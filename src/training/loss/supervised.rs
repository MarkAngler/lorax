//! Supervised loss functions for LoRA fine-tuning
//!
//! This module implements various supervised loss functions for task-specific
//! fine-tuning of models with LoRA adaptations.

use super::{LossFunction, LossConfig, LossMetrics, SupervisedTaskType, ReductionMethod};
use crate::training::{Result, Error};
use crate::training::data::SupervisedBatch;
use candle_core::{Tensor, Device, DType, D};
use candle_nn as nn;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::any::Any;
use anyhow::{Context, anyhow};
use tracing::{debug, warn, instrument};

/// Supervised loss function for task-specific fine-tuning
pub struct SupervisedLoss {
    config: LossConfig,
    device: Device,
    task_type: SupervisedTaskType,
    label_smoothing: f64,
    lora_reg_weight: f64,
    loss_fn: Box<dyn SupervisedLossFunction + Send + Sync>,
}

/// Configuration specific to supervised loss
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisedLossConfig {
    /// Base loss configuration
    pub base: LossConfig,
    /// Task type and parameters
    pub task_type: SupervisedTaskType,
    /// Label smoothing factor (0.0 = no smoothing)
    pub label_smoothing: f64,
    /// LoRA regularization weight
    pub lora_reg_weight: f64,
    /// Focal loss parameters (for imbalanced datasets)
    pub focal_loss: Option<FocalLossConfig>,
    /// Class weights for imbalanced classification
    pub class_weights: Option<Vec<f64>>,
    /// Token-level weights for sequence tasks
    pub token_weights: Option<HashMap<u32, f64>>,
    /// Ignore certain labels/tokens
    pub ignore_index: Option<i64>,
}

/// Focal loss configuration for handling imbalanced datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocalLossConfig {
    /// Focusing parameter (higher = more focus on hard examples)
    pub alpha: f64,
    /// Weighting parameter for rare classes
    pub gamma: f64,
}

/// Model predictions for supervised tasks
#[derive(Debug, Clone)]
pub struct SupervisedPredictions {
    /// Logits tensor - shape depends on task type:
    /// - Classification: [batch_size, num_classes]
    /// - Generation: [batch_size, seq_len, vocab_size]
    /// - Sequence Labeling: [batch_size, seq_len, num_labels]
    pub logits: Tensor,
    /// Optional hidden states for regularization
    pub hidden_states: Option<Vec<Tensor>>,
    /// Optional attention weights
    pub attention_weights: Option<Vec<Tensor>>,
    /// LoRA parameters (for regularization)
    pub lora_params: Option<HashMap<String, (Tensor, Tensor)>>,
}

/// Trait for different supervised loss functions
trait SupervisedLossFunction: Send + Sync {
    fn compute_loss(
        &self,
        predictions: &SupervisedPredictions,
        batch: &SupervisedBatch,
        config: &SupervisedLossConfig,
    ) -> Result<LossMetrics>;
    
    fn name(&self) -> &'static str;
}

/// Cross-entropy loss for classification
struct CrossEntropyLoss {
    num_classes: usize,
}

impl SupervisedLossFunction for CrossEntropyLoss {
    #[instrument(skip(self, predictions, batch, config))]
    fn compute_loss(
        &self,
        predictions: &SupervisedPredictions,
        batch: &SupervisedBatch,
        config: &SupervisedLossConfig,
    ) -> Result<LossMetrics> {
        let mut metrics = LossMetrics::new();
        
        let labels = batch.labels.as_ref()
            .ok_or_else(|| anyhow!("Labels required for cross-entropy loss"))?;
        
        let logits = &predictions.logits;
        
        // Apply label smoothing if specified
        let loss = if config.label_smoothing > 0.0 {
            self.cross_entropy_with_smoothing(logits, labels, config.label_smoothing)?
        } else {
            nn::loss::cross_entropy(logits, labels)?
        };
        
        let loss_value = match config.base.reduction {
            ReductionMethod::Mean => loss.mean_all()?,
            ReductionMethod::Sum => loss.sum_all()?,
            ReductionMethod::None => loss,
        };
        
        let loss_scalar = loss_value.to_scalar::<f64>()
            .context("Failed to convert loss to scalar")?;
        
        metrics.add_component("cross_entropy", loss_scalar);
        
        // Compute accuracy
        let accuracy = self.compute_accuracy(logits, labels)?;
        metrics.add_metric("accuracy", accuracy);
        
        // Compute per-class metrics if applicable
        if self.num_classes <= 10 {
            let class_metrics = self.compute_class_metrics(logits, labels)?;
            for (class_idx, metric_value) in class_metrics.iter().enumerate() {
                metrics.add_metric(&format!("class_{}_f1", class_idx), *metric_value);
            }
        }
        
        Ok(metrics)
    }
    
    fn name(&self) -> &'static str { "cross_entropy" }
}

impl CrossEntropyLoss {
    fn cross_entropy_with_smoothing(&self, logits: &Tensor, labels: &Tensor, smoothing: f64) -> Result<Tensor> {
        let num_classes = logits.dim(D::Minus1)?;
        let log_probs = nn::ops::log_softmax(logits, D::Minus1)?;
        
        // One-hot encode labels
        let one_hot = labels.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?;
        let one_hot_expanded = Tensor::zeros_like(&log_probs)?;
        let one_hot_final = one_hot_expanded.scatter_add(&one_hot, &Tensor::ones_like(&one_hot)?, D::Minus1)?;
        
        // Apply label smoothing
        let smooth_labels = &one_hot_final * (1.0 - smoothing) + smoothing / num_classes as f64;
        
        // Compute loss
        let loss = -(&log_probs * &smooth_labels)?.sum_keepdim(D::Minus1)?;
        Ok(loss)
    }
    
    fn compute_accuracy(&self, logits: &Tensor, labels: &Tensor) -> Result<f64> {
        let predictions = logits.argmax_keepdim(D::Minus1)?;
        let labels_expanded = labels.unsqueeze(D::Minus1)?;
        let correct = predictions.eq(&labels_expanded)?.to_dtype(DType::F32)?;
        let accuracy = correct.mean_all()?.to_scalar::<f64>()?;
        Ok(accuracy)
    }
    
    fn compute_class_metrics(&self, logits: &Tensor, labels: &Tensor) -> Result<Vec<f64>> {
        let predictions = logits.argmax_keepdim(D::Minus1)?.squeeze(D::Minus1)?;
        let pred_data = predictions.to_vec1::<u32>()?;
        let label_data = labels.to_vec1::<u32>()?;
        
        let mut class_metrics = vec![0.0; self.num_classes];
        
        // Simple F1 computation per class
        for class_idx in 0..self.num_classes {
            let mut tp = 0;
            let mut fp = 0;
            let mut fn_ = 0;
            
            for (pred, label) in pred_data.iter().zip(label_data.iter()) {
                let pred_class = *pred as usize;
                let label_class = *label as usize;
                
                if pred_class == class_idx && label_class == class_idx {
                    tp += 1;
                } else if pred_class == class_idx && label_class != class_idx {
                    fp += 1;
                } else if pred_class != class_idx && label_class == class_idx {
                    fn_ += 1;
                }
            }
            
            let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
            let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
            let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
            
            class_metrics[class_idx] = f1;
        }
        
        Ok(class_metrics)
    }
}

/// Generation loss for text generation tasks
struct GenerationLoss;

impl SupervisedLossFunction for GenerationLoss {
    #[instrument(skip(self, predictions, batch, config))]
    fn compute_loss(
        &self, 
        predictions: &SupervisedPredictions,
        batch: &SupervisedBatch,
        config: &SupervisedLossConfig,
    ) -> Result<LossMetrics> {
        let mut metrics = LossMetrics::new();
        
        let labels = batch.labels.as_ref()
            .ok_or_else(|| anyhow!("Labels required for generation loss"))?;
        
        let logits = &predictions.logits;
        
        // Shift logits and labels for next-token prediction
        let shift_logits = logits.narrow(1, 0, logits.dim(1)? - 1)?;
        let shift_labels = labels.narrow(1, 1, labels.dim(1)? - 1)?;
        
        // Flatten for cross-entropy computation
        let flat_logits = shift_logits.flatten_to(1)?;
        let flat_labels = shift_labels.flatten_all()?;
        
        // Compute cross-entropy loss
        let loss = nn::loss::cross_entropy(&flat_logits, &flat_labels)?;
        
        let loss_value = match config.base.reduction {
            ReductionMethod::Mean => loss.mean_all()?,
            ReductionMethod::Sum => loss.sum_all()?,
            ReductionMethod::None => loss,
        };
        
        let loss_scalar = loss_value.to_scalar::<f64>()?;
        metrics.add_component("generation", loss_scalar);
        
        // Compute perplexity
        let perplexity = loss_scalar.exp();
        metrics.add_metric("perplexity", perplexity);
        
        // Compute token-level accuracy
        let token_accuracy = self.compute_token_accuracy(&flat_logits, &flat_labels)?;
        metrics.add_metric("token_accuracy", token_accuracy);
        
        Ok(metrics)
    }
    
    fn name(&self) -> &'static str { "generation" }
}

impl GenerationLoss {
    fn compute_token_accuracy(&self, logits: &Tensor, labels: &Tensor) -> Result<f64> {
        let predictions = logits.argmax_keepdim(D::Minus1)?;
        let labels_expanded = labels.unsqueeze(D::Minus1)?;
        let correct = predictions.eq(&labels_expanded)?.to_dtype(DType::F32)?;
        let accuracy = correct.mean_all()?.to_scalar::<f64>()?;
        Ok(accuracy)
    }
}

/// Sequence labeling loss for token classification
struct SequenceLabelingLoss {
    num_labels: usize,
}

impl SupervisedLossFunction for SequenceLabelingLoss {
    #[instrument(skip(self, predictions, batch, config))]
    fn compute_loss(
        &self,
        predictions: &SupervisedPredictions,
        batch: &SupervisedBatch,
        config: &SupervisedLossConfig,
    ) -> Result<LossMetrics> {
        let mut metrics = LossMetrics::new();
        
        let labels = batch.labels.as_ref()
            .ok_or_else(|| anyhow!("Labels required for sequence labeling loss"))?;
        
        let logits = &predictions.logits;
        
        // Handle attention mask for proper loss computation
        let loss = if let Some(attention_mask) = &batch.attention_mask {
            self.masked_cross_entropy(logits, labels, attention_mask, config)?
        } else {
            let flat_logits = logits.flatten_to(1)?;
            let flat_labels = labels.flatten_all()?;
            nn::loss::cross_entropy(&flat_logits, &flat_labels)?
        };
        
        let loss_value = match config.base.reduction {
            ReductionMethod::Mean => loss.mean_all()?,
            ReductionMethod::Sum => loss.sum_all()?,
            ReductionMethod::None => loss,
        };
        
        let loss_scalar = loss_value.to_scalar::<f64>()?;
        metrics.add_component("sequence_labeling", loss_scalar);
        
        // Compute token-level accuracy
        let accuracy = self.compute_sequence_accuracy(logits, labels, batch.attention_mask.as_ref())?;
        metrics.add_metric("token_accuracy", accuracy);
        
        // Compute entity-level F1 if applicable
        if self.num_labels <= 20 {
            let entity_f1 = self.compute_entity_f1(logits, labels, batch.attention_mask.as_ref())?;
            metrics.add_metric("entity_f1", entity_f1);
        }
        
        Ok(metrics)
    }
    
    fn name(&self) -> &'static str { "sequence_labeling" }
}

impl SequenceLabelingLoss {
    fn masked_cross_entropy(
        &self,
        logits: &Tensor,
        labels: &Tensor,
        attention_mask: &Tensor,
        config: &SupervisedLossConfig,
    ) -> Result<Tensor> {
        let flat_logits = logits.flatten_to(1)?;
        let flat_labels = labels.flatten_all()?;
        let flat_mask = attention_mask.flatten_all()?.to_dtype(DType::F32)?;
        
        // Compute cross-entropy for all positions
        let loss = nn::loss::cross_entropy(&flat_logits, &flat_labels)?;
        
        // Apply mask to ignore padded positions
        let masked_loss = &loss * &flat_mask;
        let num_valid = flat_mask.sum_all()?;
        
        // Return mean over valid positions
        Ok(masked_loss.sum_all()? / num_valid)
    }
    
    fn compute_sequence_accuracy(
        &self,
        logits: &Tensor,
        labels: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<f64> {
        let predictions = logits.argmax_keepdim(D::Minus1)?;
        let labels_expanded = labels.unsqueeze(D::Minus1)?;
        let correct = predictions.eq(&labels_expanded)?.to_dtype(DType::F32)?;
        
        if let Some(mask) = attention_mask {
            let mask_expanded = mask.unsqueeze(D::Minus1)?.to_dtype(DType::F32)?;
            let masked_correct = &correct * &mask_expanded;
            let total_correct = masked_correct.sum_all()?;
            let total_valid = mask.sum_all()?.to_dtype(DType::F32)?;
            Ok((total_correct / total_valid).to_scalar::<f64>()?)
        } else {
            Ok(correct.mean_all()?.to_scalar::<f64>()?)
        }
    }
    
    fn compute_entity_f1(
        &self,
        logits: &Tensor,
        labels: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<f64> {
        // Simplified entity F1 - in practice, you'd want proper BIO tag handling
        let predictions = logits.argmax_keepdim(D::Minus1)?.squeeze(D::Minus1)?;
        
        // For now, just return token-level F1 as approximation
        self.compute_sequence_accuracy(&logits, labels, attention_mask)
    }
}

/// Masked language modeling loss
struct MaskedLMLoss;

impl SupervisedLossFunction for MaskedLMLoss {
    #[instrument(skip(self, predictions, batch, config))]
    fn compute_loss(
        &self,
        predictions: &SupervisedPredictions,
        batch: &SupervisedBatch,
        config: &SupervisedLossConfig,
    ) -> Result<LossMetrics> {
        let mut metrics = LossMetrics::new();
        
        let labels = batch.labels.as_ref()
            .ok_or_else(|| anyhow!("Labels required for masked LM loss"))?;
        
        let logits = &predictions.logits;
        
        // Compute loss only on masked positions (labels != -100 typically)
        let ignore_index = config.ignore_index.unwrap_or(-100);
        let valid_mask = labels.ne(ignore_index as f64)?;
        
        let loss = self.masked_cross_entropy_mlm(logits, labels, &valid_mask)?;
        
        let loss_value = match config.base.reduction {
            ReductionMethod::Mean => loss.mean_all()?,
            ReductionMethod::Sum => loss.sum_all()?,
            ReductionMethod::None => loss,
        };
        
        let loss_scalar = loss_value.to_scalar::<f64>()?;
        metrics.add_component("masked_lm", loss_scalar);
        
        // Compute perplexity on masked tokens
        let perplexity = loss_scalar.exp();
        metrics.add_metric("masked_perplexity", perplexity);
        
        // Compute accuracy on masked tokens
        let masked_accuracy = self.compute_masked_accuracy(logits, labels, &valid_mask)?;
        metrics.add_metric("masked_accuracy", masked_accuracy);
        
        Ok(metrics)
    }
    
    fn name(&self) -> &'static str { "masked_lm" }
}

impl MaskedLMLoss {
    fn masked_cross_entropy_mlm(&self, logits: &Tensor, labels: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let flat_logits = logits.flatten_to(1)?;
        let flat_labels = labels.flatten_all()?;
        let flat_mask = mask.flatten_all()?.to_dtype(DType::F32)?;
        
        // Compute cross-entropy for all positions
        let loss = nn::loss::cross_entropy(&flat_logits, &flat_labels)?;
        
        // Apply mask to compute loss only on masked positions
        let masked_loss = &loss * &flat_mask;
        let num_masked = flat_mask.sum_all()?;
        
        // Return mean over masked positions
        Ok(masked_loss.sum_all()? / num_masked)
    }
    
    fn compute_masked_accuracy(&self, logits: &Tensor, labels: &Tensor, mask: &Tensor) -> Result<f64> {
        let predictions = logits.argmax_keepdim(D::Minus1)?;
        let labels_expanded = labels.unsqueeze(D::Minus1)?;
        let correct = predictions.eq(&labels_expanded)?.to_dtype(DType::F32)?;
        let mask_expanded = mask.unsqueeze(D::Minus1)?.to_dtype(DType::F32)?;
        
        let masked_correct = &correct * &mask_expanded;
        let total_correct = masked_correct.sum_all()?;
        let total_masked = mask.sum_all()?.to_dtype(DType::F32)?;
        
        Ok((total_correct / total_masked).to_scalar::<f64>()?)
    }
}

impl SupervisedLoss {
    /// Create new supervised loss function
    pub fn new(config: LossConfig, device: &Device) -> Result<Self> {
        let (task_type, label_smoothing, lora_reg_weight) = match &config.loss_type {
            super::LossType::Supervised { 
                task_type, 
                label_smoothing, 
                lora_reg_weight 
            } => (task_type.clone(), *label_smoothing, *lora_reg_weight),
            _ => return Err(anyhow!("Invalid loss type for supervised loss")),
        };
        
        let loss_fn = Self::create_loss_function(&task_type)?;
        
        Ok(Self {
            config,
            device: device.clone(),
            task_type,
            label_smoothing,
            lora_reg_weight,
            loss_fn,
        })
    }
    
    /// Create supervised loss with custom configuration
    pub fn with_config(config: SupervisedLossConfig, device: &Device) -> Result<Self> {
        let loss_fn = Self::create_loss_function(&config.task_type)?;
        
        Ok(Self {
            config: config.base.clone(),
            device: device.clone(),
            task_type: config.task_type.clone(),
            label_smoothing: config.label_smoothing,
            lora_reg_weight: config.lora_reg_weight,
            loss_fn,
        })
    }
    
    /// Create task-specific loss function
    fn create_loss_function(task_type: &SupervisedTaskType) -> Result<Box<dyn SupervisedLossFunction + Send + Sync>> {
        let loss_fn: Box<dyn SupervisedLossFunction + Send + Sync> = match task_type {
            SupervisedTaskType::Classification { num_classes } => {
                Box::new(CrossEntropyLoss { num_classes: *num_classes })
            },
            SupervisedTaskType::Generation => {
                Box::new(GenerationLoss)
            },
            SupervisedTaskType::SequenceLabeling { num_labels } => {
                Box::new(SequenceLabelingLoss { num_labels: *num_labels })
            },
            SupervisedTaskType::MaskedLM => {
                Box::new(MaskedLMLoss)
            },
        };
        
        Ok(loss_fn)
    }
    
    /// Compute LoRA regularization loss
    #[instrument(skip(self, lora_params))]
    pub fn compute_lora_regularization(&self, lora_params: &HashMap<String, (Tensor, Tensor)>) -> Result<f64> {
        let mut total_reg = 0.0;
        let mut param_count = 0;
        
        for (layer_name, (a_matrix, b_matrix)) in lora_params {
            // L2 regularization on LoRA parameters
            let a_l2 = a_matrix.sqr()?.mean_all()?.to_scalar::<f64>()?;
            let b_l2 = b_matrix.sqr()?.mean_all()?.to_scalar::<f64>()?;
            
            total_reg += a_l2 + b_l2;
            param_count += 2;
            
            debug!("LoRA regularization for {}: A_L2={:.6}, B_L2={:.6}", layer_name, a_l2, b_l2);
        }
        
        if param_count > 0 {
            Ok(total_reg / param_count as f64)
        } else {
            Ok(0.0)
        }
    }
}

impl LossFunction for SupervisedLoss {
    fn forward(&self, batch: &dyn Any, predictions: &dyn Any) -> Result<Tensor> {
        let batch = batch.downcast_ref::<SupervisedBatch>()
            .ok_or_else(|| anyhow!("Invalid batch type for supervised loss"))?;
        let predictions = predictions.downcast_ref::<SupervisedPredictions>()
            .ok_or_else(|| anyhow!("Invalid predictions type for supervised loss"))?;
        
        let config = SupervisedLossConfig {
            base: self.config.clone(),
            task_type: self.task_type.clone(),
            label_smoothing: self.label_smoothing,
            lora_reg_weight: self.lora_reg_weight,
            focal_loss: None,
            class_weights: None,
            token_weights: None,
            ignore_index: None,
        };
        
        let metrics = self.loss_fn.compute_loss(predictions, batch, &config)?;
        
        // Return total loss as tensor
        Tensor::full(metrics.total_loss, (), &self.device)
            .context("Failed to create loss tensor")
    }
    
    fn compute_metrics(&self, batch: &dyn Any, predictions: &dyn Any) -> Result<LossMetrics> {
        let batch = batch.downcast_ref::<SupervisedBatch>()
            .ok_or_else(|| anyhow!("Invalid batch type for supervised loss"))?;
        let predictions = predictions.downcast_ref::<SupervisedPredictions>()
            .ok_or_else(|| anyhow!("Invalid predictions type for supervised loss"))?;
        
        let config = SupervisedLossConfig {
            base: self.config.clone(),
            task_type: self.task_type.clone(),
            label_smoothing: self.label_smoothing,
            lora_reg_weight: self.lora_reg_weight,
            focal_loss: None,
            class_weights: None,
            token_weights: None,
            ignore_index: None,
        };
        
        let mut metrics = self.loss_fn.compute_loss(predictions, batch, &config)?;
        
        // Add LoRA regularization if parameters are provided
        if let Some(lora_params) = &predictions.lora_params {
            if self.lora_reg_weight > 0.0 {
                let lora_reg = self.compute_lora_regularization(lora_params)?;
                let weighted_reg = lora_reg * self.lora_reg_weight;
                metrics.add_regularization_loss("lora_regularization", weighted_reg);
            }
        }
        
        Ok(metrics)
    }
    
    fn config(&self) -> &LossConfig {
        &self.config
    }
    
    fn clone_box(&self) -> Box<dyn LossFunction> {
        Box::new(Self {
            config: self.config.clone(),
            device: self.device.clone(),
            task_type: self.task_type.clone(),
            label_smoothing: self.label_smoothing,
            lora_reg_weight: self.lora_reg_weight,
            loss_fn: Self::create_loss_function(&self.task_type).unwrap(),
        })
    }
}

impl Default for SupervisedLossConfig {
    fn default() -> Self {
        Self {
            base: LossConfig::supervised_default(),
            task_type: SupervisedTaskType::Classification { num_classes: 2 },
            label_smoothing: 0.0,
            lora_reg_weight: 0.01,
            focal_loss: None,
            class_weights: None,
            token_weights: None,
            ignore_index: Some(-100),
        }
    }
}

impl Default for FocalLossConfig {
    fn default() -> Self {
        Self {
            alpha: 0.25,
            gamma: 2.0,
        }
    }
}