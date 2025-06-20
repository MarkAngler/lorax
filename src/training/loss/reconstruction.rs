//! Reconstruction loss for LoRA parameter training
//!
//! This module implements MSE and other losses for training the hypernetwork
//! to generate LoRA parameters from task descriptions.

use super::{LossFunction, LossConfig, LossMetrics, MatrixLossType, ReductionMethod};
use crate::training::{Result, Error};
use crate::training::data::ReconstructionBatch;
use crate::lora::parameters::{LoraParameters, LoraLayer};
use candle_core::{Tensor, Device, DType};
use candle_nn as nn;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::any::Any;
use anyhow::{Context, anyhow};
use tracing::{debug, warn, instrument};

/// Reconstruction loss function for LoRA parameter generation
pub struct ReconstructionLoss {
    config: LossConfig,
    device: Device,
    layer_weights: HashMap<String, f64>,
    matrix_loss_fn: Box<dyn MatrixLossFunction + Send + Sync>,
}

/// Configuration specific to reconstruction loss
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionLossConfig {
    /// Base loss configuration
    pub base: LossConfig,
    /// Layer-specific weights for loss computation
    pub layer_weights: HashMap<String, f64>,
    /// Matrix loss function type
    pub matrix_loss: MatrixLossType,
    /// Weight for A matrix loss vs B matrix loss
    pub matrix_balance: MatrixBalance,
    /// Consistency regularization weight
    pub consistency_weight: f64,
    /// Sparsity regularization weight
    pub sparsity_weight: f64,
    /// Enable progressive training (start with easier layers)
    pub progressive_training: Option<ProgressiveConfig>,
}

/// Balance between A and B matrix losses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixBalance {
    /// Weight for A matrix loss
    pub a_weight: f64,
    /// Weight for B matrix loss  
    pub b_weight: f64,
    /// Adaptive balancing based on magnitude
    pub adaptive: bool,
}

/// Progressive training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveConfig {
    /// Start training with these layers
    pub initial_layers: Vec<String>,
    /// Add layers every N steps
    pub add_interval: usize,
    /// Layers to add at each interval
    pub layers_per_interval: usize,
}

/// Predicted LoRA parameters from the hypernetwork
#[derive(Debug, Clone)]
pub struct PredictedLoraParams {
    /// Predicted parameters by layer
    pub layers: HashMap<String, PredictedLoraLayer>,
    /// Device for tensors
    pub device: Device,
}

/// Predicted parameters for a single LoRA layer
#[derive(Debug, Clone)]
pub struct PredictedLoraLayer {
    /// Layer name
    pub name: String,
    /// Predicted A matrix [batch_size, input_dim, rank] 
    pub a_matrix: Tensor,
    /// Predicted B matrix [batch_size, rank, output_dim]
    pub b_matrix: Tensor,
    /// Predicted scaling factors [batch_size]
    pub alpha: Option<Tensor>,
}

/// Trait for different matrix loss functions
trait MatrixLossFunction: Send + Sync {
    fn compute_loss(&self, predicted: &Tensor, target: &Tensor, reduction: &ReductionMethod) -> Result<Tensor>;
    fn name(&self) -> &'static str;
}

/// Mean Squared Error loss
struct MseLoss;

impl MatrixLossFunction for MseLoss {
    fn compute_loss(&self, predicted: &Tensor, target: &Tensor, reduction: &ReductionMethod) -> Result<Tensor> {
        let diff = (predicted - target)?;
        let squared_diff = diff.sqr()?;
        
        match reduction {
            ReductionMethod::Mean => squared_diff.mean_all(),
            ReductionMethod::Sum => squared_diff.sum_all(),
            ReductionMethod::None => Ok(squared_diff),
        }.context("Failed to compute MSE loss")
    }
    
    fn name(&self) -> &'static str { "mse" }
}

/// Mean Absolute Error loss
struct MaeLoss;

impl MatrixLossFunction for MaeLoss {
    fn compute_loss(&self, predicted: &Tensor, target: &Tensor, reduction: &ReductionMethod) -> Result<Tensor> {
        let diff = (predicted - target)?;
        let abs_diff = diff.abs()?;
        
        match reduction {
            ReductionMethod::Mean => abs_diff.mean_all(),
            ReductionMethod::Sum => abs_diff.sum_all(),
            ReductionMethod::None => Ok(abs_diff),
        }.context("Failed to compute MAE loss")
    }
    
    fn name(&self) -> &'static str { "mae" }
}

/// Huber loss (robust to outliers)
struct HuberLoss {
    delta: f64,
}

impl MatrixLossFunction for HuberLoss {
    fn compute_loss(&self, predicted: &Tensor, target: &Tensor, reduction: &ReductionMethod) -> Result<Tensor> {
        let diff = (predicted - target)?;
        let abs_diff = diff.abs()?;
        let delta_tensor = Tensor::full(self.delta, abs_diff.shape(), abs_diff.device())?;
        
        // Huber loss: 0.5 * x^2 if |x| <= delta, else delta * (|x| - 0.5 * delta)
        let quadratic_mask = abs_diff.le(&delta_tensor)?;
        let quadratic_loss = (diff.sqr()? * 0.5)?;
        let linear_loss = (&delta_tensor * (&abs_diff - &delta_tensor * 0.5)?)?;
        
        let loss = quadratic_mask.where_cond(&quadratic_loss, &linear_loss)?;
        
        match reduction {
            ReductionMethod::Mean => loss.mean_all(),
            ReductionMethod::Sum => loss.sum_all(),
            ReductionMethod::None => Ok(loss),
        }.context("Failed to compute Huber loss")
    }
    
    fn name(&self) -> &'static str { "huber" }
}

/// Cosine similarity loss
struct CosineLoss;

impl MatrixLossFunction for CosineLoss {
    fn compute_loss(&self, predicted: &Tensor, target: &Tensor, reduction: &ReductionMethod) -> Result<Tensor> {
        // Flatten last two dimensions for cosine similarity
        let pred_flat = predicted.flatten_from(predicted.dims().len() - 2)?;
        let target_flat = target.flatten_from(target.dims().len() - 2)?;
        
        // Compute cosine similarity
        let dot_product = (&pred_flat * &target_flat)?.sum_keepdim(D::Minus1)?;
        let pred_norm = pred_flat.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let target_norm = target_flat.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        
        let cosine_sim = &dot_product / (&pred_norm * &target_norm + 1e-8)?;
        let cosine_loss = (Tensor::ones_like(&cosine_sim)? - cosine_sim)?;
        
        match reduction {
            ReductionMethod::Mean => cosine_loss.mean_all(),
            ReductionMethod::Sum => cosine_loss.sum_all(), 
            ReductionMethod::None => Ok(cosine_loss),
        }.context("Failed to compute cosine loss")
    }
    
    fn name(&self) -> &'static str { "cosine" }
}

impl ReconstructionLoss {
    /// Create new reconstruction loss function
    pub fn new(config: LossConfig, device: &Device) -> Result<Self> {
        let matrix_loss_fn = Self::create_matrix_loss_fn(&config)?;
        let layer_weights = Self::extract_layer_weights(&config);
        
        Ok(Self {
            config,
            device: device.clone(),
            layer_weights,
            matrix_loss_fn,
        })
    }
    
    /// Create reconstruction loss with custom configuration
    pub fn with_config(config: ReconstructionLossConfig, device: &Device) -> Result<Self> {
        let matrix_loss_fn = Self::create_matrix_loss_fn(&config.base)?;
        
        Ok(Self {
            config: config.base,
            device: device.clone(),
            layer_weights: config.layer_weights,
            matrix_loss_fn,
        })
    }
    
    /// Create matrix loss function based on configuration
    fn create_matrix_loss_fn(config: &LossConfig) -> Result<Box<dyn MatrixLossFunction + Send + Sync>> {
        let matrix_loss = match &config.loss_type {
            super::LossType::Reconstruction { matrix_loss, .. } => matrix_loss,
            _ => return Err(anyhow!("Invalid loss type for reconstruction loss")),
        };
        
        let loss_fn: Box<dyn MatrixLossFunction + Send + Sync> = match matrix_loss {
            MatrixLossType::Mse => Box::new(MseLoss),
            MatrixLossType::Mae => Box::new(MaeLoss),
            MatrixLossType::Huber { delta } => Box::new(HuberLoss { delta: *delta }),
            MatrixLossType::Cosine => Box::new(CosineLoss),
        };
        
        Ok(loss_fn)
    }
    
    /// Extract layer weights from configuration
    fn extract_layer_weights(config: &LossConfig) -> HashMap<String, f64> {
        config.layer_configs
            .iter()
            .map(|(layer, cfg)| (layer.clone(), cfg.weight))
            .collect()
    }
    
    /// Compute reconstruction loss for predicted vs target LoRA parameters
    #[instrument(skip(self, predicted, target))]
    pub fn compute_reconstruction_loss(
        &self,
        predicted: &PredictedLoraParams,
        target: &LoraParameters,
    ) -> Result<LossMetrics> {
        let mut metrics = LossMetrics::new();
        let mut total_loss = 0.0;
        
        debug!("Computing reconstruction loss for {} layers", predicted.layers.len());
        
        // Compute loss for each layer
        for (layer_name, pred_layer) in &predicted.layers {
            if let Some(target_layer) = target.get_layer(layer_name) {
                let layer_loss = self.compute_layer_loss(pred_layer, target_layer)?;
                let layer_weight = self.layer_weights.get(layer_name).unwrap_or(&1.0);
                let weighted_loss = layer_loss * layer_weight;
                
                metrics.add_layer_loss(layer_name, weighted_loss);
                total_loss += weighted_loss;
                
                debug!("Layer {} loss: {:.6}", layer_name, weighted_loss);
            } else {
                warn!("Target layer {} not found in LoRA parameters", layer_name);
            }
        }
        
        // Add consistency regularization
        if let super::LossType::Reconstruction { consistency_weight, .. } = &self.config.loss_type {
            if *consistency_weight > 0.0 {
                let consistency_loss = self.compute_consistency_loss(predicted)?;
                let weighted_consistency = consistency_loss * consistency_weight;
                metrics.add_regularization_loss("consistency", weighted_consistency);
                total_loss += weighted_consistency;
            }
        }
        
        // Add sparsity regularization
        if let super::LossType::Reconstruction { sparsity_weight, .. } = &self.config.loss_type {
            if *sparsity_weight > 0.0 {
                let sparsity_loss = self.compute_sparsity_loss(predicted)?;
                let weighted_sparsity = sparsity_loss * sparsity_weight;
                metrics.add_regularization_loss("sparsity", weighted_sparsity);
                total_loss += weighted_sparsity;
            }
        }
        
        metrics.total_loss = total_loss;
        metrics.add_component("reconstruction", total_loss - metrics.total_regularization_loss());
        
        // Compute additional metrics
        self.compute_additional_metrics(&mut metrics, predicted, target)?;
        
        Ok(metrics)
    }
    
    /// Compute loss for a single layer
    fn compute_layer_loss(
        &self,
        predicted: &PredictedLoraLayer,
        target: &LoraLayer,
    ) -> Result<f64> {
        // Convert target parameters to tensors
        let target_a = self.vec_to_tensor(&target.a_weights, predicted.a_matrix.shape())?;
        let target_b = self.vec_to_tensor(&target.b_weights, predicted.b_matrix.shape())?;
        
        // Compute A matrix loss
        let a_loss = self.matrix_loss_fn.compute_loss(
            &predicted.a_matrix,
            &target_a,
            &self.config.reduction,
        )?;
        
        // Compute B matrix loss
        let b_loss = self.matrix_loss_fn.compute_loss(
            &predicted.b_matrix,
            &target_b,
            &self.config.reduction,
        )?;
        
        // Balance A and B losses (default: equal weight)
        let total_loss = (a_loss.to_scalar::<f64>()? + b_loss.to_scalar::<f64>()?) * 0.5;
        
        Ok(total_loss)
    }
    
    /// Convert Vec<f32> to Tensor with specified shape
    fn vec_to_tensor(&self, vec: &[f32], shape: &[usize]) -> Result<Tensor> {
        let tensor = Tensor::from_slice(vec, (vec.len(),), &self.device)?;
        tensor.reshape(shape).context("Failed to reshape tensor")
    }
    
    /// Compute consistency regularization loss
    fn compute_consistency_loss(&self, predicted: &PredictedLoraParams) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut count = 0;
        
        // Consistency across similar layer types (e.g., q_proj, k_proj, v_proj)
        let layer_groups = self.group_similar_layers(&predicted.layers);
        
        for group in layer_groups {
            if group.len() > 1 {
                let consistency = self.compute_group_consistency(&group)?;
                total_loss += consistency;
                count += 1;
            }
        }
        
        if count > 0 {
            Ok(total_loss / count as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Group similar layers for consistency regularization
    fn group_similar_layers<'a>(&self, layers: &'a HashMap<String, PredictedLoraLayer>) -> Vec<Vec<&'a PredictedLoraLayer>> {
        let mut groups: HashMap<String, Vec<_>> = HashMap::new();
        
        for layer in layers.values() {
            let layer_type = self.extract_layer_type(&layer.name);
            groups.entry(layer_type).or_default().push(layer);
        }
        
        groups.into_values().collect()
    }
    
    /// Extract layer type from layer name (e.g., "layer.0.attention.q_proj" -> "q_proj")
    fn extract_layer_type(&self, layer_name: &str) -> String {
        layer_name.split('.').last().unwrap_or(layer_name).to_string()
    }
    
    /// Compute consistency within a group of layers
    fn compute_group_consistency(&self, group: &[&PredictedLoraLayer]) -> Result<f64> {
        if group.len() < 2 {
            return Ok(0.0);
        }
        
        let mut total_variance = 0.0;
        
        // Compute variance of A matrices
        let a_matrices: Vec<_> = group.iter().map(|l| &l.a_matrix).collect();
        let a_variance = self.compute_tensor_variance(&a_matrices)?;
        total_variance += a_variance;
        
        // Compute variance of B matrices
        let b_matrices: Vec<_> = group.iter().map(|l| &l.b_matrix).collect();
        let b_variance = self.compute_tensor_variance(&b_matrices)?;
        total_variance += b_variance;
        
        Ok(total_variance)
    }
    
    /// Compute variance across a group of tensors
    fn compute_tensor_variance(&self, tensors: &[&Tensor]) -> Result<f64> {
        if tensors.is_empty() {
            return Ok(0.0);
        }
        
        // Stack tensors and compute variance
        let stacked = Tensor::stack(tensors, 0)?;
        let mean = stacked.mean_keepdim(D::Minus1)?;
        let variance = ((stacked - mean)?.sqr()?).mean_all()?;
        
        variance.to_scalar::<f64>().context("Failed to convert variance to scalar")
    }
    
    /// Compute sparsity regularization loss
    fn compute_sparsity_loss(&self, predicted: &PredictedLoraParams) -> Result<f64> {
        let mut total_sparsity = 0.0;
        let mut count = 0;
        
        for layer in predicted.layers.values() {
            // L1 norm for sparsity
            let a_l1 = layer.a_matrix.abs()?.mean_all()?.to_scalar::<f64>()?;
            let b_l1 = layer.b_matrix.abs()?.mean_all()?.to_scalar::<f64>()?;
            
            total_sparsity += a_l1 + b_l1;
            count += 2;
        }
        
        if count > 0 {
            Ok(total_sparsity / count as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Compute additional metrics like accuracy and alignment
    fn compute_additional_metrics(
        &self,
        metrics: &mut LossMetrics,
        predicted: &PredictedLoraParams,
        target: &LoraParameters,
    ) -> Result<()> {
        // Parameter alignment (cosine similarity)
        let mut total_alignment = 0.0;
        let mut layer_count = 0;
        
        for (layer_name, pred_layer) in &predicted.layers {
            if let Some(target_layer) = target.get_layer(layer_name) {
                let alignment = self.compute_parameter_alignment(pred_layer, target_layer)?;
                total_alignment += alignment;
                layer_count += 1;
            }
        }
        
        if layer_count > 0 {
            metrics.add_metric("parameter_alignment", total_alignment / layer_count as f64);
        }
        
        // Parameter magnitude ratio
        let pred_magnitude = self.compute_parameter_magnitude(predicted)?;
        let target_magnitude = self.compute_target_magnitude(target)?;
        if target_magnitude > 0.0 {
            metrics.add_metric("magnitude_ratio", pred_magnitude / target_magnitude);
        }
        
        // Effective rank (approximate)
        let avg_rank = self.compute_average_effective_rank(predicted)?;
        metrics.add_metric("effective_rank", avg_rank);
        
        Ok(())
    }
    
    /// Compute parameter alignment between predicted and target
    fn compute_parameter_alignment(&self, predicted: &PredictedLoraLayer, target: &LoraLayer) -> Result<f64> {
        let target_a = self.vec_to_tensor(&target.a_weights, predicted.a_matrix.shape())?;
        let target_b = self.vec_to_tensor(&target.b_weights, predicted.b_matrix.shape())?;
        
        // Flatten and compute cosine similarity
        let pred_flat = Tensor::cat(&[
            predicted.a_matrix.flatten_all()?,
            predicted.b_matrix.flatten_all()?,
        ], 0)?;
        
        let target_flat = Tensor::cat(&[
            target_a.flatten_all()?,
            target_b.flatten_all()?,
        ], 0)?;
        
        let dot_product = (&pred_flat * &target_flat)?.sum_all()?;
        let pred_norm = pred_flat.sqr()?.sum_all()?.sqrt()?;
        let target_norm = target_flat.sqr()?.sum_all()?.sqrt()?;
        
        let alignment = dot_product / (pred_norm * target_norm + 1e-8)?;
        alignment.to_scalar::<f64>().context("Failed to compute alignment")
    }
    
    /// Compute total parameter magnitude for predicted parameters
    fn compute_parameter_magnitude(&self, predicted: &PredictedLoraParams) -> Result<f64> {
        let mut total_magnitude = 0.0;
        
        for layer in predicted.layers.values() {
            let a_mag = layer.a_matrix.sqr()?.sum_all()?.sqrt()?.to_scalar::<f64>()?;
            let b_mag = layer.b_matrix.sqr()?.sum_all()?.sqrt()?.to_scalar::<f64>()?;
            total_magnitude += a_mag + b_mag;
        }
        
        Ok(total_magnitude)
    }
    
    /// Compute total parameter magnitude for target parameters
    fn compute_target_magnitude(&self, target: &LoraParameters) -> Result<f64> {
        let mut total_magnitude = 0.0;
        
        for layer in target.layers.values() {
            let a_mag: f64 = layer.a_weights.iter().map(|x| x * x).sum::<f32>().sqrt() as f64;
            let b_mag: f64 = layer.b_weights.iter().map(|x| x * x).sum::<f32>().sqrt() as f64;
            total_magnitude += a_mag + b_mag;
        }
        
        Ok(total_magnitude)
    }
    
    /// Compute average effective rank across layers
    fn compute_average_effective_rank(&self, predicted: &PredictedLoraParams) -> Result<f64> {
        let mut total_rank = 0.0;
        let mut layer_count = 0;
        
        for layer in predicted.layers.values() {
            // Approximate effective rank using nuclear norm / spectral norm
            let combined = (layer.a_matrix.matmul(&layer.b_matrix))?;
            
            // Simple approximation: ratio of Frobenius norm to max singular value approximation
            let frobenius_norm = combined.sqr()?.sum_all()?.sqrt()?.to_scalar::<f64>()?;
            let max_element = combined.abs()?.max_keepdim(D::Minus1)?.max_keepdim(D::Minus2)?.to_scalar::<f64>()?;
            
            if max_element > 1e-8 {
                let approx_rank = frobenius_norm / max_element;
                total_rank += approx_rank;
                layer_count += 1;
            }
        }
        
        if layer_count > 0 {
            Ok(total_rank / layer_count as f64)
        } else {
            Ok(0.0)
        }
    }
}

impl LossFunction for ReconstructionLoss {
    fn forward(&self, batch: &dyn Any, predictions: &dyn Any) -> Result<Tensor> {
        let batch = batch.downcast_ref::<ReconstructionBatch>()
            .ok_or_else(|| anyhow!("Invalid batch type for reconstruction loss"))?;
        let predictions = predictions.downcast_ref::<PredictedLoraParams>()
            .ok_or_else(|| anyhow!("Invalid predictions type for reconstruction loss"))?;
        
        // Convert batch to LoRA parameters for comparison
        let target_params = self.batch_to_lora_params(batch)?;
        let metrics = self.compute_reconstruction_loss(predictions, &target_params)?;
        
        // Return total loss as tensor
        Tensor::full(metrics.total_loss, (), &self.device)
            .context("Failed to create loss tensor")
    }
    
    fn compute_metrics(&self, batch: &dyn Any, predictions: &dyn Any) -> Result<LossMetrics> {
        let batch = batch.downcast_ref::<ReconstructionBatch>()
            .ok_or_else(|| anyhow!("Invalid batch type for reconstruction loss"))?;
        let predictions = predictions.downcast_ref::<PredictedLoraParams>()
            .ok_or_else(|| anyhow!("Invalid predictions type for reconstruction loss"))?;
        
        let target_params = self.batch_to_lora_params(batch)?;
        self.compute_reconstruction_loss(predictions, &target_params)
    }
    
    fn config(&self) -> &LossConfig {
        &self.config
    }
    
    fn clone_box(&self) -> Box<dyn LossFunction> {
        Box::new(Self {
            config: self.config.clone(),
            device: self.device.clone(),
            layer_weights: self.layer_weights.clone(),
            matrix_loss_fn: match self.matrix_loss_fn.name() {
                "mse" => Box::new(MseLoss),
                "mae" => Box::new(MaeLoss),
                "huber" => Box::new(HuberLoss { delta: 1.0 }), // Default delta
                "cosine" => Box::new(CosineLoss),
                _ => Box::new(MseLoss), // Fallback
            },
        })
    }
}

impl ReconstructionLoss {
    /// Convert batch to LoRA parameters for loss computation
    fn batch_to_lora_params(&self, batch: &ReconstructionBatch) -> Result<LoraParameters> {
        let mut params = LoraParameters::new(Default::default());
        
        for (layer_name, (a_tensor, b_tensor)) in &batch.lora_params {
            // Extract single sample from batch (assuming batch_size = 1 for simplicity)
            let a_weights = self.tensor_to_vec(a_tensor)?;
            let b_weights = self.tensor_to_vec(b_tensor)?;
            
            let layer = LoraLayer {
                name: layer_name.clone(),
                a_weights,
                b_weights,
                input_dim: a_tensor.dim(1)?,
                output_dim: b_tensor.dim(2)?,
                rank: a_tensor.dim(2)?,
                alpha: 32.0, // Default alpha
            };
            
            params.add_layer(layer)?;
        }
        
        Ok(params)
    }
    
    /// Convert tensor to Vec<f32>
    fn tensor_to_vec(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        let flattened = tensor.flatten_all()?;
        let data = flattened.to_vec1::<f32>()
            .context("Failed to convert tensor to vec")?;
        Ok(data)
    }
}

// Dimension helper for tensor operations
use candle_core::D;

impl Default for ReconstructionLossConfig {
    fn default() -> Self {
        Self {
            base: LossConfig::reconstruction_default(),
            layer_weights: HashMap::new(),
            matrix_loss: MatrixLossType::Mse,
            matrix_balance: MatrixBalance {
                a_weight: 1.0,
                b_weight: 1.0,
                adaptive: false,
            },
            consistency_weight: 0.1,
            sparsity_weight: 0.01,
            progressive_training: None,
        }
    }
}