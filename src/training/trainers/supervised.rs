//! Supervised trainer for T2L with dynamic LoRA adaptation
//!
//! This module implements a specialized trainer for supervised fine-tuning where
//! T2L generates LoRA parameters that are dynamically applied to frozen base models
//! for downstream task adaptation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, Context};
use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, VarMap, Module};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn, instrument};

use crate::hypernetwork::{HyperNetwork, HypernetworkConfig, TargetArchitecture};
use crate::lora::{LoraLayer, LoraConfig, LoraAdapter};
use crate::models::{BaseModel, ModelConfig, ModelType, ModelOutput, create_base_model};
use crate::training::{
    TrainingConfig, TrainingState, TrainingStatus, TrainingResult, TrainingEvent,
    T2LTrainer, MemoryUsage, SupervisedBatch,
    CheckpointManager, TrainingCheckpoint, MetricsTracker, TrainingMetrics,
    OptimizerState, SchedulerState, create_optimizer, create_scheduler,
    SupervisedLoss, SupervisedLossConfig, SupervisedPredictions, LossFunctionMetrics,
    SupervisedTaskType,
};
use crate::training::data::{Dataset, DataLoader};

/// Configuration for supervised T2L training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisedTrainerConfig {
    /// Base training configuration
    pub base_config: TrainingConfig,
    
    /// Base model configuration
    pub base_model: BaseModelConfig,
    
    /// Supervised training settings
    pub supervised: SupervisedSettings,
    
    /// LoRA adaptation settings
    pub lora_adaptation: LoraAdaptationConfig,
    
    /// Validation settings
    pub validation: ValidationSettings,
    
    /// Metric tracking settings
    pub metrics: MetricsSettings,
}

/// Base model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseModelConfig {
    /// Model type (llama, mistral, bert, etc.)
    pub model_type: ModelType,
    /// Path to pretrained model weights
    pub model_path: PathBuf,
    /// Model size/variant (e.g., "7b", "13b")
    pub model_size: String,
    /// Model configuration overrides
    pub config_overrides: Option<HashMap<String, serde_json::Value>>,
    /// Whether to load in 8-bit quantization
    pub load_in_8bit: bool,
    /// Whether to load in 4-bit quantization
    pub load_in_4bit: bool,
    /// Device placement strategy
    pub device_map: DeviceMapStrategy,
}

/// Device placement strategy for model parallelism
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceMapStrategy {
    /// Single device
    Single,
    /// Automatic device mapping
    Auto,
    /// Custom layer-to-device mapping
    Custom(HashMap<String, usize>),
}

/// Supervised training settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisedSettings {
    /// Task type for training
    pub task_type: SupervisedTaskType,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Tokenizer settings
    pub tokenizer: TokenizerConfig,
    /// Batch tokenization strategy
    pub batch_tokenization: BatchTokenizationStrategy,
    /// Generation settings (for generation tasks)
    pub generation: Option<GenerationConfig>,
    /// Enable gradient checkpointing for memory efficiency
    pub gradient_checkpointing: bool,
    /// Number of unfrozen layers at the top (0 = all frozen)
    pub unfreeze_top_layers: usize,
}

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Path to tokenizer
    pub tokenizer_path: Option<PathBuf>,
    /// Padding strategy
    pub padding: PaddingStrategy,
    /// Truncation strategy
    pub truncation: bool,
    /// Add special tokens
    pub add_special_tokens: bool,
    /// Return token type IDs
    pub return_token_type_ids: bool,
}

/// Padding strategies for tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PaddingStrategy {
    /// No padding
    None,
    /// Pad to max length in batch
    Max,
    /// Pad to fixed length
    Fixed(usize),
}

/// Batch tokenization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BatchTokenizationStrategy {
    /// Tokenize on CPU before training
    Eager,
    /// Tokenize on-the-fly during training
    Lazy,
    /// Pre-tokenized inputs
    PreTokenized,
}

/// Generation configuration for text generation tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum new tokens to generate
    pub max_new_tokens: usize,
    /// Temperature for sampling
    pub temperature: f64,
    /// Top-k sampling
    pub top_k: Option<usize>,
    /// Top-p (nucleus) sampling
    pub top_p: Option<f64>,
    /// Repetition penalty
    pub repetition_penalty: Option<f64>,
    /// Whether to use sampling or greedy decoding
    pub do_sample: bool,
}

/// LoRA adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdaptationConfig {
    /// LoRA rank
    pub rank: usize,
    /// LoRA alpha scaling factor
    pub alpha: f32,
    /// LoRA dropout
    pub dropout: f32,
    /// Target modules pattern (e.g., ".*attention.*")
    pub target_modules: Vec<String>,
    /// Whether to merge LoRA weights for inference
    pub merge_weights: bool,
    /// Initialization scale for LoRA
    pub init_scale: f32,
    /// Dynamic rank adjustment
    pub dynamic_rank: Option<DynamicRankConfig>,
}

/// Dynamic rank adjustment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicRankConfig {
    /// Minimum rank
    pub min_rank: usize,
    /// Maximum rank
    pub max_rank: usize,
    /// Rank adjustment interval (steps)
    pub adjustment_interval: usize,
    /// Adjustment based on metric
    pub adjustment_metric: String,
}

/// Validation settings for supervised training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSettings {
    /// Compute task-specific metrics
    pub compute_task_metrics: bool,
    /// Compute generation quality metrics
    pub compute_generation_metrics: bool,
    /// Sample predictions to log
    pub log_predictions: bool,
    /// Number of predictions to log
    pub num_predictions_to_log: usize,
    /// Evaluation strategy
    pub eval_strategy: EvalStrategy,
    /// Best model metric
    pub best_model_metric: String,
}

/// Evaluation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalStrategy {
    /// Evaluate every N steps
    Steps(usize),
    /// Evaluate every epoch
    Epoch,
    /// No evaluation during training
    No,
}

/// Metrics tracking settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSettings {
    /// Log LoRA parameter statistics
    pub log_lora_stats: bool,
    /// Log base model activation statistics
    pub log_activation_stats: bool,
    /// Track GPU memory usage
    pub track_gpu_memory: bool,
    /// Log interval for detailed metrics
    pub detailed_metrics_interval: usize,
}

/// Container for base model and its components
pub struct BaseModelContainer {
    /// The base model
    pub model: Box<dyn BaseModel>,
    /// Model configuration
    pub config: ModelConfig,
    /// Tokenizer
    pub tokenizer: Option<Box<dyn Tokenizer>>,
    /// Original state dict for restoration
    pub original_state: Option<HashMap<String, Tensor>>,
}

unsafe impl Send for BaseModelContainer {}
unsafe impl Sync for BaseModelContainer {}

/// Tokenizer trait for different model types
pub trait Tokenizer: Send + Sync {
    /// Tokenize a batch of texts
    fn tokenize_batch(&self, texts: &[String]) -> Result<TokenizedBatch>;
    /// Decode token IDs to text
    fn decode(&self, token_ids: &[u32]) -> Result<String>;
    /// Get vocabulary size
    fn vocab_size(&self) -> usize;
    /// Get padding token ID
    fn pad_token_id(&self) -> Option<u32>;
}

/// Tokenized batch output
#[derive(Debug)]
pub struct TokenizedBatch {
    /// Input IDs tensor [batch_size, seq_len]
    pub input_ids: Tensor,
    /// Attention mask [batch_size, seq_len]
    pub attention_mask: Tensor,
    /// Token type IDs (optional) [batch_size, seq_len]
    pub token_type_ids: Option<Tensor>,
}

/// LoRA-adapted model wrapper
pub struct AdaptedModel {
    /// Base model reference
    base_model: Arc<RwLock<BaseModelContainer>>,
    /// Active LoRA adapters by layer name
    lora_adapters: HashMap<String, LoraAdapter>,
    /// Device
    device: Device,
}

impl AdaptedModel {
    /// Apply LoRA parameters to create adapted model
    pub fn new(
        base_model: Arc<RwLock<BaseModelContainer>>,
        lora_params: HashMap<String, (Tensor, Tensor)>,
        config: &LoraAdaptationConfig,
        device: Device,
    ) -> Result<Self> {
        let mut lora_adapters = HashMap::new();
        
        // Create LoRA adapters for each layer
        for (layer_name, (a_matrix, b_matrix)) in lora_params {
            let adapter = LoraAdapter::from_matrices(
                a_matrix,
                b_matrix,
                config.alpha,
                config.dropout,
            )?;
            lora_adapters.insert(layer_name, adapter);
        }
        
        Ok(Self {
            base_model,
            lora_adapters,
            device,
        })
    }
    
    /// Forward pass through adapted model
    pub fn forward(&self, batch: &SupervisedBatch) -> Result<SupervisedPredictions> {
        // This is a simplified implementation - in practice, you'd need
        // to properly integrate with the specific base model architecture
        let base_model = self.base_model.read();
        
        // Get input tensors
        let input_ids = batch.input_ids.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Input IDs required for forward pass"))?;
        let attention_mask = batch.attention_mask.as_ref();
        
        // Forward through base model with LoRA adaptation
        // This would need proper implementation based on model architecture
        let logits = Tensor::zeros((batch.batch_size, 100, 50000), DType::F32, &self.device)?;
        
        Ok(SupervisedPredictions {
            logits,
            hidden_states: None,
            attention_weights: None,
            lora_params: Some(
                self.lora_adapters.iter()
                    .map(|(name, adapter)| {
                        (name.clone(), adapter.get_parameters())
                    })
                    .collect()
            ),
        })
    }
}

/// Supervised trainer for T2L
pub struct SupervisedTrainer {
    /// Configuration
    config: SupervisedTrainerConfig,
    
    /// Hypernetwork model
    hypernetwork: Arc<RwLock<HyperNetwork>>,
    
    /// Base model container
    base_model: Arc<RwLock<BaseModelContainer>>,
    
    /// Variable map for hypernetwork parameters
    var_map: VarMap,
    
    /// Optimizer state (only for hypernetwork)
    optimizer: OptimizerState,
    
    /// Learning rate scheduler
    scheduler: SchedulerState,
    
    /// Training data loader
    train_loader: DataLoader<Box<dyn Dataset>>,
    
    /// Validation data loader
    val_loader: Option<DataLoader<Box<dyn Dataset>>>,
    
    /// Supervised loss function
    loss_fn: SupervisedLoss,
    
    /// Checkpoint manager
    checkpoint_manager: CheckpointManager,
    
    /// Metrics tracker
    metrics: MetricsTracker,
    
    /// Training state
    state: TrainingState,
    
    /// Device for training
    device: Device,
    
    /// Event channel for monitoring
    event_tx: Option<mpsc::UnboundedSender<TrainingEvent>>,
    
    /// Current adapted model (reused across batches)
    current_adapted_model: Option<AdaptedModel>,
    
    /// Task-specific metrics computer
    task_metrics: TaskMetricsComputer,
}

/// Computes task-specific metrics
struct TaskMetricsComputer {
    task_type: SupervisedTaskType,
}

impl TaskMetricsComputer {
    fn new(task_type: SupervisedTaskType) -> Self {
        Self { task_type }
    }
    
    fn compute_metrics(&self, predictions: &Tensor, labels: &Tensor) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        match &self.task_type {
            SupervisedTaskType::Classification { num_classes } => {
                // Compute accuracy
                if let Ok(acc) = self.compute_accuracy(predictions, labels) {
                    metrics.insert("accuracy".to_string(), acc);
                }
                
                // Compute per-class F1 if reasonable number of classes
                if *num_classes <= 10 {
                    if let Ok(f1_scores) = self.compute_f1_scores(predictions, labels, *num_classes) {
                        for (i, f1) in f1_scores.iter().enumerate() {
                            metrics.insert(format!("f1_class_{}", i), *f1);
                        }
                    }
                }
            },
            SupervisedTaskType::Generation => {
                // Compute perplexity
                if let Ok(ppl) = self.compute_perplexity(predictions, labels) {
                    metrics.insert("perplexity".to_string(), ppl);
                }
            },
            SupervisedTaskType::SequenceLabeling { num_labels } => {
                // Token-level accuracy
                if let Ok(acc) = self.compute_token_accuracy(predictions, labels) {
                    metrics.insert("token_accuracy".to_string(), acc);
                }
            },
            SupervisedTaskType::MaskedLM => {
                // Masked token accuracy
                if let Ok(acc) = self.compute_masked_accuracy(predictions, labels) {
                    metrics.insert("masked_accuracy".to_string(), acc);
                }
            },
        }
        
        metrics
    }
    
    fn compute_accuracy(&self, predictions: &Tensor, labels: &Tensor) -> Result<f64> {
        let preds = predictions.argmax_keepdim(candle_core::D::Minus1)?;
        let correct = preds.eq(labels)?.to_dtype(DType::F32)?;
        Ok(correct.mean_all()?.to_scalar::<f64>()?)
    }
    
    fn compute_f1_scores(&self, predictions: &Tensor, labels: &Tensor, num_classes: usize) -> Result<Vec<f64>> {
        // Simplified F1 computation
        Ok(vec![0.0; num_classes])
    }
    
    fn compute_perplexity(&self, predictions: &Tensor, labels: &Tensor) -> Result<f64> {
        // Cross entropy then exp
        Ok(2.0) // Placeholder
    }
    
    fn compute_token_accuracy(&self, predictions: &Tensor, labels: &Tensor) -> Result<f64> {
        self.compute_accuracy(predictions, labels)
    }
    
    fn compute_masked_accuracy(&self, predictions: &Tensor, labels: &Tensor) -> Result<f64> {
        // Would need to handle mask properly
        self.compute_accuracy(predictions, labels)
    }
}

impl SupervisedTrainer {
    /// Create a new supervised trainer
    pub fn new(
        config: SupervisedTrainerConfig,
        hypernetwork: Arc<RwLock<HyperNetwork>>,
        base_model_container: BaseModelContainer,
        train_loader: DataLoader<Box<dyn Dataset>>,
        val_loader: Option<DataLoader<Box<dyn Dataset>>>,
        device: Device,
    ) -> Result<Self> {
        // Validate configuration
        config.base_config.validate()
            .context("Training configuration validation failed")?;
        
        // Initialize variable map for hypernetwork
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        
        // Initialize optimizer (only for hypernetwork parameters)
        let optimizer = create_optimizer(&config.base_config.optimizer, &var_map)?;
        
        // Initialize scheduler
        let scheduler = create_scheduler(
            &config.base_config.optimizer.scheduler,
            config.base_config.optimizer.learning_rate
        )?;
        
        // Create supervised loss with config
        let loss_config = SupervisedLossConfig {
            base: config.base_config.loss.clone(),
            task_type: config.supervised.task_type.clone(),
            label_smoothing: 0.1, // Could be in config
            lora_reg_weight: config.lora_adaptation.alpha as f64 * 0.01,
            focal_loss: None,
            class_weights: None,
            token_weights: None,
            ignore_index: Some(-100),
        };
        let loss_fn = SupervisedLoss::with_config(loss_config, &device)?;
        
        // Initialize checkpoint manager
        let checkpoint_manager = CheckpointManager::new(
            config.base_config.checkpointing.clone(),
            device.clone(),
        )?;
        
        // Initialize metrics tracker
        let metrics = MetricsTracker::new(config.base_config.logging.clone())?;
        
        // Initialize training state
        let state = TrainingState::new();
        
        // Wrap base model in Arc<RwLock>
        let base_model = Arc::new(RwLock::new(base_model_container));
        
        // Initialize task metrics computer
        let task_metrics = TaskMetricsComputer::new(config.supervised.task_type.clone());
        
        Ok(Self {
            config,
            hypernetwork,
            base_model,
            var_map,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            loss_fn,
            checkpoint_manager,
            metrics,
            state,
            device,
            event_tx: None,
            current_adapted_model: None,
            task_metrics,
        })
    }
    
    /// Set up event monitoring
    pub fn with_event_monitoring(mut self, tx: mpsc::UnboundedSender<TrainingEvent>) -> Self {
        self.event_tx = Some(tx);
        self
    }
    
    /// Start supervised training
    #[instrument(skip(self))]
    pub async fn train(&mut self) -> Result<TrainingResult> {
        info!("Starting supervised T2L training");
        
        // Initialize training
        self.initialize_training()?;
        
        // Send training started event
        self.send_event(TrainingEvent::EpochStarted { epoch: 0 });
        
        let training_start = Instant::now();
        
        // Main training loop
        let result = match self.run_training_loop().await {
            Ok(metrics) => {
                self.state.status = TrainingStatus::Completed;
                TrainingResult {
                    final_metrics: metrics,
                    final_state: self.state.clone(),
                    best_checkpoint_path: self.checkpoint_manager.best_checkpoint_path(),
                    training_duration: training_start.elapsed(),
                    total_steps: self.state.global_step,
                    success: true,
                    error_message: None,
                }
            }
            Err(e) => {
                let error_msg = format!("Training failed: {}", e);
                error!("{}", error_msg);
                self.state.status = TrainingStatus::Failed { error: error_msg.clone() };
                self.send_event(TrainingEvent::Error { error: error_msg.clone() });
                
                TrainingResult {
                    final_metrics: self.metrics.current_metrics(),
                    final_state: self.state.clone(),
                    best_checkpoint_path: self.checkpoint_manager.best_checkpoint_path(),
                    training_duration: training_start.elapsed(),
                    total_steps: self.state.global_step,
                    success: false,
                    error_message: Some(error_msg),
                }
            }
        };
        
        // Final cleanup
        self.finalize_training()?;
        
        info!("Supervised training completed in {:?}", result.training_duration);
        self.send_event(TrainingEvent::TrainingCompleted { 
            total_steps: result.total_steps 
        });
        
        Ok(result)
    }
    
    /// Initialize training setup
    fn initialize_training(&mut self) -> Result<()> {
        info!("Initializing supervised training setup");
        
        // Set training status
        self.state.status = TrainingStatus::Running;
        self.state.start_time = Utc::now();
        
        // Freeze base model parameters
        info!("Freezing base model parameters");
        // Note: In practice, you'd set requires_grad = False on base model params
        
        // Resume from checkpoint if specified
        if let Some(checkpoint_path) = &self.config.base_config.training.resume_from_checkpoint {
            info!("Resuming from checkpoint: {:?}", checkpoint_path);
            self.load_checkpoint(checkpoint_path)?;
        }
        
        // Log initial state
        info!("Device: {:?}", self.device);
        info!("Task type: {:?}", self.config.supervised.task_type);
        info!("LoRA rank: {}", self.config.lora_adaptation.rank);
        info!("Base model: {} ({})", self.config.base_model.model_type, self.config.base_model.model_size);
        
        Ok(())
    }
    
    /// Main training loop
    async fn run_training_loop(&mut self) -> Result<TrainingMetrics> {
        let total_epochs = self.config.base_config.training.num_epochs;
        
        for epoch in self.state.epoch..total_epochs {
            self.state.epoch = epoch;
            
            info!("Starting epoch {}/{}", epoch + 1, total_epochs);
            self.send_event(TrainingEvent::EpochStarted { epoch });
            
            // Run training epoch
            let train_metrics = self.train_epoch().await?;
            
            // Run evaluation based on strategy
            let eval_metrics = match self.config.validation.eval_strategy {
                EvalStrategy::Epoch => {
                    if self.val_loader.is_some() {
                        Some(self.evaluate().await?)
                    } else {
                        None
                    }
                },
                EvalStrategy::Steps(_) => None, // Handled in train_epoch
                EvalStrategy::No => None,
            };
            
            // Combine metrics
            let epoch_metrics = self.combine_metrics(train_metrics, eval_metrics);
            
            // Update learning rate scheduler
            let scheduler_metric = epoch_metrics.get(&self.config.validation.best_model_metric)
                .or_else(|| epoch_metrics.get("eval_loss"))
                .cloned();
            self.scheduler.step(scheduler_metric);
            self.state.current_lr = self.scheduler.get_lr();
            
            // Log epoch completion
            info!("Epoch {}/{} completed - Train Loss: {:.4}, Eval Loss: {:.4}", 
                  epoch + 1, total_epochs, 
                  epoch_metrics.get("train_loss").unwrap_or(&0.0),
                  epoch_metrics.get("eval_loss").unwrap_or(&0.0));
            
            self.send_event(TrainingEvent::EpochCompleted { 
                epoch, 
                metrics: self.metrics.current_metrics()
            });
            
            // Save checkpoint
            self.save_checkpoint(epoch, &epoch_metrics).await?;
            
            // Check early stopping
            if self.should_early_stop(&epoch_metrics) {
                info!("Early stopping triggered");
                self.state.status = TrainingStatus::EarlyStopped;
                self.send_event(TrainingEvent::EarlyStopping { 
                    reason: "No improvement in validation metrics".to_string() 
                });
                break;
            }
            
            // Check if max steps reached
            if let Some(max_steps) = self.config.base_config.training.max_steps {
                if self.state.global_step >= max_steps {
                    info!("Maximum steps reached: {}", max_steps);
                    break;
                }
            }
        }
        
        Ok(self.metrics.current_metrics())
    }
    
    /// Train for one epoch
    #[instrument(skip(self))]
    async fn train_epoch(&mut self) -> Result<HashMap<String, f64>> {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        let mut task_metrics_accum: HashMap<String, f64> = HashMap::new();
        
        // Iterate through training batches
        while let Some(batch) = self.train_loader.next_batch().await? {
            let step_start = Instant::now();
            
            // Convert to supervised batch
            let batch = batch.downcast::<SupervisedBatch>()
                .map_err(|_| anyhow::anyhow!("Invalid batch type for supervised training"))?;
            
            // Tokenize batch if needed
            let tokenized_batch = self.tokenize_batch(&batch)?;
            
            // Forward pass and loss computation
            let (loss, loss_metrics, task_metrics) = self.supervised_step(tokenized_batch)?;
            
            // Backward pass
            let gradients = loss.backward()?;
            
            // Optimizer step (only updates hypernetwork parameters)
            self.optimizer.step(&gradients)?;
            
            // Clear gradients
            self.var_map.all_vars().iter().for_each(|(_, var)| {
                // Clear gradients - implementation needed
            });
            
            // Update metrics
            epoch_loss += loss_metrics.total_loss;
            num_batches += 1;
            
            // Accumulate task metrics
            for (metric_name, value) in task_metrics {
                *task_metrics_accum.entry(metric_name).or_insert(0.0) += value;
            }
            
            // Log step progress
            if self.state.global_step % self.config.base_config.training.log_steps == 0 {
                self.log_training_step(
                    &loss_metrics,
                    step_start.elapsed(),
                    self.scheduler.get_lr()
                );
            }
            
            // Log LoRA statistics
            if self.config.metrics.log_lora_stats &&
               self.state.global_step % self.config.metrics.detailed_metrics_interval == 0 {
                self.log_lora_statistics();
            }
            
            // Evaluation based on steps strategy
            if let EvalStrategy::Steps(eval_steps) = self.config.validation.eval_strategy {
                if self.state.global_step % eval_steps == 0 && self.val_loader.is_some() {
                    let eval_metrics = self.evaluate().await?;
                    self.send_event(TrainingEvent::EvaluationCompleted { 
                        metrics: self.metrics.current_metrics() 
                    });
                }
            }
            
            // Save checkpoint
            if self.state.global_step % self.config.base_config.training.save_steps == 0 {
                let metrics = HashMap::new(); // Simplified for step saves
                self.save_checkpoint(self.state.epoch, &metrics).await?;
            }
            
            self.state.step += 1;
            self.state.global_step += 1;
            
            // Update memory usage
            self.update_memory_usage();
        }
        
        // Compute average metrics
        let avg_loss = if num_batches > 0 { epoch_loss / num_batches as f64 } else { 0.0 };
        let mut metrics = HashMap::new();
        metrics.insert("train_loss".to_string(), avg_loss);
        
        // Add average task metrics
        for (metric_name, total) in task_metrics_accum {
            let avg = if num_batches > 0 { total / num_batches as f64 } else { 0.0 };
            metrics.insert(format!("train_{}", metric_name), avg);
        }
        
        // Add loss to history
        self.state.loss_history.push(avg_loss);
        if self.state.loss_history.len() > 100 {
            self.state.loss_history.remove(0);
        }
        
        Ok(metrics)
    }
    
    /// Perform a supervised training step
    fn supervised_step(&mut self, batch: SupervisedBatch) -> Result<(Tensor, LossFunctionMetrics, HashMap<String, f64>)> {
        let hypernetwork = self.hypernetwork.read();
        
        // Generate LoRA parameters from task description
        let lora_params = self.generate_lora_params(&hypernetwork, &batch)?;
        
        // Create adapted model with LoRA parameters
        let adapted_model = AdaptedModel::new(
            self.base_model.clone(),
            lora_params,
            &self.config.lora_adaptation,
            self.device.clone(),
        )?;
        
        // Forward pass through adapted model
        let predictions = adapted_model.forward(&batch)?;
        
        // Compute loss
        let loss_metrics = self.loss_fn.compute_metrics(
            &batch as &dyn std::any::Any,
            &predictions as &dyn std::any::Any
        )?;
        
        // Create loss tensor
        let loss = Tensor::full(loss_metrics.total_loss, (), &self.device)?;
        
        // Compute task-specific metrics
        let task_metrics = if let (Some(labels), logits) = (&batch.labels, &predictions.logits) {
            self.task_metrics.compute_metrics(logits, labels)
        } else {
            HashMap::new()
        };
        
        // Store current adapted model for potential reuse
        self.current_adapted_model = Some(adapted_model);
        
        Ok((loss, loss_metrics, task_metrics))
    }
    
    /// Generate LoRA parameters using hypernetwork
    fn generate_lora_params(
        &self,
        hypernetwork: &HyperNetwork,
        batch: &SupervisedBatch
    ) -> Result<HashMap<String, (Tensor, Tensor)>> {
        let mut all_lora_params = HashMap::new();
        
        // Process each sample in the batch
        for i in 0..batch.batch_size {
            // Get task embedding for this sample
            let task_embedding = if let Some(ref embeddings) = batch.task_embeddings {
                embeddings.i(i)?
            } else {
                // Generate embedding from task description
                self.generate_task_embedding(&batch.task_descriptions[i])?
            };
            
            // Convert to ndarray for hypernetwork
            let embedding_array = self.tensor_to_array1(&task_embedding)?;
            
            // Determine target architecture based on base model
            let target_arch = self.get_target_architecture()?;
            
            // Generate LoRA parameters
            let lora_params = hypernetwork.generate_lora_params(&embedding_array, target_arch)?;
            
            // Convert to tensors and accumulate
            for (layer_name, layer_params) in lora_params.layers {
                let a_tensor = self.array_to_tensor(&layer_params.matrix_a)?;
                let b_tensor = self.array_to_tensor(&layer_params.matrix_b)?;
                
                if i == 0 {
                    all_lora_params.insert(layer_name.clone(), (a_tensor, b_tensor));
                } else {
                    // Stack with existing params
                    let (existing_a, existing_b) = all_lora_params.get_mut(&layer_name).unwrap();
                    *existing_a = Tensor::cat(&[existing_a, &a_tensor.unsqueeze(0)?], 0)?;
                    *existing_b = Tensor::cat(&[existing_b, &b_tensor.unsqueeze(0)?], 0)?;
                }
            }
        }
        
        Ok(all_lora_params)
    }
    
    /// Tokenize batch if needed
    fn tokenize_batch(&self, batch: &SupervisedBatch) -> Result<SupervisedBatch> {
        match self.config.supervised.batch_tokenization {
            BatchTokenizationStrategy::PreTokenized => Ok(batch.clone()),
            BatchTokenizationStrategy::Eager | BatchTokenizationStrategy::Lazy => {
                // Would need actual tokenizer implementation
                let mut tokenized = batch.clone();
                
                // Placeholder tokenization
                let seq_len = 128;
                let vocab_size = 50000;
                
                tokenized.input_ids = Some(Tensor::zeros(
                    (batch.batch_size, seq_len),
                    DType::I64,
                    &self.device
                )?);
                
                tokenized.attention_mask = Some(Tensor::ones(
                    (batch.batch_size, seq_len),
                    DType::I64,
                    &self.device
                )?);
                
                Ok(tokenized)
            }
        }
    }
    
    /// Evaluation loop
    #[instrument(skip(self))]
    async fn evaluate(&mut self) -> Result<HashMap<String, f64>> {
        if self.val_loader.is_none() {
            return Ok(HashMap::new());
        }
        
        info!("Running evaluation");
        
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        let mut task_metrics_accum: HashMap<String, f64> = HashMap::new();
        let mut sample_predictions = Vec::new();
        
        let val_loader = self.val_loader.as_mut().unwrap();
        
        while let Some(batch) = val_loader.next_batch().await? {
            let batch = batch.downcast::<SupervisedBatch>()
                .map_err(|_| anyhow::anyhow!("Invalid batch type for supervised evaluation"))?;
            
            // Tokenize batch if needed
            let tokenized_batch = self.tokenize_batch(&batch)?;
            
            // Forward pass without gradient computation
            let (_, loss_metrics, task_metrics) = self.supervised_step(tokenized_batch)?;
            
            // Accumulate metrics
            total_loss += loss_metrics.total_loss;
            num_batches += 1;
            
            for (metric_name, value) in task_metrics {
                *task_metrics_accum.entry(metric_name).or_insert(0.0) += value;
            }
            
            // Collect sample predictions for logging
            if self.config.validation.log_predictions && 
               sample_predictions.len() < self.config.validation.num_predictions_to_log {
                // Would collect actual predictions here
                sample_predictions.push("Sample prediction".to_string());
            }
        }
        
        // Compute average metrics
        let avg_loss = if num_batches > 0 { total_loss / num_batches as f64 } else { 0.0 };
        let mut metrics = HashMap::new();
        metrics.insert("eval_loss".to_string(), avg_loss);
        
        // Add average task metrics
        for (metric_name, total) in task_metrics_accum {
            let avg = if num_batches > 0 { total / num_batches as f64 } else { 0.0 };
            metrics.insert(format!("eval_{}", metric_name), avg);
        }
        
        // Log sample predictions
        if self.config.validation.log_predictions && !sample_predictions.is_empty() {
            info!("Sample predictions:");
            for (i, pred) in sample_predictions.iter().enumerate().take(5) {
                info!("  [{}] {}", i, pred);
            }
        }
        
        self.state.last_eval_time = Some(Utc::now());
        
        info!("Evaluation completed - Loss: {:.4}", avg_loss);
        
        Ok(metrics)
    }
    
    /// Generate task embedding from description
    fn generate_task_embedding(&self, task_description: &str) -> Result<Tensor> {
        // Placeholder - would use actual text encoder
        let embedding_dim = self.hypernetwork.read().config().input_dim;
        Ok(Tensor::randn(0.0, 1.0, (embedding_dim,), &self.device)?)
    }
    
    /// Get target architecture for base model
    fn get_target_architecture(&self) -> Result<TargetArchitecture> {
        // Based on model type, return appropriate architecture
        match self.config.base_model.model_type {
            ModelType::BERT => Ok(TargetArchitecture::BERT {
                hidden_size: 768,
                num_layers: 12,
                num_heads: 12,
            }),
            ModelType::LLaMA => Ok(TargetArchitecture::LLAMA {
                hidden_size: 4096,
                num_layers: 32,
                num_heads: 32,
                intermediate_size: 11008,
            }),
            _ => Err(anyhow::anyhow!("Unsupported model type for target architecture")),
        }
    }
    
    /// Convert Tensor to ndarray
    fn tensor_to_array1(&self, tensor: &Tensor) -> Result<ndarray::Array1<f32>> {
        let data = tensor.to_vec1::<f32>()?;
        Ok(ndarray::Array1::from_vec(data))
    }
    
    /// Convert ndarray to Tensor
    fn array_to_tensor(&self, array: &ndarray::Array2<f32>) -> Result<Tensor> {
        let shape = array.shape();
        let data = array.as_slice()
            .ok_or_else(|| anyhow::anyhow!("Failed to get slice from array"))?;
        Ok(Tensor::from_slice(data, &[shape[0], shape[1]], &self.device)?)
    }
    
    /// Log training step information
    fn log_training_step(&self, loss_metrics: &LossFunctionMetrics, step_time: Duration, lr: f64) {
        debug!(
            "Step {} - Loss: {:.4}, LR: {:.2e}, Time: {:?}", 
            self.state.global_step, 
            loss_metrics.total_loss,
            lr,
            step_time
        );
        
        self.send_event(TrainingEvent::StepCompleted { 
            step: self.state.global_step, 
            loss: loss_metrics.total_loss,
            lr,
        });
    }
    
    /// Log LoRA parameter statistics
    fn log_lora_statistics(&self) {
        if let Some(ref adapted_model) = self.current_adapted_model {
            info!("LoRA parameter statistics at step {}:", self.state.global_step);
            
            for (layer_name, adapter) in &adapted_model.lora_adapters {
                let (a, b) = adapter.get_parameters();
                
                // Compute statistics
                if let Ok(a_norm) = a.sqr().unwrap().mean_all().unwrap().sqrt().unwrap().to_scalar::<f64>() {
                    if let Ok(b_norm) = b.sqr().unwrap().mean_all().unwrap().sqrt().unwrap().to_scalar::<f64>() {
                        info!("  Layer {}: A_norm={:.6}, B_norm={:.6}", layer_name, a_norm, b_norm);
                    }
                }
            }
        }
    }
    
    /// Check if early stopping should be triggered
    fn should_early_stop(&mut self, metrics: &HashMap<String, f64>) -> bool {
        let early_stop_config = &self.config.base_config.training.early_stopping;
        
        if !early_stop_config.enabled {
            return false;
        }
        
        let monitor_metric = &self.config.validation.best_model_metric;
        let higher_is_better = early_stop_config.higher_is_better;
        let min_delta = early_stop_config.min_delta;
        let patience = early_stop_config.patience;
        
        if let Some(&current_score) = metrics.get(monitor_metric) {
            let improved = match self.state.best_score {
                None => true,
                Some(best_score) => {
                    if higher_is_better {
                        current_score > best_score + min_delta
                    } else {
                        current_score < best_score - min_delta
                    }
                }
            };
            
            if improved {
                self.state.best_score = Some(current_score);
                self.state.steps_since_best = 0;
                false
            } else {
                self.state.steps_since_best += 1;
                self.state.steps_since_best >= patience
            }
        } else {
            false
        }
    }
    
    /// Save training checkpoint
    async fn save_checkpoint(&mut self, epoch: usize, metrics: &HashMap<String, f64>) -> Result<()> {
        let checkpoint = TrainingCheckpoint {
            epoch,
            global_step: self.state.global_step,
            model_state: self.serialize_model_state()?,
            optimizer_state: self.optimizer.state_dict()?,
            scheduler_state: self.scheduler.state_dict()?,
            metrics: metrics.clone(),
            config: self.config.base_config.clone(),
            timestamp: Utc::now(),
        };
        
        let path = self.checkpoint_manager.save_checkpoint(checkpoint).await?;
        
        self.send_event(TrainingEvent::CheckpointSaved { path });
        
        Ok(())
    }
    
    /// Load training checkpoint
    fn load_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let checkpoint = self.checkpoint_manager.load_checkpoint(path)?;
        
        // Restore training state
        self.state.epoch = checkpoint.epoch;
        self.state.global_step = checkpoint.global_step;
        
        // Restore hypernetwork state
        self.deserialize_model_state(&checkpoint.model_state)?;
        
        // Restore optimizer state
        if let Some(optimizer_state) = checkpoint.optimizer_state {
            self.optimizer.load_state_dict(optimizer_state)?;
        }
        
        // Restore scheduler state
        if let Some(scheduler_state) = checkpoint.scheduler_state {
            self.scheduler.load_state_dict(scheduler_state)?;
        }
        
        info!("Loaded checkpoint from epoch {}, step {}", 
              checkpoint.epoch, checkpoint.global_step);
        
        Ok(())
    }
    
    /// Serialize model state for checkpointing
    fn serialize_model_state(&self) -> Result<Vec<u8>> {
        // This would serialize the hypernetwork parameters
        Ok(vec![])
    }
    
    /// Deserialize model state from checkpoint
    fn deserialize_model_state(&mut self, state: &[u8]) -> Result<()> {
        // This would deserialize and load hypernetwork parameters
        Ok(())
    }
    
    /// Combine training and evaluation metrics
    fn combine_metrics(
        &self, 
        train_metrics: HashMap<String, f64>, 
        eval_metrics: Option<HashMap<String, f64>>
    ) -> HashMap<String, f64> {
        let mut combined = train_metrics;
        
        if let Some(eval_metrics) = eval_metrics {
            for (key, value) in eval_metrics {
                combined.insert(key, value);
            }
        }
        
        combined
    }
    
    /// Update memory usage tracking
    fn update_memory_usage(&mut self) {
        if self.config.metrics.track_gpu_memory {
            // This would track actual memory usage
            self.state.memory_usage.current_memory_bytes = 0;
            
            // Track GPU memory if available
            if self.device.is_cuda() {
                self.state.memory_usage.gpu_memory_bytes = Some(0);
            }
        }
    }
    
    /// Finalize training cleanup
    fn finalize_training(&mut self) -> Result<()> {
        info!("Finalizing supervised training");
        
        // Remove any active LoRA adapters
        self.current_adapted_model = None;
        
        // Log final statistics
        if self.config.metrics.log_lora_stats {
            self.log_lora_statistics();
        }
        
        Ok(())
    }
    
    /// Send training event
    fn send_event(&self, event: TrainingEvent) {
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(event);
        }
    }
    
    /// Get current training state
    pub fn state(&self) -> &TrainingState {
        &self.state
    }
    
    /// Get current metrics
    pub fn metrics(&self) -> TrainingMetrics {
        self.metrics.current_metrics()
    }
}


impl Default for SupervisedTrainerConfig {
    fn default() -> Self {
        Self {
            base_config: TrainingConfig::supervised_default(),
            base_model: BaseModelConfig {
                model_type: ModelType::BERT,
                model_path: PathBuf::from("bert-base-uncased"),
                model_size: "base".to_string(),
                config_overrides: None,
                load_in_8bit: false,
                load_in_4bit: false,
                device_map: DeviceMapStrategy::Single,
            },
            supervised: SupervisedSettings {
                task_type: SupervisedTaskType::Classification { num_classes: 2 },
                max_seq_length: 512,
                tokenizer: TokenizerConfig {
                    tokenizer_path: None,
                    padding: PaddingStrategy::Max,
                    truncation: true,
                    add_special_tokens: true,
                    return_token_type_ids: true,
                },
                batch_tokenization: BatchTokenizationStrategy::Eager,
                generation: None,
                gradient_checkpointing: false,
                unfreeze_top_layers: 0,
            },
            lora_adaptation: LoraAdaptationConfig {
                rank: 16,
                alpha: 32.0,
                dropout: 0.1,
                target_modules: vec![".*attention.*".to_string()],
                merge_weights: false,
                init_scale: 0.01,
                dynamic_rank: None,
            },
            validation: ValidationSettings {
                compute_task_metrics: true,
                compute_generation_metrics: false,
                log_predictions: true,
                num_predictions_to_log: 10,
                eval_strategy: EvalStrategy::Epoch,
                best_model_metric: "eval_accuracy".to_string(),
            },
            metrics: MetricsSettings {
                log_lora_stats: true,
                log_activation_stats: false,
                track_gpu_memory: true,
                detailed_metrics_interval: 100,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trainer_config_creation() {
        let config = SupervisedTrainerConfig::default();
        assert_eq!(config.lora_adaptation.rank, 16);
        assert_eq!(config.base_model.model_type, ModelType::BERT);
    }
    
    #[test]
    fn test_task_metrics_computer() {
        let computer = TaskMetricsComputer::new(SupervisedTaskType::Classification { num_classes: 2 });
        let metrics = computer.compute_metrics(
            &Tensor::zeros((10, 2), DType::F32, &Device::Cpu).unwrap(),
            &Tensor::zeros((10,), DType::I64, &Device::Cpu).unwrap()
        );
        assert!(metrics.contains_key("accuracy"));
    }
}