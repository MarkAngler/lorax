//! Specialized trainer for reconstruction-based T2L training
//!
//! This module implements a trainer specifically designed for reconstruction-based
//! training where the hypernetwork learns to generate LoRA parameters from task
//! embeddings by reconstructing existing LoRA weights.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, Context};
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::{VarBuilder, VarMap};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn, instrument};

use crate::hypernetwork::{HyperNetwork, HypernetworkConfig, TargetArchitecture};
use crate::lora::parameters::LoraParameters;
use crate::training::{
    TrainingConfig, TrainingState, TrainingStatus, TrainingResult, TrainingEvent,
    T2LTrainer, MemoryUsage, ReconstructionBatch,
    CheckpointManager, TrainingCheckpoint, MetricsTracker, TrainingMetrics,
    OptimizerState, SchedulerState, create_optimizer, create_scheduler,
    ReconstructionLoss, LossFunctionMetrics, PredictedLoraParams, PredictedLoraLayer,
};
use crate::training::config::LogLevel;
use crate::training::loss::LossFunction;
use crate::training::data::{Dataset, DataLoader};

/// Configuration specific to reconstruction training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionTrainerConfig {
    /// Base training configuration
    pub base_config: TrainingConfig,
    
    /// Reconstruction-specific settings
    pub reconstruction: ReconstructionSettings,
    
    /// Validation settings
    pub validation: ValidationSettings,
    
    /// Metric tracking settings
    pub metrics: MetricsSettings,
}

/// Reconstruction-specific training settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionSettings {
    /// Layer-wise loss weighting strategy
    pub layer_weighting: LayerWeightingStrategy,
    
    /// Gradient accumulation for large LoRA parameters
    pub gradient_accumulation_mode: GradientAccumulationMode,
    
    /// Progressive training schedule
    pub progressive_training: Option<ProgressiveTrainingConfig>,
    
    /// Parameter norm clipping threshold
    pub param_norm_clip: Option<f64>,
    
    /// Enable parameter magnitude tracking
    pub track_param_magnitudes: bool,
    
    /// Enable gradient flow analysis
    pub analyze_gradient_flow: bool,
}

/// Layer weighting strategies for loss computation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LayerWeightingStrategy {
    /// Equal weight for all layers
    Uniform,
    /// Weight by parameter count
    ByParameterCount,
    /// Weight by layer depth (earlier layers get more weight)
    ByDepth { decay_factor: f64 },
    /// Custom weights per layer type
    Custom(HashMap<String, f64>),
}

/// Gradient accumulation modes
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GradientAccumulationMode {
    /// Standard fixed-step accumulation
    Fixed { steps: usize },
    /// Dynamic based on batch memory usage
    Dynamic { target_memory_mb: usize },
    /// Adaptive based on gradient magnitudes
    Adaptive { min_steps: usize, max_steps: usize },
}

/// Progressive training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveTrainingConfig {
    /// Start with these layer patterns
    pub initial_layers: Vec<String>,
    /// Add layers every N epochs
    pub layer_addition_interval: usize,
    /// Warmup steps for new layers
    pub layer_warmup_steps: usize,
}

/// Validation-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSettings {
    /// Compute parameter alignment metrics
    pub compute_alignment: bool,
    /// Compute effective rank metrics
    pub compute_effective_rank: bool,
    /// Track per-layer reconstruction accuracy
    pub track_layer_accuracy: bool,
    /// Save best model based on this metric
    pub best_model_metric: String,
}

/// Metrics tracking settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSettings {
    /// Log parameter statistics every N steps
    pub log_param_stats_interval: usize,
    /// Log gradient flow every N steps
    pub log_gradient_flow_interval: usize,
    /// Track memory usage
    pub track_memory_usage: bool,
    /// Enable detailed profiling
    pub enable_profiling: bool,
}

/// Specialized trainer for reconstruction-based T2L training
pub struct ReconstructionTrainer {
    /// Configuration
    config: ReconstructionTrainerConfig,
    
    /// Hypernetwork model
    model: Arc<RwLock<HyperNetwork>>,
    
    /// Variable map for parameters
    var_map: VarMap,
    
    /// Optimizer state
    optimizer: OptimizerState,
    
    /// Learning rate scheduler
    scheduler: SchedulerState,
    
    /// Training data loader
    train_loader: DataLoader<Box<dyn Dataset>>,
    
    /// Validation data loader
    val_loader: Option<DataLoader<Box<dyn Dataset>>>,
    
    /// Reconstruction loss function
    loss_fn: ReconstructionLoss,
    
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
    
    /// Layer weights for loss computation
    layer_weights: HashMap<String, f64>,
    
    /// Gradient scaler for mixed precision
    grad_scaler: Option<GradientScaler>,
    
    /// Parameter statistics tracker
    param_stats: ParameterStatistics,
    
    /// Active layers for progressive training
    active_layers: Option<Vec<String>>,
}

/// Gradient scaler for mixed precision training
#[cfg(test)]
pub(crate) struct GradientScaler {
    pub scale: f64,
    pub growth_factor: f64,
    pub backoff_factor: f64,
    pub growth_interval: usize,
    pub steps_since_update: usize,
    pub found_inf_count: usize,
}

#[cfg(not(test))]
struct GradientScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: usize,
    steps_since_update: usize,
    found_inf_count: usize,
}

impl GradientScaler {
    #[cfg(test)]
    pub(crate) fn new(initial_scale: f64) -> Self {
        Self {
            scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_update: 0,
            found_inf_count: 0,
        }
    }
    
    #[cfg(not(test))]
    fn new(initial_scale: f64) -> Self {
        Self {
            scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_update: 0,
            found_inf_count: 0,
        }
    }
    
    fn scale_loss(&self, loss: &Tensor) -> Result<Tensor> {
        Ok((loss * self.scale)?)
    }
    
    fn unscale_gradients(&mut self, gradients: &mut candle_core::backprop::GradStore) -> Result<bool> {
        // Check for inf/nan in gradients
        let has_inf_nan = false; // Would need actual implementation
        
        if !has_inf_nan {
            // Unscale gradients
            // Implementation would go here
            self.update_scale(false);
        } else {
            self.update_scale(true);
        }
        
        Ok(!has_inf_nan)
    }
    
    #[cfg(test)]
    pub(crate) fn update_scale(&mut self, found_inf: bool) {
        if found_inf {
            self.scale *= self.backoff_factor;
            self.steps_since_update = 0;
            self.found_inf_count += 1;
        } else {
            self.steps_since_update += 1;
            if self.steps_since_update >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_update = 0;
            }
        }
    }
    
    #[cfg(not(test))]
    fn update_scale(&mut self, found_inf: bool) {
        if found_inf {
            self.scale *= self.backoff_factor;
            self.steps_since_update = 0;
            self.found_inf_count += 1;
        } else {
            self.steps_since_update += 1;
            if self.steps_since_update >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_update = 0;
            }
        }
    }
}

/// Parameter statistics tracking
#[derive(Debug, Clone)]
#[cfg(test)]
pub(crate) struct ParameterStatistics {
    /// Mean parameter magnitudes by layer
    pub layer_magnitudes: HashMap<String, f64>,
    /// Parameter gradient norms by layer
    pub layer_grad_norms: HashMap<String, f64>,
    /// Update ratios (gradient_norm / param_norm)
    pub update_ratios: HashMap<String, f64>,
    /// Effective ranks by layer
    pub effective_ranks: HashMap<String, f64>,
}

#[cfg(not(test))]
struct ParameterStatistics {
    /// Mean parameter magnitudes by layer
    layer_magnitudes: HashMap<String, f64>,
    /// Parameter gradient norms by layer
    layer_grad_norms: HashMap<String, f64>,
    /// Update ratios (gradient_norm / param_norm)
    update_ratios: HashMap<String, f64>,
    /// Effective ranks by layer
    effective_ranks: HashMap<String, f64>,
}

impl ParameterStatistics {
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self {
            layer_magnitudes: HashMap::new(),
            layer_grad_norms: HashMap::new(),
            update_ratios: HashMap::new(),
            effective_ranks: HashMap::new(),
        }
    }
    
    #[cfg(not(test))]
    fn new() -> Self {
        Self {
            layer_magnitudes: HashMap::new(),
            layer_grad_norms: HashMap::new(),
            update_ratios: HashMap::new(),
            effective_ranks: HashMap::new(),
        }
    }
    
    fn update(&mut self, layer_name: &str, params: &Tensor, gradients: Option<&Tensor>) {
        // Compute parameter magnitude
        if let Ok(magnitude) = params.sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar::<f64>() {
            self.layer_magnitudes.insert(layer_name.to_string(), magnitude);
        }
        
        // Compute gradient norm if available
        if let Some(grads) = gradients {
            if let Ok(grad_norm) = grads.sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar::<f64>() {
                self.layer_grad_norms.insert(layer_name.to_string(), grad_norm);
                
                // Compute update ratio
                if let Some(&param_mag) = self.layer_magnitudes.get(layer_name) {
                    if param_mag > 1e-8 {
                        self.update_ratios.insert(layer_name.to_string(), grad_norm / param_mag);
                    }
                }
            }
        }
    }
    
    #[cfg(test)]
    pub(crate) fn to_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        // Average magnitudes
        if !self.layer_magnitudes.is_empty() {
            let avg_magnitude: f64 = self.layer_magnitudes.values().sum::<f64>() / self.layer_magnitudes.len() as f64;
            metrics.insert("avg_param_magnitude".to_string(), avg_magnitude);
        }
        
        // Average gradient norms
        if !self.layer_grad_norms.is_empty() {
            let avg_grad_norm: f64 = self.layer_grad_norms.values().sum::<f64>() / self.layer_grad_norms.len() as f64;
            metrics.insert("avg_grad_norm".to_string(), avg_grad_norm);
        }
        
        // Average update ratios
        if !self.update_ratios.is_empty() {
            let avg_update_ratio: f64 = self.update_ratios.values().sum::<f64>() / self.update_ratios.len() as f64;
            metrics.insert("avg_update_ratio".to_string(), avg_update_ratio);
        }
        
        metrics
    }
    
    #[cfg(not(test))]
    fn to_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        // Average magnitudes
        if !self.layer_magnitudes.is_empty() {
            let avg_magnitude: f64 = self.layer_magnitudes.values().sum::<f64>() / self.layer_magnitudes.len() as f64;
            metrics.insert("avg_param_magnitude".to_string(), avg_magnitude);
        }
        
        // Average gradient norms
        if !self.layer_grad_norms.is_empty() {
            let avg_grad_norm: f64 = self.layer_grad_norms.values().sum::<f64>() / self.layer_grad_norms.len() as f64;
            metrics.insert("avg_grad_norm".to_string(), avg_grad_norm);
        }
        
        // Average update ratios
        if !self.update_ratios.is_empty() {
            let avg_update_ratio: f64 = self.update_ratios.values().sum::<f64>() / self.update_ratios.len() as f64;
            metrics.insert("avg_update_ratio".to_string(), avg_update_ratio);
        }
        
        metrics
    }
}

impl ReconstructionTrainer {
    /// Create a new reconstruction trainer
    pub fn new(
        config: ReconstructionTrainerConfig,
        model: Arc<RwLock<HyperNetwork>>,
        train_loader: DataLoader<Box<dyn Dataset>>,
        val_loader: Option<DataLoader<Box<dyn Dataset>>>,
        device: Device,
    ) -> Result<Self> {
        // Validate configuration
        config.base_config.validate()
            .context("Training configuration validation failed")?;
        
        // Initialize variable map
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        
        // Initialize optimizer
        let optimizer = create_optimizer(&config.base_config.optimizer, &var_map)?;
        
        // Initialize scheduler
        let scheduler = create_scheduler(
            &config.base_config.optimizer.scheduler,
            config.base_config.optimizer.learning_rate
        )?;
        
        // Initialize reconstruction loss
        let loss_fn = ReconstructionLoss::new(config.base_config.loss.clone(), &device)?;
        
        // Initialize checkpoint manager
        let checkpoint_manager = CheckpointManager::new(
            config.base_config.checkpointing.clone(),
            device.clone(),
        )?;
        
        // Initialize metrics tracker
        let metrics = MetricsTracker::new(config.base_config.logging.clone())?;
        
        // Initialize training state
        let state = TrainingState::new();
        
        // Initialize layer weights
        let layer_weights = Self::compute_layer_weights(&config.reconstruction.layer_weighting);
        
        // Initialize gradient scaler if mixed precision is enabled
        let grad_scaler = if config.base_config.mixed_precision.enabled {
            Some(GradientScaler::new(config.base_config.mixed_precision.loss_scaling.init_scale))
        } else {
            None
        };
        
        // Initialize parameter statistics
        let param_stats = ParameterStatistics::new();
        
        // Initialize active layers for progressive training
        let active_layers = config.reconstruction.progressive_training.as_ref()
            .map(|prog| prog.initial_layers.clone());
        
        Ok(Self {
            config,
            model,
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
            layer_weights,
            grad_scaler,
            param_stats,
            active_layers,
        })
    }
    
    /// Set up event monitoring
    pub fn with_event_monitoring(mut self, tx: mpsc::UnboundedSender<TrainingEvent>) -> Self {
        self.event_tx = Some(tx);
        self
    }
    
    /// Start reconstruction training
    #[instrument(skip(self))]
    pub async fn train(&mut self) -> Result<TrainingResult> {
        info!("Starting reconstruction training");
        
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
        
        info!("Reconstruction training completed in {:?}", result.training_duration);
        self.send_event(TrainingEvent::TrainingCompleted { 
            total_steps: result.total_steps 
        });
        
        Ok(result)
    }
    
    /// Initialize training setup
    fn initialize_training(&mut self) -> Result<()> {
        info!("Initializing reconstruction training setup");
        
        // Set training status
        self.state.status = TrainingStatus::Running;
        self.state.start_time = Utc::now();
        
        // Resume from checkpoint if specified
        if let Some(checkpoint_path) = &self.config.base_config.training.resume_from_checkpoint {
            info!("Resuming from checkpoint: {:?}", checkpoint_path);
            let path = checkpoint_path.clone();
            self.load_checkpoint(&path)?;
        }
        
        // Initialize model for training
        // Set model to training mode
        // Note: HyperNetwork doesn't have set_training method
        
        // Log initial state
        info!("Device: {:?}", self.device);
        info!("Mixed precision: {}", self.config.base_config.mixed_precision.enabled);
        info!("Gradient accumulation mode: {:?}", self.config.reconstruction.gradient_accumulation_mode);
        
        if let Some(ref active_layers) = self.active_layers {
            info!("Progressive training enabled, starting with {} layers", active_layers.len());
        }
        
        Ok(())
    }
    
    /// Main training loop
    async fn run_training_loop(&mut self) -> Result<TrainingMetrics> {
        let total_epochs = self.config.base_config.training.num_epochs;
        
        for epoch in self.state.epoch..total_epochs {
            self.state.epoch = epoch;
            
            info!("Starting epoch {}/{}", epoch + 1, total_epochs);
            self.send_event(TrainingEvent::EpochStarted { epoch });
            
            // Update active layers for progressive training
            self.update_active_layers(epoch)?;
            
            // Run training epoch
            let train_metrics = self.train_epoch().await?;
            
            // Run evaluation if validation data is available
            let eval_metrics = if self.val_loader.is_some() {
                Some(self.evaluate().await?)
            } else {
                None
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
        let mut layer_losses: HashMap<String, f64> = HashMap::new();
        
        // Set model to training mode
        // Note: HyperNetwork doesn't have set_training method
        
        // Determine gradient accumulation steps
        let grad_accum_steps = self.get_gradient_accumulation_steps();
        let mut accumulated_steps = 0;
        
        // Iterate through training batches
        while let Some(batch) = self.train_loader.next_batch().await? {
            let step_start = Instant::now();
            
            // Convert batch to ReconstructionBatch
            let batch = batch.downcast::<ReconstructionBatch>()
                .map_err(|_| anyhow::anyhow!("Invalid batch type for reconstruction training"))?;
            
            // Forward pass and loss computation
            let (loss, loss_metrics) = self.reconstruction_step(*batch)?;
            
            // Scale loss for gradient accumulation
            let scaled_loss = if grad_accum_steps > 1 {
                (loss / grad_accum_steps as f64)?
            } else {
                loss
            };
            
            // Scale loss for mixed precision if enabled
            let final_loss = if let Some(ref scaler) = self.grad_scaler {
                scaler.scale_loss(&scaled_loss)?
            } else {
                scaled_loss
            };
            
            // Backward pass
            let mut gradients = final_loss.backward()?;
            
            // Update parameter statistics
            if self.config.reconstruction.track_param_magnitudes {
                self.update_parameter_statistics(&gradients)?;
            }
            
            accumulated_steps += 1;
            
            // Update parameters if gradient accumulation is complete
            if accumulated_steps >= grad_accum_steps {
                // Unscale gradients if using mixed precision
                let skip_update = if let Some(ref mut scaler) = self.grad_scaler {
                    !scaler.unscale_gradients(&mut gradients)?
                } else {
                    false
                };
                
                if !skip_update {
                    // Apply gradient clipping
                    self.apply_gradient_clipping(&gradients)?;
                    
                    // Optimizer step
                    self.optimizer.step(&gradients)?;
                }
                
                // Reset gradient accumulation
                accumulated_steps = 0;
                
                // Clear gradients
                self.var_map.all_vars().iter().for_each(|var| {
                    // Clear gradients - implementation needed
                });
            }
            
            // Update metrics
            epoch_loss += loss_metrics.total_loss;
            for (layer_name, layer_loss) in &loss_metrics.layer_losses {
                *layer_losses.entry(layer_name.clone()).or_insert(0.0) += layer_loss;
            }
            num_batches += 1;
            
            // Log step progress
            if self.state.global_step % self.config.base_config.training.log_steps == 0 {
                self.log_training_step(
                    &loss_metrics,
                    step_start.elapsed(),
                    self.scheduler.get_lr()
                );
            }
            
            // Log parameter statistics
            if self.config.reconstruction.track_param_magnitudes &&
               self.state.global_step % self.config.metrics.log_param_stats_interval == 0 {
                self.log_parameter_statistics();
            }
            
            // Evaluation step
            if self.state.global_step % self.config.base_config.training.eval_steps == 0 {
                if self.val_loader.is_some() {
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
        
        // Add average layer losses
        for (layer_name, total_loss) in layer_losses {
            let avg_layer_loss = if num_batches > 0 { total_loss / num_batches as f64 } else { 0.0 };
            metrics.insert(format!("train_loss_{}", layer_name), avg_layer_loss);
        }
        
        // Add parameter statistics to metrics
        if self.config.reconstruction.track_param_magnitudes {
            metrics.extend(self.param_stats.to_metrics());
        }
        
        // Add loss to history
        self.state.loss_history.push(avg_loss);
        if self.state.loss_history.len() > 100 {
            self.state.loss_history.remove(0);
        }
        
        Ok(metrics)
    }
    
    /// Perform a reconstruction training step
    fn reconstruction_step(&mut self, batch: ReconstructionBatch) -> Result<(Tensor, LossFunctionMetrics)> {
        let model = self.model.read();
        
        // Filter batch by active layers if progressive training is enabled
        let filtered_batch = if let Some(ref active_layers) = self.active_layers {
            self.filter_batch_by_layers(batch, active_layers)?
        } else {
            batch
        };
        
        // Forward pass through hypernetwork
        let predicted_params = self.forward_hypernetwork(&model, &filtered_batch)?;
        
        // Compute reconstruction loss
        let loss_metrics = self.loss_fn.compute_metrics(
            &filtered_batch as &dyn std::any::Any,
            &predicted_params as &dyn std::any::Any
        )?;
        
        // Create loss tensor
        let loss = Tensor::full(loss_metrics.total_loss, (), &self.device)?;
        
        Ok((loss, loss_metrics))
    }
    
    /// Perform a reconstruction step for evaluation (non-mutating)
    fn reconstruction_step_eval(&self, batch: ReconstructionBatch) -> Result<(Tensor, LossFunctionMetrics)> {
        let model = self.model.read();
        
        // Filter batch by active layers if progressive training is enabled
        let filtered_batch = if let Some(ref active_layers) = self.active_layers {
            self.filter_batch_by_layers(batch, active_layers)?
        } else {
            batch
        };
        
        // Forward pass through hypernetwork
        let predicted_params = self.forward_hypernetwork(&model, &filtered_batch)?;
        
        // Compute reconstruction loss
        let loss_metrics = self.loss_fn.compute_metrics(
            &filtered_batch as &dyn std::any::Any,
            &predicted_params as &dyn std::any::Any
        )?;
        
        // Create loss tensor
        let loss = Tensor::full(loss_metrics.total_loss, (), &self.device)?;
        
        Ok((loss, loss_metrics))
    }
    
    /// Forward pass through hypernetwork to generate LoRA parameters
    fn forward_hypernetwork(
        &self,
        model: &HyperNetwork,
        batch: &ReconstructionBatch
    ) -> Result<PredictedLoraParams> {
        let batch_size = batch.batch_size;
        let mut predicted_layers = HashMap::new();
        
        // Process each sample in the batch
        for i in 0..batch_size {
            // Extract task embedding for this sample
            let task_embedding = if let Some(ref embeddings) = batch.task_embeddings {
                embeddings.i(i)?
            } else {
                // Create dummy embedding if not provided
                Tensor::zeros((model.config().input_dim,), DType::F32, &self.device)?
            };
            
            // Forward pass through hypernetwork
            // Convert Tensor to ndarray for compatibility
            let embedding_array = self.tensor_to_array1(&task_embedding)?;
            // Default to BERT architecture for now
            let target_arch = TargetArchitecture::BERT {
                hidden_size: 768,
                num_layers: 12,
                num_heads: 12,
            }; // Would need proper target architecture from config
            let lora_params = model.generate_lora_params(&embedding_array, target_arch)?;
            
            // Convert LoRA params to layer-specific parameters
            let layer_params = self.convert_lora_params_to_tensors(&lora_params)?;
            
            // Add to predicted layers
            for (layer_name, (a_params, b_params)) in layer_params {
                let entry = predicted_layers.entry(layer_name.clone()).or_insert_with(|| {
                    (Vec::new(), Vec::new())
                });
                entry.0.push(a_params);
                entry.1.push(b_params);
            }
        }
        
        // Stack predictions into batches
        let mut layers = HashMap::new();
        for (layer_name, (a_list, b_list)) in predicted_layers {
            let a_matrix = Tensor::stack(&a_list, 0)?;
            let b_matrix = Tensor::stack(&b_list, 0)?;
            
            layers.insert(layer_name.clone(), PredictedLoraLayer {
                name: layer_name,
                a_matrix,
                b_matrix,
                alpha: None, // Could be predicted by hypernetwork
            });
        }
        
        Ok(PredictedLoraParams {
            layers,
            device: self.device.clone(),
        })
    }
    
    
    /// Get expected dimensions for a layer
    fn get_layer_dimensions(&self, layer_name: &str) -> Result<(usize, usize, usize)> {
        // This would need to be configured based on the target model architecture
        // For now, return dummy values
        Ok((768, 768, 16)) // Example: BERT-like dimensions with rank 16
    }
    
    /// Filter batch by active layers for progressive training
    fn filter_batch_by_layers(
        &self,
        mut batch: ReconstructionBatch,
        active_layers: &[String]
    ) -> Result<ReconstructionBatch> {
        let mut filtered_params = HashMap::new();
        
        for layer_name in active_layers {
            if let Some(params) = batch.lora_params.get(layer_name) {
                filtered_params.insert(layer_name.clone(), params.clone());
            }
        }
        
        batch.lora_params = filtered_params;
        Ok(batch)
    }
    
    /// Update active layers for progressive training
    fn update_active_layers(&mut self, epoch: usize) -> Result<()> {
        if let Some(ref prog_config) = self.config.reconstruction.progressive_training {
            if let Some(ref mut active_layers) = self.active_layers {
                let intervals_passed = epoch / prog_config.layer_addition_interval;
                
                // Add new layers if it's time
                if intervals_passed > 0 && epoch % prog_config.layer_addition_interval == 0 {
                    // This would need actual implementation to determine which layers to add
                    info!("Progressive training: adding new layers at epoch {}", epoch);
                }
            }
        }
        Ok(())
    }
    
    /// Evaluation loop
    #[instrument(skip(self))]
    async fn evaluate(&mut self) -> Result<HashMap<String, f64>> {
        if self.val_loader.is_none() {
            return Ok(HashMap::new());
        }
        
        info!("Running evaluation");
        // Set model to eval mode
        // Note: HyperNetwork doesn't have set_training method
        
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        let mut layer_losses: HashMap<String, f64> = HashMap::new();
        let mut alignment_scores: Vec<f64> = Vec::new();
        let mut magnitude_ratios: Vec<f64> = Vec::new();
        
        // Collect all batches first to avoid borrow checker issues
        let mut batches = Vec::new();
        if let Some(val_loader) = self.val_loader.as_mut() {
            while let Some(batch) = val_loader.next_batch().await? {
                let batch = batch.downcast::<ReconstructionBatch>()
                    .map_err(|_| anyhow::anyhow!("Invalid batch type for reconstruction evaluation"))?;
                batches.push(*batch);
            }
        }
        
        // Process the collected batches
        for batch in batches {
            // Forward pass without gradient computation
            let (_loss, loss_metrics) = self.reconstruction_step_eval(batch)?;
            
            // Accumulate metrics
            total_loss += loss_metrics.total_loss;
            for (layer_name, layer_loss) in &loss_metrics.layer_losses {
                *layer_losses.entry(layer_name.clone()).or_insert(0.0) += layer_loss;
            }
            
            // Collect additional metrics
            if self.config.validation.compute_alignment {
                if let Some(alignment) = loss_metrics.metrics.get("parameter_alignment") {
                    alignment_scores.push(*alignment);
                }
            }
            
            if let Some(ratio) = loss_metrics.metrics.get("magnitude_ratio") {
                magnitude_ratios.push(*ratio);
            }
            
            num_batches += 1;
        }
        
        // Compute average metrics
        let avg_loss = if num_batches > 0 { total_loss / num_batches as f64 } else { 0.0 };
        let mut metrics = HashMap::new();
        metrics.insert("eval_loss".to_string(), avg_loss);
        
        // Add average layer losses
        for (layer_name, total_loss) in layer_losses {
            let avg_layer_loss = if num_batches > 0 { total_loss / num_batches as f64 } else { 0.0 };
            metrics.insert(format!("eval_loss_{}", layer_name), avg_layer_loss);
        }
        
        // Add alignment metrics
        if !alignment_scores.is_empty() {
            let avg_alignment = alignment_scores.iter().sum::<f64>() / alignment_scores.len() as f64;
            metrics.insert("eval_alignment".to_string(), avg_alignment);
        }
        
        // Add magnitude ratio metrics
        if !magnitude_ratios.is_empty() {
            let avg_ratio = magnitude_ratios.iter().sum::<f64>() / magnitude_ratios.len() as f64;
            metrics.insert("eval_magnitude_ratio".to_string(), avg_ratio);
        }
        
        self.state.last_eval_time = Some(Utc::now());
        
        info!("Evaluation completed - Loss: {:.4}", avg_loss);
        
        Ok(metrics)
    }
    
    /// Get gradient accumulation steps based on mode
    fn get_gradient_accumulation_steps(&self) -> usize {
        match &self.config.reconstruction.gradient_accumulation_mode {
            GradientAccumulationMode::Fixed { steps } => *steps,
            GradientAccumulationMode::Dynamic { target_memory_mb } => {
                // Would need actual memory usage tracking
                1 // Placeholder
            }
            GradientAccumulationMode::Adaptive { min_steps, max_steps } => {
                // Would need gradient magnitude analysis
                *min_steps // Placeholder
            }
        }
    }
    
    /// Apply gradient clipping
    fn apply_gradient_clipping(&self, gradients: &candle_core::backprop::GradStore) -> Result<()> {
        let clip_config = &self.config.base_config.optimizer.gradient_clipping;
        
        if !clip_config.enabled {
            return Ok(());
        }
        
        match clip_config.method {
            crate::training::config::ClippingMethod::GlobalNorm => {
                // Implement global norm clipping
                let threshold = clip_config.threshold;
                debug!("Applying global norm clipping with threshold {}", threshold);
            }
            crate::training::config::ClippingMethod::Value => {
                // Implement value clipping
                let threshold = clip_config.threshold;
                debug!("Applying value clipping with threshold {}", threshold);
            }
            crate::training::config::ClippingMethod::Adaptive { percentile } => {
                // Implement adaptive clipping
                debug!("Applying adaptive clipping at {}th percentile", percentile);
            }
        }
        
        // Also apply parameter norm clipping if configured
        if let Some(param_norm_clip) = self.config.reconstruction.param_norm_clip {
            debug!("Applying parameter norm clipping with threshold {}", param_norm_clip);
            // Implementation would go here
        }
        
        Ok(())
    }
    
    /// Update parameter statistics
    fn update_parameter_statistics(&mut self, gradients: &candle_core::backprop::GradStore) -> Result<()> {
        // This would need actual implementation to track parameter and gradient statistics
        Ok(())
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
        
        // Log layer-specific losses if verbose
        if matches!(self.config.base_config.logging.level, LogLevel::Debug) {
            for (layer, loss) in &loss_metrics.layer_losses {
                debug!("  Layer {} loss: {:.6}", layer, loss);
            }
        }
        
        self.send_event(TrainingEvent::StepCompleted { 
            step: self.state.global_step, 
            loss: loss_metrics.total_loss,
            lr,
        });
    }
    
    /// Log parameter statistics
    fn log_parameter_statistics(&self) {
        info!("Parameter statistics at step {}:", self.state.global_step);
        
        for (metric_name, value) in self.param_stats.to_metrics() {
            info!("  {}: {:.6}", metric_name, value);
        }
        
        // Log per-layer statistics if verbose
        if matches!(self.config.base_config.logging.level, LogLevel::Debug) {
            for (layer, magnitude) in &self.param_stats.layer_magnitudes {
                debug!("  Layer {} magnitude: {:.6}", layer, magnitude);
            }
            
            for (layer, ratio) in &self.param_stats.update_ratios {
                debug!("  Layer {} update ratio: {:.6}", layer, ratio);
            }
        }
    }
    
    /// Compute layer weights based on strategy
    #[cfg(test)]
    pub(crate) fn compute_layer_weights(strategy: &LayerWeightingStrategy) -> HashMap<String, f64> {
        match strategy {
            LayerWeightingStrategy::Uniform => HashMap::new(), // Empty = uniform
            LayerWeightingStrategy::ByParameterCount => {
                // Would need actual parameter counts
                HashMap::new()
            }
            LayerWeightingStrategy::ByDepth { decay_factor } => {
                // Would need layer depth information
                HashMap::new()
            }
            LayerWeightingStrategy::Custom(weights) => weights.clone(),
        }
    }
    
    #[cfg(not(test))]
    fn compute_layer_weights(strategy: &LayerWeightingStrategy) -> HashMap<String, f64> {
        match strategy {
            LayerWeightingStrategy::Uniform => HashMap::new(), // Empty = uniform
            LayerWeightingStrategy::ByParameterCount => {
                // Would need actual parameter counts
                HashMap::new()
            }
            LayerWeightingStrategy::ByDepth { decay_factor } => {
                // Would need layer depth information
                HashMap::new()
            }
            LayerWeightingStrategy::Custom(weights) => weights.clone(),
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
            optimizer_state: Some(self.optimizer.state_dict()?),
            scheduler_state: Some(self.scheduler.state_dict()?),
            metrics: metrics.clone(),
            config: self.config.base_config.clone(),
            timestamp: Utc::now(),
            metadata: crate::training::checkpoints::CheckpointMetadata::new(),
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
        
        // Restore model state
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
        // This would serialize the model parameters
        Ok(vec![])
    }
    
    /// Deserialize model state from checkpoint
    fn deserialize_model_state(&mut self, state: &[u8]) -> Result<()> {
        // This would deserialize and load model parameters
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
        if self.config.metrics.track_memory_usage {
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
        info!("Finalizing reconstruction training");
        
        // Set model to evaluation mode
        // Set model to eval mode
        // Note: HyperNetwork doesn't have set_training method
        
        // Log final statistics
        if self.config.reconstruction.track_param_magnitudes {
            self.log_parameter_statistics();
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
    
    /// Convert Tensor to ndarray::Array1
    #[cfg(test)]
    pub(crate) fn tensor_to_array1(&self, tensor: &Tensor) -> Result<ndarray::Array1<f32>> {
        let data = tensor.to_vec1::<f32>()
            .context("Failed to convert tensor to vec")?;
        Ok(ndarray::Array1::from_vec(data))
    }
    
    #[cfg(not(test))]
    fn tensor_to_array1(&self, tensor: &Tensor) -> Result<ndarray::Array1<f32>> {
        let data = tensor.to_vec1::<f32>()
            .context("Failed to convert tensor to vec")?;
        Ok(ndarray::Array1::from_vec(data))
    }
    
    /// Convert LoRA parameters from hypernetwork to tensors
    fn convert_lora_params_to_tensors(
        &self,
        lora_params: &crate::hypernetwork::LoRAParams
    ) -> Result<HashMap<String, (Tensor, Tensor)>> {
        let mut result = HashMap::new();
        
        for (layer_name, layer_params) in &lora_params.layers {
            // Convert ndarray to Tensor for matrix A
            let a_shape = layer_params.matrix_a.shape();
            let a_data = layer_params.matrix_a.as_slice()
                .ok_or_else(|| anyhow::anyhow!("Failed to get slice from matrix A"))?;
            let a_tensor = Tensor::from_slice(a_data, &[a_shape[0], a_shape[1]], &self.device)?;
            
            // Convert ndarray to Tensor for matrix B
            let b_shape = layer_params.matrix_b.shape();
            let b_data = layer_params.matrix_b.as_slice()
                .ok_or_else(|| anyhow::anyhow!("Failed to get slice from matrix B"))?;
            let b_tensor = Tensor::from_slice(b_data, &[b_shape[0], b_shape[1]], &self.device)?;
            
            result.insert(layer_name.clone(), (a_tensor, b_tensor));
        }
        
        Ok(result)
    }
}

impl Default for ReconstructionTrainerConfig {
    fn default() -> Self {
        Self {
            base_config: TrainingConfig::reconstruction_default(),
            reconstruction: ReconstructionSettings {
                layer_weighting: LayerWeightingStrategy::Uniform,
                gradient_accumulation_mode: GradientAccumulationMode::Fixed { steps: 1 },
                progressive_training: None,
                param_norm_clip: Some(1.0),
                track_param_magnitudes: true,
                analyze_gradient_flow: false,
            },
            validation: ValidationSettings {
                compute_alignment: true,
                compute_effective_rank: false,
                track_layer_accuracy: true,
                best_model_metric: "eval_loss".to_string(),
            },
            metrics: MetricsSettings {
                log_param_stats_interval: 100,
                log_gradient_flow_interval: 500,
                track_memory_usage: true,
                enable_profiling: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_trainer_config_creation() {
        let config = ReconstructionTrainerConfig::default();
        assert!(matches!(config.reconstruction.layer_weighting, LayerWeightingStrategy::Uniform));
        assert_eq!(config.validation.best_model_metric, "eval_loss");
    }
    
    #[test]
    fn test_gradient_scaler() {
        let mut scaler = GradientScaler::new(1024.0);
        assert_eq!(scaler.scale, 1024.0);
        
        // Test scale update without inf
        scaler.update_scale(false);
        assert_eq!(scaler.steps_since_update, 1);
        
        // Test scale update with inf
        scaler.update_scale(true);
        assert_eq!(scaler.scale, 512.0); // Should be halved
        assert_eq!(scaler.steps_since_update, 0);
    }
    
    #[test]
    fn test_parameter_statistics() {
        let mut stats = ParameterStatistics::new();
        assert!(stats.layer_magnitudes.is_empty());
        
        // Add some dummy statistics
        stats.layer_magnitudes.insert("layer1".to_string(), 1.5);
        stats.layer_magnitudes.insert("layer2".to_string(), 2.5);
        
        let metrics = stats.to_metrics();
        assert_eq!(metrics.get("avg_param_magnitude"), Some(&2.0));
    }
}