//! Base trainer framework for T2L models
//!
//! This module provides the core training infrastructure including the main
//! T2LTrainer class, training loops, evaluation logic, and state management.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, Context};
use candle_core::{Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

// use crate::error::T2LError;
// use crate::hypernetwork::models::HyperNetwork;
// use crate::lora::config::LoRAConfig;

// Placeholder types until the actual modules are implemented
type HyperNetwork = ();
type T2LError = anyhow::Error;
use crate::training::config::{TrainingConfig, TrainingType};
use crate::training::data::{Dataset, DataLoader, ReconstructionBatch, SupervisedBatch};
use crate::training::checkpoints::{CheckpointManager, TrainingCheckpoint};
use crate::training::metrics::{MetricsTracker, TrainingMetrics};
use crate::training::optimizers::{OptimizerState, SchedulerState, create_optimizer, create_scheduler};

/// Main trainer for T2L models
pub struct T2LTrainer {
    /// Training configuration
    config: TrainingConfig,
    
    /// Model being trained
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
    
    /// Checkpoint manager
    checkpoint_manager: CheckpointManager,
    
    /// Metrics tracker
    metrics: MetricsTracker,
    
    /// Training state
    state: TrainingState,
    
    /// Device for training
    device: Device,
    
    /// Random number generator
    rng: StdRng,
    
    /// Event channel for monitoring
    event_tx: Option<mpsc::UnboundedSender<TrainingEvent>>,
}

/// Training state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// Current epoch
    pub epoch: usize,
    
    /// Current step within epoch
    pub step: usize,
    
    /// Global step across all epochs
    pub global_step: usize,
    
    /// Best validation score
    pub best_score: Option<f64>,
    
    /// Steps since best score
    pub steps_since_best: usize,
    
    /// Training start time
    pub start_time: DateTime<Utc>,
    
    /// Last evaluation time
    pub last_eval_time: Option<DateTime<Utc>>,
    
    /// Training status
    pub status: TrainingStatus,
    
    /// Current learning rate
    pub current_lr: f64,
    
    /// Accumulated gradient steps
    pub grad_accum_step: usize,
    
    /// Loss history for debugging
    pub loss_history: Vec<f64>,
    
    /// Memory usage tracking
    pub memory_usage: MemoryUsage,
}

/// Training status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrainingStatus {
    NotStarted,
    Running,
    Paused,
    Completed,
    Failed { error: String },
    EarlyStopped,
}

/// Memory usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    
    /// Current memory usage in bytes
    pub current_memory_bytes: u64,
    
    /// GPU memory usage if available
    pub gpu_memory_bytes: Option<u64>,
}

/// Training events for monitoring
#[derive(Debug, Clone)]
pub enum TrainingEvent {
    EpochStarted { epoch: usize },
    EpochCompleted { epoch: usize, metrics: TrainingMetrics },
    StepCompleted { step: usize, loss: f64, lr: f64 },
    EvaluationCompleted { metrics: TrainingMetrics },
    CheckpointSaved { path: PathBuf },
    EarlyStopping { reason: String },
    TrainingCompleted { total_steps: usize },
    Error { error: String },
}

/// Training result information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Final training metrics
    pub final_metrics: TrainingMetrics,
    
    /// Training state at completion
    pub final_state: TrainingState,
    
    /// Path to best model checkpoint
    pub best_checkpoint_path: Option<PathBuf>,
    
    /// Training duration
    pub training_duration: Duration,
    
    /// Total training steps
    pub total_steps: usize,
    
    /// Whether training completed successfully
    pub success: bool,
    
    /// Error message if training failed
    pub error_message: Option<String>,
}

impl T2LTrainer {
    /// Create a new trainer instance
    pub fn new(
        config: TrainingConfig,
        model: Arc<RwLock<HyperNetwork>>,
        train_loader: DataLoader<Box<dyn Dataset>>,
        val_loader: Option<DataLoader<Box<dyn Dataset>>>,
        device: Device,
    ) -> Result<Self> {
        // Validate configuration
        config.validate()
            .context("Training configuration validation failed")?;
        
        // Initialize variable map
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
        
        // Initialize optimizer
        let optimizer = create_optimizer(&config.optimizer, &var_map)?;
        
        // Initialize scheduler
        let scheduler = create_scheduler(&config.optimizer.scheduler, config.optimizer.learning_rate)?;
        
        // Initialize checkpoint manager
        let checkpoint_manager = CheckpointManager::new(
            config.checkpointing.clone(),
            device.clone(),
        )?;
        
        // Initialize metrics tracker
        let metrics = MetricsTracker::new(config.logging.clone())?;
        
        // Initialize training state
        let state = TrainingState::new();
        
        // Initialize RNG with seed
        let rng = StdRng::seed_from_u64(config.training.seed);
        
        Ok(Self {
            config,
            model,
            var_map,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            checkpoint_manager,
            metrics,
            state,
            device,
            rng,
            event_tx: None,
        })
    }
    
    /// Set up event monitoring
    pub fn with_event_monitoring(mut self, tx: mpsc::UnboundedSender<TrainingEvent>) -> Self {
        self.event_tx = Some(tx);
        self
    }
    
    /// Start training
    pub async fn train(&mut self) -> Result<TrainingResult> {
        info!("Starting T2L training");
        
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
        
        info!("Training completed in {:?}", result.training_duration);
        self.send_event(TrainingEvent::TrainingCompleted { 
            total_steps: result.total_steps 
        });
        
        Ok(result)
    }
    
    /// Initialize training setup
    fn initialize_training(&mut self) -> Result<()> {
        info!("Initializing training setup");
        
        // Set training status
        self.state.status = TrainingStatus::Running;
        self.state.start_time = Utc::now();
        
        // Resume from checkpoint if specified
        if let Some(checkpoint_path) = &self.config.training.resume_from_checkpoint {
            info!("Resuming from checkpoint: {:?}", checkpoint_path);
            let path = checkpoint_path.clone();
            self.load_checkpoint(path.as_path())?;
        }
        
        // Initialize model for training
        // TODO: Implement set_training when HyperNetwork is fully implemented
        // self.model.write().set_training(true);
        
        // Set up mixed precision if enabled
        if self.config.mixed_precision.enabled {
            info!("Mixed precision training enabled");
            // Setup mixed precision context
        }
        
        // Log initial state
        info!("Training configuration: {:?}", self.config.model.training_type);
        info!("Device: {:?}", self.device);
        info!("Model parameters: {}", self.count_parameters());
        
        Ok(())
    }
    
    /// Main training loop
    async fn run_training_loop(&mut self) -> Result<TrainingMetrics> {
        let total_epochs = self.config.training.num_epochs;
        
        for epoch in self.state.epoch..total_epochs {
            self.state.epoch = epoch;
            
            info!("Starting epoch {}/{}", epoch + 1, total_epochs);
            self.send_event(TrainingEvent::EpochStarted { epoch });
            
            // Run training epoch
            let train_metrics = self.train_epoch().await?;
            
            // Run evaluation if validation data is available
            let eval_metrics = if let Some(_) = &self.val_loader {
                Some(self.evaluate().await?)
            } else {
                None
            };
            
            // Combine metrics
            let epoch_metrics = self.combine_metrics(train_metrics, eval_metrics);
            
            // Update learning rate scheduler
            self.scheduler.step(epoch_metrics.get("eval_loss").cloned());
            self.state.current_lr = self.scheduler.get_lr();
            
            // Log epoch completion
            info!("Epoch {}/{} completed - Loss: {:.4}", 
                  epoch + 1, total_epochs, 
                  epoch_metrics.get("train_loss").unwrap_or(&0.0));
            
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
            if let Some(max_steps) = self.config.training.max_steps {
                if self.state.global_step >= max_steps {
                    info!("Maximum steps reached: {}", max_steps);
                    break;
                }
            }
        }
        
        Ok(self.metrics.current_metrics())
    }
    
    /// Train for one epoch
    async fn train_epoch(&mut self) -> Result<HashMap<String, f64>> {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        
        // TODO: Implement set_training when HyperNetwork is fully implemented
        // self.model.write().set_training(true);
        
        // Iterate through training batches
        while let Some(batch) = self.train_loader.next_batch().await? {
            let step_start = Instant::now();
            
            // Forward pass and loss computation
            let loss = match &self.config.model.training_type {
                TrainingType::Reconstruction => {
                    let reconstruction_batch = batch.downcast::<ReconstructionBatch>()
                        .map_err(|_| anyhow::anyhow!("Failed to downcast batch to ReconstructionBatch"))?;
                    self.reconstruction_step(*reconstruction_batch)?
                }
                TrainingType::Supervised => {
                    let supervised_batch = batch.downcast::<SupervisedBatch>()
                        .map_err(|_| anyhow::anyhow!("Failed to downcast batch to SupervisedBatch"))?;
                    self.supervised_step(*supervised_batch)?
                }
                TrainingType::MultiTask { tasks, task_weights } => {
                    // Clone to avoid borrow issues
                    let tasks_cloned = tasks.clone();
                    let weights_cloned = task_weights.clone();
                    self.multi_task_step(batch, &tasks_cloned, &weights_cloned)?
                }
            };
            
            // Store loss value before backward pass
            let loss_value = loss.to_scalar::<f64>()?;
            
            // Backward pass
            self.backward_step(loss)?;
            
            // Update metrics
            epoch_loss += loss_value;
            num_batches += 1;
            
            // Log step
            if self.state.global_step % self.config.training.log_steps == 0 {
                let lr = self.scheduler.get_lr();
                debug!("Step {} - Loss: {:.4}, LR: {:.2e}, Time: {:?}", 
                       self.state.global_step, 
                       loss_value, 
                       lr,
                       step_start.elapsed());
                
                self.send_event(TrainingEvent::StepCompleted { 
                    step: self.state.global_step, 
                    loss: loss_value,
                    lr,
                });
            }
            
            // Evaluation step
            if self.state.global_step % self.config.training.eval_steps == 0 {
                if let Some(_) = &self.val_loader {
                    let eval_metrics = self.evaluate().await?;
                    self.send_event(TrainingEvent::EvaluationCompleted { 
                        metrics: self.metrics.current_metrics() 
                    });
                }
            }
            
            // Save checkpoint
            if self.state.global_step % self.config.training.save_steps == 0 {
                let metrics = HashMap::new(); // Simplified for step saves
                self.save_checkpoint(self.state.epoch, &metrics).await?;
            }
            
            self.state.step += 1;
            self.state.global_step += 1;
            
            // Update memory usage
            self.update_memory_usage();
        }
        
        let avg_loss = if num_batches > 0 { epoch_loss / num_batches as f64 } else { 0.0 };
        let mut metrics = HashMap::new();
        metrics.insert("train_loss".to_string(), avg_loss);
        
        // Add loss to history
        self.state.loss_history.push(avg_loss);
        
        Ok(metrics)
    }
    
    /// Reconstruction training step
    fn reconstruction_step(&mut self, batch: ReconstructionBatch) -> Result<Tensor> {
        // TODO: Implement actual forward pass when HyperNetwork is fully implemented
        // For now, return a dummy loss tensor
        
        // let model = self.model.read();
        // let lora_weights = model.forward(&batch.text_embeddings, &batch.task_embeddings)?;
        
        // Compute reconstruction loss using target weights directly as a placeholder
        let reconstruction_loss = self.compute_reconstruction_loss(&batch.target_weights, &batch.target_weights)?;
        
        // Add regularization if configured
        let total_loss = if self.config.regularization.l2_reg > 0.0 {
            let l2_penalty = self.compute_l2_penalty(&batch.target_weights)?;
            (reconstruction_loss + l2_penalty * self.config.regularization.l2_reg)?
        } else {
            reconstruction_loss
        };
        
        Ok(total_loss)
    }
    
    /// Supervised training step
    fn supervised_step(&mut self, batch: SupervisedBatch) -> Result<Tensor> {
        // TODO: Implement actual forward pass when HyperNetwork is fully implemented
        // For now, return a dummy loss tensor
        
        // let model = self.model.read();
        // let logits = model.forward_supervised(&batch.input_ids, &batch.attention_mask)?;
        
        // Compute supervised loss - check if labels are present
        if let Some(labels) = &batch.labels {
            // Create dummy logits for now
            let dummy_logits = Tensor::zeros_like(labels)?;
            let loss = self.compute_supervised_loss(&dummy_logits, labels)?;
            Ok(loss)
        } else {
            // Return a dummy loss if no labels
            Ok(Tensor::new(&[1.0f32], &self.device)?)
        }
    }
    
    /// Multi-task training step
    fn multi_task_step(
        &mut self, 
        batch: Box<dyn std::any::Any>, 
        tasks: &[String], 
        task_weights: &[f64]
    ) -> Result<Tensor> {
        // This would need to be implemented based on specific multi-task requirements
        todo!("Multi-task training not yet implemented")
    }
    
    /// Backward pass and optimizer step
    fn backward_step(&mut self, loss: Tensor) -> Result<()> {
        // Scale loss for gradient accumulation
        let scaled_loss = if self.config.training.gradient_accumulation_steps > 1 {
            (loss / self.config.training.gradient_accumulation_steps as f64)?
        } else {
            loss
        };
        
        // Backward pass
        let gradients = scaled_loss.backward()?;
        
        self.state.grad_accum_step += 1;
        
        // Update parameters if gradient accumulation is complete
        if self.state.grad_accum_step >= self.config.training.gradient_accumulation_steps {
            // Apply gradient clipping if enabled
            if self.config.optimizer.gradient_clipping.enabled {
                self.apply_gradient_clipping(&gradients)?;
            }
            
            // Optimizer step
            self.optimizer.step(&gradients)?;
            
            // Reset gradient accumulation
            self.state.grad_accum_step = 0;
            
            // Clear gradients
            // TODO: Implement gradient clearing when var_map properly supports it
            // self.var_map.all_vars().iter().for_each(|var| {
            //     // Clear gradients - this would need proper implementation
            // });
        }
        
        Ok(())
    }
    
    /// Evaluation loop
    async fn evaluate(&mut self) -> Result<HashMap<String, f64>> {
        if self.val_loader.is_none() {
            return Ok(HashMap::new());
        }
        
        info!("Running evaluation");
        // TODO: Implement set_training when HyperNetwork is fully implemented
        // self.model.write().set_training(false);
        
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        // Clone val_loader to avoid borrow issues
        let mut val_loader = self.val_loader.take().unwrap();
        
        while let Some(batch) = val_loader.next_batch().await? {
            let loss = match &self.config.model.training_type {
                TrainingType::Reconstruction => {
                    let reconstruction_batch = batch.downcast::<ReconstructionBatch>()
                        .map_err(|_| anyhow::anyhow!("Failed to downcast batch to ReconstructionBatch"))?;
                    self.reconstruction_step(*reconstruction_batch)?
                }
                TrainingType::Supervised => {
                    let supervised_batch = batch.downcast::<SupervisedBatch>()
                        .map_err(|_| anyhow::anyhow!("Failed to downcast batch to SupervisedBatch"))?;
                    self.supervised_step(*supervised_batch)?
                }
                TrainingType::MultiTask { tasks, task_weights } => {
                    // Clone to avoid borrow issues
                    let tasks_cloned = tasks.clone();
                    let weights_cloned = task_weights.clone();
                    self.multi_task_step(batch, &tasks_cloned, &weights_cloned)?
                }
            };
            
            total_loss += loss.to_scalar::<f64>()?;
            num_batches += 1;
        }
        
        let avg_loss = if num_batches > 0 { total_loss / num_batches as f64 } else { 0.0 };
        
        let mut metrics = HashMap::new();
        metrics.insert("eval_loss".to_string(), avg_loss);
        
        // Restore val_loader
        self.val_loader = Some(val_loader);
        
        self.state.last_eval_time = Some(Utc::now());
        
        info!("Evaluation completed - Loss: {:.4}", avg_loss);
        
        Ok(metrics)
    }
    
    /// Compute reconstruction loss
    fn compute_reconstruction_loss(&self, predicted: &Tensor, target: &Tensor) -> Result<Tensor> {
        // MSE loss for reconstruction
        let diff = (predicted - target)?;
        let squared_diff = diff.sqr()?;
        let loss = squared_diff.mean_all()?;
        Ok(loss)
    }
    
    /// Compute supervised loss (cross-entropy)
    fn compute_supervised_loss(&self, logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
        // Cross-entropy loss implementation
        // TODO: Use proper log_softmax when available
        // For now, use a simplified version
        let max_logits = logits.max_keepdim(candle_core::D::Minus1)?;
        let shifted_logits = (logits - max_logits)?;
        let exp_logits = shifted_logits.exp()?;
        let sum_exp = exp_logits.sum_keepdim(candle_core::D::Minus1)?;
        let log_probs = (shifted_logits - sum_exp.log())?;
        
        let nll_loss = labels.broadcast_mul(&log_probs)?.sum_all()?.neg()?;
        Ok(nll_loss)
    }
    
    /// Compute L2 regularization penalty
    fn compute_l2_penalty(&self, weights: &Tensor) -> Result<Tensor> {
        let squared = weights.sqr()?;
        let l2_norm = squared.sum_all()?;
        Ok(l2_norm)
    }
    
    /// Apply gradient clipping
    fn apply_gradient_clipping(&self, gradients: &candle_core::backprop::GradStore) -> Result<()> {
        match self.config.optimizer.gradient_clipping.method {
            crate::training::config::ClippingMethod::GlobalNorm => {
                // Implement global norm clipping
                let threshold = self.config.optimizer.gradient_clipping.threshold;
                // Implementation would go here
            }
            crate::training::config::ClippingMethod::Value => {
                // Implement value clipping
                let threshold = self.config.optimizer.gradient_clipping.threshold;
                // Implementation would go here
            }
            crate::training::config::ClippingMethod::Adaptive { percentile } => {
                // Implement adaptive clipping
                // Implementation would go here
            }
        }
        
        Ok(())
    }
    
    /// Check if early stopping should be triggered
    fn should_early_stop(&mut self, metrics: &HashMap<String, f64>) -> bool {
        if !self.config.training.early_stopping.enabled {
            return false;
        }
        
        let monitor_metric = &self.config.training.early_stopping.monitor_metric;
        let higher_is_better = self.config.training.early_stopping.higher_is_better;
        let min_delta = self.config.training.early_stopping.min_delta;
        let patience = self.config.training.early_stopping.patience;
        
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
            config: self.config.clone(),
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
        // Implementation depends on how model state is stored
        Ok(vec![])
    }
    
    /// Deserialize model state from checkpoint
    fn deserialize_model_state(&mut self, state: &[u8]) -> Result<()> {
        // This would deserialize and load model parameters
        // Implementation depends on how model state is stored
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
    
    /// Count total model parameters
    fn count_parameters(&self) -> usize {
        // This would count the total number of trainable parameters
        // Implementation depends on model structure
        0
    }
    
    /// Update memory usage tracking
    fn update_memory_usage(&mut self) {
        // This would track actual memory usage
        // Implementation would depend on platform-specific memory APIs
        self.state.memory_usage.current_memory_bytes = 0;
    }
    
    /// Finalize training cleanup
    fn finalize_training(&mut self) -> Result<()> {
        info!("Finalizing training");
        
        // Set model to evaluation mode
        // TODO: Implement set_training when HyperNetwork is fully implemented
        // self.model.write().set_training(false);
        
        // Cleanup resources
        // Implementation-specific cleanup
        
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
    
    /// Pause training
    pub fn pause(&mut self) {
        self.state.status = TrainingStatus::Paused;
        info!("Training paused");
    }
    
    /// Resume training
    pub fn resume(&mut self) {
        if self.state.status == TrainingStatus::Paused {
            self.state.status = TrainingStatus::Running;
            info!("Training resumed");
        }
    }
    
    /// Stop training
    pub fn stop(&mut self) {
        self.state.status = TrainingStatus::Completed;
        info!("Training stopped");
    }
}

impl TrainingState {
    /// Create new training state
    pub fn new() -> Self {
        Self {
            epoch: 0,
            step: 0,
            global_step: 0,
            best_score: None,
            steps_since_best: 0,
            start_time: Utc::now(),
            last_eval_time: None,
            status: TrainingStatus::NotStarted,
            current_lr: 0.0,
            grad_accum_step: 0,
            loss_history: Vec::new(),
            memory_usage: MemoryUsage::new(),
        }
    }
    
    /// Get training progress as percentage
    pub fn progress_percentage(&self, total_epochs: usize) -> f64 {
        if total_epochs == 0 {
            0.0
        } else {
            (self.epoch as f64 / total_epochs as f64) * 100.0
        }
    }
    
    /// Get estimated time remaining
    pub fn estimated_time_remaining(&self, total_epochs: usize) -> Option<Duration> {
        if self.epoch == 0 || total_epochs <= self.epoch {
            return None;
        }
        
        let elapsed = Utc::now().signed_duration_since(self.start_time);
        let elapsed_secs = elapsed.num_seconds() as f64;
        
        let progress = self.epoch as f64 / total_epochs as f64;
        let total_time_estimate = elapsed_secs / progress;
        let remaining = total_time_estimate - elapsed_secs;
        
        if remaining > 0.0 {
            Some(Duration::from_secs(remaining as u64))
        } else {
            None
        }
    }
}

impl MemoryUsage {
    pub fn new() -> Self {
        Self {
            peak_memory_bytes: 0,
            current_memory_bytes: 0,
            gpu_memory_bytes: None,
        }
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MemoryUsage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_training_state_creation() {
        let state = TrainingState::new();
        assert_eq!(state.epoch, 0);
        assert_eq!(state.step, 0);
        assert_eq!(state.global_step, 0);
        assert_eq!(state.status, TrainingStatus::NotStarted);
    }

    #[test]
    fn test_progress_calculation() {
        let mut state = TrainingState::new();
        state.epoch = 5;
        
        let progress = state.progress_percentage(10);
        assert_eq!(progress, 50.0);
    }

    #[test]
    fn test_memory_usage_creation() {
        let memory = MemoryUsage::new();
        assert_eq!(memory.peak_memory_bytes, 0);
        assert_eq!(memory.current_memory_bytes, 0);
        assert!(memory.gpu_memory_bytes.is_none());
    }
}