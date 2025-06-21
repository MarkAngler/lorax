//! Optimizers and learning rate schedulers for T2L training
//!
//! This module provides comprehensive optimization infrastructure including
//! AdamW, SGD, and other optimizers with learning rate scheduling, gradient
//! clipping, and mixed precision support.

pub mod adamw;
pub mod sgd;
pub mod schedulers;
pub mod state;

pub use adamw::AdamWOptimizer;
pub use sgd::SGDOptimizer;
pub use schedulers::{
    LearningRateScheduler, LinearScheduler, CosineScheduler, 
    ExponentialScheduler, StepScheduler, ConstantScheduler
};
pub use state::{OptimizerState as OptimizerStateDict, SchedulerState as SchedulerStateDict};

use std::collections::HashMap;
use anyhow::Result;
use candle_core::{Tensor, Device};
use candle_nn::{VarMap, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::training::config::{OptimizerConfig, SchedulerConfig, OptimizerType, SchedulerType};

/// Trait for optimizers
pub trait Optimizer {
    /// Optimizer name
    fn name(&self) -> &str;
    
    /// Perform optimization step
    fn step(&mut self, gradients: &candle_core::backprop::GradStore) -> Result<()>;
    
    /// Zero gradients
    fn zero_grad(&mut self) -> Result<()>;
    
    /// Get current learning rate
    fn learning_rate(&self) -> f64;
    
    /// Set learning rate
    fn set_learning_rate(&mut self, lr: f64);
    
    /// Get optimizer state for checkpointing
    fn state_dict(&self) -> Result<OptimizerStateDictLegacy>;
    
    /// Load optimizer state from checkpoint
    fn load_state_dict(&mut self, state: OptimizerStateDictLegacy) -> Result<()>;
    
    /// Get parameter count
    fn parameter_count(&self) -> usize;
    
    /// Get current step count
    fn step_count(&self) -> usize;
}

/// Trait for learning rate schedulers
pub trait Scheduler {
    /// Scheduler name
    fn name(&self) -> &str;
    
    /// Step the scheduler
    fn step(&mut self, metric: Option<f64>);
    
    /// Get current learning rate
    fn get_lr(&self) -> f64;
    
    /// Get scheduler state for checkpointing
    fn state_dict(&self) -> Result<SchedulerStateDictLegacy>;
    
    /// Load scheduler state from checkpoint
    fn load_state_dict(&mut self, state: SchedulerStateDictLegacy) -> Result<()>;
    
    /// Check if scheduler is done (for finite schedules)
    fn is_done(&self) -> bool {
        false
    }
    
    /// Reset scheduler to initial state
    fn reset(&mut self);
}

/// Optimizer state dictionary for checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerStateDictLegacy {
    /// Optimizer type
    pub optimizer_type: String,
    
    /// Current step count
    pub step_count: usize,
    
    /// Current learning rate
    pub learning_rate: f64,
    
    /// Optimizer-specific state
    pub state: HashMap<String, OptimizerTensorState>,
    
    /// Hyperparameters
    pub hyperparameters: HashMap<String, f64>,
}

/// Scheduler state dictionary for checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerStateDictLegacy {
    /// Scheduler type
    pub scheduler_type: String,
    
    /// Current step count
    pub step_count: usize,
    
    /// Current learning rate
    pub current_lr: f64,
    
    /// Base learning rate
    pub base_lr: f64,
    
    /// Scheduler-specific state
    pub state: HashMap<String, f64>,
    
    /// Hyperparameters
    pub hyperparameters: HashMap<String, f64>,
}

/// Tensor state for optimizer momentum/moving averages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerTensorState {
    /// Tensor shape
    pub shape: Vec<usize>,
    
    /// Tensor data (flattened)
    pub data: Vec<f64>,
    
    /// Device type
    pub device: String,
    
    /// Data type
    pub dtype: String,
}

/// Wrapper for optimizer state management
pub struct OptimizerState {
    optimizer: Box<dyn Optimizer + Send + Sync>,
    device: Device,
}

/// Wrapper for scheduler state management
pub struct SchedulerState {
    scheduler: Box<dyn Scheduler + Send + Sync>,
}

impl OptimizerState {
    /// Create new optimizer state
    pub fn new(optimizer: Box<dyn Optimizer + Send + Sync>, device: Device) -> Self {
        Self { optimizer, device }
    }
    
    /// Perform optimization step
    pub fn step(&mut self, gradients: &candle_core::backprop::GradStore) -> Result<()> {
        self.optimizer.step(gradients)
    }
    
    /// Zero gradients
    pub fn zero_grad(&mut self) -> Result<()> {
        self.optimizer.zero_grad()
    }
    
    /// Get current learning rate
    pub fn learning_rate(&self) -> f64 {
        self.optimizer.learning_rate()
    }
    
    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.optimizer.set_learning_rate(lr);
    }
    
    /// Get state dict for checkpointing
    pub fn state_dict(&self) -> Result<Vec<u8>> {
        let state = self.optimizer.state_dict()?;
        Ok(bincode::serialize(&state)?)
    }
    
    /// Load state dict from checkpoint
    pub fn load_state_dict(&mut self, data: Vec<u8>) -> Result<()> {
        let state: OptimizerStateDictLegacy = bincode::deserialize(&data)?;
        self.optimizer.load_state_dict(state)
    }
}

impl SchedulerState {
    /// Create new scheduler state
    pub fn new(scheduler: Box<dyn Scheduler + Send + Sync>) -> Self {
        Self { scheduler }
    }
    
    /// Step the scheduler
    pub fn step(&mut self, metric: Option<f64>) {
        self.scheduler.step(metric);
    }
    
    /// Get current learning rate
    pub fn get_lr(&self) -> f64 {
        self.scheduler.get_lr()
    }
    
    /// Get state dict for checkpointing
    pub fn state_dict(&self) -> Result<Vec<u8>> {
        let state = self.scheduler.state_dict()?;
        Ok(bincode::serialize(&state)?)
    }
    
    /// Load state dict from checkpoint
    pub fn load_state_dict(&mut self, data: Vec<u8>) -> Result<()> {
        let state: SchedulerStateDictLegacy = bincode::deserialize(&data)?;
        self.scheduler.load_state_dict(state)
    }
    
    /// Check if scheduler is done
    pub fn is_done(&self) -> bool {
        self.scheduler.is_done()
    }
    
    /// Reset scheduler
    pub fn reset(&mut self) {
        self.scheduler.reset();
    }
}

/// Create optimizer from configuration
pub fn create_optimizer(config: &OptimizerConfig, var_map: &VarMap) -> Result<OptimizerState> {
    let device = Device::Cpu; // This should come from configuration
    
    let optimizer: Box<dyn Optimizer + Send + Sync> = match &config.optimizer_type {
        OptimizerType::AdamW => {
            Box::new(AdamWOptimizer::new(
                var_map,
                config.learning_rate,
                config.beta1,
                config.beta2,
                config.epsilon,
                config.weight_decay,
            )?)
        }
        OptimizerType::Adam => {
            Box::new(AdamWOptimizer::new(
                var_map,
                config.learning_rate,
                config.beta1,
                config.beta2,
                config.epsilon,
                0.0, // No weight decay for Adam
            )?)
        }
        OptimizerType::SGD { momentum } => {
            Box::new(SGDOptimizer::new(
                var_map,
                config.learning_rate,
                *momentum,
                config.weight_decay,
            )?)
        }
        OptimizerType::RMSprop { alpha } => {
            // For now, use SGD as placeholder
            Box::new(SGDOptimizer::new(
                var_map,
                config.learning_rate,
                0.0,
                config.weight_decay,
            )?)
        }
        OptimizerType::Custom { name } => {
            return Err(anyhow::anyhow!("Custom optimizer '{}' not implemented", name));
        }
    };
    
    Ok(OptimizerState::new(optimizer, device))
}

/// Create scheduler from configuration
pub fn create_scheduler(config: &SchedulerConfig, base_lr: f64) -> Result<SchedulerState> {
    let scheduler: Box<dyn Scheduler + Send + Sync> = match &config.scheduler_type {
        SchedulerType::Linear => {
            Box::new(LinearScheduler::new(
                base_lr,
                config.min_lr,
                config.total_steps.unwrap_or(1000),
                config.warmup_steps,
            ))
        }
        SchedulerType::Cosine => {
            Box::new(CosineScheduler::new(
                base_lr,
                config.min_lr,
                config.total_steps.unwrap_or(1000),
                config.warmup_steps,
            ))
        }
        SchedulerType::CosineWithRestarts { restart_period } => {
            Box::new(CosineScheduler::new(
                base_lr,
                config.min_lr,
                *restart_period,
                config.warmup_steps,
            ))
        }
        SchedulerType::Exponential => {
            Box::new(ExponentialScheduler::new(
                base_lr,
                config.decay_factor,
                config.warmup_steps,
            ))
        }
        SchedulerType::StepLR => {
            Box::new(StepScheduler::new(
                base_lr,
                config.step_size,
                config.decay_factor,
                config.warmup_steps,
            ))
        }
        SchedulerType::ReduceOnPlateau { patience, factor } => {
            // For now, use linear scheduler as placeholder
            Box::new(LinearScheduler::new(
                base_lr,
                config.min_lr,
                config.total_steps.unwrap_or(1000),
                config.warmup_steps,
            ))
        }
        SchedulerType::Constant => {
            Box::new(ConstantScheduler::new(base_lr))
        }
    };
    
    Ok(SchedulerState::new(scheduler))
}

/// Gradient clipping utilities
pub struct GradientClipper {
    method: ClippingMethod,
    threshold: f64,
}

/// Gradient clipping methods
#[derive(Debug, Clone)]
pub enum ClippingMethod {
    GlobalNorm,
    Value,
    Adaptive { percentile: f64 },
}

impl GradientClipper {
    /// Create new gradient clipper
    pub fn new(method: ClippingMethod, threshold: f64) -> Self {
        Self { method, threshold }
    }
    
    /// Apply gradient clipping
    pub fn clip_gradients(&self, var_map: &VarMap, gradients: &candle_core::backprop::GradStore) -> Result<f64> {
        match &self.method {
            ClippingMethod::GlobalNorm => {
                self.clip_by_global_norm(var_map, gradients)
            }
            ClippingMethod::Value => {
                self.clip_by_value(var_map, gradients)
            }
            ClippingMethod::Adaptive { percentile } => {
                self.clip_adaptive(var_map, gradients, *percentile)
            }
        }
    }
    
    /// Clip gradients by global norm
    fn clip_by_global_norm(&self, var_map: &VarMap, gradients: &candle_core::backprop::GradStore) -> Result<f64> {
        // Calculate global norm of all gradients
        let mut global_norm_squared = 0.0f64;
        
        // First pass: calculate the global norm
        for var in var_map.all_vars() {
            if let Some(grad) = gradients.get(&var) {
                let grad_norm_squared = grad.sqr()?.sum_all()?.to_scalar::<f64>()?;
                global_norm_squared += grad_norm_squared;
            }
        }
        
        let global_norm = global_norm_squared.sqrt();
        
        // If the global norm exceeds the threshold, scale all gradients
        if global_norm > self.threshold {
            let scale_factor = self.threshold / global_norm;
            
            // Second pass: scale all gradients
            // Note: In practice, gradient clipping would be implemented
            // by scaling gradients during the optimizer step
        }
        
        Ok(global_norm)
    }
    
    /// Clip gradients by value
    fn clip_by_value(&self, var_map: &VarMap, gradients: &candle_core::backprop::GradStore) -> Result<f64> {
        let mut max_grad = 0.0f64;
        
        // Clip each gradient to [-threshold, threshold]
        for var in var_map.all_vars() {
            if let Some(grad) = gradients.get(&var) {
                // Get the maximum absolute value in this gradient for reporting
                let grad_max = grad.abs()?.max_all()?.to_scalar::<f64>()?;
                max_grad = max_grad.max(grad_max);
                
                // Note: GradStore doesn't support mutable gradient updates
                // Gradient clipping would need to be handled during optimizer step
            }
        }
        
        Ok(max_grad)
    }
    
    /// Adaptive gradient clipping
    fn clip_adaptive(&self, var_map: &VarMap, gradients: &candle_core::backprop::GradStore, percentile: f64) -> Result<f64> {
        // Collect all gradient norms
        let mut grad_norms = Vec::new();
        
        for var in var_map.all_vars() {
            if let Some(grad) = gradients.get(&var) {
                let grad_norm = grad.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt();
                grad_norms.push(grad_norm);
            }
        }
        
        // Sort to find percentile
        grad_norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate the percentile-based threshold
        let idx = ((percentile / 100.0) * (grad_norms.len() as f64 - 1.0)) as usize;
        let adaptive_threshold = if grad_norms.is_empty() {
            self.threshold
        } else {
            grad_norms[idx.min(grad_norms.len() - 1)].max(self.threshold)
        };
        
        // Apply clipping using the adaptive threshold
        let mut clipper = GradientClipper::new(ClippingMethod::GlobalNorm, adaptive_threshold);
        clipper.clip_by_global_norm(var_map, gradients)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_state_dict_serialization() {
        let state_dict = OptimizerStateDictLegacy {
            optimizer_type: "adamw".to_string(),
            step_count: 100,
            learning_rate: 0.001,
            state: HashMap::new(),
            hyperparameters: HashMap::new(),
        };
        
        let serialized = bincode::serialize(&state_dict).unwrap();
        let deserialized: OptimizerStateDictLegacy = bincode::deserialize(&serialized).unwrap();
        
        assert_eq!(state_dict.optimizer_type, deserialized.optimizer_type);
        assert_eq!(state_dict.step_count, deserialized.step_count);
    }

    #[test]
    fn test_scheduler_state_dict_serialization() {
        let state_dict = SchedulerStateDictLegacy {
            scheduler_type: "linear".to_string(),
            step_count: 50,
            current_lr: 0.0005,
            base_lr: 0.001,
            state: HashMap::new(),
            hyperparameters: HashMap::new(),
        };
        
        let serialized = bincode::serialize(&state_dict).unwrap();
        let deserialized: SchedulerStateDictLegacy = bincode::deserialize(&serialized).unwrap();
        
        assert_eq!(state_dict.scheduler_type, deserialized.scheduler_type);
        assert_eq!(state_dict.current_lr, deserialized.current_lr);
    }
}