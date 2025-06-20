//! Optimizer and scheduler state management
//!
//! This module provides state management utilities for optimizers and schedulers,
//! including serialization, deserialization, and state transfer capabilities.

use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Comprehensive optimizer state for checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Optimizer configuration
    pub config: OptimizerConfig,
    
    /// Current training step
    pub step: usize,
    
    /// Current learning rate
    pub learning_rate: f64,
    
    /// Parameter-specific state
    pub parameter_states: HashMap<String, ParameterState>,
    
    /// Global optimizer state
    pub global_state: HashMap<String, f64>,
    
    /// Optimizer metadata
    pub metadata: OptimizerMetadata,
}

/// Scheduler state for checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerState {
    /// Scheduler configuration
    pub config: SchedulerConfig,
    
    /// Current step
    pub step: usize,
    
    /// Current learning rate
    pub current_lr: f64,
    
    /// Base learning rate
    pub base_lr: f64,
    
    /// Scheduler-specific state
    pub scheduler_state: HashMap<String, f64>,
    
    /// Scheduler metadata
    pub metadata: SchedulerMetadata,
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Optimizer type
    pub optimizer_type: String,
    
    /// Learning rate
    pub learning_rate: f64,
    
    /// Optimizer hyperparameters
    pub hyperparameters: HashMap<String, f64>,
    
    /// Gradient clipping settings
    pub gradient_clipping: Option<GradientClippingConfig>,
    
    /// Weight decay settings
    pub weight_decay: f64,
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduler type
    pub scheduler_type: String,
    
    /// Base learning rate
    pub base_lr: f64,
    
    /// Scheduler hyperparameters
    pub hyperparameters: HashMap<String, f64>,
    
    /// Warmup configuration
    pub warmup: Option<WarmupConfig>,
}

/// Gradient clipping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientClippingConfig {
    /// Clipping method
    pub method: String,
    
    /// Clipping threshold
    pub threshold: f64,
    
    /// Additional parameters
    pub params: HashMap<String, f64>,
}

/// Warmup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupConfig {
    /// Warmup steps
    pub steps: usize,
    
    /// Warmup method
    pub method: String,
    
    /// Initial learning rate factor
    pub initial_lr_factor: f64,
}

/// Per-parameter optimizer state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterState {
    /// Parameter name
    pub name: String,
    
    /// Parameter shape
    pub shape: Vec<usize>,
    
    /// Step count for this parameter
    pub step: usize,
    
    /// Momentum/moving average states
    pub momentum_states: HashMap<String, TensorState>,
    
    /// Variance/second moment states
    pub variance_states: HashMap<String, TensorState>,
    
    /// Parameter-specific metadata
    pub metadata: ParameterMetadata,
}

/// Tensor state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorState {
    /// Tensor shape
    pub shape: Vec<usize>,
    
    /// Flattened tensor data
    pub data: Vec<f64>,
    
    /// Data type
    pub dtype: String,
    
    /// Device
    pub device: String,
    
    /// Creation timestamp
    pub created_at: String,
    
    /// Last update timestamp
    pub updated_at: String,
}

/// Parameter metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterMetadata {
    /// Parameter initialization method
    pub initialization: String,
    
    /// Parameter update count
    pub update_count: usize,
    
    /// Last gradient norm
    pub last_grad_norm: f64,
    
    /// Last parameter norm
    pub last_param_norm: f64,
    
    /// Average gradient magnitude
    pub avg_grad_magnitude: f64,
    
    /// Parameter statistics
    pub statistics: ParameterStatistics,
}

/// Parameter statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterStatistics {
    /// Mean value
    pub mean: f64,
    
    /// Standard deviation
    pub std: f64,
    
    /// Minimum value
    pub min: f64,
    
    /// Maximum value
    pub max: f64,
    
    /// L1 norm
    pub l1_norm: f64,
    
    /// L2 norm
    pub l2_norm: f64,
    
    /// Sparsity ratio
    pub sparsity: f64,
}

/// Optimizer metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerMetadata {
    /// Creation timestamp
    pub created_at: String,
    
    /// Last update timestamp
    pub updated_at: String,
    
    /// Total steps taken
    pub total_steps: usize,
    
    /// Total parameters
    pub total_parameters: usize,
    
    /// Optimizer version
    pub version: String,
    
    /// Performance statistics
    pub performance: OptimizerPerformance,
}

/// Scheduler metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerMetadata {
    /// Creation timestamp
    pub created_at: String,
    
    /// Last update timestamp
    pub updated_at: String,
    
    /// Total steps taken
    pub total_steps: usize,
    
    /// Scheduler version
    pub version: String,
    
    /// Learning rate history (last N values)
    pub lr_history: Vec<f64>,
    
    /// Performance statistics
    pub performance: SchedulerPerformance,
}

/// Optimizer performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerPerformance {
    /// Average step time in milliseconds
    pub avg_step_time_ms: f64,
    
    /// Total optimization time in seconds
    pub total_time_seconds: f64,
    
    /// Steps per second
    pub steps_per_second: f64,
    
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    
    /// Gradient norms history
    pub gradient_norms: Vec<f64>,
    
    /// Parameter update magnitudes
    pub update_magnitudes: Vec<f64>,
}

/// Scheduler performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerPerformance {
    /// Average update time in microseconds
    pub avg_update_time_us: f64,
    
    /// Total updates
    pub total_updates: usize,
    
    /// Learning rate variance
    pub lr_variance: f64,
    
    /// Effective learning rate (weighted average)
    pub effective_lr: f64,
}

impl OptimizerState {
    /// Create a new optimizer state
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            step: 0,
            learning_rate: 0.0,
            parameter_states: HashMap::new(),
            global_state: HashMap::new(),
            metadata: OptimizerMetadata::new(),
        }
    }
    
    /// Add parameter state
    pub fn add_parameter(&mut self, name: String, shape: Vec<usize>) {
        let param_state = ParameterState::new(name.clone(), shape);
        self.parameter_states.insert(name, param_state);
    }
    
    /// Update parameter state
    pub fn update_parameter(&mut self, name: &str, momentum: Option<TensorState>, variance: Option<TensorState>) {
        if let Some(param_state) = self.parameter_states.get_mut(name) {
            param_state.step += 1;
            
            if let Some(momentum) = momentum {
                param_state.momentum_states.insert("momentum".to_string(), momentum);
            }
            
            if let Some(variance) = variance {
                param_state.variance_states.insert("variance".to_string(), variance);
            }
            
            param_state.metadata.update_count += 1;
            param_state.metadata.updated_at();
        }
    }
    
    /// Get parameter state
    pub fn get_parameter(&self, name: &str) -> Option<&ParameterState> {
        self.parameter_states.get(name)
    }
    
    /// Update global state
    pub fn update_global_state(&mut self, key: String, value: f64) {
        self.global_state.insert(key, value);
    }
    
    /// Step the optimizer
    pub fn step(&mut self) {
        self.step += 1;
        self.metadata.total_steps += 1;
        self.metadata.updated_at();
    }
    
    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.step = 0;
        self.parameter_states.clear();
        self.global_state.clear();
        self.metadata = OptimizerMetadata::new();
    }
    
    /// Get memory usage estimate
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = 0;
        
        for param_state in self.parameter_states.values() {
            for tensor_state in param_state.momentum_states.values() {
                total += tensor_state.data.len() * std::mem::size_of::<f64>();
            }
            for tensor_state in param_state.variance_states.values() {
                total += tensor_state.data.len() * std::mem::size_of::<f64>();
            }
        }
        
        total
    }
    
    /// Validate state consistency
    pub fn validate(&self) -> Result<()> {
        // Check that all required fields are present
        if self.config.optimizer_type.is_empty() {
            return Err(anyhow::anyhow!("Optimizer type not specified"));
        }
        
        // Check parameter states
        for (name, param_state) in &self.parameter_states {
            if param_state.name != *name {
                return Err(anyhow::anyhow!("Parameter name mismatch: {} vs {}", name, param_state.name));
            }
            
            // Validate tensor states
            for tensor_state in param_state.momentum_states.values() {
                tensor_state.validate()?;
            }
            for tensor_state in param_state.variance_states.values() {
                tensor_state.validate()?;
            }
        }
        
        Ok(())
    }
}

impl SchedulerState {
    /// Create a new scheduler state
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            step: 0,
            current_lr: 0.0,
            base_lr: 0.0,
            scheduler_state: HashMap::new(),
            metadata: SchedulerMetadata::new(),
        }
    }
    
    /// Step the scheduler
    pub fn step(&mut self, new_lr: f64) {
        self.step += 1;
        self.current_lr = new_lr;
        self.metadata.total_steps += 1;
        self.metadata.lr_history.push(new_lr);
        
        // Keep only last 1000 learning rates
        if self.metadata.lr_history.len() > 1000 {
            self.metadata.lr_history.remove(0);
        }
        
        self.metadata.updated_at();
    }
    
    /// Reset scheduler state
    pub fn reset(&mut self) {
        self.step = 0;
        self.current_lr = self.base_lr;
        self.scheduler_state.clear();
        self.metadata = SchedulerMetadata::new();
    }
    
    /// Update scheduler state
    pub fn update_state(&mut self, key: String, value: f64) {
        self.scheduler_state.insert(key, value);
    }
    
    /// Get scheduler state value
    pub fn get_state(&self, key: &str) -> Option<f64> {
        self.scheduler_state.get(key).copied()
    }
    
    /// Calculate learning rate statistics
    pub fn lr_statistics(&self) -> (f64, f64, f64, f64) {
        if self.metadata.lr_history.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        
        let mean = self.metadata.lr_history.iter().sum::<f64>() / self.metadata.lr_history.len() as f64;
        let min = self.metadata.lr_history.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = self.metadata.lr_history.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        let variance = self.metadata.lr_history.iter()
            .map(|&lr| (lr - mean).powi(2))
            .sum::<f64>() / self.metadata.lr_history.len() as f64;
        let std_dev = variance.sqrt();
        
        (mean, std_dev, min, max)
    }
}

impl ParameterState {
    /// Create a new parameter state
    pub fn new(name: String, shape: Vec<usize>) -> Self {
        Self {
            name,
            shape,
            step: 0,
            momentum_states: HashMap::new(),
            variance_states: HashMap::new(),
            metadata: ParameterMetadata::new(),
        }
    }
    
    /// Update parameter metadata
    pub fn update_metadata(&mut self, grad_norm: f64, param_norm: f64) {
        self.metadata.last_grad_norm = grad_norm;
        self.metadata.last_param_norm = param_norm;
        self.metadata.update_count += 1;
        
        // Update running average of gradient magnitude
        let alpha = 0.1; // Exponential moving average factor
        self.metadata.avg_grad_magnitude = 
            alpha * grad_norm + (1.0 - alpha) * self.metadata.avg_grad_magnitude;
    }
}

impl TensorState {
    /// Create a new tensor state
    pub fn new(shape: Vec<usize>, data: Vec<f64>, dtype: String, device: String) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            shape,
            data,
            dtype,
            device,
            created_at: now.clone(),
            updated_at: now,
        }
    }
    
    /// Update tensor data
    pub fn update(&mut self, data: Vec<f64>) {
        self.data = data;
        self.updated_at = chrono::Utc::now().to_rfc3339();
    }
    
    /// Validate tensor state
    pub fn validate(&self) -> Result<()> {
        let expected_size: usize = self.shape.iter().product();
        if self.data.len() != expected_size {
            return Err(anyhow::anyhow!(
                "Data size mismatch: expected {}, got {}", 
                expected_size, 
                self.data.len()
            ));
        }
        Ok(())
    }
    
    /// Get tensor statistics
    pub fn statistics(&self) -> ParameterStatistics {
        if self.data.is_empty() {
            return ParameterStatistics::default();
        }
        
        let mean = self.data.iter().sum::<f64>() / self.data.len() as f64;
        let variance = self.data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / self.data.len() as f64;
        let std = variance.sqrt();
        
        let min = self.data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = self.data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        let l1_norm = self.data.iter().map(|&x| x.abs()).sum::<f64>();
        let l2_norm = self.data.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
        
        let zeros = self.data.iter().filter(|&&x| x.abs() < 1e-8).count();
        let sparsity = zeros as f64 / self.data.len() as f64;
        
        ParameterStatistics {
            mean,
            std,
            min,
            max,
            l1_norm,
            l2_norm,
            sparsity,
        }
    }
}

impl ParameterMetadata {
    /// Create new parameter metadata
    pub fn new() -> Self {
        Self {
            initialization: "unknown".to_string(),
            update_count: 0,
            last_grad_norm: 0.0,
            last_param_norm: 0.0,
            avg_grad_magnitude: 0.0,
            statistics: ParameterStatistics::default(),
        }
    }
    
    /// Update timestamp
    pub fn updated_at(&mut self) {
        // In a real implementation, you might store a timestamp
    }
}

impl OptimizerMetadata {
    /// Create new optimizer metadata
    pub fn new() -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            created_at: now.clone(),
            updated_at: now,
            total_steps: 0,
            total_parameters: 0,
            version: "1.0.0".to_string(),
            performance: OptimizerPerformance::new(),
        }
    }
    
    /// Update timestamp
    pub fn updated_at(&mut self) {
        self.updated_at = chrono::Utc::now().to_rfc3339();
    }
}

impl SchedulerMetadata {
    /// Create new scheduler metadata
    pub fn new() -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            created_at: now.clone(),
            updated_at: now,
            total_steps: 0,
            version: "1.0.0".to_string(),
            lr_history: Vec::new(),
            performance: SchedulerPerformance::new(),
        }
    }
    
    /// Update timestamp
    pub fn updated_at(&mut self) {
        self.updated_at = chrono::Utc::now().to_rfc3339();
    }
}

impl OptimizerPerformance {
    /// Create new optimizer performance stats
    pub fn new() -> Self {
        Self {
            avg_step_time_ms: 0.0,
            total_time_seconds: 0.0,
            steps_per_second: 0.0,
            memory_usage_bytes: 0,
            gradient_norms: Vec::new(),
            update_magnitudes: Vec::new(),
        }
    }
}

impl SchedulerPerformance {
    /// Create new scheduler performance stats
    pub fn new() -> Self {
        Self {
            avg_update_time_us: 0.0,
            total_updates: 0,
            lr_variance: 0.0,
            effective_lr: 0.0,
        }
    }
}

impl Default for ParameterStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            l1_norm: 0.0,
            l2_norm: 0.0,
            sparsity: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_state_creation() {
        let config = OptimizerConfig {
            optimizer_type: "adamw".to_string(),
            learning_rate: 0.001,
            hyperparameters: HashMap::new(),
            gradient_clipping: None,
            weight_decay: 0.01,
        };
        
        let state = OptimizerState::new(config);
        assert_eq!(state.step, 0);
        assert!(state.parameter_states.is_empty());
    }

    #[test]
    fn test_parameter_state_creation() {
        let param_state = ParameterState::new("test_param".to_string(), vec![10, 20]);
        assert_eq!(param_state.name, "test_param");
        assert_eq!(param_state.shape, vec![10, 20]);
        assert_eq!(param_state.step, 0);
    }

    #[test]
    fn test_tensor_state_validation() {
        let tensor_state = TensorState::new(
            vec![2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f64".to_string(),
            "cpu".to_string(),
        );
        
        assert!(tensor_state.validate().is_ok());
        
        let invalid_tensor_state = TensorState::new(
            vec![2, 3],
            vec![1.0, 2.0], // Wrong size
            "f64".to_string(),
            "cpu".to_string(),
        );
        
        assert!(invalid_tensor_state.validate().is_err());
    }

    #[test]
    fn test_tensor_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor_state = TensorState::new(
            vec![5],
            data.clone(),
            "f64".to_string(),
            "cpu".to_string(),
        );
        
        let stats = tensor_state.statistics();
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }
}