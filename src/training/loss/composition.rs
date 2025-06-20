//! Loss composition and multi-task balancing
//!
//! This module provides utilities for combining multiple loss functions
//! and balancing them for multi-task learning scenarios.

use super::{LossFunction, LossConfig, LossMetrics, TaskBalancingStrategy, LossType};
use crate::training::{Result, Error};
use candle_core::{Tensor, Device};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::any::Any;
use anyhow::{Context, anyhow};
use tracing::{debug, warn, instrument};

/// Composite loss function for multi-task learning
pub struct CompositeLoss {
    tasks: HashMap<String, TaskLoss>,
    weight_manager: TaskWeightManager,
    device: Device,
    config: LossConfig,
}

/// Individual task loss with its weight and function
#[derive(Clone)]
pub struct TaskLoss {
    pub name: String,
    pub weight: f64,
    pub loss_fn: Box<dyn LossFunction>,
    pub enabled: bool,
}

/// Task weight management for dynamic balancing
pub struct TaskWeightManager {
    strategy: TaskBalancingStrategy,
    weights: HashMap<String, f64>,
    loss_history: HashMap<String, Vec<f64>>,
    gradient_history: HashMap<String, Vec<f64>>,
    uncertainty_weights: HashMap<String, f64>,
    step_count: usize,
}

/// Configuration for task weight balancing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskBalancingConfig {
    /// Balancing strategy
    pub strategy: TaskBalancingStrategy,
    /// Update frequency (steps)
    pub update_frequency: usize,
    /// History window size for adaptive methods
    pub history_window: usize,
    /// Temperature for uncertainty weighting
    pub temperature: f64,
    /// Minimum weight for any task
    pub min_weight: f64,
    /// Maximum weight for any task
    pub max_weight: f64,
    /// Smoothing factor for weight updates
    pub smoothing_factor: f64,
}

/// Gradient-based task balancing (GradNorm)
pub struct GradNormBalancer {
    alpha: f64,
    target_losses: HashMap<String, f64>,
    shared_params: Vec<String>,
}

/// Uncertainty-based task weighting
pub struct UncertaintyWeighter {
    log_vars: HashMap<String, f64>,
    update_rate: f64,
}

/// Dynamic weight adjustment strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightUpdateStrategy {
    /// Simple loss magnitude balancing
    LossMagnitude { 
        target_balance: f64,
        smoothing: f64,
    },
    /// Gradient norm balancing
    GradientNorm {
        alpha: f64,
        target_losses: HashMap<String, f64>,
    },
    /// Uncertainty-based weighting
    Uncertainty {
        temperature: f64,
        update_rate: f64,
    },
    /// Learning rate based on task difficulty
    TaskDifficulty {
        difficulty_threshold: f64,
        boost_factor: f64,
    },
}

impl CompositeLoss {
    /// Create new composite loss function
    pub fn new(config: LossConfig) -> Result<Self> {
        let device = Device::Cpu; // Default device
        let weight_manager = TaskWeightManager::new(
            TaskBalancingStrategy::Fixed,
            TaskBalancingConfig::default(),
        );
        
        Ok(Self {
            tasks: HashMap::new(),
            weight_manager,
            device,
            config,
        })
    }
    
    /// Create composite loss with specific device
    pub fn with_device(config: LossConfig, device: &Device) -> Result<Self> {
        let weight_manager = TaskWeightManager::new(
            TaskBalancingStrategy::Fixed,
            TaskBalancingConfig::default(),
        );
        
        Ok(Self {
            tasks: HashMap::new(),
            weight_manager,
            device: device.clone(),
            config,
        })
    }
    
    /// Add a task with its loss function and weight
    pub fn add_task(
        &mut self,
        name: &str,
        loss_fn: Box<dyn LossFunction>,
        weight: f64,
    ) -> Result<()> {
        let task_loss = TaskLoss {
            name: name.to_string(),
            weight,
            loss_fn,
            enabled: true,
        };
        
        self.tasks.insert(name.to_string(), task_loss);
        self.weight_manager.add_task(name, weight);
        
        debug!("Added task '{}' with weight {:.4}", name, weight);
        Ok(())
    }
    
    /// Remove a task
    pub fn remove_task(&mut self, name: &str) -> Result<()> {
        self.tasks.remove(name);
        self.weight_manager.remove_task(name);
        debug!("Removed task '{}'", name);
        Ok(())
    }
    
    /// Enable/disable a task
    pub fn set_task_enabled(&mut self, name: &str, enabled: bool) -> Result<()> {
        if let Some(task) = self.tasks.get_mut(name) {
            task.enabled = enabled;
            debug!("Task '{}' enabled: {}", name, enabled);
            Ok(())
        } else {
            Err(anyhow!("Task '{}' not found", name))
        }
    }
    
    /// Update task weights based on current performance
    #[instrument(skip(self, task_losses, gradients))]
    pub fn update_weights(
        &mut self,
        task_losses: &HashMap<String, f64>,
        gradients: Option<&HashMap<String, HashMap<String, Tensor>>>,
    ) -> Result<()> {
        self.weight_manager.update_weights(task_losses, gradients)
    }
    
    /// Get current task weights
    pub fn get_task_weights(&self) -> &HashMap<String, f64> {
        self.weight_manager.get_weights()
    }
    
    /// Set balancing strategy
    pub fn set_balancing_strategy(&mut self, strategy: TaskBalancingStrategy) {
        self.weight_manager.set_strategy(strategy);
    }
    
    /// Compute weighted combination of task losses
    #[instrument(skip(self, batches, predictions))]
    pub fn compute_combined_loss(
        &self,
        batches: &HashMap<String, Box<dyn Any>>,
        predictions: &HashMap<String, Box<dyn Any>>,
    ) -> Result<LossMetrics> {
        let mut combined_metrics = LossMetrics::new();
        let mut task_losses = HashMap::new();
        
        // Compute loss for each enabled task
        for (task_name, task) in &self.tasks {
            if !task.enabled {
                continue;
            }
            
            if let (Some(batch), Some(pred)) = (batches.get(task_name), predictions.get(task_name)) {
                let task_metrics = task.loss_fn.compute_metrics(batch.as_ref(), pred.as_ref())?;
                let weighted_loss = task_metrics.total_loss * task.weight;
                
                task_losses.insert(task_name.clone(), task_metrics.total_loss);
                combined_metrics.add_component(task_name, weighted_loss);
                
                // Add task-specific metrics
                for (metric_name, metric_value) in task_metrics.metrics {
                    combined_metrics.add_metric(
                        &format!("{}_{}", task_name, metric_name),
                        metric_value,
                    );
                }
                
                // Add task-specific regularization losses
                for (reg_name, reg_value) in task_metrics.regularization_losses {
                    combined_metrics.add_regularization_loss(
                        &format!("{}_{}", task_name, reg_name),
                        reg_value * task.weight,
                    );
                }
                
                debug!("Task '{}': loss={:.6}, weight={:.4}, weighted={:.6}", 
                    task_name, task_metrics.total_loss, task.weight, weighted_loss);
            } else {
                warn!("Missing batch or predictions for task '{}'", task_name);
            }
        }
        
        // Compute task balance metrics
        let balance_metrics = self.compute_balance_metrics(&task_losses)?;
        for (name, value) in balance_metrics {
            combined_metrics.add_metric(&name, value);
        }
        
        Ok(combined_metrics)
    }
    
    /// Compute task balance metrics
    fn compute_balance_metrics(&self, task_losses: &HashMap<String, f64>) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();
        
        if task_losses.len() < 2 {
            return Ok(metrics);
        }
        
        let losses: Vec<f64> = task_losses.values().cloned().collect();
        let weights: Vec<f64> = task_losses.keys()
            .map(|name| self.weight_manager.get_weight(name))
            .collect();
        
        // Loss variance (how balanced are the losses)
        let mean_loss = losses.iter().sum::<f64>() / losses.len() as f64;
        let loss_variance = losses.iter()
            .map(|x| (x - mean_loss).powi(2))
            .sum::<f64>() / losses.len() as f64;
        metrics.insert("task_loss_variance".to_string(), loss_variance);
        
        // Weight entropy (how uniform are the weights)
        let weight_sum = weights.iter().sum::<f64>();
        let weight_entropy = if weight_sum > 0.0 {
            -weights.iter()
                .map(|w| {
                    let p = w / weight_sum;
                    if p > 0.0 { p * p.ln() } else { 0.0 }
                })
                .sum::<f64>()
        } else {
            0.0
        };
        metrics.insert("task_weight_entropy".to_string(), weight_entropy);
        
        // Task dominance (max weight / mean weight)
        let max_weight = weights.iter().fold(0.0, |a, &b| a.max(b));
        let mean_weight = weight_sum / weights.len() as f64;
        if mean_weight > 0.0 {
            metrics.insert("task_dominance".to_string(), max_weight / mean_weight);
        }
        
        Ok(metrics)
    }
}

impl LossFunction for CompositeLoss {
    fn forward(&self, batch: &dyn Any, predictions: &dyn Any) -> Result<Tensor> {
        // This is a simplified version - in practice, you'd handle multiple batches/predictions
        let mut total_loss = 0.0;
        
        for (task_name, task) in &self.tasks {
            if task.enabled {
                // In a real implementation, you'd extract task-specific data
                let task_loss_tensor = task.loss_fn.forward(batch, predictions)?;
                let task_loss_value = task_loss_tensor.to_scalar::<f64>()?;
                total_loss += task_loss_value * task.weight;
            }
        }
        
        Tensor::full(total_loss, (), &self.device)
            .context("Failed to create combined loss tensor")
    }
    
    fn compute_metrics(&self, batch: &dyn Any, predictions: &dyn Any) -> Result<LossMetrics> {
        // Simplified version - in practice, you'd handle task-specific batches
        let mut combined_metrics = LossMetrics::new();
        
        for (task_name, task) in &self.tasks {
            if task.enabled {
                let task_metrics = task.loss_fn.compute_metrics(batch, predictions)?;
                let weighted_loss = task_metrics.total_loss * task.weight;
                combined_metrics.add_component(task_name, weighted_loss);
            }
        }
        
        Ok(combined_metrics)
    }
    
    fn config(&self) -> &LossConfig {
        &self.config
    }
    
    fn clone_box(&self) -> Box<dyn LossFunction> {
        // Simplified clone - in practice, you'd properly clone all tasks
        Box::new(Self {
            tasks: HashMap::new(), // Simplified
            weight_manager: TaskWeightManager::new(
                TaskBalancingStrategy::Fixed,
                TaskBalancingConfig::default(),
            ),
            device: self.device.clone(),
            config: self.config.clone(),
        })
    }
}

impl TaskWeightManager {
    /// Create new task weight manager
    pub fn new(strategy: TaskBalancingStrategy, config: TaskBalancingConfig) -> Self {
        Self {
            strategy,
            weights: HashMap::new(),
            loss_history: HashMap::new(),
            gradient_history: HashMap::new(),
            uncertainty_weights: HashMap::new(),
            step_count: 0,
        }
    }
    
    /// Add task with initial weight
    pub fn add_task(&mut self, name: &str, weight: f64) {
        self.weights.insert(name.to_string(), weight);
        self.loss_history.insert(name.to_string(), Vec::new());
        self.gradient_history.insert(name.to_string(), Vec::new());
        self.uncertainty_weights.insert(name.to_string(), 1.0);
    }
    
    /// Remove task
    pub fn remove_task(&mut self, name: &str) {
        self.weights.remove(name);
        self.loss_history.remove(name);
        self.gradient_history.remove(name);
        self.uncertainty_weights.remove(name);
    }
    
    /// Update weights based on current losses and gradients
    #[instrument(skip(self, task_losses, gradients))]
    pub fn update_weights(
        &mut self,
        task_losses: &HashMap<String, f64>,
        gradients: Option<&HashMap<String, HashMap<String, Tensor>>>,
    ) -> Result<()> {
        self.step_count += 1;
        
        // Store loss history
        for (task_name, &loss_value) in task_losses {
            if let Some(history) = self.loss_history.get_mut(task_name) {
                history.push(loss_value);
                if history.len() > 100 { // Keep last 100 values
                    history.remove(0);
                }
            }
        }
        
        // Update weights based on strategy
        match &self.strategy {
            TaskBalancingStrategy::Fixed => {
                // No update needed for fixed weights
            },
            TaskBalancingStrategy::Dynamic => {
                self.update_dynamic_weights(task_losses)?;
            },
            TaskBalancingStrategy::Uncertainty => {
                self.update_uncertainty_weights(task_losses)?;
            },
            TaskBalancingStrategy::GradNorm => {
                if let Some(grads) = gradients {
                    self.update_gradnorm_weights(task_losses, grads)?;
                }
            },
        }
        
        debug!("Updated task weights at step {}: {:?}", self.step_count, self.weights);
        Ok(())
    }
    
    /// Update weights using dynamic balancing
    fn update_dynamic_weights(&mut self, task_losses: &HashMap<String, f64>) -> Result<()> {
        let loss_sum: f64 = task_losses.values().sum();
        if loss_sum <= 0.0 {
            return Ok(());
        }
        
        // Inverse loss weighting (higher weight for higher loss)
        let max_loss = task_losses.values().fold(0.0, |a, &b| a.max(b));
        
        for (task_name, &loss_value) in task_losses {
            if let Some(current_weight) = self.weights.get_mut(task_name) {
                let target_weight = if max_loss > 0.0 {
                    loss_value / max_loss
                } else {
                    1.0
                };
                
                // Smooth update
                *current_weight = 0.9 * *current_weight + 0.1 * target_weight;
            }
        }
        
        // Normalize weights
        self.normalize_weights();
        Ok(())
    }
    
    /// Update weights using uncertainty weighting
    fn update_uncertainty_weights(&mut self, task_losses: &HashMap<String, f64>) -> Result<()> {
        let window_size = 10;
        
        for (task_name, &current_loss) in task_losses {
            if let Some(history) = self.loss_history.get(task_name) {
                if history.len() >= window_size {
                    let recent_losses = &history[history.len() - window_size..];
                    let mean_loss = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
                    let variance = recent_losses.iter()
                        .map(|x| (x - mean_loss).powi(2))
                        .sum::<f64>() / (recent_losses.len() - 1) as f64;
                    
                    // Higher uncertainty -> lower weight
                    let uncertainty = variance.sqrt();
                    let uncertainty_weight = 1.0 / (1.0 + uncertainty);
                    
                    if let Some(weight) = self.uncertainty_weights.get_mut(task_name) {
                        *weight = 0.9 * *weight + 0.1 * uncertainty_weight;
                    }
                    
                    if let Some(task_weight) = self.weights.get_mut(task_name) {
                        *task_weight = self.uncertainty_weights[task_name];
                    }
                }
            }
        }
        
        self.normalize_weights();
        Ok(())
    }
    
    /// Update weights using GradNorm algorithm
    fn update_gradnorm_weights(
        &mut self,
        task_losses: &HashMap<String, f64>,
        gradients: &HashMap<String, HashMap<String, Tensor>>,
    ) -> Result<()> {
        // Simplified GradNorm implementation
        // In practice, you'd compute gradients w.r.t. shared parameters
        
        let mut grad_norms = HashMap::new();
        
        // Compute gradient norms for each task
        for (task_name, task_grads) in gradients {
            let mut total_norm = 0.0;
            for (_, grad) in task_grads {
                let grad_norm = grad.sqr()?.sum_all()?.to_scalar::<f64>()?;
                total_norm += grad_norm;
            }
            grad_norms.insert(task_name.clone(), total_norm.sqrt());
        }
        
        // Balance gradient norms
        let mean_grad_norm = grad_norms.values().sum::<f64>() / grad_norms.len() as f64;
        
        for (task_name, &grad_norm) in &grad_norms {
            if let Some(weight) = self.weights.get_mut(task_name) {
                let target_ratio = mean_grad_norm / grad_norm.max(1e-8);
                *weight *= target_ratio.min(2.0).max(0.5); // Limit weight changes
            }
        }
        
        self.normalize_weights();
        Ok(())
    }
    
    /// Normalize weights to sum to number of tasks
    fn normalize_weights(&mut self) {
        let weight_sum: f64 = self.weights.values().sum();
        let num_tasks = self.weights.len() as f64;
        
        if weight_sum > 0.0 && num_tasks > 0.0 {
            let scale = num_tasks / weight_sum;
            for weight in self.weights.values_mut() {
                *weight *= scale;
                *weight = weight.max(0.1).min(10.0); // Clamp weights
            }
        }
    }
    
    /// Get current weights
    pub fn get_weights(&self) -> &HashMap<String, f64> {
        &self.weights
    }
    
    /// Get weight for specific task
    pub fn get_weight(&self, task_name: &str) -> f64 {
        self.weights.get(task_name).copied().unwrap_or(1.0)
    }
    
    /// Set balancing strategy
    pub fn set_strategy(&mut self, strategy: TaskBalancingStrategy) {
        self.strategy = strategy;
    }
    
    /// Get loss history for a task
    pub fn get_loss_history(&self, task_name: &str) -> Option<&Vec<f64>> {
        self.loss_history.get(task_name)
    }
}

impl Default for TaskBalancingConfig {
    fn default() -> Self {
        Self {
            strategy: TaskBalancingStrategy::Fixed,
            update_frequency: 100,
            history_window: 50,
            temperature: 1.0,
            min_weight: 0.1,
            max_weight: 10.0,
            smoothing_factor: 0.9,
        }
    }
}