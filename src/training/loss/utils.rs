//! Loss computation utilities and numerical stability helpers
//!
//! This module provides utilities for loss computation, numerical stability,
//! metrics calculation, and gradient handling.

use super::{StabilityConfig, LossScalingConfig};
use crate::training::Result;
use candle_core::{Tensor, Device, DType, D};
use candle_nn as nn;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Context;
use tracing::{debug, warn, instrument};

/// Loss computation utilities
pub struct LossUtils {
    device: Device,
    stability_config: StabilityConfig,
}

/// Numerical stability helpers
pub struct NumericalStability {
    config: StabilityConfig,
    device: Device,
}

/// Metrics computation engine
pub struct MetricsComputer {
    device: Device,
    metric_cache: HashMap<String, f64>,
}

/// Loss scaling manager for mixed precision training
pub struct LossScaler {
    config: LossScalingConfig,
    current_scale: f64,
    steps_since_update: usize,
    consecutive_good_steps: usize,
}

/// Gradient statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientStats {
    /// Gradient norms by parameter name
    pub norms: HashMap<String, f64>,
    /// Total gradient norm
    pub total_norm: f64,
    /// Maximum gradient value
    pub max_grad: f64,
    /// Minimum gradient value
    pub min_grad: f64,
    /// Number of NaN/Inf gradients
    pub num_nan_inf: usize,
    /// Gradient sparsity (fraction of zero gradients)
    pub sparsity: f64,
}

/// Moving average tracker for smooth metrics
pub struct MovingAverage {
    values: Vec<f64>,
    window_size: usize,
    current_sum: f64,
    current_count: usize,
}

impl LossUtils {
    /// Create new loss utilities
    pub fn new(device: &Device, stability_config: StabilityConfig) -> Self {
        Self {
            device: device.clone(),
            stability_config,
        }
    }
    
    /// Safely compute loss with numerical stability checks
    #[instrument(skip(self, loss_tensor))]
    pub fn compute_stable_loss(&self, loss_tensor: &Tensor) -> Result<f64> {
        // Check for NaN/Inf values
        if self.has_nan_or_inf(loss_tensor)? {
            warn!("Loss contains NaN or Inf values, applying stability correction");
            return Ok(self.stability_config.max_loss_value);
        }
        
        let loss_value = loss_tensor.to_scalar::<f64>()
            .context("Failed to convert loss tensor to scalar")?;
        
        // Clamp loss if enabled
        if self.stability_config.clamp_losses {
            let clamped_loss = loss_value.min(self.stability_config.max_loss_value).max(0.0);
            if clamped_loss != loss_value {
                debug!("Clamped loss from {:.6} to {:.6}", loss_value, clamped_loss);
            }
            Ok(clamped_loss)
        } else {
            Ok(loss_value)
        }
    }
    
    /// Check if tensor contains NaN or Inf values
    fn has_nan_or_inf(&self, tensor: &Tensor) -> Result<bool> {
        // Check for NaN by comparing tensor with itself (NaN != NaN)
        let ne_result = tensor.ne(tensor)?;
        let has_nan = ne_result.sum_all()?.to_scalar::<f64>()? > 0.0;
        if has_nan {
            return Ok(true);
        }
        
        // Check for infinity by comparing absolute value with a large number
        let max_val = tensor.abs()?.max(D::Minus1)?.max(D::Minus1)?.to_scalar::<f64>()?;
        Ok(max_val.is_infinite())
    }
    
    /// Weighted loss combination for multi-task learning
    pub fn combine_weighted_losses(
        &self,
        losses: &HashMap<String, f64>,
        weights: &HashMap<String, f64>,
    ) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut total_weight = 0.0;
        
        for (task_name, loss_value) in losses {
            let weight = weights.get(task_name).unwrap_or(&1.0);
            total_loss += loss_value * weight;
            total_weight += weight;
        }
        
        if total_weight > 0.0 {
            Ok(total_loss / total_weight)
        } else {
            Ok(0.0)
        }
    }
    
    /// Compute loss gradients with respect to parameters
    pub fn compute_loss_gradients(
        &self,
        loss: &Tensor,
        parameters: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut gradients = HashMap::new();
        
        for (name, param) in parameters {
            if let Ok(grad) = loss.backward() {
                // In practice, you'd extract the gradient for this specific parameter
                // This is a simplified version
                // In practice, you'd extract the gradient for this specific parameter
                // This is a simplified placeholder
                gradients.insert(name.clone(), loss.clone());
            }
        }
        
        Ok(gradients)
    }
    
    /// Apply gradient clipping to prevent exploding gradients
    pub fn clip_gradients(
        &self,
        gradients: &mut HashMap<String, Tensor>,
        max_norm: f64,
    ) -> Result<f64> {
        let total_norm = self.compute_gradient_norm(gradients)?;
        
        if total_norm > max_norm {
            let clip_coeff = max_norm / total_norm;
            for (name, grad) in gradients.iter_mut() {
                *grad = (grad.as_ref() * clip_coeff)
                    .with_context(|| format!("Failed to clip gradient for {}", name))?;
            }
            debug!("Clipped gradients: norm {:.6} -> {:.6}", total_norm, max_norm);
        }
        
        Ok(total_norm)
    }
    
    /// Compute total gradient norm
    pub fn compute_gradient_norm(&self, gradients: &HashMap<String, Tensor>) -> Result<f64> {
        let mut total_norm_sq = 0.0;
        
        for (_, grad) in gradients {
            let grad_norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f64>()?;
            total_norm_sq += grad_norm_sq;
        }
        
        Ok(total_norm_sq.sqrt())
    }
    
    /// Smooth loss using exponential moving average
    pub fn smooth_loss(&self, current_loss: f64, previous_smooth: f64, beta: f64) -> f64 {
        beta * previous_smooth + (1.0 - beta) * current_loss
    }
    
    /// Compute loss variance for stability monitoring
    pub fn compute_loss_variance(&self, loss_history: &[f64], window_size: usize) -> f64 {
        if loss_history.len() < window_size || window_size < 2 {
            return 0.0;
        }
        
        let recent_losses = &loss_history[loss_history.len() - window_size..];
        let mean = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
        let variance = recent_losses.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (recent_losses.len() - 1) as f64;
        
        variance
    }
}

impl NumericalStability {
    /// Create new numerical stability helper
    pub fn new(config: StabilityConfig, device: &Device) -> Self {
        Self {
            config,
            device: device.clone(),
        }
    }
    
    /// Add epsilon for numerical stability
    pub fn add_epsilon(&self, tensor: &Tensor) -> Result<Tensor> {
        (tensor + self.config.epsilon).context("Failed to add epsilon")
    }
    
    /// Safe division with epsilon
    pub fn safe_divide(&self, numerator: &Tensor, denominator: &Tensor) -> Result<Tensor> {
        let safe_denom = self.add_epsilon(denominator)?;
        (numerator / safe_denom).context("Failed to perform safe division")
    }
    
    /// Safe logarithm
    pub fn safe_log(&self, tensor: &Tensor) -> Result<Tensor> {
        let safe_input = self.add_epsilon(tensor)?;
        safe_input.log().context("Failed to compute safe logarithm")
    }
    
    /// Safe square root
    pub fn safe_sqrt(&self, tensor: &Tensor) -> Result<Tensor> {
        let safe_input = self.add_epsilon(tensor)?;
        safe_input.sqrt().context("Failed to compute safe square root")
    }
    
    /// Clamp tensor values to prevent overflow
    pub fn clamp_tensor(&self, tensor: &Tensor, min_val: f64, max_val: f64) -> Result<Tensor> {
        tensor.clamp(min_val, max_val).context("Failed to clamp tensor")
    }
    
    /// Replace NaN/Inf values with safe defaults
    pub fn replace_nan_inf(&self, tensor: &Tensor, replacement: f64) -> Result<Tensor> {
        // Create masks for NaN and Inf values
        let is_nan = tensor.ne(tensor)?; // NaN != NaN
        let is_inf = tensor.abs()?.gt(&Tensor::full(1e30, tensor.shape().dims(), tensor.device())?)?;
        // Combine NaN and Inf masks using element-wise maximum
        let is_invalid = (&is_nan.to_dtype(DType::F32)? + &is_inf.to_dtype(DType::F32)?)?.ge(&Tensor::full(0.5, is_nan.shape(), is_nan.device())?)?;
        
        let replacement_tensor = Tensor::full(replacement, tensor.shape().dims(), tensor.device())?;
        is_invalid.where_cond(&replacement_tensor, tensor)
            .context("Failed to replace NaN/Inf values")
    }
    
    /// Normalize tensor for numerical stability
    pub fn normalize_tensor(&self, tensor: &Tensor, dim: usize) -> Result<Tensor> {
        let norm = tensor.sqr()?.sum_keepdim(dim)?.sqrt()?;
        let safe_norm = self.add_epsilon(&norm)?;
        (tensor / safe_norm).context("Failed to normalize tensor")
    }
    
    /// Check tensor health (NaN, Inf, extreme values)
    pub fn check_tensor_health(&self, tensor: &Tensor, name: &str) -> Result<TensorHealthReport> {
        let ne_result = tensor.ne(tensor)?;
        let has_nan = ne_result.sum_all()?.to_scalar::<f64>()? > 0.0;
        let abs_tensor = tensor.abs()?;
        let max_val = abs_tensor.max(D::Minus1)?.max(D::Minus1)?.to_scalar::<f64>()?;
        let min_val = abs_tensor.min(D::Minus1)?.min(D::Minus1)?.to_scalar::<f64>()?;
        let mean_val = tensor.mean_all()?.to_scalar::<f64>()?;
        
        let health_status = if has_nan {
            TensorHealthStatus::HasNaN
        } else if max_val > 1e6 {
            TensorHealthStatus::ExtremeValues
        } else if max_val < 1e-8 {
            TensorHealthStatus::VanishingValues
        } else {
            TensorHealthStatus::Healthy
        };
        
        Ok(TensorHealthReport {
            name: name.to_string(),
            status: health_status,
            has_nan,
            max_value: max_val,
            min_value: min_val,
            mean_value: mean_val,
            shape: tensor.shape().dims().to_vec(),
        })
    }
}

/// Tensor health report
#[derive(Debug, Clone)]
pub struct TensorHealthReport {
    pub name: String,
    pub status: TensorHealthStatus,
    pub has_nan: bool,
    pub max_value: f64,
    pub min_value: f64,
    pub mean_value: f64,
    pub shape: Vec<usize>,
}

/// Tensor health status
#[derive(Debug, Clone, PartialEq)]
pub enum TensorHealthStatus {
    Healthy,
    HasNaN,
    ExtremeValues,
    VanishingValues,
}

impl MetricsComputer {
    /// Create new metrics computer
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            metric_cache: HashMap::new(),
        }
    }
    
    /// Compute comprehensive loss metrics
    #[instrument(skip(self, predictions, targets))]
    pub fn compute_comprehensive_metrics(
        &mut self,
        predictions: &Tensor,
        targets: &Tensor,
        task_type: &str,
    ) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();
        
        match task_type {
            "classification" => {
                metrics.extend(self.compute_classification_metrics(predictions, targets)?);
            },
            "regression" => {
                metrics.extend(self.compute_regression_metrics(predictions, targets)?);
            },
            "generation" => {
                metrics.extend(self.compute_generation_metrics(predictions, targets)?);
            },
            _ => {
                metrics.insert("basic_mse".to_string(), 
                    self.compute_mse_metric(predictions, targets)?);
            }
        }
        
        // Cache metrics for later use
        for (name, value) in &metrics {
            self.metric_cache.insert(name.clone(), *value);
        }
        
        Ok(metrics)
    }
    
    /// Compute classification metrics
    fn compute_classification_metrics(&self, predictions: &Tensor, targets: &Tensor) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();
        
        // Accuracy
        let pred_classes = predictions.argmax_keepdim(D::Minus1)?;
        let target_expanded = targets.unsqueeze(D::Minus1)?;
        let correct = pred_classes.eq(&target_expanded)?.to_dtype(DType::F32)?;
        let accuracy = correct.mean_all()?.to_scalar::<f64>()?;
        metrics.insert("accuracy".to_string(), accuracy);
        
        // Top-k accuracy (if applicable)
        if predictions.dim(D::Minus1)? >= 5 {
            let top5_acc = self.compute_topk_accuracy(predictions, targets, 5)?;
            metrics.insert("top5_accuracy".to_string(), top5_acc);
        }
        
        // Cross-entropy loss
        let ce_loss = nn::loss::cross_entropy(predictions, targets)?;
        let ce_value = ce_loss.mean_all()?.to_scalar::<f64>()?;
        metrics.insert("cross_entropy".to_string(), ce_value);
        
        Ok(metrics)
    }
    
    /// Compute regression metrics
    fn compute_regression_metrics(&self, predictions: &Tensor, targets: &Tensor) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();
        
        // MSE
        let mse = self.compute_mse_metric(predictions, targets)?;
        metrics.insert("mse".to_string(), mse);
        
        // RMSE
        metrics.insert("rmse".to_string(), mse.sqrt());
        
        // MAE
        let mae = (predictions - targets)?.abs()?.mean_all()?.to_scalar::<f64>()?;
        metrics.insert("mae".to_string(), mae);
        
        // R²
        let r_squared = self.compute_r_squared(predictions, targets)?;
        metrics.insert("r_squared".to_string(), r_squared);
        
        Ok(metrics)
    }
    
    /// Compute generation metrics
    fn compute_generation_metrics(&self, predictions: &Tensor, targets: &Tensor) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();
        
        // Perplexity
        let ce_loss = nn::loss::cross_entropy(predictions, targets)?;
        let perplexity = ce_loss.mean_all()?.exp()?.to_scalar::<f64>()?;
        metrics.insert("perplexity".to_string(), perplexity);
        
        // Token accuracy
        let token_acc = self.compute_token_accuracy(predictions, targets)?;
        metrics.insert("token_accuracy".to_string(), token_acc);
        
        Ok(metrics)
    }
    
    /// Compute MSE metric
    fn compute_mse_metric(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64> {
        let diff = (predictions - targets)?;
        let mse = diff.sqr()?.mean_all()?.to_scalar::<f64>()?;
        Ok(mse)
    }
    
    /// Compute top-k accuracy
    fn compute_topk_accuracy(&self, predictions: &Tensor, targets: &Tensor, k: usize) -> Result<f64> {
        let batch_size = predictions.dim(0)?;
        let num_classes = predictions.dim(1)?;
        
        if k > num_classes {
            return Ok(0.0);
        }
        
        // Get top-k predictions
        let topk_indices = self.get_topk_indices(predictions, k)?;
        let targets_expanded = targets.unsqueeze(1)?.expand(&[batch_size, k])?;
        
        // Check if target is in top-k
        let matches = topk_indices.eq(&targets_expanded)?;
        let matches_sum = matches.to_dtype(DType::F32)?.sum_keepdim(1)?;
        let has_match = matches_sum.ge(&Tensor::full(0.5, matches_sum.shape(), matches_sum.device())?)?;
        let accuracy = has_match.to_dtype(DType::F32)?.mean_all()?.to_scalar::<f64>()?;
        
        Ok(accuracy)
    }
    
    /// Get top-k indices (simplified implementation)
    fn get_topk_indices(&self, tensor: &Tensor, k: usize) -> Result<Tensor> {
        // Simplified version - in practice, you'd implement proper top-k
        let mut indices = Vec::new();
        for i in 0..k.min(tensor.dim(1)?) {
            indices.push(i as u32);
        }
        
        let batch_size = tensor.dim(0)?;
        let indices_tensor = Tensor::from_slice(
            &indices.repeat(batch_size),
            (batch_size, k),
            tensor.device()
        )?;
        
        Ok(indices_tensor)
    }
    
    /// Compute R-squared metric
    fn compute_r_squared(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64> {
        let mean_target = targets.mean_all()?;
        let ss_tot = (targets - &mean_target)?.sqr()?.sum_all()?.to_scalar::<f64>()?;
        let ss_res = (targets - predictions)?.sqr()?.sum_all()?.to_scalar::<f64>()?;
        
        if ss_tot > 0.0 {
            Ok(1.0 - ss_res / ss_tot)
        } else {
            Ok(0.0)
        }
    }
    
    /// Compute token accuracy for generation tasks
    fn compute_token_accuracy(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64> {
        let pred_tokens = predictions.argmax_keepdim(D::Minus1)?;
        let target_expanded = targets.unsqueeze(D::Minus1)?;
        let correct = pred_tokens.eq(&target_expanded)?.to_dtype(DType::F32)?;
        let accuracy = correct.mean_all()?.to_scalar::<f64>()?;
        Ok(accuracy)
    }
    
    /// Get cached metric
    pub fn get_cached_metric(&self, name: &str) -> Option<f64> {
        self.metric_cache.get(name).copied()
    }
    
    /// Clear metric cache
    pub fn clear_cache(&mut self) {
        self.metric_cache.clear();
    }
}

impl LossScaler {
    /// Create new loss scaler
    pub fn new(config: LossScalingConfig) -> Self {
        Self {
            current_scale: config.initial_scale,
            config,
            steps_since_update: 0,
            consecutive_good_steps: 0,
        }
    }
    
    /// Scale loss tensor
    pub fn scale_loss(&self, loss: &Tensor) -> Result<Tensor> {
        if self.config.enabled {
            (loss * self.current_scale).context("Failed to scale loss")
        } else {
            Ok(loss.clone())
        }
    }
    
    /// Unscale gradients
    pub fn unscale_gradients(&self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        
        for (name, grad) in gradients.iter_mut() {
            *grad = (grad.as_ref() / self.current_scale)
                .with_context(|| format!("Failed to unscale gradient for {}", name))?;
        }
        
        Ok(())
    }
    
    /// Update loss scale based on gradient overflow
    pub fn update_scale(&mut self, has_overflow: bool) {
        self.steps_since_update += 1;
        
        if has_overflow {
            // Decrease scale on overflow
            self.current_scale *= self.config.backoff_factor;
            self.consecutive_good_steps = 0;
            debug!("Loss scale decreased to {:.1} due to overflow", self.current_scale);
        } else {
            self.consecutive_good_steps += 1;
            
            // Increase scale after good steps
            if self.consecutive_good_steps >= self.config.growth_interval {
                self.current_scale *= self.config.growth_factor;
                self.consecutive_good_steps = 0;
                debug!("Loss scale increased to {:.1}", self.current_scale);
            }
        }
    }
    
    /// Get current scale
    pub fn get_scale(&self) -> f64 {
        self.current_scale
    }
    
    /// Check for gradient overflow
    pub fn check_gradient_overflow(&self, gradients: &HashMap<String, Tensor>) -> Result<bool> {
        for (_, grad) in gradients {
            // Check for NaN or Inf in gradients
            let ne_result = grad.ne(grad)?;
            let has_nan = ne_result.sum_all()?.to_scalar::<f64>()? > 0.0;
            if has_nan {
                return Ok(true);
            }
            let max_val = grad.abs()?.max(D::Minus1)?.max(D::Minus1)?.to_scalar::<f64>()?;
            if max_val.is_infinite() {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl MovingAverage {
    /// Create new moving average tracker
    pub fn new(window_size: usize) -> Self {
        Self {
            values: Vec::with_capacity(window_size),
            window_size,
            current_sum: 0.0,
            current_count: 0,
        }
    }
    
    /// Add new value
    pub fn update(&mut self, value: f64) {
        if self.values.len() < self.window_size {
            self.values.push(value);
            self.current_sum += value;
            self.current_count += 1;
        } else {
            let old_value = self.values[self.current_count % self.window_size];
            self.values[self.current_count % self.window_size] = value;
            self.current_sum = self.current_sum - old_value + value;
        }
        self.current_count += 1;
    }
    
    /// Get current average
    pub fn get_average(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.current_sum / self.values.len() as f64
        }
    }
    
    /// Get current variance
    pub fn get_variance(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        
        let mean = self.get_average();
        let variance = self.values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (self.values.len() - 1) as f64;
        
        variance
    }
    
    /// Reset the tracker
    pub fn reset(&mut self) {
        self.values.clear();
        self.current_sum = 0.0;
        self.current_count = 0;
    }
}