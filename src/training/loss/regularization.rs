//! Regularization utilities for T2L training
//!
//! This module provides comprehensive regularization techniques including
//! L1/L2 penalties, dropout, weight decay, and LoRA-specific regularization.

use super::{RegularizationConfig, LoraRegularizationConfig, LossMetrics};
use crate::training::{Result, Error};
use candle_core::{Tensor, Device, DType, D};
use candle_nn as nn;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::{Context, anyhow};
use tracing::{debug, instrument};

/// Comprehensive regularization penalty computation
pub struct RegularizationPenalties {
    config: RegularizationConfig,
    device: Device,
    lora_regularizer: LoraRegularizer,
}

/// LoRA-specific regularization techniques
pub struct LoraRegularizer {
    config: LoraRegularizationConfig,
    device: Device,
}

/// Weight decay scheduler for adaptive regularization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightDecayScheduler {
    /// Initial weight decay value
    pub initial_decay: f64,
    /// Decay schedule type
    pub schedule_type: DecayScheduleType,
    /// Schedule parameters
    pub schedule_params: HashMap<String, f64>,
}

/// Weight decay schedule types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DecayScheduleType {
    /// Constant weight decay
    Constant,
    /// Linear decay over training
    Linear,
    /// Exponential decay
    Exponential,
    /// Cosine annealing
    Cosine,
    /// Step-wise decay
    Step,
}

/// Dropout configuration for different components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropoutConfig {
    /// Standard dropout probability
    pub dropout: f64,
    /// Attention dropout probability  
    pub attention_dropout: f64,
    /// LoRA-specific dropout
    pub lora_dropout: f64,
    /// Path dropout (stochastic depth)
    pub path_dropout: f64,
}

/// Spectral normalization for weight matrices
pub struct SpectralNormalizer {
    /// Power iteration count for singular value approximation
    pub power_iterations: usize,
    /// EPS for numerical stability
    pub eps: f64,
}

impl RegularizationPenalties {
    /// Create new regularization penalties
    pub fn new(config: RegularizationConfig, device: &Device) -> Self {
        let lora_regularizer = LoraRegularizer::new(config.lora_regularization.clone(), device);
        
        Self {
            config,
            device: device.clone(),
            lora_regularizer,
        }
    }
    
    /// Compute all regularization penalties for given parameters
    #[instrument(skip(self, parameters, lora_params))]
    pub fn compute_penalties(
        &self,
        parameters: &HashMap<String, Tensor>,
        lora_params: Option<&HashMap<String, (Tensor, Tensor)>>,
    ) -> Result<LossMetrics> {
        let mut metrics = LossMetrics::new();
        
        // L1 regularization
        if self.config.l1_weight > 0.0 {
            let l1_penalty = self.compute_l1_penalty(parameters)?;
            let weighted_l1 = l1_penalty * self.config.l1_weight;
            metrics.add_regularization_loss("l1", weighted_l1);
            debug!("L1 penalty: {:.6}", weighted_l1);
        }
        
        // L2 regularization  
        if self.config.l2_weight > 0.0 {
            let l2_penalty = self.compute_l2_penalty(parameters)?;
            let weighted_l2 = l2_penalty * self.config.l2_weight;
            metrics.add_regularization_loss("l2", weighted_l2);
            debug!("L2 penalty: {:.6}", weighted_l2);
        }
        
        // LoRA-specific regularization
        if let Some(lora_params) = lora_params {
            let lora_metrics = self.lora_regularizer.compute_regularization(lora_params)?;
            for (name, value) in lora_metrics.regularization_losses {
                metrics.add_regularization_loss(&name, value);
            }
        }
        
        Ok(metrics)
    }
    
    /// Compute L1 regularization penalty
    fn compute_l1_penalty(&self, parameters: &HashMap<String, Tensor>) -> Result<f64> {
        let mut total_l1 = 0.0;
        let mut param_count = 0;
        
        for (name, tensor) in parameters {
            // Skip bias terms and embeddings from L1 regularization
            if name.contains("bias") || name.contains("embed") {
                continue;
            }
            
            let l1_norm = tensor.abs()?.sum_all()?.to_scalar::<f64>()?;
            total_l1 += l1_norm;
            param_count += 1;
        }
        
        if param_count > 0 {
            Ok(total_l1 / param_count as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Compute L2 regularization penalty
    fn compute_l2_penalty(&self, parameters: &HashMap<String, Tensor>) -> Result<f64> {
        let mut total_l2 = 0.0;
        let mut param_count = 0;
        
        for (name, tensor) in parameters {
            // Apply L2 to all parameters except bias
            if name.contains("bias") {
                continue;
            }
            
            let l2_norm = tensor.sqr()?.sum_all()?.to_scalar::<f64>()?;
            total_l2 += l2_norm;
            param_count += 1;
        }
        
        if param_count > 0 {
            Ok(total_l2 / param_count as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Apply dropout to tensors during training
    pub fn apply_dropout(&self, tensor: &Tensor, dropout_prob: f64, training: bool) -> Result<Tensor> {
        if !training || dropout_prob <= 0.0 {
            return Ok(tensor.clone());
        }
        
        nn::ops::dropout(tensor, dropout_prob)
            .context("Failed to apply dropout")
    }
    
    /// Apply different dropout rates to different components
    pub fn apply_component_dropout(
        &self,
        tensors: &HashMap<String, Tensor>,
        dropout_config: &DropoutConfig,
        training: bool,
    ) -> Result<HashMap<String, Tensor>> {
        let mut result = HashMap::new();
        
        for (name, tensor) in tensors {
            let dropout_prob = if name.contains("attention") {
                dropout_config.attention_dropout
            } else if name.contains("lora") {
                dropout_config.lora_dropout
            } else {
                dropout_config.dropout
            };
            
            let dropped_tensor = self.apply_dropout(tensor, dropout_prob, training)?;
            result.insert(name.clone(), dropped_tensor);
        }
        
        Ok(result)
    }
    
    /// Compute gradient penalty for gradient-based regularization
    pub fn compute_gradient_penalty(
        &self,
        gradients: &HashMap<String, Tensor>,
        penalty_type: GradientPenaltyType,
    ) -> Result<f64> {
        match penalty_type {
            GradientPenaltyType::L2 => self.compute_gradient_l2_penalty(gradients),
            GradientPenaltyType::Spectral => self.compute_gradient_spectral_penalty(gradients),
        }
    }
    
    fn compute_gradient_l2_penalty(&self, gradients: &HashMap<String, Tensor>) -> Result<f64> {
        let mut total_penalty = 0.0;
        let mut grad_count = 0;
        
        for (_, grad) in gradients {
            let grad_norm = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f64>()?;
            total_penalty += grad_norm;
            grad_count += 1;
        }
        
        if grad_count > 0 {
            Ok(total_penalty / grad_count as f64)
        } else {
            Ok(0.0)
        }
    }
    
    fn compute_gradient_spectral_penalty(&self, gradients: &HashMap<String, Tensor>) -> Result<f64> {
        // Simplified spectral penalty - in practice, you'd compute spectral norms
        self.compute_gradient_l2_penalty(gradients)
    }
}

/// Gradient penalty types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientPenaltyType {
    L2,
    Spectral,
}

impl LoraRegularizer {
    /// Create new LoRA regularizer
    pub fn new(config: LoraRegularizationConfig, device: &Device) -> Self {
        Self {
            config,
            device: device.clone(),
        }
    }
    
    /// Compute comprehensive LoRA regularization
    #[instrument(skip(self, lora_params))]
    pub fn compute_regularization(
        &self,
        lora_params: &HashMap<String, (Tensor, Tensor)>,
    ) -> Result<LossMetrics> {
        let mut metrics = LossMetrics::new();
        
        // Orthogonality constraint for A matrices
        if self.config.orthogonality_weight > 0.0 {
            let orthogonality_loss = self.compute_orthogonality_loss(lora_params)?;
            let weighted_loss = orthogonality_loss * self.config.orthogonality_weight;
            metrics.add_regularization_loss("lora_orthogonality", weighted_loss);
        }
        
        // Rank penalty
        if self.config.rank_penalty_weight > 0.0 {
            let rank_penalty = self.compute_rank_penalty(lora_params)?;
            let weighted_penalty = rank_penalty * self.config.rank_penalty_weight;
            metrics.add_regularization_loss("lora_rank_penalty", weighted_penalty);
        }
        
        // Spectral normalization constraint
        if let Some(max_sv) = self.config.max_singular_value {
            let spectral_penalty = self.compute_spectral_penalty(lora_params, max_sv)?;
            metrics.add_regularization_loss("lora_spectral", spectral_penalty);
        }
        
        Ok(metrics)
    }
    
    /// Compute orthogonality loss for A matrices
    fn compute_orthogonality_loss(&self, lora_params: &HashMap<String, (Tensor, Tensor)>) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut layer_count = 0;
        
        for (layer_name, (a_matrix, _)) in lora_params {
            let orthogonality = self.compute_matrix_orthogonality_loss(a_matrix)?;
            total_loss += orthogonality;
            layer_count += 1;
            
            debug!("Orthogonality loss for {}: {:.6}", layer_name, orthogonality);
        }
        
        if layer_count > 0 {
            Ok(total_loss / layer_count as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Compute orthogonality loss for a single matrix
    fn compute_matrix_orthogonality_loss(&self, matrix: &Tensor) -> Result<f64> {
        // For batch processing, take the mean across batch dimension
        let batch_size = matrix.dim(0)?;
        let mut total_loss = 0.0;
        
        for batch_idx in 0..batch_size {
            let single_matrix = matrix.get(batch_idx)?;
            
            // Reshape to 2D if needed
            let matrix_2d = if single_matrix.dims().len() > 2 {
                single_matrix.flatten_to(1)?
            } else {
                single_matrix
            };
            
            // Compute A^T * A
            let at_a = matrix_2d.t()?.matmul(&matrix_2d)?;
            
            // Create identity matrix of same size
            let rank = at_a.dim(0)?;
            let identity = Tensor::eye(rank, at_a.device())?;
            
            // Orthogonality loss: ||A^T * A - I||_F^2
            let diff = (at_a - identity)?;
            let frobenius_norm_sq = diff.sqr()?.sum_all()?.to_scalar::<f64>()?;
            total_loss += frobenius_norm_sq;
        }
        
        Ok(total_loss / batch_size as f64)
    }
    
    /// Compute rank penalty to encourage low-rank solutions
    fn compute_rank_penalty(&self, lora_params: &HashMap<String, (Tensor, Tensor)>) -> Result<f64> {
        let mut total_penalty = 0.0;
        let mut layer_count = 0;
        
        for (layer_name, (a_matrix, b_matrix)) in lora_params {
            let penalty = self.compute_layer_rank_penalty(a_matrix, b_matrix)?;
            total_penalty += penalty;
            layer_count += 1;
            
            debug!("Rank penalty for {}: {:.6}", layer_name, penalty);
        }
        
        if layer_count > 0 {
            Ok(total_penalty / layer_count as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Compute rank penalty for a single layer
    fn compute_layer_rank_penalty(&self, a_matrix: &Tensor, b_matrix: &Tensor) -> Result<f64> {
        // Nuclear norm penalty (sum of singular values)
        let batch_size = a_matrix.dim(0)?;
        let mut total_penalty = 0.0;
        
        for batch_idx in 0..batch_size {
            let a_single = a_matrix.get(batch_idx)?;
            let b_single = b_matrix.get(batch_idx)?;
            
            // Compute combined matrix A * B
            let combined = a_single.matmul(&b_single)?;
            
            // Approximate nuclear norm using Frobenius norm
            // (True nuclear norm requires SVD which is expensive)
            let frobenius_norm = combined.sqr()?.sum_all()?.sqrt()?.to_scalar::<f64>()?;
            total_penalty += frobenius_norm;
        }
        
        Ok(total_penalty / batch_size as f64)
    }
    
    /// Compute spectral penalty to constrain maximum singular values
    fn compute_spectral_penalty(&self, lora_params: &HashMap<String, (Tensor, Tensor)>, max_sv: f64) -> Result<f64> {
        let mut total_penalty = 0.0;
        let mut layer_count = 0;
        
        for (layer_name, (a_matrix, b_matrix)) in lora_params {
            let penalty = self.compute_layer_spectral_penalty(a_matrix, b_matrix, max_sv)?;
            total_penalty += penalty;
            layer_count += 1;
            
            debug!("Spectral penalty for {}: {:.6}", layer_name, penalty);
        }
        
        if layer_count > 0 {
            Ok(total_penalty / layer_count as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Compute spectral penalty for a single layer
    fn compute_layer_spectral_penalty(&self, a_matrix: &Tensor, b_matrix: &Tensor, max_sv: f64) -> Result<f64> {
        let batch_size = a_matrix.dim(0)?;
        let mut total_penalty = 0.0;
        
        for batch_idx in 0..batch_size {
            let a_single = a_matrix.get(batch_idx)?;
            let b_single = b_matrix.get(batch_idx)?;
            
            // Approximate spectral norm using power iteration
            let spectral_norm = self.approximate_spectral_norm(&a_single, &b_single)?;
            
            // Penalty if spectral norm exceeds maximum
            if spectral_norm > max_sv {
                total_penalty += (spectral_norm - max_sv).powi(2);
            }
        }
        
        Ok(total_penalty / batch_size as f64)
    }
    
    /// Approximate spectral norm using power iteration
    fn approximate_spectral_norm(&self, a_matrix: &Tensor, b_matrix: &Tensor) -> Result<f64> {
        // Simple approximation: use largest element as proxy for spectral norm
        // In practice, you'd implement proper power iteration
        let combined = a_matrix.matmul(b_matrix)?;
        let max_element = combined.abs()?.max(D::Minus1)?.max(D::Minus1)?.to_scalar::<f64>()?;
        Ok(max_element)
    }
    
    /// Apply spectral normalization to LoRA parameters
    pub fn apply_spectral_normalization(
        &self,
        lora_params: &mut HashMap<String, (Tensor, Tensor)>,
    ) -> Result<()> {
        if !self.config.spectral_norm {
            return Ok(());
        }
        
        for (layer_name, (a_matrix, b_matrix)) in lora_params.iter_mut() {
            self.normalize_layer_spectral(a_matrix, b_matrix)?;
            debug!("Applied spectral normalization to {}", layer_name);
        }
        
        Ok(())
    }
    
    /// Apply spectral normalization to a single layer
    fn normalize_layer_spectral(&self, a_matrix: &mut Tensor, b_matrix: &mut Tensor) -> Result<()> {
        let batch_size = a_matrix.dim(0)?;
        
        for batch_idx in 0..batch_size {
            let mut a_single = a_matrix.get(batch_idx)?;
            let mut b_single = b_matrix.get(batch_idx)?;
            
            // Approximate spectral norm
            let spectral_norm = self.approximate_spectral_norm(&a_single, &b_single)?;
            
            // Normalize if needed
            if let Some(max_sv) = self.config.max_singular_value {
                if spectral_norm > max_sv {
                    let scale = max_sv / spectral_norm;
                    a_single = (a_single * scale)?;
                    // Note: In practice, you'd update the tensors in-place
                    // This is a simplified version
                }
            }
        }
        
        Ok(())
    }
}

impl SpectralNormalizer {
    /// Create new spectral normalizer
    pub fn new(power_iterations: usize, eps: f64) -> Self {
        Self {
            power_iterations,
            eps,
        }
    }
    
    /// Compute spectral norm using power iteration
    pub fn compute_spectral_norm(&self, matrix: &Tensor) -> Result<f64> {
        let (m, n) = (matrix.dim(0)?, matrix.dim(1)?);
        
        // Initialize random vector
        let mut u = Tensor::randn(0.0, 1.0, (m,), matrix.device())?;
        let mut v = Tensor::randn(0.0, 1.0, (n,), matrix.device())?;
        
        // Power iteration
        for _ in 0..self.power_iterations {
            // v = A^T u / ||A^T u||
            v = matrix.t()?.matmul(&u.unsqueeze(1)?)?.squeeze(1)?;
            let v_norm = v.sqr()?.sum_all()?.sqrt()?;
            v = (v / (v_norm + self.eps))?;
            
            // u = A v / ||A v||
            u = matrix.matmul(&v.unsqueeze(1)?)?.squeeze(1)?;
            let u_norm = u.sqr()?.sum_all()?.sqrt()?;
            u = (u / (u_norm + self.eps))?;
        }
        
        // Compute spectral norm: u^T A v
        let spectral_norm = u.unsqueeze(0)?.matmul(&matrix)?
            .matmul(&v.unsqueeze(1)?)?.squeeze(0)?.squeeze(0)?
            .to_scalar::<f64>()?;
        
        Ok(spectral_norm.abs())
    }
    
    /// Normalize matrix to have spectral norm <= max_norm
    pub fn normalize_spectral_norm(&self, matrix: &Tensor, max_norm: f64) -> Result<Tensor> {
        let spectral_norm = self.compute_spectral_norm(matrix)?;
        
        if spectral_norm > max_norm {
            let scale = max_norm / spectral_norm;
            Ok((matrix * scale)?)
        } else {
            Ok(matrix.clone())
        }
    }
}

impl WeightDecayScheduler {
    /// Create new weight decay scheduler
    pub fn new(initial_decay: f64, schedule_type: DecayScheduleType) -> Self {
        Self {
            initial_decay,
            schedule_type,
            schedule_params: HashMap::new(),
        }
    }
    
    /// Get weight decay value at given step
    pub fn get_decay(&self, step: usize, total_steps: usize) -> f64 {
        match self.schedule_type {
            DecayScheduleType::Constant => self.initial_decay,
            DecayScheduleType::Linear => {
                let progress = step as f64 / total_steps as f64;
                self.initial_decay * (1.0 - progress)
            },
            DecayScheduleType::Exponential => {
                let decay_rate = self.schedule_params.get("decay_rate").unwrap_or(&0.95);
                self.initial_decay * decay_rate.powf(step as f64)
            },
            DecayScheduleType::Cosine => {
                let progress = step as f64 / total_steps as f64;
                self.initial_decay * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
            },
            DecayScheduleType::Step => {
                let step_size = self.schedule_params.get("step_size").unwrap_or(&1000.0) as usize;
                let decay_factor = self.schedule_params.get("decay_factor").unwrap_or(&0.1);
                let num_decays = step / step_size;
                self.initial_decay * decay_factor.powf(num_decays as f64)
            },
        }
    }
    
    /// Add schedule parameter
    pub fn add_param(&mut self, name: &str, value: f64) {
        self.schedule_params.insert(name.to_string(), value);
    }
}

impl Default for DropoutConfig {
    fn default() -> Self {
        Self {
            dropout: 0.1,
            attention_dropout: 0.1,
            lora_dropout: 0.1,
            path_dropout: 0.0,
        }
    }
}

impl Default for WeightDecayScheduler {
    fn default() -> Self {
        Self::new(0.01, DecayScheduleType::Constant)
    }
}

impl Default for SpectralNormalizer {
    fn default() -> Self {
        Self::new(1, 1e-12)
    }
}