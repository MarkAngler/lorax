//! Loss functions for T2L model training
//!
//! This module provides comprehensive loss function implementations for both
//! reconstruction and supervised training modes in the Text-to-LoRA system.
//!
//! # Key Components
//!
//! - **ReconstructionLoss**: MSE loss for LoRA parameter reconstruction training
//! - **SupervisedLoss**: Cross-entropy and generation losses for fine-tuning
//! - **Regularization**: L1/L2 penalties, dropout, weight decay utilities
//! - **Loss Composition**: Multi-task loss balancing and weighting
//! - **Utilities**: Numerical stability, metrics computation, loss scaling
//!
//! # Usage Examples
//!
//! ## Reconstruction Training
//! ```rust,ignore
//! use lorax::training::loss::{ReconstructionLoss, LossConfig};
//! use candle_core::Device;
//!
//! let config = LossConfig::reconstruction_default();
//! let loss_fn = ReconstructionLoss::new(config, &Device::Cpu)?;
//! 
//! // Compute loss for a batch
//! let loss_value = loss_fn.forward(&batch, &predicted_params)?;
//! let metrics = loss_fn.compute_metrics(&batch, &predicted_params)?;
//! ```
//!
//! ## Supervised Training
//! ```rust,ignore
//! use lorax::training::loss::{SupervisedLoss, LossConfig};
//!
//! let config = LossConfig::supervised_default();
//! let loss_fn = SupervisedLoss::new(config, &Device::Cpu)?;
//!
//! // Compute loss for classification/generation
//! let loss_value = loss_fn.forward(&batch, &logits)?;
//! ```
//!
//! ## Multi-task Training
//! ```rust,ignore
//! use lorax::training::loss::{CompositeLoss, LossConfig};
//!
//! let config = LossConfig::multitask_default();
//! let loss_fn = CompositeLoss::new(config)?;
//! loss_fn.add_task("reconstruction", reconstruction_loss, 0.7)?;
//! loss_fn.add_task("supervised", supervised_loss, 0.3)?;
//!
//! let total_loss = loss_fn.forward(&batch)?;
//! ```

pub mod reconstruction;
pub mod supervised;
pub mod regularization;
pub mod utils;
pub mod composition;

// Core loss trait and types
use crate::training::{Result, Error};
use crate::training::data::{ReconstructionBatch, SupervisedBatch};
use candle_core::{Tensor, Device};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main trait for all loss functions
pub trait LossFunction: Send + Sync {
    /// Compute the loss value
    fn forward(&self, batch: &dyn std::any::Any, predictions: &dyn std::any::Any) -> Result<Tensor>;
    
    /// Compute detailed loss metrics
    fn compute_metrics(&self, batch: &dyn std::any::Any, predictions: &dyn std::any::Any) -> Result<LossMetrics>;
    
    /// Get loss configuration
    fn config(&self) -> &LossConfig;
    
    /// Clone the loss function
    fn clone_box(&self) -> Box<dyn LossFunction>;
}

/// Configuration for loss functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    /// Type of loss function
    pub loss_type: LossType,
    
    /// Regularization configuration
    pub regularization: RegularizationConfig,
    
    /// Loss scaling configuration
    pub scaling: LossScalingConfig,
    
    /// Numerical stability settings
    pub stability: StabilityConfig,
    
    /// Task-specific weights (for multi-task learning)
    pub task_weights: HashMap<String, f64>,
    
    /// Loss reduction method
    pub reduction: ReductionMethod,
    
    /// Layer-specific configurations
    pub layer_configs: HashMap<String, LayerLossConfig>,
}

/// Type of loss function
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LossType {
    /// Reconstruction loss for LoRA parameters
    Reconstruction {
        /// Loss function for A/B matrices
        matrix_loss: MatrixLossType,
        /// Weight for consistency regularization
        consistency_weight: f64,
        /// Weight for sparsity regularization
        sparsity_weight: f64,
    },
    /// Supervised learning loss
    Supervised {
        /// Task type
        task_type: SupervisedTaskType,
        /// Label smoothing factor
        label_smoothing: f64,
        /// LoRA regularization weight
        lora_reg_weight: f64,
    },
    /// Combined multi-task loss
    MultiTask {
        /// Task configurations
        tasks: HashMap<String, LossType>,
        /// Task balancing strategy
        balancing: TaskBalancingStrategy,
    },
}

/// Matrix loss types for reconstruction
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MatrixLossType {
    /// Mean Squared Error
    Mse,
    /// Mean Absolute Error
    Mae,
    /// Huber loss (robust to outliers)
    Huber { delta: f64 },
    /// Cosine similarity loss
    Cosine,
}

/// Supervised task types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SupervisedTaskType {
    /// Classification task
    Classification { num_classes: usize },
    /// Text generation task
    Generation,
    /// Sequence labeling task
    SequenceLabeling { num_labels: usize },
    /// Masked language modeling
    MaskedLM,
}

/// Task balancing strategies for multi-task learning
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskBalancingStrategy {
    /// Fixed weights
    Fixed,
    /// Dynamic weight adjustment based on loss magnitudes
    Dynamic,
    /// Uncertainty-based weighting
    Uncertainty,
    /// Gradient normalization
    GradNorm,
}

/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// L1 regularization weight
    pub l1_weight: f64,
    /// L2 regularization weight  
    pub l2_weight: f64,
    /// Dropout probability
    pub dropout: f64,
    /// Weight decay factor
    pub weight_decay: f64,
    /// LoRA-specific regularization
    pub lora_regularization: LoraRegularizationConfig,
}

/// LoRA-specific regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraRegularizationConfig {
    /// Orthogonality constraint weight for A matrices
    pub orthogonality_weight: f64,
    /// Rank penalty weight
    pub rank_penalty_weight: f64,
    /// Spectral normalization for stability
    pub spectral_norm: bool,
    /// Maximum singular value constraint
    pub max_singular_value: Option<f64>,
}

/// Loss scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossScalingConfig {
    /// Enable loss scaling
    pub enabled: bool,
    /// Initial scale factor
    pub initial_scale: f64,
    /// Scale growth factor
    pub growth_factor: f64,
    /// Scale backoff factor
    pub backoff_factor: f64,
    /// Growth interval (steps)
    pub growth_interval: usize,
}

/// Numerical stability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityConfig {
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Gradient clipping threshold
    pub gradient_clip_threshold: f64,
    /// Maximum loss value (for clipping)
    pub max_loss_value: f64,
    /// Enable loss clamping
    pub clamp_losses: bool,
}

/// Loss reduction methods
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReductionMethod {
    /// Take mean of losses
    Mean,
    /// Sum all losses
    Sum,
    /// No reduction (return per-sample losses)
    None,
}

/// Layer-specific loss configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerLossConfig {
    /// Weight for this layer's loss
    pub weight: f64,
    /// Layer-specific regularization
    pub regularization: Option<RegularizationConfig>,
    /// Custom loss parameters
    pub custom_params: HashMap<String, f64>,
}

/// Comprehensive loss metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossMetrics {
    /// Total loss value
    pub total_loss: f64,
    /// Individual loss components
    pub components: HashMap<String, f64>,
    /// Per-layer losses (if applicable)
    pub layer_losses: HashMap<String, f64>,
    /// Regularization losses
    pub regularization_losses: HashMap<String, f64>,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
    /// Gradient norms (if computed)
    pub gradient_norms: Option<HashMap<String, f64>>,
}

impl LossConfig {
    /// Default configuration for reconstruction training
    pub fn reconstruction_default() -> Self {
        Self {
            loss_type: LossType::Reconstruction {
                matrix_loss: MatrixLossType::Mse,
                consistency_weight: 0.1,
                sparsity_weight: 0.01,
            },
            regularization: RegularizationConfig::default(),
            scaling: LossScalingConfig::default(),
            stability: StabilityConfig::default(),
            task_weights: HashMap::new(),
            reduction: ReductionMethod::Mean,
            layer_configs: HashMap::new(),
        }
    }
    
    /// Default configuration for supervised training
    pub fn supervised_default() -> Self {
        Self {
            loss_type: LossType::Supervised {
                task_type: SupervisedTaskType::Classification { num_classes: 2 },
                label_smoothing: 0.0,
                lora_reg_weight: 0.01,
            },
            regularization: RegularizationConfig::default(),
            scaling: LossScalingConfig::default(),
            stability: StabilityConfig::default(),
            task_weights: HashMap::new(),
            reduction: ReductionMethod::Mean,
            layer_configs: HashMap::new(),
        }
    }
    
    /// Default configuration for multi-task training
    pub fn multitask_default() -> Self {
        Self {
            loss_type: LossType::MultiTask {
                tasks: HashMap::new(),
                balancing: TaskBalancingStrategy::Fixed,
            },
            regularization: RegularizationConfig::default(),
            scaling: LossScalingConfig::default(),
            stability: StabilityConfig::default(),
            task_weights: HashMap::new(),
            reduction: ReductionMethod::Mean,
            layer_configs: HashMap::new(),
        }
    }
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            l1_weight: 0.0,
            l2_weight: 0.01,
            dropout: 0.1,
            weight_decay: 0.01,
            lora_regularization: LoraRegularizationConfig::default(),
        }
    }
}

impl Default for LoraRegularizationConfig {
    fn default() -> Self {
        Self {
            orthogonality_weight: 0.01,
            rank_penalty_weight: 0.001,
            spectral_norm: false,
            max_singular_value: None,
        }
    }
}

impl Default for LossScalingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            initial_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
        }
    }
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-8,
            gradient_clip_threshold: 1.0,
            max_loss_value: 1e6,
            clamp_losses: true,
        }
    }
}

impl LossMetrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self {
            total_loss: 0.0,
            components: HashMap::new(),
            layer_losses: HashMap::new(),
            regularization_losses: HashMap::new(),
            metrics: HashMap::new(),
            gradient_norms: None,
        }
    }
    
    /// Add a loss component
    pub fn add_component(&mut self, name: &str, value: f64) {
        self.components.insert(name.to_string(), value);
        self.total_loss += value;
    }
    
    /// Add a layer loss
    pub fn add_layer_loss(&mut self, layer: &str, value: f64) {
        self.layer_losses.insert(layer.to_string(), value);
    }
    
    /// Add a regularization loss
    pub fn add_regularization_loss(&mut self, name: &str, value: f64) {
        self.regularization_losses.insert(name.to_string(), value);
        self.total_loss += value;
    }
    
    /// Add a metric
    pub fn add_metric(&mut self, name: &str, value: f64) {
        self.metrics.insert(name.to_string(), value);
    }
    
    /// Set gradient norms
    pub fn set_gradient_norms(&mut self, norms: HashMap<String, f64>) {
        self.gradient_norms = Some(norms);
    }
    
    /// Get accuracy metric (if available)
    pub fn accuracy(&self) -> Option<f64> {
        self.metrics.get("accuracy").copied()
    }
    
    /// Get perplexity metric (if available)
    pub fn perplexity(&self) -> Option<f64> {
        self.metrics.get("perplexity").copied()
    }
    
    /// Get total regularization loss
    pub fn total_regularization_loss(&self) -> f64 {
        self.regularization_losses.values().sum()
    }
    
    /// Get main loss (total minus regularization)
    pub fn main_loss(&self) -> f64 {
        self.total_loss - self.total_regularization_loss()
    }
}

// Re-exports from submodules
pub use reconstruction::{ReconstructionLoss, ReconstructionLossConfig};
pub use supervised::{SupervisedLoss, SupervisedLossConfig, SupervisedPredictions};
pub use regularization::{RegularizationPenalties, LoraRegularizer};
pub use utils::{LossUtils, NumericalStability, MetricsComputer};
pub use composition::{CompositeLoss, TaskWeightManager};