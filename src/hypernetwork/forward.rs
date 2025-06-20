//! Forward pass implementation for the hypernetwork

use anyhow::Result;
use ndarray::{Array1, Array2};

/// Trait for forward pass operations
pub trait ForwardPass {
    /// Execute forward pass with input tensor
    fn forward(&self, input: &Array1<f32>) -> Result<Array1<f32>>;
    
    /// Execute forward pass with batch input
    fn forward_batch(&self, inputs: &Array2<f32>) -> Result<Array2<f32>>;
}

/// Advanced forward pass operations
pub struct ForwardPassContext {
    /// Whether to collect intermediate activations
    pub collect_activations: bool,
    /// Whether to apply gradient checkpointing
    pub gradient_checkpointing: bool,
    /// Collected activations if enabled
    pub activations: Vec<Array1<f32>>,
}

impl Default for ForwardPassContext {
    fn default() -> Self {
        Self {
            collect_activations: false,
            gradient_checkpointing: false,
            activations: Vec::new(),
        }
    }
}

impl ForwardPassContext {
    /// Create a new context with activation collection enabled
    pub fn with_activation_collection() -> Self {
        Self {
            collect_activations: true,
            ..Default::default()
        }
    }
    
    /// Store an activation tensor
    pub fn store_activation(&mut self, activation: Array1<f32>) {
        if self.collect_activations {
            self.activations.push(activation);
        }
    }
    
    /// Clear stored activations
    pub fn clear_activations(&mut self) {
        self.activations.clear();
    }
}