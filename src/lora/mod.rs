//! LoRA (Low-Rank Adaptation) module for efficient model adaptation

use anyhow::Result;
use candle_core::{Tensor, Device};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod parameters;
pub mod generation;
pub mod config;

pub use parameters::{LoraParameters, LoraLayer};
pub use generation::LoraGenerator;
pub use config::{LoraConfig, BiasType};

/// LoRA weight matrices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraWeights {
    /// A matrix (input_dim x rank)
    pub a: Tensor,
    /// B matrix (rank x output_dim)  
    pub b: Tensor,
    /// Scaling factor
    pub alpha: f32,
    /// LoRA rank
    pub rank: usize,
}

impl LoraWeights {
    /// Create new LoRA weights
    pub fn new(input_dim: usize, output_dim: usize, rank: usize, alpha: f32, device: &Device) -> Result<Self> {
        // Initialize A matrix with Xavier normal distribution
        let a = Tensor::randn(0.0, (input_dim as f64).sqrt().recip(), (input_dim, rank), device)?;
        
        // Initialize B matrix with zeros
        let b = Tensor::zeros((rank, output_dim), candle_core::DType::F32, device)?;
        
        Ok(Self {
            a,
            b,
            alpha,
            rank,
        })
    }
    
    /// Apply LoRA to input tensor
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // input @ A @ B * alpha
        let temp = input.matmul(&self.a)?;
        let output = temp.matmul(&self.b)?;
        output.mul(&Tensor::new(self.alpha, input.device())?)
    }
    
    /// Get merged weight matrix (for inference optimization)
    pub fn get_merged_weights(&self) -> Result<Tensor> {
        // A @ B * alpha
        let merged = self.a.matmul(&self.b)?;
        merged.mul(&Tensor::new(self.alpha, self.a.device())?)
    }
    
    /// Update from raw parameters
    pub fn update_from_params(&mut self, a_weights: &[f32], b_weights: &[f32]) -> Result<()> {
        let device = self.a.device();
        
        // Reshape and update A matrix
        let a_shape = self.a.dims();
        self.a = Tensor::from_slice(a_weights, a_shape, device)?;
        
        // Reshape and update B matrix
        let b_shape = self.b.dims();
        self.b = Tensor::from_slice(b_weights, b_shape, device)?;
        
        Ok(())
    }
    
    /// Get number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        self.a.elem_count() + self.b.elem_count()
    }
}

/// LoRA adapter for a specific layer
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Layer name/identifier
    pub name: String,
    /// LoRA weights
    pub weights: LoraWeights,
    /// Whether adapter is enabled
    pub enabled: bool,
    /// Dropout rate
    pub dropout: f32,
}

impl LoraAdapter {
    /// Create new LoRA adapter
    pub fn new(
        name: String,
        input_dim: usize,
        output_dim: usize,
        rank: usize,
        alpha: f32,
        device: &Device,
    ) -> Result<Self> {
        let weights = LoraWeights::new(input_dim, output_dim, rank, alpha, device)?;
        
        Ok(Self {
            name,
            weights,
            enabled: true,
            dropout: 0.0,
        })
    }
    
    /// Create from pre-computed matrices
    pub fn from_matrices(a_matrix: Tensor, b_matrix: Tensor, alpha: f32, dropout: f32) -> Result<Self> {
        let rank = a_matrix.dim(1)?;
        let weights = LoraWeights {
            a: a_matrix,
            b: b_matrix,
            alpha,
            rank,
        };
        
        Ok(Self {
            name: String::new(),
            weights,
            enabled: true,
            dropout,
        })
    }
    
    /// Apply adapter to input if enabled
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.enabled {
            self.weights.forward(input)
        } else {
            Ok(input.clone())
        }
    }
    
    /// Enable/disable adapter
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Get adapter name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get weights reference
    pub fn weights(&self) -> &LoraWeights {
        &self.weights
    }
    
    /// Get LoRA parameters as tuple
    pub fn get_parameters(&self) -> (Tensor, Tensor) {
        (self.weights.a.clone(), self.weights.b.clone())
    }
}

/// Collection of LoRA adapters for a model
#[derive(Debug)]
pub struct LoraModel {
    /// Map of layer name to adapter
    adapters: HashMap<String, LoraAdapter>,
    /// Global scaling factor
    global_alpha: f32,
    /// Device for computations
    device: Device,
}

impl LoraModel {
    /// Create new LoRA model
    pub fn new(device: Device) -> Self {
        Self {
            adapters: HashMap::new(),
            global_alpha: 1.0,
            device,
        }
    }
    
    /// Add adapter for a layer
    pub fn add_adapter(
        &mut self,
        layer_name: String,
        input_dim: usize,
        output_dim: usize,
        rank: usize,
        alpha: f32,
    ) -> Result<()> {
        let adapter = LoraAdapter::new(layer_name.clone(), input_dim, output_dim, rank, alpha, &self.device)?;
        self.adapters.insert(layer_name, adapter);
        Ok(())
    }
    
    /// Get adapter by name
    pub fn get_adapter(&self, layer_name: &str) -> Option<&LoraAdapter> {
        self.adapters.get(layer_name)
    }
    
    /// Get mutable adapter by name
    pub fn get_adapter_mut(&mut self, layer_name: &str) -> Option<&mut LoraAdapter> {
        self.adapters.get_mut(layer_name)
    }
    
    /// Apply LoRA to specific layer
    pub fn apply_layer(&self, layer_name: &str, input: &Tensor) -> Result<Tensor> {
        if let Some(adapter) = self.get_adapter(layer_name) {
            let lora_output = adapter.forward(input)?;
            // Add to original input (residual connection)
            input.add(&lora_output)
        } else {
            Ok(input.clone())
        }
    }
    
    /// List all adapter names
    pub fn adapter_names(&self) -> Vec<&String> {
        self.adapters.keys().collect()
    }
    
    /// Enable/disable all adapters
    pub fn set_all_enabled(&mut self, enabled: bool) {
        for adapter in self.adapters.values_mut() {
            adapter.set_enabled(enabled);
        }
    }
    
    /// Get total number of parameters
    pub fn total_parameters(&self) -> usize {
        self.adapters.values().map(|a| a.weights.num_parameters()).sum()
    }
    
    /// Set global alpha scaling
    pub fn set_global_alpha(&mut self, alpha: f32) {
        self.global_alpha = alpha;
    }
    
    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
}