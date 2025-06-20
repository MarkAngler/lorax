//! LoRA parameter generation module

use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// LoRA parameters for a single layer
#[derive(Debug, Clone)]
pub struct LoRALayerParams {
    /// Matrix A (down projection)
    pub matrix_a: Array2<f32>,
    /// Matrix B (up projection)
    pub matrix_b: Array2<f32>,
    /// Optional scaling factor
    pub alpha: f32,
}

/// Complete LoRA parameters for a model
#[derive(Debug, Clone)]
pub struct LoRAParams {
    /// Parameters organized by layer name
    pub layers: HashMap<String, LoRALayerParams>,
    /// Global LoRA rank
    pub rank: usize,
    /// Model architecture these params are for
    pub target_architecture: String,
}

impl LoRAParams {
    pub fn new(rank: usize, target_architecture: String) -> Self {
        Self {
            layers: HashMap::new(),
            rank,
            target_architecture,
        }
    }
    
    /// Add parameters for a specific layer
    pub fn add_layer(&mut self, name: String, params: LoRALayerParams) {
        self.layers.insert(name, params);
    }
    
    /// Get parameters for a specific layer
    pub fn get_layer(&self, name: &str) -> Option<&LoRALayerParams> {
        self.layers.get(name)
    }
    
    /// Calculate total number of parameters
    pub fn total_params(&self) -> usize {
        self.layers.values().map(|layer| {
            layer.matrix_a.len() + layer.matrix_b.len()
        }).sum()
    }
}

/// Layer configuration for LoRA generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub name: String,
    pub in_features: usize,
    pub out_features: usize,
    pub layer_type: LayerType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LayerType {
    Attention,
    MLP,
    Embedding,
    Output,
}

/// LoRA parameter generator
pub struct LoRAGenerator {
    rank: usize,
}

impl LoRAGenerator {
    pub fn new(rank: usize) -> Self {
        Self { rank }
    }
    
    /// Generate LoRA parameters from hypernetwork output
    pub fn generate(
        &self,
        hypernetwork_output: &Array1<f32>,
        layer_configs: &[LayerConfig],
    ) -> Result<LoRAParams> {
        let mut lora_params = LoRAParams::new(
            self.rank,
            "generated".to_string(),
        );
        
        let mut offset = 0;
        
        for config in layer_configs {
            let layer_params = self.generate_layer_params(
                hypernetwork_output,
                config,
                &mut offset,
            )?;
            
            lora_params.add_layer(config.name.clone(), layer_params);
        }
        
        Ok(lora_params)
    }
    
    /// Generate parameters for a single layer
    fn generate_layer_params(
        &self,
        output: &Array1<f32>,
        config: &LayerConfig,
        offset: &mut usize,
    ) -> Result<LoRALayerParams> {
        // Calculate sizes for A and B matrices
        let a_size = config.in_features * self.rank;
        let b_size = self.rank * config.out_features;
        let total_size = a_size + b_size + 1; // +1 for alpha
        
        // Check if we have enough values
        if *offset + total_size > output.len() {
            return Err(anyhow!(
                "Not enough values in hypernetwork output for layer {}",
                config.name
            ));
        }
        
        // Extract values for matrix A
        let a_values = output.slice(ndarray::s![*offset..*offset + a_size]);
        let matrix_a = Array2::from_shape_vec(
            (config.in_features, self.rank),
            a_values.to_vec(),
        )?;
        *offset += a_size;
        
        // Extract values for matrix B
        let b_values = output.slice(ndarray::s![*offset..*offset + b_size]);
        let matrix_b = Array2::from_shape_vec(
            (self.rank, config.out_features),
            b_values.to_vec(),
        )?;
        *offset += b_size;
        
        // Extract alpha scaling factor
        let alpha = output[*offset].abs() + 1.0; // Ensure positive
        *offset += 1;
        
        // Apply initialization scaling based on layer type
        let scale = match config.layer_type {
            LayerType::Attention => 0.01,
            LayerType::MLP => 0.02,
            LayerType::Embedding => 0.1,
            LayerType::Output => 0.05,
        };
        
        Ok(LoRALayerParams {
            matrix_a: matrix_a * scale,
            matrix_b: matrix_b * scale,
            alpha,
        })
    }
    
    /// Calculate required output size for given layer configurations
    pub fn calculate_output_size(&self, layer_configs: &[LayerConfig]) -> usize {
        layer_configs.iter().map(|config| {
            config.in_features * self.rank + // Matrix A
            self.rank * config.out_features + // Matrix B
            1 // Alpha
        }).sum()
    }
}

/// Apply LoRA to a weight matrix
pub fn apply_lora(
    weight: &Array2<f32>,
    lora_params: &LoRALayerParams,
) -> Array2<f32> {
    let delta = lora_params.matrix_a.dot(&lora_params.matrix_b);
    weight + &(delta * (lora_params.alpha / lora_params.matrix_a.ncols() as f32))
}