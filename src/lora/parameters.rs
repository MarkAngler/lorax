//! LoRA parameter structures and management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LoRA parameters for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraLayer {
    /// Layer name
    pub name: String,
    /// A matrix weights (flattened)
    pub a_weights: Vec<f32>,
    /// B matrix weights (flattened)
    pub b_weights: Vec<f32>,
    /// Matrix dimensions
    pub input_dim: usize,
    pub output_dim: usize,
    pub rank: usize,
    /// Scaling factor
    pub alpha: f32,
}

impl LoraLayer {
    /// Create new LoRA layer parameters
    pub fn new(
        name: String,
        input_dim: usize,
        output_dim: usize,
        rank: usize,
        alpha: f32,
    ) -> Self {
        let a_size = input_dim * rank;
        let b_size = rank * output_dim;
        
        Self {
            name,
            a_weights: vec![0.0; a_size],
            b_weights: vec![0.0; b_size],
            input_dim,
            output_dim,
            rank,
            alpha,
        }
    }
    
    /// Initialize with random weights
    pub fn with_random_weights(
        name: String,
        input_dim: usize,
        output_dim: usize,
        rank: usize,
        alpha: f32,
    ) -> Self {
        let mut layer = Self::new(name, input_dim, output_dim, rank, alpha);
        layer.randomize_weights();
        layer
    }
    
    /// Randomize weights using Xavier initialization
    pub fn randomize_weights(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Xavier initialization for A matrix
        let fan_in = self.input_dim as f32;
        let fan_out = self.rank as f32;
        let xavier_bound = (6.0 / (fan_in + fan_out)).sqrt();
        
        for weight in &mut self.a_weights {
            *weight = rng.gen_range(-xavier_bound..xavier_bound);
        }
        
        // Initialize B matrix to zeros (standard practice)
        for weight in &mut self.b_weights {
            *weight = 0.0;
        }
    }
    
    /// Get A matrix as 2D shape
    pub fn a_matrix_shape(&self) -> (usize, usize) {
        (self.input_dim, self.rank)
    }
    
    /// Get B matrix as 2D shape
    pub fn b_matrix_shape(&self) -> (usize, usize) {
        (self.rank, self.output_dim)
    }
    
    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.a_weights.len() + self.b_weights.len()
    }
    
    /// Validate parameter dimensions
    pub fn validate(&self) -> Result<(), crate::error::Error> {
        let expected_a_size = self.input_dim * self.rank;
        let expected_b_size = self.rank * self.output_dim;
        
        if self.a_weights.len() != expected_a_size {
            return Err(crate::error::Error::InvalidInput(format!(
                "Invalid A matrix size: expected {}, got {}",
                expected_a_size,
                self.a_weights.len()
            )));
        }
        
        if self.b_weights.len() != expected_b_size {
            return Err(crate::error::Error::InvalidInput(format!(
                "Invalid B matrix size: expected {}, got {}",
                expected_b_size,
                self.b_weights.len()
            )));
        }
        
        if self.rank == 0 {
            return Err(crate::error::Error::InvalidInput("LoRA rank cannot be zero".to_string()));
        }
        
        if self.alpha <= 0.0 {
            return Err(crate::error::Error::InvalidInput("LoRA alpha must be positive".to_string()));
        }
        
        Ok(())
    }
}

/// Complete set of LoRA parameters for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraParameters {
    /// LoRA layers by name
    pub layers: HashMap<String, LoraLayer>,
    /// Global configuration
    pub config: LoraParameterConfig,
    /// Metadata
    pub metadata: Option<ParameterMetadata>,
}

/// Configuration for LoRA parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraParameterConfig {
    /// Target model architecture
    pub target_architecture: String,
    /// Default rank (if not specified per layer)
    pub default_rank: usize,
    /// Default alpha (if not specified per layer)
    pub default_alpha: f32,
    /// Target modules/layers
    pub target_modules: Vec<String>,
    /// Whether to merge weights for inference
    pub merge_weights: bool,
}

impl Default for LoraParameterConfig {
    fn default() -> Self {
        Self {
            target_architecture: "llama".to_string(),
            default_rank: 16,
            default_alpha: 32.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            merge_weights: false,
        }
    }
}

impl From<crate::config::LoraConfig> for LoraParameterConfig {
    fn from(config: crate::config::LoraConfig) -> Self {
        Self {
            target_architecture: "llama".to_string(), // Default architecture
            default_rank: config.rank,
            default_alpha: config.alpha,
            target_modules: config.target_modules,
            merge_weights: false, // Default to false
        }
    }
}

/// Metadata for LoRA parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterMetadata {
    /// Task description used for generation
    pub task_description: String,
    /// Generation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Model version used for generation
    pub generator_version: String,
    /// Generation hyperparameters
    pub hyperparameters: HashMap<String, serde_json::Value>,
    /// Performance metrics (if available)
    pub metrics: Option<HashMap<String, f64>>,
}

impl LoraParameters {
    /// Create new LoRA parameters
    pub fn new(config: LoraParameterConfig) -> Self {
        Self {
            layers: HashMap::new(),
            config,
            metadata: None,
        }
    }
    
    /// Add layer parameters
    pub fn add_layer(&mut self, layer: LoraLayer) -> Result<(), crate::error::Error> {
        layer.validate()?;
        self.layers.insert(layer.name.clone(), layer);
        Ok(())
    }
    
    /// Get layer by name
    pub fn get_layer(&self, name: &str) -> Option<&LoraLayer> {
        self.layers.get(name)
    }
    
    /// Get mutable layer by name
    pub fn get_layer_mut(&mut self, name: &str) -> Option<&mut LoraLayer> {
        self.layers.get_mut(name)
    }
    
    /// Remove layer
    pub fn remove_layer(&mut self, name: &str) -> Option<LoraLayer> {
        self.layers.remove(name)
    }
    
    /// List all layer names
    pub fn layer_names(&self) -> Vec<&String> {
        self.layers.keys().collect()
    }
    
    /// Get total number of parameters
    pub fn total_parameters(&self) -> usize {
        self.layers.values().map(|layer| layer.num_parameters()).sum()
    }
    
    /// Validate all layers
    pub fn validate(&self) -> Result<(), crate::error::Error> {
        for layer in self.layers.values() {
            layer.validate()?;
        }
        Ok(())
    }
    
    /// Set metadata
    pub fn set_metadata(&mut self, metadata: ParameterMetadata) {
        self.metadata = Some(metadata);
    }
    
    /// Get metadata
    pub fn metadata(&self) -> Option<&ParameterMetadata> {
        self.metadata.as_ref()
    }
    
    /// Get configuration
    pub fn config(&self) -> &LoraParameterConfig {
        &self.config
    }
    
    /// Get summary statistics
    pub fn summary(&self) -> ParameterSummary {
        let total_params = self.total_parameters();
        let num_layers = self.layers.len();
        let avg_rank = if num_layers > 0 {
            self.layers.values().map(|l| l.rank).sum::<usize>() as f64 / num_layers as f64
        } else {
            0.0
        };
        let avg_alpha = if num_layers > 0 {
            self.layers.values().map(|l| l.alpha as f64).sum::<f64>() / num_layers as f64
        } else {
            0.0
        };
        
        ParameterSummary {
            total_parameters: total_params,
            num_layers,
            avg_rank,
            avg_alpha,
            target_architecture: self.config.target_architecture.clone(),
        }
    }
}

/// Summary statistics for LoRA parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSummary {
    pub total_parameters: usize,
    pub num_layers: usize,
    pub avg_rank: f64,
    pub avg_alpha: f64,
    pub target_architecture: String,
}