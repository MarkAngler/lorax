//! Projection layer for dimension mapping between encoder and hypernetwork

use crate::error::Result;
use candle_core::{Tensor, Device, DType};
use candle_nn::{Linear, VarBuilder, Module};
use serde::{Deserialize, Serialize};

/// Activation function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ActivationType {
    ReLU,
    GELU,
    SiLU,
    Tanh,
    Sigmoid,
    Identity,
}

/// Configuration for projection layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionConfig {
    /// Input dimension (from encoder)
    pub input_dim: usize,
    /// Output dimension (to hypernetwork)
    pub output_dim: usize,
    /// Activation function
    pub activation: ActivationType,
    /// Dropout probability
    pub dropout: f32,
    /// Whether to use layer normalization
    pub normalize: bool,
    /// Whether to use bias in linear layer
    pub use_bias: bool,
    /// Number of hidden layers (0 for single linear projection)
    pub num_hidden_layers: usize,
    /// Hidden dimension (if using hidden layers)
    pub hidden_dim: Option<usize>,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            input_dim: 384,      // Common sentence transformer dimension
            output_dim: 768,     // Common hypernetwork input dimension
            activation: ActivationType::GELU,
            dropout: 0.1,
            normalize: true,
            use_bias: true,
            num_hidden_layers: 1,
            hidden_dim: Some(512),
        }
    }
}

/// Projection layer for mapping encoder outputs to hypernetwork inputs
pub struct ProjectionLayer {
    config: ProjectionConfig,
    layers: Vec<Linear>,
    device: Device,
}

impl ProjectionLayer {
    /// Create new projection layer
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        activation: ActivationType,
    ) -> Result<Self> {
        let config = ProjectionConfig {
            input_dim,
            output_dim,
            activation,
            ..Default::default()
        };
        
        Self::with_config(config)
    }
    
    /// Create projection layer with custom configuration
    pub fn with_config(config: ProjectionConfig) -> Result<Self> {
        let device = Device::Cpu; // TODO: Make device configurable
        let mut layers = Vec::new();
        
        if config.num_hidden_layers == 0 {
            // Single linear projection
            let layer = Self::create_linear_layer(
                config.input_dim,
                config.output_dim,
                config.use_bias,
                &device,
            )?;
            layers.push(layer);
        } else {
            // Multi-layer projection
            let hidden_dim = config.hidden_dim.unwrap_or(
                (config.input_dim + config.output_dim) / 2
            );
            
            // First layer: input -> hidden
            let first_layer = Self::create_linear_layer(
                config.input_dim,
                hidden_dim,
                config.use_bias,
                &device,
            )?;
            layers.push(first_layer);
            
            // Hidden layers
            for _ in 1..config.num_hidden_layers {
                let hidden_layer = Self::create_linear_layer(
                    hidden_dim,
                    hidden_dim,
                    config.use_bias,
                    &device,
                )?;
                layers.push(hidden_layer);
            }
            
            // Final layer: hidden -> output
            let final_layer = Self::create_linear_layer(
                hidden_dim,
                config.output_dim,
                config.use_bias,
                &device,
            )?;
            layers.push(final_layer);
        }
        
        Ok(Self {
            config,
            layers,
            device,
        })
    }
    
    /// Create a linear layer with Xavier initialization
    fn create_linear_layer(
        input_dim: usize,
        output_dim: usize,
        use_bias: bool,
        device: &Device,
    ) -> Result<Linear> {
        // Xavier/Glorot initialization
        let fan_in = input_dim as f32;
        let fan_out = output_dim as f32;
        let bound = (6.0f32 / (fan_in + fan_out)).sqrt();
        
        // Create weight matrix
        let weight = Tensor::rand(-bound, bound, (output_dim, input_dim), device)?;
        
        // Create bias if needed
        let bias = if use_bias {
            Some(Tensor::zeros(output_dim, DType::F32, device)?)
        } else {
            None
        };
        
        Ok(Linear::new(weight, bias))
    }
    
    /// Forward pass through projection layer
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        // Convert input to tensor
        let input_tensor = Tensor::from_slice(
            input,
            (1, self.config.input_dim),
            &self.device,
        )?;
        
        let mut x = input_tensor;
        
        // Forward through all layers
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            
            // Apply activation (except possibly on last layer)
            if i < self.layers.len() - 1 || self.should_apply_final_activation() {
                x = self.apply_activation(&x)?;
            }
            
            // Apply dropout during training
            if self.config.dropout > 0.0 && i < self.layers.len() - 1 {
                x = self.apply_dropout(&x)?;
            }
        }
        
        // Apply layer normalization if configured
        if self.config.normalize {
            x = self.apply_layer_norm(&x)?;
        }
        
        // Convert back to Vec<f32>
        let output = x.flatten_all()?.to_vec1::<f32>()?;
        Ok(output)
    }
    
    /// Apply activation function
    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        match self.config.activation {
            ActivationType::ReLU => Ok(x.relu()?),
            ActivationType::GELU => {
                // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                let x_cubed = x.powf(3.0)?;
                let coeff = Tensor::new(0.044715f32, x.device())?;
                let inner = x.add(&x_cubed.mul(&coeff)?)?;
                let sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();
                let sqrt_2_pi_tensor = Tensor::new(sqrt_2_pi, x.device())?;
                let tanh_input = inner.mul(&sqrt_2_pi_tensor)?;
                let tanh_output = tanh_input.tanh()?;
                let one_plus_tanh = tanh_output.add(&Tensor::ones_like(&tanh_output)?)?;
                let half = Tensor::new(0.5f32, x.device())?;
                Ok(x.mul(&one_plus_tanh.mul(&half)?)?)
            }
            ActivationType::SiLU => {
                // SiLU: x * sigmoid(x)
                let ones = Tensor::ones_like(x)?;
                let neg_x = x.neg()?;
                let exp_neg_x = neg_x.exp()?;
                let sigmoid = ones.div(&(ones.add(&exp_neg_x)?))?;
                Ok(x.mul(&sigmoid)?)
            }
            ActivationType::Tanh => Ok(x.tanh()?),
            ActivationType::Sigmoid => {
                let ones = Tensor::ones_like(x)?;
                let neg_x = x.neg()?;
                let exp_neg_x = neg_x.exp()?;
                Ok(ones.div(&ones.add(&exp_neg_x)?)?)
            }
            ActivationType::Identity => Ok(x.clone()),
        }
    }
    
    /// Apply dropout (simplified for inference)
    fn apply_dropout(&self, x: &Tensor) -> Result<Tensor> {
        // For inference, we just return the input as-is
        // In training, this would apply dropout
        Ok(x.clone())
    }
    
    /// Apply layer normalization
    fn apply_layer_norm(&self, x: &Tensor) -> Result<Tensor> {
        // Simple layer normalization: (x - mean) / (std + eps)
        let eps = 1e-5;
        
        // For a [1, 512] tensor, we want to normalize across the feature dimension
        // mean_keepdim(1) would give [1, 1], but we need to broadcast properly
        let mean = x.mean(1)?;
        let mean_expanded = mean.unsqueeze(1)?; // Now [1, 1] shape
        let centered = x.broadcast_sub(&mean_expanded)?;
        
        let variance = centered.powf(2.0)?.mean(1)?;
        let variance_expanded = variance.unsqueeze(1)?;
        let eps_tensor = Tensor::new(eps as f32, x.device())?;
        let std = variance_expanded.broadcast_add(&eps_tensor)?.sqrt()?;
        
        Ok(centered.broadcast_div(&std)?)
    }
    
    /// Whether to apply activation on final layer
    fn should_apply_final_activation(&self) -> bool {
        // Usually we don't apply activation on the final projection
        // unless specifically configured
        false
    }
    
    /// Forward pass with batch input
    pub fn forward_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        
        for input in inputs {
            let output = self.forward(input)?;
            outputs.push(output);
        }
        
        Ok(outputs)
    }
    
    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.config.input_dim
    }
    
    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.config.output_dim
    }
    
    /// Get configuration
    pub fn config(&self) -> &ProjectionConfig {
        &self.config
    }
    
    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;
        
        if self.config.num_hidden_layers == 0 {
            // Single layer
            total += self.config.input_dim * self.config.output_dim;
            if self.config.use_bias {
                total += self.config.output_dim;
            }
        } else {
            // Multi-layer
            let hidden_dim = self.config.hidden_dim.unwrap_or(
                (self.config.input_dim + self.config.output_dim) / 2
            );
            
            // First layer
            total += self.config.input_dim * hidden_dim;
            if self.config.use_bias {
                total += hidden_dim;
            }
            
            // Hidden layers
            for _ in 1..self.config.num_hidden_layers {
                total += hidden_dim * hidden_dim;
                if self.config.use_bias {
                    total += hidden_dim;
                }
            }
            
            // Final layer
            total += hidden_dim * self.config.output_dim;
            if self.config.use_bias {
                total += self.config.output_dim;
            }
        }
        
        total
    }
}

/// Builder for creating projection layers
pub struct ProjectionLayerBuilder {
    config: ProjectionConfig,
}

impl ProjectionLayerBuilder {
    /// Create new builder
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let config = ProjectionConfig {
            input_dim,
            output_dim,
            ..Default::default()
        };
        
        Self { config }
    }
    
    /// Set activation function
    pub fn activation(mut self, activation: ActivationType) -> Self {
        self.config.activation = activation;
        self
    }
    
    /// Set dropout probability
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.config.dropout = dropout;
        self
    }
    
    /// Enable/disable layer normalization
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }
    
    /// Enable/disable bias
    pub fn use_bias(mut self, use_bias: bool) -> Self {
        self.config.use_bias = use_bias;
        self
    }
    
    /// Set number of hidden layers
    pub fn hidden_layers(mut self, num_layers: usize, hidden_dim: Option<usize>) -> Self {
        self.config.num_hidden_layers = num_layers;
        self.config.hidden_dim = hidden_dim;
        self
    }
    
    /// Build the projection layer
    pub fn build(self) -> Result<ProjectionLayer> {
        ProjectionLayer::with_config(self.config)
    }
}