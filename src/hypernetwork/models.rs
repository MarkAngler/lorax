//! Hypernetwork model implementations (L, M, S)

use super::{ActivationType, HypernetworkConfig};
use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::thread_rng;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

/// Model size variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelSize {
    /// Small model: 2 layers, 256 hidden units
    Large,
    /// Medium model: 4 layers, 512 hidden units
    Medium,
    /// Large model: 8 layers, 1024 hidden units
    Small,
}

impl ModelSize {
    pub fn num_layers(&self) -> usize {
        match self {
            ModelSize::Large => 2,
            ModelSize::Medium => 4,
            ModelSize::Small => 8,
        }
    }

    pub fn hidden_dim(&self) -> usize {
        match self {
            ModelSize::Large => 256,
            ModelSize::Medium => 512,
            ModelSize::Small => 1024,
        }
    }
}

/// Linear layer
struct Linear {
    weight: Array2<f32>,
    bias: Array1<f32>,
}

impl Linear {
    fn new(in_features: usize, out_features: usize) -> Result<Self> {
        let mut rng = thread_rng();
        let std = (2.0 / in_features as f32).sqrt();
        let normal = Normal::new(0.0, std)?;
        
        let weight = Array2::from_shape_fn((out_features, in_features), |_| {
            normal.sample(&mut rng)
        });
        
        let bias = Array1::zeros(out_features);
        
        Ok(Self { weight, bias })
    }

    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        self.weight.dot(input) + &self.bias
    }
}

/// Layer normalization
struct LayerNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    eps: f32,
}

impl LayerNorm {
    fn new(normalized_shape: usize) -> Self {
        Self {
            gamma: Array1::ones(normalized_shape),
            beta: Array1::zeros(normalized_shape),
            eps: 1e-5,
        }
    }

    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let mean = input.mean().unwrap();
        let var = input.mapv(|x| (x - mean).powi(2)).mean().unwrap();
        let std = (var + self.eps).sqrt();
        
        let normalized = input.mapv(|x| (x - mean) / std);
        &normalized * &self.gamma + &self.beta
    }
}

/// Hypernetwork model
pub struct HyperNetworkModel {
    layers: Vec<Linear>,
    layer_norms: Vec<LayerNorm>,
    dropout_prob: f32,
    activation: ActivationType,
    output_dim: usize,
}

impl HyperNetworkModel {
    pub fn new(config: &HypernetworkConfig) -> Result<Self> {
        let num_layers = config.model_size.num_layers();
        let hidden_dim = config.model_size.hidden_dim();
        let input_dim = config.input_dim;
        
        // Calculate output dimension based on LoRA rank and typical layer sizes
        // This will be dynamically adjusted based on target architecture
        let output_dim = hidden_dim * 2;  // Base output dimension
        
        let mut layers = Vec::new();
        let mut layer_norms = Vec::new();
        
        // Input layer
        layers.push(Linear::new(input_dim, hidden_dim)?);        
        layer_norms.push(LayerNorm::new(hidden_dim));
        
        // Hidden layers
        for _ in 1..num_layers - 1 {
            layers.push(Linear::new(hidden_dim, hidden_dim)?);
            layer_norms.push(LayerNorm::new(hidden_dim));
        }
        
        // Output layer
        layers.push(Linear::new(hidden_dim, output_dim)?);
        // No layer norm after output
        
        Ok(Self {
            layers,
            layer_norms,
            dropout_prob: config.dropout,
            activation: config.activation,
            output_dim,
        })
    }

    pub fn forward(&self, input: &Array1<f32>) -> Result<Array1<f32>> {
        if input.len() == 0 {
            return Err(anyhow!("Empty input"));
        }
        
        let mut x = input.clone();
        
        // Apply all layers except the last
        for i in 0..self.layers.len() - 1 {
            x = self.layers[i].forward(&x);
            x = self.layer_norms[i].forward(&x);
            x = self.apply_activation(&x);
            x = self.apply_dropout(&x);
        }
        
        // Output layer (no activation or dropout)
        x = self.layers.last().unwrap().forward(&x);
        
        Ok(x)
    }

    fn apply_activation(&self, input: &Array1<f32>) -> Array1<f32> {
        match self.activation {
            ActivationType::ReLU => input.mapv(|x| x.max(0.0)),
            ActivationType::GELU => input.mapv(|x| {
                0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * x).tanh())
            }),
            ActivationType::SiLU => input.mapv(|x| x / (1.0 + (-x).exp())),
        }
    }

    fn apply_dropout(&self, input: &Array1<f32>) -> Array1<f32> {
        if self.dropout_prob == 0.0 {
            return input.clone();
        }
        
        let mut rng = thread_rng();
        let scale = 1.0 / (1.0 - self.dropout_prob);
        
        input.mapv(|x| {
            if rng.gen::<f32>() > self.dropout_prob {
                x * scale
            } else {
                0.0
            }
        })
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}