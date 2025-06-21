//! LoRA parameter generation from hypernetwork outputs

use super::{LoraParameters, LoraLayer};
use super::parameters::{LoraParameterConfig, ParameterMetadata};
use crate::error::Result;
use std::collections::HashMap;

/// LoRA parameter generator
#[derive(Debug, Clone)]
pub struct LoraGenerator {
    config: LoraGeneratorConfig,
}

/// Configuration for LoRA generation
#[derive(Debug, Clone)]
pub struct LoraGeneratorConfig {
    /// Default rank for generated LoRAs
    pub default_rank: usize,
    /// Default alpha scaling factor
    pub default_alpha: f32,
    /// Layer specifications
    pub layer_specs: HashMap<String, LayerSpec>,
    /// Whether to add noise during generation
    pub add_noise: bool,
    /// Noise level (standard deviation)
    pub noise_level: f32,
}

/// Specification for a single layer
#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub input_dim: usize,
    pub output_dim: usize,
    pub rank: Option<usize>,  // Override default rank
    pub alpha: Option<f32>,   // Override default alpha
}

impl Default for LoraGeneratorConfig {
    fn default() -> Self {
        Self {
            default_rank: 16,
            default_alpha: 32.0,
            layer_specs: Self::default_llama_specs(),
            add_noise: false,
            noise_level: 0.01,
        }
    }
}

impl From<crate::config::LoraConfig> for LoraGeneratorConfig {
    fn from(config: crate::config::LoraConfig) -> Self {
        let mut layer_specs = HashMap::new();
        
        // Create layer specs based on target modules and model dimensions
        for module in &config.target_modules {
            let spec = match module.as_str() {
                "q_proj" | "k_proj" | "v_proj" | "o_proj" => LayerSpec {
                    input_dim: config.model_dim,
                    output_dim: config.model_dim,
                    rank: Some(config.rank),
                    alpha: Some(config.alpha),
                },
                "gate_proj" | "up_proj" => LayerSpec {
                    input_dim: config.model_dim,
                    output_dim: (config.model_dim as f64 * 2.7) as usize, // Approximate MLP expansion
                    rank: Some(config.rank),
                    alpha: Some(config.alpha),
                },
                "down_proj" => LayerSpec {
                    input_dim: (config.model_dim as f64 * 2.7) as usize,
                    output_dim: config.model_dim,
                    rank: Some(config.rank),
                    alpha: Some(config.alpha),
                },
                _ => LayerSpec {
                    input_dim: config.model_dim,
                    output_dim: config.model_dim,
                    rank: Some(config.rank),
                    alpha: Some(config.alpha),
                },
            };
            layer_specs.insert(module.clone(), spec);
        }
        
        Self {
            default_rank: config.rank,
            default_alpha: config.alpha,
            layer_specs,
            add_noise: false,
            noise_level: 0.01,
        }
    }
}

impl LoraGeneratorConfig {
    /// Default layer specifications for LLaMA architecture
    pub fn default_llama_specs() -> HashMap<String, LayerSpec> {
        let mut specs = HashMap::new();
        
        // Attention layers (assuming 4096 hidden size)
        specs.insert("q_proj".to_string(), LayerSpec {
            input_dim: 4096,
            output_dim: 4096,
            rank: None,
            alpha: None,
        });
        
        specs.insert("k_proj".to_string(), LayerSpec {
            input_dim: 4096,
            output_dim: 4096,
            rank: None,
            alpha: None,
        });
        
        specs.insert("v_proj".to_string(), LayerSpec {
            input_dim: 4096,
            output_dim: 4096,
            rank: None,
            alpha: None,
        });
        
        specs.insert("o_proj".to_string(), LayerSpec {
            input_dim: 4096,
            output_dim: 4096,
            rank: None,
            alpha: None,
        });
        
        // MLP layers
        specs.insert("gate_proj".to_string(), LayerSpec {
            input_dim: 4096,
            output_dim: 11008,
            rank: None,
            alpha: None,
        });
        
        specs.insert("up_proj".to_string(), LayerSpec {
            input_dim: 4096,
            output_dim: 11008,
            rank: None,
            alpha: None,
        });
        
        specs.insert("down_proj".to_string(), LayerSpec {
            input_dim: 11008,
            output_dim: 4096,
            rank: None,
            alpha: None,
        });
        
        specs
    }
    
    /// Create specifications for different model architectures
    pub fn for_architecture(architecture: &str) -> Self {
        let layer_specs = match architecture.to_lowercase().as_str() {
            "llama" | "llama2" | "llama3" => Self::default_llama_specs(),
            "mistral" => Self::mistral_specs(),
            "gemma" => Self::gemma_specs(),
            _ => Self::default_llama_specs(), // Fallback to LLaMA
        };
        
        Self {
            layer_specs,
            ..Default::default()
        }
    }
    
    /// Mistral architecture specifications
    pub fn mistral_specs() -> HashMap<String, LayerSpec> {
        let mut specs = HashMap::new();
        
        // Similar to LLaMA but different dimensions
        specs.insert("q_proj".to_string(), LayerSpec {
            input_dim: 4096,
            output_dim: 4096,
            rank: None,
            alpha: None,
        });
        
        specs.insert("k_proj".to_string(), LayerSpec {
            input_dim: 4096,
            output_dim: 1024, // Mistral uses different key dimensions
            rank: None,
            alpha: None,
        });
        
        specs.insert("v_proj".to_string(), LayerSpec {
            input_dim: 4096,
            output_dim: 1024,
            rank: None,
            alpha: None,
        });
        
        specs.insert("o_proj".to_string(), LayerSpec {
            input_dim: 4096,
            output_dim: 4096,
            rank: None,
            alpha: None,
        });
        
        specs.insert("gate_proj".to_string(), LayerSpec {
            input_dim: 4096,
            output_dim: 14336,
            rank: None,
            alpha: None,
        });
        
        specs.insert("up_proj".to_string(), LayerSpec {
            input_dim: 4096,
            output_dim: 14336,
            rank: None,
            alpha: None,
        });
        
        specs.insert("down_proj".to_string(), LayerSpec {
            input_dim: 14336,
            output_dim: 4096,
            rank: None,
            alpha: None,
        });
        
        specs
    }
    
    /// Gemma architecture specifications
    pub fn gemma_specs() -> HashMap<String, LayerSpec> {
        // Similar to LLaMA but with different dimensions
        Self::default_llama_specs() // Simplified for now
    }
}

impl LoraGenerator {
    /// Create new LoRA generator
    pub fn new(config: LoraGeneratorConfig) -> Self {
        Self { config }
    }
    
    /// Create with default configuration for architecture
    pub fn for_architecture(architecture: &str) -> Self {
        let config = LoraGeneratorConfig::for_architecture(architecture);
        Self::new(config)
    }
    
    /// Generate LoRA parameters from raw hypernetwork output
    pub fn generate(&self, raw_params: Vec<f32>) -> Result<LoraParameters> {
        let param_config = LoraParameterConfig {
            target_architecture: "auto".to_string(), // Will be set based on layer specs
            default_rank: self.config.default_rank,
            default_alpha: self.config.default_alpha,
            target_modules: self.config.layer_specs.keys().cloned().collect(),
            merge_weights: false,
        };
        
        let mut lora_params = LoraParameters::new(param_config);
        
        // Distribute raw parameters across layers
        let total_params_needed = self.calculate_total_params_needed();
        if raw_params.len() < total_params_needed {
            return Err(crate::error::Error::LoraGeneration(format!(
                "Insufficient parameters: need {}, got {}",
                total_params_needed,
                raw_params.len()
            )));
        }
        
        let mut param_offset = 0;
        
        for (layer_name, spec) in &self.config.layer_specs {
            let rank = spec.rank.unwrap_or(self.config.default_rank);
            let alpha = spec.alpha.unwrap_or(self.config.default_alpha);
            
            let a_size = spec.input_dim * rank;
            let b_size = rank * spec.output_dim;
            let total_layer_params = a_size + b_size;
            
            if param_offset + total_layer_params > raw_params.len() {
                return Err(crate::error::Error::LoraGeneration(
                    "Parameter allocation exceeds available parameters".to_string()
                ));
            }
            
            // Extract A and B matrix parameters
            let a_weights = raw_params[param_offset..param_offset + a_size].to_vec();
            param_offset += a_size;
            
            let b_weights = raw_params[param_offset..param_offset + b_size].to_vec();
            param_offset += b_size;
            
            // Apply noise if configured
            let (a_weights, b_weights) = if self.config.add_noise {
                (
                    self.add_noise_to_weights(a_weights),
                    self.add_noise_to_weights(b_weights)
                )
            } else {
                (a_weights, b_weights)
            };
            
            // Create LoRA layer
            let mut layer = LoraLayer::new(
                layer_name.clone(),
                spec.input_dim,
                spec.output_dim,
                rank,
                alpha,
            );
            
            layer.a_weights = a_weights;
            layer.b_weights = b_weights;
            
            lora_params.add_layer(layer)?;
        }
        
        Ok(lora_params)
    }
    
    /// Generate LoRA parameters for specific layers only
    pub fn generate_for_layers(
        &self,
        raw_params: Vec<f32>,
        target_layers: &[String],
    ) -> Result<LoraParameters> {
        // Filter layer specs to only include target layers
        let filtered_specs: HashMap<String, LayerSpec> = self.config.layer_specs
            .iter()
            .filter(|(name, _)| target_layers.contains(name))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        
        let temp_config = LoraGeneratorConfig {
            layer_specs: filtered_specs,
            ..self.config.clone()
        };
        
        let temp_generator = LoraGenerator::new(temp_config);
        temp_generator.generate(raw_params)
    }
    
    /// Calculate total parameters needed for all layers
    pub fn calculate_total_params_needed(&self) -> usize {
        self.config.layer_specs
            .iter()
            .map(|(_, spec)| {
                let rank = spec.rank.unwrap_or(self.config.default_rank);
                let a_size = spec.input_dim * rank;
                let b_size = rank * spec.output_dim;
                a_size + b_size
            })
            .sum()
    }
    
    /// Add Gaussian noise to weights
    fn add_noise_to_weights(&self, mut weights: Vec<f32>) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for weight in &mut weights {
            let noise: f32 = rng.gen_range(-1.0..1.0) * self.config.noise_level;
            *weight += noise;
        }
        
        weights
    }
    
    /// Get configuration
    pub fn config(&self) -> &LoraGeneratorConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: LoraGeneratorConfig) {
        self.config = config;
    }
    
    /// Add custom layer specification
    pub fn add_layer_spec(&mut self, name: String, spec: LayerSpec) {
        self.config.layer_specs.insert(name, spec);
    }
    
    /// Remove layer specification
    pub fn remove_layer_spec(&mut self, name: &str) {
        self.config.layer_specs.remove(name);
    }
    
    /// List supported layer names
    pub fn supported_layers(&self) -> Vec<&String> {
        self.config.layer_specs.keys().collect()
    }
}

/// Builder for creating LoRA generators with custom configurations
pub struct LoraGeneratorBuilder {
    config: LoraGeneratorConfig,
}

impl LoraGeneratorBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: LoraGeneratorConfig::default(),
        }
    }
    
    /// Set architecture
    pub fn architecture(mut self, arch: &str) -> Self {
        self.config = LoraGeneratorConfig::for_architecture(arch);
        self
    }
    
    /// Set default rank
    pub fn default_rank(mut self, rank: usize) -> Self {
        self.config.default_rank = rank;
        self
    }
    
    /// Set default alpha
    pub fn default_alpha(mut self, alpha: f32) -> Self {
        self.config.default_alpha = alpha;
        self
    }
    
    /// Enable noise injection
    pub fn with_noise(mut self, noise_level: f32) -> Self {
        self.config.add_noise = true;
        self.config.noise_level = noise_level;
        self
    }
    
    /// Add custom layer
    pub fn add_layer(mut self, name: String, spec: LayerSpec) -> Self {
        self.config.layer_specs.insert(name, spec);
        self
    }
    
    /// Build the generator
    pub fn build(self) -> LoraGenerator {
        LoraGenerator::new(self.config)
    }
}

impl Default for LoraGeneratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}