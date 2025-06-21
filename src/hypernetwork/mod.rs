//! Hypernetwork module for T2L

mod models;
mod forward;
mod lora;
mod architectures;

#[cfg(test)]
mod tests;

pub use models::{ModelSize, HyperNetworkModel};
pub use forward::ForwardPass;
pub use lora::{LoRAParams, LoRAGenerator};
pub use architectures::{TargetArchitecture, ArchitectureHandler};

use anyhow::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Configuration for the hypernetwork
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypernetworkConfig {
    pub model_size: ModelSize,
    pub input_dim: usize,
    pub lora_rank: usize,
    pub dropout: f32,
    pub activation: ActivationType,
}

impl Default for HypernetworkConfig {
    fn default() -> Self {
        Self {
            model_size: ModelSize::Medium,
            input_dim: 768,  // Default text embedding dimension
            lora_rank: 16,
            dropout: 0.1,
            activation: ActivationType::ReLU,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    GELU,
    SiLU,
}

/// Main hypernetwork struct
pub struct HyperNetwork {
    config: HypernetworkConfig,
    model: HyperNetworkModel,
    lora_generator: LoRAGenerator,
    architecture_handler: ArchitectureHandler,
}

impl HyperNetwork {
    /// Create a new hypernetwork with the given configuration
    pub fn new(config: HypernetworkConfig) -> Result<Self> {
        let model = HyperNetworkModel::new(&config)?;
        let lora_generator = LoRAGenerator::new(config.lora_rank);
        let architecture_handler = ArchitectureHandler::new();

        Ok(Self {
            config,
            model,
            lora_generator,
            architecture_handler,
        })
    }

    /// Generate LoRA parameters for a given input and target architecture
    pub fn generate_lora_params(
        &self,
        input: &Array1<f32>,
        target_arch: TargetArchitecture,
    ) -> Result<LoRAParams> {
        // Forward pass through the hypernetwork
        let hidden = self.model.forward(input)?;
        
        // Get architecture-specific dimensions
        let layer_configs = self.architecture_handler.get_layer_configs(&target_arch)?;
        
        // Generate LoRA parameters
        let lora_params = self.lora_generator.generate(&hidden, &layer_configs)?;
        
        Ok(lora_params)
    }

    /// Get the current configuration
    pub fn config(&self) -> &HypernetworkConfig {
        &self.config
    }
}