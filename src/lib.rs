//! LoRAX - Production-ready Text-to-LoRA (T2L) implementation
//!
//! This crate provides a complete implementation of Text-to-LoRA, enabling
//! dynamic model adaptation through natural language task descriptions.

#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

pub mod config;
pub mod encoder;
pub mod error;
pub mod hypernetwork;
pub mod lora;
pub mod models;
pub mod projection;
pub mod training;
pub mod utils;

// Re-exports
pub use config::{Config, HypernetworkConfig, ModelSize};
pub use encoder::{Encoder, TaskEncoder};
pub use error::{Error, Result};
pub use hypernetwork::{HyperNetwork, HyperNetworkModel};
pub use lora::{LoraConfig, LoraGenerator, LoraParameters};
pub use training::{DataLoader, ReconstructionDataset, SupervisedDataset};

use std::sync::Arc;
use tracing::{debug, info, instrument};

/// Main Text-to-LoRA system
pub struct TextToLora {
    /// Task encoder for converting text to embeddings
    encoder: Arc<dyn TaskEncoder>,
    /// Projection layer for dimension mapping
    projection: projection::ProjectionLayer,
    /// Hypernetwork for generating LoRA parameters
    hypernetwork: hypernetwork::HyperNetwork,
    /// LoRA generator for creating final parameters
    generator: LoraGenerator,
    /// System configuration
    config: Config,
}

impl TextToLora {
    /// Create a new Text-to-LoRA system with the given configuration
    #[instrument(skip(config))]
    pub async fn new(config: Config) -> Result<Self> {
        info!("Initializing Text-to-LoRA system");
        
        // Initialize encoder
        let encoder = encoder::create_encoder(&config.encoder)?;
        debug!("Task encoder initialized");
        
        // Initialize projection layer
        let projection = projection::ProjectionLayer::new(
            config.encoder.embedding_dim,
            config.hypernetwork.input_dim,
            config.projection.activation.clone(),
        )?;
        debug!("Projection layer initialized");
        
        // Initialize hypernetwork
        let hypernetwork = hypernetwork::HyperNetwork::new(config.hypernetwork.clone())?;
        debug!("Hypernetwork initialized");
        
        // Initialize LoRA generator
        let generator = LoraGenerator::new(config.lora.clone());
        debug!("LoRA generator initialized");
        
        Ok(Self {
            encoder,
            projection,
            hypernetwork,
            generator,
            config,
        })
    }
    
    /// Generate LoRA parameters from a task description
    #[instrument(skip(self))]
    pub async fn generate(&self, task_description: &str) -> Result<LoraParameters> {
        debug!("Generating LoRA parameters for task: {}", task_description);
        
        // Encode task description
        let task_embedding = self.encoder.encode(task_description).await?;
        debug!("Task encoded to {} dimensions", task_embedding.len());
        
        // Project to hypernetwork input dimension
        let projected = self.projection.forward(&task_embedding)?;
        debug!("Projected to {} dimensions", projected.len());
        
        // Generate raw parameters via hypernetwork
        let raw_params = self.hypernetwork.forward(&projected)?;
        debug!("Generated {} raw parameters", raw_params.len());
        
        // Generate LoRA parameters
        let lora_params = self.generator.generate(raw_params)?;
        info!("Successfully generated LoRA parameters");
        
        Ok(lora_params)
    }
    
    /// Generate LoRA parameters for multiple tasks (batched)
    #[instrument(skip(self, task_descriptions))]
    pub async fn generate_batch(
        &self,
        task_descriptions: &[String],
    ) -> Result<Vec<LoraParameters>> {
        info!("Generating LoRA parameters for {} tasks", task_descriptions.len());
        
        // Encode all tasks
        let embeddings = self.encoder.encode_batch(task_descriptions).await?;
        
        // Process through pipeline
        let mut results = Vec::with_capacity(task_descriptions.len());
        for embedding in embeddings {
            let projected = self.projection.forward(&embedding)?;
            let raw_params = self.hypernetwork.forward(&projected)?;
            let lora_params = self.generator.generate(raw_params)?;
            results.push(lora_params);
        }
        
        info!("Successfully generated {} LoRA parameter sets", results.len());
        Ok(results)
    }
    
    /// Get the current configuration
    pub fn config(&self) -> &Config {
        &self.config
    }
    
    /// Update configuration (requires restart for some changes)
    pub fn update_config(&mut self, config: Config) -> Result<()> {
        // Validate config changes
        config.validate()?;
        
        // Update components if needed
        if config.lora != self.config.lora {
            self.generator = LoraGenerator::new(config.lora.clone());
        }
        
        self.config = config;
        Ok(())
    }
}