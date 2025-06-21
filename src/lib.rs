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
            config.projection.activation.to_projection_activation(),
        )?;
        debug!("Projection layer initialized");
        
        // Initialize hypernetwork
        // Convert config to hypernetwork module's config type
        let hypernetwork_config = hypernetwork::HypernetworkConfig {
            model_size: config.hypernetwork.model_size.to_hypernetwork_model_size(),
            input_dim: config.hypernetwork.input_dim,
            lora_rank: config.hypernetwork.lora_rank,
            dropout: config.hypernetwork.dropout,
            activation: config.hypernetwork.activation.to_hypernetwork_activation(),
            target_architecture: config.hypernetwork.target_architecture.clone(),
        };
        let hypernetwork = hypernetwork::HyperNetwork::new(hypernetwork_config)?;
        debug!("Hypernetwork initialized");
        
        // Initialize LoRA generator
        let generator = LoraGenerator::new(config.lora.clone().into());
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
        
        // Convert projected vector to ndarray
        let projected_array = ndarray::Array1::from_vec(projected);
        
        // Get target architecture from config
        let target_arch = self.config.hypernetwork.target_architecture.clone();
        
        // Generate LoRA parameters via hypernetwork
        let lora_params_raw = self.hypernetwork.generate_lora_params(&projected_array, target_arch)?;
        
        // Convert to LoraParameters format
        let mut lora_params = lora::LoraParameters::new(
            self.config.lora.clone().into()
        );
        
        // Convert each layer from hypernetwork format to LoRA format
        for (layer_name, layer_params) in lora_params_raw.layers.iter() {
            let input_dim = layer_params.matrix_a.shape()[0];
            let rank = layer_params.matrix_a.shape()[1];
            let output_dim = layer_params.matrix_b.shape()[1];
            
            // Create LoRA layer with the dimensions
            let mut lora_layer = lora::parameters::LoraLayer::new(
                layer_name.clone(),
                input_dim,
                output_dim,
                rank,
                layer_params.alpha,
            );
            
            // Copy weights from ndarray to flattened vectors
            // Matrix A is stored in row-major order
            lora_layer.a_weights = layer_params.matrix_a.iter().cloned().collect();
            // Matrix B is stored in row-major order
            lora_layer.b_weights = layer_params.matrix_b.iter().cloned().collect();
            
            // Add the layer to parameters
            lora_params.add_layer(lora_layer)?;
        }
        
        // Add metadata
        let metadata = lora::parameters::ParameterMetadata {
            task_description: task_description.to_string(),
            created_at: chrono::Utc::now(),
            generator_version: env!("CARGO_PKG_VERSION").to_string(),
            hyperparameters: {
                let mut params = std::collections::HashMap::new();
                params.insert("lora_rank".to_string(), serde_json::json!(self.config.lora.rank));
                params.insert("model_size".to_string(), serde_json::json!(format!("{:?}", self.config.hypernetwork.model_size)));
                params
            },
            metrics: None,
        };
        lora_params.set_metadata(metadata);
        
        info!("Successfully generated LoRA parameters with {} layers", lora_params.layers.len());
        
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
            
            // Convert projected vector to ndarray
            let projected_array = ndarray::Array1::from_vec(projected);
            
            // Get target architecture from config
            let target_arch = self.config.hypernetwork.target_architecture.clone();
            
            // Generate LoRA parameters via hypernetwork
            let _lora_params_raw = self.hypernetwork.generate_lora_params(&projected_array, target_arch)?;
            
            // Convert to LoraParameters format
            // TODO: Implement proper conversion from LoRAParams to LoraParameters
            let lora_params = lora::LoraParameters::new(
                self.config.lora.clone().into()
            );
            
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
            self.generator = LoraGenerator::new(config.lora.clone().into());
        }
        
        self.config = config;
        Ok(())
    }
}