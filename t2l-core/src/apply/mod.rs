//! LoRA adapter application module
//! 
//! This module provides functionality to apply LoRA adapters to base models,
//! enabling seamless integration of T2L-generated adapters with existing models.

pub mod loader;
pub mod merger;
pub mod saver;

use lorax::lora::LoraParameters;
use anyhow::Result;
use candle_core::Device;
use std::path::Path;

/// Output format for adapted models
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum OutputFormat {
    #[value(name = "safetensors")]
    SafeTensors,
    #[value(name = "pytorch")]
    PyTorch,
    #[value(name = "gguf")]
    GGUF,
}

/// Model applicator for applying LoRA adapters to base models
pub struct ModelApplicator {
    device: Device,
    merge_weights: bool,
}

impl ModelApplicator {
    /// Create a new model applicator
    pub fn new(device: Device, merge_weights: bool) -> Self {
        Self { device, merge_weights }
    }

    /// Apply LoRA adapter to a base model
    pub async fn apply(
        &self,
        base_model_path: &str,
        adapter: &LoraParameters,
        output_path: &Path,
        format: OutputFormat,
    ) -> Result<()> {
        tracing::info!("Loading base model from: {}", base_model_path);
        
        // 1. Load base model
        let mut base_model = loader::load_base_model(base_model_path, &self.device).await?;
        
        // 2. Apply LoRA adapter
        if self.merge_weights {
            tracing::info!("Merging LoRA weights into base model");
            merger::merge_lora_weights(&mut base_model, adapter)?;
        } else {
            tracing::info!("Attaching LoRA adapters to base model");
            merger::attach_lora_adapters(&mut base_model, adapter)?;
        }
        
        // 3. Save adapted model
        tracing::info!("Saving adapted model to: {}", output_path.display());
        saver::save_model(&base_model, output_path, format).await?;
        
        Ok(())
    }
}