//! Format conversion module
//!
//! This module provides functionality to convert T2L adapters to various formats
//! including PEFT, GGML, and HuggingFace, enabling integration with existing ML frameworks.

pub mod peft;
pub mod ggml;
pub mod huggingface;
pub mod openai;

use crate::lora::LoraParameters;
use anyhow::Result;
use std::path::Path;

/// Export format for T2L adapters
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum ExportFormat {
    /// PEFT (Parameter-Efficient Fine-Tuning) format for HuggingFace
    Peft,
    /// GGML format for llama.cpp
    Ggml,
    /// HuggingFace Transformers format
    Hf,
    /// OpenAI API-compatible format
    OpenAI,
}

/// Precision for exported weights
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum Precision {
    Fp16,
    Fp32,
    Int8,
}

/// Export engine for converting T2L adapters to different formats
pub struct ExportEngine {
    precision: Precision,
    merge_weights: bool,
}

impl ExportEngine {
    /// Create a new export engine
    pub fn new(precision: Precision, merge_weights: bool) -> Self {
        Self { precision, merge_weights }
    }

    /// Export adapter to specified format
    pub async fn export(
        &self,
        adapter: &LoraParameters,
        format: ExportFormat,
        target_model: Option<&str>,
        output_path: &Path,
    ) -> Result<()> {
        tracing::info!("Exporting adapter to {:?} format at: {}", format, output_path.display());
        
        match format {
            ExportFormat::Peft => {
                peft::export_to_peft(adapter, target_model, output_path, self.precision).await
            }
            ExportFormat::Ggml => {
                ggml::export_to_ggml(adapter, output_path, self.precision).await
            }
            ExportFormat::Hf => {
                huggingface::export_to_hf(adapter, target_model, output_path, self.precision).await
            }
            ExportFormat::OpenAI => {
                openai::export_to_openai(adapter, output_path).await
            }
        }
    }
}