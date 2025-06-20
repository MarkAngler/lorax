use crate::cli::{config::Config, error::CliResult, progress::ProgressReporter};
use crate::export::{ExportEngine, ExportFormat, Precision};
use crate::lora::LoraParameters;
use anyhow::{Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};
use tracing::{info, warn};

#[derive(Args, Debug)]
pub struct ExportCommand {
    /// Path to T2L adapter file
    #[arg(
        short = 'a',
        long = "adapter", 
        help = "Path to T2L adapter file (e.g., french_adapter.safetensors)"
    )]
    pub adapter: PathBuf,

    /// Output format
    #[arg(
        short = 'f',
        long = "format",
        help = "Output format (peft, ggml, hf, openai)"
    )]
    pub format: ExportFormat,

    /// Target model for format compatibility
    #[arg(
        long = "target-model",
        help = "Target model ID for format compatibility (e.g., meta-llama/Llama-2-7b-hf)"
    )]
    pub target_model: Option<String>,

    /// Output directory
    #[arg(
        short = 'o',
        long = "output",
        help = "Output directory for exported adapter"
    )]
    pub output: PathBuf,

    /// Precision for exported weights
    #[arg(
        long = "precision",
        default_value = "fp16",
        help = "Weight precision (fp16, fp32, int8)"
    )]
    pub precision: Precision,

    /// Whether to merge weights into single tensors
    #[arg(
        long = "merge-weights",
        help = "Merge LoRA weights into base tensors"
    )]
    pub merge_weights: bool,

    /// Force overwrite existing output directory
    #[arg(
        long = "force",
        help = "Overwrite existing output directory"
    )]
    pub force: bool,

    /// Skip validation of adapter file
    #[arg(
        long = "skip-validation",
        help = "Skip adapter file validation"
    )]
    pub skip_validation: bool,

    /// Include metadata in the output
    #[arg(
        long = "include-metadata",
        help = "Include export metadata in output"
    )]
    pub include_metadata: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExportMetadata {
    pub adapter_path: PathBuf,
    pub format: String,
    pub target_model: Option<String>,
    pub precision: String,
    pub merge_weights: bool,
    pub export_time_ms: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub t2l_version: String,
}

pub async fn execute(cmd: ExportCommand, config: Config) -> CliResult<()> {
    info!("Starting T2L adapter export");

    // Validate command arguments
    validate_command(&cmd)?;

    let start_time = std::time::Instant::now();
    
    // Initialize progress reporter
    let progress = ProgressReporter::new("Exporting adapter")?;
    
    progress.set_message("Validating inputs...");
    
    // Load and validate adapter
    let adapter = load_adapter(&cmd.adapter, &cmd)?;
    
    progress.advance("Initializing export engine...");
    
    // Create export engine
    let export_engine = ExportEngine::new(cmd.precision, cmd.merge_weights);
    
    progress.advance(&format!("Exporting to {} format...", format_name(&cmd.format)));
    
    // Perform export
    export_engine
        .export(
            &adapter,
            cmd.format,
            cmd.target_model.as_deref(),
            &cmd.output,
        )
        .await
        .context("Failed to export adapter")?;
    
    progress.advance("Saving export metadata...");
    
    // Save metadata if requested
    if cmd.include_metadata {
        save_export_metadata(&cmd, start_time.elapsed().as_millis() as u64)?;
    }
    
    progress.finish("Adapter exported successfully!");
    
    info!("Exported adapter: {}", cmd.adapter.display());
    info!("Format: {:?}", cmd.format);
    info!("Output directory: {}", cmd.output.display());
    if let Some(target) = &cmd.target_model {
        info!("Target model: {}", target);
    }
    info!("Precision: {:?}", cmd.precision);
    info!("Weights merged: {}", cmd.merge_weights);
    
    Ok(())
}

fn validate_command(cmd: &ExportCommand) -> CliResult<()> {
    // Check if adapter file exists
    if !cmd.adapter.exists() {
        return Err(crate::cli::error::CliError::FileNotFound(
            cmd.adapter.to_string_lossy().to_string()
        ));
    }

    // Check output directory doesn't exist or force is specified
    if cmd.output.exists() && !cmd.force {
        return Err(crate::cli::error::CliError::FileExists(cmd.output.clone()));
    }

    // Validate adapter file extension
    let adapter_ext = cmd.adapter
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    
    let supported_extensions = ["safetensors", "bin", "pt", "pth", "ckpt"];
    if !supported_extensions.contains(&adapter_ext) {
        warn!(
            "Adapter file extension '{}' may not be supported. Supported: {}",
            adapter_ext,
            supported_extensions.join(", ")
        );
    }

    // Format-specific validation
    match cmd.format {
        ExportFormat::Peft | ExportFormat::Hf => {
            if cmd.target_model.is_none() {
                warn!("Target model not specified for {} format. This may cause compatibility issues.", format_name(&cmd.format));
            }
        }
        ExportFormat::Ggml => {
            if cmd.merge_weights {
                warn!("GGML format with merged weights may not be supported by all llama.cpp versions");
            }
        }
        ExportFormat::OpenAI => {
            if cmd.target_model.is_none() {
                warn!("Target model not specified for OpenAI format. Using generic configuration.");
            }
        }
    }

    // Validate precision for specific formats
    match (cmd.format, cmd.precision) {
        (ExportFormat::Ggml, Precision::Fp32) => {
            warn!("GGML format typically uses fp16 or int8 for better performance");
        }
        (ExportFormat::OpenAI, Precision::Int8) => {
            warn!("OpenAI format may not support int8 precision");
        }
        _ => {}
    }

    // Validate target model format if provided
    if let Some(target) = &cmd.target_model {
        if target.trim().is_empty() {
            return Err(crate::cli::error::CliError::InvalidArgument(
                "Target model cannot be empty".to_string()
            ));
        }
        
        // Check if it looks like a HuggingFace model ID
        if target.contains('/') && target.split('/').count() == 2 {
            let parts: Vec<&str> = target.split('/').collect();
            if parts[0].is_empty() || parts[1].is_empty() {
                return Err(crate::cli::error::CliError::InvalidArgument(
                    format!("Invalid HuggingFace model ID format: {}", target)
                ));
            }
        }
    }

    Ok(())
}

fn load_adapter(adapter_path: &PathBuf, cmd: &ExportCommand) -> CliResult<LoraParameters> {
    info!("Loading T2L adapter from: {}", adapter_path.display());
    
    // TODO: Implement proper adapter loading based on file format
    // This would involve:
    // 1. Detecting file format (safetensors, pickle, etc.)
    // 2. Loading the adapter parameters
    // 3. Validating adapter structure
    // 4. Checking compatibility with target format
    
    if !cmd.skip_validation {
        info!("Validating adapter structure...");
        // TODO: Implement adapter validation
        // - Check if all required fields are present
        // - Validate tensor shapes and data types
        // - Verify LoRA rank consistency
        // - Check layer name format compatibility
    }
    
    // Placeholder implementation
    warn!("Adapter loading not yet fully implemented - using placeholder");
    // TODO: Implement proper adapter loading
    // For now, we'll return an error since we can't create a real placeholder
    Err(crate::cli::error::CliError::FeatureNotAvailable(
        "Adapter loading is not yet implemented".to_string()
    ))
}

fn save_export_metadata(
    cmd: &ExportCommand,
    export_time_ms: u64,
) -> CliResult<()> {
    let metadata = ExportMetadata {
        adapter_path: cmd.adapter.clone(),
        format: format_name(&cmd.format).to_string(),
        target_model: cmd.target_model.clone(),
        precision: format!("{:?}", cmd.precision).to_lowercase(),
        merge_weights: cmd.merge_weights,
        export_time_ms,
        timestamp: chrono::Utc::now(),
        t2l_version: crate::VERSION.to_string(),
    };

    // Save metadata in the output directory
    let metadata_path = cmd.output.join("export_metadata.json");
    
    // Create output directory if it doesn't exist
    if let Some(parent) = metadata_path.parent() {
        fs::create_dir_all(parent)?;
    }
    
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .context("Failed to serialize export metadata")?;
    
    fs::write(&metadata_path, metadata_json)
        .context("Failed to write export metadata file")?;
    
    info!("Export metadata saved to: {}", metadata_path.display());
    Ok(())
}

fn format_name(format: &ExportFormat) -> &'static str {
    match format {
        ExportFormat::Peft => "PEFT",
        ExportFormat::Ggml => "GGML",
        ExportFormat::Hf => "HuggingFace",
        ExportFormat::OpenAI => "OpenAI",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_validate_command_missing_adapter() {
        let cmd = ExportCommand {
            adapter: PathBuf::from("nonexistent.safetensors"),
            format: ExportFormat::Peft,
            target_model: Some("meta-llama/Llama-2-7b-hf".to_string()),
            output: PathBuf::from("./output"),
            precision: Precision::Fp16,
            merge_weights: false,
            force: false,
            skip_validation: false,
            include_metadata: false,
        };
        
        assert!(validate_command(&cmd).is_err());
    }
    
    #[test]
    fn test_validate_command_invalid_target_model() {
        let cmd = ExportCommand {
            adapter: PathBuf::from("test.safetensors"),
            format: ExportFormat::Peft,
            target_model: Some("/invalid".to_string()),
            output: PathBuf::from("./output"),
            precision: Precision::Fp16,
            merge_weights: false,
            force: false,
            skip_validation: false,
            include_metadata: false,
        };
        
        // This should fail because the target model format is invalid
        assert!(validate_command(&cmd).is_err());
    }
    
    #[test]
    fn test_format_name() {
        assert_eq!(format_name(&ExportFormat::Peft), "PEFT");
        assert_eq!(format_name(&ExportFormat::Ggml), "GGML");
        assert_eq!(format_name(&ExportFormat::Hf), "HuggingFace");
        assert_eq!(format_name(&ExportFormat::OpenAI), "OpenAI");
    }
}