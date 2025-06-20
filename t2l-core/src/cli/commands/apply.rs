use crate::cli::{config::Config, error::CliResult, progress::ProgressReporter};
use crate::apply::{ModelApplicator, OutputFormat};
use crate::lora::LoraParameters;
use anyhow::{Context, Result};
use clap::Args;
use candle_core::Device;
use chrono;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};
use tracing::{info, warn};

#[derive(Args, Debug)]
pub struct ApplyCommand {
    /// Path to the base model to apply the adapter to
    #[arg(
        short = 'b', 
        long = "base-model", 
        help = "Path to the base model (e.g., 'llama-7b' or path to model files)"
    )]
    pub base_model: String,

    /// Path to the LoRA adapter file to apply
    #[arg(
        short = 'a', 
        long = "adapter", 
        help = "Path to the LoRA adapter file (e.g., french_adapter.safetensors)"
    )]
    pub adapter: PathBuf,

    /// Output path for the adapted model
    #[arg(
        short = 'o', 
        long = "output", 
        help = "Output path for the adapted model"
    )]
    pub output: PathBuf,

    /// Output format for the adapted model
    #[arg(
        short = 'f', 
        long = "format", 
        default_value = "safetensors", 
        help = "Output format (safetensors, pytorch, gguf)"
    )]
    pub format: OutputFormat,

    /// Merge LoRA weights into base model instead of keeping them separate
    #[arg(
        long = "merge-weights", 
        help = "Merge LoRA weights into base model weights (creates a single merged model)"
    )]
    pub merge_weights: bool,

    /// Device to use for computation
    #[arg(
        short = 'd', 
        long = "device", 
        default_value = "auto", 
        help = "Device to use (cuda, cpu, auto)"
    )]
    pub device: DeviceType,

    /// Force overwrite existing output file
    #[arg(
        long = "force", 
        help = "Overwrite existing output file"
    )]
    pub force: bool,

    /// Skip model validation checks
    #[arg(
        long = "skip-validation", 
        help = "Skip compatibility validation between base model and adapter"
    )]
    pub skip_validation: bool,

    /// Adapter scaling factor (alpha value)
    #[arg(
        long = "alpha", 
        help = "Override adapter alpha scaling factor"
    )]
    pub alpha: Option<f32>,

    /// Use half precision (fp16) for reduced memory usage
    #[arg(
        long = "half-precision", 
        help = "Use half precision (fp16) for reduced memory usage"
    )]
    pub half_precision: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, clap::ValueEnum)]
pub enum DeviceType {
    #[value(name = "cuda")]
    Cuda,
    #[value(name = "cpu")]
    Cpu,
    #[value(name = "auto")]
    Auto,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApplicationMetadata {
    pub base_model: String,
    pub adapter_path: PathBuf,
    pub output_format: String,
    pub merge_weights: bool,
    pub device_used: String,
    pub alpha_override: Option<f32>,
    pub application_time_ms: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub t2l_version: String,
}

pub async fn execute(cmd: ApplyCommand, config: Config) -> CliResult<()> {
    info!("Starting LoRA adapter application with T2L");

    // Validate command arguments
    validate_command(&cmd)?;

    let start_time = std::time::Instant::now();
    
    // Initialize progress reporter
    let progress = ProgressReporter::new("Applying LoRA adapter")?;
    
    progress.set_message("Validating inputs...");
    
    // Load and validate adapter
    let adapter = load_adapter(&cmd.adapter, &cmd)?;
    
    progress.advance("Initializing device...");
    
    // Initialize device
    let device = initialize_device(&cmd.device)?;
    info!("Using device: {:?}", device);
    
    progress.advance("Setting up model applicator...");
    
    // Create model applicator
    let applicator = ModelApplicator::new(device, cmd.merge_weights);
    
    progress.advance("Applying LoRA adapter to base model...");
    
    // Apply adapter to base model
    applicator
        .apply(&cmd.base_model, &adapter, &cmd.output, cmd.format)
        .await
        .context("Failed to apply LoRA adapter to base model")?;
    
    progress.advance("Saving application metadata...");
    
    // Save metadata
    save_application_metadata(&cmd, start_time.elapsed().as_millis() as u64, &device)?;
    
    progress.finish("LoRA adapter applied successfully!");
    
    info!("Applied LoRA adapter: {}", cmd.adapter.display());
    info!("Base model: {}", cmd.base_model);
    info!("Output saved to: {}", cmd.output.display());
    info!("Format: {:?}", cmd.format);
    info!("Weights merged: {}", cmd.merge_weights);
    
    Ok(())
}

fn validate_command(cmd: &ApplyCommand) -> CliResult<()> {
    // Check if adapter file exists
    if !cmd.adapter.exists() {
        return Err(crate::cli::error::CliError::FileNotFound(
            cmd.adapter.to_string_lossy().to_string()
        ));
    }

    // Check output path doesn't exist or force is specified
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

    // Validate alpha value if provided
    if let Some(alpha) = cmd.alpha {
        if alpha <= 0.0 {
            return Err(crate::cli::error::CliError::InvalidArgument(
                format!("Alpha value must be positive, got: {}", alpha)
            ));
        }
    }

    // Validate base model path (basic check)
    if cmd.base_model.trim().is_empty() {
        return Err(crate::cli::error::CliError::InvalidArgument(
            "Base model path cannot be empty".to_string()
        ));
    }

    Ok(())
}

fn load_adapter(adapter_path: &PathBuf, cmd: &ApplyCommand) -> CliResult<LoraParameters> {
    info!("Loading LoRA adapter from: {}", adapter_path.display());
    
    // TODO: Implement proper adapter loading based on file format
    // This would involve:
    // 1. Detecting file format (safetensors, pickle, etc.)
    // 2. Loading the adapter parameters
    // 3. Validating adapter structure
    // 4. Applying alpha override if specified
    
    if !cmd.skip_validation {
        info!("Validating adapter compatibility...");
        // TODO: Implement adapter validation
    }
    
    if let Some(alpha) = cmd.alpha {
        info!("Overriding adapter alpha with: {}", alpha);
        // TODO: Apply alpha override
    }
    
    // Placeholder implementation
    warn!("Adapter loading not yet fully implemented - using placeholder");
    // TODO: Implement proper adapter loading
    // For now, we'll return an error since we can't create a real placeholder
    Err(crate::cli::error::CliError::FeatureNotAvailable(
        "Adapter loading is not yet implemented".to_string()
    ).into())
}

fn initialize_device(device_type: &DeviceType) -> CliResult<Device> {
    match device_type {
        DeviceType::Cuda => {
            // Check if CUDA is available
            match Device::new_cuda(0) {
                Ok(device) => {
                    info!("Successfully initialized CUDA device");
                    Ok(device)
                }
                Err(e) => {
                    warn!("CUDA requested but not available: {}", e);
                    info!("Falling back to CPU");
                    Ok(Device::Cpu)
                }
            }
        }
        DeviceType::Cpu => {
            info!("Using CPU device");
            Ok(Device::Cpu)
        }
        DeviceType::Auto => {
            // Try CUDA first, fall back to CPU
            match Device::new_cuda(0) {
                Ok(device) => {
                    info!("Auto-detected CUDA device");
                    Ok(device)
                }
                Err(_) => {
                    info!("Auto-detected CPU device (CUDA not available)");
                    Ok(Device::Cpu)
                }
            }
        }
    }
}

fn save_application_metadata(
    cmd: &ApplyCommand,
    application_time_ms: u64,
    device: &Device,
) -> CliResult<()> {
    let metadata = ApplicationMetadata {
        base_model: cmd.base_model.clone(),
        adapter_path: cmd.adapter.clone(),
        output_format: format!("{:?}", cmd.format).to_lowercase(),
        merge_weights: cmd.merge_weights,
        device_used: match device {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(_) => "cuda".to_string(),
            Device::Metal(_) => "metal".to_string(),
        },
        alpha_override: cmd.alpha,
        application_time_ms,
        timestamp: chrono::Utc::now(),
        t2l_version: crate::VERSION.to_string(),
    };

    // Save metadata next to the output file
    let metadata_path = cmd.output.with_extension("metadata.json");
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .context("Failed to serialize metadata")?;
    
    fs::write(&metadata_path, metadata_json)
        .context("Failed to write metadata file")?;
    
    info!("Metadata saved to: {}", metadata_path.display());
    Ok(())
}

