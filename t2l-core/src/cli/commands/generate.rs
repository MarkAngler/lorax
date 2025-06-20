use crate::cli::{config::Config, error::CliResult, progress::ProgressReporter};
use anyhow::{Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};
use tracing::{info, warn};

#[derive(Args, Debug)]
pub struct GenerateCommand {
    /// Task description for which to generate a LoRA adapter
    #[arg(short, long, help = "Natural language description of the task")]
    pub task: String,

    /// Output file for the generated LoRA parameters
    #[arg(short, long, help = "Path to save the generated LoRA adapter")]
    pub output: PathBuf,

    /// Model architecture to target (e.g., "llama", "mistral", "gemma")
    #[arg(short, long, default_value = "llama", help = "Target model architecture")]
    pub architecture: String,

    /// Model size variant (L, M, S)
    #[arg(long, default_value = "M", help = "Hypernetwork size variant")]
    pub variant: String,

    /// LoRA rank (overrides config default)
    #[arg(long, help = "LoRA rank parameter")]
    pub rank: Option<usize>,

    /// LoRA alpha scaling factor
    #[arg(long, help = "LoRA alpha scaling factor")]
    pub alpha: Option<f32>,

    /// Force overwrite existing output file
    #[arg(short, long, help = "Overwrite existing output file")]
    pub force: bool,

    /// Export format (safetensors, pickle, json)
    #[arg(long, default_value = "safetensors", help = "Output format")]
    pub format: ExportFormat,

    /// Include metadata in the output
    #[arg(long, help = "Include generation metadata")]
    pub include_metadata: bool,

    /// Batch generation from file containing multiple tasks
    #[arg(long, conflicts_with = "task", help = "Path to file with multiple tasks")]
    pub batch_file: Option<PathBuf>,

    /// Number of parallel workers for batch generation
    #[arg(long, default_value = "1", help = "Number of parallel workers")]
    pub workers: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, clap::ValueEnum)]
pub enum ExportFormat {
    #[value(name = "safetensors")]
    SafeTensors,
    #[value(name = "pickle")]
    Pickle,
    #[value(name = "json")]
    Json,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerationMetadata {
    pub task_description: String,
    pub architecture: String,
    pub variant: String,
    pub rank: usize,
    pub alpha: f32,
    pub generation_time_ms: u64,
    pub hypernetwork_version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchTask {
    pub id: String,
    pub task: String,
    pub output: PathBuf,
    pub architecture: Option<String>,
    pub rank: Option<usize>,
    pub alpha: Option<f32>,
}

pub async fn execute(cmd: GenerateCommand, config: Config) -> CliResult<()> {
    info!("Starting LoRA generation with T2L");

    // Validate inputs
    validate_command(&cmd)?;

    // Check if batch generation is requested
    if let Some(batch_file) = &cmd.batch_file {
        return execute_batch_generation(cmd, config, batch_file).await;
    }

    // Single task generation
    execute_single_generation(cmd, config).await
}

async fn execute_single_generation(cmd: GenerateCommand, config: Config) -> CliResult<()> {
    let start_time = std::time::Instant::now();
    
    // Initialize progress reporter
    let progress = ProgressReporter::new("Generating LoRA adapter")?;
    
    progress.set_message("Initializing T2L system...");
    
    // Initialize T2L system
    let t2l_config = build_t2l_config(&cmd, &config)?;
    let t2l_system = lorax::TextToLora::new(t2l_config)
        .await
        .context("Failed to initialize T2L system")?;
    
    progress.advance("Encoding task description...");
    
    // Generate LoRA parameters
    let lora_params = t2l_system
        .generate(&cmd.task)
        .await
        .context("Failed to generate LoRA parameters")?;
    
    progress.advance("Exporting LoRA adapter...");
    
    // Export to specified format
    export_lora_adapter(&cmd, &lora_params, start_time.elapsed().as_millis() as u64)?;
    
    progress.finish("LoRA adapter generated successfully!");
    
    info!("Generated LoRA adapter for task: {}", cmd.task);
    info!("Output saved to: {}", cmd.output.display());
    
    Ok(())
}

async fn execute_batch_generation(
    cmd: GenerateCommand,
    config: Config,
    batch_file: &PathBuf,
) -> CliResult<()> {
    info!("Starting batch LoRA generation");
    
    // Load batch tasks
    let tasks = load_batch_tasks(batch_file)?;
    info!("Loaded {} tasks for batch generation", tasks.len());
    
    // Initialize T2L system
    let t2l_config = build_t2l_config(&cmd, &config)?;
    let t2l_system = lorax::TextToLora::new(t2l_config)
        .await
        .context("Failed to initialize T2L system")?;
    
    // Create progress bar for batch processing
    let progress = ProgressReporter::new_with_total("Batch generation", tasks.len() as u64)?;
    
    // Process tasks in parallel using the specified number of workers
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(cmd.workers));
    let mut handles = Vec::new();
    
    for task in tasks {
        let t2l_system = std::sync::Arc::clone(&std::sync::Arc::new(t2l_system));
        let semaphore = std::sync::Arc::clone(&semaphore);
        let progress = progress.clone();
        let cmd = cmd.clone();
        
        let handle = tokio::spawn(async move {
            let _permit = semaphore.acquire().await.unwrap();
            
            let start_time = std::time::Instant::now();
            
            // Generate LoRA parameters
            let lora_params = t2l_system
                .generate(&task.task)
                .await
                .context("Failed to generate LoRA parameters")?;
            
            // Create individual command for this task
            let task_cmd = GenerateCommand {
                task: task.task.clone(),
                output: task.output.clone(),
                architecture: task.architecture.unwrap_or_else(|| cmd.architecture.clone()),
                rank: task.rank.or(cmd.rank),
                alpha: task.alpha.or(cmd.alpha),
                ..cmd
            };
            
            // Export the adapter
            export_lora_adapter(&task_cmd, &lora_params, start_time.elapsed().as_millis() as u64)?;
            
            progress.advance(&format!("Completed task: {}", task.id));
            
            Ok::<(), anyhow::Error>(())
        });
        
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        handle.await??;
    }
    
    progress.finish("Batch generation completed!");
    info!("All {} tasks completed successfully", tasks.len());
    
    Ok(())
}

fn validate_command(cmd: &GenerateCommand) -> CliResult<()> {
    // Check output path doesn't exist or force is specified
    if cmd.output.exists() && !cmd.force {
        return Err(crate::cli::error::CliError::FileExists(cmd.output.clone()));
    }
    
    // Validate variant
    if !["L", "M", "S"].contains(&cmd.variant.as_str()) {
        return Err(crate::cli::error::CliError::InvalidVariant(cmd.variant.clone()));
    }
    
    // Validate architecture
    let supported_archs = ["llama", "mistral", "gemma", "phi", "qwen"];
    if !supported_archs.contains(&cmd.architecture.as_str()) {
        warn!("Architecture '{}' may not be fully supported", cmd.architecture);
    }
    
    // Validate rank if specified
    if let Some(rank) = cmd.rank {
        if rank == 0 || rank > 512 {
            return Err(crate::cli::error::CliError::InvalidRank(rank));
        }
    }
    
    Ok(())
}

fn build_t2l_config(cmd: &GenerateCommand, config: &Config) -> CliResult<lorax::Config> {
    let mut t2l_config = config.t2l.clone();
    
    // Override with command-line arguments
    if let Some(rank) = cmd.rank {
        t2l_config.lora.rank = rank;
    }
    
    if let Some(alpha) = cmd.alpha {
        t2l_config.lora.alpha = alpha;
    }
    
    // Set hypernetwork variant
    t2l_config.hypernetwork.model_size = match cmd.variant.as_str() {
        "L" => lorax::ModelSize::Large,
        "M" => lorax::ModelSize::Medium,
        "S" => lorax::ModelSize::Small,
        _ => unreachable!(), // Already validated
    };
    
    Ok(t2l_config)
}

fn export_lora_adapter(
    cmd: &GenerateCommand,
    lora_params: &lorax::LoraParameters,
    generation_time_ms: u64,
) -> CliResult<()> {
    // Create output directory if it doesn't exist
    if let Some(parent) = cmd.output.parent() {
        fs::create_dir_all(parent)?;
    }
    
    // Create metadata if requested
    let metadata = if cmd.include_metadata {
        Some(GenerationMetadata {
            task_description: cmd.task.clone(),
            architecture: cmd.architecture.clone(),
            variant: cmd.variant.clone(),
            rank: cmd.rank.unwrap_or(16), // Default from config
            alpha: cmd.alpha.unwrap_or(32.0), // Default from config
            generation_time_ms,
            hypernetwork_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now(),
        })
    } else {
        None
    };
    
    // Export based on format
    match cmd.format {
        ExportFormat::SafeTensors => {
            export_safetensors(&cmd.output, lora_params, metadata.as_ref())?;
        }
        ExportFormat::Pickle => {
            export_pickle(&cmd.output, lora_params, metadata.as_ref())?;
        }
        ExportFormat::Json => {
            export_json(&cmd.output, lora_params, metadata.as_ref())?;
        }
    }
    
    Ok(())
}

fn load_batch_tasks(batch_file: &PathBuf) -> CliResult<Vec<BatchTask>> {
    let content = fs::read_to_string(batch_file)
        .context("Failed to read batch file")?;
    
    // Try to parse as different formats
    if batch_file.extension().map_or(false, |ext| ext == "json") {
        let tasks: Vec<BatchTask> = serde_json::from_str(&content)
            .context("Failed to parse batch file as JSON")?;
        Ok(tasks)
    } else if batch_file.extension().map_or(false, |ext| ext == "yaml" || ext == "yml") {
        let tasks: Vec<BatchTask> = serde_yaml::from_str(&content)
            .context("Failed to parse batch file as YAML")?;
        Ok(tasks)
    } else {
        // Assume simple text format: one task per line
        let tasks: Vec<BatchTask> = content
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .map(|(i, line)| BatchTask {
                id: format!("task_{}", i + 1),
                task: line.trim().to_string(),
                output: PathBuf::from(format!("output_{}.safetensors", i + 1)),
                architecture: None,
                rank: None,
                alpha: None,
            })
            .collect();
        Ok(tasks)
    }
}

fn export_safetensors(
    output: &PathBuf,
    _lora_params: &lorax::LoraParameters,
    _metadata: Option<&GenerationMetadata>,
) -> CliResult<()> {
    // TODO: Implement SafeTensors export
    // This would use the safetensors crate to serialize the LoRA parameters
    warn!("SafeTensors export not yet fully implemented - creating placeholder");
    fs::write(output, b"placeholder safetensors data")?;
    Ok(())
}

fn export_pickle(
    output: &PathBuf,
    _lora_params: &lorax::LoraParameters,
    _metadata: Option<&GenerationMetadata>,
) -> CliResult<()> {
    // TODO: Implement Pickle export for Python compatibility
    warn!("Pickle export not yet fully implemented - creating placeholder");
    fs::write(output, b"placeholder pickle data")?;
    Ok(())
}

fn export_json(
    output: &PathBuf,
    _lora_params: &lorax::LoraParameters,
    metadata: Option<&GenerationMetadata>,
) -> CliResult<()> {
    // TODO: Implement JSON export with proper serialization
    let json_data = serde_json::json!({
        "lora_parameters": "placeholder - implement serialization",
        "metadata": metadata
    });
    
    fs::write(output, serde_json::to_string_pretty(&json_data)?)?;
    Ok(())
}