use crate::cli::{config::Config, error::CliResult, progress::ProgressReporter};
use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf, sync::Arc};
use tracing::{info, warn, error};
use tokio::runtime::Runtime;
use parking_lot::RwLock;
use candle_core::Device;

// Import from the main lorax crate
use lorax::training::{
    config::{
        TrainingConfig, TrainingType, ModelConfig, ModelArchitecture,
        OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType,
        TrainingParams, DataConfig, DataFormat, CheckpointingConfig,
        LoggingConfig, WandbConfig, MixedPrecisionConfig, RegularizationConfig,
        RuntimeConfig, DeviceConfig, DeviceType,
    },
    checkpoints::{CheckpointManager, TrainingCheckpoint},
    trainer::{TrainingEvent, TrainingResult},
    data::DataLoader,
    trainers::{
        reconstruction::{
            ReconstructionTrainer, ReconstructionTrainerConfig, ReconstructionSettings,
            LayerWeightingStrategy, GradientAccumulationMode, 
            ValidationSettings as ReconstructionValidationSettings, MetricsSettings,
        },
        supervised::{
            SupervisedTrainer, SupervisedTrainerConfig, SupervisedSettings,
            ValidationSettings as SupervisedValidationSettings,
        },
    },
};
use lorax::hypernetwork::{HyperNetwork, HypernetworkConfig};

#[derive(Args, Debug)]
pub struct TrainCommand {
    #[command(subcommand)]
    pub command: TrainSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum TrainSubcommand {
    /// Train T2L hypernetwork with reconstruction loss
    Reconstruction(ReconstructionCommand),
    
    /// Train T2L with supervised fine-tuning
    Supervised(SupervisedCommand),
    
    /// Validate a trained model
    Validate(ValidateCommand),
}

#[derive(Args, Debug)]
pub struct ReconstructionCommand {
    /// Training configuration file (JSON or YAML)
    #[arg(short, long, help = "Training configuration file")]
    pub config: Option<PathBuf>,

    /// Path to LoRA dataset directory
    #[arg(short, long, help = "Path to LoRA dataset")]
    pub data: PathBuf,

    /// T2L model variant (S/M/L)
    #[arg(short, long, help = "T2L model variant")]
    pub model: Option<String>,

    /// Number of training epochs
    #[arg(short, long, help = "Number of training epochs")]
    pub epochs: Option<usize>,

    /// Training batch size
    #[arg(short, long, help = "Training batch size")]
    pub batch_size: Option<usize>,

    /// Learning rate
    #[arg(short, long, help = "Override learning rate")]
    pub learning_rate: Option<f64>,

    /// Resume from checkpoint
    #[arg(long, help = "Resume from checkpoint")]
    pub checkpoint: Option<PathBuf>,

    /// Output directory for checkpoints
    #[arg(short, long, help = "Output directory for checkpoints")]
    pub output: Option<PathBuf>,

    /// LoRA rank for reconstruction
    #[arg(long, help = "LoRA rank")]
    pub lora_rank: Option<usize>,

    /// Gradient accumulation steps
    #[arg(long, help = "Gradient accumulation steps")]
    pub gradient_accumulation: Option<usize>,

    /// Enable mixed precision training
    #[arg(long, help = "Enable mixed precision")]
    pub mixed_precision: bool,

    /// Layer weighting strategy
    #[arg(long, help = "Layer weighting strategy")]
    pub layer_weighting: Option<String>,

    /// Progressive training
    #[arg(long, help = "Enable progressive training")]
    pub progressive: bool,

    /// W&B project name
    #[arg(long, help = "Weights & Biases project")]
    pub wandb_project: Option<String>,

    /// Device to use (cpu/cuda/metal)
    #[arg(long, default_value = "cpu", help = "Device to use")]
    pub device: String,

    /// Validation split ratio
    #[arg(long, help = "Validation split ratio")]
    pub val_split: Option<f64>,

    /// Early stopping patience
    #[arg(long, help = "Early stopping patience")]
    pub early_stopping: Option<usize>,

    /// Dry run without training
    #[arg(long, help = "Dry run mode")]
    pub dry_run: bool,
}

#[derive(Args, Debug)]
pub struct SupervisedCommand {
    /// Training configuration file (JSON or YAML)
    #[arg(short, long, help = "Training configuration file")]
    pub config: Option<PathBuf>,

    /// Path to task dataset
    #[arg(short, long, help = "Path to task dataset")]
    pub data: PathBuf,

    /// Base model to adapt
    #[arg(short, long, help = "Base model to adapt")]
    pub base_model: Option<String>,

    /// Task type (classification/qa/generation)
    #[arg(short, long, help = "Task type")]
    pub task_type: Option<String>,

    /// Number of training epochs
    #[arg(short, long, help = "Number of training epochs")]
    pub epochs: Option<usize>,

    /// Training batch size
    #[arg(short, long, help = "Training batch size")]
    pub batch_size: Option<usize>,

    /// Learning rate
    #[arg(short, long, help = "Override learning rate")]
    pub learning_rate: Option<f64>,

    /// Resume from checkpoint
    #[arg(long, help = "Resume from checkpoint")]
    pub checkpoint: Option<PathBuf>,

    /// Output directory for checkpoints
    #[arg(short, long, help = "Output directory for checkpoints")]
    pub output: Option<PathBuf>,

    /// Maximum sequence length
    #[arg(long, help = "Maximum sequence length")]
    pub max_seq_length: Option<usize>,

    /// Enable mixed precision training
    #[arg(long, help = "Enable mixed precision")]
    pub mixed_precision: bool,

    /// W&B project name
    #[arg(long, help = "Weights & Biases project")]
    pub wandb_project: Option<String>,

    /// Device to use (cpu/cuda/metal)
    #[arg(long, default_value = "cpu", help = "Device to use")]
    pub device: String,

    /// Early stopping patience
    #[arg(long, help = "Early stopping patience")]
    pub early_stopping: Option<usize>,

    /// Multi-task training
    #[arg(long, help = "Enable multi-task training")]
    pub multi_task: bool,

    /// Task weights for multi-task training
    #[arg(long, help = "Task weights (comma-separated)")]
    pub task_weights: Option<String>,

    /// Dry run without training
    #[arg(long, help = "Dry run mode")]
    pub dry_run: bool,
}

#[derive(Args, Debug)]
pub struct ValidateCommand {
    /// Path to model checkpoint
    #[arg(short, long, help = "Path to checkpoint")]
    pub model: PathBuf,

    /// Validation dataset
    #[arg(short, long, help = "Validation dataset")]
    pub data: PathBuf,

    /// Metrics to compute (comma-separated)
    #[arg(long, default_value = "loss,accuracy,perplexity", help = "Metrics to compute")]
    pub metrics: String,

    /// Batch size for validation
    #[arg(long, default_value = "32", help = "Validation batch size")]
    pub batch_size: usize,

    /// Device to use (cpu/cuda/metal)
    #[arg(long, default_value = "cpu", help = "Device to use")]
    pub device: String,

    /// Output file for results
    #[arg(short, long, help = "Output file for results")]
    pub output: Option<PathBuf>,

    /// Detailed layer-wise analysis
    #[arg(long, help = "Enable detailed analysis")]
    pub detailed: bool,
}

// Remove this - we use TrainingConfig from lorax::training

// Remove this - we use TrainingMetrics from lorax::training

pub async fn execute(cmd: TrainCommand, config: Config) -> CliResult<()> {
    match cmd.command {
        TrainSubcommand::Reconstruction(rec_cmd) => {
            execute_reconstruction_training(rec_cmd, config).await
        }
        TrainSubcommand::Supervised(sup_cmd) => {
            execute_supervised_training(sup_cmd, config).await
        }
        TrainSubcommand::Validate(val_cmd) => {
            execute_validation(val_cmd, config).await
        }
    }
}

async fn execute_reconstruction_training(cmd: ReconstructionCommand, cli_config: Config) -> CliResult<()> {
    info!("Starting reconstruction training");

    // Initialize progress reporter
    let progress = ProgressReporter::new("Reconstruction Training")?;
    
    progress.set_message("Loading configuration...");
    
    // Load or create training configuration
    let mut training_config = if let Some(config_path) = &cmd.config {
        info!("Loading training configuration from: {}", config_path.display());
        TrainingConfig::from_file(config_path)
            .context("Failed to load training configuration")?  
    } else {
        info!("Using default reconstruction training configuration");
        create_reconstruction_config(&cmd)?
    };
    
    // Apply command-line overrides
    apply_reconstruction_overrides(&mut training_config, &cmd)?;
    
    // Validate configuration
    training_config.validate()
        .context("Training configuration validation failed")?;
    
    if cmd.dry_run {
        info!("Dry run mode - configuration validated successfully");
        if let Some(output) = &cmd.output {
            let config_path = output.join("training_config.yaml");
            training_config.to_file(&config_path)?;
            info!("Configuration saved to: {}", config_path.display());
        }
        return Ok(());
    }
    
    progress.advance("Initializing training components...");
    
    // Set up device
    let device = parse_device(&cmd.device)?;
    info!("Using device: {:?}", device);
    
    // Initialize hypernetwork
    let hypernetwork_config = create_hypernetwork_config(&training_config)?;
    let hypernetwork = HyperNetwork::new(hypernetwork_config)
        .context("Failed to create hypernetwork")?;
    let hypernetwork = Arc::new(RwLock::new(hypernetwork));
    
    progress.advance("Loading training data...");
    
    // Create data loaders
    let (train_loader, val_loader) = create_reconstruction_data_loaders(
        &training_config.data,
        &device
    ).await?;
    
    info!("Training data loaded successfully");
    
    // Create reconstruction trainer config
    let trainer_config = ReconstructionTrainerConfig {
        base_config: training_config,
        reconstruction: ReconstructionSettings::default(),
        validation: ReconstructionValidationSettings::default(),
        metrics: MetricsSettings::default(),
    };
    
    // Initialize trainer
    let mut trainer = ReconstructionTrainer::new(
        trainer_config,
        hypernetwork.clone(),
        train_loader,
        val_loader,
        device,
    ).context("Failed to create trainer")?;
    
    progress.advance("Starting training...");
    
    // Set up event monitoring
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    trainer = trainer.with_event_monitoring(tx);
    
    // Spawn task to handle training events
    let progress_handle = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                TrainingEvent::StepCompleted { step, loss, lr } => {
                    progress.update_metrics(vec![
                        ("step", step as f64),
                        ("loss", loss),
                        ("lr", lr),
                    ]);
                }
                TrainingEvent::EpochCompleted { epoch, metrics } => {
                    progress.set_message(&format!("Epoch {} completed", epoch + 1));
                }
                TrainingEvent::CheckpointSaved { path } => {
                    info!("Checkpoint saved: {}", path.display());
                }
                _ => {}
            }
        }
    });
    
    // Run training
    let result = trainer.train().await?;
    
    // Wait for progress updates to complete
    drop(progress_handle);
    
    progress.finish("Reconstruction training completed!");
    
    // Log results
    info!("Training completed successfully");
    info!("Final loss: {:.4}", result.final_metrics.get("train_loss").unwrap_or(&0.0));
    info!("Total steps: {}", result.total_steps);
    info!("Training duration: {:?}", result.training_duration);
    
    if let Some(checkpoint_path) = &result.best_checkpoint_path {
        info!("Best model saved to: {}", checkpoint_path.display());
    }
    
    Ok(())
}

async fn execute_supervised_training(cmd: SupervisedCommand, cli_config: Config) -> CliResult<()> {
    info!("Starting supervised fine-tuning");

    let progress = ProgressReporter::new("Supervised Training")?;
    
    progress.set_message("Loading configuration...");
    
    // Load or create training configuration
    let mut training_config = if let Some(config_path) = &cmd.config {
        info!("Loading training configuration from: {}", config_path.display());
        TrainingConfig::from_file(config_path)
            .context("Failed to load training configuration")?  
    } else {
        info!("Using default supervised training configuration");
        create_supervised_config(&cmd)?
    };
    
    // Apply command-line overrides
    apply_supervised_overrides(&mut training_config, &cmd)?;
    
    // Validate configuration
    training_config.validate()
        .context("Training configuration validation failed")?;
    
    if cmd.dry_run {
        info!("Dry run mode - configuration validated successfully");
        if let Some(output) = &cmd.output {
            let config_path = output.join("training_config.yaml");
            training_config.to_file(&config_path)?;
            info!("Configuration saved to: {}", config_path.display());
        }
        return Ok(());
    }
    
    progress.advance("Initializing training components...");
    
    // Set up device
    let device = parse_device(&cmd.device)?;
    info!("Using device: {:?}", device);
    
    // Initialize hypernetwork
    let hypernetwork_config = create_hypernetwork_config(&training_config)?;
    let hypernetwork = HyperNetwork::new(hypernetwork_config)
        .context("Failed to create hypernetwork")?;
    let hypernetwork = Arc::new(RwLock::new(hypernetwork));
    
    progress.advance("Loading training data...");
    
    // Create data loaders
    let (train_loader, val_loader) = create_supervised_data_loaders(
        &training_config.data,
        &device
    ).await?;
    
    info!("Training data loaded successfully");
    
    // Create supervised trainer config
    let trainer_config = SupervisedTrainerConfig {
        base_config: training_config,
        supervised: SupervisedSettings::default(),
        validation: SupervisedValidationSettings::default(),
        metrics: MetricsSettings {
            log_param_stats_interval: 100,
            log_gradient_flow_interval: 500,
            track_memory_usage: true,
            enable_profiling: false,
        },
    };
    
    // Initialize trainer
    let mut trainer = SupervisedTrainer::new(
        trainer_config,
        hypernetwork.clone(),
        train_loader,
        val_loader,
        device,
    ).context("Failed to create trainer")?;
    
    progress.advance("Starting training...");
    
    // Set up event monitoring
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    trainer = trainer.with_event_monitoring(tx);
    
    // Spawn task to handle training events
    let progress_handle = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                TrainingEvent::StepCompleted { step, loss, lr } => {
                    progress.update_metrics(vec![
                        ("step", step as f64),
                        ("loss", loss),
                        ("lr", lr),
                    ]);
                }
                TrainingEvent::EpochCompleted { epoch, metrics } => {
                    progress.set_message(&format!("Epoch {} completed", epoch + 1));
                }
                TrainingEvent::CheckpointSaved { path } => {
                    info!("Checkpoint saved: {}", path.display());
                }
                _ => {}
            }
        }
    });
    
    // Run training
    let result = trainer.train().await?;
    
    // Wait for progress updates to complete
    drop(progress_handle);
    
    progress.finish("Supervised training completed!");
    
    // Log results
    info!("Training completed successfully");
    info!("Final loss: {:.4}", result.final_metrics.get("train_loss").unwrap_or(&0.0));
    info!("Total steps: {}", result.total_steps);
    info!("Training duration: {:?}", result.training_duration);
    
    if let Some(checkpoint_path) = &result.best_checkpoint_path {
        info!("Best model saved to: {}", checkpoint_path.display());
    }
    
    Ok(())
}

async fn execute_validation(cmd: ValidateCommand, cli_config: Config) -> CliResult<()> {
    info!("Starting model validation");
    
    let progress = ProgressReporter::new("Model Validation")?;
    
    progress.set_message("Loading model checkpoint...");
    
    // Validate paths
    if !cmd.model.exists() {
        return Err(crate::cli::error::CliError::FileNotFound(cmd.model));
    }
    
    if !cmd.data.exists() {
        return Err(crate::cli::error::CliError::DatasetNotFound(cmd.data));
    }
    
    // Set up device
    let device = parse_device(&cmd.device)?;
    info!("Using device: {:?}", device);
    
    // Load checkpoint
    let checkpoint_manager = CheckpointManager::new(
        CheckpointingConfig::default(),
        device.clone(),
    )?;
    let checkpoint = checkpoint_manager.load_checkpoint(&cmd.model)?;
    
    progress.advance("Loading validation data...");
    
    // Create data loader for validation
    // Note: This is a placeholder - in a real implementation,
    // you would create the appropriate dataset and data loader
    // based on the checkpoint configuration and data format
    warn!("Validation data loader creation not fully implemented");
    
    // For now, we'll return a placeholder error
    Err(crate::cli::error::CliError::FeatureNotAvailable(
        "Model validation is not yet fully implemented".to_string()
    ))
}

/// Create reconstruction training configuration from command
fn create_reconstruction_config(cmd: &ReconstructionCommand) -> Result<TrainingConfig> {
    let mut config = TrainingConfig::reconstruction_default();
    
    // Set model variant
    if let Some(ref variant) = cmd.model {
        config.model.hidden_size = match variant.as_str() {
            "S" => 2048,
            "M" => 4096,
            "L" => 8192,
            _ => return Err(anyhow::anyhow!("Invalid model variant: {}", variant)),
        };
    }
    
    // Set data path
    config.data.train_data_path = cmd.data.clone();
    
    // Set output directory
    if let Some(ref output) = cmd.output {
        config.checkpointing.output_dir = output.clone();
    }
    
    Ok(config)
}

/// Apply command-line overrides to reconstruction config
fn apply_reconstruction_overrides(config: &mut TrainingConfig, cmd: &ReconstructionCommand) -> Result<()> {
    // Training parameters
    if let Some(epochs) = cmd.epochs {
        config.training.num_epochs = epochs;
    }
    
    if let Some(batch_size) = cmd.batch_size {
        config.training.batch_size = batch_size;
    }
    
    if let Some(learning_rate) = cmd.learning_rate {
        config.optimizer.learning_rate = learning_rate;
    }
    
    if let Some(lora_rank) = cmd.lora_rank {
        config.model.lora_rank = lora_rank;
    }
    
    if let Some(grad_accum) = cmd.gradient_accumulation {
        config.training.gradient_accumulation_steps = grad_accum;
    }
    
    // Checkpoint resume
    if let Some(ref checkpoint) = cmd.checkpoint {
        config.training.resume_from_checkpoint = Some(checkpoint.clone());
    }
    
    // Mixed precision
    config.mixed_precision.enabled = cmd.mixed_precision;
    
    // Early stopping
    if let Some(patience) = cmd.early_stopping {
        config.training.early_stopping.enabled = true;
        config.training.early_stopping.patience = patience;
    }
    
    // W&B logging
    if let Some(ref project) = cmd.wandb_project {
        config.logging.wandb.enabled = true;
        config.logging.wandb.project = Some(project.clone());
    }
    
    Ok(())
}

/// Create supervised training configuration from command
fn create_supervised_config(cmd: &SupervisedCommand) -> Result<TrainingConfig> {
    let mut config = TrainingConfig::supervised_default();
    
    // Set data path
    config.data.train_data_path = cmd.data.clone();
    
    // Set output directory
    if let Some(ref output) = cmd.output {
        config.checkpointing.output_dir = output.clone();
    }
    
    // Set base model architecture
    if let Some(ref base_model) = cmd.base_model {
        config.model.architecture = match base_model.to_lowercase().as_str() {
            "llama" | "llama-7b" => ModelArchitecture::Llama,
            "mistral" | "mistral-7b" => ModelArchitecture::Mistral,
            "gemma" => ModelArchitecture::Gemma,
            "bert" => ModelArchitecture::Bert,
            _ => ModelArchitecture::Custom { name: base_model.clone() },
        };
    }
    
    // Set task type for multi-task training
    if cmd.multi_task {
        let tasks = if let Some(ref task_type) = cmd.task_type {
            vec![task_type.clone()]
        } else {
            vec!["default".to_string()]
        };
        
        let task_weights = if let Some(ref weights_str) = cmd.task_weights {
            weights_str.split(',').map(|w| w.parse::<f64>().unwrap_or(1.0)).collect()
        } else {
            vec![1.0; tasks.len()]
        };
        
        config.model.training_type = TrainingType::MultiTask { tasks, task_weights };
    }
    
    Ok(config)
}

/// Apply command-line overrides to supervised config
fn apply_supervised_overrides(config: &mut TrainingConfig, cmd: &SupervisedCommand) -> Result<()> {
    // Training parameters
    if let Some(epochs) = cmd.epochs {
        config.training.num_epochs = epochs;
    }
    
    if let Some(batch_size) = cmd.batch_size {
        config.training.batch_size = batch_size;
    }
    
    if let Some(learning_rate) = cmd.learning_rate {
        config.optimizer.learning_rate = learning_rate;
    }
    
    if let Some(max_seq_length) = cmd.max_seq_length {
        config.model.max_sequence_length = max_seq_length;
    }
    
    // Checkpoint resume
    if let Some(ref checkpoint) = cmd.checkpoint {
        config.training.resume_from_checkpoint = Some(checkpoint.clone());
    }
    
    // Mixed precision
    config.mixed_precision.enabled = cmd.mixed_precision;
    
    // Early stopping
    if let Some(patience) = cmd.early_stopping {
        config.training.early_stopping.enabled = true;
        config.training.early_stopping.patience = patience;
    }
    
    // W&B logging
    if let Some(ref project) = cmd.wandb_project {
        config.logging.wandb.enabled = true;
        config.logging.wandb.project = Some(project.clone());
    }
    
    Ok(())
}

/// Parse device string to Device enum
fn parse_device(device_str: &str) -> Result<Device> {
    match device_str.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" | "gpu" => {
            if candle_core::cuda_is_available() {
                Ok(Device::new_cuda(0)?)
            } else {
                Err(anyhow::anyhow!("CUDA is not available on this system"))
            }
        }
        "metal" => {
            if candle_core::metal_is_available() {
                Ok(Device::new_metal(0)?)
            } else {
                Err(anyhow::anyhow!("Metal is not available on this system"))
            }
        }
        _ => Err(anyhow::anyhow!("Unknown device: {}", device_str)),
    }
}

/// Create hypernetwork configuration from training config
fn create_hypernetwork_config(training_config: &TrainingConfig) -> Result<HypernetworkConfig> {
    // Map model architecture to hypernetwork config
    let input_dim = match training_config.model.architecture {
        ModelArchitecture::Llama => 4096,
        ModelArchitecture::Mistral => 4096,
        ModelArchitecture::Gemma => 3072,
        ModelArchitecture::Bert => 768,
        ModelArchitecture::Custom { .. } => training_config.model.hidden_size,
    };
    
    Ok(HypernetworkConfig {
        input_dim,
        hidden_dim: training_config.model.hidden_size / 2,
        num_layers: 6,
        lora_rank: training_config.model.lora_rank,
        activation: "gelu".to_string(),
        use_layer_norm: true,
        dropout_rate: training_config.regularization.dropout,
        initialization: "xavier_uniform".to_string(),
        use_bias: true,
    })
}

/// Create data loaders for reconstruction training
async fn create_reconstruction_data_loaders(
    data_config: &DataConfig,
    device: &Device,
) -> Result<(DataLoader, Option<DataLoader>)> {
    // Note: This is a placeholder implementation
    // In a real implementation, you would:
    // 1. Create the appropriate dataset based on data_config.data_format
    // 2. Create proper batch collators
    // 3. Configure the DataLoader properly
    
    // For now, return an error indicating this needs implementation
    Err(anyhow::anyhow!("Reconstruction data loader creation not yet implemented. Please implement dataset loading based on your data format."))
}

/// Create data loaders for supervised training
async fn create_supervised_data_loaders(
    data_config: &DataConfig,
    device: &Device,
) -> Result<(DataLoader, Option<DataLoader>)> {
    // Note: This is a placeholder implementation
    // In a real implementation, you would:
    // 1. Create the appropriate dataset based on data_config.data_format
    // 2. Create proper batch collators for supervised learning
    // 3. Configure the DataLoader properly
    
    // For now, return an error indicating this needs implementation
    Err(anyhow::anyhow!("Supervised data loader creation not yet implemented. Please implement dataset loading based on your data format."))
}

// Add missing imports if needed
use std::collections::HashMap;