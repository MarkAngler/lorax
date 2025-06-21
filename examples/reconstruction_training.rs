//! Example of reconstruction-based T2L training
//!
//! This example demonstrates how to use the ReconstructionTrainer to train
//! a hypernetwork to generate LoRA parameters from task embeddings.

use anyhow::Result;
use candle_core::Device;
use lorax::hypernetwork::{HyperNetwork, HypernetworkConfig, ModelSize};
use lorax::training::{
    ReconstructionTrainer, ReconstructionTrainerConfig,
    ReconstructionDataset, DataLoader, DataLoaderConfig,
    TrainingConfig, TrainingType, ModelConfig, ModelArchitecture,
    OptimizerConfig, OptimizerType, SchedulerType, SchedulerConfig,
    CheckpointingConfig, LoggingConfig, MixedPrecisionConfig,
    TrainingParams, DataConfig,
    PrecisionType, LossScalingConfig, LossScalingMethod,
};
use std::path::Path;
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("Starting reconstruction training example");
    
    // Set device
    let device = Device::cuda_if_available(0)?;
    info!("Using device: {:?}", device);
    
    // Create hypernetwork configuration
    let hypernetwork_config = HypernetworkConfig {
        model_size: ModelSize::Medium,
        input_dim: 768,  // BERT embedding dimension
        lora_rank: 16,
        dropout: 0.1,
        activation: lorax::hypernetwork::ActivationType::GELU,
    };
    
    // Create hypernetwork model
    let model = Arc::new(RwLock::new(HyperNetwork::new(hypernetwork_config)?));
    
    // Create training configuration
    let mut base_config = TrainingConfig {
        model: ModelConfig {
            architecture: ModelArchitecture::Hypernetwork { 
                size: "medium".to_string() 
            },
            training_type: TrainingType::Reconstruction,
            checkpoint_path: None,
        },
        optimizer: OptimizerConfig {
            optimizer_type: OptimizerType::AdamW {
                betas: (0.9, 0.999),
                eps: 1e-8,
                weight_decay: 0.01,
            },
            learning_rate: 1e-4,
            scheduler: SchedulerConfig {
                scheduler_type: SchedulerType::Cosine {
                    num_warmup_steps: 1000,
                    num_training_steps: 10000,
                },
            },
            gradient_clipping: lorax::training::GradientClippingConfig {
                enabled: true,
                method: lorax::training::ClippingMethod::GlobalNorm,
                threshold: 1.0,
            },
        },
        training: TrainingParams {
            num_epochs: 10,
            batch_size: 16,
            gradient_accumulation_steps: 4,
            eval_steps: 500,
            save_steps: 1000,
            log_steps: 100,
            max_steps: None,
            seed: 42,
            resume_from_checkpoint: None,
            early_stopping: lorax::training::EarlyStoppingConfig {
                enabled: true,
                monitor_metric: "eval_loss".to_string(),
                patience: 3,
                min_delta: 0.001,
                higher_is_better: false,
            },
        },
        data: DataConfig {
            train_data_path: "data/reconstruction/train".to_string(),
            eval_data_path: Some("data/reconstruction/val".to_string()),
            cache_data: true,
            num_workers: 4,
            shuffle: true,
            drop_last: false,
        },
        checkpointing: CheckpointingConfig {
            save_dir: "checkpoints/reconstruction".to_string(),
            save_total_limit: 5,
            save_best_only: true,
            monitor_metric: "eval_loss".to_string(),
            mode: lorax::training::CheckpointMode::Min,
            save_on_each_epoch: true,
            save_optimizer_state: true,
            save_scheduler_state: true,
            compression_level: 6,
        },
        logging: LoggingConfig {
            log_dir: "logs/reconstruction".to_string(),
            log_level: "info".to_string(),
            log_to_file: true,
            log_to_console: true,
            report_to: vec!["tensorboard".to_string()],
            logging_steps: 10,
            include_system_metrics: true,
        },
        mixed_precision: MixedPrecisionConfig {
            enabled: true,
            precision: PrecisionType::FP16,
            loss_scaling: LossScalingConfig {
                method: LossScalingMethod::Dynamic,
                init_scale: 65536.0,
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2000,
            },
            gradient_scaling: true,
        },
        ..Default::default()
    };
    
    // Create specialized reconstruction trainer config
    let trainer_config = ReconstructionTrainerConfig {
        base_config: base_config.clone(),
        reconstruction: lorax::training::trainers::reconstruction::ReconstructionSettings {
            layer_weighting: lorax::training::trainers::reconstruction::LayerWeightingStrategy::Uniform,
            gradient_accumulation_mode: lorax::training::trainers::reconstruction::GradientAccumulationMode::Fixed { 
                steps: 4 
            },
            progressive_training: None,
            param_norm_clip: Some(1.0),
            track_param_magnitudes: true,
            analyze_gradient_flow: false,
        },
        validation: lorax::training::trainers::reconstruction::ValidationSettings {
            compute_alignment: true,
            compute_effective_rank: false,
            track_layer_accuracy: true,
            best_model_metric: "eval_loss".to_string(),
        },
        metrics: lorax::training::trainers::reconstruction::MetricsSettings {
            log_param_stats_interval: 100,
            log_gradient_flow_interval: 500,
            track_memory_usage: true,
            enable_profiling: false,
        },
    };
    
    // Create datasets
    info!("Loading datasets...");
    let train_dataset = ReconstructionDataset::new(
        &base_config.data.train_data_path,
        device.clone(),
        None::<&Path>,
        base_config.data.cache_data,
    )?;
    
    let val_dataset = if let Some(ref eval_path) = base_config.data.eval_data_path {
        Some(ReconstructionDataset::new(
            eval_path,
            device.clone(),
            None::<&Path>,
            base_config.data.cache_data,
        )?)
    } else {
        None
    };
    
    // Create data loaders
    let train_loader = DataLoader::new(
        Arc::new(train_dataset),
        DataLoaderConfig {
            batch_size: base_config.training.batch_size,
            shuffle: base_config.data.shuffle,
            drop_last: base_config.data.drop_last,
            num_workers: base_config.data.num_workers,
            pin_memory: true,
            prefetch_factor: Some(2),
        },
        device.clone(),
    )?;
    
    let val_loader = val_dataset.map(|dataset| {
        DataLoader::new(
            Arc::new(dataset),
            DataLoaderConfig {
                batch_size: base_config.training.batch_size,
                shuffle: false,
                drop_last: false,
                num_workers: base_config.data.num_workers,
                pin_memory: true,
                prefetch_factor: Some(2),
            },
            device.clone(),
        )
    }).transpose()?;
    
    // Create trainer
    info!("Creating reconstruction trainer...");
    let mut trainer = ReconstructionTrainer::new(
        trainer_config,
        model,
        train_loader,
        val_loader,
        device,
    )?;
    
    // Set up event monitoring
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    trainer = trainer.with_event_monitoring(event_tx);
    
    // Spawn task to handle training events
    tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            match event {
                lorax::training::TrainingEvent::EpochStarted { epoch } => {
                    info!("Epoch {} started", epoch + 1);
                }
                lorax::training::TrainingEvent::EpochCompleted { epoch, metrics } => {
                    info!("Epoch {} completed", epoch + 1);
                    info!("Metrics: {:?}", metrics);
                }
                lorax::training::TrainingEvent::StepCompleted { step, loss, lr } => {
                    if step % 100 == 0 {
                        info!("Step {}: loss={:.4}, lr={:.2e}", step, loss, lr);
                    }
                }
                lorax::training::TrainingEvent::EvaluationCompleted { metrics } => {
                    info!("Evaluation completed: {:?}", metrics);
                }
                lorax::training::TrainingEvent::CheckpointSaved { path } => {
                    info!("Checkpoint saved to: {:?}", path);
                }
                lorax::training::TrainingEvent::EarlyStopping { reason } => {
                    info!("Early stopping: {}", reason);
                }
                lorax::training::TrainingEvent::TrainingCompleted { total_steps } => {
                    info!("Training completed after {} steps", total_steps);
                }
                lorax::training::TrainingEvent::Error { error } => {
                    eprintln!("Training error: {}", error);
                }
            }
        }
    });
    
    // Start training
    info!("Starting training...");
    let result = trainer.train().await?;
    
    // Print final results
    info!("Training completed!");
    info!("Final metrics: {:?}", result.final_metrics);
    info!("Total steps: {}", result.total_steps);
    info!("Training duration: {:?}", result.training_duration);
    
    if let Some(best_checkpoint) = result.best_checkpoint_path {
        info!("Best checkpoint saved at: {:?}", best_checkpoint);
    }
    
    Ok(())
}