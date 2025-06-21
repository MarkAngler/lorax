//! Example of supervised fine-tuning with T2L
//!
//! This example demonstrates how to use the SupervisedTrainer to train
//! a hypernetwork that generates task-specific LoRA parameters for
//! downstream task adaptation.

use anyhow::Result;
use candle_core::Device;
use lorax::hypernetwork::{HyperNetwork, HypernetworkConfig, ModelSize, TargetArchitecture};
use lorax::models::{ModelType, create_base_model};
use lorax::training::{
    SupervisedTrainer, SupervisedTrainerConfig,
    SupervisedDataset, DataLoader, DataLoaderConfig,
    TrainingConfig, TrainingType, ModelConfig, ModelArchitecture,
    OptimizerConfig, OptimizerType, SchedulerType, SchedulerConfig,
    CheckpointingConfig, LoggingConfig, MixedPrecisionConfig,
    TrainingParams, DataConfig, GradientClippingConfig, ClippingMethod,
    EarlyStoppingConfig, CheckpointMode, SupervisedTaskType,
    PrecisionType, LossScalingConfig, LossScalingMethod,
};
use lorax::training::trainers::supervised::{
    BaseModelConfig, SupervisedSettings, LoraAdaptationConfig, ValidationSettings,
    GradientAccumulationMode, TaskBalancingStrategy, EvaluationMode,
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
    
    info!("Starting supervised fine-tuning example");
    
    // Set device (use GPU if available)
    let device = Device::cuda_if_available(0)?;
    info!("Using device: {:?}", device);
    
    // Create hypernetwork configuration for Llama-based model
    let hypernetwork_config = HypernetworkConfig {
        model_size: ModelSize::Large,  // Larger model for complex tasks
        input_dim: 768,  // BERT embedding dimension for task encodings
        lora_rank: 32,   // Higher rank for more expressiveness
        dropout: 0.1,
        activation: lorax::hypernetwork::ActivationType::SiLU,
        target_architecture: Some(TargetArchitecture::Llama),
    };
    
    // Create hypernetwork model
    let hypernetwork = Arc::new(RwLock::new(HyperNetwork::new(hypernetwork_config)?));
    
    // Create base training configuration
    let base_config = TrainingConfig {
        model: ModelConfig {
            architecture: ModelArchitecture::Hypernetwork { 
                size: "large".to_string() 
            },
            training_type: TrainingType::Supervised,
            checkpoint_path: None,
        },
        optimizer: OptimizerConfig {
            optimizer_type: OptimizerType::AdamW {
                betas: (0.9, 0.999),
                eps: 1e-8,
                weight_decay: 0.01,
            },
            learning_rate: 5e-5,  // Lower LR for fine-tuning
            scheduler: SchedulerConfig {
                scheduler_type: SchedulerType::CosineWithWarmup {
                    num_warmup_steps: 500,
                    num_training_steps: 10000,
                    min_lr_ratio: 0.1,
                },
            },
            gradient_clipping: GradientClippingConfig {
                enabled: true,
                method: ClippingMethod::GlobalNorm,
                threshold: 1.0,
            },
        },
        training: TrainingParams {
            num_epochs: 3,  // Fewer epochs for fine-tuning
            batch_size: 8,  // Smaller batch size for larger models
            gradient_accumulation_steps: 8,  // More accumulation for effective batch size
            eval_steps: 250,
            save_steps: 500,
            log_steps: 50,
            max_steps: None,
            seed: 42,
            resume_from_checkpoint: None,
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                monitor_metric: "eval_accuracy".to_string(),
                patience: 5,
                min_delta: 0.001,
                higher_is_better: true,  // Accuracy should increase
            },
        },
        data: DataConfig {
            train_data_path: "data/supervised/train".to_string(),
            eval_data_path: Some("data/supervised/val".to_string()),
            cache_data: true,
            num_workers: 4,
            shuffle: true,
            drop_last: false,
        },
        checkpointing: CheckpointingConfig {
            save_dir: "checkpoints/supervised".to_string(),
            save_total_limit: 3,
            save_best_only: true,
            monitor_metric: "eval_accuracy".to_string(),
            mode: CheckpointMode::Max,  // Higher accuracy is better
            save_on_each_epoch: true,
            save_optimizer_state: true,
            save_scheduler_state: true,
            compression_level: 6,
        },
        logging: LoggingConfig {
            log_dir: "logs/supervised".to_string(),
            log_level: "info".to_string(),
            log_to_file: true,
            log_to_console: true,
            report_to: vec!["tensorboard".to_string(), "wandb".to_string()],
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
    
    // Create supervised trainer configuration
    let trainer_config = SupervisedTrainerConfig {
        base_config: base_config.clone(),
        base_model: BaseModelConfig {
            model_type: ModelType::Llama2_7B,
            checkpoint_path: Some("models/llama2-7b".to_string()),
            freeze_base: true,  // Freeze base model during training
            device_map: None,   // Let trainer handle device placement
        },
        supervised: SupervisedSettings {
            task_type: SupervisedTaskType::MultiTask {
                tasks: vec![
                    "sentiment_analysis".to_string(),
                    "named_entity_recognition".to_string(),
                    "question_answering".to_string(),
                ],
                task_weights: Some(vec![1.0, 1.5, 2.0]), // QA is more important
            },
            gradient_accumulation_mode: GradientAccumulationMode::Dynamic {
                min_steps: 4,
                max_steps: 16,
                scale_factor: 2.0,
            },
            task_balancing: TaskBalancingStrategy::DynamicWeighting {
                update_frequency: 100,
                smoothing_factor: 0.9,
            },
            auxiliary_objectives: vec![],  // Add auxiliary losses if needed
            use_gradient_checkpointing: true,
            max_sequence_length: 512,
        },
        lora_adaptation: LoraAdaptationConfig {
            apply_to_layers: vec![
                "self_attn.q_proj".to_string(),
                "self_attn.k_proj".to_string(),
                "self_attn.v_proj".to_string(),
                "self_attn.o_proj".to_string(),
                "mlp.gate_proj".to_string(),
                "mlp.up_proj".to_string(),
                "mlp.down_proj".to_string(),
            ],
            alpha: 32.0,  // LoRA scaling factor
            dropout: 0.05,
            merge_weights: false,  // Keep LoRA separate for flexibility
            init_strategy: lorax::training::trainers::supervised::LoraInitStrategy::Gaussian {
                std: 0.02,
            },
            rank_allocation: lorax::training::trainers::supervised::RankAllocation::Adaptive {
                min_rank: 8,
                max_rank: 64,
                importance_threshold: 0.1,
            },
        },
        validation: ValidationSettings {
            evaluation_mode: EvaluationMode::Full,  // Evaluate on full dataset
            compute_perplexity: true,
            compute_task_metrics: true,
            track_layer_metrics: true,
            best_model_metric: "eval_accuracy".to_string(),
            eval_batch_size: None,  // Use training batch size
            use_teacher_forcing: true,
            temperature: 1.0,
        },
        metrics: lorax::training::trainers::supervised::MetricsSettings {
            log_param_stats_interval: 100,
            log_gradient_flow_interval: 200,
            track_memory_usage: true,
            enable_profiling: false,
            track_activation_stats: true,
        },
    };
    
    // Load base model
    info!("Loading base model: {:?}", trainer_config.base_model.model_type);
    let base_model_path = trainer_config.base_model.checkpoint_path.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Base model checkpoint path required"))?;
    
    let base_model = create_base_model(
        trainer_config.base_model.model_type,
        base_model_path,
        device.clone(),
    )?;
    
    // Create datasets with task information
    info!("Loading datasets...");
    let train_dataset = SupervisedDataset::new(
        &base_config.data.train_data_path,
        device.clone(),
        Some(&trainer_config.supervised.task_type),
        base_config.data.cache_data,
        trainer_config.supervised.max_sequence_length,
    )?;
    
    let val_dataset = if let Some(ref eval_path) = base_config.data.eval_data_path {
        Some(SupervisedDataset::new(
            eval_path,
            device.clone(),
            Some(&trainer_config.supervised.task_type),
            base_config.data.cache_data,
            trainer_config.supervised.max_sequence_length,
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
                batch_size: trainer_config.validation.eval_batch_size
                    .unwrap_or(base_config.training.batch_size),
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
    info!("Creating supervised trainer...");
    let mut trainer = SupervisedTrainer::new(
        trainer_config,
        hypernetwork,
        Arc::new(RwLock::new(base_model)),
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
                    info!("========== Epoch {} Started ==========", epoch + 1);
                }
                lorax::training::TrainingEvent::EpochCompleted { epoch, metrics } => {
                    info!("========== Epoch {} Completed ==========", epoch + 1);
                    info!("Avg Loss: {:.4}", metrics.get("train_loss").unwrap_or(&0.0));
                    info!("Avg Accuracy: {:.2}%", metrics.get("train_accuracy").unwrap_or(&0.0) * 100.0);
                    if let Some(perplexity) = metrics.get("train_perplexity") {
                        info!("Perplexity: {:.2}", perplexity);
                    }
                }
                lorax::training::TrainingEvent::StepCompleted { step, loss, lr } => {
                    if step % 50 == 0 {
                        info!("Step {}: loss={:.4}, lr={:.2e}", step, loss, lr);
                    }
                }
                lorax::training::TrainingEvent::EvaluationCompleted { metrics } => {
                    info!("===== Evaluation Results =====");
                    info!("Eval Loss: {:.4}", metrics.get("eval_loss").unwrap_or(&0.0));
                    info!("Eval Accuracy: {:.2}%", metrics.get("eval_accuracy").unwrap_or(&0.0) * 100.0);
                    if let Some(perplexity) = metrics.get("eval_perplexity") {
                        info!("Eval Perplexity: {:.2}", perplexity);
                    }
                    // Log task-specific metrics
                    for (key, value) in metrics.iter() {
                        if key.contains("task_") {
                            info!("{}: {:.4}", key, value);
                        }
                    }
                }
                lorax::training::TrainingEvent::CheckpointSaved { path } => {
                    info!("✓ Checkpoint saved to: {:?}", path);
                }
                lorax::training::TrainingEvent::EarlyStopping { reason } => {
                    info!("⚠️  Early stopping triggered: {}", reason);
                }
                lorax::training::TrainingEvent::TrainingCompleted { total_steps } => {
                    info!("✅ Training completed successfully after {} steps", total_steps);
                }
                lorax::training::TrainingEvent::Error { error } => {
                    error!("❌ Training error: {}", error);
                }
            }
        }
    });
    
    // Optional: Load pretrained checkpoint if continuing training
    if let Some(checkpoint_path) = &base_config.training.resume_from_checkpoint {
        info!("Resuming from checkpoint: {}", checkpoint_path);
        trainer.load_checkpoint(checkpoint_path)?;
    }
    
    // Start training
    info!("Starting supervised fine-tuning...");
    let result = trainer.train().await?;
    
    // Print final results
    info!("\n========== Training Summary ==========");
    info!("Total steps: {}", result.total_steps);
    info!("Total epochs: {}", result.epochs_completed);
    info!("Training duration: {:?}", result.training_duration);
    info!("\nFinal Metrics:");
    for (metric, value) in result.final_metrics.iter() {
        info!("  {}: {:.4}", metric, value);
    }
    
    if let Some(best_checkpoint) = result.best_checkpoint_path {
        info!("\n✅ Best checkpoint saved at: {:?}", best_checkpoint);
    }
    
    // Example of using the trained model for inference
    info!("\n===== Testing Inference =====");
    // The trainer provides access to the trained model
    // You can now use it for generating LoRA parameters for new tasks
    
    Ok(())
}