//! Example of multi-task learning with T2L
//!
//! This example demonstrates how to train a hypernetwork to handle multiple
//! tasks simultaneously, with dynamic task balancing and specialized LoRA
//! generation for each task.

use anyhow::Result;
use candle_core::{Device, Tensor};
use lorax::hypernetwork::{HyperNetwork, HypernetworkConfig, ModelSize, TargetArchitecture};
use lorax::models::{ModelType, create_base_model};
use lorax::training::{
    SupervisedTrainer, SupervisedTrainerConfig,
    DataLoader, DataLoaderConfig,
    TrainingConfig, TrainingType, ModelConfig, ModelArchitecture,
    OptimizerConfig, OptimizerType, SchedulerType, SchedulerConfig,
    CheckpointingConfig, LoggingConfig, MixedPrecisionConfig,
    TrainingParams, DataConfig, GradientClippingConfig, ClippingMethod,
    EarlyStoppingConfig, CheckpointMode, SupervisedTaskType,
    RegularizationConfig, RegularizationType,
    PrecisionType, LossScalingConfig, LossScalingMethod,
};
use lorax::training::trainers::supervised::{
    BaseModelConfig, SupervisedSettings, LoraAdaptationConfig, ValidationSettings,
    GradientAccumulationMode, TaskBalancingStrategy, EvaluationMode,
    LoraInitStrategy, RankAllocation, AuxiliaryObjective,
};
use lorax::training::data::{MultiTaskDataset, TaskSampler, SamplingStrategy};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

/// Task configuration for multi-task learning
#[derive(Debug, Clone)]
struct TaskConfig {
    name: String,
    data_path: PathBuf,
    weight: f32,
    max_samples: Option<usize>,
    task_type: String,  // classification, generation, qa, etc.
    num_labels: Option<usize>,  // For classification tasks
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging with more detailed formatting
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::DEBUG)
        .with_line_number(true)
        .with_thread_ids(true)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("Starting multi-task learning example with T2L");
    
    // Set device
    let device = Device::cuda_if_available(0)?;
    info!("Using device: {:?}", device);
    
    // Define tasks for multi-task learning
    let tasks = vec![
        TaskConfig {
            name: "sentiment".to_string(),
            data_path: PathBuf::from("data/tasks/sentiment"),
            weight: 1.0,
            max_samples: Some(50000),
            task_type: "classification".to_string(),
            num_labels: Some(3),  // positive, negative, neutral
        },
        TaskConfig {
            name: "nli".to_string(),
            data_path: PathBuf::from("data/tasks/nli"),
            weight: 1.5,  // Natural Language Inference is harder
            max_samples: Some(100000),
            task_type: "classification".to_string(),
            num_labels: Some(3),  // entailment, contradiction, neutral
        },
        TaskConfig {
            name: "qa".to_string(),
            data_path: PathBuf::from("data/tasks/squad"),
            weight: 2.0,  // Question answering is most complex
            max_samples: None,
            task_type: "generation".to_string(),
            num_labels: None,
        },
        TaskConfig {
            name: "summarization".to_string(),
            data_path: PathBuf::from("data/tasks/cnn_dailymail"),
            weight: 1.8,
            max_samples: Some(30000),
            task_type: "generation".to_string(),
            num_labels: None,
        },
        TaskConfig {
            name: "ner".to_string(),
            data_path: PathBuf::from("data/tasks/conll2003"),
            weight: 1.2,
            max_samples: None,
            task_type: "token_classification".to_string(),
            num_labels: Some(9),  // B-PER, I-PER, B-ORG, etc.
        },
    ];
    
    // Create hypernetwork with task-aware configuration
    let hypernetwork_config = HypernetworkConfig {
        model_size: ModelSize::XLarge,  // Larger model for multi-task
        input_dim: 1024,  // Larger embedding for task diversity
        lora_rank: 64,    // Higher rank for task-specific adaptation
        dropout: 0.1,
        activation: lorax::hypernetwork::ActivationType::GELU,
        target_architecture: Some(TargetArchitecture::Llama),
        // Enable task-specific components
        task_embedding_dim: Some(256),
        num_tasks: Some(tasks.len()),
        use_task_embeddings: true,
        task_mixing_alpha: Some(0.3),  // Mix task embeddings with input
    };
    
    let hypernetwork = Arc::new(RwLock::new(HyperNetwork::new(hypernetwork_config)?));
    
    // Create base training configuration optimized for multi-task
    let base_config = TrainingConfig {
        model: ModelConfig {
            architecture: ModelArchitecture::Hypernetwork { 
                size: "xlarge".to_string() 
            },
            training_type: TrainingType::Supervised,
            checkpoint_path: None,
        },
        optimizer: OptimizerConfig {
            optimizer_type: OptimizerType::AdamW {
                betas: (0.9, 0.999),
                eps: 1e-8,
                weight_decay: 0.05,  // Stronger regularization for multi-task
            },
            learning_rate: 3e-5,
            scheduler: SchedulerConfig {
                scheduler_type: SchedulerType::PolynomialDecay {
                    num_warmup_steps: 2000,  // Longer warmup for task adaptation
                    num_training_steps: 50000,
                    power: 1.0,
                    end_lr: 1e-6,
                },
            },
            gradient_clipping: GradientClippingConfig {
                enabled: true,
                method: ClippingMethod::AdaptiveNorm {
                    percentile: 90.0,
                    history_size: 1000,
                },
                threshold: 1.0,
            },
        },
        training: TrainingParams {
            num_epochs: 5,
            batch_size: 4,  // Small batch per task
            gradient_accumulation_steps: 16,  // Large effective batch
            eval_steps: 500,
            save_steps: 1000,
            log_steps: 25,
            max_steps: Some(50000),
            seed: 42,
            resume_from_checkpoint: None,
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                monitor_metric: "eval_avg_task_performance".to_string(),
                patience: 10,
                min_delta: 0.0005,
                higher_is_better: true,
            },
        },
        data: DataConfig {
            train_data_path: "data/multi_task/train".to_string(),
            eval_data_path: Some("data/multi_task/val".to_string()),
            cache_data: true,
            num_workers: 8,  // More workers for multiple datasets
            shuffle: true,
            drop_last: false,
        },
        regularization: RegularizationConfig {
            l2_lambda: 0.01,
            l1_lambda: 0.0,
            dropout_rate: 0.1,
            label_smoothing: 0.1,
            gradient_penalty: None,
            spectral_norm: false,
            regularization_types: vec![
                RegularizationType::TaskOrthogonality {
                    lambda: 0.1,
                    apply_to: vec!["lora_weights".to_string()],
                },
                RegularizationType::ParameterDiversity {
                    lambda: 0.05,
                    temperature: 1.0,
                },
            ],
        },
        checkpointing: CheckpointingConfig {
            save_dir: "checkpoints/multi_task".to_string(),
            save_total_limit: 5,
            save_best_only: false,  // Save periodically for multi-task
            monitor_metric: "eval_avg_task_performance".to_string(),
            mode: CheckpointMode::Max,
            save_on_each_epoch: true,
            save_optimizer_state: true,
            save_scheduler_state: true,
            compression_level: 6,
        },
        logging: LoggingConfig {
            log_dir: "logs/multi_task".to_string(),
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
    
    // Create multi-task specific configuration
    let task_names: Vec<String> = tasks.iter().map(|t| t.name.clone()).collect();
    let task_weights: Vec<f32> = tasks.iter().map(|t| t.weight).collect();
    
    let trainer_config = SupervisedTrainerConfig {
        base_config: base_config.clone(),
        base_model: BaseModelConfig {
            model_type: ModelType::Llama2_13B,  // Larger model for multi-task
            checkpoint_path: Some("models/llama2-13b".to_string()),
            freeze_base: true,
            device_map: Some(HashMap::from([
                ("embeddings".to_string(), 0),
                ("layers.0-15".to_string(), 0),
                ("layers.16-31".to_string(), 1),
                ("layers.32-39".to_string(), 1),
                ("output".to_string(), 1),
            ])),  // Model parallelism for large model
        },
        supervised: SupervisedSettings {
            task_type: SupervisedTaskType::MultiTask {
                tasks: task_names.clone(),
                task_weights: Some(task_weights),
            },
            gradient_accumulation_mode: GradientAccumulationMode::TaskAdaptive {
                base_steps: 4,
                task_scaling: HashMap::from([
                    ("qa".to_string(), 2.0),          // More accumulation for complex tasks
                    ("summarization".to_string(), 1.5),
                    ("ner".to_string(), 1.2),
                    ("sentiment".to_string(), 0.8),   // Less for simple tasks
                    ("nli".to_string(), 1.0),
                ]),
            },
            task_balancing: TaskBalancingStrategy::UncertaintyWeighting {
                initial_weights: HashMap::from([
                    ("sentiment".to_string(), 1.0),
                    ("nli".to_string(), 1.5),
                    ("qa".to_string(), 2.0),
                    ("summarization".to_string(), 1.8),
                    ("ner".to_string(), 1.2),
                ]),
                update_frequency: 100,
                temperature: 2.0,
            },
            auxiliary_objectives: vec![
                AuxiliaryObjective::TaskSimilarity {
                    weight: 0.1,
                    distance_metric: "cosine".to_string(),
                },
                AuxiliaryObjective::GradientAlignment {
                    weight: 0.05,
                    target_alignment: 0.8,
                },
                AuxiliaryObjective::ParameterRegularization {
                    weight: 0.1,
                    reg_type: "l2".to_string(),
                },
            ],
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
            alpha: 64.0,  // Higher alpha for multi-task
            dropout: 0.05,
            merge_weights: false,
            init_strategy: LoraInitStrategy::TaskAware {
                base_std: 0.02,
                task_specific_std: HashMap::from([
                    ("qa".to_string(), 0.03),
                    ("summarization".to_string(), 0.025),
                    ("ner".to_string(), 0.02),
                    ("sentiment".to_string(), 0.015),
                    ("nli".to_string(), 0.02),
                ]),
            },
            rank_allocation: RankAllocation::TaskSpecific {
                base_rank: 32,
                task_ranks: HashMap::from([
                    ("qa".to_string(), 64),          // Higher rank for complex tasks
                    ("summarization".to_string(), 48),
                    ("ner".to_string(), 32),
                    ("sentiment".to_string(), 16),    // Lower rank for simple tasks
                    ("nli".to_string(), 32),
                ]),
                dynamic_adjustment: true,
            },
        },
        validation: ValidationSettings {
            evaluation_mode: EvaluationMode::TaskStratified {
                samples_per_task: 1000,
                balanced: true,
            },
            compute_perplexity: true,
            compute_task_metrics: true,
            track_layer_metrics: true,
            best_model_metric: "eval_avg_task_performance".to_string(),
            eval_batch_size: Some(8),
            use_teacher_forcing: true,
            temperature: 1.0,
            task_specific_metrics: HashMap::from([
                ("sentiment".to_string(), vec!["accuracy", "f1_macro"]),
                ("nli".to_string(), vec!["accuracy", "confusion_matrix"]),
                ("qa".to_string(), vec!["exact_match", "f1_score"]),
                ("summarization".to_string(), vec!["rouge1", "rouge2", "rougeL"]),
                ("ner".to_string(), vec!["entity_f1", "entity_precision", "entity_recall"]),
            ]),
        },
        metrics: lorax::training::trainers::supervised::MetricsSettings {
            log_param_stats_interval: 100,
            log_gradient_flow_interval: 200,
            track_memory_usage: true,
            enable_profiling: true,  // Profile multi-task performance
            track_activation_stats: true,
            task_specific_tracking: true,
            cross_task_metrics: true,
        },
    };
    
    // Create multi-task dataset with custom sampling
    info!("Creating multi-task datasets...");
    let mut train_datasets = HashMap::new();
    let mut val_datasets = HashMap::new();
    
    for task in &tasks {
        info!("Loading dataset for task: {}", task.name);
        
        // Create task-specific dataset
        let train_dataset = MultiTaskDataset::new(
            &task.data_path.join("train"),
            &task.name,
            &task.task_type,
            task.num_labels,
            device.clone(),
            trainer_config.supervised.max_sequence_length,
            task.max_samples,
        )?;
        
        let val_dataset = MultiTaskDataset::new(
            &task.data_path.join("val"),
            &task.name,
            &task.task_type,
            task.num_labels,
            device.clone(),
            trainer_config.supervised.max_sequence_length,
            Some(1000),  // Limit validation samples
        )?;
        
        train_datasets.insert(task.name.clone(), Arc::new(train_dataset));
        val_datasets.insert(task.name.clone(), Arc::new(val_dataset));
    }
    
    // Create task sampler for balanced multi-task training
    let task_sampler = TaskSampler::new(
        train_datasets.clone(),
        SamplingStrategy::ProportionalToSize {
            temperature: 0.75,  // Smooth size differences
            min_probability: 0.1,
        },
    )?;
    
    // Create multi-task data loaders
    let train_loader = DataLoader::multi_task(
        task_sampler,
        DataLoaderConfig {
            batch_size: base_config.training.batch_size,
            shuffle: true,
            drop_last: false,
            num_workers: base_config.data.num_workers,
            pin_memory: true,
            prefetch_factor: Some(4),  // More prefetch for multiple datasets
        },
        device.clone(),
    )?;
    
    let val_loader = DataLoader::multi_task_eval(
        val_datasets,
        DataLoaderConfig {
            batch_size: trainer_config.validation.eval_batch_size.unwrap_or(8),
            shuffle: false,
            drop_last: false,
            num_workers: base_config.data.num_workers,
            pin_memory: true,
            prefetch_factor: Some(2),
        },
        device.clone(),
    )?;
    
    // Load base model
    info!("Loading base model: {:?}", trainer_config.base_model.model_type);
    let base_model = create_base_model(
        trainer_config.base_model.model_type,
        trainer_config.base_model.checkpoint_path.as_ref().unwrap(),
        device.clone(),
    )?;
    
    // Create trainer
    info!("Creating multi-task trainer...");
    let mut trainer = SupervisedTrainer::new(
        trainer_config,
        hypernetwork,
        Arc::new(RwLock::new(base_model)),
        train_loader,
        Some(val_loader),
        device,
    )?;
    
    // Set up comprehensive event monitoring
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    trainer = trainer.with_event_monitoring(event_tx);
    
    // Task performance tracking
    let mut task_performances: HashMap<String, Vec<f32>> = HashMap::new();
    for task_name in &task_names {
        task_performances.insert(task_name.clone(), Vec::new());
    }
    
    // Spawn monitoring task
    let task_performances_clone = Arc::new(RwLock::new(task_performances));
    tokio::spawn(async move {
        let task_perfs = task_performances_clone;
        
        while let Some(event) = event_rx.recv().await {
            match event {
                lorax::training::TrainingEvent::EpochStarted { epoch } => {
                    info!("\n{'='*60}");
                    info!("EPOCH {} STARTED", epoch + 1);
                    info!("{'='*60}\n");
                }
                lorax::training::TrainingEvent::EpochCompleted { epoch, metrics } => {
                    info!("\n{'='*60}");
                    info!("EPOCH {} COMPLETED", epoch + 1);
                    info!("{'='*60}");
                    
                    // Log overall metrics
                    info!("\nOverall Metrics:");
                    info!("  Average Loss: {:.4}", metrics.get("train_loss").unwrap_or(&0.0));
                    info!("  Average Performance: {:.2}%", 
                        metrics.get("train_avg_task_performance").unwrap_or(&0.0) * 100.0);
                    
                    // Log task-specific metrics
                    info!("\nTask-Specific Performance:");
                    for task_name in &task_names {
                        let task_acc_key = format!("train_task_{}_accuracy", task_name);
                        let task_loss_key = format!("train_task_{}_loss", task_name);
                        
                        if let Some(acc) = metrics.get(&task_acc_key) {
                            info!("  {}: Accuracy={:.2}%, Loss={:.4}", 
                                task_name, 
                                acc * 100.0,
                                metrics.get(&task_loss_key).unwrap_or(&0.0)
                            );
                            
                            // Track performance history
                            let mut perfs = task_perfs.write();
                            perfs.get_mut(task_name).unwrap().push(*acc as f32);
                        }
                    }
                }
                lorax::training::TrainingEvent::StepCompleted { step, loss, lr } => {
                    if step % 25 == 0 {
                        info!("Step {}: loss={:.4}, lr={:.2e}", step, loss, lr);
                    }
                }
                lorax::training::TrainingEvent::EvaluationCompleted { metrics } => {
                    info!("\n{'='*40}");
                    info!("EVALUATION RESULTS");
                    info!("{'='*40}");
                    
                    // Overall evaluation metrics
                    info!("\nOverall Performance:");
                    info!("  Average Task Performance: {:.2}%",
                        metrics.get("eval_avg_task_performance").unwrap_or(&0.0) * 100.0);
                    info!("  Average Loss: {:.4}", metrics.get("eval_loss").unwrap_or(&0.0));
                    
                    // Task-specific evaluation
                    info!("\nTask-Specific Results:");
                    for task_name in &task_names {
                        info!("\n  {}:", task_name.to_uppercase());
                        
                        // Standard metrics
                        let acc_key = format!("eval_task_{}_accuracy", task_name);
                        let loss_key = format!("eval_task_{}_loss", task_name);
                        
                        if let Some(acc) = metrics.get(&acc_key) {
                            info!("    Accuracy: {:.2}%", acc * 100.0);
                            info!("    Loss: {:.4}", metrics.get(&loss_key).unwrap_or(&0.0));
                        }
                        
                        // Task-specific metrics
                        match task_name.as_str() {
                            "qa" => {
                                if let Some(em) = metrics.get(&format!("eval_task_{}_exact_match", task_name)) {
                                    info!("    Exact Match: {:.2}%", em * 100.0);
                                }
                                if let Some(f1) = metrics.get(&format!("eval_task_{}_f1_score", task_name)) {
                                    info!("    F1 Score: {:.2}%", f1 * 100.0);
                                }
                            }
                            "summarization" => {
                                for rouge_type in ["rouge1", "rouge2", "rougeL"] {
                                    let key = format!("eval_task_{}_{}", task_name, rouge_type);
                                    if let Some(score) = metrics.get(&key) {
                                        info!("    {}: {:.2}", rouge_type.to_uppercase(), score);
                                    }
                                }
                            }
                            "ner" => {
                                if let Some(f1) = metrics.get(&format!("eval_task_{}_entity_f1", task_name)) {
                                    info!("    Entity F1: {:.2}%", f1 * 100.0);
                                }
                            }
                            _ => {}
                        }
                    }
                    
                    // Task balance analysis
                    info!("\n  Task Weight Distribution:");
                    for task_name in &task_names {
                        let weight_key = format!("task_weight_{}", task_name);
                        if let Some(weight) = metrics.get(&weight_key) {
                            info!("    {}: {:.3}", task_name, weight);
                        }
                    }
                }
                lorax::training::TrainingEvent::CheckpointSaved { path } => {
                    info!("✓ Checkpoint saved: {:?}", path);
                }
                lorax::training::TrainingEvent::EarlyStopping { reason } => {
                    warn!("⚠️  Early stopping: {}", reason);
                }
                lorax::training::TrainingEvent::TrainingCompleted { total_steps } => {
                    info!("\n{'='*60}");
                    info!("✅ TRAINING COMPLETED SUCCESSFULLY");
                    info!("{'='*60}");
                    info!("Total steps: {}", total_steps);
                    
                    // Print performance evolution
                    let perfs = task_perfs.read();
                    info!("\nTask Performance Evolution:");
                    for (task_name, history) in perfs.iter() {
                        if !history.is_empty() {
                            let start = history.first().unwrap();
                            let end = history.last().unwrap();
                            let improvement = (end - start) * 100.0;
                            info!("  {}: {:.2}% → {:.2}% ({:+.2}%)",
                                task_name,
                                start * 100.0,
                                end * 100.0,
                                improvement
                            );
                        }
                    }
                }
                lorax::training::TrainingEvent::Error { error } => {
                    error!("❌ Training error: {}", error);
                }
            }
        }
    });
    
    // Start training
    info!("\nStarting multi-task training with {} tasks...", tasks.len());
    let start_time = std::time::Instant::now();
    let result = trainer.train().await?;
    
    // Print comprehensive final summary
    info!("\n\n{'='*80}");
    info!("TRAINING SUMMARY");
    info!("{'='*80}");
    info!("Total duration: {:?}", start_time.elapsed());
    info!("Total steps: {}", result.total_steps);
    info!("Total epochs: {}", result.epochs_completed);
    info!("\nFinal Performance:");
    
    // Group metrics by task
    let mut task_results: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (metric_name, value) in result.final_metrics.iter() {
        if metric_name.contains("task_") {
            let parts: Vec<&str> = metric_name.split('_').collect();
            if parts.len() >= 3 {
                let task_name = parts[2];
                let metric_type = parts[3..].join("_");
                task_results
                    .entry(task_name.to_string())
                    .or_insert_with(HashMap::new)
                    .insert(metric_type, *value);
            }
        }
    }
    
    // Print task results
    for (task_name, metrics) in task_results.iter() {
        info!("\n  {}:", task_name.to_uppercase());
        for (metric_type, value) in metrics.iter() {
            if metric_type.contains("accuracy") || metric_type.contains("f1") {
                info!("    {}: {:.2}%", metric_type, value * 100.0);
            } else {
                info!("    {}: {:.4}", metric_type, value);
            }
        }
    }
    
    if let Some(best_checkpoint) = result.best_checkpoint_path {
        info!("\n✅ Best model checkpoint: {:?}", best_checkpoint);
    }
    
    Ok(())
}