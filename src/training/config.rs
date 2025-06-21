//! Training configuration system for T2L models
//!
//! This module provides comprehensive configuration structures for training,
//! including model parameters, optimization settings, regularization options,
//! and checkpointing configuration.

use std::path::PathBuf;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use crate::training::loss::LossConfig;

/// Main training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Model configuration
    pub model: ModelConfig,
    
    /// Optimization configuration
    pub optimizer: OptimizerConfig,
    
    /// Training parameters
    pub training: TrainingParams,
    
    /// Data configuration
    pub data: DataConfig,
    
    /// Checkpointing configuration
    pub checkpointing: CheckpointingConfig,
    
    /// Logging and monitoring configuration
    pub logging: LoggingConfig,
    
    /// Mixed precision training settings
    pub mixed_precision: MixedPrecisionConfig,
    
    /// Regularization settings
    pub regularization: RegularizationConfig,
    
    /// Runtime configuration
    pub runtime: RuntimeConfig,
    
    /// Loss function configuration
    pub loss: LossConfig,
}

/// Model-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Type of training (reconstruction or supervised)
    pub training_type: TrainingType,
    
    /// LoRA rank for adapters
    pub lora_rank: usize,
    
    /// LoRA alpha parameter
    pub lora_alpha: f64,
    
    /// Dropout rate for LoRA layers
    pub lora_dropout: f64,
    
    /// Target modules for LoRA adaptation
    pub target_modules: Vec<String>,
    
    /// Model architecture type
    pub architecture: ModelArchitecture,
    
    /// Maximum sequence length
    pub max_sequence_length: usize,
    
    /// Hidden dimension size
    pub hidden_size: usize,
    
    /// Number of attention heads
    pub num_attention_heads: usize,
    
    /// Number of layers
    pub num_layers: usize,
    
    /// Vocabulary size
    pub vocab_size: usize,
}

/// Training type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainingType {
    /// Reconstruction training for text-to-LoRA
    Reconstruction,
    /// Supervised fine-tuning
    Supervised,
    /// Multi-task training
    MultiTask {
        tasks: Vec<String>,
        task_weights: Vec<f64>,
    },
}

/// Supported model architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelArchitecture {
    Llama,
    Mistral,
    Gemma,
    Bert,
    Custom { name: String },
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Optimizer type
    pub optimizer_type: OptimizerType,
    
    /// Base learning rate
    pub learning_rate: f64,
    
    /// Learning rate scheduler
    pub scheduler: SchedulerConfig,
    
    /// Weight decay coefficient
    pub weight_decay: f64,
    
    /// Beta1 parameter for Adam-based optimizers
    pub beta1: f64,
    
    /// Beta2 parameter for Adam-based optimizers
    pub beta2: f64,
    
    /// Epsilon for numerical stability
    pub epsilon: f64,
    
    /// Gradient clipping settings
    pub gradient_clipping: GradientClippingConfig,
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerType {
    AdamW,
    Adam,
    SGD { momentum: f64 },
    RMSprop { alpha: f64 },
    Custom { name: String },
}

/// Learning rate scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduler type
    pub scheduler_type: SchedulerType,
    
    /// Warmup steps
    pub warmup_steps: usize,
    
    /// Total training steps (for cosine annealing)
    pub total_steps: Option<usize>,
    
    /// Decay factor for step/exponential schedulers
    pub decay_factor: f64,
    
    /// Step size for step scheduler
    pub step_size: usize,
    
    /// Minimum learning rate
    pub min_lr: f64,
}

/// Learning rate scheduler types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerType {
    Linear,
    Cosine,
    CosineWithRestarts { restart_period: usize },
    Exponential,
    StepLR,
    ReduceOnPlateau { patience: usize, factor: f64 },
    Constant,
}

/// Gradient clipping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientClippingConfig {
    /// Enable gradient clipping
    pub enabled: bool,
    
    /// Clipping method
    pub method: ClippingMethod,
    
    /// Clipping threshold
    pub threshold: f64,
}

/// Gradient clipping methods
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClippingMethod {
    /// Clip by global norm
    GlobalNorm,
    /// Clip by value
    Value,
    /// Adaptive clipping
    Adaptive { percentile: f64 },
}

/// Core training parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    /// Number of training epochs
    pub num_epochs: usize,
    
    /// Batch size for training
    pub batch_size: usize,
    
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    
    /// Evaluation frequency (in steps)
    pub eval_steps: usize,
    
    /// Logging frequency (in steps)
    pub log_steps: usize,
    
    /// Save frequency (in steps)
    pub save_steps: usize,
    
    /// Maximum number of steps (overrides epochs if set)
    pub max_steps: Option<usize>,
    
    /// Early stopping configuration
    pub early_stopping: EarlyStoppingConfig,
    
    /// Seed for reproducibility
    pub seed: u64,
    
    /// Resume from checkpoint path
    pub resume_from_checkpoint: Option<PathBuf>,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Enable early stopping
    pub enabled: bool,
    
    /// Patience (number of evaluations without improvement)
    pub patience: usize,
    
    /// Minimum improvement threshold
    pub min_delta: f64,
    
    /// Metric to monitor for early stopping
    pub monitor_metric: String,
    
    /// Whether higher values are better for the metric
    pub higher_is_better: bool,
}

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Training data path
    pub train_data_path: PathBuf,
    
    /// Validation data path
    pub val_data_path: Option<PathBuf>,
    
    /// Test data path
    pub test_data_path: Option<PathBuf>,
    
    /// Data format
    pub data_format: DataFormat,
    
    /// Maximum number of training samples
    pub max_train_samples: Option<usize>,
    
    /// Maximum number of validation samples
    pub max_val_samples: Option<usize>,
    
    /// Data preprocessing settings
    pub preprocessing: PreprocessingConfig,
    
    /// Data loading settings
    pub loading: DataLoadingConfig,
}

/// Supported data formats
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataFormat {
    Json,
    Jsonl,
    Parquet,
    Csv,
    HuggingFace { dataset_name: String },
    Custom { format: String },
}

/// Data preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Tokenizer settings
    pub tokenizer: TokenizerConfig,
    
    /// Text normalization settings
    pub normalization: NormalizationConfig,
    
    /// Data augmentation settings
    pub augmentation: AugmentationConfig,
}

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Tokenizer model path or name
    pub model_name: String,
    
    /// Add special tokens
    pub add_special_tokens: bool,
    
    /// Padding strategy
    pub padding: PaddingStrategy,
    
    /// Truncation strategy
    pub truncation: TruncationStrategy,
    
    /// Return attention mask
    pub return_attention_mask: bool,
}

/// Padding strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaddingStrategy {
    MaxLength,
    LongestFirst,
    DoNotPad,
}

/// Truncation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruncationStrategy {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
    DoNotTruncate,
}

/// Text normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    /// Lowercase text
    pub lowercase: bool,
    
    /// Strip accents
    pub strip_accents: bool,
    
    /// Remove extra whitespace
    pub strip_whitespace: bool,
    
    /// Normalize unicode
    pub unicode_normalization: Option<String>,
}

/// Data augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    /// Enable augmentation
    pub enabled: bool,
    
    /// Augmentation probability
    pub probability: f64,
    
    /// Augmentation techniques
    pub techniques: Vec<AugmentationTechnique>,
}

/// Data augmentation techniques
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AugmentationTechnique {
    RandomMask { mask_prob: f64 },
    RandomSwap { swap_prob: f64 },
    Synonym { replacement_prob: f64 },
    BackTranslation { languages: Vec<String> },
}

/// Data loading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoadingConfig {
    /// Number of data loading workers
    pub num_workers: usize,
    
    /// Prefetch factor
    pub prefetch_factor: usize,
    
    /// Pin memory for GPU training
    pub pin_memory: bool,
    
    /// Drop last incomplete batch
    pub drop_last: bool,
    
    /// Shuffle training data
    pub shuffle: bool,
}

/// Checkpointing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointingConfig {
    /// Output directory for checkpoints
    pub output_dir: PathBuf,
    
    /// Save only the best model
    pub save_best_only: bool,
    
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,
    
    /// Metric to use for best model selection
    pub best_metric: String,
    
    /// Whether higher values are better for the metric
    pub best_metric_higher_is_better: bool,
    
    /// Save optimizer state
    pub save_optimizer_state: bool,
    
    /// Save scheduler state
    pub save_scheduler_state: bool,
    
    /// Checkpoint format
    pub format: CheckpointFormat,
}

/// Checkpoint formats
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckpointFormat {
    Safetensors,
    Pytorch,
    Candle,
}

/// Logging and monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    
    /// Enable Weights & Biases logging
    pub wandb: WandbConfig,
    
    /// Enable TensorBoard logging
    pub tensorboard: TensorBoardConfig,
    
    /// Prometheus metrics
    pub prometheus: PrometheusConfig,
    
    /// Custom logging handlers
    pub handlers: Vec<LogHandler>,
}

/// Log levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Weights & Biases configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WandbConfig {
    /// Enable W&B logging
    pub enabled: bool,
    
    /// Project name
    pub project: Option<String>,
    
    /// Run name
    pub name: Option<String>,
    
    /// Tags
    pub tags: Vec<String>,
    
    /// Entity (team/user)
    pub entity: Option<String>,
}

/// TensorBoard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorBoardConfig {
    /// Enable TensorBoard logging
    pub enabled: bool,
    
    /// Log directory
    pub log_dir: PathBuf,
    
    /// Update frequency
    pub update_freq: usize,
}

/// Prometheus metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusConfig {
    /// Enable Prometheus metrics
    pub enabled: bool,
    
    /// Metrics port
    pub port: u16,
    
    /// Metrics endpoint
    pub endpoint: String,
}

/// Custom log handlers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogHandler {
    /// Handler name
    pub name: String,
    
    /// Handler type
    pub handler_type: String,
    
    /// Handler configuration
    pub config: serde_json::Value,
}

/// Mixed precision training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionConfig {
    /// Enable mixed precision training
    pub enabled: bool,
    
    /// Precision type
    pub precision: PrecisionType,
    
    /// Loss scaling
    pub loss_scaling: LossScalingConfig,
    
    /// Gradient scaling
    pub gradient_scaling: bool,
}

/// Precision types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrecisionType {
    FP16,
    BF16,
    FP32,
}

/// Loss scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossScalingConfig {
    /// Scaling method
    pub method: LossScalingMethod,
    
    /// Initial scale
    pub init_scale: f64,
    
    /// Growth factor
    pub growth_factor: f64,
    
    /// Backoff factor
    pub backoff_factor: f64,
    
    /// Growth interval
    pub growth_interval: usize,
}

/// Loss scaling methods
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossScalingMethod {
    Static,
    Dynamic,
}

/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// Dropout rate
    pub dropout: f64,
    
    /// Attention dropout rate
    pub attention_dropout: f64,
    
    /// Hidden dropout rate
    pub hidden_dropout: f64,
    
    /// Label smoothing
    pub label_smoothing: f64,
    
    /// Layer dropout (stochastic depth)
    pub layer_dropout: f64,
    
    /// Spectral normalization
    pub spectral_norm: bool,
    
    /// L1 regularization
    pub l1_reg: f64,
    
    /// L2 regularization
    pub l2_reg: f64,
}

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Device configuration
    pub device: DeviceConfig,
    
    /// Parallelization settings
    pub parallelization: ParallelizationConfig,
    
    /// Memory optimization
    pub memory: MemoryConfig,
    
    /// Compilation settings
    pub compilation: CompilationConfig,
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Device type
    pub device_type: DeviceType,
    
    /// Device IDs to use
    pub device_ids: Vec<usize>,
    
    /// Enable automatic device selection
    pub auto_select: bool,
}

/// Device types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    CPU,
    CUDA,
    Metal,
    Vulkan,
}

/// Parallelization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelizationConfig {
    /// Data parallel strategy
    pub data_parallel: bool,
    
    /// Model parallel strategy
    pub model_parallel: bool,
    
    /// Pipeline parallel stages
    pub pipeline_parallel_stages: Option<usize>,
    
    /// Number of threads for CPU operations
    pub num_threads: Option<usize>,
}

/// Memory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Enable gradient checkpointing
    pub gradient_checkpointing: bool,
    
    /// Memory efficient attention
    pub memory_efficient_attention: bool,
    
    /// Maximum memory usage (in GB)
    pub max_memory_gb: Option<f64>,
    
    /// Enable memory profiling
    pub profile_memory: bool,
}

/// Compilation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationConfig {
    /// Enable compilation optimizations
    pub enabled: bool,
    
    /// Compilation backend
    pub backend: CompilationBackend,
    
    /// Optimization level
    pub optimization_level: usize,
}

/// Compilation backends
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompilationBackend {
    Default,
    LLVM,
    TensorRT,
    Custom { name: String },
}

impl TrainingConfig {
    /// Create a new training configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Load configuration from a file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .context("Failed to read configuration file")?;
        
        let config = if path.as_ref().extension().and_then(|s| s.to_str()) == Some("json") {
            serde_json::from_str(&content)
                .context("Failed to parse JSON configuration")?
        } else {
            serde_yaml::from_str(&content)
                .context("Failed to parse YAML configuration")?
        };
        
        Ok(config)
    }
    
    /// Save configuration to a file
    pub fn to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let content = if path.as_ref().extension().and_then(|s| s.to_str()) == Some("json") {
            serde_json::to_string_pretty(self)
                .context("Failed to serialize configuration to JSON")?
        } else {
            serde_yaml::to_string(self)
                .context("Failed to serialize configuration to YAML")?
        };
        
        std::fs::write(path.as_ref(), content)
            .context("Failed to write configuration file")?;
        
        Ok(())
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate model configuration
        if self.model.lora_rank == 0 {
            return Err(anyhow::anyhow!("LoRA rank must be greater than 0"));
        }
        
        if self.model.lora_alpha <= 0.0 {
            return Err(anyhow::anyhow!("LoRA alpha must be positive"));
        }
        
        if self.model.lora_dropout < 0.0 || self.model.lora_dropout >= 1.0 {
            return Err(anyhow::anyhow!("LoRA dropout must be in [0.0, 1.0)"));
        }
        
        // Validate training parameters
        if self.training.num_epochs == 0 {
            return Err(anyhow::anyhow!("Number of epochs must be greater than 0"));
        }
        
        if self.training.batch_size == 0 {
            return Err(anyhow::anyhow!("Batch size must be greater than 0"));
        }
        
        if self.training.gradient_accumulation_steps == 0 {
            return Err(anyhow::anyhow!("Gradient accumulation steps must be greater than 0"));
        }
        
        // Validate optimizer configuration
        if self.optimizer.learning_rate <= 0.0 {
            return Err(anyhow::anyhow!("Learning rate must be positive"));
        }
        
        if self.optimizer.weight_decay < 0.0 {
            return Err(anyhow::anyhow!("Weight decay must be non-negative"));
        }
        
        // Validate data configuration
        if !self.data.train_data_path.exists() {
            return Err(anyhow::anyhow!("Training data path does not exist"));
        }
        
        // Validate checkpointing configuration
        if self.checkpointing.max_checkpoints == 0 {
            return Err(anyhow::anyhow!("Max checkpoints must be greater than 0"));
        }
        
        // Validate multi-task configuration
        if let TrainingType::MultiTask { tasks, task_weights } = &self.model.training_type {
            if tasks.len() != task_weights.len() {
                return Err(anyhow::anyhow!("Number of tasks must match number of task weights"));
            }
            
            if task_weights.iter().any(|&w| w <= 0.0) {
                return Err(anyhow::anyhow!("All task weights must be positive"));
            }
        }
        
        Ok(())
    }
    
    /// Get effective batch size (batch_size * gradient_accumulation_steps)
    pub fn effective_batch_size(&self) -> usize {
        self.training.batch_size * self.training.gradient_accumulation_steps
    }
    
    /// Get total training steps
    pub fn total_steps(&self) -> Option<usize> {
        if let Some(max_steps) = self.training.max_steps {
            Some(max_steps)
        } else {
            // This would need to be calculated based on dataset size
            None
        }
    }

    /// Create a default configuration for reconstruction training
    pub fn reconstruction_default() -> Self {
        let mut config = Self::default();
        config.model.training_type = TrainingType::Reconstruction;
        config
    }
    
    /// Create a default configuration for supervised training
    pub fn supervised_default() -> Self {
        let mut config = Self::default();
        config.model.training_type = TrainingType::Supervised;
        config
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            optimizer: OptimizerConfig::default(),
            training: TrainingParams::default(),
            data: DataConfig::default(),
            checkpointing: CheckpointingConfig::default(),
            logging: LoggingConfig::default(),
            mixed_precision: MixedPrecisionConfig::default(),
            regularization: RegularizationConfig::default(),
            runtime: RuntimeConfig::default(),
            loss: LossConfig::default(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            training_type: TrainingType::Reconstruction,
            lora_rank: 16,
            lora_alpha: 32.0,
            lora_dropout: 0.1,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            architecture: ModelArchitecture::Llama,
            max_sequence_length: 2048,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_layers: 32,
            vocab_size: 32000,
        }
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::AdamW,
            learning_rate: 2e-4,
            scheduler: SchedulerConfig::default(),
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            gradient_clipping: GradientClippingConfig::default(),
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            scheduler_type: SchedulerType::Linear,
            warmup_steps: 100,
            total_steps: None,
            decay_factor: 0.1,
            step_size: 100,
            min_lr: 1e-6,
        }
    }
}

impl Default for GradientClippingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            method: ClippingMethod::GlobalNorm,
            threshold: 1.0,
        }
    }
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            num_epochs: 3,
            batch_size: 8,
            gradient_accumulation_steps: 1,
            eval_steps: 500,
            log_steps: 10,
            save_steps: 1000,
            max_steps: None,
            early_stopping: EarlyStoppingConfig::default(),
            seed: 42,
            resume_from_checkpoint: None,
        }
    }
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            patience: 3,
            min_delta: 0.001,
            monitor_metric: "eval_loss".to_string(),
            higher_is_better: false,
        }
    }
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            train_data_path: PathBuf::from("data/train.jsonl"),
            val_data_path: Some(PathBuf::from("data/val.jsonl")),
            test_data_path: None,
            data_format: DataFormat::Jsonl,
            max_train_samples: None,
            max_val_samples: None,
            preprocessing: PreprocessingConfig::default(),
            loading: DataLoadingConfig::default(),
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            tokenizer: TokenizerConfig::default(),
            normalization: NormalizationConfig::default(),
            augmentation: AugmentationConfig::default(),
        }
    }
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            model_name: "gpt2".to_string(),
            add_special_tokens: true,
            padding: PaddingStrategy::MaxLength,
            truncation: TruncationStrategy::LongestFirst,
            return_attention_mask: true,
        }
    }
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            lowercase: false,
            strip_accents: false,
            strip_whitespace: true,
            unicode_normalization: None,
        }
    }
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            probability: 0.1,
            techniques: vec![],
        }
    }
}

impl Default for DataLoadingConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            prefetch_factor: 2,
            pin_memory: true,
            drop_last: true,
            shuffle: true,
        }
    }
}

impl Default for CheckpointingConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./checkpoints"),
            save_best_only: false,
            max_checkpoints: 3,
            best_metric: "eval_loss".to_string(),
            best_metric_higher_is_better: false,
            save_optimizer_state: true,
            save_scheduler_state: true,
            format: CheckpointFormat::Safetensors,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            wandb: WandbConfig::default(),
            tensorboard: TensorBoardConfig::default(),
            prometheus: PrometheusConfig::default(),
            handlers: vec![],
        }
    }
}

impl Default for WandbConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            project: None,
            name: None,
            tags: vec![],
            entity: None,
        }
    }
}

impl Default for TensorBoardConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            log_dir: PathBuf::from("./logs"),
            update_freq: 100,
        }
    }
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            port: 9090,
            endpoint: "/metrics".to_string(),
        }
    }
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            precision: PrecisionType::FP16,
            loss_scaling: LossScalingConfig::default(),
            gradient_scaling: true,
        }
    }
}

impl Default for LossScalingConfig {
    fn default() -> Self {
        Self {
            method: LossScalingMethod::Dynamic,
            init_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
        }
    }
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            dropout: 0.1,
            attention_dropout: 0.1,
            hidden_dropout: 0.1,
            label_smoothing: 0.0,
            layer_dropout: 0.0,
            spectral_norm: false,
            l1_reg: 0.0,
            l2_reg: 0.0,
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            device: DeviceConfig::default(),
            parallelization: ParallelizationConfig::default(),
            memory: MemoryConfig::default(),
            compilation: CompilationConfig::default(),
        }
    }
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::CPU,
            device_ids: vec![0],
            auto_select: true,
        }
    }
}

impl Default for ParallelizationConfig {
    fn default() -> Self {
        Self {
            data_parallel: false,
            model_parallel: false,
            pipeline_parallel_stages: None,
            num_threads: None,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            gradient_checkpointing: false,
            memory_efficient_attention: false,
            max_memory_gb: None,
            profile_memory: false,
        }
    }
}

impl Default for CompilationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: CompilationBackend::Default,
            optimization_level: 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = TrainingConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let config = TrainingConfig::default();
        
        // Test JSON serialization
        let json_str = serde_json::to_string(&config).unwrap();
        let deserialized: TrainingConfig = serde_json::from_str(&json_str).unwrap();
        assert_eq!(config.model.lora_rank, deserialized.model.lora_rank);
        
        // Test YAML serialization
        let yaml_str = serde_yaml::to_string(&config).unwrap();
        let deserialized: TrainingConfig = serde_yaml::from_str(&yaml_str).unwrap();
        assert_eq!(config.optimizer.learning_rate, deserialized.optimizer.learning_rate);
    }

    #[test]
    fn test_config_file_operations() {
        let config = TrainingConfig::default();
        
        // Test JSON file
        let json_file = NamedTempFile::new().unwrap();
        config.to_file(json_file.path().with_extension("json")).unwrap();
        let loaded_config = TrainingConfig::from_file(json_file.path().with_extension("json")).unwrap();
        assert_eq!(config.training.batch_size, loaded_config.training.batch_size);
        
        // Test YAML file
        let yaml_file = NamedTempFile::new().unwrap();
        config.to_file(yaml_file.path().with_extension("yaml")).unwrap();
        let loaded_config = TrainingConfig::from_file(yaml_file.path().with_extension("yaml")).unwrap();
        assert_eq!(config.training.num_epochs, loaded_config.training.num_epochs);
    }

    #[test]
    fn test_config_validation() {
        let mut config = TrainingConfig::default();
        
        // Valid config should pass
        assert!(config.validate().is_ok());
        
        // Invalid LoRA rank
        config.model.lora_rank = 0;
        assert!(config.validate().is_err());
        
        // Reset and test invalid learning rate
        config = TrainingConfig::default();
        config.optimizer.learning_rate = -0.1;
        assert!(config.validate().is_err());
        
        // Reset and test invalid batch size
        config = TrainingConfig::default();
        config.training.batch_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_effective_batch_size() {
        let mut config = TrainingConfig::default();
        config.training.batch_size = 4;
        config.training.gradient_accumulation_steps = 8;
        assert_eq!(config.effective_batch_size(), 32);
    }

    #[test]
    fn test_specialized_configs() {
        let reconstruction_config = TrainingConfig::reconstruction_default();
        match reconstruction_config.model.training_type {
            TrainingType::Reconstruction => {},
            _ => panic!("Expected reconstruction training type"),
        }
        
        let supervised_config = TrainingConfig::supervised_default();
        match supervised_config.model.training_type {
            TrainingType::Supervised => {},
            _ => panic!("Expected supervised training type"),
        }
    }
}