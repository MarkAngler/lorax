//! Configuration structures for the LoRAX system

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::error::{Error, Result};
use crate::hypernetwork::TargetArchitecture;

/// Main configuration for the Text-to-LoRA system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Encoder configuration
    pub encoder: EncoderConfig,
    /// Projection layer configuration
    pub projection: ProjectionConfig,
    /// Hypernetwork configuration
    pub hypernetwork: HypernetworkConfig,
    /// LoRA generation configuration
    pub lora: LoraConfig,
    /// System-wide settings
    pub system: SystemConfig,
}

impl Config {
    /// Load configuration from a file
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate encoder config
        if self.encoder.embedding_dim == 0 {
            return Err(Error::config("Encoder embedding dimension must be > 0"));
        }
        
        // Validate projection config
        if self.projection.input_dim != self.encoder.embedding_dim {
            return Err(Error::config(
                "Projection input dim must match encoder embedding dim"
            ));
        }
        
        // Validate hypernetwork config
        if self.hypernetwork.input_dim != self.projection.output_dim {
            return Err(Error::config(
                "Hypernetwork input dim must match projection output dim"
            ));
        }
        
        // Validate LoRA config
        if self.lora.rank == 0 || self.lora.rank > 64 {
            return Err(Error::config("LoRA rank must be between 1 and 64"));
        }
        
        Ok(())
    }
    
    /// Create default configuration
    pub fn default() -> Self {
        Self {
            encoder: EncoderConfig::default(),
            projection: ProjectionConfig::default(),
            hypernetwork: HypernetworkConfig::default(),
            lora: LoraConfig::default(),
            system: SystemConfig::default(),
        }
    }
}

/// Encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Type of encoder to use
    pub encoder_type: EncoderType,
    /// Model name or path
    pub model_name: String,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Pooling strategy
    pub pooling_strategy: PoolingStrategy,
    /// Cache directory for models
    pub cache_dir: Option<PathBuf>,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            encoder_type: EncoderType::Bert,
            model_name: "bert-base-uncased".to_string(),
            embedding_dim: 768,
            max_sequence_length: 512,
            pooling_strategy: PoolingStrategy::Mean,
            cache_dir: None,
        }
    }
}

/// Encoder type options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncoderType {
    /// BERT encoder
    Bert,
    /// RoBERTa encoder
    Roberta,
    /// DistilBERT encoder
    Distilbert,
    /// Custom encoder
    Custom(String),
}

/// Pooling strategy for encoder outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PoolingStrategy {
    /// Use CLS token
    Cls,
    /// Mean pooling
    Mean,
    /// Max pooling
    Max,
    /// Mean of first and last layers
    MeanFirstLast,
}

/// Projection layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function
    pub activation: Activation,
    /// Dropout rate
    pub dropout: f32,
    /// Use batch normalization
    pub use_batch_norm: bool,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            input_dim: 768,
            output_dim: 256,
            activation: Activation::Relu,
            dropout: 0.1,
            use_batch_norm: true,
        }
    }
}

/// Activation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    /// ReLU activation
    Relu,
    /// GELU activation
    Gelu,
    /// SiLU/Swish activation
    Silu,
    /// Tanh activation
    Tanh,
    /// No activation
    None,
}

/// Hypernetwork configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypernetworkConfig {
    /// Model size variant
    pub model_size: ModelSize,
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimensions for each layer
    pub hidden_dims: Vec<usize>,
    /// Output dimension (total LoRA parameters)
    pub output_dim: usize,
    /// Activation function
    pub activation: Activation,
    /// Dropout rate
    pub dropout: f32,
    /// Use residual connections
    pub use_residual: bool,
    /// Layer normalization
    pub use_layer_norm: bool,
    /// LoRA rank for generated adapters
    pub lora_rank: usize,
    /// Target model architecture
    pub target_architecture: TargetArchitecture,
}

impl Default for HypernetworkConfig {
    fn default() -> Self {
        Self {
            model_size: ModelSize::Small,
            input_dim: 256,
            hidden_dims: vec![256, 128],
            output_dim: 4096,  // For rank-8 LoRA on 512-dim model
            activation: Activation::Gelu,
            dropout: 0.1,
            use_residual: true,
            use_layer_norm: true,
            lora_rank: 8,
            target_architecture: TargetArchitecture::GPT {
                hidden_size: 768,
                num_layers: 12,
                num_heads: 12,
            },
        }
    }
}

/// Model size variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelSize {
    /// Small model (<1B parameters)
    Small,
    /// Medium model (1B-7B parameters)
    Medium,
    /// Large model (7B+ parameters)
    Large,
}

impl ModelSize {
    /// Get recommended hidden dimensions
    pub fn hidden_dims(&self) -> Vec<usize> {
        match self {
            ModelSize::Small => vec![256, 128],
            ModelSize::Medium => vec![512, 256, 128],
            ModelSize::Large => vec![1024, 512, 256, 128],
        }
    }
    
    /// Get recommended parameter count
    pub fn param_count(&self) -> usize {
        match self {
            ModelSize::Small => 2_000_000,
            ModelSize::Medium => 5_000_000,
            ModelSize::Large => 10_000_000,
        }
    }
    
    /// Convert to hypernetwork module's ModelSize
    pub fn to_hypernetwork_model_size(&self) -> crate::hypernetwork::ModelSize {
        match self {
            ModelSize::Small => crate::hypernetwork::ModelSize::Small,
            ModelSize::Medium => crate::hypernetwork::ModelSize::Medium,
            ModelSize::Large => crate::hypernetwork::ModelSize::Large,
        }
    }
}

impl Activation {
    /// Convert to hypernetwork module's ActivationType
    pub fn to_hypernetwork_activation(&self) -> crate::hypernetwork::ActivationType {
        match self {
            Activation::Relu => crate::hypernetwork::ActivationType::ReLU,
            Activation::Gelu => crate::hypernetwork::ActivationType::GELU,
            Activation::Silu => crate::hypernetwork::ActivationType::SiLU,
            _ => crate::hypernetwork::ActivationType::ReLU, // Default for unsupported types
        }
    }
    
    /// Convert to projection module's ActivationType
    pub fn to_projection_activation(&self) -> crate::projection::ActivationType {
        match self {
            Activation::Relu => crate::projection::ActivationType::ReLU,
            Activation::Gelu => crate::projection::ActivationType::GELU,
            Activation::Silu => crate::projection::ActivationType::SiLU,
            Activation::Tanh => crate::projection::ActivationType::Tanh,
            Activation::None => crate::projection::ActivationType::Identity,
        }
    }
}

/// LoRA generation configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoraConfig {
    /// LoRA rank
    pub rank: usize,
    /// Alpha scaling factor
    pub alpha: f32,
    /// Target modules (e.g., ["q_proj", "v_proj"])
    pub target_modules: Vec<String>,
    /// Model dimension
    pub model_dim: usize,
    /// Initialization strategy
    pub init_strategy: InitStrategy,
    /// Enable bias terms
    pub use_bias: bool,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            model_dim: 512,
            init_strategy: InitStrategy::Kaiming,
            use_bias: false,
        }
    }
}

/// Weight initialization strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum InitStrategy {
    /// Kaiming/He initialization
    Kaiming,
    /// Xavier/Glorot initialization
    Xavier,
    /// Normal distribution
    Normal { mean: f32, std: f32 },
    /// Uniform distribution
    Uniform { low: f32, high: f32 },
    /// Zero initialization
    Zero,
}

/// System-wide configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Number of worker threads
    pub num_threads: usize,
    /// Batch size for inference
    pub batch_size: usize,
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics port
    pub metrics_port: u16,
    /// Log level
    pub log_level: String,
    /// Model cache size in MB
    pub model_cache_size_mb: usize,
    /// Device configuration
    pub device: DeviceConfig,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            batch_size: 32,
            max_concurrent_requests: 100,
            enable_metrics: true,
            metrics_port: 9090,
            log_level: "info".to_string(),
            model_cache_size_mb: 1024,
            device: DeviceConfig::default(),
        }
    }
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Device type
    pub device_type: DeviceType,
    /// Device ID (for multi-GPU)
    pub device_id: usize,
    /// Enable mixed precision
    pub mixed_precision: bool,
    /// Memory fraction to use (0.0-1.0)
    pub memory_fraction: f32,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            device_id: 0,
            mixed_precision: false,
            memory_fraction: 0.9,
        }
    }
}

/// Device types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    /// CPU device
    Cpu,
    /// CUDA GPU
    Cuda,
    /// Metal (Apple Silicon)
    Metal,
    /// Intel/AMD accelerators
    Accelerate,
}