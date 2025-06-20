use colored::*;
use std::fmt;
use thiserror::Error;

pub type CliResult<T> = Result<T, CliError>;

#[derive(Error, Debug)]
pub enum CliError {
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    #[error("Training error: {0}")]
    Training(String),
    
    #[error("Generation error: {0}")]
    Generation(String),
    
    #[error("Evaluation error: {0}")]
    Evaluation(String),
    
    #[error("Server error: {0}")]
    Server(String),
    
    #[error("Model error: {0}")]
    Model(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Network error: {0}")]
    Network(String),

    // Additional T2L-specific errors
    #[error("File already exists: {0}")]
    FileExists(std::path::PathBuf),

    #[error("Dataset not found: {0}")]
    DatasetNotFound(std::path::PathBuf),

    #[error("Model not found: {0}")]
    ModelNotFound(std::path::PathBuf),

    #[error("Invalid variant: {0}. Must be one of: L, M, S")]
    InvalidVariant(String),

    #[error("Invalid LoRA rank: {0}. Must be between 1 and 512")]
    InvalidRank(usize),

    #[error("Invalid learning rate: {0}. Must be between 0.0 and 1.0")]
    InvalidLearningRate(f64),

    #[error("Invalid validation split: {0}. Must be between 0.0 and 1.0")]
    InvalidValidationSplit(f64),

    #[error("Invalid batch size: {0}. Must be greater than 0")]
    InvalidBatchSize(usize),

    #[error("Invalid temperature: {0}. Must be between 0.0 and 2.0")]
    InvalidTemperature(f32),

    #[error("Invalid top-p: {0}. Must be between 0.0 and 1.0")]
    InvalidTopP(f32),

    #[error("No benchmarks specified")]
    NoBenchmarksSpecified,

    #[error("Configuration file not found: {0}")]
    ConfigNotFound(std::path::PathBuf),

    #[error("Invalid configuration key: {0}")]
    InvalidConfigKey(String),

    #[error("Training failed: {0}")]
    TrainingFailed(String),

    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    #[error("Evaluation failed: {0}")]
    EvaluationFailed(String),

    #[error("Model loading failed: {0}")]
    ModelLoadingFailed(String),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Memory allocation error: {0}")]
    MemoryError(String),

    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Hardware not supported: {0}")]
    HardwareNotSupported(String),

    #[error("Feature not available: {0}")]
    FeatureNotAvailable(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    // Inference-specific errors
    #[error("Adapter not found: {0}")]
    AdapterNotFound(std::path::PathBuf),

    #[error("Invalid max tokens: {0}. Must be between 1 and 4096")]
    InvalidMaxTokens(usize),

    #[error("Invalid repetition penalty: {0}. Must be between 0.0 and 2.0")]
    InvalidRepetitionPenalty(f32),

    #[error("History file not found: {0}")]
    HistoryFileNotFound(std::path::PathBuf),

    #[error("Batch file not found: {0}")]
    BatchFileNotFound(std::path::PathBuf),

    #[error("Empty batch file: {0}")]
    EmptyBatchFile(std::path::PathBuf),

    #[error("CUDA not available")]
    CudaNotAvailable,
    
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl CliError {
    /// Print a user-friendly error message
    pub fn print_error(&self) {
        eprintln!("{} {}", "Error:".red().bold(), self);
        
        // Add helpful suggestions based on error type
        match self {
            CliError::Config(msg) if msg.contains("not found") => {
                eprintln!(
                    "\n{} Run {} to create a default configuration file",
                    "Hint:".yellow(),
                    "t2l config init".cyan()
                );
            }
            CliError::FileNotFound(path) => {
                eprintln!(
                    "\n{} Make sure the file exists and the path is correct: {}",
                    "Hint:".yellow(),
                    path.cyan()
                );
            }
            CliError::InvalidArgument(msg) => {
                eprintln!(
                    "\n{} Use {} for more information",
                    "Hint:".yellow(),
                    "t2l --help".cyan()
                );
            }
            CliError::Server(_) => {
                eprintln!(
                    "\n{} Check that the port is not already in use",
                    "Hint:".yellow()
                );
            }
            _ => {}
        }
    }
}

impl From<serde_json::Error> for CliError {
    fn from(err: serde_json::Error) -> Self {
        CliError::Serialization(err.to_string())
    }
}

impl From<serde_yaml::Error> for CliError {
    fn from(err: serde_yaml::Error) -> Self {
        CliError::Serialization(err.to_string())
    }
}

impl From<toml::de::Error> for CliError {
    fn from(err: toml::de::Error) -> Self {
        CliError::Config(err.to_string())
    }
}

impl From<config::ConfigError> for CliError {
    fn from(err: config::ConfigError) -> Self {
        CliError::Config(err.to_string())
    }
}