//! Error types for the LoRAX system

use thiserror::Error;

/// Main error type for LoRAX operations
#[derive(Error, Debug)]
pub enum Error {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Encoder error
    #[error("Encoder error: {0}")]
    Encoder(String),
    
    /// Hypernetwork error
    #[error("Hypernetwork error: {0}")]
    Hypernetwork(String),
    
    /// LoRA generation error
    #[error("LoRA generation error: {0}")]
    LoraGeneration(String),
    
    /// Tensor operation error
    #[error("Tensor operation error: {0}")]
    Tensor(#[from] candle_core::Error),
    
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Model loading error
    #[error("Model loading error: {0}")]
    ModelLoading(String),
    
    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Resource exhausted
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    
    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
    
    /// Other errors
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result type alias for LoRAX operations
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create a configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
    
    /// Create an encoder error
    pub fn encoder(msg: impl Into<String>) -> Self {
        Self::Encoder(msg.into())
    }
    
    /// Create a hypernetwork error
    pub fn hypernetwork(msg: impl Into<String>) -> Self {
        Self::Hypernetwork(msg.into())
    }
    
    /// Create a LoRA generation error
    pub fn lora_generation(msg: impl Into<String>) -> Self {
        Self::LoraGeneration(msg.into())
    }
    
    /// Create a model loading error
    pub fn model_loading(msg: impl Into<String>) -> Self {
        Self::ModelLoading(msg.into())
    }
    
    /// Create an invalid input error
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }
    
    /// Create a resource exhausted error
    pub fn resource_exhausted(msg: impl Into<String>) -> Self {
        Self::ResourceExhausted(msg.into())
    }
    
    /// Create an internal error
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}