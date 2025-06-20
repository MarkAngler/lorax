//! Data loading infrastructure for T2L training
//!
//! This module provides dataset abstractions, data loaders, and batch collation
//! functions for both reconstruction and supervised training modes.

use anyhow::Result;
use candle_core::Tensor;
use std::collections::HashMap;

pub mod datasets;
pub mod loaders;
pub mod batching;

#[cfg(test)]
mod test_module;

// Re-exports
pub use datasets::{ReconstructionDataset, SupervisedDataset};
pub use loaders::{DataLoader, DataLoaderConfig};
pub use batching::{BatchCollator, ReconstructionBatch, SupervisedBatch};

/// Common trait for all training datasets
pub trait Dataset: Send + Sync {
    /// Get the number of samples in the dataset
    fn len(&self) -> usize;
    
    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get a sample by index
    fn get(&self, index: usize) -> Result<DataSample>;
    
    /// Get dataset metadata
    fn metadata(&self) -> &DatasetMetadata;
}

/// Represents a single training sample
#[derive(Debug, Clone)]
pub struct DataSample {
    /// Unique identifier for the sample
    pub id: String,
    /// Task description text
    pub task_description: String,
    /// Task embeddings (if pre-computed)
    pub task_embeddings: Option<Tensor>,
    /// Sample-specific data
    pub data: SampleData,
}

/// Sample-specific data for different training modes
#[derive(Debug, Clone)]
pub enum SampleData {
    /// LoRA parameters for reconstruction training
    Reconstruction {
        lora_params: HashMap<String, (Tensor, Tensor)>, // layer_name -> (A, B)
    },
    /// Text data for supervised training
    Supervised {
        input_text: String,
        target_text: Option<String>,
        labels: Option<Tensor>,
    },
}

/// Dataset metadata
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    /// Dataset name/identifier
    pub name: String,
    /// Number of samples
    pub num_samples: usize,
    /// LoRA configuration (for reconstruction datasets)
    pub lora_config: Option<LoraDatasetConfig>,
    /// Task information
    pub task_info: Option<TaskInfo>,
}

/// LoRA-specific dataset configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoraDatasetConfig {
    /// Layer names with LoRA parameters
    pub layer_names: Vec<String>,
    /// LoRA rank
    pub rank: usize,
    /// LoRA alpha scaling factor
    pub alpha: f32,
    /// Parameter dimensions per layer
    pub layer_dims: HashMap<String, (usize, usize)>, // layer_name -> (input_dim, output_dim)
}

/// Task information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskInfo {
    /// Task type (classification, qa, generation, etc.)
    pub task_type: String,
    /// Number of classes (for classification)
    pub num_classes: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// Maximum sequence length
    pub max_seq_len: Option<usize>,
}

/// Error types for data loading
#[derive(thiserror::Error, Debug)]
pub enum DataError {
    #[error("Dataset not found: {path}")]
    DatasetNotFound { path: String },
    
    #[error("Invalid sample index: {index} >= {dataset_size}")]
    InvalidIndex { index: usize, dataset_size: usize },
    
    #[error("Malformed data at index {index}: {reason}")]
    MalformedData { index: usize, reason: String },
    
    #[error("Missing metadata: {field}")]
    MissingMetadata { field: String },
    
    #[error("HDF5 error: {0}")]
    Hdf5Error(#[from] hdf5::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Tensor error: {0}")]
    TensorError(#[from] candle_core::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Batch collation error: {reason}")]
    BatchCollationError { reason: String },
}