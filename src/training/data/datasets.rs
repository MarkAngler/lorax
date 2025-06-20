//! Dataset implementations for T2L training

use super::{
    Dataset, DataSample, SampleData, DatasetMetadata, LoraDatasetConfig, TaskInfo, DataError
};
use anyhow::Result;
use candle_core::{Tensor, Device};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn, instrument};

/// Dataset for reconstruction training with LoRA parameters
pub struct ReconstructionDataset {
    /// Dataset metadata
    metadata: DatasetMetadata,
    /// Path to dataset directory
    data_path: PathBuf,
    /// Path to task embeddings (optional)
    embeddings_path: Option<PathBuf>,
    /// Device for tensor operations
    device: Device,
    /// In-memory cache for loaded samples
    cache: Arc<RwLock<HashMap<usize, DataSample>>>,
    /// Whether to cache all samples in memory
    cache_in_memory: bool,
    /// Sample information loaded from metadata
    samples: Vec<SampleInfo>,
}

/// Information about a single sample
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SampleInfo {
    /// Sample ID
    id: String,
    /// Task description
    description: String,
    /// File path relative to dataset directory
    file_path: String,
    /// Optional metadata
    metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Metadata file structure for reconstruction datasets
#[derive(Debug, Serialize, Deserialize)]
struct ReconstructionMetadata {
    /// Dataset name
    name: String,
    /// Dataset version
    version: String,
    /// Sample information
    samples: Vec<SampleInfo>,
    /// LoRA configuration
    lora_config: LoraDatasetConfig,
    /// Task information
    task_info: Option<TaskInfo>,
    /// Creation timestamp
    created_at: chrono::DateTime<chrono::Utc>,
}

impl ReconstructionDataset {
    /// Create a new reconstruction dataset
    pub fn new(
        data_path: impl AsRef<Path>,
        device: Device,
        embeddings_path: Option<impl AsRef<Path>>,
        cache_in_memory: bool,
    ) -> Result<Self> {
        let data_path = data_path.as_ref().to_path_buf();
        let embeddings_path = embeddings_path.map(|p| p.as_ref().to_path_buf());
        
        info!("Loading reconstruction dataset from {:?}", data_path);
        
        // Load metadata
        let metadata_path = data_path.join("metadata.json");
        if !metadata_path.exists() {
            return Err(DataError::DatasetNotFound { 
                path: metadata_path.to_string_lossy().to_string() 
            }.into());
        }
        
        let metadata_str = std::fs::read_to_string(&metadata_path)?;
        let reconstruction_metadata: ReconstructionMetadata = serde_json::from_str(&metadata_str)?;
        
        let metadata = DatasetMetadata {
            name: reconstruction_metadata.name.clone(),
            num_samples: reconstruction_metadata.samples.len(),
            lora_config: Some(reconstruction_metadata.lora_config.clone()),
            task_info: reconstruction_metadata.task_info.clone(),
        };
        
        info!(
            "Loaded reconstruction dataset '{}' with {} samples",
            metadata.name,
            metadata.num_samples
        );
        
        Ok(Self {
            metadata,
            data_path,
            embeddings_path,
            device,
            cache: Arc::new(RwLock::new(HashMap::new())),
            cache_in_memory,
            samples: reconstruction_metadata.samples,
        })
    }
    
    /// Load LoRA parameters from HDF5 file
    #[instrument(skip(self))]
    fn load_lora_params(&self, file_path: &Path) -> Result<HashMap<String, (Tensor, Tensor)>> {
        debug!("Loading LoRA parameters from {:?}", file_path);
        
        let file = hdf5::File::open(file_path)?;
        let mut lora_params = HashMap::new();
        
        let lora_config = self.metadata.lora_config.as_ref()
            .ok_or_else(|| DataError::MissingMetadata { field: "lora_config".to_string() })?;
        
        for layer_name in &lora_config.layer_names {
            // Load A matrix
            let a_dataset = file.dataset(&format!("{}/A", layer_name))?;
            let a_data: ndarray::Array2<f32> = a_dataset.read()?;
            let a_tensor = Tensor::from_slice(
                a_data.as_slice().unwrap(),
                a_data.dim(),
                &self.device
            )?;
            
            // Load B matrix
            let b_dataset = file.dataset(&format!("{}/B", layer_name))?;
            let b_data: ndarray::Array2<f32> = b_dataset.read()?;
            let b_tensor = Tensor::from_slice(
                b_data.as_slice().unwrap(),
                b_data.dim(),
                &self.device
            )?;
            
            lora_params.insert(layer_name.clone(), (a_tensor, b_tensor));
        }
        
        debug!("Loaded {} LoRA parameter sets", lora_params.len());
        Ok(lora_params)
    }
    
    /// Load task embeddings if available
    #[instrument(skip(self))]
    fn load_task_embeddings(&self, sample_id: &str) -> Result<Option<Tensor>> {
        if let Some(embeddings_path) = &self.embeddings_path {
            let embedding_file = embeddings_path.join(format!("{}.bin", sample_id));
            if embedding_file.exists() {
                debug!("Loading task embeddings for sample {}", sample_id);
                let data = std::fs::read(&embedding_file)?;
                let embeddings: Vec<f32> = bincode::deserialize(&data)?;
                let tensor = Tensor::from_slice(&embeddings, embeddings.len(), &self.device)?;
                return Ok(Some(tensor));
            }
        }
        Ok(None)
    }
}

impl Dataset for ReconstructionDataset {
    fn len(&self) -> usize {
        self.metadata.num_samples
    }
    
    #[instrument(skip(self))]
    fn get(&self, index: usize) -> Result<DataSample> {
        if index >= self.len() {
            return Err(DataError::InvalidIndex { 
                index, 
                dataset_size: self.len() 
            }.into());
        }
        
        // Check cache first
        if let Some(sample) = self.cache.read().get(&index) {
            debug!("Retrieved sample {} from cache", index);
            return Ok(sample.clone());
        }
        
        let sample_info = &self.samples[index];
        debug!("Loading sample {} ({})", index, sample_info.id);
        
        // Load LoRA parameters
        let lora_file_path = self.data_path.join(&sample_info.file_path);
        let lora_params = self.load_lora_params(&lora_file_path)
            .map_err(|e| DataError::MalformedData { 
                index, 
                reason: format!("Failed to load LoRA parameters: {}", e) 
            })?;
        
        // Load task embeddings if available
        let task_embeddings = self.load_task_embeddings(&sample_info.id)?;
        
        let sample = DataSample {
            id: sample_info.id.clone(),
            task_description: sample_info.description.clone(),
            task_embeddings,
            data: SampleData::Reconstruction { lora_params },
        };
        
        // Cache if enabled
        if self.cache_in_memory {
            self.cache.write().insert(index, sample.clone());
        }
        
        Ok(sample)
    }
    
    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }
}

/// Dataset for supervised training with text data
pub struct SupervisedDataset {
    /// Dataset metadata
    metadata: DatasetMetadata,
    /// Path to dataset file
    data_path: PathBuf,
    /// Device for tensor operations
    device: Device,
    /// Task type
    task_type: String,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Samples loaded from file
    samples: Vec<SupervisedSample>,
    /// In-memory cache
    cache: Arc<RwLock<HashMap<usize, DataSample>>>,
}

/// A single supervised training sample
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SupervisedSample {
    /// Sample ID
    id: String,
    /// Task description
    task_description: String,
    /// Input text
    input_text: String,
    /// Target text (for generation tasks)
    target_text: Option<String>,
    /// Labels (for classification tasks)
    labels: Option<Vec<i64>>,
    /// Additional metadata
    metadata: Option<HashMap<String, serde_json::Value>>,
}

impl SupervisedDataset {
    /// Create a new supervised dataset
    pub fn new(
        data_path: impl AsRef<Path>,
        device: Device,
        task_type: String,
        max_seq_len: usize,
    ) -> Result<Self> {
        let data_path = data_path.as_ref().to_path_buf();
        
        info!("Loading supervised dataset from {:?}", data_path);
        
        if !data_path.exists() {
            return Err(DataError::DatasetNotFound { 
                path: data_path.to_string_lossy().to_string() 
            }.into());
        }
        
        // Load samples from JSONL file
        let file_content = std::fs::read_to_string(&data_path)?;
        let samples: Vec<SupervisedSample> = file_content
            .lines()
            .enumerate()
            .map(|(i, line)| {
                serde_json::from_str(line)
                    .map_err(|e| DataError::MalformedData { 
                        index: i, 
                        reason: format!("Invalid JSON: {}", e) 
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        
        // Determine task info
        let num_classes = if task_type == "classification" {
            samples.iter()
                .filter_map(|s| s.labels.as_ref())
                .flat_map(|labels| labels.iter())
                .map(|&label| label as usize)
                .max()
                .map(|max_label| max_label + 1)
        } else {
            None
        };
        
        let task_info = TaskInfo {
            task_type: task_type.clone(),
            num_classes,
            vocab_size: None, // Will be set by tokenizer
            max_seq_len: Some(max_seq_len),
        };
        
        let metadata = DatasetMetadata {
            name: data_path.file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            num_samples: samples.len(),
            lora_config: None,
            task_info: Some(task_info),
        };
        
        info!(
            "Loaded supervised dataset '{}' with {} samples",
            metadata.name,
            metadata.num_samples
        );
        
        Ok(Self {
            metadata,
            data_path,
            device,
            task_type,
            max_seq_len,
            samples,
            cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Convert labels to tensor
    fn labels_to_tensor(&self, labels: &[i64]) -> Result<Tensor> {
        Ok(Tensor::from_slice(labels, labels.len(), &self.device)?)
    }
}

impl Dataset for SupervisedDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }
    
    #[instrument(skip(self))]
    fn get(&self, index: usize) -> Result<DataSample> {
        if index >= self.len() {
            return Err(DataError::InvalidIndex { 
                index, 
                dataset_size: self.len() 
            }.into());
        }
        
        // Check cache first
        if let Some(sample) = self.cache.read().get(&index) {
            debug!("Retrieved sample {} from cache", index);
            return Ok(sample.clone());
        }
        
        let supervised_sample = &self.samples[index];
        debug!("Loading sample {} ({})", index, supervised_sample.id);
        
        // Convert labels to tensor if present
        let labels = if let Some(ref label_data) = supervised_sample.labels {
            Some(self.labels_to_tensor(label_data)?)
        } else {
            None
        };
        
        let sample = DataSample {
            id: supervised_sample.id.clone(),
            task_description: supervised_sample.task_description.clone(),
            task_embeddings: None, // Will be computed during training
            data: SampleData::Supervised {
                input_text: supervised_sample.input_text.clone(),
                target_text: supervised_sample.target_text.clone(),
                labels,
            },
        };
        
        // Cache the sample
        self.cache.write().insert(index, sample.clone());
        
        Ok(sample)
    }
    
    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    
    fn create_test_reconstruction_metadata() -> ReconstructionMetadata {
        ReconstructionMetadata {
            name: "test_dataset".to_string(),
            version: "1.0".to_string(),
            samples: vec![
                SampleInfo {
                    id: "sample_001".to_string(),
                    description: "Test task description".to_string(),
                    file_path: "sample_001.h5".to_string(),
                    metadata: None,
                },
            ],
            lora_config: LoraDatasetConfig {
                layer_names: vec!["layer1".to_string()],
                rank: 16,
                alpha: 32.0,
                layer_dims: {
                    let mut dims = HashMap::new();
                    dims.insert("layer1".to_string(), (512, 512));
                    dims
                },
            },
            task_info: None,
            created_at: chrono::Utc::now(),
        }
    }
    
    #[test]
    fn test_reconstruction_dataset_creation() {
        let temp_dir = TempDir::new().unwrap();
        let metadata = create_test_reconstruction_metadata();
        
        // Write metadata file
        let metadata_path = temp_dir.path().join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&metadata).unwrap();
        fs::write(&metadata_path, metadata_json).unwrap();
        
        // Create dataset
        let device = Device::Cpu;
        let dataset = ReconstructionDataset::new(
            temp_dir.path(),
            device,
            None::<PathBuf>,
            false,
        );
        
        assert!(dataset.is_ok());
        let dataset = dataset.unwrap();
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.metadata().name, "test_dataset");
    }
    
    #[test]
    fn test_supervised_dataset_creation() {
        let temp_dir = TempDir::new().unwrap();
        let data_path = temp_dir.path().join("test_data.jsonl");
        
        // Create test data
        let sample = SupervisedSample {
            id: "sample_001".to_string(),
            task_description: "Classify sentiment".to_string(),
            input_text: "This is a great movie!".to_string(),
            target_text: None,
            labels: Some(vec![1]), // Positive sentiment
            metadata: None,
        };
        
        let sample_json = serde_json::to_string(&sample).unwrap();
        fs::write(&data_path, sample_json).unwrap();
        
        // Create dataset
        let device = Device::Cpu;
        let dataset = SupervisedDataset::new(
            data_path,
            device,
            "classification".to_string(),
            512,
        );
        
        assert!(dataset.is_ok());
        let dataset = dataset.unwrap();
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.metadata().task_info.as_ref().unwrap().task_type, "classification");
    }
}