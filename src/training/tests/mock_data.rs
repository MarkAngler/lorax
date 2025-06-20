//! Mock data generation for testing

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

use crate::training::data::{
    Dataset, DataSample, SampleData, DatasetMetadata, LoraDatasetConfig,
    TaskInfo, ReconstructionDataset, SupervisedDataset, ReconstructionBatch,
    SupervisedBatch,
};

/// Mock dataset for testing
pub struct MockDataset {
    samples: Vec<DataSample>,
    metadata: DatasetMetadata,
}

impl MockDataset {
    pub fn new(num_samples: usize, dataset_type: DatasetType) -> Self {
        let samples = (0..num_samples)
            .map(|i| match dataset_type {
                DatasetType::Reconstruction => create_reconstruction_sample(i),
                DatasetType::Supervised => create_supervised_sample(i),
            })
            .collect();
        
        let metadata = DatasetMetadata {
            name: format!("mock_{:?}_dataset", dataset_type),
            num_samples,
            lora_config: match dataset_type {
                DatasetType::Reconstruction => Some(LoraDatasetConfig {
                    layer_names: vec!["layer_0".to_string(), "layer_1".to_string()],
                    rank: 4,
                    alpha: 16.0,
                    layer_dims: HashMap::from([
                        ("layer_0".to_string(), (64, 64)),
                        ("layer_1".to_string(), (64, 64)),
                    ]),
                }),
                DatasetType::Supervised => None,
            },
            task_info: Some(TaskInfo {
                task_type: match dataset_type {
                    DatasetType::Reconstruction => "reconstruction".to_string(),
                    DatasetType::Supervised => "classification".to_string(),
                },
                num_classes: Some(10),
                vocab_size: Some(1000),
                max_seq_len: Some(128),
            }),
        };
        
        Self { samples, metadata }
    }
}

impl Dataset for MockDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }
    
    fn get(&self, index: usize) -> Result<DataSample> {
        self.samples.get(index)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Index {} out of bounds", index))
    }
    
    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DatasetType {
    Reconstruction,
    Supervised,
}

fn create_reconstruction_sample(index: usize) -> DataSample {
    let device = Device::Cpu;
    let mut lora_params = HashMap::new();
    
    // Create mock LoRA parameters
    for i in 0..2 {
        let layer_name = format!("layer_{}", i);
        let a = Tensor::randn(0f32, 1f32, &[64, 4], &device).unwrap();
        let b = Tensor::randn(0f32, 1f32, &[4, 64], &device).unwrap();
        lora_params.insert(layer_name, (a, b));
    }
    
    DataSample {
        id: format!("sample_{}", index),
        task_description: format!("Task description for sample {}", index),
        task_embeddings: Some(
            Tensor::randn(0f32, 1f32, &[64], &device).unwrap()
        ),
        data: SampleData::Reconstruction { lora_params },
    }
}

fn create_supervised_sample(index: usize) -> DataSample {
    let device = Device::Cpu;
    
    DataSample {
        id: format!("sample_{}", index),
        task_description: format!("Task description for sample {}", index),
        task_embeddings: Some(
            Tensor::randn(0f32, 1f32, &[64], &device).unwrap()
        ),
        data: SampleData::Supervised {
            input_text: format!("Input text for sample {}", index),
            target_text: Some(format!("Target text for sample {}", index)),
            labels: Some(
                Tensor::from_vec(vec![index as f32 % 10.0], &[1], &device).unwrap()
            ),
        },
    }
}

/// Create a mock dataset for testing
pub fn create_mock_dataset(num_samples: usize, dataset_type: DatasetType) -> Arc<dyn Dataset> {
    Arc::new(MockDataset::new(num_samples, dataset_type))
}

/// Create a mock reconstruction batch
pub fn create_mock_batch(batch_size: usize, device: &Device) -> Result<ReconstructionBatch> {
    let task_embeddings = Tensor::randn(0f32, 1f32, &[batch_size, 64], device)?;
    
    let mut target_params = HashMap::new();
    let mut layer_mask = HashMap::new();
    
    for i in 0..2 {
        let layer_name = format!("layer_{}", i);
        let a = Tensor::randn(0f32, 1f32, &[batch_size, 64, 4], device)?;
        let b = Tensor::randn(0f32, 1f32, &[batch_size, 4, 64], device)?;
        target_params.insert(layer_name.clone(), (a, b));
        
        // All layers active in mock data
        layer_mask.insert(layer_name, Tensor::ones(&[batch_size], DType::F32, device)?);
    }
    
    Ok(ReconstructionBatch {
        task_embeddings,
        target_params,
        layer_mask: Some(layer_mask),
        sample_ids: (0..batch_size).map(|i| format!("sample_{}", i)).collect(),
        metadata: HashMap::new(),
    })
}

/// Create mock LoRA parameters
pub fn create_mock_lora_params(
    num_layers: usize,
    hidden_size: usize,
    rank: usize,
    device: &Device,
) -> Result<HashMap<String, (Tensor, Tensor)>> {
    let mut params = HashMap::new();
    
    for i in 0..num_layers {
        let layer_name = format!("layer_{}", i);
        let a = Tensor::randn(0f32, 1f32, &[hidden_size, rank], device)?;
        let b = Tensor::randn(0f32, 1f32, &[rank, hidden_size], device)?;
        params.insert(layer_name, (a, b));
    }
    
    Ok(params)
}

/// Create mock embeddings
pub fn create_mock_embeddings(
    batch_size: usize,
    hidden_size: usize,
    device: &Device,
) -> Result<Tensor> {
    Tensor::randn(0f32, 1f32, &[batch_size, hidden_size], device)
}

/// Create a mock supervised batch
pub fn create_mock_supervised_batch(
    batch_size: usize,
    seq_length: usize,
    vocab_size: usize,
    device: &Device,
) -> Result<SupervisedBatch> {
    use rand::{thread_rng, Rng};
    let mut rng = thread_rng();
    
    // Create random input IDs
    let input_ids_vec: Vec<u32> = (0..batch_size * seq_length)
        .map(|_| rng.gen_range(0..vocab_size) as u32)
        .collect();
    let input_ids = Tensor::from_vec(
        input_ids_vec,
        &[batch_size, seq_length],
        device,
    )?;
    
    // Create attention mask (all ones for simplicity)
    let attention_mask = Tensor::ones(
        &[batch_size, seq_length],
        DType::F32,
        device,
    )?;
    
    // Create labels (shifted input_ids)
    let labels = input_ids.clone();
    
    // Create task embeddings
    let task_embeddings = Tensor::randn(0f32, 1f32, &[batch_size, 64], device)?;
    
    Ok(SupervisedBatch {
        input_ids,
        attention_mask,
        labels: Some(labels),
        task_embeddings,
        sample_ids: (0..batch_size).map(|i| format!("sample_{}", i)).collect(),
        metadata: HashMap::new(),
    })
}

/// Mock data loader for testing
pub struct MockDataLoader {
    dataset: Arc<dyn Dataset>,
    batch_size: usize,
    current_idx: Arc<RwLock<usize>>,
}

impl MockDataLoader {
    pub fn new(dataset: Arc<dyn Dataset>, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            current_idx: Arc::new(RwLock::new(0)),
        }
    }
    
    pub fn next_batch(&self, device: &Device) -> Result<Option<ReconstructionBatch>> {
        let mut idx = self.current_idx.write();
        
        if *idx >= self.dataset.len() {
            *idx = 0;
            return Ok(None);
        }
        
        let end_idx = (*idx + self.batch_size).min(self.dataset.len());
        let batch_size = end_idx - *idx;
        
        // Create batch from dataset samples
        let batch = create_mock_batch(batch_size, device)?;
        
        *idx = end_idx;
        Ok(Some(batch))
    }
    
    pub fn reset(&self) {
        *self.current_idx.write() = 0;
    }
}