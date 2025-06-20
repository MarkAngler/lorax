//! Batch collation functions for T2L training

use super::{DataSample, SampleData, DataError};
use anyhow::Result;
use candle_core::{Tensor, Device};
use std::collections::HashMap;
use std::any::Any;
use tracing::{debug, warn, instrument};

/// Trait for batch collation
pub trait BatchCollator: Send + Sync {
    /// Collate a batch of samples
    fn collate_batch(&self, samples: Vec<DataSample>) -> Result<Box<dyn Any + Send>, DataError>;
    
    /// Clone the collator (for multi-threading)
    fn clone_box(&self) -> Box<dyn BatchCollator + Send + Sync>;
}

/// Batch of samples for reconstruction training  
#[derive(Debug)]
pub struct ReconstructionBatch {
    /// Batch size
    pub batch_size: usize,
    /// Task IDs
    pub task_ids: Vec<String>,
    /// Task descriptions
    pub task_descriptions: Vec<String>,
    /// Task embeddings tensor [batch_size, embedding_dim]
    pub task_embeddings: Option<Tensor>,
    /// LoRA parameters by layer: layer_name -> (A_batch, B_batch)
    /// A_batch: [batch_size, input_dim, rank]
    /// B_batch: [batch_size, rank, output_dim]
    pub lora_params: HashMap<String, (Tensor, Tensor)>,
    /// Device used for tensors
    pub device: Device,
}

impl ReconstructionBatch {
    /// Get the number of parameters in the batch
    pub fn num_parameters(&self) -> usize {
        self.lora_params
            .values()
            .map(|(a, b)| a.elem_count() + b.elem_count())
            .sum()
    }
    
    /// Get layer names
    pub fn layer_names(&self) -> Vec<&String> {
        self.lora_params.keys().collect()
    }
    
    /// Get LoRA parameters for a specific layer
    pub fn get_layer_params(&self, layer_name: &str) -> Option<&(Tensor, Tensor)> {
        self.lora_params.get(layer_name)
    }
    
    /// Move batch to a different device
    pub fn to_device(&mut self, device: &Device) -> Result<(), candle_core::Error> {
        if let Some(embeddings) = &self.task_embeddings {
            self.task_embeddings = Some(embeddings.to_device(device)?);
        }
        
        for (a, b) in self.lora_params.values_mut() {
            *a = a.to_device(device)?;
            *b = b.to_device(device)?;
        }
        
        self.device = device.clone();
        Ok(())
    }
}

/// Batch of samples for supervised training
#[derive(Debug)]
pub struct SupervisedBatch {
    /// Batch size
    pub batch_size: usize,
    /// Task descriptions
    pub task_descriptions: Vec<String>,
    /// Input texts
    pub input_texts: Vec<String>,
    /// Target texts (for generation tasks)
    pub target_texts: Vec<Option<String>>,
    /// Input token IDs [batch_size, seq_len]
    pub input_ids: Option<Tensor>,
    /// Attention mask [batch_size, seq_len]
    pub attention_mask: Option<Tensor>,
    /// Labels tensor [batch_size, ...] (shape depends on task)
    pub labels: Option<Tensor>,
    /// Task embeddings [batch_size, embedding_dim]
    pub task_embeddings: Option<Tensor>,
    /// Device used for tensors
    pub device: Device,
}

impl SupervisedBatch {
    /// Get sequence length (if tokenized)
    pub fn seq_len(&self) -> Option<usize> {
        self.input_ids.as_ref().map(|ids| ids.dim(1).unwrap_or(0))
    }
    
    /// Move batch to a different device
    pub fn to_device(&mut self, device: &Device) -> Result<(), candle_core::Error> {
        if let Some(input_ids) = &self.input_ids {
            self.input_ids = Some(input_ids.to_device(device)?);
        }
        
        if let Some(attention_mask) = &self.attention_mask {
            self.attention_mask = Some(attention_mask.to_device(device)?);
        }
        
        if let Some(labels) = &self.labels {
            self.labels = Some(labels.to_device(device)?);
        }
        
        if let Some(embeddings) = &self.task_embeddings {
            self.task_embeddings = Some(embeddings.to_device(device)?);
        }
        
        self.device = device.clone();
        Ok(())
    }
}

/// Collator for reconstruction training batches
#[derive(Debug, Clone)]
pub struct ReconstructionCollator {
    /// Device for tensor operations
    device: Device,
    /// Whether to pad LoRA parameters to same dimensions
    pad_parameters: bool,
}

impl ReconstructionCollator {
    /// Create a new reconstruction collator
    pub fn new(device: Device, pad_parameters: bool) -> Self {
        Self {
            device,
            pad_parameters,
        }
    }
    
    /// Stack LoRA parameters for a specific layer
    #[instrument(skip(self, layer_params))]
    fn stack_layer_params(&self, layer_name: &str, layer_params: Vec<(Tensor, Tensor)>) -> Result<(Tensor, Tensor), DataError> {
        if layer_params.is_empty() {
            return Err(DataError::BatchCollationError {
                reason: format!("No parameters for layer {}", layer_name)
            });
        }
        
        // Check dimension consistency
        let first_a_dims = layer_params[0].0.dims();
        let first_b_dims = layer_params[0].1.dims();
        
        for (i, (a, b)) in layer_params.iter().enumerate() {
            if a.dims() != first_a_dims {
                return Err(DataError::BatchCollationError {
                    reason: format!(
                        "Inconsistent A matrix dimensions for layer {} at index {}: expected {:?}, got {:?}",
                        layer_name, i, first_a_dims, a.dims()
                    )
                });
            }
            if b.dims() != first_b_dims {
                return Err(DataError::BatchCollationError {
                    reason: format!(
                        "Inconsistent B matrix dimensions for layer {} at index {}: expected {:?}, got {:?}",
                        layer_name, i, first_b_dims, b.dims()
                    )
                });
            }
        }
        
        // Stack A matrices: [batch_size, input_dim, rank]
        let a_matrices: Result<Vec<_>, _> = layer_params
            .iter()
            .map(|(a, _)| a.unsqueeze(0))
            .collect();
        let a_matrices = a_matrices?;
        let a_stacked = Tensor::cat(&a_matrices, 0)?;
        
        // Stack B matrices: [batch_size, rank, output_dim]
        let b_matrices: Result<Vec<_>, _> = layer_params
            .iter()
            .map(|(_, b)| b.unsqueeze(0))
            .collect();
        let b_matrices = b_matrices?;
        let b_stacked = Tensor::cat(&b_matrices, 0)?;
        
        Ok((a_stacked, b_stacked))
    }
    
    /// Stack task embeddings
    #[instrument(skip(self, embeddings))]
    fn stack_task_embeddings(&self, embeddings: Vec<Option<Tensor>>) -> Result<Option<Tensor>, DataError> {
        let valid_embeddings: Vec<_> = embeddings.into_iter().flatten().collect();
        
        if valid_embeddings.is_empty() {
            return Ok(None);
        }
        
        // Check dimension consistency
        let first_dims = valid_embeddings[0].dims();
        for (i, embedding) in valid_embeddings.iter().enumerate() {
            if embedding.dims() != first_dims {
                return Err(DataError::BatchCollationError {
                    reason: format!(
                        "Inconsistent embedding dimensions at index {}: expected {:?}, got {:?}",
                        i, first_dims, embedding.dims()
                    )
                });
            }
        }
        
        // Stack embeddings: [batch_size, embedding_dim]
        let unsqueezed: Result<Vec<_>, _> = valid_embeddings
            .iter()
            .map(|emb| emb.unsqueeze(0))
            .collect();
        let unsqueezed = unsqueezed?;
        let stacked = Tensor::cat(&unsqueezed, 0)?;
        
        Ok(Some(stacked))
    }
}

impl BatchCollator for ReconstructionCollator {
    #[instrument(skip(self, samples))]
    fn collate_batch(&self, samples: Vec<DataSample>) -> Result<Box<dyn Any + Send>, DataError> {
        if samples.is_empty() {
            return Err(DataError::BatchCollationError {
                reason: "Empty batch".to_string()
            });
        }
        
        let batch_size = samples.len();
        debug!("Collating reconstruction batch of size {}", batch_size);
        
        let mut task_ids = Vec::with_capacity(batch_size);
        let mut task_descriptions = Vec::with_capacity(batch_size);
        let mut task_embeddings = Vec::with_capacity(batch_size);
        let mut layer_params_map: HashMap<String, Vec<(Tensor, Tensor)>> = HashMap::new();
        
        // Extract data from samples
        for sample in samples {
            task_ids.push(sample.id);
            task_descriptions.push(sample.task_description);
            task_embeddings.push(sample.task_embeddings);
            
            match sample.data {
                SampleData::Reconstruction { lora_params } => {
                    for (layer_name, (a, b)) in lora_params {
                        layer_params_map
                            .entry(layer_name)
                            .or_insert_with(Vec::new)
                            .push((a, b));
                    }
                },
                _ => {
                    return Err(DataError::BatchCollationError {
                        reason: "Expected reconstruction data".to_string()
                    });
                }
            }
        }
        
        // Stack LoRA parameters by layer
        let mut batched_lora_params = HashMap::new();
        for (layer_name, params) in layer_params_map {
            let (a_batch, b_batch) = self.stack_layer_params(&layer_name, params)?;
            batched_lora_params.insert(layer_name, (a_batch, b_batch));
        }
        
        // Stack task embeddings
        let batched_embeddings = self.stack_task_embeddings(task_embeddings)?;
        
        let batch = ReconstructionBatch {
            batch_size,
            task_ids,
            task_descriptions,
            task_embeddings: batched_embeddings,
            lora_params: batched_lora_params,
            device: self.device.clone(),
        };
        
        debug!("Created reconstruction batch with {} parameters", batch.num_parameters());
        Ok(Box::new(batch))
    }
    
    fn clone_box(&self) -> Box<dyn BatchCollator + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Collator for supervised training batches
#[derive(Debug, Clone)]
pub struct SupervisedCollator {
    /// Device for tensor operations
    device: Device,
    /// Maximum sequence length
    max_seq_len: Option<usize>,
    /// Padding token ID
    pad_token_id: Option<i64>,
}

impl SupervisedCollator {
    /// Create a new supervised collator
    pub fn new(device: Device, max_seq_len: Option<usize>, pad_token_id: Option<i64>) -> Self {
        Self {
            device,
            max_seq_len,
            pad_token_id,
        }
    }
    
    /// Stack and pad labels
    #[instrument(skip(self, labels))]
    fn collate_labels(&self, labels: Vec<Option<Tensor>>) -> Result<Option<Tensor>, DataError> {
        let valid_labels: Vec<_> = labels.into_iter().flatten().collect();
        
        if valid_labels.is_empty() {
            return Ok(None);
        }
        
        // For now, assume all labels have the same shape
        // TODO: Add padding support for variable-length labels
        let first_dims = valid_labels[0].dims();
        for (i, label) in valid_labels.iter().enumerate() {
            if label.dims() != first_dims {
                warn!(
                    "Inconsistent label dimensions at index {}: expected {:?}, got {:?}",
                    i, first_dims, label.dims()
                );
            }
        }
        
        // Stack labels
        let unsqueezed: Result<Vec<_>, _> = valid_labels
            .iter()
            .map(|label| label.unsqueeze(0))
            .collect();
        let unsqueezed = unsqueezed?;
        let stacked = Tensor::cat(&unsqueezed, 0)?;
        
        Ok(Some(stacked))
    }
}

impl BatchCollator for SupervisedCollator {
    #[instrument(skip(self, samples))]
    fn collate_batch(&self, samples: Vec<DataSample>) -> Result<Box<dyn Any + Send>, DataError> {
        if samples.is_empty() {
            return Err(DataError::BatchCollationError {
                reason: "Empty batch".to_string()
            });
        }
        
        let batch_size = samples.len();
        debug!("Collating supervised batch of size {}", batch_size);
        
        let mut task_descriptions = Vec::with_capacity(batch_size);
        let mut input_texts = Vec::with_capacity(batch_size);
        let mut target_texts = Vec::with_capacity(batch_size);
        let mut labels = Vec::with_capacity(batch_size);
        let mut task_embeddings = Vec::with_capacity(batch_size);
        
        // Extract data from samples
        for sample in samples {
            task_descriptions.push(sample.task_description);
            task_embeddings.push(sample.task_embeddings);
            
            match sample.data {
                SampleData::Supervised { input_text, target_text, labels: sample_labels } => {
                    input_texts.push(input_text);
                    target_texts.push(target_text);
                    labels.push(sample_labels);
                },
                _ => {
                    return Err(DataError::BatchCollationError {
                        reason: "Expected supervised data".to_string()
                    });
                }
            }
        }
        
        // Collate labels
        let batched_labels = self.collate_labels(labels)?;
        
        // Stack task embeddings (reuse logic from ReconstructionCollator)
        let batched_embeddings = if task_embeddings.iter().any(|emb| emb.is_some()) {
            let valid_embeddings: Vec<_> = task_embeddings.into_iter().flatten().collect();
            if !valid_embeddings.is_empty() {
                let unsqueezed: Result<Vec<_>, _> = valid_embeddings
                    .iter()
                    .map(|emb| emb.unsqueeze(0))
                    .collect();
                let unsqueezed = unsqueezed?;
                Some(Tensor::cat(&unsqueezed, 0)?)
            } else {
                None
            }
        } else {
            None
        };
        
        let batch = SupervisedBatch {
            batch_size,
            task_descriptions,
            input_texts,
            target_texts,
            input_ids: None, // Will be filled by tokenizer
            attention_mask: None, // Will be filled by tokenizer
            labels: batched_labels,
            task_embeddings: batched_embeddings,
            device: self.device.clone(),
        };
        
        debug!("Created supervised batch with {} samples", batch.batch_size);
        Ok(Box::new(batch))
    }
    
    fn clone_box(&self) -> Box<dyn BatchCollator + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Factory function to create appropriate collator based on sample type
pub fn create_collator(
    samples: &[DataSample],
    device: Device,
    max_seq_len: Option<usize>,
    pad_token_id: Option<i64>,
) -> Result<Box<dyn BatchCollator + Send + Sync>, DataError> {
    if samples.is_empty() {
        return Err(DataError::BatchCollationError {
            reason: "Cannot determine collator type from empty samples".to_string()
        });
    }
    
    match &samples[0].data {
        SampleData::Reconstruction { .. } => {
            Ok(Box::new(ReconstructionCollator::new(device, false)))
        },
        SampleData::Supervised { .. } => {
            Ok(Box::new(SupervisedCollator::new(device, max_seq_len, pad_token_id)))
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;
    
    fn create_test_lora_params() -> HashMap<String, (Tensor, Tensor)> {
        let device = Device::Cpu;
        let mut params = HashMap::new();
        
        // Create test A and B matrices
        let a = Tensor::randn(0.0, 1.0, (512, 16), &device).unwrap();
        let b = Tensor::randn(0.0, 1.0, (16, 512), &device).unwrap();
        
        params.insert("layer1".to_string(), (a, b));
        params
    }
    
    fn create_test_reconstruction_sample(id: &str) -> DataSample {
        DataSample {
            id: id.to_string(),
            task_description: format!("Test task {}", id),
            task_embeddings: None,
            data: SampleData::Reconstruction {
                lora_params: create_test_lora_params(),
            },
        }
    }
    
    fn create_test_supervised_sample(id: &str) -> DataSample {
        let device = Device::Cpu;
        let labels = Tensor::from_slice(&[1i64], 1, &device).unwrap();
        
        DataSample {
            id: id.to_string(),
            task_description: format!("Test task {}", id),
            task_embeddings: None,
            data: SampleData::Supervised {
                input_text: format!("Input text {}", id),
                target_text: Some(format!("Target text {}", id)),
                labels: Some(labels),
            },
        }
    }
    
    #[test]
    fn test_reconstruction_collator() {
        let device = Device::Cpu;
        let collator = ReconstructionCollator::new(device, false);
        
        let samples = vec![
            create_test_reconstruction_sample("1"),
            create_test_reconstruction_sample("2"),
        ];
        
        let result = collator.collate_batch(samples);
        assert!(result.is_ok());
        
        let batch = result.unwrap();
        let batch = batch.downcast::<ReconstructionBatch>().unwrap();
        
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.task_ids.len(), 2);
        assert!(batch.lora_params.contains_key("layer1"));
        
        let (a_batch, b_batch) = batch.lora_params.get("layer1").unwrap();
        assert_eq!(a_batch.dims(), &[2, 512, 16]); // [batch_size, input_dim, rank]
        assert_eq!(b_batch.dims(), &[2, 16, 512]); // [batch_size, rank, output_dim]
    }
    
    #[test]
    fn test_supervised_collator() {
        let device = Device::Cpu;
        let collator = SupervisedCollator::new(device, Some(512), Some(0));
        
        let samples = vec![
            create_test_supervised_sample("1"),
            create_test_supervised_sample("2"),
        ];
        
        let result = collator.collate_batch(samples);
        assert!(result.is_ok());
        
        let batch = result.unwrap();
        let batch = batch.downcast::<SupervisedBatch>().unwrap();
        
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.input_texts.len(), 2);
        assert_eq!(batch.target_texts.len(), 2);
        assert!(batch.labels.is_some());
        
        let labels = batch.labels.as_ref().unwrap();
        assert_eq!(labels.dims(), &[2, 1]); // [batch_size, label_dim]
    }
    
    #[test]
    fn test_empty_batch_error() {
        let device = Device::Cpu;
        let collator = ReconstructionCollator::new(device, false);
        
        let result = collator.collate_batch(vec![]);
        assert!(result.is_err());
        
        if let Err(DataError::BatchCollationError { reason }) = result {
            assert_eq!(reason, "Empty batch");
        } else {
            panic!("Expected BatchCollationError with 'Empty batch' message");
        }
    }
    
    #[test]
    fn test_create_collator_factory() {
        let device = Device::Cpu;
        
        // Test reconstruction collator creation
        let reconstruction_samples = vec![create_test_reconstruction_sample("1")];
        let collator = create_collator(&reconstruction_samples, device.clone(), None, None);
        assert!(collator.is_ok());
        
        // Test supervised collator creation
        let supervised_samples = vec![create_test_supervised_sample("1")];
        let collator = create_collator(&supervised_samples, device, Some(512), Some(0));
        assert!(collator.is_ok());
        
        // Test empty samples error
        let empty_samples: Vec<DataSample> = vec![];
        let collator = create_collator(&empty_samples, Device::Cpu, None, None);
        assert!(collator.is_err());
    }
}