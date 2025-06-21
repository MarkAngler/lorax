//! Data loaders for T2L training with async support and error handling

use super::{Dataset, BatchCollator, DataError};
use anyhow::Result;
use futures::stream::{Stream, StreamExt};
use rand::seq::SliceRandom;
use rand::rng;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn, instrument, error};

/// Configuration for DataLoader
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoaderConfig {
    /// Batch size
    pub batch_size: usize,
    /// Whether to shuffle data
    pub shuffle: bool,
    /// Number of worker threads for data loading
    pub num_workers: usize,
    /// Buffer size for async loading
    pub buffer_size: usize,
    /// Whether to drop the last incomplete batch
    pub drop_last: bool,
    /// Prefetch factor (number of batches to prefetch)
    pub prefetch_factor: usize,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            num_workers: 4,
            buffer_size: 100,
            drop_last: false,
            prefetch_factor: 2,
        }
    }
}

/// High-performance async data loader for training
pub struct DataLoader<D: Dataset + 'static> {
    /// Dataset reference
    dataset: Arc<D>,
    /// DataLoader configuration
    config: DataLoaderConfig,
    /// Batch collator
    collator: Box<dyn BatchCollator + Send + Sync>,
    /// Current epoch
    epoch: Arc<Mutex<usize>>,
    /// Current stream (for next_batch interface)
    current_stream: Arc<Mutex<Option<DataLoaderStream>>>,
}

impl<D: Dataset + 'static> DataLoader<D> {
    /// Create a new DataLoader
    pub fn new(
        dataset: Arc<D>,
        config: DataLoaderConfig,
        collator: Box<dyn BatchCollator + Send + Sync>,
    ) -> Self {
        info!(
            "Creating DataLoader with batch_size={}, num_workers={}, shuffle={}",
            config.batch_size,
            config.num_workers,
            config.shuffle
        );
        
        Self {
            dataset,
            config,
            collator,
            epoch: Arc::new(Mutex::new(0)),
            current_stream: Arc::new(Mutex::new(None)),
        }
    }
    
    /// Get the number of batches per epoch
    pub fn num_batches(&self) -> usize {
        let dataset_size = self.dataset.len();
        if self.config.drop_last {
            dataset_size / self.config.batch_size
        } else {
            (dataset_size + self.config.batch_size - 1) / self.config.batch_size
        }
    }
    
    /// Get dataset size
    pub fn dataset_size(&self) -> usize {
        self.dataset.len()
    }
    
    /// Create a stream of batches for an epoch
    #[instrument(skip(self))]
    pub async fn epoch_stream(&self) -> Result<DataLoaderStream> {
        let mut epoch = self.epoch.lock().await;
        *epoch += 1;
        let current_epoch = *epoch;
        drop(epoch);
        
        info!("Starting epoch {} with {} batches", current_epoch, self.num_batches());
        
        // Generate indices for this epoch
        let mut indices: Vec<usize> = (0..self.dataset.len()).collect();
        if self.config.shuffle {
            indices.shuffle(&mut rng());
            debug!("Shuffled {} indices", indices.len());
        }
        
        // Create batches
        let batches = self.create_batches(indices);
        
        // Create the stream
        Ok(DataLoaderStream::new(
            self.dataset.clone(),
            self.collator.clone_box(),
            batches,
            self.config.num_workers,
            self.config.buffer_size,
            current_epoch,
        ))
    }
    
    /// Create batches from indices
    fn create_batches(&self, indices: Vec<usize>) -> Vec<Vec<usize>> {
        let mut batches = Vec::new();
        
        for chunk in indices.chunks(self.config.batch_size) {
            if self.config.drop_last && chunk.len() < self.config.batch_size {
                break;
            }
            batches.push(chunk.to_vec());
        }
        
        debug!("Created {} batches", batches.len());
        batches
    }
    
    /// Get dataset metadata
    pub fn metadata(&self) -> &super::DatasetMetadata {
        self.dataset.metadata()
    }
    
    /// Get current epoch
    pub async fn current_epoch(&self) -> usize {
        *self.epoch.lock().await
    }
    
    /// Get next batch from the current epoch stream
    /// This method provides a simpler interface compared to epoch_stream()
    pub async fn next_batch(&self) -> Result<Option<Box<dyn std::any::Any + Send>>> {
        let mut stream_guard = self.current_stream.lock().await;
        
        // If we don't have a current stream, create one for the new epoch
        if stream_guard.is_none() {
            let new_stream = self.epoch_stream().await?;
            *stream_guard = Some(new_stream);
        }
        
        // Get the next batch from the stream
        if let Some(stream) = stream_guard.as_mut() {
            match stream.next().await {
                Some(Ok(batch)) => Ok(Some(batch)),
                Some(Err(e)) => Err(e),
                None => {
                    // Stream is exhausted, clear it for the next epoch
                    *stream_guard = None;
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }
}

/// Stream implementation for DataLoader batches
pub struct DataLoaderStream {
    /// Channel receiver for batches
    receiver: mpsc::Receiver<Result<Box<dyn std::any::Any + Send>>>,
    /// Number of batches processed
    batches_processed: usize,
    /// Total number of batches
    total_batches: usize,
    /// Current epoch number
    epoch: usize,
}

impl DataLoaderStream {
    /// Create a new DataLoaderStream
    #[instrument(skip(dataset, collator, batches))]
    fn new<D: Dataset + 'static>(
        dataset: Arc<D>,
        collator: Box<dyn BatchCollator + Send + Sync>,
        batches: Vec<Vec<usize>>,
        num_workers: usize,
        buffer_size: usize,
        epoch: usize,
    ) -> Self {
        let total_batches = batches.len();
        let (sender, receiver) = mpsc::channel(buffer_size);
        
        debug!(
            "Creating DataLoaderStream with {} workers for {} batches",
            num_workers,
            total_batches
        );
        
        // Spawn worker tasks
        for worker_id in 0..num_workers {
            let dataset = dataset.clone();
            let collator = collator.clone_box();
            let sender = sender.clone();
            let worker_batches: Vec<_> = batches
                .iter()
                .skip(worker_id)
                .step_by(num_workers)
                .cloned()
                .collect();
            
            tokio::spawn(async move {
                Self::worker_task(worker_id, dataset, collator, worker_batches, sender).await;
            });
        }
        
        // Drop the original sender so the channel closes when workers finish
        drop(sender);
        
        Self {
            receiver,
            batches_processed: 0,
            total_batches,
            epoch,
        }
    }
    
    /// Worker task for loading and collating batches
    #[instrument(skip(dataset, collator, batches, sender))]
    async fn worker_task<D: Dataset + 'static>(
        worker_id: usize,
        dataset: Arc<D>,
        collator: Box<dyn BatchCollator + Send + Sync>,
        batches: Vec<Vec<usize>>,
        sender: mpsc::Sender<Result<Box<dyn std::any::Any + Send>>>,
    ) {
        debug!("Worker {} starting with {} batches", worker_id, batches.len());
        
        for (batch_idx, indices) in batches.into_iter().enumerate() {
            let batch_result = tokio::task::spawn_blocking({
                let dataset = dataset.clone();
                let collator = collator.clone_box();
                move || -> Result<Box<dyn std::any::Any + Send>> {
                    // Load samples
                    let mut samples = Vec::with_capacity(indices.len());
                    for &idx in &indices {
                        match dataset.get(idx) {
                            Ok(sample) => samples.push(sample),
                            Err(e) => {
                                error!("Failed to load sample {}: {}", idx, e);
                                return Err(e);
                            }
                        }
                    }
                    
                    // Collate batch
                    collator.collate_batch(samples)
                        .map_err(|e| e.into())
                }
            }).await;
            
            match batch_result {
                Ok(Ok(batch)) => {
                    if sender.send(Ok(batch)).await.is_err() {
                        debug!("Worker {} channel closed, stopping", worker_id);
                        break;
                    }
                },
                Ok(Err(e)) => {
                    error!("Worker {} batch {} failed: {}", worker_id, batch_idx, e);
                    if sender.send(Err(e)).await.is_err() {
                        break;
                    }
                },
                Err(e) => {
                    error!("Worker {} task {} panicked: {}", worker_id, batch_idx, e);
                    if sender.send(Err(e.into())).await.is_err() {
                        break;
                    }
                }
            }
        }
        
        debug!("Worker {} finished", worker_id);
    }
    
    /// Get the number of batches processed
    pub fn batches_processed(&self) -> usize {
        self.batches_processed
    }
    
    /// Get the total number of batches
    pub fn total_batches(&self) -> usize {
        self.total_batches
    }
    
    /// Get the current epoch
    pub fn epoch(&self) -> usize {
        self.epoch
    }
    
    /// Get progress as a fraction (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        if self.total_batches == 0 {
            1.0
        } else {
            self.batches_processed as f32 / self.total_batches as f32
        }
    }
}

impl Stream for DataLoaderStream {
    type Item = Result<Box<dyn std::any::Any + Send>>;
    
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.receiver.poll_recv(cx) {
            Poll::Ready(Some(batch)) => {
                self.batches_processed += 1;
                
                if self.batches_processed % 100 == 0 {
                    debug!(
                        "Processed {}/{} batches in epoch {} ({:.1}%)",
                        self.batches_processed,
                        self.total_batches,
                        self.epoch,
                        self.progress() * 100.0
                    );
                }
                
                Poll::Ready(Some(batch))
            },
            Poll::Ready(None) => {
                info!(
                    "Epoch {} completed: {}/{} batches processed",
                    self.epoch,
                    self.batches_processed,
                    self.total_batches
                );
                Poll::Ready(None)
            },
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Builder for creating DataLoaders with fluent API
pub struct DataLoaderBuilder<D: Dataset + 'static> {
    dataset: Arc<D>,
    config: DataLoaderConfig,
    collator: Option<Box<dyn BatchCollator + Send + Sync>>,
}

impl<D: Dataset + 'static> DataLoaderBuilder<D> {
    /// Create a new DataLoaderBuilder
    pub fn new(dataset: Arc<D>) -> Self {
        Self {
            dataset,
            config: DataLoaderConfig::default(),
            collator: None,
        }
    }
    
    /// Set batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }
    
    /// Set shuffle
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.config.shuffle = shuffle;
        self
    }
    
    /// Set number of workers
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.config.num_workers = num_workers;
        self
    }
    
    /// Set buffer size
    pub fn buffer_size(mut self, buffer_size: usize) -> Self {
        self.config.buffer_size = buffer_size;
        self
    }
    
    /// Set drop last
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.config.drop_last = drop_last;
        self
    }
    
    /// Set prefetch factor
    pub fn prefetch_factor(mut self, prefetch_factor: usize) -> Self {
        self.config.prefetch_factor = prefetch_factor;
        self
    }
    
    /// Set batch collator
    pub fn collator(mut self, collator: Box<dyn BatchCollator + Send + Sync>) -> Self {
        self.collator = Some(collator);
        self
    }
    
    /// Build the DataLoader
    pub fn build(self) -> Result<DataLoader<D>> {
        let collator = self.collator
            .ok_or_else(|| DataError::MissingMetadata { field: "collator".to_string() })?;
        
        Ok(DataLoader::new(self.dataset, self.config, collator))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::data::{DataSample, SampleData, DatasetMetadata};
    use std::collections::HashMap;
    use tokio;
    
    // Mock dataset for testing
    struct MockDataset {
        samples: Vec<String>,
        metadata: DatasetMetadata,
    }
    
    impl MockDataset {
        fn new(size: usize) -> Self {
            let samples: Vec<String> = (0..size)
                .map(|i| format!("sample_{}", i))
                .collect();
            
            let metadata = DatasetMetadata {
                name: "mock_dataset".to_string(),
                num_samples: size,
                lora_config: None,
                task_info: None,
            };
            
            Self { samples, metadata }
        }
    }
    
    impl Dataset for MockDataset {
        fn len(&self) -> usize {
            self.samples.len()
        }
        
        fn get(&self, index: usize) -> Result<DataSample> {
            if index >= self.len() {
                return Err(DataError::InvalidIndex { 
                    index, 
                    dataset_size: self.len() 
                }.into());
            }
            
            Ok(DataSample {
                id: format!("sample_{}", index),
                task_description: self.samples[index].clone(),
                task_embeddings: None,
                data: SampleData::Supervised {
                    input_text: self.samples[index].clone(),
                    target_text: None,
                    labels: None,
                },
            })
        }
        
        fn metadata(&self) -> &DatasetMetadata {
            &self.metadata
        }
    }
    
    // Mock collator for testing
    struct MockCollator;
    
    impl BatchCollator for MockCollator {
        fn collate_batch(&self, samples: Vec<DataSample>) -> Result<Box<dyn std::any::Any + Send>, DataError> {
            Ok(Box::new(samples))
        }
        
        fn clone_box(&self) -> Box<dyn BatchCollator + Send + Sync> {
            Box::new(MockCollator)
        }
    }
    
    #[tokio::test]
    async fn test_dataloader_creation() {
        let dataset = Arc::new(MockDataset::new(100));
        let config = DataLoaderConfig {
            batch_size: 10,
            shuffle: false,
            num_workers: 2,
            buffer_size: 5,
            drop_last: false,
            prefetch_factor: 1,
        };
        let collator = Box::new(MockCollator);
        
        let dataloader = DataLoader::new(dataset, config, collator);
        assert_eq!(dataloader.dataset_size(), 100);
        assert_eq!(dataloader.num_batches(), 10);
    }
    
    #[tokio::test]
    async fn test_dataloader_stream() {
        let dataset = Arc::new(MockDataset::new(25));
        let config = DataLoaderConfig {
            batch_size: 10,
            shuffle: false,
            num_workers: 1,
            buffer_size: 5,
            drop_last: false,
            prefetch_factor: 1,
        };
        let collator = Box::new(MockCollator);
        
        let dataloader = DataLoader::new(dataset, config, collator);
        let mut stream = dataloader.epoch_stream().await.unwrap();
        
        let mut batch_count = 0;
        while let Some(batch_result) = stream.next().await {
            assert!(batch_result.is_ok());
            batch_count += 1;
        }
        
        assert_eq!(batch_count, 3); // 25 samples / 10 batch_size = 3 batches
        assert_eq!(stream.batches_processed(), 3);
        assert_eq!(stream.total_batches(), 3);
    }
    
    #[tokio::test]
    async fn test_dataloader_builder() {
        let dataset = Arc::new(MockDataset::new(50));
        
        let dataloader = DataLoaderBuilder::new(dataset)
            .batch_size(5)
            .shuffle(true)
            .num_workers(2)
            .buffer_size(10)
            .drop_last(true)
            .collator(Box::new(MockCollator))
            .build()
            .unwrap();
        
        assert_eq!(dataloader.dataset_size(), 50);
        assert_eq!(dataloader.num_batches(), 10); // 50 samples / 5 batch_size = 10 batches (drop_last=true)
    }
}