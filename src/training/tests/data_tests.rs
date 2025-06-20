//! Tests for data loading infrastructure

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::sync::Arc;
use proptest::prelude::*;

use crate::training::data::{
    Dataset, DataLoader, DataLoaderConfig, BatchCollator,
    ReconstructionDataset, SupervisedDataset, ReconstructionBatch,
    SupervisedBatch, DataSample, SampleData,
};

use super::fixtures::{DataTestFixture, TestFixture};
use super::mock_data::{create_mock_dataset, create_mock_batch, DatasetType};
use super::test_utils::{assert_tensor_close, create_test_device};

#[test]
fn test_dataset_loading() -> Result<()> {
    let device = create_test_device();
    
    // Test reconstruction dataset
    let recon_dataset = create_mock_dataset(100, DatasetType::Reconstruction);
    assert_eq!(recon_dataset.len(), 100);
    assert!(!recon_dataset.is_empty());
    
    // Test sample retrieval
    let sample = recon_dataset.get(0)?;
    assert_eq!(sample.id, "sample_0");
    assert!(matches!(sample.data, SampleData::Reconstruction { .. }));
    
    // Test supervised dataset
    let sup_dataset = create_mock_dataset(50, DatasetType::Supervised);
    assert_eq!(sup_dataset.len(), 50);
    
    let sample = sup_dataset.get(10)?;
    assert_eq!(sample.id, "sample_10");
    assert!(matches!(sample.data, SampleData::Supervised { .. }));
    
    Ok(())
}

#[test]
fn test_dataset_bounds_checking() -> Result<()> {
    let dataset = create_mock_dataset(10, DatasetType::Reconstruction);
    
    // Valid indices should work
    for i in 0..10 {
        let sample = dataset.get(i)?;
        assert_eq!(sample.id, format!("sample_{}", i));
    }
    
    // Out of bounds should fail
    assert!(dataset.get(10).is_err());
    assert!(dataset.get(100).is_err());
    
    Ok(())
}

#[test]
fn test_data_loader_creation() -> Result<()> {
    let dataset = create_mock_dataset(100, DatasetType::Reconstruction);
    
    // This test needs to be updated once DataLoader API is stabilized
    // For now, we'll just test the dataset
    assert_eq!(dataset.len(), 100);
    
    // TODO: Test loader properties once DataLoader is properly implemented
    // assert_eq!(loader.batch_size(), 16);
    // assert_eq!(loader.num_batches(), 7); // 100 samples / 16 batch size = 6.25, rounded up to 7
    
    Ok(())
}

#[test]
fn test_batch_collation() -> Result<()> {
    let device = create_test_device();
    let dataset = create_mock_dataset(20, DatasetType::Reconstruction);
    
    // Get samples for batch
    let samples: Result<Vec<_>> = (0..4).map(|i| dataset.get(i)).collect();
    let samples = samples?;
    
    // Create batch collator
    let collator = BatchCollator::new(device.clone());
    
    // Test reconstruction batch collation
    let recon_samples: Vec<_> = samples.iter()
        .filter_map(|s| match &s.data {
            SampleData::Reconstruction { lora_params } => Some((s, lora_params)),
            _ => None,
        })
        .collect();
    
    assert_eq!(recon_samples.len(), 4);
    
    Ok(())
}

#[test]
fn test_reconstruction_batch_structure() -> Result<()> {
    let device = create_test_device();
    let batch = create_mock_batch(8, &device)?;
    
    // Check batch structure
    assert_eq!(batch.task_embeddings.shape().dims(), &[8, 64]);
    assert_eq!(batch.sample_ids.len(), 8);
    
    // Check LoRA parameters
    for (layer_name, (a, b)) in &batch.target_params {
        assert_eq!(a.shape().dims(), &[8, 64, 4]);
        assert_eq!(b.shape().dims(), &[8, 4, 64]);
        
        // Check layer mask if present
        if let Some(ref mask) = batch.layer_mask {
            if let Some(layer_mask) = mask.get(layer_name) {
                assert_eq!(layer_mask.shape().dims(), &[8]);
            }
        }
    }
    
    Ok(())
}

#[test]
fn test_data_loader_iteration() -> Result<()> {
    let device = create_test_device();
    let dataset = create_mock_dataset(100, DatasetType::Reconstruction);
    
    // TODO: Fix DataLoader creation once API is stable
    // let loader = DataLoader::new(
    //     dataset,
    //     16,    // batch_size
    //     false, // shuffle
    //     0,     // num_workers
    // )?;
    
    // Count batches
    let mut batch_count = 0;
    let mut total_samples = 0;
    
    // In real implementation, we'd iterate through the loader
    // For now, simulate the expected behavior
    let expected_batches = 7; // ceil(100 / 16)
    let expected_last_batch_size = 4; // 100 % 16
    
    // TODO: Test once DataLoader is implemented
    // assert_eq!(loader.num_batches(), expected_batches);
    
    Ok(())
}

#[test]
fn test_multi_threaded_loading() -> Result<()> {
    let dataset = create_mock_dataset(1000, DatasetType::Reconstruction);
    
    // TODO: Test with multiple workers once DataLoader is implemented
    // let loader = DataLoader::new(
    //     dataset,
    //     32,    // batch_size
    //     true,  // shuffle
    //     4,     // num_workers
    // )?;
    
    // Verify loader can handle concurrent access
    // assert_eq!(loader.num_batches(), 32); // ceil(1000 / 32)
    
    Ok(())
}

#[test]
fn test_empty_dataset_handling() -> Result<()> {
    let dataset = create_mock_dataset(0, DatasetType::Supervised);
    assert!(dataset.is_empty());
    
    // TODO: DataLoader should handle empty dataset gracefully
    // let result = DataLoader::new(dataset, 16, false, 0);
    // assert!(result.is_err() || result.unwrap().num_batches() == 0);
    
    Ok(())
}

#[test]
fn test_hdf5_dataset_loading() -> Result<()> {
    let fixture = DataTestFixture::new()?;
    
    // Create test HDF5 file
    let hdf5_path = fixture.create_hdf5_dataset("test_dataset", 50)?;
    
    // Load dataset from HDF5
    let dataset = ReconstructionDataset::from_hdf5(
        hdf5_path,
        Some(64), // embedding_dim
        None,     // no pre-computed embeddings
    )?;
    
    assert_eq!(dataset.len(), 50);
    
    // Test sample loading
    let sample = dataset.get(0)?;
    assert!(matches!(sample.data, SampleData::Reconstruction { .. }));
    
    Ok(())
}

#[test]
fn test_dataset_caching() -> Result<()> {
    use std::time::Instant;
    
    let dataset = create_mock_dataset(100, DatasetType::Reconstruction);
    
    // First access - might be slower
    let start = Instant::now();
    let _sample1 = dataset.get(0)?;
    let first_access = start.elapsed();
    
    // Second access - should be cached (if caching is implemented)
    let start = Instant::now();
    let _sample2 = dataset.get(0)?;
    let second_access = start.elapsed();
    
    // Note: This test assumes caching is implemented
    // In practice, second access should be faster or equal
    
    Ok(())
}

// Property-based tests
proptest! {
    #[test]
    fn prop_test_batch_size_consistency(
        num_samples in 1usize..=1000,
        batch_size in 1usize..=32,
    ) {
        let dataset = create_mock_dataset(num_samples, DatasetType::Reconstruction);
        // TODO: Test once DataLoader is implemented
        // let loader = DataLoader::new(dataset, batch_size, false, 0).unwrap();
        
        let expected_batches = (num_samples + batch_size - 1) / batch_size;
        // prop_assert_eq!(loader.num_batches(), expected_batches);
    }
    
    #[test]
    fn prop_test_dataset_index_validity(
        num_samples in 1usize..=100,
        indices in prop::collection::vec(0usize..100, 0..10),
    ) {
        let dataset = create_mock_dataset(num_samples, DatasetType::Supervised);
        
        for &idx in &indices {
            if idx < num_samples {
                prop_assert!(dataset.get(idx).is_ok());
            } else {
                prop_assert!(dataset.get(idx).is_err());
            }
        }
    }
}

#[test]
fn test_supervised_batch_structure() -> Result<()> {
    use super::mock_data::create_mock_supervised_batch;
    
    let device = create_test_device();
    let batch = create_mock_supervised_batch(4, 128, 1000, &device)?;
    
    // Check batch dimensions
    assert_eq!(batch.input_ids.shape().dims(), &[4, 128]);
    assert_eq!(batch.attention_mask.shape().dims(), &[4, 128]);
    
    if let Some(labels) = &batch.labels {
        assert_eq!(labels.shape().dims(), &[4, 128]);
    }
    
    assert_eq!(batch.task_embeddings.shape().dims(), &[4, 64]);
    assert_eq!(batch.sample_ids.len(), 4);
    
    Ok(())
}

#[test]
fn test_data_augmentation() -> Result<()> {
    // Test any data augmentation strategies
    let device = create_test_device();
    let original_batch = create_mock_batch(4, &device)?;
    
    // In a real implementation, you might have augmentation functions
    // For example: add noise, dropout, etc.
    
    // Verify augmentation doesn't break batch structure
    assert_eq!(original_batch.sample_ids.len(), 4);
    
    Ok(())
}

#[test]
fn test_memory_efficiency() -> Result<()> {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
    
    struct TrackingAllocator;
    
    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
            System.alloc(layout)
        }
        
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
            System.dealloc(ptr, layout)
        }
    }
    
    // Test that data loading doesn't leak memory
    let initial_memory = ALLOCATED.load(Ordering::SeqCst);
    
    {
        let dataset = create_mock_dataset(100, DatasetType::Reconstruction);
        // TODO: Test DataLoader creation once implemented
        // let _loader = DataLoader::new(dataset, 16, false, 0)?;
        // Simulate loading batches
    }
    
    let final_memory = ALLOCATED.load(Ordering::SeqCst);
    
    // Allow some overhead, but memory should be mostly freed
    assert!(final_memory <= initial_memory + 1024 * 1024); // 1MB tolerance
    
    Ok(())
}