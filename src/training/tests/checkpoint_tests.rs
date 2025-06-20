//! Tests for checkpointing functionality

use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use chrono::Utc;

use crate::training::checkpoints::{
    CheckpointManager, TrainingCheckpoint, CheckpointInfo, CheckpointMetadata,
    RecoveryManager, RecoveryOptions, RecoveryResult, ValidationResult,
    ModelMetadata, TrainingMetadata, ValidationStatus, CheckpointLoadOptions,
};
use crate::training::{TrainingConfig, TrainingState, TrainingStatus};

use super::fixtures::{CheckpointTestFixture, TestFixture};
use super::test_utils::{create_test_config, create_temp_dir};

#[test]
fn test_checkpoint_save_load() {
    // Placeholder test to be referenced by mod.rs
}

#[test]
fn test_checkpoint_manager_creation() -> Result<()> {
    let temp_dir = create_temp_dir()?;
    let checkpoint_dir = temp_dir.path().join("checkpoints");
    
    let manager = CheckpointManager::new(
        &checkpoint_dir,
        Some(3), // save_total_limit
        false,   // compression
    )?;
    
    // Verify directory was created
    assert!(checkpoint_dir.exists());
    assert!(checkpoint_dir.is_dir());
    
    Ok(())
}

#[test]
fn test_checkpoint_saving() -> Result<()> {
    let fixture = CheckpointTestFixture::new()?;
    let manager = fixture.create_checkpoint_manager()?;
    
    // Create test checkpoint
    let checkpoint = TrainingCheckpoint {
        epoch: 1,
        global_step: 100,
        model_state: vec![1, 2, 3, 4, 5],
        optimizer_state: Some(vec![6, 7, 8, 9]),
        scheduler_state: Some(vec![10, 11]),
        metrics: HashMap::from([
            ("loss".to_string(), 0.5),
            ("accuracy".to_string(), 0.85),
        ]),
        config: fixture.config.clone(),
        timestamp: Utc::now(),
        metadata: CheckpointMetadata {
            model_metadata: ModelMetadata {
                architecture: "bert_t2l".to_string(),
                num_parameters: 1000000,
                model_version: "1.0.0".to_string(),
                lora_config: crate::training::checkpoints::metadata::LoRAMetadata {
                    rank: 4,
                    alpha: 16.0,
                    target_modules: vec!["attention".to_string(), "mlp".to_string()],
                    dropout: 0.0,
                    custom_params: HashMap::new(),
                },
            },
            training_metadata: TrainingMetadata {
                total_steps: 1000,
                training_time_seconds: 3600,
                dataset_info: HashMap::from([
                    ("name".to_string(), "test_dataset".to_string()),
                    ("size".to_string(), "1000".to_string()),
                ]),
                framework_version: env!("CARGO_PKG_VERSION").to_string(),
            },
            validation_status: ValidationStatus::Validated,
            custom_metadata: HashMap::new(),
        },
    };
    
    // Save checkpoint
    let checkpoint_path = manager.save_checkpoint(checkpoint.clone())?;
    assert!(checkpoint_path.exists());
    
    // Load checkpoint
    let loaded = manager.load_checkpoint(&checkpoint_path)?;
    assert_eq!(loaded.epoch, checkpoint.epoch);
    assert_eq!(loaded.global_step, checkpoint.global_step);
    assert_eq!(loaded.model_state, checkpoint.model_state);
    
    Ok(())
}

#[test]
fn test_checkpoint_limit_enforcement() -> Result<()> {
    let temp_dir = create_temp_dir()?;
    let checkpoint_dir = temp_dir.path().join("checkpoints");
    
    let manager = CheckpointManager::new(
        &checkpoint_dir,
        Some(3), // Keep only 3 checkpoints
        false,
    )?;
    
    // Save 5 checkpoints
    for i in 0..5 {
        let checkpoint = TrainingCheckpoint {
            epoch: i,
            global_step: i * 100,
            model_state: vec![i as u8; 10],
            optimizer_state: Some(vec![i as u8; 5]),
            scheduler_state: None,
            metrics: HashMap::from([("loss".to_string(), 1.0 / (i + 1) as f64)]),
            config: create_test_config(),
            timestamp: Utc::now() + chrono::Duration::seconds(i as i64),
            metadata: Default::default(),
        };
        
        manager.save_checkpoint(checkpoint)?;
        std::thread::sleep(std::time::Duration::from_millis(10)); // Ensure different timestamps
    }
    
    // Only 3 checkpoints should remain
    let checkpoints = manager.list_checkpoints()?;
    assert_eq!(checkpoints.len(), 3);
    
    // Verify oldest checkpoints were removed
    let epochs: Vec<usize> = checkpoints.iter().map(|c| c.epoch).collect();
    assert!(epochs.contains(&2));
    assert!(epochs.contains(&3));
    assert!(epochs.contains(&4));
    assert!(!epochs.contains(&0));
    assert!(!epochs.contains(&1));
    
    Ok(())
}

#[test]
fn test_checkpoint_best_model_tracking() -> Result<()> {
    let fixture = CheckpointTestFixture::new()?;
    let manager = fixture.create_checkpoint_manager()?;
    
    let mut best_score = f64::MAX;
    let mut best_checkpoint_path = None;
    
    // Save checkpoints with different scores
    for i in 0..5 {
        let loss = 1.0 / (i + 1) as f64;
        
        let checkpoint = TrainingCheckpoint {
            epoch: i,
            global_step: i * 100,
            model_state: vec![i as u8; 10],
            optimizer_state: None,
            scheduler_state: None,
            metrics: HashMap::from([("loss".to_string(), loss)]),
            config: fixture.config.clone(),
            timestamp: Utc::now(),
            metadata: Default::default(),
        };
        
        let path = manager.save_checkpoint(checkpoint)?;
        
        // Track best model (lower loss is better)
        if loss < best_score {
            best_score = loss;
            best_checkpoint_path = Some(path);
        }
    }
    
    // Mark best checkpoint
    if let Some(best_path) = best_checkpoint_path {
        manager.mark_as_best(&best_path)?;
        
        // Verify best checkpoint can be retrieved
        let best = manager.get_best_checkpoint()?;
        assert!(best.is_some());
        
        if let Some(best_info) = best {
            assert_eq!(best_info.metrics.get("loss"), Some(&best_score));
        }
    }
    
    Ok(())
}

#[test]
fn test_checkpoint_compression() -> Result<()> {
    let temp_dir = create_temp_dir()?;
    let checkpoint_dir = temp_dir.path().join("checkpoints");
    
    // Create manager with compression enabled
    let manager = CheckpointManager::new(
        &checkpoint_dir,
        None,
        true, // Enable compression
    )?;
    
    // Create large checkpoint
    let large_data = vec![42u8; 1_000_000]; // 1MB of data
    let checkpoint = TrainingCheckpoint {
        epoch: 1,
        global_step: 100,
        model_state: large_data.clone(),
        optimizer_state: Some(large_data.clone()),
        scheduler_state: None,
        metrics: HashMap::new(),
        config: create_test_config(),
        timestamp: Utc::now(),
        metadata: Default::default(),
    };
    
    // Save with compression
    let compressed_path = manager.save_checkpoint(checkpoint.clone())?;
    
    // Check that compressed file is smaller than uncompressed data
    let file_size = std::fs::metadata(&compressed_path)?.len();
    assert!(file_size < (large_data.len() * 2) as u64);
    
    // Load and verify
    let loaded = manager.load_checkpoint(&compressed_path)?;
    assert_eq!(loaded.model_state.len(), checkpoint.model_state.len());
    
    Ok(())
}

#[test]
fn test_recovery_manager() -> Result<()> {
    let fixture = CheckpointTestFixture::new()?;
    let checkpoint_manager = fixture.create_checkpoint_manager()?;
    
    // Save a checkpoint
    let checkpoint = TrainingCheckpoint {
        epoch: 5,
        global_step: 500,
        model_state: vec![1, 2, 3, 4],
        optimizer_state: Some(vec![5, 6, 7, 8]),
        scheduler_state: Some(vec![9, 10]),
        metrics: HashMap::from([("loss".to_string(), 0.3)]),
        config: fixture.config.clone(),
        timestamp: Utc::now(),
        metadata: Default::default(),
    };
    
    let checkpoint_path = checkpoint_manager.save_checkpoint(checkpoint)?;
    
    // Create recovery manager
    let recovery_manager = RecoveryManager::new(&fixture.checkpoint_dir)?;
    
    // Test recovery
    let options = RecoveryOptions {
        validate_checkpoints: true,
        repair_corrupted: false,
        max_recovery_attempts: 3,
    };
    
    let result = recovery_manager.recover_training_state(options)?;
    
    match result {
        RecoveryResult::Success { checkpoint_path: recovered_path, validation } => {
            assert_eq!(recovered_path, checkpoint_path);
            assert!(matches!(validation.status, ValidationStatus::Validated));
        }
        _ => panic!("Expected successful recovery"),
    }
    
    Ok(())
}

#[test]
fn test_checkpoint_metadata_validation() -> Result<()> {
    let fixture = CheckpointTestFixture::new()?;
    let manager = fixture.create_checkpoint_manager()?;
    
    // Create checkpoint with metadata
    let metadata = CheckpointMetadata {
        model_metadata: ModelMetadata {
            architecture: "bert_t2l".to_string(),
            num_parameters: 10_000_000,
            model_version: "2.0.0".to_string(),
            lora_config: crate::training::checkpoints::metadata::LoRAMetadata {
                rank: 8,
                alpha: 32.0,
                target_modules: vec!["query".to_string(), "value".to_string()],
                dropout: 0.1,
                custom_params: HashMap::new(),
            },
        },
        training_metadata: TrainingMetadata {
            total_steps: 10000,
            training_time_seconds: 7200,
            dataset_info: HashMap::from([
                ("name".to_string(), "large_dataset".to_string()),
                ("size".to_string(), "1000000".to_string()),
                ("version".to_string(), "v1.2".to_string()),
            ]),
            framework_version: env!("CARGO_PKG_VERSION").to_string(),
        },
        validation_status: ValidationStatus::Validated,
        custom_metadata: HashMap::from([
            ("experiment_id".to_string(), "exp_001".to_string()),
            ("hyperparameter_search".to_string(), "grid_search".to_string()),
        ]),
    };
    
    let checkpoint = TrainingCheckpoint {
        epoch: 10,
        global_step: 1000,
        model_state: vec![1; 100],
        optimizer_state: None,
        scheduler_state: None,
        metrics: HashMap::new(),
        config: fixture.config.clone(),
        timestamp: Utc::now(),
        metadata: metadata.clone(),
    };
    
    // Save and load
    let path = manager.save_checkpoint(checkpoint)?;
    let loaded = manager.load_checkpoint(&path)?;
    
    // Verify metadata
    assert_eq!(loaded.metadata.model_metadata.architecture, "bert_t2l");
    assert_eq!(loaded.metadata.model_metadata.num_parameters, 10_000_000);
    assert_eq!(loaded.metadata.training_metadata.total_steps, 10000);
    assert_eq!(
        loaded.metadata.custom_metadata.get("experiment_id"),
        Some(&"exp_001".to_string())
    );
    
    Ok(())
}

#[test]
fn test_partial_checkpoint_loading() -> Result<()> {
    let fixture = CheckpointTestFixture::new()?;
    let manager = fixture.create_checkpoint_manager()?;
    
    // Save full checkpoint
    let checkpoint = TrainingCheckpoint {
        epoch: 3,
        global_step: 300,
        model_state: vec![1; 1000],
        optimizer_state: Some(vec![2; 500]),
        scheduler_state: Some(vec![3; 100]),
        metrics: HashMap::from([("loss".to_string(), 0.4)]),
        config: fixture.config.clone(),
        timestamp: Utc::now(),
        metadata: Default::default(),
    };
    
    let path = manager.save_checkpoint(checkpoint)?;
    
    // Test loading only model state
    let load_options = CheckpointLoadOptions {
        load_model: true,
        load_optimizer: false,
        load_scheduler: false,
        strict: false,
        map_location: None,
    };
    
    let partial = manager.load_checkpoint_with_options(&path, load_options)?;
    assert!(!partial.model_state.is_empty());
    assert!(partial.optimizer_state.is_none());
    assert!(partial.scheduler_state.is_none());
    
    Ok(())
}

#[test]
fn test_checkpoint_corruption_detection() -> Result<()> {
    let temp_dir = create_temp_dir()?;
    let checkpoint_path = temp_dir.path().join("checkpoints").join("checkpoint_epoch_1.pt");
    std::fs::create_dir_all(checkpoint_path.parent().unwrap())?;
    
    // Write corrupted data
    std::fs::write(&checkpoint_path, b"corrupted checkpoint data")?;
    
    let manager = CheckpointManager::new(
        checkpoint_path.parent().unwrap(),
        None,
        false,
    )?;
    
    // Loading should fail
    let result = manager.load_checkpoint(&checkpoint_path);
    assert!(result.is_err());
    
    Ok(())
}

#[test]
fn test_checkpoint_versioning() -> Result<()> {
    let fixture = CheckpointTestFixture::new()?;
    let manager = fixture.create_checkpoint_manager()?;
    
    // Save checkpoints from different versions
    for version in ["1.0.0", "1.1.0", "2.0.0"] {
        let mut metadata = CheckpointMetadata::default();
        metadata.model_metadata.model_version = version.to_string();
        
        let checkpoint = TrainingCheckpoint {
            epoch: 1,
            global_step: 100,
            model_state: vec![1, 2, 3],
            optimizer_state: None,
            scheduler_state: None,
            metrics: HashMap::new(),
            config: fixture.config.clone(),
            timestamp: Utc::now(),
            metadata,
        };
        
        manager.save_checkpoint(checkpoint)?;
    }
    
    // List checkpoints and verify versions
    let checkpoints = manager.list_checkpoints()?;
    let versions: Vec<String> = checkpoints
        .iter()
        .map(|c| c.metadata.model_metadata.model_version.clone())
        .collect();
    
    assert!(versions.contains(&"1.0.0".to_string()));
    assert!(versions.contains(&"1.1.0".to_string()));
    assert!(versions.contains(&"2.0.0".to_string()));
    
    Ok(())
}

#[test]
fn test_atomic_checkpoint_saving() -> Result<()> {
    use std::sync::{Arc, Barrier};
    use std::thread;
    
    let fixture = CheckpointTestFixture::new()?;
    let manager = Arc::new(fixture.create_checkpoint_manager()?);
    
    // Test concurrent checkpoint saving
    let barrier = Arc::new(Barrier::new(3));
    let mut handles = vec![];
    
    for i in 0..3 {
        let manager_clone = manager.clone();
        let barrier_clone = barrier.clone();
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            let checkpoint = TrainingCheckpoint {
                epoch: i,
                global_step: i * 100,
                model_state: vec![i as u8; 100],
                optimizer_state: None,
                scheduler_state: None,
                metrics: HashMap::from([("thread_id".to_string(), i as f64)]),
                config: create_test_config(),
                timestamp: Utc::now(),
                metadata: Default::default(),
            };
            
            manager_clone.save_checkpoint(checkpoint)
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap()?;
    }
    
    // Verify all checkpoints were saved correctly
    let checkpoints = manager.list_checkpoints()?;
    assert_eq!(checkpoints.len(), 3);
    
    Ok(())
}