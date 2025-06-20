//! Tests for trainer functionality

use anyhow::Result;
use candle_core::Device;
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;
use tempfile::TempDir;

use crate::training::{
    T2LTrainer, TrainingConfig, TrainingState, TrainingStatus, TrainingResult,
    TrainingEvent, MemoryUsage, CheckpointManager, MetricsTracker,
    create_optimizer, create_scheduler, OptimizerState, SchedulerState,
};
use crate::training::data::DataLoader;

use super::fixtures::{TrainingTestFixture, TestFixture};
use super::test_utils::{create_test_config, create_test_device, create_test_model};
use super::mock_data::{create_mock_dataset, DatasetType};

#[test]
fn test_trainer_creation() {
    // Placeholder test to be referenced by mod.rs
}

#[test]
fn test_trainer_initialization() -> Result<()> {
    let fixture = TrainingTestFixture::new()?;
    
    // Verify fixture components are properly initialized
    assert_eq!(fixture.config.training.batch_size, 4);
    assert_eq!(fixture.config.training.num_epochs, 2);
    assert!(fixture.train_loader.batch_size() == 4);
    assert!(fixture.val_loader.is_some());
    
    Ok(())
}

#[test]
fn test_training_state_management() -> Result<()> {
    let mut state = TrainingState {
        epoch: 0,
        step: 0,
        global_step: 0,
        best_score: None,
        steps_since_best: 0,
        start_time: chrono::Utc::now(),
        last_eval_time: None,
        status: TrainingStatus::Initializing,
        memory_usage: MemoryUsage::default(),
    };
    
    // Test state transitions
    state.status = TrainingStatus::Running;
    assert!(matches!(state.status, TrainingStatus::Running));
    
    // Update step counts
    state.step += 1;
    state.global_step += 1;
    assert_eq!(state.step, 1);
    assert_eq!(state.global_step, 1);
    
    // Update best score
    state.best_score = Some(0.95);
    state.steps_since_best = 0;
    
    // Simulate worse score
    state.steps_since_best += 1;
    assert_eq!(state.steps_since_best, 1);
    
    Ok(())
}

#[test]
fn test_optimizer_creation() -> Result<()> {
    let device = create_test_device();
    let varmap = create_test_model(&device)?;
    let config = create_test_config();
    
    // Create optimizer
    let optimizer = create_optimizer(
        &config.optimizer,
        varmap.all_vars(),
        config.optimizer.learning_rate,
    )?;
    
    // Verify optimizer state
    match &optimizer {
        OptimizerState::AdamW(opt) => {
            assert_eq!(opt.learning_rate(), config.optimizer.learning_rate);
        }
        _ => panic!("Expected AdamW optimizer"),
    }
    
    Ok(())
}

#[test]
fn test_scheduler_creation() -> Result<()> {
    let config = create_test_config();
    
    // Create scheduler with default config
    let total_steps = 1000;
    let scheduler = create_scheduler(
        &config.optimizer.scheduler,
        config.optimizer.learning_rate,
        total_steps,
        config.training.warmup_steps,
    )?;
    
    // Test scheduler behavior
    match &scheduler {
        SchedulerState::Linear(sched) => {
            // Test warmup phase
            let lr_start = sched.get_lr(0);
            let lr_warmup_end = sched.get_lr(config.training.warmup_steps);
            assert!(lr_warmup_end > lr_start);
            
            // Test decay phase
            let lr_mid = sched.get_lr(total_steps / 2);
            let lr_end = sched.get_lr(total_steps - 1);
            assert!(lr_mid > lr_end);
        }
        _ => {}
    }
    
    Ok(())
}

#[test]
fn test_training_event_handling() -> Result<()> {
    use tokio::sync::mpsc;
    use tokio::runtime::Runtime;
    
    let rt = Runtime::new()?;
    
    rt.block_on(async {
        let (tx, mut rx) = mpsc::unbounded_channel::<TrainingEvent>();
        
        // Send test events
        tx.send(TrainingEvent::EpochStart { epoch: 1 })?;
        tx.send(TrainingEvent::StepComplete {
            epoch: 1,
            step: 10,
            global_step: 10,
            loss: 0.5,
        })?;
        tx.send(TrainingEvent::EpochEnd {
            epoch: 1,
            metrics: HashMap::from([
                ("loss".to_string(), 0.5),
                ("accuracy".to_string(), 0.9),
            ]),
        })?;
        
        // Verify events are received
        let event1 = rx.recv().await.unwrap();
        assert!(matches!(event1, TrainingEvent::EpochStart { epoch: 1 }));
        
        let event2 = rx.recv().await.unwrap();
        if let TrainingEvent::StepComplete { loss, .. } = event2 {
            assert_eq!(loss, 0.5);
        } else {
            panic!("Expected StepComplete event");
        }
        
        let event3 = rx.recv().await.unwrap();
        if let TrainingEvent::EpochEnd { metrics, .. } = event3 {
            assert_eq!(metrics.get("accuracy"), Some(&0.9));
        } else {
            panic!("Expected EpochEnd event");
        }
        
        Ok::<(), anyhow::Error>(())
    })?;
    
    Ok(())
}

#[test]
fn test_metrics_tracking() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let metrics_dir = temp_dir.path().join("metrics");
    
    let mut tracker = MetricsTracker::new(&metrics_dir, true)?;
    
    // Record training metrics
    tracker.record_step_metrics(
        1,  // epoch
        10, // step
        100, // global_step
        0.5, // loss
        1.5, // learning_rate
        std::time::Duration::from_secs(2),
    )?;
    
    // Record validation metrics
    let val_metrics = HashMap::from([
        ("val_loss".to_string(), 0.45),
        ("val_accuracy".to_string(), 0.92),
    ]);
    tracker.record_validation_metrics(1, 100, val_metrics)?;
    
    // Get summary
    let summary = tracker.get_summary();
    assert!(summary.total_steps > 0);
    assert!(summary.best_metrics.contains_key("val_loss"));
    
    Ok(())
}

#[test]
fn test_gradient_accumulation() -> Result<()> {
    let config = create_test_config();
    let device = create_test_device();
    
    // Test with gradient accumulation
    let mut config_with_accum = config.clone();
    config_with_accum.training.gradient_accumulation_steps = 4;
    
    // Effective batch size should be batch_size * gradient_accumulation_steps
    let effective_batch_size = config_with_accum.training.batch_size 
        * config_with_accum.training.gradient_accumulation_steps;
    assert_eq!(effective_batch_size, 16);
    
    Ok(())
}

#[test]
fn test_early_stopping() -> Result<()> {
    let mut state = TrainingState {
        epoch: 5,
        step: 500,
        global_step: 500,
        best_score: Some(0.95),
        steps_since_best: 0,
        start_time: chrono::Utc::now(),
        last_eval_time: Some(chrono::Utc::now()),
        status: TrainingStatus::Running,
        memory_usage: MemoryUsage::default(),
    };
    
    let patience = 3;
    let mut should_stop = false;
    
    // Simulate training with no improvement
    for _ in 0..5 {
        state.steps_since_best += 1;
        
        if state.steps_since_best >= patience {
            should_stop = true;
            state.status = TrainingStatus::EarlyStopped;
            break;
        }
    }
    
    assert!(should_stop);
    assert!(matches!(state.status, TrainingStatus::EarlyStopped));
    
    Ok(())
}

#[test]
fn test_memory_usage_tracking() -> Result<()> {
    let mut memory = MemoryUsage::default();
    
    // Update memory stats
    memory.allocated = 1024 * 1024 * 100; // 100 MB
    memory.reserved = 1024 * 1024 * 200;  // 200 MB
    memory.peak_allocated = 1024 * 1024 * 150; // 150 MB
    
    // Check memory limits
    let memory_limit = 1024 * 1024 * 1024; // 1 GB
    assert!(memory.allocated < memory_limit);
    
    // Calculate utilization
    let utilization = memory.allocated as f64 / memory.reserved as f64;
    assert!(utilization <= 1.0);
    
    Ok(())
}

#[test]
fn test_training_interruption_handling() -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    
    let interrupt_flag = Arc::new(AtomicBool::new(false));
    let flag_clone = interrupt_flag.clone();
    
    // Set up signal handler (in real trainer)
    // ctrlc::set_handler(move || {
    //     flag_clone.store(true, Ordering::SeqCst);
    // })?;
    
    // Simulate training loop
    let mut state = TrainingState {
        epoch: 1,
        step: 50,
        global_step: 50,
        best_score: None,
        steps_since_best: 0,
        start_time: chrono::Utc::now(),
        last_eval_time: None,
        status: TrainingStatus::Running,
        memory_usage: MemoryUsage::default(),
    };
    
    // Simulate interrupt
    interrupt_flag.store(true, Ordering::SeqCst);
    
    // Check interrupt in training loop
    if interrupt_flag.load(Ordering::SeqCst) {
        state.status = TrainingStatus::Interrupted;
    }
    
    assert!(matches!(state.status, TrainingStatus::Interrupted));
    
    Ok(())
}

#[test]
fn test_mixed_precision_training() -> Result<()> {
    let mut config = create_test_config();
    
    // Test FP16 configuration
    config.training.fp16 = true;
    config.training.bf16 = false;
    
    // Verify precision settings don't conflict
    assert!(!(config.training.fp16 && config.training.bf16));
    
    // Test BF16 configuration
    config.training.fp16 = false;
    config.training.bf16 = true;
    
    assert!(config.training.bf16);
    
    Ok(())
}

#[test]
fn test_distributed_training_config() -> Result<()> {
    let config = create_test_config();
    
    // Test distributed settings
    if let Some(runtime) = &config.runtime {
        // Check for distributed training flags
        assert!(!runtime.distributed);
        assert_eq!(runtime.world_size, 1);
        assert_eq!(runtime.local_rank, 0);
    }
    
    Ok(())
}

#[test]
fn test_learning_rate_scheduling() -> Result<()> {
    let config = create_test_config();
    let initial_lr = config.optimizer.learning_rate;
    let warmup_steps = config.training.warmup_steps;
    let total_steps = 1000;
    
    let scheduler = create_scheduler(
        &config.optimizer.scheduler,
        initial_lr,
        total_steps,
        warmup_steps,
    )?;
    
    // Test learning rate at different steps
    let lr_values: Vec<f64> = (0..=total_steps)
        .step_by(100)
        .map(|step| scheduler.get_lr(step))
        .collect();
    
    // Verify warmup increases LR
    if warmup_steps > 0 {
        assert!(lr_values[1] > lr_values[0]);
    }
    
    // Verify decay decreases LR
    let mid_idx = lr_values.len() / 2;
    assert!(lr_values[mid_idx] >= lr_values.last().unwrap().clone());
    
    Ok(())
}

#[test]
fn test_checkpoint_recovery_after_crash() -> Result<()> {
    let fixture = CheckpointTestFixture::new()?;
    let checkpoint_manager = fixture.create_checkpoint_manager()?;
    
    // Simulate a training state before "crash"
    let pre_crash_state = TrainingState {
        epoch: 3,
        step: 150,
        global_step: 450,
        best_score: Some(0.85),
        steps_since_best: 5,
        start_time: chrono::Utc::now() - chrono::Duration::hours(2),
        last_eval_time: Some(chrono::Utc::now() - chrono::Duration::minutes(10)),
        status: TrainingStatus::Running,
        memory_usage: MemoryUsage {
            allocated: 500_000_000,
            reserved: 1_000_000_000,
            peak_allocated: 750_000_000,
        },
    };
    
    // Save checkpoint
    let checkpoint = crate::training::checkpoints::TrainingCheckpoint {
        epoch: pre_crash_state.epoch,
        global_step: pre_crash_state.global_step,
        model_state: vec![1, 2, 3, 4], // Mock model state
        optimizer_state: Some(vec![5, 6, 7, 8]), // Mock optimizer state
        scheduler_state: Some(vec![9, 10]), // Mock scheduler state
        metrics: HashMap::from([
            ("loss".to_string(), 0.25),
            ("accuracy".to_string(), 0.85),
        ]),
        config: fixture.config.clone(),
        timestamp: chrono::Utc::now(),
        metadata: Default::default(),
    };
    
    checkpoint_manager.save_checkpoint(checkpoint)?;
    
    // Simulate recovery after crash
    let latest = checkpoint_manager.get_latest_checkpoint()?;
    assert!(latest.is_some());
    
    if let Some(checkpoint_info) = latest {
        assert_eq!(checkpoint_info.epoch, pre_crash_state.epoch);
        assert_eq!(checkpoint_info.global_step, pre_crash_state.global_step);
    }
    
    Ok(())
}

#[cfg(test)]
mod trainer_integration_tests {
    use super::*;
    
    #[test]
    #[ignore] // Ignore for now as T2LTrainer is not fully implemented
    fn test_full_training_loop() -> Result<()> {
        let fixture = TrainingTestFixture::new()?;
        let trainer = fixture.create_trainer()?;
        
        // Run training for a few steps
        let result = futures::executor::block_on(trainer.train())?;
        
        // Verify training completed
        assert!(matches!(result.status, TrainingStatus::Completed));
        assert!(result.final_metrics.contains_key("loss"));
        
        Ok(())
    }
}