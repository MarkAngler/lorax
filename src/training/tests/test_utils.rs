//! Common utilities for testing the training pipeline

use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

use crate::training::{
    TrainingConfig, ModelConfig, OptimizerConfig, TrainingParams,
    DataConfig, CheckpointingConfig, TrainingType, ModelArchitecture,
    OptimizerType, DeviceType,
};
use crate::training::data::{Dataset, DataLoader, DataLoaderConfig, BatchCollator};

/// Create a test device (CPU for CI compatibility)
pub fn create_test_device() -> Device {
    Device::Cpu
}

/// Create a minimal test configuration
pub fn create_test_config() -> TrainingConfig {
    TrainingConfig {
        name: "test_training".to_string(),
        model: ModelConfig {
            architecture: ModelArchitecture::BertT2L,
            hidden_size: 64,  // Small for testing
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            lora_rank: 4,
            lora_alpha: 16.0,
            lora_dropout: 0.0,
            checkpoint_path: None,
            gradient_checkpointing: false,
            flash_attention: false,
        },
        optimizer: OptimizerConfig {
            optimizer_type: OptimizerType::AdamW,
            learning_rate: 1e-3,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            momentum: None,
            gradient_clip_value: Some(1.0),
            gradient_clip_norm: Some(1.0),
        },
        training: TrainingParams {
            num_epochs: 2,
            batch_size: 4,
            gradient_accumulation_steps: 1,
            warmup_steps: 10,
            logging_steps: 5,
            eval_steps: 10,
            save_steps: 20,
            eval_strategy: "steps".to_string(),
            save_strategy: "steps".to_string(),
            training_type: TrainingType::Reconstruction,
            fp16: false,
            bf16: false,
            max_grad_norm: 1.0,
            seed: 42,
            dataloader_num_workers: 0,
            remove_unused_columns: true,
            label_smoothing_factor: 0.0,
        },
        data: DataConfig {
            train_path: "test_data/train.h5".to_string(),
            val_path: Some("test_data/val.h5".to_string()),
            test_path: None,
            max_seq_length: 128,
            preprocessing_num_workers: 1,
            overwrite_cache: true,
            cache_dir: None,
        },
        checkpointing: CheckpointingConfig {
            output_dir: "test_checkpoints".to_string(),
            resume_from_checkpoint: None,
            save_total_limit: Some(3),
            save_best_only: false,
            metric_for_best_model: Some("loss".to_string()),
            greater_is_better: false,
            load_best_model_at_end: false,
            checkpoint_compression: false,
        },
        ..Default::default()
    }
}

/// Create a mock model for testing
pub fn create_test_model(device: &Device) -> Result<VarMap> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    
    // Create minimal model parameters
    let hidden_size = 64;
    let num_heads = 4;
    let num_layers = 2;
    
    // Create encoder parameters
    for layer_idx in 0..num_layers {
        let layer_prefix = format!("encoder.layer.{}", layer_idx);
        
        // Self-attention
        vb.get_with_hints((hidden_size, hidden_size), &format!("{}.attention.self.query.weight", layer_prefix), candle_nn::init::DEFAULT_KAIMING_UNIFORM)?;
        vb.get_with_hints((hidden_size, hidden_size), &format!("{}.attention.self.key.weight", layer_prefix), candle_nn::init::DEFAULT_KAIMING_UNIFORM)?;
        vb.get_with_hints((hidden_size, hidden_size), &format!("{}.attention.self.value.weight", layer_prefix), candle_nn::init::DEFAULT_KAIMING_UNIFORM)?;
        vb.get_with_hints((hidden_size, hidden_size), &format!("{}.attention.output.dense.weight", layer_prefix), candle_nn::init::DEFAULT_KAIMING_UNIFORM)?;
        
        // FFN
        vb.get_with_hints((hidden_size * 4, hidden_size), &format!("{}.intermediate.dense.weight", layer_prefix), candle_nn::init::DEFAULT_KAIMING_UNIFORM)?;
        vb.get_with_hints((hidden_size, hidden_size * 4), &format!("{}.output.dense.weight", layer_prefix), candle_nn::init::DEFAULT_KAIMING_UNIFORM)?;
    }
    
    Ok(varmap)
}

/// Assert two tensors are close within tolerance
pub fn assert_tensor_close(a: &Tensor, b: &Tensor, rtol: f64, atol: f64) -> Result<()> {
    assert_eq!(a.shape(), b.shape(), "Tensor shapes must match");
    
    let diff = (a - b)?.abs()?;
    let a_abs = a.abs()?;
    let b_abs = b.abs()?;
    let max_abs = a_abs.maximum(&b_abs)?;
    
    let tol = max_abs.mul_scalar(rtol)?.add_scalar(atol)?;
    let is_close = diff.le(&tol)?;
    
    let all_close = is_close.flatten_all()?.to_vec1::<u8>()?
        .iter()
        .all(|&x| x == 1);
    
    assert!(all_close, "Tensors are not close within tolerance");
    Ok(())
}

/// Assert metrics are valid (no NaN/Inf values)
pub fn assert_metrics_valid(metrics: &HashMap<String, f64>) {
    for (name, value) in metrics {
        assert!(!value.is_nan(), "Metric {} is NaN", name);
        assert!(!value.is_infinite(), "Metric {} is infinite", name);
    }
}

/// Create a temporary directory for testing
pub fn create_temp_dir() -> Result<TempDir> {
    Ok(tempfile::tempdir()?)
}

/// Create test checkpoint path
pub fn create_test_checkpoint_dir() -> Result<PathBuf> {
    let temp_dir = create_temp_dir()?;
    let checkpoint_dir = temp_dir.path().join("checkpoints");
    std::fs::create_dir_all(&checkpoint_dir)?;
    Ok(checkpoint_dir)
}

/// Generate random tensor with given shape
pub fn random_tensor(shape: &[usize], device: &Device) -> Result<Tensor> {
    use rand::{thread_rng, Rng};
    let mut rng = thread_rng();
    
    let total_size: usize = shape.iter().product();
    let data: Vec<f32> = (0..total_size)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    Tensor::from_vec(data, shape, device)
}

/// Create a small embedding tensor for testing
pub fn create_test_embeddings(batch_size: usize, hidden_size: usize, device: &Device) -> Result<Tensor> {
    random_tensor(&[batch_size, hidden_size], device)
}

/// Create test LoRA parameters
pub fn create_test_lora_params(
    num_layers: usize,
    hidden_size: usize,
    rank: usize,
    device: &Device,
) -> Result<HashMap<String, (Tensor, Tensor)>> {
    let mut params = HashMap::new();
    
    for i in 0..num_layers {
        let layer_name = format!("layer_{}", i);
        let a = random_tensor(&[hidden_size, rank], device)?;
        let b = random_tensor(&[rank, hidden_size], device)?;
        params.insert(layer_name, (a, b));
    }
    
    Ok(params)
}

/// Setup test logging
pub fn setup_test_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("lorax=debug")
        .with_test_writer()
        .try_init();
}

/// Create a DataLoader for testing
pub fn create_test_dataloader(
    dataset: Arc<dyn Dataset>,
    batch_size: usize,
    shuffle: bool,
    num_workers: usize,
) -> Result<Box<dyn std::any::Any>> {
    // This is a workaround for the generic DataLoader issue in tests
    // In real usage, you would use DataLoader<YourDatasetType>::new()
    Ok(Box::new(()))
}

/// Property-based testing utilities
pub mod prop_test {
    use proptest::prelude::*;
    use super::*;
    
    /// Strategy for generating tensor shapes
    pub fn tensor_shape_strategy() -> impl Strategy<Value = Vec<usize>> {
        prop::collection::vec(1usize..=64, 1..=4)
    }
    
    /// Strategy for generating learning rates
    pub fn learning_rate_strategy() -> impl Strategy<Value = f64> {
        (1e-5..1e-1).prop_map(|x| x)
    }
    
    /// Strategy for generating batch sizes
    pub fn batch_size_strategy() -> impl Strategy<Value = usize> {
        1usize..=32
    }
    
    /// Strategy for generating LoRA ranks
    pub fn lora_rank_strategy() -> impl Strategy<Value = usize> {
        1usize..=16
    }
}