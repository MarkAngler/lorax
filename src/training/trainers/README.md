# T2L Trainers

This module provides specialized trainer implementations optimized for different T2L training scenarios.

## ReconstructionTrainer

The `ReconstructionTrainer` is a specialized trainer for reconstruction-based T2L training, where the hypernetwork learns to generate LoRA parameters by reconstructing existing pre-trained LoRA weights.

### Key Features

- **Specialized Loss Computation**: Integrates with `ReconstructionLoss` for MSE-based parameter reconstruction
- **Mixed Precision Training**: Built-in support for gradient scaling and mixed precision operations
- **Gradient Accumulation**: Flexible gradient accumulation with fixed, dynamic, or adaptive modes
- **Progressive Training**: Optional layer-wise progressive training schedule
- **Comprehensive Metrics**: Tracks parameter magnitudes, gradient norms, alignment scores, and effective ranks
- **Layer-wise Weighting**: Support for uniform, parameter-count-based, depth-based, or custom layer weights

### Configuration

```rust
use lorax::training::trainers::reconstruction::{
    ReconstructionTrainerConfig, ReconstructionSettings,
    LayerWeightingStrategy, GradientAccumulationMode,
    ValidationSettings, MetricsSettings
};

let config = ReconstructionTrainerConfig {
    base_config: TrainingConfig::reconstruction_default(),
    reconstruction: ReconstructionSettings {
        layer_weighting: LayerWeightingStrategy::Uniform,
        gradient_accumulation_mode: GradientAccumulationMode::Fixed { steps: 4 },
        progressive_training: None,
        param_norm_clip: Some(1.0),
        track_param_magnitudes: true,
        analyze_gradient_flow: false,
    },
    validation: ValidationSettings {
        compute_alignment: true,
        compute_effective_rank: false,
        track_layer_accuracy: true,
        best_model_metric: "eval_loss".to_string(),
    },
    metrics: MetricsSettings {
        log_param_stats_interval: 100,
        log_gradient_flow_interval: 500,
        track_memory_usage: true,
        enable_profiling: false,
    },
};
```

### Usage Example

```rust
use lorax::training::{
    ReconstructionTrainer, ReconstructionDataset, DataLoader
};
use std::sync::Arc;
use parking_lot::RwLock;

// Create hypernetwork model
let model = Arc::new(RwLock::new(HyperNetwork::new(config)?));

// Create datasets and data loaders
let train_dataset = ReconstructionDataset::new(train_path, device.clone(), None, true)?;
let train_loader = DataLoader::new(Arc::new(train_dataset), loader_config, device.clone())?;

// Create trainer
let mut trainer = ReconstructionTrainer::new(
    trainer_config,
    model,
    train_loader,
    val_loader,
    device,
)?;

// Start training
let result = trainer.train().await?;
```

### Gradient Accumulation Modes

1. **Fixed**: Accumulate gradients for a fixed number of steps
   ```rust
   GradientAccumulationMode::Fixed { steps: 4 }
   ```

2. **Dynamic**: Adjust accumulation based on memory usage
   ```rust
   GradientAccumulationMode::Dynamic { target_memory_mb: 8192 }
   ```

3. **Adaptive**: Adjust based on gradient magnitudes
   ```rust
   GradientAccumulationMode::Adaptive { min_steps: 1, max_steps: 8 }
   ```

### Layer Weighting Strategies

1. **Uniform**: Equal weight for all layers
2. **ByParameterCount**: Weight proportional to parameter count
3. **ByDepth**: Earlier layers get more weight with exponential decay
4. **Custom**: Specify custom weights per layer

### Metrics Tracked

- **Loss Metrics**:
  - Total reconstruction loss
  - Per-layer reconstruction losses
  - Consistency regularization loss
  - Sparsity regularization loss

- **Parameter Metrics**:
  - Parameter magnitudes by layer
  - Gradient norms by layer
  - Update ratios (gradient_norm / param_norm)
  - Parameter alignment scores
  - Effective rank estimates

- **Training Metrics**:
  - Learning rate
  - Training/validation loss
  - Memory usage
  - Step timing

### Progressive Training

Enable progressive training to start with a subset of layers and gradually add more:

```rust
use lorax::training::trainers::reconstruction::ProgressiveTrainingConfig;

let progressive_config = Some(ProgressiveTrainingConfig {
    initial_layers: vec!["attention.q_proj".to_string(), "attention.v_proj".to_string()],
    layer_addition_interval: 2, // Add layers every 2 epochs
    layer_warmup_steps: 100,    // Warmup for new layers
});
```

### Best Practices

1. **Start with Small Learning Rate**: Reconstruction training can be sensitive to learning rates
2. **Use Gradient Clipping**: Enable global norm clipping to prevent gradient explosions
3. **Monitor Parameter Alignment**: Track alignment metrics to ensure generated parameters match targets
4. **Adjust Layer Weights**: Fine-tune layer weights based on reconstruction accuracy
5. **Enable Mixed Precision**: Use mixed precision for faster training and reduced memory usage

### Integration with Other Components

- **Data Loading**: Works with `ReconstructionDataset` and `ReconstructionBatch`
- **Loss Functions**: Integrates with `ReconstructionLoss` and supports various matrix loss types
- **Checkpointing**: Full integration with `CheckpointManager` for robust model saving
- **Metrics**: Compatible with `MetricsTracker` for comprehensive monitoring