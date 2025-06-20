# T2L Training Examples

This directory contains comprehensive examples for training T2L (Text-to-LoRA) models using different approaches.

## Examples

### 1. Reconstruction Training (`reconstruction_training.rs`)

Learn to generate LoRA parameters by reconstructing known good adapters.

```bash
# Run with default configuration
cargo run --example reconstruction_training

# Run with custom config
cargo run --example reconstruction_training -- --config configs/reconstruction_config.toml
```

**Use Cases:**
- Initial hypernetwork pretraining
- Learning the mapping from task descriptions to LoRA parameters
- Creating a foundation model for further fine-tuning

### 2. Supervised Training (`supervised_training.rs`)

End-to-end training with downstream tasks, where the hypernetwork generates LoRA adapters that are immediately applied to a frozen base model.

```bash
# Run supervised training
cargo run --example supervised_training

# With custom configuration
cargo run --example supervised_training -- --config configs/supervised_config.toml
```

**Use Cases:**
- Task-specific optimization
- Domain adaptation
- Creating specialized adapters

### 3. Multi-Task Training (`multi_task_training.rs`)

Train a single hypernetwork to handle multiple tasks simultaneously with dynamic task balancing.

```bash
# Run multi-task training
cargo run --example multi_task_training

# With distributed training
cargo run --example multi_task_training -- --distributed --world-size 8
```

**Use Cases:**
- Building versatile models
- Leveraging task synergies
- Efficient parameter sharing across tasks

## Configuration Files

The `configs/` directory contains example configurations:

- `reconstruction_config.toml` - Configuration for reconstruction training
- `supervised_config.toml` - Configuration for supervised fine-tuning
- `multi_task_config.toml` - Configuration for multi-task learning

## Quick Start

1. **Prepare your data:**
   ```bash
   # Convert your data to the appropriate format
   # See the Data Format section below for required schemas
   
   # Example: Create training data directory
   mkdir -p data/train data/val
   ```

2. **Choose a training mode:**
   - Reconstruction: Best for initial training
   - Supervised: Best for task-specific performance
   - Multi-task: Best for versatile models

3. **Run training:**
   ```bash
   # Example: supervised training with custom settings
   cargo run --example supervised_training -- \
       --batch-size 8 \
       --learning-rate 3e-5 \
       --num-epochs 5
   ```

4. **Monitor progress:**
   - Training logs are output to the console
   - Checkpoints are saved to `checkpoints/`
   - Metrics are tracked in the training output

## Data Format

### Reconstruction Training Data
```json
{
  "task_description": "Classify movie reviews as positive or negative",
  "task_embedding": [0.1, 0.2, ...],  // 768-dim vector
  "lora_parameters": {
    "layer_0": {
      "lora_A": [[...], ...],
      "lora_B": [[...], ...]
    }
  }
}
```

### Supervised Training Data
```json
{
  "task_description": "Sentiment analysis of product reviews",
  "task_id": "sentiment",
  "input": "This product is amazing!",
  "target": "positive"
}
```

### Multi-Task Training Data
```json
{
  "task_id": "qa",
  "task_type": "question_answering",
  "question": "What is the capital of France?",
  "context": "France is a country in Europe...",
  "answer": "Paris"
}
```

## Advanced Usage

### Custom Training Loop

```rust
use lorax::training::{TrainingConfig, T2LTrainer};

// Load configuration
let config = TrainingConfig::from_file("my_config.toml")?;

// Create custom trainer
let trainer = T2LTrainer::new(config, model, train_loader, val_loader, device)?;

// Add custom callbacks
trainer.add_callback(Box::new(MyCustomCallback));

// Start training
let result = trainer.train().await?;
```

### Distributed Training

```bash
# Distributed training is built into the Rust implementation
# Use the --distributed flag with appropriate settings
cargo run --example multi_task_training -- \
    --distributed \
    --world-size 8 \
    --rank 0 \
    --master-addr "10.0.0.1"
```

### Memory Optimization

For large models or limited GPU memory:

1. Enable gradient checkpointing
2. Use mixed precision training
3. Reduce batch size and increase gradient accumulation
4. Enable CPU offloading

See the configuration files for examples.

## Performance Benchmarks

| Training Mode | Model Size | GPU | Throughput | Memory Usage |
|--------------|------------|-----|------------|--------------|
| Reconstruction | Medium | A100 40GB | 500 samples/s | 12GB |
| Supervised | Large | A100 80GB | 200 samples/s | 35GB |
| Multi-Task | XLarge | 8xA100 | 1500 samples/s | 65GB/GPU |

## Troubleshooting

### Out of Memory
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision (bf16)
- Enable CPU offloading

### Slow Training
- Check data loading (use more workers)
- Enable mixed precision (bf16)
- Enable CUDA optimizations
- Profile with included profiling tools

### Poor Convergence
- Adjust learning rate
- Check data quality
- Try different initialization
- Use gradient clipping

## Next Steps

1. Read the [Training Guide](../../docs/training/training-guide.md)
2. Check [Configuration Reference](../../docs/training/configuration.md)
3. See [Performance Optimization](../../docs/training/performance.md)
4. Review [Troubleshooting Guide](../../docs/training/troubleshooting.md)

## Contributing

Feel free to add more examples! See the contribution guidelines in the main repository.