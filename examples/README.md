# T2L Examples

This directory contains practical examples demonstrating various T2L use cases and training patterns.

## Training Examples

The `training/` subdirectory contains complete Rust examples for different training scenarios:

### Core Training Examples

- [reconstruction_training.rs](reconstruction_training.rs) - Train T2L by reconstructing known LoRA adapters
- [supervised_training.rs](training/supervised_training.rs) - End-to-end supervised fine-tuning
- [multi_task_training.rs](training/multi_task_training.rs) - Multi-task learning across different domains

### Configuration Examples

The `training/configs/` directory contains sample configuration files:

- [reconstruction_config.toml](training/configs/reconstruction_config.toml) - Configuration for reconstruction training
- [supervised_config.toml](training/configs/supervised_config.toml) - Supervised training configuration
- [multi_task_config.toml](training/configs/multi_task_config.toml) - Multi-task training setup

## Running the Examples

### Quick Start

```bash
# Run reconstruction training example
cargo run --example reconstruction_training

# Run supervised training with custom config
cargo run --example supervised_training -- --config examples/training/configs/supervised_config.toml

# Run multi-task training
cargo run --example multi_task_training
```

### With CLI

After installing the T2L CLI:

```bash
# Generate an adapter
t2l generate --task "Translate English to French" --output translator.safetensors

# Train a new model
t2l train --mode reconstruction --config examples/training/configs/reconstruction_config.toml

# Export to different formats
t2l export --adapter translator.safetensors --format peft --output ./peft_export/
```

## Requirements

- Rust 1.70+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB+ RAM for training
- 4GB+ VRAM for GPU training

## Python Support Deprecated

**Note**: Python examples have been removed. T2L is now a pure Rust implementation. For information about migrating from Python, see [MIGRATION.md](../MIGRATION.md).

## Example Structure

Each example is self-contained and demonstrates:

1. **Configuration** - How to set up training parameters
2. **Data Loading** - Loading and preprocessing training data
3. **Model Setup** - Initializing the hypernetwork
4. **Training Loop** - Complete training implementation
5. **Checkpointing** - Saving and resuming training
6. **Evaluation** - Validating model performance

## Advanced Usage

For production deployments and advanced patterns, see:

- [Training Guide](../docs/training/training-guide.md) - Comprehensive training documentation
- [Configuration Reference](../docs/training/configuration.md) - All configuration options
- [Performance Guide](../docs/training/performance.md) - Optimization tips
- [CLI Reference](../docs/cli-reference.md) - Complete CLI documentation