//! Training infrastructure for T2L models
//!
//! This module provides comprehensive training infrastructure including data loading,
//! training loops, optimization, checkpointing, metrics tracking, and monitoring
//! components for training Text-to-LoRA models.
//!
//! # Main Components
//!
//! - **Configuration**: Comprehensive training configuration system
//! - **Trainer**: Base trainer framework with support for reconstruction and supervised training
//! - **Data**: Data loading and batch processing infrastructure
//! - **Checkpoints**: Robust checkpointing system with metadata and recovery
//! - **Metrics**: Real-time metrics collection and monitoring
//! - **Optimizers**: AdamW, SGD, and other optimizers with scheduling
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use lorax::training::{TrainingConfig, T2LTrainer, DataLoader};
//! use candle_core::Device;
//!
//! // Create training configuration
//! let config = TrainingConfig::reconstruction_default();
//!
//! // Initialize trainer
//! let trainer = T2LTrainer::new(
//!     config,
//!     model,
//!     train_loader,
//!     val_loader,
//!     Device::Cpu,
//! )?;
//!
//! // Start training
//! let result = trainer.train().await?;
//! ```

pub mod config;
pub mod trainer;
pub mod trainers;
pub mod data;
pub mod checkpoints;
pub mod metrics;
pub mod optimizers;
pub mod loss;

// Tests module
#[cfg(test)]
pub mod tests;

// Configuration re-exports
pub use config::{
    TrainingConfig, ModelConfig, OptimizerConfig, TrainingParams, 
    DataConfig, CheckpointingConfig, LoggingConfig, MixedPrecisionConfig,
    RegularizationConfig, RuntimeConfig, TrainingType, ModelArchitecture,
    OptimizerType, SchedulerType, SchedulerConfig, DeviceType
};

// Trainer re-exports
pub use trainer::{
    T2LTrainer, TrainingState, TrainingStatus, TrainingResult, 
    TrainingEvent, MemoryUsage
};

// Data re-exports
pub use data::{
    DataLoader, ReconstructionDataset, SupervisedDataset,
    BatchCollator, ReconstructionBatch, SupervisedBatch
};

// Checkpoint re-exports
pub use checkpoints::{
    CheckpointManager, TrainingCheckpoint, CheckpointInfo, CheckpointMetadata,
    RecoveryManager, RecoveryOptions, RecoveryResult, ValidationResult
};

// Metrics re-exports
pub use metrics::{
    MetricsTracker, TrainingMetrics, MetricValue, LossMetrics, 
    PerformanceMetrics, ModelMetrics, SystemMetrics, OptimizationMetrics,
    ValidationMetrics, MetricsCollector, MetricsExporter, MetricsAggregator
};

// Optimizer re-exports
pub use optimizers::{
    AdamWOptimizer, SGDOptimizer, OptimizerState, SchedulerState,
    LinearScheduler, CosineScheduler, ExponentialScheduler, StepScheduler,
    ConstantScheduler, Optimizer, Scheduler, create_optimizer, create_scheduler
};

// Loss re-exports
pub use loss::{
    LossFunction, LossConfig, LossType, MatrixLossType, SupervisedTaskType,
    TaskBalancingStrategy, LoraRegularizationConfig,
    LossScalingConfig, StabilityConfig, ReductionMethod, LayerLossConfig,
    ReconstructionLoss, ReconstructionLossConfig, SupervisedLoss, SupervisedLossConfig,
    SupervisedPredictions, RegularizationPenalties, LoraRegularizer, LossUtils, 
    NumericalStability, MetricsComputer, CompositeLoss, TaskWeightManager
};

// Re-export loss-specific LossMetrics with a different name to avoid conflict
pub use loss::LossMetrics as LossFunctionMetrics;

// Re-export PredictedLoraParams types from loss module
pub use loss::reconstruction::{PredictedLoraParams, PredictedLoraLayer};

// Trainers re-exports
pub use trainers::{ReconstructionTrainer, ReconstructionTrainerConfig, SupervisedTrainer, SupervisedTrainerConfig};

/// Training result type alias
pub type Result<T> = anyhow::Result<T>;

/// Training error type alias
pub type Error = anyhow::Error;