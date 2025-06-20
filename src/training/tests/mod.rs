//! Comprehensive test suite for T2L training infrastructure
//!
//! This module provides extensive testing coverage for all components of the
//! training pipeline including data loading, loss functions, optimization,
//! checkpointing, and end-to-end training loops.

// Test modules
pub mod data_tests;
pub mod loss_tests;
pub mod trainer_tests;
pub mod checkpoint_tests;
pub mod integration_tests;

// Utility modules for testing
pub mod test_utils;
pub mod mock_data;
pub mod fixtures;

// Re-export commonly used test utilities
pub use test_utils::{
    create_test_device,
    create_test_config,
    create_test_model,
    assert_tensor_close,
    assert_metrics_valid,
};

pub use mock_data::{
    create_mock_dataset,
    create_mock_batch,
    create_mock_lora_params,
    create_mock_embeddings,
};

pub use fixtures::{
    TestFixture,
    TrainingTestFixture,
    DataTestFixture,
    CheckpointTestFixture,
};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_imports() {
        // Ensure all test modules compile and are accessible
        let _ = std::hint::black_box(data_tests::test_dataset_loading);
        let _ = std::hint::black_box(loss_tests::test_loss_functions);
        let _ = std::hint::black_box(trainer_tests::test_trainer_creation);
        let _ = std::hint::black_box(checkpoint_tests::test_checkpoint_save_load);
        let _ = std::hint::black_box(integration_tests::test_full_training_loop);
    }
}