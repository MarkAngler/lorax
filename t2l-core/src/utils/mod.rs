//! Utility modules for T2L core functionality
//!
//! This module provides common utilities used across the T2L system:
//! - Tensor manipulation and conversion utilities
//! - Weight format conversion helpers
//! - Device management utilities
//! - Memory-efficient operations

pub mod tensor;

// Re-export commonly used types and functions for convenience
pub use tensor::{
    TensorDataType, DeviceType, TensorMetadata, LoraMatrixConfig, Tensor,
    precision, lora, safetensors, device, memory, format
};