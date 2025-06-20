//! Specialized trainers for different T2L training modes
//!
//! This module provides specialized trainer implementations optimized for
//! specific training scenarios such as reconstruction and supervised training.

pub mod reconstruction;
pub mod supervised;

#[cfg(test)]
mod tests;

// Re-exports
pub use reconstruction::{ReconstructionTrainer, ReconstructionTrainerConfig};
pub use supervised::{SupervisedTrainer, SupervisedTrainerConfig};