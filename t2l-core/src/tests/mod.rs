//! Comprehensive test suite for T2L (Text-to-LoRA) implementation
//!
//! This module contains unit tests, integration tests, and benchmarks
//! for all T2L components including hypernetwork, encoder, and training.

#[cfg(test)]
pub mod unit;

#[cfg(test)]
pub mod integration;

#[cfg(test)]
pub mod benchmarks;

#[cfg(test)]
pub mod evaluation;