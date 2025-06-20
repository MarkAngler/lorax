//! SIMD-accelerated operations for T2L (Text-to-LoRA)
//! 
//! This module provides highly optimized SIMD implementations for
//! vector and matrix operations critical to T2L performance.

#![feature(portable_simd)]
#![feature(stdsimd)]

use std::arch::x86_64::*;
use bytemuck::{Pod, Zeroable};
use half::f16;

pub mod ops;
pub mod matmul;
pub mod quantize;
pub mod memory;

/// SIMD vector width detection
#[derive(Debug, Clone, Copy)]
pub enum SimdCapability {
    Avx512,
    Avx2,
    Sse4,
    Neon,
    Scalar,
}

impl SimdCapability {
    /// Detect the best available SIMD capability
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdCapability::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdCapability::Avx2;
            }
            if is_x86_feature_detected!("sse4.1") {
                return SimdCapability::Sse4;
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return SimdCapability::Neon;
            }
        }
        
        SimdCapability::Scalar
    }
    
    /// Get the vector width in f32 elements
    pub fn vector_width_f32(&self) -> usize {
        match self {
            SimdCapability::Avx512 => 16,
            SimdCapability::Avx2 => 8,
            SimdCapability::Sse4 => 4,
            SimdCapability::Neon => 4,
            SimdCapability::Scalar => 1,
        }
    }
    
    /// Get the vector width in f16 elements
    pub fn vector_width_f16(&self) -> usize {
        self.vector_width_f32() * 2
    }
}

/// Aligned memory allocation for SIMD operations
#[repr(C, align(64))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AlignedVec<T: Pod, const N: usize> {
    data: [T; N],
}

impl<T: Pod, const N: usize> AlignedVec<T, N> {
    pub fn new() -> Self {
        Self { data: [T::zeroed(); N] }
    }
    
    pub fn from_slice(slice: &[T]) -> Self {
        let mut aligned = Self::new();
        aligned.data[..slice.len()].copy_from_slice(slice);
        aligned
    }
}

/// Global SIMD capability detector
pub static SIMD_CAPABILITY: once_cell::sync::Lazy<SimdCapability> = 
    once_cell::sync::Lazy::new(SimdCapability::detect);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_detection() {
        let cap = SimdCapability::detect();
        println!("Detected SIMD capability: {:?}", cap);
        assert!(cap.vector_width_f32() >= 1);
    }
}