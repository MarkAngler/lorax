//! High-performance SIMD operations for T2L

use std::arch::x86_64::*;
use half::f16;
use bytemuck::Pod;

/// Vectorized addition for f32
#[inline]
pub unsafe fn add_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;
    
    // Process 8 elements at a time with AVX2
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        let result = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out.as_mut_ptr().add(idx), result);
    }
    
    // Handle remainder
    let offset = chunks * 8;
    for i in 0..remainder {
        out[offset + i] = a[offset + i] + b[offset + i];
    }
}

/// Vectorized multiplication for f32
#[inline]
pub unsafe fn mul_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;
    
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        let result = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out.as_mut_ptr().add(idx), result);
    }
    
    let offset = chunks * 8;
    for i in 0..remainder {
        out[offset + i] = a[offset + i] * b[offset + i];
    }
}

/// Fused multiply-add for f32
#[inline]
pub unsafe fn fma_f32_avx2(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    assert_eq!(a.len(), out.len());
    
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;
    
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        let vc = _mm256_loadu_ps(c.as_ptr().add(idx));
        let result = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(out.as_mut_ptr().add(idx), result);
    }
    
    let offset = chunks * 8;
    for i in 0..remainder {
        out[offset + i] = a[offset + i].mul_add(b[offset + i], c[offset + i]);
    }
}

/// Vectorized ReLU activation
#[inline]
pub unsafe fn relu_f32_avx2(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    
    let zero = _mm256_setzero_ps();
    let chunks = input.len() / 8;
    let remainder = input.len() % 8;
    
    for i in 0..chunks {
        let idx = i * 8;
        let v = _mm256_loadu_ps(input.as_ptr().add(idx));
        let result = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(output.as_mut_ptr().add(idx), result);
    }
    
    let offset = chunks * 8;
    for i in 0..remainder {
        output[offset + i] = input[offset + i].max(0.0);
    }
}

/// Vectorized LayerNorm
pub unsafe fn layer_norm_f32_avx2(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
) {
    assert_eq!(input.len(), output.len());
    assert_eq!(input.len(), gamma.len());
    assert_eq!(input.len(), beta.len());
    
    let n = input.len();
    
    // Compute mean
    let mut sum = _mm256_setzero_ps();
    let chunks = n / 8;
    
    for i in 0..chunks {
        let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
        sum = _mm256_add_ps(sum, v);
    }
    
    // Horizontal sum
    let mut mean_arr = [0f32; 8];
    _mm256_storeu_ps(mean_arr.as_mut_ptr(), sum);
    let mut mean = mean_arr.iter().sum::<f32>();
    
    // Add remainder
    for i in (chunks * 8)..n {
        mean += input[i];
    }
    mean /= n as f32;
    
    // Compute variance
    let mean_vec = _mm256_set1_ps(mean);
    let mut var_sum = _mm256_setzero_ps();
    
    for i in 0..chunks {
        let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
        let diff = _mm256_sub_ps(v, mean_vec);
        let squared = _mm256_mul_ps(diff, diff);
        var_sum = _mm256_add_ps(var_sum, squared);
    }
    
    // Horizontal sum for variance
    _mm256_storeu_ps(mean_arr.as_mut_ptr(), var_sum);
    let mut variance = mean_arr.iter().sum::<f32>();
    
    // Add remainder
    for i in (chunks * 8)..n {
        let diff = input[i] - mean;
        variance += diff * diff;
    }
    variance = (variance / n as f32 + epsilon).sqrt();
    
    // Normalize and scale
    let inv_std = _mm256_set1_ps(1.0 / variance);
    
    for i in 0..chunks {
        let idx = i * 8;
        let v = _mm256_loadu_ps(input.as_ptr().add(idx));
        let centered = _mm256_sub_ps(v, mean_vec);
        let normalized = _mm256_mul_ps(centered, inv_std);
        let g = _mm256_loadu_ps(gamma.as_ptr().add(idx));
        let b = _mm256_loadu_ps(beta.as_ptr().add(idx));
        let scaled = _mm256_mul_ps(normalized, g);
        let result = _mm256_add_ps(scaled, b);
        _mm256_storeu_ps(output.as_mut_ptr().add(idx), result);
    }
    
    // Handle remainder
    for i in (chunks * 8)..n {
        let normalized = (input[i] - mean) / variance;
        output[i] = normalized * gamma[i] + beta[i];
    }
}

/// Optimized GELU activation using polynomial approximation
#[inline]
pub unsafe fn gelu_f32_avx2(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    
    // GELU approximation constants
    let c1 = _mm256_set1_ps(0.044715);
    let c2 = _mm256_set1_ps(0.7978845608);
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    
    let chunks = input.len() / 8;
    let remainder = input.len() % 8;
    
    for i in 0..chunks {
        let idx = i * 8;
        let x = _mm256_loadu_ps(input.as_ptr().add(idx));
        
        // Compute x^3
        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x2, x);
        
        // 0.044715 * x^3
        let term = _mm256_mul_ps(c1, x3);
        
        // x + 0.044715 * x^3
        let inner = _mm256_add_ps(x, term);
        
        // sqrt(2/pi) * (x + 0.044715 * x^3)
        let scaled = _mm256_mul_ps(c2, inner);
        
        // Fast tanh approximation
        let tanh = fast_tanh_avx2(scaled);
        
        // 1 + tanh(...)
        let sum = _mm256_add_ps(one, tanh);
        
        // 0.5 * x * (1 + tanh(...))
        let mul1 = _mm256_mul_ps(half, x);
        let result = _mm256_mul_ps(mul1, sum);
        
        _mm256_storeu_ps(output.as_mut_ptr().add(idx), result);
    }
    
    // Handle remainder
    let offset = chunks * 8;
    for i in 0..remainder {
        let x = input[offset + i];
        output[offset + i] = 0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() 
            * (x + 0.044715 * x * x * x)).tanh());
    }
}

/// Fast tanh approximation for AVX2
#[inline]
unsafe fn fast_tanh_avx2(x: __m256) -> __m256 {
    // Clamp input to [-4.97, 4.97] for numerical stability
    let max_val = _mm256_set1_ps(4.97);
    let min_val = _mm256_set1_ps(-4.97);
    let x_clamped = _mm256_max_ps(_mm256_min_ps(x, max_val), min_val);
    
    // Polynomial approximation
    let x2 = _mm256_mul_ps(x_clamped, x_clamped);
    
    // Coefficients for tanh approximation
    let c1 = _mm256_set1_ps(0.03138777);
    let c2 = _mm256_set1_ps(0.276281267);
    let c3 = _mm256_set1_ps(1.0);
    
    // Compute polynomial
    let p = _mm256_fmadd_ps(x2, c1, c2);
    let q = _mm256_fmadd_ps(x2, p, c3);
    
    // tanh(x) ≈ x / (1 + x²/3 + ...)
    _mm256_div_ps(x_clamped, q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_add_f32_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        
        let a = vec![1.0f32; 100];
        let b = vec![2.0f32; 100];
        let mut out = vec![0.0f32; 100];
        
        unsafe {
            add_f32_avx2(&a, &b, &mut out);
        }
        
        for val in &out {
            assert_relative_eq!(*val, 3.0f32, epsilon = 1e-6);
        }
    }
}