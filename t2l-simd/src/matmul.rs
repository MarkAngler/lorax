//! SIMD-accelerated matrix multiplication for T2L

use std::arch::x86_64::*;
use rayon::prelude::*;

/// Optimized matrix multiplication with AVX2
/// C = A * B where A is MxK, B is KxN, C is MxN
pub unsafe fn matmul_f32_avx2(
    a: &[f32],
    b: &[f32], 
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    
    // Use parallel processing for large matrices
    if m * n > 10000 {
        matmul_f32_avx2_parallel(a, b, c, m, k, n);
        return;
    }
    
    // Block size for cache optimization
    const BLOCK_SIZE: usize = 64;
    
    // Zero output matrix
    c.fill(0.0);
    
    // Blocked matrix multiplication
    for i_block in (0..m).step_by(BLOCK_SIZE) {
        for j_block in (0..n).step_by(BLOCK_SIZE) {
            for k_block in (0..k).step_by(BLOCK_SIZE) {
                // Process block
                let i_end = (i_block + BLOCK_SIZE).min(m);
                let j_end = (j_block + BLOCK_SIZE).min(n);
                let k_end = (k_block + BLOCK_SIZE).min(k);
                
                for i in i_block..i_end {
                    for j in (j_block..j_end).step_by(8) {
                        let j_simd_end = (j + 8).min(j_end);
                        
                        // Load accumulator
                        let mut acc = if j + 8 <= j_end {
                            _mm256_loadu_ps(c.as_ptr().add(i * n + j))
                        } else {
                            _mm256_setzero_ps()
                        };
                        
                        // Compute dot product
                        for k_idx in k_block..k_end {
                            let a_scalar = a[i * k + k_idx];
                            let a_vec = _mm256_set1_ps(a_scalar);
                            
                            if j + 8 <= j_end {
                                let b_vec = _mm256_loadu_ps(b.as_ptr().add(k_idx * n + j));
                                acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
                            } else {
                                // Handle remainder
                                for j_idx in j..j_simd_end {
                                    c[i * n + j_idx] += a_scalar * b[k_idx * n + j_idx];
                                }
                            }
                        }
                        
                        // Store result
                        if j + 8 <= j_end {
                            _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), acc);
                        }
                    }
                }
            }
        }
    }
}

/// Parallel matrix multiplication using Rayon
unsafe fn matmul_f32_avx2_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    const TILE_SIZE: usize = 32;
    
    // Parallelize over output rows
    c.par_chunks_mut(n * TILE_SIZE)
        .enumerate()
        .for_each(|(tile_i, c_tile)| {
            let i_start = tile_i * TILE_SIZE;
            let i_end = (i_start + TILE_SIZE).min(m);
            
            // Zero the output tile
            c_tile.fill(0.0);
            
            // Compute the tile
            for i in i_start..i_end {
                let i_local = i - i_start;
                
                for k_idx in 0..k {
                    let a_scalar = a[i * k + k_idx];
                    let a_vec = _mm256_set1_ps(a_scalar);
                    
                    // Process 8 columns at a time
                    for j in (0..n).step_by(8) {
                        if j + 8 <= n {
                            let b_vec = _mm256_loadu_ps(b.as_ptr().add(k_idx * n + j));
                            let c_ptr = c_tile.as_mut_ptr().add(i_local * n + j);
                            let c_vec = _mm256_loadu_ps(c_ptr);
                            let result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(c_ptr, result);
                        } else {
                            // Handle remainder
                            for j_idx in j..n {
                                c_tile[i_local * n + j_idx] += a_scalar * b[k_idx * n + j_idx];
                            }
                        }
                    }
                }
            }
        });
}

/// Optimized batch matrix multiplication
/// Performs multiple matrix multiplications in parallel
pub fn batch_matmul_f32(
    batch_a: &[&[f32]],
    batch_b: &[&[f32]],
    batch_c: &mut [&mut [f32]],
    m: usize,
    k: usize,
    n: usize,
) {
    assert_eq!(batch_a.len(), batch_b.len());
    assert_eq!(batch_a.len(), batch_c.len());
    
    // Process batches in parallel
    batch_a.par_iter()
        .zip(batch_b.par_iter())
        .zip(batch_c.par_iter_mut())
        .for_each(|((a, b), c)| {
            unsafe {
                matmul_f32_avx2(a, b, c, m, k, n);
            }
        });
}

/// Specialized matrix multiplication for LoRA updates
/// Computes C = A * B^T + C where A is MxR, B is NxR
pub unsafe fn lora_update_avx2(
    a: &[f32],  // MxR
    b: &[f32],  // NxR  
    c: &mut [f32],  // MxN
    m: usize,
    n: usize,
    r: usize,  // LoRA rank
    scale: f32,
) {
    assert_eq!(a.len(), m * r);
    assert_eq!(b.len(), n * r);
    assert_eq!(c.len(), m * n);
    
    let scale_vec = _mm256_set1_ps(scale);
    
    // Optimize for small rank (typical LoRA case)
    if r <= 32 {
        // Process multiple rows in parallel
        for i in 0..m {
            for j in (0..n).step_by(8) {
                let mut acc = _mm256_setzero_ps();
                
                // Compute dot product A[i,:] @ B[j:j+8,:]^T
                for k in 0..r {
                    let a_scalar = a[i * r + k];
                    let a_vec = _mm256_set1_ps(a_scalar);
                    
                    if j + 8 <= n {
                        // Load 8 elements from B
                        let mut b_vals = [0f32; 8];
                        for idx in 0..8 {
                            b_vals[idx] = b[(j + idx) * r + k];
                        }
                        let b_vec = _mm256_loadu_ps(b_vals.as_ptr());
                        acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
                    } else {
                        // Handle remainder
                        for j_idx in j..n {
                            c[i * n + j_idx] += scale * a_scalar * b[j_idx * r + k];
                        }
                    }
                }
                
                if j + 8 <= n {
                    // Scale and add to C
                    let scaled = _mm256_mul_ps(acc, scale_vec);
                    let c_vec = _mm256_loadu_ps(c.as_ptr().add(i * n + j));
                    let result = _mm256_add_ps(c_vec, scaled);
                    _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), result);
                }
            }
        }
    } else {
        // Fall back to regular matmul for larger ranks
        let mut temp = vec![0.0f32; m * n];
        matmul_transposed_b_avx2(a, b, &mut temp, m, r, n);
        
        // Add scaled result to C
        for i in 0..m * n {
            c[i] += scale * temp[i];
        }
    }
}

/// Matrix multiplication with transposed B: C = A * B^T
unsafe fn matmul_transposed_b_avx2(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    c.fill(0.0);
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            
            // Vectorized dot product
            let chunks = k / 8;
            let remainder = k % 8;
            
            if chunks > 0 {
                let mut acc = _mm256_setzero_ps();
                
                for chunk in 0..chunks {
                    let idx = chunk * 8;
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * k + idx));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(j * k + idx));
                    acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
                }
                
                // Horizontal sum
                let mut result = [0f32; 8];
                _mm256_storeu_ps(result.as_mut_ptr(), acc);
                sum = result.iter().sum();
            }
            
            // Handle remainder
            for idx in (chunks * 8)..k {
                sum += a[i * k + idx] * b[j * k + idx];
            }
            
            c[i * n + j] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_matmul_correctness() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        
        let m = 4;
        let k = 3;
        let n = 5;
        
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ];
        
        let b = vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
        ];
        
        let mut c = vec![0.0; m * n];
        
        unsafe {
            matmul_f32_avx2(&a, &b, &mut c, m, k, n);
        }
        
        // Verify first row
        assert_relative_eq!(c[0], 46.0, epsilon = 1e-6);
        assert_relative_eq!(c[1], 52.0, epsilon = 1e-6);
        assert_relative_eq!(c[2], 58.0, epsilon = 1e-6);
        assert_relative_eq!(c[3], 64.0, epsilon = 1e-6);
        assert_relative_eq!(c[4], 70.0, epsilon = 1e-6);
    }
}