//! T2L Benchmarking Suite
//!
//! This crate provides comprehensive benchmarking capabilities for T2L operations
//! including LoRA generation, SIMD operations, memory management, and more.

#![warn(missing_docs)]

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Batch sizes to benchmark
    pub batch_sizes: Vec<usize>,
    /// Matrix dimensions to test
    pub matrix_dims: Vec<(usize, usize)>,
    /// LoRA ranks to benchmark
    pub lora_ranks: Vec<usize>,
    /// Number of warmup iterations
    pub warmup_iterations: u32,
    /// Measurement time per benchmark
    pub measurement_time: Duration,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            batch_sizes: vec![1, 8, 16, 32, 64],
            matrix_dims: vec![
                (768, 768),
                (1024, 1024),
                (2048, 2048),
                (4096, 4096),
            ],
            lora_ranks: vec![4, 8, 16, 32, 64],
            warmup_iterations: 3,
            measurement_time: Duration::from_secs(5),
        }
    }
}

/// SIMD operation benchmarks
pub mod simd_benchmarks {
    use super::*;
    use t2l_simd::ops::*;
    
    /// Benchmark SIMD vector addition
    pub fn bench_vector_add(c: &mut Criterion, config: &BenchConfig) {
        let mut group = c.benchmark_group("simd_vector_add");
        
        for &size in &[1024, 4096, 16384, 65536] {
            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(
                BenchmarkId::new("size", size),
                &size,
                |b, &size| {
                    let a = vec![1.0f32; size];
                    let b = vec![2.0f32; size];
                    let mut c = vec![0.0f32; size];
                    
                    b.iter(|| {
                        // TODO: Call actual SIMD vector add
                        for i in 0..size {
                            c[i] = a[i] + b[i];
                        }
                    });
                },
            );
        }
        
        group.finish();
    }
    
    /// Benchmark SIMD matrix multiplication
    pub fn bench_matrix_mul(c: &mut Criterion, config: &BenchConfig) {
        let mut group = c.benchmark_group("simd_matrix_mul");
        
        for &(m, n) in &config.matrix_dims {
            let k = n; // Square matrices
            group.throughput(Throughput::Elements((m * n * k) as u64));
            group.bench_with_input(
                BenchmarkId::new("dims", format!("{}x{}x{}", m, n, k)),
                &(m, n, k),
                |b, &(m, n, k)| {
                    let a = vec![1.0f32; m * k];
                    let b_mat = vec![2.0f32; k * n];
                    let mut c = vec![0.0f32; m * n];
                    
                    b.iter(|| {
                        // TODO: Call actual SIMD matrix multiplication
                        for i in 0..m {
                            for j in 0..n {
                                let mut sum = 0.0;
                                for l in 0..k {
                                    sum += a[i * k + l] * b_mat[l * n + j];
                                }
                                c[i * n + j] = sum;
                            }
                        }
                    });
                },
            );
        }
        
        group.finish();
    }
}

/// LoRA generation benchmarks
pub mod lora_benchmarks {
    use super::*;
    
    /// Benchmark LoRA parameter generation
    pub fn bench_lora_generation(c: &mut Criterion, config: &BenchConfig) {
        let mut group = c.benchmark_group("lora_generation");
        
        for &rank in &config.lora_ranks {
            for &batch_size in &config.batch_sizes {
                group.bench_with_input(
                    BenchmarkId::new("rank_batch", format!("r{}_b{}", rank, batch_size)),
                    &(rank, batch_size),
                    |b, &(rank, batch_size)| {
                        // TODO: Set up LoRA generation benchmark
                        let task_embeddings = vec![vec![1.0f32; 768]; batch_size];
                        
                        b.iter(|| {
                            // TODO: Call actual LoRA generation
                            for embedding in &task_embeddings {
                                let _lora_a = vec![vec![0.1f32; rank]; 768];
                                let _lora_b = vec![vec![0.1f32; 768]; rank];
                            }
                        });
                    },
                );
            }
        }
        
        group.finish();
    }
    
    /// Benchmark LoRA application (forward pass)
    pub fn bench_lora_forward(c: &mut Criterion, config: &BenchConfig) {
        let mut group = c.benchmark_group("lora_forward");
        
        for &rank in &config.lora_ranks {
            for &batch_size in &config.batch_sizes {
                group.throughput(Throughput::Elements((batch_size * 768) as u64));
                group.bench_with_input(
                    BenchmarkId::new("rank_batch", format!("r{}_b{}", rank, batch_size)),
                    &(rank, batch_size),
                    |b, &(rank, batch_size)| {
                        let input = vec![vec![1.0f32; 768]; batch_size];
                        let lora_a = vec![vec![0.1f32; rank]; 768];
                        let lora_b = vec![vec![0.1f32; 768]; rank];
                        let mut output = vec![vec![0.0f32; 768]; batch_size];
                        
                        b.iter(|| {
                            // TODO: Implement actual LoRA forward pass
                            for (i, inp) in input.iter().enumerate() {
                                for j in 0..768 {
                                    let mut sum = 0.0;
                                    for k in 0..rank {
                                        let mut temp = 0.0;
                                        for l in 0..768 {
                                            temp += inp[l] * lora_a[l][k];
                                        }
                                        sum += temp * lora_b[k][j];
                                    }
                                    output[i][j] = sum;
                                }
                            }
                        });
                    },
                );
            }
        }
        
        group.finish();
    }
}

/// Memory management benchmarks
pub mod memory_benchmarks {
    use super::*;
    
    /// Benchmark memory allocation patterns
    pub fn bench_memory_allocation(c: &mut Criterion, config: &BenchConfig) {
        let mut group = c.benchmark_group("memory_allocation");
        
        for &size in &[1024, 4096, 16384, 65536, 262144] {
            group.throughput(Throughput::Bytes(size as u64 * 4)); // f32 = 4 bytes
            group.bench_with_input(
                BenchmarkId::new("size", size),
                &size,
                |b, &size| {
                    b.iter_batched(
                        || size,
                        |size| {
                            let _data: Vec<f32> = vec![0.0; size];
                            // Force allocation by writing to memory
                            std::hint::black_box(_data);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
        
        group.finish();
    }
    
    /// Benchmark memory copy operations
    pub fn bench_memory_copy(c: &mut Criterion, config: &BenchConfig) {
        let mut group = c.benchmark_group("memory_copy");
        
        for &size in &[1024, 4096, 16384, 65536, 262144] {
            group.throughput(Throughput::Bytes(size as u64 * 4)); // f32 = 4 bytes
            group.bench_with_input(
                BenchmarkId::new("size", size),
                &size,
                |b, &size| {
                    let src = vec![1.0f32; size];
                    let mut dst = vec![0.0f32; size];
                    
                    b.iter(|| {
                        dst.copy_from_slice(&src);
                        std::hint::black_box(&mut dst);
                    });
                },
            );
        }
        
        group.finish();
    }
}

/// Quantization benchmarks
pub mod quantization_benchmarks {
    use super::*;
    
    /// Benchmark float to int8 quantization
    pub fn bench_f32_to_i8_quantization(c: &mut Criterion, config: &BenchConfig) {
        let mut group = c.benchmark_group("f32_to_i8_quantization");
        
        for &size in &[1024, 4096, 16384, 65536] {
            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(
                BenchmarkId::new("size", size),
                &size,
                |b, &size| {
                    let input = vec![1.0f32; size];
                    let mut output = vec![0i8; size];
                    let scale = 127.0f32;
                    
                    b.iter(|| {
                        for i in 0..size {
                            output[i] = (input[i] * scale).clamp(-128.0, 127.0) as i8;
                        }
                        std::hint::black_box(&mut output);
                    });
                },
            );
        }
        
        group.finish();
    }
    
    /// Benchmark int8 to float dequantization
    pub fn bench_i8_to_f32_dequantization(c: &mut Criterion, config: &BenchConfig) {
        let mut group = c.benchmark_group("i8_to_f32_dequantization");
        
        for &size in &[1024, 4096, 16384, 65536] {
            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(
                BenchmarkId::new("size", size),
                &size,
                |b, &size| {
                    let input = vec![64i8; size];
                    let mut output = vec![0.0f32; size];
                    let scale = 1.0 / 127.0f32;
                    
                    b.iter(|| {
                        for i in 0..size {
                            output[i] = input[i] as f32 * scale;
                        }
                        std::hint::black_box(&mut output);
                    });
                },
            );
        }
        
        group.finish();
    }
}

/// Run all benchmarks with the given configuration
pub fn run_all_benchmarks(c: &mut Criterion, config: &BenchConfig) {
    // Configure criterion
    c.warm_up_time(Duration::from_secs(1))
        .measurement_time(config.measurement_time)
        .sample_size(50);
    
    // Run SIMD benchmarks
    simd_benchmarks::bench_vector_add(c, config);
    simd_benchmarks::bench_matrix_mul(c, config);
    
    // Run LoRA benchmarks
    lora_benchmarks::bench_lora_generation(c, config);
    lora_benchmarks::bench_lora_forward(c, config);
    
    // Run memory benchmarks
    memory_benchmarks::bench_memory_allocation(c, config);
    memory_benchmarks::bench_memory_copy(c, config);
    
    // Run quantization benchmarks
    quantization_benchmarks::bench_f32_to_i8_quantization(c, config);
    quantization_benchmarks::bench_i8_to_f32_dequantization(c, config);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bench_config_default() {
        let config = BenchConfig::default();
        assert!(!config.batch_sizes.is_empty());
        assert!(!config.matrix_dims.is_empty());
        assert!(!config.lora_ranks.is_empty());
    }
}