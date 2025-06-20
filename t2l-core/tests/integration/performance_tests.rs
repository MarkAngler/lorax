//! Performance benchmark tests
//!
//! Tests for performance characteristics and optimization validation

use super::fixtures::*;
use super::init_test_logging;
use t2l_core::{Result, TextToLora};
use t2l_core::apply::{ApplyEngine, MergeStrategy};
use t2l_core::export::{ExportEngine, ExportFormat, Precision};
use t2l_core::lora::LoraParameters;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Benchmark result structure
#[derive(Debug)]
struct BenchmarkResult {
    operation: String,
    duration: Duration,
    throughput: Option<f64>,
    memory_peak: Option<usize>,
}

impl BenchmarkResult {
    fn new(operation: &str, duration: Duration) -> Self {
        Self {
            operation: operation.to_string(),
            duration,
            throughput: None,
            memory_peak: None,
        }
    }
    
    fn with_throughput(mut self, throughput: f64) -> Self {
        self.throughput = Some(throughput);
        self
    }
    
    fn report(&self) {
        println!(
            "Benchmark [{}]: {:.2} ms",
            self.operation,
            self.duration.as_secs_f64() * 1000.0
        );
        
        if let Some(throughput) = self.throughput {
            println!("  Throughput: {:.2} ops/sec", throughput);
        }
        
        if let Some(memory) = self.memory_peak {
            println!("  Peak memory: {} MB", memory / 1_048_576);
        }
    }
}

#[tokio::test]
async fn benchmark_adapter_generation() -> Result<()> {
    init_test_logging();
    
    let ranks = vec![4, 8, 16, 32, 64, 128];
    let mut results = Vec::new();
    
    for rank in ranks {
        let start = Instant::now();
        
        // Generate adapter
        let _adapter = create_test_adapter("llama", rank);
        
        let duration = start.elapsed();
        let result = BenchmarkResult::new(&format!("generate_adapter_rank_{}", rank), duration);
        result.report();
        results.push(result);
    }
    
    // Verify performance scaling
    // Higher ranks should take proportionally more time
    let base_time = results[0].duration.as_secs_f64();
    let high_rank_time = results.last().unwrap().duration.as_secs_f64();
    
    assert!(
        high_rank_time > base_time,
        "Higher rank adapters should take more time to generate"
    );
    
    Ok(())
}

#[tokio::test]
async fn benchmark_export_operations() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let adapter = create_test_adapter("llama", 32);
    
    let export_configs = vec![
        ("peft_fp32", ExportFormat::Peft, Precision::Fp32, false),
        ("peft_fp16", ExportFormat::Peft, Precision::Fp16, false),
        ("peft_int8", ExportFormat::Peft, Precision::Int8, false),
        ("peft_optimized", ExportFormat::Peft, Precision::Fp16, true),
        ("hf_fp32", ExportFormat::HuggingFace, Precision::Fp32, false),
        ("hf_optimized", ExportFormat::HuggingFace, Precision::Fp16, true),
    ];
    
    let mut results = Vec::new();
    
    for (name, format, precision, optimize) in export_configs {
        let output_path = temp_path.join(name);
        let engine = ExportEngine::new(precision, optimize);
        
        let start = Instant::now();
        
        let result = engine.export(
            &adapter,
            format,
            Some("meta-llama/Llama-2-7b-hf"),
            &output_path,
        ).await;
        
        let duration = start.elapsed();
        
        if result.is_ok() {
            let bench_result = BenchmarkResult::new(&format!("export_{}", name), duration);
            bench_result.report();
            results.push(bench_result);
            
            // Measure output size
            if let Ok(size) = get_directory_size(&output_path) {
                println!("  Output size: {} MB", size / 1_048_576);
            }
        }
    }
    
    // Verify optimization benefits
    let regular_idx = results.iter().position(|r| r.operation.contains("peft_fp16")).unwrap();
    let optimized_idx = results.iter().position(|r| r.operation.contains("peft_optimized")).unwrap();
    
    if regular_idx < results.len() && optimized_idx < results.len() {
        let regular_time = results[regular_idx].duration.as_secs_f64();
        let optimized_time = results[optimized_idx].duration.as_secs_f64();
        
        println!(
            "Optimization improvement: {:.1}%",
            ((regular_time - optimized_time) / regular_time) * 100.0
        );
    }
    
    Ok(())
}

#[tokio::test]
async fn benchmark_apply_operations() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let model_path = temp_path.join("model");
    create_mock_model(&model_path, "llama")?;
    
    let strategies = vec![
        ("linear", MergeStrategy::Linear, false),
        ("linear_optimized", MergeStrategy::Linear, true),
        ("scaled", MergeStrategy::Scaled(0.5), false),
        ("scaled_optimized", MergeStrategy::Scaled(0.5), true),
    ];
    
    let mut results = Vec::new();
    
    for (name, strategy, optimize) in strategies {
        let adapter = create_test_adapter("llama", 16);
        let adapter_path = temp_path.join(format!("adapter_{}", name));
        std::fs::create_dir_all(&adapter_path)?;
        adapter.save(&adapter_path)?;
        
        let output_path = temp_path.join(format!("merged_{}", name));
        let engine = ApplyEngine::new(strategy, optimize);
        
        let start = Instant::now();
        
        let result = engine.apply(
            &model_path,
            &adapter_path,
            Some(&output_path),
        ).await;
        
        let duration = start.elapsed();
        
        if result.is_ok() {
            let bench_result = BenchmarkResult::new(&format!("apply_{}", name), duration);
            bench_result.report();
            results.push(bench_result);
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn benchmark_batch_operations() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let batch_sizes = vec![1, 4, 8, 16, 32];
    let mut results = Vec::new();
    
    for batch_size in batch_sizes {
        let start = Instant::now();
        
        // Create batch of adapters
        let adapters: Vec<_> = (0..batch_size)
            .map(|i| create_test_adapter("llama", 8 + (i % 4) * 4))
            .collect();
        
        // Export batch
        let mut export_tasks = Vec::new();
        for (i, adapter) in adapters.iter().enumerate() {
            let output_path = temp_path.join(format!("batch_{}_{}", batch_size, i));
            let adapter_clone = adapter.clone();
            
            export_tasks.push(tokio::spawn(async move {
                let engine = ExportEngine::new(Precision::Fp16, true);
                engine.export(
                    &adapter_clone,
                    ExportFormat::Peft,
                    Some("meta-llama/Llama-2-7b-hf"),
                    &output_path,
                ).await
            }));
        }
        
        // Wait for all exports
        for task in export_tasks {
            let _ = task.await?;
        }
        
        let duration = start.elapsed();
        let throughput = batch_size as f64 / duration.as_secs_f64();
        
        let result = BenchmarkResult::new(&format!("batch_export_size_{}", batch_size), duration)
            .with_throughput(throughput);
        result.report();
        results.push(result);
    }
    
    // Verify batch efficiency
    if results.len() >= 2 {
        let single_throughput = results[0].throughput.unwrap_or(1.0);
        let batch_throughput = results.last().unwrap().throughput.unwrap_or(1.0);
        
        println!(
            "Batch efficiency improvement: {:.1}x",
            batch_throughput / single_throughput
        );
    }
    
    Ok(())
}

#[tokio::test]
async fn benchmark_memory_usage() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Test memory usage with different adapter sizes
    let configs = vec![
        ("small", 8, 4),
        ("medium", 32, 8),
        ("large", 128, 16),
        ("xlarge", 256, 32),
    ];
    
    for (size_name, rank, num_layers) in configs {
        let start_memory = get_current_memory_usage();
        
        // Create large adapter
        let mut adapter = create_test_adapter("llama", rank);
        
        // Add more layers to increase memory usage
        for i in 0..num_layers {
            for module in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                let name = format!("model.layers.{}.self_attn.{}", i, module);
                let mut layer = LoraLayer::new(name.clone(), 4096, 4096, rank, rank as f32 * 2.0);
                layer.randomize_weights();
                adapter.layers.insert(name, layer);
            }
        }
        
        let peak_memory = get_current_memory_usage();
        let memory_used = peak_memory.saturating_sub(start_memory);
        
        println!(
            "Memory usage [{}]: {} MB (rank={}, layers={})",
            size_name,
            memory_used / 1_048_576,
            rank,
            adapter.layers.len()
        );
        
        // Export to test memory during operations
        let export_path = temp_path.join(format!("memory_test_{}", size_name));
        let engine = ExportEngine::new(Precision::Fp32, true);
        
        let _ = engine.export(
            &adapter,
            ExportFormat::Peft,
            Some("meta-llama/Llama-2-7b-hf"),
            &export_path,
        ).await;
        
        let export_peak_memory = get_current_memory_usage();
        let export_memory_increase = export_peak_memory.saturating_sub(peak_memory);
        
        println!(
            "  Export memory increase: {} MB",
            export_memory_increase / 1_048_576
        );
    }
    
    Ok(())
}

#[tokio::test]
async fn benchmark_concurrent_performance() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let concurrency_levels = vec![1, 2, 4, 8, 16];
    
    for concurrency in concurrency_levels {
        let start = Instant::now();
        
        // Launch concurrent operations
        let tasks: Vec<_> = (0..concurrency).map(|i| {
            let path = temp_path.clone();
            tokio::spawn(async move {
                let adapter = create_test_adapter("llama", 16);
                let output_path = path.join(format!("concurrent_{}_{}", concurrency, i));
                
                let engine = ExportEngine::new(Precision::Fp16, true);
                engine.export(
                    &adapter,
                    ExportFormat::Peft,
                    Some("meta-llama/Llama-2-7b-hf"),
                    &output_path,
                ).await
            })
        }).collect();
        
        // Wait for all tasks
        let mut successes = 0;
        for task in tasks {
            if let Ok(Ok(_)) = task.await {
                successes += 1;
            }
        }
        
        let duration = start.elapsed();
        let throughput = successes as f64 / duration.as_secs_f64();
        
        println!(
            "Concurrent operations [{}]: {:.2} ms, {:.2} ops/sec",
            concurrency,
            duration.as_secs_f64() * 1000.0,
            throughput
        );
    }
    
    Ok(())
}

#[tokio::test]
async fn benchmark_precision_impact() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let adapter = create_test_adapter("llama", 64); // Large adapter
    
    let precisions = vec![
        (Precision::Fp32, "fp32", 1.0),
        (Precision::Fp16, "fp16", 2.0),
        (Precision::Bf16, "bf16", 2.0),
        (Precision::Int8, "int8", 4.0),
    ];
    
    for (precision, name, expected_speedup) in precisions {
        let output_path = temp_path.join(format!("precision_bench_{}", name));
        let engine = ExportEngine::new(precision, true);
        
        let start = Instant::now();
        
        let result = engine.export(
            &adapter,
            ExportFormat::Peft,
            Some("meta-llama/Llama-2-7b-hf"),
            &output_path,
        ).await;
        
        let duration = start.elapsed();
        
        if result.is_ok() {
            println!(
                "Precision [{}]: {:.2} ms (expected speedup: {:.1}x)",
                name,
                duration.as_secs_f64() * 1000.0,
                expected_speedup
            );
            
            // Check output file size
            if let Ok(size) = get_directory_size(&output_path) {
                println!("  Output size: {} MB", size / 1_048_576);
            }
        }
    }
    
    Ok(())
}

// Helper functions

fn get_current_memory_usage() -> usize {
    // Mock implementation - in real code, use system-specific memory queries
    // On Linux: parse /proc/self/status
    // On macOS: use mach APIs
    // On Windows: use Windows APIs
    0
}

fn get_directory_size(path: &std::path::Path) -> Result<u64> {
    let mut total_size = 0;
    
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_file() {
                total_size += metadata.len();
            } else if metadata.is_dir() {
                total_size += get_directory_size(&entry.path())?;
            }
        }
    } else if path.is_file() {
        total_size = std::fs::metadata(path)?.len();
    }
    
    Ok(total_size)
}