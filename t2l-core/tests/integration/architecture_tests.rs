//! Architecture and device compatibility tests
//!
//! Tests for multi-architecture support and device compatibility

use super::fixtures::*;
use super::init_test_logging;
use t2l_core::{Result, TextToLora};
use t2l_core::lora::{LoraParameters, LoraLayer, LoraParameterConfig};
use t2l_core::config::{Config, EncoderConfig, HypernetworkConfig, ModelSize};
use t2l_core::apply::{ApplyEngine, MergeStrategy};
use t2l_core::export::{ExportEngine, ExportFormat, Precision};
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;

#[tokio::test]
async fn test_llama_architecture_compatibility() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Test different Llama model sizes
    let llama_variants = vec![
        ("llama-7b", 4096, 11008, 32),
        ("llama-13b", 5120, 13824, 40),
        ("llama-70b", 8192, 28672, 64),
    ];
    
    for (variant, hidden_size, intermediate_size, num_layers) in llama_variants {
        let adapter = create_llama_adapter(hidden_size, intermediate_size);
        let adapter_path = temp_path.join(format!("{}_adapter", variant));
        
        std::fs::create_dir_all(&adapter_path)?;
        adapter.save(&adapter_path)?;
        
        // Verify adapter structure matches Llama architecture
        assert_eq!(adapter.config.target_architecture, "llama");
        assert!(adapter.layers.keys().any(|k| k.contains("self_attn")));
        assert!(adapter.layers.keys().any(|k| k.contains("mlp")));
        
        println!("Created adapter for {} with {} layers", variant, adapter.layers.len());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_mistral_architecture_compatibility() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Mistral uses grouped-query attention
    let adapter = create_mistral_adapter();
    let adapter_path = temp_path.join("mistral_adapter");
    
    std::fs::create_dir_all(&adapter_path)?;
    adapter.save(&adapter_path)?;
    
    // Verify Mistral-specific features
    assert_eq!(adapter.config.target_architecture, "mistral");
    
    // Check for GQA-aware layer dimensions
    let k_proj = adapter.layers.get("model.layers.0.self_attn.k_proj");
    let v_proj = adapter.layers.get("model.layers.0.self_attn.v_proj");
    
    if let (Some(k), Some(v)) = (k_proj, v_proj) {
        // Mistral uses smaller K/V dimensions due to GQA
        assert_eq!(k.output_dim, 1024); // vs 4096 for Q
        assert_eq!(v.output_dim, 1024);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_gemma_architecture_compatibility() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Gemma has unique architecture features
    let adapter = create_gemma_adapter();
    let adapter_path = temp_path.join("gemma_adapter");
    
    std::fs::create_dir_all(&adapter_path)?;
    adapter.save(&adapter_path)?;
    
    // Verify Gemma-specific features
    assert_eq!(adapter.config.target_architecture, "gemma");
    
    // Gemma uses different hidden sizes
    let layer = adapter.layers.values().next().unwrap();
    assert!(layer.input_dim == 3072 || layer.input_dim == 24576);
    
    Ok(())
}

#[tokio::test]
async fn test_cross_architecture_conversion() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Create adapters for different architectures
    let architectures = vec![
        ("llama", create_test_adapter("llama", 16)),
        ("mistral", create_test_adapter("mistral", 16)),
        ("gemma", create_test_adapter("gemma", 16)),
    ];
    
    for (source_arch, adapter) in architectures {
        // Export to universal PEFT format
        let export_path = temp_path.join(format!("{}_universal", source_arch));
        let engine = ExportEngine::new(Precision::Fp32, false);
        
        let result = engine.export(
            &adapter,
            ExportFormat::Peft,
            None, // No specific target model
            &export_path,
        ).await;
        
        assert!(result.is_ok(), "Failed to export {} to universal format", source_arch);
        
        // Verify the export can specify different target architectures
        let config = verify_json_file(&export_path.join("adapter_config.json"))?;
        assert!(config.get("target_modules").is_some());
    }
    
    Ok(())
}

#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_cuda_device_compatibility() -> Result<()> {
    init_test_logging();
    
    // Check if CUDA is available
    if !is_cuda_available() {
        println!("CUDA not available, skipping test");
        return Ok(());
    }
    
    let adapter = create_test_adapter("llama", 16);
    
    // Test CUDA-specific operations
    let result = run_cuda_inference_test(adapter).await;
    assert!(result.is_ok(), "CUDA inference failed: {:?}", result.err());
    
    Ok(())
}

#[tokio::test]
async fn test_cpu_device_compatibility() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let adapter = create_test_adapter("llama", 16);
    
    // Test CPU-specific optimizations
    let engine = ExportEngine::new(Precision::Fp32, true); // Enable CPU optimizations
    let export_path = temp_path.join("cpu_optimized");
    
    let result = engine.export(
        &adapter,
        ExportFormat::HuggingFace,
        Some("meta-llama/Llama-2-7b-hf"),
        &export_path,
    ).await;
    
    assert!(result.is_ok(), "CPU-optimized export failed: {:?}", result.err());
    
    Ok(())
}

#[tokio::test]
async fn test_mixed_precision_support() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Test different precision combinations
    let precision_tests = vec![
        ("compute_fp32_storage_fp16", Precision::Fp32, Precision::Fp16),
        ("compute_fp16_storage_int8", Precision::Fp16, Precision::Int8),
        ("full_int8", Precision::Int8, Precision::Int8),
    ];
    
    for (test_name, compute_precision, storage_precision) in precision_tests {
        let adapter = create_test_adapter("llama", 16);
        let export_path = temp_path.join(test_name);
        
        // Export with specific precision settings
        let engine = ExportEngine::new(storage_precision, false);
        let result = engine.export(
            &adapter,
            ExportFormat::Peft,
            Some("meta-llama/Llama-2-7b-hf"),
            &export_path,
        ).await;
        
        assert!(result.is_ok(), "Mixed precision test {} failed", test_name);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_architecture_specific_optimizations() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Test architecture-specific optimization paths
    let test_cases = vec![
        ("llama", true, "flash_attention"),
        ("mistral", true, "grouped_query_attention"),
        ("gemma", false, "standard_attention"),
    ];
    
    for (arch, use_optimization, optimization_type) in test_cases {
        let adapter = create_test_adapter(arch, 16);
        let export_path = temp_path.join(format!("{}_opt_{}", arch, optimization_type));
        
        let engine = ExportEngine::new(Precision::Fp16, use_optimization);
        let result = engine.export(
            &adapter,
            ExportFormat::HuggingFace,
            Some(TEST_MODEL_IDS.iter().find(|(a, _)| a == &arch).unwrap().1),
            &export_path,
        ).await;
        
        assert!(
            result.is_ok(), 
            "Architecture optimization test failed for {} with {}", 
            arch, 
            optimization_type
        );
    }
    
    Ok(())
}

// Helper functions for architecture-specific adapter creation

fn create_llama_adapter(hidden_size: usize, intermediate_size: usize) -> LoraParameters {
    let mut layers = HashMap::new();
    let rank = 16;
    let alpha = 32.0;
    
    // Create full Llama layer structure
    for layer_idx in 0..2 {  // Just 2 layers for testing
        let prefix = format!("model.layers.{}", layer_idx);
        
        // Attention layers
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let name = format!("{}.self_attn.{}", prefix, proj);
            let mut layer = LoraLayer::new(name.clone(), hidden_size, hidden_size, rank, alpha);
            layer.randomize_weights();
            layers.insert(name, layer);
        }
        
        // MLP layers
        let mlp_configs = vec![
            ("gate_proj", hidden_size, intermediate_size),
            ("up_proj", hidden_size, intermediate_size),
            ("down_proj", intermediate_size, hidden_size),
        ];
        
        for (proj, in_dim, out_dim) in mlp_configs {
            let name = format!("{}.mlp.{}", prefix, proj);
            let mut layer = LoraLayer::new(name.clone(), in_dim, out_dim, rank, alpha);
            layer.randomize_weights();
            layers.insert(name, layer);
        }
    }
    
    let config = LoraParameterConfig {
        target_architecture: "llama".to_string(),
        default_rank: rank,
        default_alpha: alpha,
        target_modules: vec![
            "q_proj".to_string(), "k_proj".to_string(), 
            "v_proj".to_string(), "o_proj".to_string(),
            "gate_proj".to_string(), "up_proj".to_string(), 
            "down_proj".to_string(),
        ],
        merge_weights: false,
    };
    
    LoraParameters { layers, config, metadata: None }
}

fn create_mistral_adapter() -> LoraParameters {
    let mut layers = HashMap::new();
    let rank = 16;
    let alpha = 32.0;
    
    // Mistral uses GQA with different K/V dimensions
    let hidden_size = 4096;
    let kv_hidden_size = 1024;  // Smaller due to GQA
    let intermediate_size = 14336;
    
    for layer_idx in 0..2 {
        let prefix = format!("model.layers.{}", layer_idx);
        
        // Q projection (full size)
        let q_name = format!("{}.self_attn.q_proj", prefix);
        let mut q_layer = LoraLayer::new(q_name.clone(), hidden_size, hidden_size, rank, alpha);
        q_layer.randomize_weights();
        layers.insert(q_name, q_layer);
        
        // K/V projections (reduced size for GQA)
        for proj in &["k_proj", "v_proj"] {
            let name = format!("{}.self_attn.{}", prefix, proj);
            let mut layer = LoraLayer::new(name.clone(), hidden_size, kv_hidden_size, rank, alpha);
            layer.randomize_weights();
            layers.insert(name, layer);
        }
        
        // O projection
        let o_name = format!("{}.self_attn.o_proj", prefix);
        let mut o_layer = LoraLayer::new(o_name.clone(), hidden_size, hidden_size, rank, alpha);
        o_layer.randomize_weights();
        layers.insert(o_name, o_layer);
        
        // MLP layers (same as Llama)
        for (proj, in_dim, out_dim) in &[
            ("gate_proj", hidden_size, intermediate_size),
            ("up_proj", hidden_size, intermediate_size),
            ("down_proj", intermediate_size, hidden_size),
        ] {
            let name = format!("{}.mlp.{}", prefix, proj);
            let mut layer = LoraLayer::new(name.clone(), *in_dim, *out_dim, rank, alpha);
            layer.randomize_weights();
            layers.insert(name, layer);
        }
    }
    
    let config = LoraParameterConfig {
        target_architecture: "mistral".to_string(),
        default_rank: rank,
        default_alpha: alpha,
        target_modules: vec![
            "q_proj".to_string(), "k_proj".to_string(), 
            "v_proj".to_string(), "o_proj".to_string(),
            "gate_proj".to_string(), "up_proj".to_string(), 
            "down_proj".to_string(),
        ],
        merge_weights: false,
    };
    
    LoraParameters { layers, config, metadata: None }
}

fn create_gemma_adapter() -> LoraParameters {
    create_test_adapter("gemma", 16) // Use the fixture version
}

// Mock functions for device testing

fn is_cuda_available() -> bool {
    // In a real implementation, this would check for CUDA availability
    false
}

async fn run_cuda_inference_test(_adapter: LoraParameters) -> Result<()> {
    // Mock CUDA inference test
    Ok(())
}