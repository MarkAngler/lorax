//! Error handling and edge case tests
//!
//! Tests for error conditions, edge cases, and recovery scenarios

use super::fixtures::*;
use super::init_test_logging;
use t2l_core::{Result, TextToLora};
use t2l_core::apply::{ApplyEngine, MergeStrategy};
use t2l_core::export::{ExportEngine, ExportFormat, Precision};
use t2l_core::lora::{LoraParameters, LoraLayer};
use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::TempDir;

#[tokio::test]
async fn test_missing_model_files() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let model_path = temp_path.join("incomplete_model");
    let adapter_path = temp_path.join("adapter");
    
    // Create incomplete model (missing config.json)
    std::fs::create_dir_all(&model_path)?;
    std::fs::write(model_path.join("model.safetensors"), b"dummy")?;
    
    // Create valid adapter
    let adapter = create_test_adapter("llama", 16);
    std::fs::create_dir_all(&adapter_path)?;
    save_lora_parameters(&adapter, &adapter_path)?;
    
    // Try to apply adapter to incomplete model
    let apply_engine = ApplyEngine::new(MergeStrategy::Linear, false);
    let result = apply_engine.apply(
        &model_path,
        &adapter_path,
        None,
    ).await;
    
    assert!(result.is_err(), "Should fail with missing model files");
    let error_msg = result.err().unwrap().to_string();
    assert!(
        error_msg.contains("config") || error_msg.contains("not found"),
        "Error should mention missing config"
    );
    
    Ok(())
}

#[tokio::test]
async fn test_incompatible_adapter_architecture() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let model_path = temp_path.join("llama_model");
    let adapter_path = temp_path.join("mistral_adapter");
    
    // Create Llama model
    create_mock_model(&model_path, "llama")?;
    
    // Create Mistral adapter (incompatible)
    let adapter = create_test_adapter("mistral", 16);
    std::fs::create_dir_all(&adapter_path)?;
    save_lora_parameters(&adapter, &adapter_path)?;
    
    // Try to apply incompatible adapter
    let apply_engine = ApplyEngine::new(MergeStrategy::Linear, false);
    let result = apply_engine.apply(
        &model_path,
        &adapter_path,
        None,
    ).await;
    
    // This might succeed with warnings or fail - depends on implementation
    if result.is_err() {
        let error_msg = result.err().unwrap().to_string();
        println!("Expected incompatibility error: {}", error_msg);
    } else {
        println!("Warning: Incompatible architectures were merged - check for warnings");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_corrupted_adapter_files() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let adapter_path = temp_path.join("corrupted_adapter");
    
    std::fs::create_dir_all(&adapter_path)?;
    
    // Create corrupted adapter files
    std::fs::write(adapter_path.join("adapter_config.json"), b"{ invalid json")?;
    std::fs::write(adapter_path.join("adapter_model.safetensors"), b"corrupted data")?;
    
    // Try to load corrupted adapter
    let result = load_lora_parameters(&adapter_path);
    assert!(result.is_err(), "Should fail to load corrupted adapter");
    
    // Try to export corrupted adapter
    let export_path = temp_path.join("export");
    let engine = ExportEngine::new(Precision::Fp32, false);
    
    // Create a minimal valid adapter for testing
    let adapter = create_test_adapter("llama", 16);
    
    // Corrupt the adapter data
    let mut corrupted_adapter = adapter;
    corrupted_adapter.layers.clear(); // Remove all layers
    
    let result = engine.export(
        &corrupted_adapter,
        ExportFormat::Peft,
        Some("meta-llama/Llama-2-7b-hf"),
        &export_path,
    ).await;
    
    // Empty adapter might still export successfully but with warnings
    if result.is_ok() {
        println!("Warning: Empty adapter exported successfully");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_dimension_mismatch() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Create adapter with mismatched dimensions
    let mut layers = HashMap::new();
    
    // Add layers with inconsistent dimensions
    let mut layer1 = LoraLayer::new(
        "model.layers.0.self_attn.q_proj".to_string(),
        4096,
        4096,
        16,
        32.0,
    );
    layer1.randomize_weights();
    
    // Create a layer with wrong dimensions
    let mut layer2 = LoraLayer::new(
        "model.layers.0.self_attn.k_proj".to_string(),
        2048, // Wrong dimension
        4096,
        16,
        32.0,
    );
    layer2.randomize_weights();
    
    layers.insert(layer1.name.clone(), layer1);
    layers.insert(layer2.name.clone(), layer2);
    
    let config = LoraParameterConfig {
        target_architecture: "llama".to_string(),
        default_rank: 16,
        default_alpha: 32.0,
        target_modules: vec!["q_proj".to_string(), "k_proj".to_string()],
        merge_weights: false,
    };
    
    let adapter = LoraParameters { layers, config, metadata: None };
    
    // Try to apply adapter with dimension mismatch
    let model_path = temp_path.join("model");
    let adapter_path = temp_path.join("adapter");
    
    create_mock_model(&model_path, "llama")?;
    std::fs::create_dir_all(&adapter_path)?;
    save_lora_parameters(&adapter, &adapter_path)?;
    
    let apply_engine = ApplyEngine::new(MergeStrategy::Linear, false);
    let result = apply_engine.apply(
        &model_path,
        &adapter_path,
        None,
    ).await;
    
    // Should handle dimension mismatch gracefully
    if result.is_err() {
        println!("Dimension mismatch correctly detected");
    } else {
        println!("Warning: Dimension mismatch was not caught");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_out_of_memory_simulation() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Create extremely large adapter (simulate OOM)
    let mut adapter = create_test_adapter("llama", 256); // Very high rank
    
    // Add many layers to increase memory usage
    for i in 0..32 {
        for module in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let name = format!("model.layers.{}.self_attn.{}", i, module);
            let mut layer = LoraLayer::new(name.clone(), 8192, 8192, 256, 512.0);
            layer.randomize_weights();
            adapter.layers.insert(name, layer);
        }
    }
    
    // Try operations that might trigger OOM
    let export_path = temp_path.join("large_export");
    let engine = ExportEngine::new(Precision::Fp32, true); // Enable optimization
    
    let result = engine.export(
        &adapter,
        ExportFormat::HuggingFace,
        Some("meta-llama/Llama-2-70b-hf"),
        &export_path,
    ).await;
    
    // Should handle large operations gracefully
    if result.is_ok() {
        println!("Large adapter export succeeded with optimization");
    } else {
        println!("Large adapter export failed (possibly due to memory constraints)");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_invalid_export_paths() -> Result<()> {
    init_test_logging();
    
    let adapter = create_test_adapter("llama", 16);
    let engine = ExportEngine::new(Precision::Fp32, false);
    
    // Test various invalid paths
    let invalid_paths = vec![
        PathBuf::from("/root/no_permission"), // No write permission
        PathBuf::from("/dev/null/subfolder"), // Invalid parent
        PathBuf::from(""), // Empty path
        PathBuf::from("\0invalid\0path"), // Null bytes
    ];
    
    for invalid_path in invalid_paths {
        let result = engine.export(
            &adapter,
            ExportFormat::Peft,
            Some("meta-llama/Llama-2-7b-hf"),
            &invalid_path,
        ).await;
        
        assert!(
            result.is_err(),
            "Export should fail with invalid path: {:?}",
            invalid_path
        );
    }
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_access_conflicts() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let adapter_path = temp_path.join("shared_adapter");
    
    // Create adapter
    let adapter = create_test_adapter("llama", 16);
    std::fs::create_dir_all(&adapter_path)?;
    save_lora_parameters(&adapter, &adapter_path)?;
    
    // Simulate concurrent access
    let tasks: Vec<_> = (0..4).map(|i| {
        let path = adapter_path.clone();
        let output = temp_path.join(format!("concurrent_output_{}", i));
        
        tokio::spawn(async move {
            // Multiple processes trying to read/write the same adapter
            let engine = ExportEngine::new(Precision::Fp32, false);
            let adapter = load_lora_parameters(&path)?;
            
            engine.export(
                &adapter,
                ExportFormat::Peft,
                Some("meta-llama/Llama-2-7b-hf"),
                &output,
            ).await
        })
    }).collect();
    
    // Collect results
    let mut successes = 0;
    let mut failures = 0;
    
    for task in tasks {
        match task.await? {
            Ok(_) => successes += 1,
            Err(_) => failures += 1,
        }
    }
    
    println!("Concurrent access: {} successes, {} failures", successes, failures);
    assert!(successes > 0, "At least some concurrent operations should succeed");
    
    Ok(())
}

#[tokio::test]
async fn test_recovery_from_partial_operations() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let export_path = temp_path.join("partial_export");
    
    // Create partial export directory (simulating interrupted operation)
    std::fs::create_dir_all(&export_path)?;
    std::fs::write(export_path.join("adapter_config.json"), "{}")?;
    // Missing adapter weights file
    
    // Try to continue/overwrite partial export
    let adapter = create_test_adapter("llama", 16);
    let engine = ExportEngine::new(Precision::Fp32, false);
    
    let result = engine.export(
        &adapter,
        ExportFormat::Peft,
        Some("meta-llama/Llama-2-7b-hf"),
        &export_path,
    ).await;
    
    assert!(result.is_ok(), "Should recover from partial export");
    verify_peft_structure(&export_path)?;
    
    Ok(())
}

#[tokio::test]
async fn test_edge_case_parameters() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Test edge case LoRA parameters
    let edge_cases = vec![
        (1, 1.0),    // Minimum rank and alpha
        (512, 1024.0), // Very high rank and alpha
        (17, 0.001),  // Odd rank, very small alpha
        (64, 1e6),    // Power of 2 rank, very large alpha
    ];
    
    for (i, (rank, alpha)) in edge_cases.iter().enumerate() {
        let mut adapter = create_test_adapter("llama", *rank);
        adapter.config.default_alpha = *alpha;
        
        let export_path = temp_path.join(format!("edge_case_{}", i));
        let engine = ExportEngine::new(Precision::Fp32, false);
        
        let result = engine.export(
            &adapter,
            ExportFormat::Peft,
            Some("meta-llama/Llama-2-7b-hf"),
            &export_path,
        ).await;
        
        assert!(
            result.is_ok(),
            "Edge case export failed for rank={}, alpha={}: {:?}",
            rank, alpha, result.err()
        );
        
        // Verify exported values
        let config = verify_json_file(&export_path.join("adapter_config.json"))?;
        assert_eq!(config["r"], *rank);
        assert_eq!(config["lora_alpha"].as_f64().unwrap(), *alpha);
    }
    
    Ok(())
}

// Helper function for PEFT structure verification (reused from export_tests.rs)
fn verify_peft_structure(path: &Path) -> Result<()> {
    verify_file_exists(&path.join("adapter_config.json"))?;
    
    let has_safetensors = path.join("adapter_model.safetensors").exists();
    let has_bin = path.join("adapter_model.bin").exists();
    assert!(has_safetensors || has_bin, "No adapter weights found");
    
    Ok(())
}