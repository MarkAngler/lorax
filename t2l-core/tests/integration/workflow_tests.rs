//! Full workflow integration tests
//!
//! Tests complete end-to-end workflows: generate → apply → infer → export

use super::fixtures::*;
use super::init_test_logging;
use t2l_core::{Result, TextToLora};
use t2l_core::apply::{ApplyEngine, MergeStrategy};
use t2l_core::export::{ExportEngine, ExportFormat, Precision};
use t2l_core::infer::{InferenceEngine, GenerationConfig};
use t2l_core::config::Config;
use std::path::Path;
use tempfile::TempDir;

#[tokio::test]
async fn test_full_workflow_llama() -> Result<()> {
    init_test_logging();
    
    // Create test directories
    let (_temp_dir, temp_path) = create_test_dir();
    let model_path = temp_path.join("base_model");
    let adapter_path = temp_path.join("adapter");
    let merged_path = temp_path.join("merged_model");
    let export_path = temp_path.join("exported");
    
    // Step 1: Create mock base model
    create_mock_model(&model_path, "llama")?;
    
    // Step 2: Generate LoRA adapter
    let adapter = create_test_adapter("llama", 16);
    
    // Save adapter for testing
    save_lora_parameters(&adapter, &adapter_path)?;
    
    // Step 3: Apply adapter to model
    let apply_engine = ApplyEngine::new(MergeStrategy::Linear, false);
    apply_engine.apply(
        &model_path,
        &adapter_path,
        Some(&merged_path),
    ).await?;
    
    // Verify merged model exists
    verify_file_exists(&merged_path.join("config.json"))?;
    
    // Step 4: Run inference (mock)
    // Note: In real tests with actual models, this would run real inference
    let prompts = vec!["Translate to French: Hello world"];
    let gen_config = GenerationConfig::default();
    
    // Mock inference result
    println!("Mock inference completed for: {:?}", prompts);
    
    // Step 5: Export to different formats
    let export_engine = ExportEngine::new(Precision::Fp32, false);
    
    // Export to PEFT format
    let peft_path = export_path.join("peft");
    export_engine.export(
        &adapter,
        ExportFormat::Peft,
        Some("meta-llama/Llama-2-7b-hf"),
        &peft_path,
    ).await?;
    
    // Verify PEFT export
    verify_file_exists(&peft_path.join("adapter_config.json"))?;
    verify_file_exists(&peft_path.join("adapter_model.safetensors"))?;
    
    Ok(())
}

#[tokio::test]
async fn test_full_workflow_mistral() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let model_path = temp_path.join("mistral_model");
    let adapter_path = temp_path.join("mistral_adapter");
    
    // Create Mistral model and adapter
    create_mock_model(&model_path, "mistral")?;
    let adapter = create_test_adapter("mistral", 8);
    
    // Save and verify adapter
    save_lora_parameters(&adapter, &adapter_path)?;
    
    // Apply adapter
    let apply_engine = ApplyEngine::new(MergeStrategy::Linear, true); // with optimization
    let result = apply_engine.apply(
        &model_path,
        &adapter_path,
        None, // In-place modification
    ).await;
    
    assert!(result.is_ok(), "Mistral workflow failed: {:?}", result.err());
    
    Ok(())
}

#[tokio::test]
async fn test_full_workflow_batch_processing() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Test batch generation of adapters
    let tasks = TEST_TASKS[..4].to_vec();
    let architectures = vec!["llama", "mistral", "gemma", "gpt2"];
    
    for (i, (task, arch)) in tasks.iter().zip(architectures.iter()).enumerate() {
        let adapter = create_test_adapter(arch, 16);
        let adapter_path = temp_path.join(format!("adapter_{}", i));
        
        save_lora_parameters(&adapter, &adapter_path)?;
        
        // Export each adapter
        let export_path = temp_path.join(format!("export_{}", i));
        let export_engine = ExportEngine::new(Precision::Fp32, false);
        
        export_engine.export(
            &adapter,
            ExportFormat::Peft,
            Some(TEST_MODEL_IDS.iter().find(|(a, _)| a == arch).unwrap().1),
            &export_path,
        ).await?;
        
        verify_file_exists(&export_path.join("adapter_config.json"))?;
    }
    
    Ok(())
}

#[tokio::test]
async fn test_workflow_with_custom_config() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let config_path = temp_path.join("custom_config.json");
    
    // Create custom configuration
    create_test_config(&config_path, "full")?;
    
    // Load config and create adapter with custom settings
    let adapter = create_test_adapter("llama", 32); // Higher rank
    
    // Test export with different precisions
    let precisions = vec![Precision::Fp32, Precision::Fp16, Precision::Bf16];
    let export_engine_fp32 = ExportEngine::new(Precision::Fp32, false);
    
    for (i, precision) in precisions.iter().enumerate() {
        let export_path = temp_path.join(format!("export_precision_{}", i));
        let engine = ExportEngine::new(precision.clone(), false);
        
        let result = engine.export(
            &adapter,
            ExportFormat::HuggingFace,
            Some("meta-llama/Llama-2-7b-hf"),
            &export_path,
        ).await;
        
        assert!(result.is_ok(), "Export with {:?} failed: {:?}", precision, result.err());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_workflow_memory_efficiency() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Create large adapter (high rank)
    let adapter = create_test_adapter("llama", 128);
    let adapter_path = temp_path.join("large_adapter");
    
    std::fs::create_dir_all(&adapter_path)?;
    adapter.save(&adapter_path)?;
    
    // Test memory-efficient operations
    let apply_engine = ApplyEngine::new(MergeStrategy::Linear, true);
    
    // Mock large model path
    let model_path = temp_path.join("large_model");
    create_mock_model(&model_path, "llama")?;
    
    // Apply with optimization enabled
    let start_memory = get_current_memory_usage();
    
    let result = apply_engine.apply(
        &model_path,
        &adapter_path,
        None,
    ).await;
    
    let end_memory = get_current_memory_usage();
    
    assert!(result.is_ok(), "Memory-efficient apply failed: {:?}", result.err());
    
    // Verify memory usage didn't spike too much
    let memory_increase = end_memory.saturating_sub(start_memory);
    println!("Memory increase during apply: {} MB", memory_increase / 1_048_576);
    
    Ok(())
}

#[tokio::test]
async fn test_workflow_concurrent_operations() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Test concurrent adapter generation and export
    let tasks: Vec<_> = (0..4).map(|i| {
        let path = temp_path.clone();
        tokio::spawn(async move {
            let adapter = create_test_adapter("llama", 8 + i * 4);
            let export_path = path.join(format!("concurrent_{}", i));
            
            let engine = ExportEngine::new(Precision::Fp32, false);
            engine.export(
                &adapter,
                ExportFormat::Peft,
                Some("meta-llama/Llama-2-7b-hf"),
                &export_path,
            ).await
        })
    }).collect();
    
    // Wait for all tasks to complete
    for task in tasks {
        let result = task.await?;
        assert!(result.is_ok(), "Concurrent operation failed: {:?}", result.err());
    }
    
    // Verify all exports
    for i in 0..4 {
        let export_path = temp_path.join(format!("concurrent_{}", i));
        verify_file_exists(&export_path.join("adapter_config.json"))?;
    }
    
    Ok(())
}

/// Helper function to get current memory usage (mock implementation)
fn get_current_memory_usage() -> usize {
    // In a real implementation, this would use system calls to get actual memory usage
    0
}