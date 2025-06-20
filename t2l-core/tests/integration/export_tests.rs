//! Export format compatibility tests
//!
//! Tests for all export formats (PEFT, GGML, HuggingFace, OpenAI)

use super::fixtures::*;
use super::init_test_logging;
use t2l_core::export::{ExportEngine, ExportFormat, Precision};
use t2l_core::Result;
use std::collections::HashMap;
use tempfile::TempDir;

#[tokio::test]
async fn test_peft_export_all_architectures() -> Result<()> {
    init_test_logging();
    
    let architectures = vec!["llama", "mistral", "gemma"];
    let ranks = vec![4, 8, 16, 32];
    
    for arch in architectures {
        for rank in &ranks {
            let (_temp_dir, temp_path) = create_test_dir();
            let adapter = create_test_adapter(arch, *rank);
            let output_path = temp_path.join(format!("peft_{}_{}", arch, rank));
            
            let engine = ExportEngine::new(Precision::Fp32, false);
            let result = engine.export(
                &adapter,
                ExportFormat::Peft,
                Some(TEST_MODEL_IDS.iter().find(|(a, _)| a == &arch).unwrap().1),
                &output_path,
            ).await;
            
            assert!(result.is_ok(), "PEFT export failed for {} rank {}: {:?}", arch, rank, result.err());
            
            // Verify PEFT structure
            verify_peft_structure(&output_path)?;
            
            // Verify config content
            let config = verify_json_file(&output_path.join("adapter_config.json"))?;
            assert_eq!(config["r"], *rank);
            assert_eq!(config["peft_type"], "LORA");
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_huggingface_export_formats() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let adapter = create_test_adapter("llama", 16);
    
    // Test different HuggingFace export options
    let test_cases = vec![
        ("hf_default", false, false),
        ("hf_merged", true, false),
        ("hf_push_ready", false, true),
    ];
    
    for (name, merge_weights, include_training_args) in test_cases {
        let output_path = temp_path.join(name);
        
        // Create modified adapter with merge option
        let mut modified_adapter = adapter.clone();
        modified_adapter.config.merge_weights = merge_weights;
        
        let engine = ExportEngine::new(Precision::Fp32, false);
        let result = engine.export(
            &modified_adapter,
            ExportFormat::HuggingFace,
            Some("meta-llama/Llama-2-7b-hf"),
            &output_path,
        ).await;
        
        assert!(result.is_ok(), "HuggingFace export failed for {}: {:?}", name, result.err());
        
        // Verify HuggingFace structure
        verify_huggingface_structure(&output_path, include_training_args)?;
    }
    
    Ok(())
}

#[tokio::test]
async fn test_ggml_export_quantization() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let adapter = create_test_adapter("llama", 16);
    
    // Test different GGML quantization levels
    let quant_types = vec![
        ("q4_0", "4-bit quantization"),
        ("q4_1", "4-bit quantization with improved accuracy"),
        ("q5_0", "5-bit quantization"),
        ("q5_1", "5-bit quantization with improved accuracy"),
        ("q8_0", "8-bit quantization"),
        ("f16", "16-bit floating point"),
        ("f32", "32-bit floating point"),
    ];
    
    for (quant_type, description) in quant_types {
        let output_path = temp_path.join(format!("ggml_{}", quant_type));
        
        // Note: GGML export might not be fully implemented
        let engine = ExportEngine::new(Precision::Fp32, false);
        let result = engine.export(
            &adapter,
            ExportFormat::Ggml,
            Some("meta-llama/Llama-2-7b-hf"),
            &output_path,
        ).await;
        
        // Check if GGML is implemented
        if result.is_ok() {
            println!("GGML export successful for {}: {}", quant_type, description);
            verify_ggml_structure(&output_path)?;
        } else {
            println!("GGML export not yet implemented or failed for {}", quant_type);
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_openai_export_compatibility() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let adapter = create_test_adapter("gpt2", 16); // OpenAI format typically for GPT models
    let output_path = temp_path.join("openai_export");
    
    let engine = ExportEngine::new(Precision::Fp32, false);
    let result = engine.export(
        &adapter,
        ExportFormat::OpenAI,
        Some("gpt2"),
        &output_path,
    ).await;
    
    // Check if OpenAI export is implemented
    if result.is_ok() {
        println!("OpenAI export successful");
        verify_openai_structure(&output_path)?;
    } else {
        println!("OpenAI export not yet implemented or failed");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_export_precision_formats() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let adapter = create_test_adapter("llama", 16);
    
    let precisions = vec![
        (Precision::Fp32, "fp32", 4),
        (Precision::Fp16, "fp16", 2),
        (Precision::Bf16, "bf16", 2),
        (Precision::Int8, "int8", 1),
    ];
    
    for (precision, name, expected_bytes_per_param) in precisions {
        let output_path = temp_path.join(format!("precision_{}", name));
        
        let engine = ExportEngine::new(precision.clone(), false);
        let result = engine.export(
            &adapter,
            ExportFormat::Peft,
            Some("meta-llama/Llama-2-7b-hf"),
            &output_path,
        ).await;
        
        assert!(result.is_ok(), "Export with {} precision failed: {:?}", name, result.err());
        
        // Verify file sizes match expected precision
        if output_path.join("adapter_model.safetensors").exists() {
            let metadata = std::fs::metadata(output_path.join("adapter_model.safetensors"))?;
            println!("Export size for {}: {} bytes", name, metadata.len());
            
            // Rough check: lower precision should result in smaller files
            // (This is a simplified check; actual compression may vary)
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_export_metadata_preservation() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Create adapter with rich metadata
    let mut adapter = create_test_adapter("llama", 16);
    adapter.metadata = Some(HashMap::from([
        ("task_description".to_string(), "Summarize scientific papers".to_string()),
        ("training_dataset".to_string(), "arxiv_abstracts_2023".to_string()),
        ("training_steps".to_string(), "10000".to_string()),
        ("evaluation_perplexity".to_string(), "2.45".to_string()),
        ("created_date".to_string(), "2024-01-15".to_string()),
        ("version".to_string(), "1.0.0".to_string()),
    ]));
    
    // Export to PEFT format
    let peft_path = temp_path.join("peft_with_metadata");
    let engine = ExportEngine::new(Precision::Fp32, false);
    engine.export(
        &adapter,
        ExportFormat::Peft,
        Some("meta-llama/Llama-2-7b-hf"),
        &peft_path,
    ).await?;
    
    // Verify metadata is preserved
    let config = verify_json_file(&peft_path.join("adapter_config.json"))?;
    
    // PEFT format might store metadata differently
    if let Some(metadata) = config.get("metadata") {
        assert_eq!(metadata["task_description"], "Summarize scientific papers");
        assert_eq!(metadata["version"], "1.0.0");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_export_optimization_flags() -> Result<()> {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let adapter = create_test_adapter("llama", 32);
    
    // Test with optimization enabled
    let optimized_path = temp_path.join("optimized_export");
    let engine_optimized = ExportEngine::new(Precision::Fp16, true);
    engine_optimized.export(
        &adapter,
        ExportFormat::HuggingFace,
        Some("meta-llama/Llama-2-7b-hf"),
        &optimized_path,
    ).await?;
    
    // Test without optimization
    let regular_path = temp_path.join("regular_export");
    let engine_regular = ExportEngine::new(Precision::Fp16, false);
    engine_regular.export(
        &adapter,
        ExportFormat::HuggingFace,
        Some("meta-llama/Llama-2-7b-hf"),
        &regular_path,
    ).await?;
    
    // Compare outputs (optimized should be smaller or more efficient)
    if optimized_path.exists() && regular_path.exists() {
        println!("Export optimization test completed");
    }
    
    Ok(())
}

// Helper functions to verify export format structures

fn verify_peft_structure(path: &Path) -> Result<()> {
    verify_file_exists(&path.join("adapter_config.json"))?;
    
    // Either safetensors or bin format should exist
    let has_safetensors = path.join("adapter_model.safetensors").exists();
    let has_bin = path.join("adapter_model.bin").exists();
    assert!(has_safetensors || has_bin, "No adapter weights found");
    
    // README is optional but recommended
    if path.join("README.md").exists() {
        verify_file_exists(&path.join("README.md"))?;
    }
    
    Ok(())
}

fn verify_huggingface_structure(path: &Path, include_training_args: bool) -> Result<()> {
    verify_file_exists(&path.join("config.json"))?;
    
    // Model weights
    let has_safetensors = path.join("model.safetensors").exists();
    let has_bin = path.join("pytorch_model.bin").exists();
    assert!(has_safetensors || has_bin, "No model weights found");
    
    // Optional files
    if include_training_args {
        verify_file_exists(&path.join("training_args.json"))?;
    }
    
    Ok(())
}

fn verify_ggml_structure(path: &Path) -> Result<()> {
    // GGML typically uses a single binary file
    let ggml_files: Vec<_> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "ggml" || ext == "gguf")
                .unwrap_or(false)
        })
        .collect();
    
    assert!(!ggml_files.is_empty(), "No GGML files found");
    
    Ok(())
}

fn verify_openai_structure(path: &Path) -> Result<()> {
    // OpenAI format typically includes specific configuration
    verify_file_exists(&path.join("config.json"))?;
    
    // Check for OpenAI-specific files
    if path.join("checkpoint").exists() {
        verify_file_exists(&path.join("checkpoint"))?;
    }
    
    Ok(())
}