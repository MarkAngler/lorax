//! Integration tests for PEFT export functionality

use t2l_core::export::{ExportEngine, ExportFormat, Precision};
use t2l_core::lora::{LoraParameters, LoraLayer, LoraParameterConfig};
use std::collections::HashMap;
use std::path::Path;
use tempfile::TempDir;

/// Create a test adapter with sample data
fn create_test_adapter() -> LoraParameters {
    let mut layers = HashMap::new();
    
    // Add multiple test layers
    let layers_info = vec![
        ("layers.0.self_attn.q_proj", 768, 768, 16),
        ("layers.0.self_attn.v_proj", 768, 768, 16),
        ("layers.1.self_attn.q_proj", 768, 768, 16),
        ("layers.1.self_attn.v_proj", 768, 768, 16),
    ];
    
    for (name, input_dim, output_dim, rank) in layers_info {
        let mut layer = LoraLayer::new(
            name.to_string(),
            input_dim,
            output_dim,
            rank,
            32.0, // alpha
        );
        
        // Initialize with some test data
        layer.randomize_weights();
        layers.insert(layer.name.clone(), layer);
    }

    let config = LoraParameterConfig {
        target_architecture: "llama".to_string(),
        default_rank: 16,
        default_alpha: 32.0,
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
        ],
        merge_weights: false,
    };

    LoraParameters {
        layers,
        config,
        metadata: None,
    }
}

#[tokio::test]
async fn test_peft_export_basic() {
    let adapter = create_test_adapter();
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("peft_adapter");
    
    let engine = ExportEngine::new(Precision::Fp32, false);
    let result = engine.export(
        &adapter,
        ExportFormat::Peft,
        Some("meta-llama/Llama-2-7b-hf"),
        &output_path,
    ).await;
    
    assert!(result.is_ok(), "PEFT export failed: {:?}", result.err());
    
    // Verify output files exist
    assert!(output_path.join("adapter_config.json").exists());
    assert!(output_path.join("adapter_model.safetensors").exists());
    assert!(output_path.join("adapter_model.bin").exists());
    assert!(output_path.join("README.md").exists());
}

#[tokio::test]
async fn test_peft_export_fp16() {
    let adapter = create_test_adapter();
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("peft_adapter_fp16");
    
    let engine = ExportEngine::new(Precision::Fp16, false);
    let result = engine.export(
        &adapter,
        ExportFormat::Peft,
        Some("meta-llama/Llama-2-7b-hf"),
        &output_path,
    ).await;
    
    assert!(result.is_ok(), "PEFT export with FP16 failed: {:?}", result.err());
}

#[tokio::test]
async fn test_peft_config_content() {
    let adapter = create_test_adapter();
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("peft_config_test");
    
    let engine = ExportEngine::new(Precision::Fp32, false);
    engine.export(
        &adapter,
        ExportFormat::Peft,
        Some("meta-llama/Llama-2-7b-hf"),
        &output_path,
    ).await.unwrap();
    
    // Read and verify adapter_config.json
    let config_content = std::fs::read_to_string(output_path.join("adapter_config.json")).unwrap();
    let config: serde_json::Value = serde_json::from_str(&config_content).unwrap();
    
    assert_eq!(config["peft_type"], "LORA");
    assert_eq!(config["task_type"], "CAUSAL_LM");
    assert_eq!(config["r"], 16);
    assert_eq!(config["lora_alpha"], 32.0);
    assert_eq!(config["base_model_name_or_path"], "meta-llama/Llama-2-7b-hf");
}

#[tokio::test]
async fn test_peft_export_without_target_model() {
    let adapter = create_test_adapter();
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("peft_no_target");
    
    let engine = ExportEngine::new(Precision::Fp32, false);
    let result = engine.export(
        &adapter,
        ExportFormat::Peft,
        None, // No target model specified
        &output_path,
    ).await;
    
    assert!(result.is_ok(), "PEFT export without target model failed: {:?}", result.err());
    
    // Verify the config uses "unknown" as base model
    let config_content = std::fs::read_to_string(output_path.join("adapter_config.json")).unwrap();
    let config: serde_json::Value = serde_json::from_str(&config_content).unwrap();
    assert_eq!(config["base_model_name_or_path"], "unknown");
}