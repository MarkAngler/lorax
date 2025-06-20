//! Tests for OpenAI format export

use anyhow::Result;
use t2l_core::lora::{LoraParameters, LoraParameterConfig, LoraLayer, ParameterMetadata};
use t2l_core::export::openai::export_to_openai;
use tempfile::tempdir;
use std::collections::HashMap;
use chrono::Utc;

/// Create test LoRA parameters
fn create_test_lora_params() -> LoraParameters {
    let config = LoraParameterConfig {
        target_architecture: "llama".to_string(),
        default_rank: 8,
        default_alpha: 16.0,
        target_modules: vec![
            "q_proj".to_string(),
            "v_proj".to_string(),
            "k_proj".to_string(),
            "o_proj".to_string(),
        ],
        merge_weights: false,
    };
    
    let mut params = LoraParameters::new(config);
    
    // Add test layers
    let layer1 = LoraLayer::with_random_weights(
        "model.layers.0.self_attn.q_proj".to_string(),
        4096,
        4096,
        8,
        16.0,
    );
    params.add_layer(layer1).unwrap();
    
    let layer2 = LoraLayer::with_random_weights(
        "model.layers.0.self_attn.v_proj".to_string(),
        4096,
        4096,
        8,
        16.0,
    );
    params.add_layer(layer2).unwrap();
    
    // Add metadata
    let metadata = ParameterMetadata {
        task_description: "Generate helpful AI assistant responses".to_string(),
        created_at: Utc::now(),
        generator_version: "t2l-v0.1.0".to_string(),
        hyperparameters: HashMap::new(),
        metrics: None,
    };
    params.set_metadata(metadata);
    
    params
}

#[tokio::test]
async fn test_openai_export_basic() -> Result<()> {
    let adapter = create_test_lora_params();
    let temp_dir = tempdir()?;
    let output_path = temp_dir.path().join("openai_export");
    
    // Export to OpenAI format
    export_to_openai(&adapter, &output_path).await?;
    
    // Verify key files exist
    assert!(output_path.join("model_metadata.json").exists());
    assert!(output_path.join("deployment_config.json").exists());
    assert!(output_path.join("openai_api_spec.json").exists());
    assert!(output_path.join("model_capabilities.json").exists());
    assert!(output_path.join("finetune_config.json").exists());
    assert!(output_path.join("README.md").exists());
    assert!(output_path.join("weights/adapter_config.json").exists());
    assert!(output_path.join("weights/weight_index.json").exists());
    assert!(output_path.join("examples/basic_chat.py").exists());
    assert!(output_path.join("examples/streaming_chat.py").exists());
    
    Ok(())
}

#[tokio::test]
async fn test_openai_metadata_generation() -> Result<()> {
    let adapter = create_test_lora_params();
    let temp_dir = tempdir()?;
    let output_path = temp_dir.path().join("openai_export");
    
    export_to_openai(&adapter, &output_path).await?;
    
    // Read and verify model metadata
    let metadata_content = tokio::fs::read_to_string(output_path.join("model_metadata.json")).await?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_content)?;
    
    assert_eq!(metadata["object"], "model");
    assert!(metadata["id"].as_str().unwrap().starts_with("ft:t2l-llama:t2l-org:"));
    assert_eq!(metadata["owned_by"], "t2l-org");
    assert!(metadata["permission"].is_array());
    assert_eq!(metadata["root"], "meta-llama/Llama-2-7b-hf");
    
    Ok(())
}

#[tokio::test]
async fn test_openai_deployment_config() -> Result<()> {
    let adapter = create_test_lora_params();
    let temp_dir = tempdir()?;
    let output_path = temp_dir.path().join("openai_export");
    
    export_to_openai(&adapter, &output_path).await?;
    
    // Read and verify deployment config
    let config_content = tokio::fs::read_to_string(output_path.join("deployment_config.json")).await?;
    let config: serde_json::Value = serde_json::from_str(&config_content)?;
    
    assert_eq!(config["endpoint"]["api_version"], "v1");
    assert_eq!(config["serving"]["engine"], "t2l-inference");
    assert_eq!(config["serving"]["max_tokens"], 4096);
    assert_eq!(config["rate_limits"]["rpm"], 3500);
    
    Ok(())
}

#[tokio::test]
async fn test_openai_api_spec() -> Result<()> {
    let adapter = create_test_lora_params();
    let temp_dir = tempdir()?;
    let output_path = temp_dir.path().join("openai_export");
    
    export_to_openai(&adapter, &output_path).await?;
    
    // Read and verify API spec
    let spec_content = tokio::fs::read_to_string(output_path.join("openai_api_spec.json")).await?;
    let spec: serde_json::Value = serde_json::from_str(&spec_content)?;
    
    assert_eq!(spec["openapi"], "3.0.0");
    assert!(spec["paths"]["/chat/completions"].is_object());
    assert!(spec["components"]["schemas"]["ChatCompletionRequest"].is_object());
    
    Ok(())
}

#[tokio::test]
async fn test_openai_weight_export() -> Result<()> {
    let adapter = create_test_lora_params();
    let temp_dir = tempdir()?;
    let output_path = temp_dir.path().join("openai_export");
    
    export_to_openai(&adapter, &output_path).await?;
    
    // Read and verify adapter config
    let adapter_config_content = tokio::fs::read_to_string(output_path.join("weights/adapter_config.json")).await?;
    let adapter_config: serde_json::Value = serde_json::from_str(&adapter_config_content)?;
    
    assert_eq!(adapter_config["adapter_type"], "lora");
    assert_eq!(adapter_config["r"], 8);
    assert_eq!(adapter_config["lora_alpha"], 16.0);
    assert_eq!(adapter_config["architecture"], "llama");
    
    // Verify weight index
    let index_content = tokio::fs::read_to_string(output_path.join("weights/weight_index.json")).await?;
    let index: serde_json::Value = serde_json::from_str(&index_content)?;
    
    assert_eq!(index["metadata"]["format"], "t2l-openai");
    assert!(index["weight_map"].is_object());
    
    Ok(())
}

#[tokio::test]
async fn test_openai_capabilities_doc() -> Result<()> {
    let adapter = create_test_lora_params();
    let temp_dir = tempdir()?;
    let output_path = temp_dir.path().join("openai_export");
    
    export_to_openai(&adapter, &output_path).await?;
    
    // Read and verify capabilities
    let capabilities_content = tokio::fs::read_to_string(output_path.join("model_capabilities.json")).await?;
    let capabilities: serde_json::Value = serde_json::from_str(&capabilities_content)?;
    
    assert_eq!(capabilities["model_type"], "text-generation");
    assert_eq!(capabilities["capabilities"]["chat"], true);
    assert_eq!(capabilities["capabilities"]["streaming"], true);
    assert_eq!(capabilities["parameters"]["adapter_type"], "lora");
    assert_eq!(capabilities["context_window"]["max_tokens"], 4096);
    
    Ok(())
}

#[tokio::test]
async fn test_openai_readme_generation() -> Result<()> {
    let adapter = create_test_lora_params();
    let temp_dir = tempdir()?;
    let output_path = temp_dir.path().join("openai_export");
    
    export_to_openai(&adapter, &output_path).await?;
    
    // Read and verify README
    let readme_content = tokio::fs::read_to_string(output_path.join("README.md")).await?;
    
    assert!(readme_content.contains("T2L OpenAI-Compatible Model"));
    assert!(readme_content.contains("Model Information"));
    assert!(readme_content.contains("Quick Start"));
    assert!(readme_content.contains("API Endpoints"));
    assert!(readme_content.contains("from openai import OpenAI"));
    
    Ok(())
}

#[tokio::test]
async fn test_openai_examples() -> Result<()> {
    let adapter = create_test_lora_params();
    let temp_dir = tempdir()?;
    let output_path = temp_dir.path().join("openai_export");
    
    export_to_openai(&adapter, &output_path).await?;
    
    // Verify example scripts
    let basic_example = tokio::fs::read_to_string(output_path.join("examples/basic_chat.py")).await?;
    assert!(basic_example.contains("from openai import OpenAI"));
    assert!(basic_example.contains("client.chat.completions.create"));
    
    let streaming_example = tokio::fs::read_to_string(output_path.join("examples/streaming_chat.py")).await?;
    assert!(streaming_example.contains("stream=True"));
    assert!(streaming_example.contains("for chunk in stream:"));
    
    Ok(())
}

#[tokio::test]
async fn test_openai_finetune_config() -> Result<()> {
    let adapter = create_test_lora_params();
    let temp_dir = tempdir()?;
    let output_path = temp_dir.path().join("openai_export");
    
    export_to_openai(&adapter, &output_path).await?;
    
    // Read and verify fine-tuning config
    let finetune_content = tokio::fs::read_to_string(output_path.join("finetune_config.json")).await?;
    let finetune: serde_json::Value = serde_json::from_str(&finetune_content)?;
    
    assert_eq!(finetune["fine_tuning"]["adapter_config"]["r"], 8);
    assert_eq!(finetune["fine_tuning"]["adapter_config"]["task_type"], "CAUSAL_LM");
    assert_eq!(finetune["fine_tuning"]["training_config"]["learning_rate"], 1e-4);
    assert_eq!(finetune["fine_tuning"]["data_config"]["max_seq_length"], 2048);
    
    Ok(())
}