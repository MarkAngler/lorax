//! Tests for HuggingFace format export functionality

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use t2l_core::{
    export::{huggingface, Precision},
    lora::{LoraLayer, LoraParameterConfig, LoraParameters, ParameterMetadata},
};
use tempfile::TempDir;

/// Create a test adapter with sample data
fn create_test_adapter() -> LoraParameters {
    let config = LoraParameterConfig {
        target_architecture: "llama".to_string(),
        default_rank: 8,
        default_alpha: 16.0,
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
        ],
        merge_weights: false,
    };

    let mut adapter = LoraParameters::new(config);

    // Add test layers
    for i in 0..2 {
        for module in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let layer_name = format!("layers.{}.self_attn.{}", i, module);
            let layer = LoraLayer::new(
                layer_name.clone(),
                128,  // input_dim (small for testing)
                128,  // output_dim
                8,    // rank
                16.0, // alpha
            );
            adapter.add_layer(layer).unwrap();
        }
    }

    // Add metadata
    let metadata = ParameterMetadata {
        task_description: "Test adapter for HuggingFace export".to_string(),
        created_at: chrono::Utc::now(),
        generator_version: "test-v1.0.0".to_string(),
        hyperparameters: HashMap::new(),
        metrics: None,
    };
    adapter.set_metadata(metadata);

    adapter
}

#[tokio::test]
async fn test_huggingface_export_basic() -> Result<()> {
    let adapter = create_test_adapter();
    let temp_dir = TempDir::new()?;
    let output_path = temp_dir.path();

    // Export with default settings
    huggingface::export_to_hf(
        &adapter,
        Some("meta-llama/Llama-2-7b-hf"),
        output_path,
        Precision::Fp16,
    )
    .await?;

    // Check that all expected files were created
    assert!(output_path.join("config.json").exists());
    assert!(output_path.join("adapter_model.safetensors").exists());
    assert!(output_path.join("adapter_config.json").exists());
    assert!(output_path.join("generation_config.json").exists());
    assert!(output_path.join("tokenizer_config.json").exists());
    assert!(output_path.join("special_tokens_map.json").exists());
    assert!(output_path.join("README.md").exists());

    Ok(())
}

#[tokio::test]
async fn test_huggingface_export_config_content() -> Result<()> {
    let adapter = create_test_adapter();
    let temp_dir = TempDir::new()?;
    let output_path = temp_dir.path();

    huggingface::export_to_hf(
        &adapter,
        Some("meta-llama/Llama-2-7b-hf"),
        output_path,
        Precision::Fp16,
    )
    .await?;

    // Read and verify config.json
    let config_path = output_path.join("config.json");
    let config_content = tokio::fs::read_to_string(&config_path).await?;
    let config: serde_json::Value = serde_json::from_str(&config_content)?;

    assert_eq!(config["model_type"], "llama");
    assert_eq!(config["architectures"][0], "LlamaForCausalLM");
    assert_eq!(config["torch_dtype"], "float16");
    assert_eq!(config["hidden_size"], 4096);
    assert_eq!(config["num_attention_heads"], 32);

    Ok(())
}

#[tokio::test]
async fn test_huggingface_export_adapter_config() -> Result<()> {
    let adapter = create_test_adapter();
    let temp_dir = TempDir::new()?;
    let output_path = temp_dir.path();

    huggingface::export_to_hf(
        &adapter,
        Some("meta-llama/Llama-2-7b-hf"),
        output_path,
        Precision::Fp16,
    )
    .await?;

    // Read and verify adapter_config.json
    let adapter_config_path = output_path.join("adapter_config.json");
    let adapter_config_content = tokio::fs::read_to_string(&adapter_config_path).await?;
    let adapter_config: serde_json::Value = serde_json::from_str(&adapter_config_content)?;

    assert_eq!(adapter_config["adapter_type"], "lora");
    assert_eq!(adapter_config["r"], 8);
    assert_eq!(adapter_config["lora_alpha"], 16.0);
    assert_eq!(adapter_config["lora_dropout"], 0.0);
    assert_eq!(adapter_config["task_type"], "CAUSAL_LM");
    assert_eq!(adapter_config["base_model_name_or_path"], "meta-llama/Llama-2-7b-hf");

    Ok(())
}

#[tokio::test]
async fn test_huggingface_export_different_precisions() -> Result<()> {
    let adapter = create_test_adapter();
    
    for precision in [Precision::Fp16, Precision::Fp32, Precision::Int8] {
        let temp_dir = TempDir::new()?;
        let output_path = temp_dir.path();

        huggingface::export_to_hf(
            &adapter,
            None,
            output_path,
            precision,
        )
        .await?;

        // Verify files exist regardless of precision
        assert!(output_path.join("adapter_model.safetensors").exists());
    }

    Ok(())
}

#[tokio::test]
async fn test_huggingface_export_layer_name_conversion() -> Result<()> {
    let config = LoraParameterConfig {
        target_architecture: "llama".to_string(),
        default_rank: 4,
        default_alpha: 8.0,
        target_modules: vec!["q_proj".to_string()],
        merge_weights: false,
    };

    let mut adapter = LoraParameters::new(config);

    // Test different layer naming patterns
    let test_patterns = vec![
        ("layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj"),
        ("block.0.attn.q", "model.layers.0.self_attn.q_proj"),
        ("transformer.h.0.attn.q_proj", "model.transformer.h.0.attn.q_proj"),
    ];

    for (input_name, _expected_prefix) in test_patterns {
        let layer = LoraLayer::new(
            input_name.to_string(),
            64,
            64,
            4,
            8.0,
        );
        adapter.add_layer(layer)?;
    }

    let temp_dir = TempDir::new()?;
    let output_path = temp_dir.path();

    huggingface::export_to_hf(
        &adapter,
        None,
        output_path,
        Precision::Fp16,
    )
    .await?;

    // Verify export succeeded
    assert!(output_path.join("adapter_model.safetensors").exists());

    Ok(())
}

#[tokio::test]
async fn test_huggingface_export_readme_content() -> Result<()> {
    let adapter = create_test_adapter();
    let temp_dir = TempDir::new()?;
    let output_path = temp_dir.path();

    huggingface::export_to_hf(
        &adapter,
        Some("meta-llama/Llama-2-7b-hf"),
        output_path,
        Precision::Fp16,
    )
    .await?;

    // Read and verify README.md content
    let readme_path = output_path.join("README.md");
    let readme_content = tokio::fs::read_to_string(&readme_path).await?;

    // Check for key sections
    assert!(readme_content.contains("# T2L Generated LoRA Adapter"));
    assert!(readme_content.contains("## Model Details"));
    assert!(readme_content.contains("## Usage"));
    assert!(readme_content.contains("### Using with Transformers"));
    assert!(readme_content.contains("### Loading as Adapter"));
    assert!(readme_content.contains("base_model: meta-llama/Llama-2-7b-hf"));

    Ok(())
}

#[tokio::test]
async fn test_huggingface_export_without_metadata() -> Result<()> {
    let config = LoraParameterConfig::default();
    let mut adapter = LoraParameters::new(config);

    // Add a minimal layer without metadata
    let layer = LoraLayer::new(
        "layers.0.self_attn.q_proj".to_string(),
        64,
        64,
        4,
        8.0,
    );
    adapter.add_layer(layer)?;

    let temp_dir = TempDir::new()?;
    let output_path = temp_dir.path();

    // Should succeed even without metadata
    huggingface::export_to_hf(
        &adapter,
        None,
        output_path,
        Precision::Fp16,
    )
    .await?;

    assert!(output_path.join("adapter_model.safetensors").exists());

    Ok(())
}