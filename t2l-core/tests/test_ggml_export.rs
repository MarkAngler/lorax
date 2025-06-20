//! Integration tests for GGML export functionality

use t2l_core::export::{ggml, Precision};
use t2l_core::lora::{LoraLayer, LoraParameters, LoraParameterConfig};
use std::path::Path;
use tempfile::TempDir;

/// Create a test LoRA adapter with sample data
fn create_test_adapter() -> LoraParameters {
    let config = LoraParameterConfig {
        target_architecture: "llama".to_string(),
        default_rank: 16,
        default_alpha: 32.0,
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
        ],
        merge_weights: false,
    };
    
    let mut params = LoraParameters::new(config);
    
    // Add a few test layers
    for i in 0..3 {
        let layer_name = format!("layers.{}.self_attn.q_proj", i);
        let layer = LoraLayer::with_random_weights(
            layer_name,
            4096, // input_dim
            4096, // output_dim
            16,   // rank
            32.0, // alpha
        );
        params.add_layer(layer).unwrap();
    }
    
    params
}

#[tokio::test]
async fn test_ggml_export_fp32() {
    let adapter = create_test_adapter();
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("test_adapter.ggml");
    
    // Export with FP32 precision
    let result = ggml::export_to_ggml(&adapter, &output_path, Precision::Fp32).await;
    assert!(result.is_ok(), "Failed to export: {:?}", result);
    
    // Verify file was created
    assert!(output_path.exists());
    
    // Verify file has content
    let metadata = std::fs::metadata(&output_path).unwrap();
    assert!(metadata.len() > 0);
}

#[tokio::test]
async fn test_ggml_export_fp16() {
    let adapter = create_test_adapter();
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("test_adapter");
    
    // Export with FP16 precision (should add .ggml extension)
    let result = ggml::export_to_ggml(&adapter, &output_path, Precision::Fp16).await;
    assert!(result.is_ok(), "Failed to export: {:?}", result);
    
    // Verify file was created with extension
    let expected_path = output_path.with_extension("ggml");
    assert!(expected_path.exists());
}

#[tokio::test]
async fn test_ggml_export_int8() {
    let adapter = create_test_adapter();
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("test_adapter.ggml");
    
    // Export with INT8 quantization
    let result = ggml::export_to_ggml(&adapter, &output_path, Precision::Int8).await;
    assert!(result.is_ok(), "Failed to export: {:?}", result);
    
    // Verify file was created
    assert!(output_path.exists());
}

#[tokio::test]
async fn test_ggml_binary_format() {
    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::Read;
    
    let adapter = create_test_adapter();
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("test_adapter.ggml");
    
    // Export adapter
    ggml::export_to_ggml(&adapter, &output_path, Precision::Fp32).await.unwrap();
    
    // Read and verify binary format
    let mut file = std::fs::File::open(&output_path).unwrap();
    
    // Check magic number
    let magic = file.read_u32::<LittleEndian>().unwrap();
    assert_eq!(magic, 0x67676d6c); // "ggml"
    
    // Check version
    let version = file.read_u32::<LittleEndian>().unwrap();
    assert_eq!(version, 1);
    
    // Check tensor count (3 layers * 2 tensors per layer)
    let tensor_count = file.read_u32::<LittleEndian>().unwrap();
    assert_eq!(tensor_count, 6);
    
    // Check architecture
    let arch_len = file.read_u32::<LittleEndian>().unwrap();
    let mut arch_bytes = vec![0u8; arch_len as usize];
    file.read_exact(&mut arch_bytes).unwrap();
    let architecture = String::from_utf8(arch_bytes).unwrap();
    assert_eq!(architecture, "llama");
}

#[tokio::test]
async fn test_ggml_export_with_metadata() {
    use t2l_core::lora::ParameterMetadata;
    use chrono::Utc;
    use std::collections::HashMap;
    
    let mut adapter = create_test_adapter();
    
    // Add metadata
    let metadata = ParameterMetadata {
        task_description: "Test LoRA adapter".to_string(),
        created_at: Utc::now(),
        generator_version: "1.0.0".to_string(),
        hyperparameters: HashMap::new(),
        metrics: None,
    };
    adapter.set_metadata(metadata);
    
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("test_adapter.ggml");
    
    // Export should include metadata
    let result = ggml::export_to_ggml(&adapter, &output_path, Precision::Fp32).await;
    assert!(result.is_ok());
    assert!(output_path.exists());
}