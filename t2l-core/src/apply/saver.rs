//! Model saver for saving adapted models in various formats
//!
//! This module provides functionality to save models with applied LoRA adapters
//! in different formats including SafeTensors, PyTorch, and GGUF.

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{Device, Tensor};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, info, warn};

use crate::apply::loader::BaseModel;
use crate::apply::OutputFormat;
use crate::utils::tensor::{
    self as tensor_utils,
    safetensors::{SafeTensorsHeader, TensorInfo, write_safetensors},
    format::{PeftConfig, GgmlConfig, to_peft_format, to_ggml_format},
    TensorDataType, TensorMetadata, DeviceType, Tensor as UtilsTensor,
};

/// Model metadata for saving
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model architecture
    pub architecture: String,
    /// Model configuration as JSON
    pub config: Value,
    /// Tokenizer configuration (optional)
    pub tokenizer_config: Option<Value>,
    /// Additional metadata
    pub extra_metadata: HashMap<String, String>,
    /// Generation info
    pub generation_info: Option<GenerationInfo>,
}

/// Information about the LoRA generation process
#[derive(Debug, Clone)]
pub struct GenerationInfo {
    /// Original task description
    pub task_description: String,
    /// Generation timestamp
    pub timestamp: String,
    /// Generator version
    pub version: String,
    /// Applied LoRA parameters summary
    pub lora_summary: LoraSummary,
}

/// Summary of applied LoRA parameters
#[derive(Debug, Clone)]
pub struct LoraSummary {
    /// Number of adapted layers
    pub num_layers: usize,
    /// Total parameters added
    pub total_params: usize,
    /// Average rank
    pub avg_rank: f64,
    /// Average alpha
    pub avg_alpha: f64,
    /// Target modules
    pub target_modules: Vec<String>,
}

/// Save adapted model to specified path and format
pub async fn save_model(
    model: &BaseModel,
    output_path: &Path,
    format: OutputFormat,
) -> Result<()> {
    info!(
        "Saving {} model to {} in {:?} format",
        model.architecture_name(),
        output_path.display(),
        format
    );

    // Create output directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).await
            .with_context(|| format!("Failed to create output directory: {}", parent.display()))?;
    }

    // Extract model metadata
    let metadata = extract_model_metadata(model).await?;

    // Save model based on format
    match format {
        OutputFormat::SafeTensors => save_safetensors_model(model, output_path, &metadata).await,
        OutputFormat::PyTorch => save_pytorch_model(model, output_path, &metadata).await,
        OutputFormat::GGUF => save_gguf_model(model, output_path, &metadata).await,
    }
}

/// Save model in SafeTensors format
async fn save_safetensors_model(
    model: &BaseModel,
    output_path: &Path,
    metadata: &ModelMetadata,
) -> Result<()> {
    info!("Saving model in SafeTensors format");

    // Create model directory
    let model_dir = if output_path.extension().is_some() {
        // If output_path has extension, create parent directory
        output_path.parent().unwrap_or(output_path).to_path_buf()
    } else {
        // If no extension, treat as directory
        output_path.to_path_buf()
    };

    fs::create_dir_all(&model_dir).await
        .with_context(|| format!("Failed to create model directory: {}", model_dir.display()))?;

    // Extract tensors from model
    let tensors = extract_model_tensors(model).await?;

    // Convert to SafeTensors format
    let safetensors_path = model_dir.join("model.safetensors");
    let metadata_map = create_safetensors_metadata(metadata)?;
    
    write_safetensors(&safetensors_path, &tensors, Some(&metadata_map))
        .context("Failed to write SafeTensors file")?;

    // Save configuration files
    save_model_config(&model_dir, metadata).await?;
    save_tokenizer_config(&model_dir, metadata).await?;
    save_generation_info(&model_dir, metadata).await?;

    info!("Successfully saved SafeTensors model to: {}", model_dir.display());
    Ok(())
}

/// Save model in PyTorch format (placeholder implementation)
async fn save_pytorch_model(
    model: &BaseModel,
    output_path: &Path,
    metadata: &ModelMetadata,
) -> Result<()> {
    info!("Saving model in PyTorch format (placeholder implementation)");

    // Create model directory
    let model_dir = if output_path.extension().is_some() {
        output_path.parent().unwrap_or(output_path).to_path_buf()
    } else {
        output_path.to_path_buf()
    };

    fs::create_dir_all(&model_dir).await
        .with_context(|| format!("Failed to create model directory: {}", model_dir.display()))?;

    // For now, we'll save a placeholder file and the configuration
    let pytorch_path = model_dir.join("pytorch_model.bin");
    let placeholder_data = b"PyTorch model saving not fully implemented - use SafeTensors format instead";
    fs::write(&pytorch_path, placeholder_data).await
        .context("Failed to write PyTorch placeholder file")?;

    // Save configuration files
    save_model_config(&model_dir, metadata).await?;
    save_tokenizer_config(&model_dir, metadata).await?;
    save_generation_info(&model_dir, metadata).await?;

    warn!("PyTorch format saving is not fully implemented. Consider using SafeTensors format for full compatibility.");
    info!("Saved PyTorch model placeholder to: {}", model_dir.display());

    Ok(())
}

/// Save model in GGUF format (placeholder implementation)
async fn save_gguf_model(
    model: &BaseModel,
    output_path: &Path,
    metadata: &ModelMetadata,
) -> Result<()> {
    info!("Saving model in GGUF format (placeholder implementation)");

    // Extract tensors from model
    let tensors = extract_model_tensors(model).await?;

    // Create GGUF configuration (simplified)
    let config = GgmlConfig {
        version: 1,
        vocab_size: extract_vocab_size(&metadata.config).unwrap_or(32000),
        embedding_size: extract_embedding_size(&metadata.config).unwrap_or(4096),
        quantization: TensorDataType::Float16, // Default to f16 for GGUF
    };

    // Convert tensors to GGML format
    let gguf_data = to_ggml_format(&tensors, &config)
        .context("Failed to convert tensors to GGML format")?;

    // Write GGUF file
    let gguf_path = if output_path.extension() == Some(std::ffi::OsStr::new("gguf")) {
        output_path.to_path_buf()
    } else {
        output_path.with_extension("gguf")
    };

    fs::write(&gguf_path, &gguf_data).await
        .context("Failed to write GGUF file")?;

    // Save additional metadata
    if let Some(parent) = gguf_path.parent() {
        save_generation_info(parent, metadata).await?;
    }

    warn!("GGUF format saving is experimental. The format may not be fully compatible with all GGUF readers.");
    info!("Saved GGUF model to: {}", gguf_path.display());

    Ok(())
}

/// Extract model metadata from base model
async fn extract_model_metadata(model: &BaseModel) -> Result<ModelMetadata> {
    let architecture = model.architecture_name().to_string();
    let config = model.config_json()?;
    
    // Create basic metadata
    let mut extra_metadata = HashMap::new();
    extra_metadata.insert("format".to_string(), "t2l-adapted".to_string());
    extra_metadata.insert("architecture".to_string(), architecture.clone());
    extra_metadata.insert("created_by".to_string(), "t2l-core".to_string());
    extra_metadata.insert("created_at".to_string(), chrono::Utc::now().to_rfc3339());

    Ok(ModelMetadata {
        architecture,
        config,
        tokenizer_config: None, // TODO: Extract if available
        extra_metadata,
        generation_info: None, // TODO: Extract from LoRA parameters if available
    })
}

/// Extract tensors from base model
async fn extract_model_tensors(model: &BaseModel) -> Result<HashMap<String, UtilsTensor>> {
    let mut tensors = HashMap::new();

    match model {
        BaseModel::Llama { model: _, config: _ } => {
            // For Llama models, we need to extract tensors from the model structure
            // This is a placeholder implementation
            info!("Extracting tensors from Llama model (placeholder implementation)");
            
            // In a real implementation, you would:
            // 1. Iterate through all model layers
            // 2. Extract weight tensors from each layer
            // 3. Convert to our tensor format
            // 4. Store in the HashMap
            
            // For now, create a placeholder tensor
            create_placeholder_tensors(&mut tensors, "llama")?;
        }
        BaseModel::Mistral { weights, config: _ } => {
            info!("Extracting tensors from Mistral model");
            convert_candle_tensors_to_utils(weights, &mut tensors).await?;
        }
        BaseModel::Gemma { weights, config: _ } => {
            info!("Extracting tensors from Gemma model");
            convert_candle_tensors_to_utils(weights, &mut tensors).await?;
        }
    }

    info!("Extracted {} tensors from model", tensors.len());
    Ok(tensors)
}

/// Convert candle tensors to our utility tensor format
async fn convert_candle_tensors_to_utils(
    candle_tensors: &HashMap<String, Tensor>,
    output_tensors: &mut HashMap<String, UtilsTensor>,
) -> Result<()> {
    for (name, tensor) in candle_tensors {
        let tensor_data = extract_tensor_data(tensor)?;
        let metadata = create_tensor_metadata(name, tensor)?;
        
        let utils_tensor = UtilsTensor {
            data: tensor_data,
            metadata,
        };
        
        output_tensors.insert(name.clone(), utils_tensor);
    }
    
    Ok(())
}

/// Extract raw data from candle tensor
fn extract_tensor_data(tensor: &Tensor) -> Result<Vec<u8>> {
    // This is a simplified implementation
    // In practice, you'd need to handle different dtypes properly
    match tensor.dtype() {
        candle_core::DType::F32 => {
            let f32_data = tensor.to_vec1::<f32>()
                .context("Failed to convert tensor to f32 vector")?;
            Ok(bytemuck::cast_slice::<f32, u8>(&f32_data).to_vec())
        }
        candle_core::DType::F16 => {
            let f16_data = tensor.to_vec1::<half::f16>()
                .context("Failed to convert tensor to f16 vector")?;
            let mut bytes = Vec::with_capacity(f16_data.len() * 2);
            for f16_val in f16_data {
                bytes.extend_from_slice(&f16_val.to_le_bytes());
            }
            Ok(bytes)
        }
        _ => {
            bail!("Unsupported tensor dtype: {:?}", tensor.dtype());
        }
    }
}

/// Create tensor metadata from candle tensor
fn create_tensor_metadata(name: &str, tensor: &Tensor) -> Result<TensorMetadata> {
    let shape = tensor.dims().to_vec();
    
    let dtype = match tensor.dtype() {
        candle_core::DType::F32 => TensorDataType::Float32,
        candle_core::DType::F16 => TensorDataType::Float16,
        candle_core::DType::BF16 => TensorDataType::BFloat16,
        candle_core::DType::I8 => TensorDataType::Int8,
        _ => return Err(anyhow!("Unsupported tensor dtype: {:?}", tensor.dtype())),
    };
    
    let device = match tensor.device() {
        Device::Cpu => DeviceType::Cpu,
        Device::Cuda(idx) => DeviceType::Cuda(*idx as u32),
        Device::Metal(_) => DeviceType::Metal,
        _ => DeviceType::Cpu,
    };

    Ok(TensorMetadata {
        shape,
        dtype,
        device,
        name: name.to_string(),
    })
}

/// Create placeholder tensors for testing
fn create_placeholder_tensors(
    tensors: &mut HashMap<String, UtilsTensor>,
    architecture: &str,
) -> Result<()> {
    // Create a few placeholder tensors for demonstration
    let tensor_configs = vec![
        ("embed_tokens.weight", vec![32000, 4096]),
        ("layers.0.self_attn.q_proj.weight", vec![4096, 4096]),
        ("layers.0.self_attn.k_proj.weight", vec![4096, 4096]),
        ("layers.0.self_attn.v_proj.weight", vec![4096, 4096]),
        ("layers.0.self_attn.o_proj.weight", vec![4096, 4096]),
        ("lm_head.weight", vec![32000, 4096]),
    ];

    for (name, shape) in tensor_configs {
        let total_elements = shape.iter().product::<usize>();
        let data = vec![0u8; total_elements * 4]; // f32 = 4 bytes per element
        
        let metadata = TensorMetadata {
            shape,
            dtype: TensorDataType::Float32,
            device: DeviceType::Cpu,
            name: name.to_string(),
        };
        
        let tensor = UtilsTensor { data, metadata };
        tensors.insert(name.to_string(), tensor);
    }
    
    info!("Created {} placeholder tensors for {}", tensors.len(), architecture);
    Ok(())
}

/// Create metadata for SafeTensors format
fn create_safetensors_metadata(metadata: &ModelMetadata) -> Result<HashMap<String, String>> {
    let mut meta_map = HashMap::new();
    
    // Add basic metadata
    meta_map.insert("architecture".to_string(), metadata.architecture.clone());
    meta_map.insert("format".to_string(), "safetensors".to_string());
    
    // Add extra metadata
    for (key, value) in &metadata.extra_metadata {
        meta_map.insert(key.clone(), value.clone());
    }
    
    // Add generation info if available
    if let Some(gen_info) = &metadata.generation_info {
        meta_map.insert("task_description".to_string(), gen_info.task_description.clone());
        meta_map.insert("generation_timestamp".to_string(), gen_info.timestamp.clone());
        meta_map.insert("generator_version".to_string(), gen_info.version.clone());
        meta_map.insert("lora_layers".to_string(), gen_info.lora_summary.num_layers.to_string());
        meta_map.insert("lora_total_params".to_string(), gen_info.lora_summary.total_params.to_string());
    }
    
    Ok(meta_map)
}

/// Save model configuration file
async fn save_model_config(output_dir: &Path, metadata: &ModelMetadata) -> Result<()> {
    let config_path = output_dir.join("config.json");
    let config_str = serde_json::to_string_pretty(&metadata.config)
        .context("Failed to serialize model config")?;
    
    fs::write(&config_path, config_str).await
        .with_context(|| format!("Failed to write config file: {}", config_path.display()))?;
    
    debug!("Saved model config to: {}", config_path.display());
    Ok(())
}

/// Save tokenizer configuration file
async fn save_tokenizer_config(output_dir: &Path, metadata: &ModelMetadata) -> Result<()> {
    if let Some(tokenizer_config) = &metadata.tokenizer_config {
        let tokenizer_path = output_dir.join("tokenizer_config.json");
        let tokenizer_str = serde_json::to_string_pretty(tokenizer_config)
            .context("Failed to serialize tokenizer config")?;
        
        fs::write(&tokenizer_path, tokenizer_str).await
            .with_context(|| format!("Failed to write tokenizer config: {}", tokenizer_path.display()))?;
        
        debug!("Saved tokenizer config to: {}", tokenizer_path.display());
    }
    
    Ok(())
}

/// Save generation information file
async fn save_generation_info(output_dir: &Path, metadata: &ModelMetadata) -> Result<()> {
    if let Some(gen_info) = &metadata.generation_info {
        let info_path = output_dir.join("t2l_generation_info.json");
        let info_str = serde_json::to_string_pretty(gen_info)
            .context("Failed to serialize generation info")?;
        
        fs::write(&info_path, info_str).await
            .with_context(|| format!("Failed to write generation info: {}", info_path.display()))?;
        
        debug!("Saved generation info to: {}", info_path.display());
    }
    
    Ok(())
}

/// Extract vocabulary size from model config
fn extract_vocab_size(config: &Value) -> Option<usize> {
    config.get("vocab_size")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
}

/// Extract embedding size from model config
fn extract_embedding_size(config: &Value) -> Option<usize> {
    config.get("hidden_size")
        .or_else(|| config.get("d_model"))
        .or_else(|| config.get("embedding_size"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
}

/// Get file size in a human-readable format
fn format_file_size(size: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = size as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    format!("{:.2} {}", size, UNITS[unit_index])
}

/// Validate saved model integrity
pub async fn validate_saved_model(model_path: &Path, format: OutputFormat) -> Result<()> {
    info!("Validating saved model at: {}", model_path.display());
    
    match format {
        OutputFormat::SafeTensors => validate_safetensors_model(model_path).await,
        OutputFormat::PyTorch => validate_pytorch_model(model_path).await,
        OutputFormat::GGUF => validate_gguf_model(model_path).await,
    }
}

/// Validate SafeTensors model
async fn validate_safetensors_model(model_path: &Path) -> Result<()> {
    let model_dir = if model_path.is_dir() {
        model_path
    } else {
        model_path.parent().unwrap_or(model_path)
    };
    
    // Check required files
    let required_files = ["config.json", "model.safetensors"];
    for file in &required_files {
        let file_path = model_dir.join(file);
        if !file_path.exists() {
            return Err(anyhow!("Required file missing: {}", file));
        }
    }
    
    // Validate SafeTensors file
    let safetensors_path = model_dir.join("model.safetensors");
    let _header = tensor_utils::safetensors::read_header(&safetensors_path)
        .context("Failed to read SafeTensors header")?;
    
    // Check file size
    let metadata = fs::metadata(&safetensors_path).await?;
    let file_size = metadata.len();
    
    info!(
        "SafeTensors model validation successful: {} ({}) tensors",
        _header.tensors.len(),
        format_file_size(file_size)
    );
    
    Ok(())
}

/// Validate PyTorch model
async fn validate_pytorch_model(model_path: &Path) -> Result<()> {
    let model_dir = if model_path.is_dir() {
        model_path
    } else {
        model_path.parent().unwrap_or(model_path)
    };
    
    // Check required files
    let pytorch_file = model_dir.join("pytorch_model.bin");
    let config_file = model_dir.join("config.json");
    
    if !pytorch_file.exists() {
        return Err(anyhow!("PyTorch model file missing: pytorch_model.bin"));
    }
    
    if !config_file.exists() {
        return Err(anyhow!("Config file missing: config.json"));
    }
    
    // Check file size
    let metadata = fs::metadata(&pytorch_file).await?;
    let file_size = metadata.len();
    
    info!(
        "PyTorch model validation successful: {}",
        format_file_size(file_size)
    );
    
    Ok(())
}

/// Validate GGUF model
async fn validate_gguf_model(model_path: &Path) -> Result<()> {
    let gguf_path = if model_path.extension() == Some(std::ffi::OsStr::new("gguf")) {
        model_path
    } else {
        return Err(anyhow!("GGUF file must have .gguf extension"));
    };
    
    if !gguf_path.exists() {
        return Err(anyhow!("GGUF file does not exist: {}", gguf_path.display()));
    }
    
    // Check file size
    let metadata = fs::metadata(gguf_path).await?;
    let file_size = metadata.len();
    
    if file_size < 100 {
        return Err(anyhow!("GGUF file appears to be too small: {} bytes", file_size));
    }
    
    info!(
        "GGUF model validation successful: {}",
        format_file_size(file_size)
    );
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_create_model_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let config = serde_json::json!({
            "architecture": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096
        });
        
        let metadata = ModelMetadata {
            architecture: "llama".to_string(),
            config,
            tokenizer_config: None,
            extra_metadata: HashMap::new(),
            generation_info: None,
        };
        
        assert_eq!(metadata.architecture, "llama");
        assert_eq!(extract_vocab_size(&metadata.config), Some(32000));
        assert_eq!(extract_embedding_size(&metadata.config), Some(4096));
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(1024), "1.00 KB");
        assert_eq!(format_file_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_file_size(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_tensor_data_type_conversion() {
        let dtype = TensorDataType::Float32;
        assert_eq!(format!("{:?}", dtype), "Float32");
        
        let device = DeviceType::Cpu;
        assert_eq!(format!("{:?}", device), "Cpu");
    }

    #[tokio::test]
    async fn test_safetensors_metadata_creation() {
        let mut extra_metadata = HashMap::new();
        extra_metadata.insert("test_key".to_string(), "test_value".to_string());
        
        let metadata = ModelMetadata {
            architecture: "llama".to_string(),
            config: serde_json::json!({"test": "config"}),
            tokenizer_config: None,
            extra_metadata,
            generation_info: None,
        };
        
        let safetensors_meta = create_safetensors_metadata(&metadata).unwrap();
        assert_eq!(safetensors_meta.get("architecture"), Some(&"llama".to_string()));
        assert_eq!(safetensors_meta.get("test_key"), Some(&"test_value".to_string()));
    }
}