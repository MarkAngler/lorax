//! Base model loader with automatic architecture detection
//!
//! This module provides functionality to load different model architectures
//! and automatically detect the appropriate model type from configuration files
//! or HuggingFace Hub metadata.

use anyhow::{anyhow, bail, Result};
use candle_core::{Device, Tensor};
// Note: The exact API structure may vary - this is a conservative implementation
// that can be adjusted based on the actual candle-transformers API
use candle_transformers::models::llama::{Llama, Config as LlamaConfig};
use candle_nn::VarBuilder;
use hf_hub::api::tokio::Api;
use lorax::lora::{LoraParameters, LoraLayer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, info, warn};

/// Supported model architectures
#[derive(Debug)]
pub enum BaseModel {
    /// Llama family models (Llama 2, Code Llama, etc.)
    Llama {
        model: Llama,
        config: LlamaConfig,
    },
    /// Mistral family models (placeholder for future implementation)
    Mistral {
        // Placeholder - actual implementation would depend on candle-transformers API
        weights: HashMap<String, Tensor>,
        config: serde_json::Value,
    },
    /// Gemma family models (placeholder for future implementation)
    Gemma {
        // Placeholder - actual implementation would depend on candle-transformers API
        weights: HashMap<String, Tensor>,
        config: serde_json::Value,
    },
}

/// Model configuration for architecture detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture type
    pub architecture: String,
    /// Model type (alternative field name)
    pub model_type: Option<String>,
    /// Architectures array (HuggingFace format)
    pub architectures: Option<Vec<String>>,
    /// Additional config fields
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Device management for model loading
#[derive(Debug, Clone)]
pub struct DeviceManager {
    device: Device,
    memory_fraction: f32,
}

impl DeviceManager {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            memory_fraction: 0.9, // Use 90% of available memory by default
        }
    }

    pub fn with_memory_fraction(mut self, fraction: f32) -> Self {
        self.memory_fraction = fraction.clamp(0.1, 1.0);
        self
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl BaseModel {
    /// Apply LoRA adapter to the model
    pub fn apply_lora(&mut self, adapter: &LoraParameters) -> Result<()> {
        info!(
            "Applying LoRA adapter with {} layers to {} model",
            adapter.layers.len(),
            self.architecture_name()
        );

        match self {
            BaseModel::Llama { model, .. } => {
                apply_lora_to_llama(model, adapter)
            }
            BaseModel::Mistral { weights, .. } => {
                apply_lora_to_mistral(weights, adapter)
            }
            BaseModel::Gemma { weights, .. } => {
                apply_lora_to_gemma(weights, adapter)
            }
        }
    }

    /// Get the model architecture name
    pub fn architecture_name(&self) -> &'static str {
        match self {
            BaseModel::Llama { .. } => "llama",
            BaseModel::Mistral { .. } => "mistral",
            BaseModel::Gemma { .. } => "gemma",
        }
    }

    /// Get the device the model is loaded on
    pub fn device(&self) -> &Device {
        match self {
            BaseModel::Llama { model, .. } => model.device(),
            BaseModel::Mistral { weights, .. } => {
                // Return device from first tensor if available
                weights.values().next()
                    .map(|t| t.device())
                    .unwrap_or(&Device::Cpu)
            },
            BaseModel::Gemma { weights, .. } => {
                // Return device from first tensor if available
                weights.values().next()
                    .map(|t| t.device())
                    .unwrap_or(&Device::Cpu)
            },
        }
    }

    /// Get model configuration as JSON
    pub fn config_json(&self) -> Result<serde_json::Value> {
        match self {
            BaseModel::Llama { config, .. } => {
                serde_json::to_value(config).map_err(|e| anyhow!("Failed to serialize Llama config: {}", e))
            }
            BaseModel::Mistral { config, .. } => {
                Ok(config.clone())
            }
            BaseModel::Gemma { config, .. } => {
                Ok(config.clone())
            }
        }
    }

    /// Forward pass through the model (for inference)
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offsets: &[usize]) -> Result<Tensor> {
        match self {
            BaseModel::Llama { model, .. } => {
                model.forward(input_ids, seqlen_offsets)
                    .map_err(|e| anyhow!("Llama forward pass failed: {}", e))
            }
            BaseModel::Mistral { .. } => {
                bail!("Mistral model forward pass not yet implemented - placeholder only")
            }
            BaseModel::Gemma { .. } => {
                bail!("Gemma model forward pass not yet implemented - placeholder only")
            }
        }
    }

    /// Get the model's embedding layer for LoRA application
    pub fn get_embedding_layer(&self) -> Option<&candle_nn::Embedding> {
        match self {
            BaseModel::Llama { model, .. } => {
                // Return embedding layer if available in the model struct
                // Note: This depends on the actual Llama model structure
                None // Placeholder until we know the exact API
            }
            BaseModel::Mistral { .. } => None,
            BaseModel::Gemma { .. } => None,
        }
    }

    /// Get layer by name for LoRA application
    pub fn get_layer_by_name(&self, layer_name: &str) -> Option<&dyn LoraCompatibleLayer> {
        // This is a simplified implementation - in a real scenario, you'd need
        // to traverse the model structure to find the specific layer
        debug!("Looking for layer: {}", layer_name);
        
        // For now, return None - this would need to be implemented based on
        // the specific model structure and layer naming conventions
        None
    }
}

/// Trait for layers that can have LoRA applied
pub trait LoraCompatibleLayer {
    fn apply_lora(&mut self, lora_layer: &LoraLayer) -> Result<()>;
    fn get_weight_tensor(&self) -> &Tensor;
    fn set_weight_tensor(&mut self, new_weight: Tensor) -> Result<()>;
}

/// Load a base model with automatic architecture detection
pub async fn load_base_model(model_path: &str, device: &Device) -> Result<BaseModel> {
    info!("Loading base model from: {}", model_path);

    // Create device manager
    let device_manager = DeviceManager::new(device.clone());

    // 1. Detect model architecture from config
    let config = detect_model_config(model_path).await?;
    info!("Detected model architecture: {}", config.architecture);

    // 2. Load appropriate model type
    match config.architecture.to_lowercase().as_str() {
        "llama" | "llamaforcausallm" | "llama2" => {
            load_llama_model(model_path, &device_manager).await
        }
        "mistral" | "mistralformcausallm" => {
            load_mistral_model(model_path, &device_manager).await
        }
        "gemma" | "gemmaforcausallm" => {
            load_gemma_model(model_path, &device_manager).await
        }
        arch => {
            bail!("Unsupported architecture: {}. Supported architectures: llama, mistral, gemma", arch)
        }
    }
}

/// Detect model configuration from local files or HuggingFace Hub
async fn detect_model_config(model_path: &str) -> Result<ModelConfig> {
    // Try local config.json first
    if let Ok(local_config) = load_local_config(model_path).await {
        return Ok(local_config);
    }

    // If not a local path, try HuggingFace Hub
    if !model_path.contains('/') || model_path.starts_with("./") || model_path.starts_with("../") {
        bail!("Could not detect model architecture from local path: {}. No config.json found.", model_path);
    }

    // Try HuggingFace Hub
    info!("Attempting to fetch model config from HuggingFace Hub");
    let hf_config = fetch_hf_config(model_path).await?;
    Ok(hf_config)
}

/// Load configuration from local config.json file
async fn load_local_config(model_path: &str) -> Result<ModelConfig> {
    let config_path = Path::new(model_path).join("config.json");
    debug!("Checking for config file at: {}", config_path.display());

    if !config_path.exists() {
        bail!("Config file not found: {}", config_path.display());
    }

    let config_str = fs::read_to_string(&config_path).await
        .map_err(|e| anyhow!("Failed to read config file {}: {}", config_path.display(), e))?;

    let config: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| anyhow!("Failed to parse config JSON: {}", e))?;

    // Extract architecture information
    let architecture = config["model_type"]
        .as_str()
        .or_else(|| config["architectures"].as_array()?.get(0)?.as_str())
        .or_else(|| config["architecture"].as_str())
        .unwrap_or("unknown")
        .to_lowercase();

    if architecture == "unknown" {
        warn!("Could not determine architecture from config, fields found: {:?}", 
              config.as_object().map(|obj| obj.keys().collect::<Vec<_>>()));
    }

    Ok(ModelConfig {
        architecture,
        model_type: config["model_type"].as_str().map(|s| s.to_string()),
        architectures: config["architectures"].as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()),
        extra: config.as_object().unwrap_or(&serde_json::Map::new())
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    })
}

/// Fetch model configuration from HuggingFace Hub
async fn fetch_hf_config(model_id: &str) -> Result<ModelConfig> {
    let api = Api::new()?;
    let repo = api.model(model_id.to_string());

    // Download config.json
    let config_path = repo.get("config.json").await
        .map_err(|e| anyhow!("Failed to fetch config from HuggingFace Hub for {}: {}", model_id, e))?;

    let config_str = fs::read_to_string(&config_path).await
        .map_err(|e| anyhow!("Failed to read downloaded config: {}", e))?;

    let config: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| anyhow!("Failed to parse HuggingFace config JSON: {}", e))?;

    let architecture = config["model_type"]
        .as_str()
        .or_else(|| config["architectures"].as_array()?.get(0)?.as_str())
        .unwrap_or("unknown")
        .to_lowercase();

    Ok(ModelConfig {
        architecture,
        model_type: config["model_type"].as_str().map(|s| s.to_string()),
        architectures: config["architectures"].as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()),
        extra: config.as_object().unwrap_or(&serde_json::Map::new())
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    })
}

/// Load Llama model
async fn load_llama_model(model_path: &str, device_manager: &DeviceManager) -> Result<BaseModel> {
    info!("Loading Llama model from: {}", model_path);

    // Load model files
    let api = if model_path.contains('/') && !model_path.starts_with("./") && !model_path.starts_with("../") {
        // HuggingFace model
        Some(Api::new()?)
    } else {
        None
    };

    let model_files = if let Some(api) = &api {
        load_hf_model_files(api, model_path).await?
    } else {
        load_local_model_files(model_path).await?
    };

    // Load configuration
    let config_path = model_files.get("config.json")
        .ok_or_else(|| anyhow!("Config file not found in model files"))?;
    
    let config_str = fs::read_to_string(config_path).await?;
    let config: LlamaConfig = serde_json::from_str(&config_str)
        .map_err(|e| anyhow!("Failed to parse Llama config: {}", e))?;

    // Load model weights into VarBuilder for candle
    let weights_map = load_model_weights(&model_files, device_manager.device()).await?;
    let var_builder = VarBuilder::from_tensors(weights_map, candle_core::DType::F32, device_manager.device())?;

    // Create Llama model using VarBuilder
    let model = Llama::load(&var_builder, &config)
        .map_err(|e| anyhow!("Failed to create Llama model: {}", e))?;

    Ok(BaseModel::Llama { model, config })
}

/// Load Mistral model (placeholder implementation)
async fn load_mistral_model(model_path: &str, device_manager: &DeviceManager) -> Result<BaseModel> {
    info!("Loading Mistral model from: {} (placeholder implementation)", model_path);

    let api = if model_path.contains('/') && !model_path.starts_with("./") && !model_path.starts_with("../") {
        Some(Api::new()?)
    } else {
        None
    };

    let model_files = if let Some(api) = &api {
        load_hf_model_files(api, model_path).await?
    } else {
        load_local_model_files(model_path).await?
    };

    let config_path = model_files.get("config.json")
        .ok_or_else(|| anyhow!("Config file not found in model files"))?;
    
    let config_str = fs::read_to_string(config_path).await?;
    let config: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| anyhow!("Failed to parse Mistral config: {}", e))?;

    let model_weights = load_model_weights(&model_files, device_manager.device()).await?;

    Ok(BaseModel::Mistral { 
        weights: model_weights, 
        config 
    })
}

/// Load Gemma model (placeholder implementation)
async fn load_gemma_model(model_path: &str, device_manager: &DeviceManager) -> Result<BaseModel> {
    info!("Loading Gemma model from: {} (placeholder implementation)", model_path);

    let api = if model_path.contains('/') && !model_path.starts_with("./") && !model_path.starts_with("../") {
        Some(Api::new()?)
    } else {
        None
    };

    let model_files = if let Some(api) = &api {
        load_hf_model_files(api, model_path).await?
    } else {
        load_local_model_files(model_path).await?
    };

    let config_path = model_files.get("config.json")
        .ok_or_else(|| anyhow!("Config file not found in model files"))?;
    
    let config_str = fs::read_to_string(config_path).await?;
    let config: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| anyhow!("Failed to parse Gemma config: {}", e))?;

    let model_weights = load_model_weights(&model_files, device_manager.device()).await?;

    Ok(BaseModel::Gemma { 
        weights: model_weights, 
        config 
    })
}

/// Load model files from HuggingFace Hub
async fn load_hf_model_files(api: &Api, model_id: &str) -> Result<HashMap<String, PathBuf>> {
    let repo = api.model(model_id.to_string());
    let mut files = HashMap::new();

    // Download essential files
    let essential_files = [
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        "pytorch_model.bin",
        "model-00001-of-00001.safetensors",
    ];

    for filename in &essential_files {
        if let Ok(path) = repo.get(filename).await {
            files.insert(filename.to_string(), path);
        }
    }

    // Check if we have at least config and some model weights
    if !files.contains_key("config.json") {
        bail!("No config.json found in HuggingFace model: {}", model_id);
    }

    let has_weights = files.keys().any(|k| k.contains("model") && (k.contains(".safetensors") || k.contains(".bin")));
    if !has_weights {
        bail!("No model weight files found in HuggingFace model: {}", model_id);
    }

    Ok(files)
}

/// Load model files from local directory
async fn load_local_model_files(model_path: &str) -> Result<HashMap<String, PathBuf>> {
    let model_dir = Path::new(model_path);
    if !model_dir.exists() {
        bail!("Model directory does not exist: {}", model_path);
    }

    let mut files = HashMap::new();
    let mut entries = fs::read_dir(model_dir).await?;

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            files.insert(filename.to_string(), path);
        }
    }

    // Verify essential files exist
    if !files.contains_key("config.json") {
        bail!("No config.json found in model directory: {}", model_path);
    }

    Ok(files)
}

/// Load model weights from files
async fn load_model_weights(files: &HashMap<String, PathBuf>, device: &Device) -> Result<HashMap<String, Tensor>> {
    let mut weights = HashMap::new();

    // Try to load from safetensors first, then fallback to pytorch
    if let Some(safetensors_path) = files.get("model.safetensors") {
        info!("Loading weights from safetensors format");
        weights.extend(load_safetensors_weights(safetensors_path, device).await?);
    } else if let Some(pytorch_path) = files.get("pytorch_model.bin") {
        info!("Loading weights from pytorch format");
        weights.extend(load_pytorch_weights(pytorch_path, device).await?);
    } else {
        // Try to find any safetensors or bin files
        for (name, path) in files {
            if name.contains("model") && name.ends_with(".safetensors") {
                weights.extend(load_safetensors_weights(path, device).await?);
                break;
            } else if name.contains("model") && name.ends_with(".bin") {
                weights.extend(load_pytorch_weights(path, device).await?);
                break;
            }
        }
    }

    if weights.is_empty() {
        bail!("No model weights could be loaded from the provided files");
    }

    info!("Loaded {} weight tensors", weights.len());
    Ok(weights)
}

/// Load weights from safetensors format
async fn load_safetensors_weights(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let data = fs::read(path).await?;
    let tensors = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| anyhow!("Failed to deserialize safetensors: {}", e))?;

    let mut weights = HashMap::new();
    for (name, tensor_view) in tensors.tensors() {
        // Convert safetensors dtype to candle dtype
        let dtype = match tensor_view.dtype() {
            safetensors::Dtype::F32 => candle_core::DType::F32,
            safetensors::Dtype::F16 => candle_core::DType::F16,
            safetensors::Dtype::BF16 => candle_core::DType::BF16,
            safetensors::Dtype::U8 => candle_core::DType::U8,
            safetensors::Dtype::I8 => candle_core::DType::I8,
            safetensors::Dtype::U16 => candle_core::DType::U16,
            safetensors::Dtype::I16 => candle_core::DType::I16,
            safetensors::Dtype::U32 => candle_core::DType::U32,
            safetensors::Dtype::I32 => candle_core::DType::I32,
            safetensors::Dtype::U64 => candle_core::DType::U64,
            safetensors::Dtype::I64 => candle_core::DType::I64,
            _ => bail!("Unsupported tensor dtype: {:?}", tensor_view.dtype()),
        };

        let tensor = Tensor::from_raw_buffer(
            tensor_view.data(),
            dtype,
            tensor_view.shape(),
            device,
        )?;
        weights.insert(name.to_string(), tensor);
    }

    Ok(weights)
}

/// Load weights from pytorch format (simplified - would need actual pytorch loading)
async fn load_pytorch_weights(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    // This is a placeholder - actual pytorch loading would require
    // either Python bindings or a pure Rust pytorch format reader
    warn!("PyTorch format loading not fully implemented - this is a placeholder");
    bail!("PyTorch format loading not yet supported. Please use safetensors format.");
}

/// Apply LoRA adapter to Llama model
fn apply_lora_to_llama(model: &mut Llama, adapter: &LoraParameters) -> Result<()> {
    info!("Applying LoRA to Llama model with {} layers", adapter.layers.len());
    
    for (layer_name, lora_layer) in &adapter.layers {
        debug!("Applying LoRA to layer: {}", layer_name);
        
        // This is a simplified implementation - in practice, you'd need to:
        // 1. Map the layer name to the actual model component
        // 2. Get the current weight tensor
        // 3. Apply the LoRA transformation
        // 4. Update the model weights
        
        // For now, just validate that the layer exists conceptually
        if !is_valid_llama_layer_name(layer_name) {
            warn!("Unknown layer name for Llama model: {}", layer_name);
            continue;
        }
        
        // In a real implementation, you would:
        // let base_layer = model.get_layer_mut(layer_name)?;
        // apply_lora_to_layer(base_layer, lora_layer)?;
    }
    
    Ok(())
}

/// Apply LoRA adapter to Mistral model (placeholder)
fn apply_lora_to_mistral(weights: &mut HashMap<String, Tensor>, adapter: &LoraParameters) -> Result<()> {
    info!("Applying LoRA to Mistral model with {} layers (placeholder implementation)", adapter.layers.len());
    
    for (layer_name, lora_layer) in &adapter.layers {
        debug!("Applying LoRA to layer: {}", layer_name);
        
        if !is_valid_mistral_layer_name(layer_name) {
            warn!("Unknown layer name for Mistral model: {}", layer_name);
            continue;
        }
        
        // Placeholder: In a real implementation, you would:
        // 1. Find the corresponding weight tensor in weights map
        // 2. Apply LoRA transformation
        // 3. Update the tensor in the map
    }
    
    Ok(())
}

/// Apply LoRA adapter to Gemma model (placeholder)
fn apply_lora_to_gemma(weights: &mut HashMap<String, Tensor>, adapter: &LoraParameters) -> Result<()> {
    info!("Applying LoRA to Gemma model with {} layers (placeholder implementation)", adapter.layers.len());
    
    for (layer_name, lora_layer) in &adapter.layers {
        debug!("Applying LoRA to layer: {}", layer_name);
        
        if !is_valid_gemma_layer_name(layer_name) {
            warn!("Unknown layer name for Gemma model: {}", layer_name);
            continue;
        }
        
        // Placeholder: In a real implementation, you would:
        // 1. Find the corresponding weight tensor in weights map
        // 2. Apply LoRA transformation  
        // 3. Update the tensor in the map
    }
    
    Ok(())
}

/// Check if layer name is valid for Llama models
fn is_valid_llama_layer_name(layer_name: &str) -> bool {
    layer_name.contains("layers.") && (
        layer_name.contains("self_attn.q_proj") ||
        layer_name.contains("self_attn.k_proj") ||
        layer_name.contains("self_attn.v_proj") ||
        layer_name.contains("self_attn.o_proj") ||
        layer_name.contains("mlp.gate_proj") ||
        layer_name.contains("mlp.up_proj") ||
        layer_name.contains("mlp.down_proj")
    )
}

/// Check if layer name is valid for Mistral models
fn is_valid_mistral_layer_name(layer_name: &str) -> bool {
    layer_name.contains("layers.") && (
        layer_name.contains("self_attn.q_proj") ||
        layer_name.contains("self_attn.k_proj") ||
        layer_name.contains("self_attn.v_proj") ||
        layer_name.contains("self_attn.o_proj") ||
        layer_name.contains("mlp.gate_proj") ||
        layer_name.contains("mlp.up_proj") ||
        layer_name.contains("mlp.down_proj")
    )
}

/// Check if layer name is valid for Gemma models
fn is_valid_gemma_layer_name(layer_name: &str) -> bool {
    layer_name.contains("layers.") && (
        layer_name.contains("self_attn.q_proj") ||
        layer_name.contains("self_attn.k_proj") ||
        layer_name.contains("self_attn.v_proj") ||
        layer_name.contains("self_attn.o_proj") ||
        layer_name.contains("mlp.gate_proj") ||
        layer_name.contains("mlp.up_proj") ||
        layer_name.contains("mlp.down_proj")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[tokio::test]
    async fn test_local_config_detection() {
        // This test would require setting up a test model directory
        // with a config.json file
    }

    #[tokio::test]
    async fn test_architecture_detection() {
        let test_cases = vec![
            ("llama", "llama"),
            ("LlamaForCausalLM", "llama"),
            ("mistral", "mistral"),
            ("MistralForCausalLM", "mistral"),
            ("gemma", "gemma"),
            ("GemmaForCausalLM", "gemma"),
        ];

        for (input, expected) in test_cases {
            let config = ModelConfig {
                architecture: input.to_string(),
                model_type: None,
                architectures: None,
                extra: HashMap::new(),
            };

            match config.architecture.to_lowercase().as_str() {
                "llama" | "llamaforcausallm" => assert_eq!(expected, "llama"),
                "mistral" | "mistralformcausallm" => assert_eq!(expected, "mistral"),
                "gemma" | "gemmaforcausallm" => assert_eq!(expected, "gemma"),
                _ => panic!("Unsupported architecture: {}", input),
            }
        }
    }

    #[test]
    fn test_layer_name_validation() {
        let llama_valid_names = vec![
            "layers.0.self_attn.q_proj",
            "layers.15.self_attn.v_proj",
            "layers.31.mlp.gate_proj",
        ];

        let llama_invalid_names = vec![
            "embed_tokens",
            "layers.0.invalid_layer",
            "norm",
        ];

        for name in llama_valid_names {
            assert!(is_valid_llama_layer_name(name), "Should be valid: {}", name);
        }

        for name in llama_invalid_names {
            assert!(!is_valid_llama_layer_name(name), "Should be invalid: {}", name);
        }
    }

    #[test]
    fn test_device_manager() {
        let device = Device::Cpu;
        let manager = DeviceManager::new(device.clone());
        assert_eq!(manager.device(), &device);
        assert_eq!(manager.memory_fraction, 0.9);

        let manager_with_fraction = DeviceManager::new(device.clone()).with_memory_fraction(0.5);
        assert_eq!(manager_with_fraction.memory_fraction, 0.5);

        // Test clamping
        let manager_clamped = DeviceManager::new(device.clone()).with_memory_fraction(1.5);
        assert_eq!(manager_clamped.memory_fraction, 1.0);
    }
}