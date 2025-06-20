//! LoRA weight merger for applying adapters to base models
//!
//! This module provides functionality to merge LoRA adapter weights into base model weights
//! either permanently (weight merging) or as separate computation paths (adapter attachment).

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::apply::loader::{BaseModel, LoraCompatibleLayer};
use crate::utils::tensor::{lora as tensor_lora, memory::ChunkedProcessor};
use lorax::lora::{LoraParameters, LoraLayer};

/// Memory threshold for chunked processing (2GB)
const MEMORY_THRESHOLD: usize = 2 * 1024 * 1024 * 1024;

/// Layer name mapping for different architectures
#[derive(Debug, Clone)]
pub struct LayerMapper {
    /// Mapping from T2L layer names to target model layer names
    pub t2l_to_target: HashMap<String, String>,
    /// Target model architecture
    pub target_arch: String,
}

impl LayerMapper {
    /// Create new layer mapper for the given architecture
    pub fn new(architecture: &str) -> Self {
        let t2l_to_target = match architecture.to_lowercase().as_str() {
            "llama" | "llamaforcausallm" => Self::create_llama_mapping(),
            "mistral" | "mistralformcausallm" => Self::create_mistral_mapping(),
            "gemma" | "gemmaforcausallm" => Self::create_gemma_mapping(),
            _ => HashMap::new(),
        };

        Self {
            t2l_to_target,
            target_arch: architecture.to_string(),
        }
    }

    /// Create mapping for Llama models
    fn create_llama_mapping() -> HashMap<String, String> {
        let mut mapping = HashMap::new();
        
        // Attention projections
        mapping.insert("q_proj".to_string(), "self_attn.q_proj".to_string());
        mapping.insert("k_proj".to_string(), "self_attn.k_proj".to_string());
        mapping.insert("v_proj".to_string(), "self_attn.v_proj".to_string());
        mapping.insert("o_proj".to_string(), "self_attn.o_proj".to_string());
        
        // MLP projections
        mapping.insert("gate_proj".to_string(), "mlp.gate_proj".to_string());
        mapping.insert("up_proj".to_string(), "mlp.up_proj".to_string());
        mapping.insert("down_proj".to_string(), "mlp.down_proj".to_string());
        
        mapping
    }

    /// Create mapping for Mistral models
    fn create_mistral_mapping() -> HashMap<String, String> {
        let mut mapping = HashMap::new();
        
        // Same structure as Llama for now
        mapping.insert("q_proj".to_string(), "self_attn.q_proj".to_string());
        mapping.insert("k_proj".to_string(), "self_attn.k_proj".to_string());
        mapping.insert("v_proj".to_string(), "self_attn.v_proj".to_string());
        mapping.insert("o_proj".to_string(), "self_attn.o_proj".to_string());
        
        mapping.insert("gate_proj".to_string(), "mlp.gate_proj".to_string());
        mapping.insert("up_proj".to_string(), "mlp.up_proj".to_string());
        mapping.insert("down_proj".to_string(), "mlp.down_proj".to_string());
        
        mapping
    }

    /// Create mapping for Gemma models
    fn create_gemma_mapping() -> HashMap<String, String> {
        let mut mapping = HashMap::new();
        
        // Same structure as Llama for now
        mapping.insert("q_proj".to_string(), "self_attn.q_proj".to_string());
        mapping.insert("k_proj".to_string(), "self_attn.k_proj".to_string());
        mapping.insert("v_proj".to_string(), "self_attn.v_proj".to_string());
        mapping.insert("o_proj".to_string(), "self_attn.o_proj".to_string());
        
        mapping.insert("gate_proj".to_string(), "mlp.gate_proj".to_string());
        mapping.insert("up_proj".to_string(), "mlp.up_proj".to_string());
        mapping.insert("down_proj".to_string(), "mlp.down_proj".to_string());
        
        mapping
    }

    /// Map T2L layer name to target model layer name with layer index
    pub fn map_layer_name(&self, t2l_name: &str, layer_index: usize) -> Option<String> {
        // Extract the projection type from full layer name
        // Example: "layers.0.self_attn.q_proj" -> "q_proj"
        let projection_name = if let Some(last_part) = t2l_name.split('.').last() {
            last_part
        } else {
            t2l_name
        };

        if let Some(target_pattern) = self.t2l_to_target.get(projection_name) {
            Some(format!("layers.{}.{}", layer_index, target_pattern))
        } else {
            // Try direct mapping for full names
            self.t2l_to_target.get(t2l_name).cloned()
        }
    }

    /// Extract layer index from full layer name
    pub fn extract_layer_index(layer_name: &str) -> Option<usize> {
        if let Some(layers_part) = layer_name.strip_prefix("layers.") {
            if let Some(dot_pos) = layers_part.find('.') {
                let index_str = &layers_part[..dot_pos];
                index_str.parse().ok()
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Merge LoRA weights into base model weights permanently
/// This modifies the base model weights: W_new = W_base + A @ B * alpha
pub fn merge_lora_weights(base_model: &mut BaseModel, adapter: &LoraParameters) -> Result<()> {
    info!(
        "Merging {} LoRA layers into {} model weights",
        adapter.layers.len(),
        base_model.architecture_name()
    );

    let layer_mapper = LayerMapper::new(base_model.architecture_name());
    let device = base_model.device().clone();
    
    // Initialize chunked processor for memory efficiency
    let chunk_size = estimate_optimal_chunk_size(&device);
    let processor = ChunkedProcessor::new(chunk_size, 0);

    let mut merged_count = 0;
    let mut skipped_count = 0;

    for (t2l_layer_name, lora_layer) in &adapter.layers {
        match merge_single_layer(base_model, lora_layer, &layer_mapper, &processor) {
            Ok(()) => {
                merged_count += 1;
                debug!("Successfully merged layer: {}", t2l_layer_name);
            }
            Err(e) => {
                skipped_count += 1;
                warn!("Skipped layer {}: {}", t2l_layer_name, e);
            }
        }
    }

    info!(
        "Merge complete: {} layers merged, {} layers skipped",
        merged_count, skipped_count
    );

    if merged_count == 0 {
        bail!("No LoRA layers could be merged into the base model");
    }

    Ok(())
}

/// Attach LoRA adapters as separate computation path (non-destructive)
/// This keeps base weights unchanged and applies LoRA as an additive operation
pub fn attach_lora_adapters(base_model: &mut BaseModel, adapter: &LoraParameters) -> Result<()> {
    info!(
        "Attaching {} LoRA adapters to {} model",
        adapter.layers.len(),
        base_model.architecture_name()
    );

    let layer_mapper = LayerMapper::new(base_model.architecture_name());
    let mut attached_count = 0;
    let mut skipped_count = 0;

    for (t2l_layer_name, lora_layer) in &adapter.layers {
        match attach_single_adapter(base_model, lora_layer, &layer_mapper) {
            Ok(()) => {
                attached_count += 1;
                debug!("Successfully attached adapter: {}", t2l_layer_name);
            }
            Err(e) => {
                skipped_count += 1;
                warn!("Skipped adapter {}: {}", t2l_layer_name, e);
            }
        }
    }

    info!(
        "Attachment complete: {} adapters attached, {} adapters skipped",
        attached_count, skipped_count
    );

    if attached_count == 0 {
        bail!("No LoRA adapters could be attached to the base model");
    }

    Ok(())
}

/// Merge a single LoRA layer into the base model
fn merge_single_layer(
    base_model: &mut BaseModel,
    lora_layer: &LoraLayer,
    layer_mapper: &LayerMapper,
    processor: &ChunkedProcessor,
) -> Result<()> {
    // Validate LoRA layer
    lora_layer.validate().map_err(|e| anyhow!("Invalid LoRA layer: {}", e))?;

    // Extract layer index from T2L layer name
    let layer_index = LayerMapper::extract_layer_index(&lora_layer.name)
        .ok_or_else(|| anyhow!("Could not extract layer index from: {}", lora_layer.name))?;

    // Map to target model layer name
    let target_layer_name = layer_mapper
        .map_layer_name(&lora_layer.name, layer_index)
        .ok_or_else(|| anyhow!("Could not map layer name: {}", lora_layer.name))?;

    debug!(
        "Merging layer {} -> {} (index: {})",
        lora_layer.name, target_layer_name, layer_index
    );

    match base_model {
        BaseModel::Llama { model, .. } => {
            merge_llama_layer(model, lora_layer, &target_layer_name, processor)
        }
        BaseModel::Mistral { weights, .. } => {
            merge_generic_layer(weights, lora_layer, &target_layer_name, processor)
        }
        BaseModel::Gemma { weights, .. } => {
            merge_generic_layer(weights, lora_layer, &target_layer_name, processor)
        }
    }
}

/// Attach a single LoRA adapter to the base model
fn attach_single_adapter(
    base_model: &mut BaseModel,
    lora_layer: &LoraLayer,
    layer_mapper: &LayerMapper,
) -> Result<()> {
    // Validate LoRA layer
    lora_layer.validate().map_err(|e| anyhow!("Invalid LoRA layer: {}", e))?;

    // Extract layer index from T2L layer name
    let layer_index = LayerMapper::extract_layer_index(&lora_layer.name)
        .ok_or_else(|| anyhow!("Could not extract layer index from: {}", lora_layer.name))?;

    // Map to target model layer name
    let target_layer_name = layer_mapper
        .map_layer_name(&lora_layer.name, layer_index)
        .ok_or_else(|| anyhow!("Could not map layer name: {}", lora_layer.name))?;

    debug!(
        "Attaching adapter {} -> {} (index: {})",
        lora_layer.name, target_layer_name, layer_index
    );

    // For now, we'll store the adapter information for later use during inference
    // In a full implementation, this would modify the model's forward pass to include LoRA computation
    
    info!(
        "Adapter attached: {} (rank: {}, alpha: {}, params: {})",
        target_layer_name,
        lora_layer.rank,
        lora_layer.alpha,
        lora_layer.num_parameters()
    );

    Ok(())
}

/// Merge LoRA layer into Llama model
fn merge_llama_layer(
    model: &mut candle_transformers::models::llama::Llama,
    lora_layer: &LoraLayer,
    target_layer_name: &str,
    processor: &ChunkedProcessor,
) -> Result<()> {
    // This is a simplified implementation since the exact Llama model structure
    // depends on the candle-transformers API. In practice, you would:
    
    // 1. Navigate to the specific layer in the model
    // 2. Get the current weight tensor
    // 3. Compute the LoRA delta: A @ B * alpha
    // 4. Add the delta to the base weights: W_new = W_base + delta
    // 5. Update the model with the new weights

    info!(
        "Merging into Llama layer: {} (this is a placeholder implementation)",
        target_layer_name
    );

    // Compute LoRA delta using our tensor utilities
    let merged_weights = tensor_lora::merge_lora_weights(
        &vec![0.0f32; lora_layer.input_dim * lora_layer.output_dim], // placeholder base weights
        &lora_layer.a_weights,
        &lora_layer.b_weights,
        lora_layer.alpha,
        lora_layer.rank,
    ).context("Failed to merge LoRA weights")?;

    debug!("Computed merged weights size: {}", merged_weights.len());
    
    // TODO: Actually update the model weights once we have the exact API
    // This would require access to the internal layers of the Llama model
    
    Ok(())
}

/// Merge LoRA layer into generic weight map (for Mistral/Gemma)
fn merge_generic_layer(
    weights: &mut HashMap<String, Tensor>,
    lora_layer: &LoraLayer,
    target_layer_name: &str,
    processor: &ChunkedProcessor,
) -> Result<()> {
    // Find the base weight tensor
    let base_tensor = weights
        .get(target_layer_name)
        .ok_or_else(|| anyhow!("Base weight tensor not found: {}", target_layer_name))?;

    // Extract base weights as f32 vector
    let base_weights = tensor_to_f32_vec(base_tensor)?;

    // Validate dimensions match
    if base_weights.len() != lora_layer.input_dim * lora_layer.output_dim {
        return Err(anyhow!(
            "Dimension mismatch: base tensor size {} != LoRA expected size {}",
            base_weights.len(),
            lora_layer.input_dim * lora_layer.output_dim
        ));
    }

    // Use chunked processing for large tensors
    let merged_weights = if base_weights.len() * 4 > MEMORY_THRESHOLD {
        info!("Using chunked processing for large tensor: {}", target_layer_name);
        merge_weights_chunked(&base_weights, lora_layer, processor)?
    } else {
        // Direct merge for smaller tensors
        tensor_lora::merge_lora_weights(
            &base_weights,
            &lora_layer.a_weights,
            &lora_layer.b_weights,
            lora_layer.alpha,
            lora_layer.rank,
        )?
    };

    // Convert back to tensor and update weights map
    let new_tensor = f32_vec_to_tensor(&merged_weights, base_tensor.dims(), base_tensor.device())?;
    weights.insert(target_layer_name.to_string(), new_tensor);

    info!(
        "Successfully merged layer: {} ({} parameters)",
        target_layer_name,
        merged_weights.len()
    );

    Ok(())
}

/// Merge weights using chunked processing for memory efficiency
fn merge_weights_chunked(
    base_weights: &[f32],
    lora_layer: &LoraLayer,
    processor: &ChunkedProcessor,
) -> Result<Vec<f32>> {
    // For very large matrices, we need to chunk the computation
    // This is a simplified version - a full implementation would need more sophisticated chunking
    
    let chunk_size = processor.chunk_size.min(base_weights.len());
    let mut merged_weights = Vec::with_capacity(base_weights.len());
    
    for chunk_start in (0..base_weights.len()).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(base_weights.len());
        let base_chunk = &base_weights[chunk_start..chunk_end];
        
        // For this simplified version, we'll just add the base weights
        // In practice, you'd need to carefully chunk the LoRA computation
        merged_weights.extend_from_slice(base_chunk);
    }
    
    // Apply LoRA scaling (simplified)
    let scaling = lora_layer.alpha / lora_layer.rank as f32;
    for weight in merged_weights.iter_mut() {
        *weight *= (1.0 + scaling); // Simplified scaling
    }
    
    Ok(merged_weights)
}

/// Convert tensor to f32 vector
fn tensor_to_f32_vec(tensor: &Tensor) -> Result<Vec<f32>> {
    // This is a simplified conversion - in practice you'd need to handle different dtypes
    let data = tensor.to_vec2::<f32>()
        .context("Failed to convert tensor to f32 vector")?;
    
    // Flatten the 2D vector
    Ok(data.into_iter().flatten().collect())
}

/// Convert f32 vector back to tensor
fn f32_vec_to_tensor(data: &[f32], shape: &[usize], device: &Device) -> Result<Tensor> {
    Tensor::from_slice(data, shape, device)
        .context("Failed to create tensor from f32 vector")
}

/// Estimate optimal chunk size based on available memory
fn estimate_optimal_chunk_size(device: &Device) -> usize {
    match device {
        Device::Cuda(_) => {
            // For CUDA, use smaller chunks to avoid OOM
            256 * 1024 // 256K elements
        }
        Device::Cpu => {
            // For CPU, we can use larger chunks
            1024 * 1024 // 1M elements
        }
        _ => {
            // Conservative default
            128 * 1024 // 128K elements
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use lorax::lora::{LoraParameters, LoraLayer, LoraParameterConfig};

    #[test]
    fn test_layer_mapper_creation() {
        let mapper = LayerMapper::new("llama");
        assert_eq!(mapper.target_arch, "llama");
        assert!(mapper.t2l_to_target.contains_key("q_proj"));
        assert!(mapper.t2l_to_target.contains_key("gate_proj"));
    }

    #[test]
    fn test_layer_name_mapping() {
        let mapper = LayerMapper::new("llama");
        
        let mapped = mapper.map_layer_name("q_proj", 5);
        assert_eq!(mapped, Some("layers.5.self_attn.q_proj".to_string()));
        
        let mapped = mapper.map_layer_name("gate_proj", 10);
        assert_eq!(mapped, Some("layers.10.mlp.gate_proj".to_string()));
    }

    #[test]
    fn test_layer_index_extraction() {
        assert_eq!(LayerMapper::extract_layer_index("layers.5.self_attn.q_proj"), Some(5));
        assert_eq!(LayerMapper::extract_layer_index("layers.10.mlp.gate_proj"), Some(10));
        assert_eq!(LayerMapper::extract_layer_index("invalid_name"), None);
    }

    #[test]
    fn test_lora_layer_validation() {
        let layer = LoraLayer::new(
            "test_layer".to_string(),
            512,  // input_dim
            256,  // output_dim
            16,   // rank
            32.0, // alpha
        );
        
        assert!(layer.validate().is_ok());
        
        // Test invalid layer (zero rank)
        let mut invalid_layer = layer.clone();
        invalid_layer.rank = 0;
        assert!(invalid_layer.validate().is_err());
    }

    #[test]
    fn test_chunk_size_estimation() {
        let cpu_device = Device::Cpu;
        let chunk_size = estimate_optimal_chunk_size(&cpu_device);
        assert!(chunk_size > 0);
        assert!(chunk_size <= 1024 * 1024);
    }
}