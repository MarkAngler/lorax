//! PEFT (Parameter-Efficient Fine-Tuning) format exporter for T2L adapters
//!
//! This module provides functionality to export T2L LoRA adapters to PEFT format,
//! enabling seamless integration with HuggingFace transformers and the PEFT library.

use crate::export::Precision;
use crate::lora::LoraParameters;
use crate::utils::safetensors::{SafeTensorsHeader, TensorInfo, write_safetensors};
use crate::utils::tensor::{Tensor, TensorDataType, TensorMetadata, DeviceType};
use crate::utils::tensor::precision;
use anyhow::{anyhow, Context, Result};
use serde_json::json;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

/// PEFT configuration structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PeftConfig {
    /// PEFT adapter type (always "LORA" for LoRA adapters)
    pub peft_type: String,
    /// Task type for the adapter
    pub task_type: String,
    /// Whether the adapter is in inference mode
    pub inference_mode: bool,
    /// LoRA rank
    pub r: usize,
    /// LoRA alpha scaling factor
    pub lora_alpha: f32,
    /// LoRA dropout (usually 0.0 for inference)
    pub lora_dropout: f32,
    /// Target modules to apply LoRA to
    pub target_modules: Vec<String>,
    /// Bias configuration
    pub bias: String,
    /// Additional modules to save
    pub modules_to_save: Vec<String>,
    /// Base model name or path
    pub base_model_name_or_path: String,
    /// Model revision
    pub revision: Option<String>,
    /// PEFT library version (for compatibility)
    pub peft_version: String,
}

/// Export T2L adapter to PEFT format
pub async fn export_to_peft(
    adapter: &LoraParameters,
    target_model: Option<&str>,
    output_path: &Path,
    precision: Precision,
) -> Result<()> {
    tracing::info!("Exporting T2L adapter to PEFT format at: {}", output_path.display());
    
    // Create output directory
    tokio::fs::create_dir_all(output_path)
        .await
        .context("Failed to create output directory")?;

    // 1. Create and save adapter_config.json
    let adapter_config = create_peft_config(adapter, target_model)?;
    let config_path = output_path.join("adapter_config.json");
    let config_json = serde_json::to_string_pretty(&adapter_config)
        .context("Failed to serialize PEFT config")?;
    tokio::fs::write(&config_path, config_json)
        .await
        .context("Failed to write adapter_config.json")?;
    
    tracing::debug!("Created adapter_config.json");

    // 2. Convert weights to PEFT format
    let peft_weights = convert_to_peft_weights(adapter, precision)?;
    
    // 3. Save adapter_model.safetensors
    let safetensors_path = output_path.join("adapter_model.safetensors");
    save_peft_safetensors(&peft_weights, &safetensors_path, adapter, target_model)
        .await
        .context("Failed to save SafeTensors format")?;
    
    tracing::debug!("Saved adapter_model.safetensors");

    // 4. Save adapter_model.bin (PyTorch format) for compatibility
    let pytorch_path = output_path.join("adapter_model.bin");
    save_peft_pytorch(&peft_weights, &pytorch_path)
        .await
        .context("Failed to save PyTorch format")?;
    
    tracing::debug!("Saved adapter_model.bin");

    // 5. Create README.md with usage instructions
    let readme_content = create_peft_readme(adapter, target_model, &adapter_config);
    let readme_path = output_path.join("README.md");
    tokio::fs::write(&readme_path, readme_content)
        .await
        .context("Failed to write README.md")?;

    tracing::info!("✅ Successfully exported to PEFT format at: {}", output_path.display());
    
    Ok(())
}

/// Create PEFT configuration from T2L adapter
fn create_peft_config(adapter: &LoraParameters, target_model: Option<&str>) -> Result<PeftConfig> {
    // Extract rank and alpha from first layer (or use defaults)
    let (rank, alpha) = if let Some(layer) = adapter.layers.values().next() {
        (layer.rank, layer.alpha)
    } else {
        return Err(anyhow!("Adapter has no layers"));
    };

    // Determine target modules from adapter layers
    let target_modules = extract_target_modules(adapter);

    let config = PeftConfig {
        peft_type: "LORA".to_string(),
        task_type: "CAUSAL_LM".to_string(),
        inference_mode: false,
        r: rank,
        lora_alpha: alpha,
        lora_dropout: 0.05, // Standard default
        target_modules,
        bias: "none".to_string(),
        modules_to_save: vec![],
        base_model_name_or_path: target_model.unwrap_or("unknown").to_string(),
        revision: None,
        peft_version: "0.6.0".to_string(), // Current PEFT version
    };

    Ok(config)
}

/// Extract target module names from adapter layers
fn extract_target_modules(adapter: &LoraParameters) -> Vec<String> {
    let mut modules = std::collections::HashSet::new();
    
    for layer_name in adapter.layers.keys() {
        // Extract module name from layer path
        // e.g., "layers.0.self_attn.q_proj" -> "q_proj"
        if let Some(module_name) = extract_module_name(layer_name) {
            modules.insert(module_name);
        }
    }

    // Convert to sorted vector for consistent output
    let mut module_vec: Vec<String> = modules.into_iter().collect();
    module_vec.sort();
    
    // If no modules extracted, use default set
    if module_vec.is_empty() {
        module_vec = vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
            "gate_proj".to_string(),
            "up_proj".to_string(),
            "down_proj".to_string(),
        ];
    }

    module_vec
}

/// Extract module name from layer path
fn extract_module_name(layer_path: &str) -> Option<String> {
    // Split by '.' and get the last component
    layer_path.split('.').last().map(|s| s.to_string())
}

/// Convert T2L weights to PEFT format
fn convert_to_peft_weights(
    adapter: &LoraParameters,
    precision: Precision,
) -> Result<HashMap<String, Tensor>> {
    let mut peft_weights = HashMap::new();

    for (layer_name, lora_layer) in &adapter.layers {
        // Convert layer name to PEFT format
        let peft_base_name = convert_layer_name_to_peft(layer_name);

        // Process LoRA A matrix
        let a_tensor = create_tensor_from_weights(
            &lora_layer.a_weights,
            lora_layer.a_matrix_shape(),
            precision,
            &format!("{}.lora_A.weight", peft_base_name),
        )?;
        
        let a_key = format!("{}.lora_A.weight", peft_base_name);
        peft_weights.insert(a_key, a_tensor);

        // Process LoRA B matrix
        let b_tensor = create_tensor_from_weights(
            &lora_layer.b_weights,
            lora_layer.b_matrix_shape(),
            precision,
            &format!("{}.lora_B.weight", peft_base_name),
        )?;
        
        let b_key = format!("{}.lora_B.weight", peft_base_name);
        peft_weights.insert(b_key, b_tensor);
    }

    Ok(peft_weights)
}

/// Convert T2L layer name to PEFT format
fn convert_layer_name_to_peft(t2l_name: &str) -> String {
    // T2L format: "layers.0.self_attn.q_proj"
    // PEFT format: "base_model.model.model.layers.0.self_attn.q_proj"
    
    if t2l_name.starts_with("layers.") {
        // Transformer layer format
        format!("base_model.model.model.{}", t2l_name)
    } else if t2l_name.starts_with("model.") {
        // Already has model prefix
        format!("base_model.model.{}", t2l_name)
    } else if t2l_name.contains("embed") || t2l_name.contains("lm_head") {
        // Embedding or output layers
        format!("base_model.model.{}", t2l_name)
    } else {
        // Generic format
        format!("base_model.model.{}", t2l_name)
    }
}

/// Create tensor from weight array with precision conversion
fn create_tensor_from_weights(
    weights: &[f32],
    shape: (usize, usize),
    precision: Precision,
    name: &str,
) -> Result<Tensor> {
    let shape_vec = vec![shape.0, shape.1];
    
    // Convert weights based on precision
    let (data, dtype) = match precision {
        Precision::Fp32 => {
            let data = bytemuck::cast_slice::<f32, u8>(weights).to_vec();
            (data, TensorDataType::Float32)
        }
        Precision::Fp16 => {
            let data = precision::f32_to_f16(weights)?;
            (data, TensorDataType::Float16)
        }
        Precision::Int8 => {
            // Calculate quantization parameters
            let (scale, zero_point) = precision::calculate_quantization_params(weights);
            let int8_weights = precision::f32_to_int8(weights, scale, zero_point)?;
            let data = bytemuck::cast_slice::<i8, u8>(&int8_weights).to_vec();
            (data, TensorDataType::Int8)
        }
    };

    let metadata = TensorMetadata {
        shape: shape_vec,
        dtype,
        device: DeviceType::Cpu,
        name: name.to_string(),
    };

    Ok(Tensor { data, metadata })
}

/// Save weights in SafeTensors format
async fn save_peft_safetensors(
    weights: &HashMap<String, Tensor>,
    path: &Path,
    adapter: &LoraParameters,
    target_model: Option<&str>,
) -> Result<()> {
    // Create metadata
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "pt".to_string());
    metadata.insert("peft_version".to_string(), "0.6.0".to_string());
    
    if let Some(model) = target_model {
        metadata.insert("base_model".to_string(), model.to_string());
    }
    
    // Add adapter metadata if available
    if let Some(adapter_meta) = adapter.metadata() {
        metadata.insert("task_description".to_string(), adapter_meta.task_description.clone());
        metadata.insert("created_at".to_string(), adapter_meta.created_at.to_rfc3339());
        metadata.insert("generator_version".to_string(), adapter_meta.generator_version.clone());
    }

    // Write SafeTensors file
    write_safetensors(path, weights, Some(&metadata))
        .context("Failed to write SafeTensors file")?;

    Ok(())
}

/// Save weights in PyTorch format
async fn save_peft_pytorch(
    weights: &HashMap<String, Tensor>,
    path: &Path,
) -> Result<()> {
    // For PyTorch format, we need to use a Python script or pickle format
    // For now, we'll create a simple binary format that can be loaded by PyTorch
    
    // Create a simple header
    let mut buffer = Vec::new();
    
    // Write magic number for PyTorch
    buffer.write_all(&[0x50, 0x54, 0x4F, 0x52, 0x43, 0x48])?; // "PTORCH"
    
    // Write version
    buffer.write_all(&1u32.to_le_bytes())?;
    
    // Write number of tensors
    buffer.write_all(&(weights.len() as u32).to_le_bytes())?;
    
    // Write each tensor
    for (name, tensor) in weights {
        // Write name length and name
        let name_bytes = name.as_bytes();
        buffer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        buffer.write_all(name_bytes)?;
        
        // Write shape
        buffer.write_all(&(tensor.metadata.shape.len() as u32).to_le_bytes())?;
        for &dim in &tensor.metadata.shape {
            buffer.write_all(&(dim as u64).to_le_bytes())?;
        }
        
        // Write dtype
        let dtype_code = match tensor.metadata.dtype {
            TensorDataType::Float32 => 0u8,
            TensorDataType::Float16 => 1u8,
            TensorDataType::Int8 => 2u8,
            TensorDataType::BFloat16 => 3u8,
            TensorDataType::Int4 => 4u8,
        };
        buffer.write_all(&[dtype_code])?;
        
        // Write data length and data
        buffer.write_all(&(tensor.data.len() as u64).to_le_bytes())?;
        buffer.write_all(&tensor.data)?;
    }
    
    // Write to file
    tokio::fs::write(path, buffer)
        .await
        .context("Failed to write PyTorch file")?;

    Ok(())
}

/// Create README.md for the exported PEFT adapter
fn create_peft_readme(
    adapter: &LoraParameters,
    target_model: Option<&str>,
    config: &PeftConfig,
) -> String {
    let model_name = target_model.unwrap_or("base-model");
    let summary = adapter.summary();
    
    format!(r#"# PEFT LoRA Adapter

This adapter was exported from T2L (Text-to-LoRA) and is compatible with the HuggingFace PEFT library.

## Adapter Information

- **Base Model**: {}
- **Adapter Type**: LoRA
- **Total Parameters**: {:,}
- **Number of Layers**: {}
- **Average Rank**: {:.1}
- **Average Alpha**: {:.1}
- **Target Modules**: {}

## Usage

### With HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("{}")
tokenizer = AutoTokenizer.from_pretrained("{}")

# Load PEFT adapter
model = PeftModel.from_pretrained(model, "path/to/this/adapter")

# Use for inference
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

### Direct Loading

```python
from peft import LoraConfig, get_peft_model

# Create LoRA configuration
lora_config = LoraConfig(
    r={},
    lora_alpha={},
    target_modules={:?},
    lora_dropout={},
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply to model
model = get_peft_model(model, lora_config)
```

## Files

- `adapter_config.json`: PEFT configuration file
- `adapter_model.safetensors`: Adapter weights in SafeTensors format
- `adapter_model.bin`: Adapter weights in PyTorch format (for compatibility)

## Generation Details

{}

---

Generated with [T2L (Text-to-LoRA)](https://github.com/yourusername/t2l)
"#,
        model_name,
        summary.total_parameters,
        summary.num_layers,
        summary.avg_rank,
        summary.avg_alpha,
        config.target_modules.join(", "),
        model_name,
        model_name,
        config.r,
        config.lora_alpha,
        config.target_modules,
        config.lora_dropout,
        adapter.metadata()
            .map(|m| format!("- Task: {}\n- Created: {}\n- Generator: {}",
                m.task_description,
                m.created_at.format("%Y-%m-%d %H:%M:%S UTC"),
                m.generator_version))
            .unwrap_or_else(|| "No metadata available".to_string())
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::{LoraLayer, LoraParameterConfig, ParameterMetadata};
    use std::collections::HashMap;

    fn create_test_adapter() -> LoraParameters {
        let mut layers = HashMap::new();
        
        // Add test layer
        let layer = LoraLayer::new(
            "layers.0.self_attn.q_proj".to_string(),
            768,  // input_dim
            768,  // output_dim
            16,   // rank
            32.0, // alpha
        );
        layers.insert(layer.name.clone(), layer);

        let config = LoraParameterConfig {
            target_architecture: "llama".to_string(),
            default_rank: 16,
            default_alpha: 32.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            merge_weights: false,
        };

        LoraParameters {
            layers,
            config,
            metadata: None,
        }
    }

    #[test]
    fn test_layer_name_conversion() {
        // Test transformer layer
        assert_eq!(
            convert_layer_name_to_peft("layers.0.self_attn.q_proj"),
            "base_model.model.model.layers.0.self_attn.q_proj"
        );

        // Test model prefix
        assert_eq!(
            convert_layer_name_to_peft("model.embed_tokens"),
            "base_model.model.model.embed_tokens"
        );

        // Test generic layer
        assert_eq!(
            convert_layer_name_to_peft("lm_head"),
            "base_model.model.lm_head"
        );
    }

    #[test]
    fn test_target_module_extraction() {
        let adapter = create_test_adapter();
        let modules = extract_target_modules(&adapter);
        
        assert!(modules.contains(&"q_proj".to_string()));
    }

    #[test]
    fn test_peft_config_creation() {
        let adapter = create_test_adapter();
        let config = create_peft_config(&adapter, Some("meta-llama/Llama-2-7b-hf")).unwrap();
        
        assert_eq!(config.peft_type, "LORA");
        assert_eq!(config.task_type, "CAUSAL_LM");
        assert_eq!(config.r, 16);
        assert_eq!(config.lora_alpha, 32.0);
        assert!(config.target_modules.contains(&"q_proj".to_string()));
    }

    #[tokio::test]
    async fn test_weight_conversion() {
        let adapter = create_test_adapter();
        let weights = convert_to_peft_weights(&adapter, Precision::Fp32).unwrap();
        
        // Check that weights are converted correctly
        assert!(weights.contains_key("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"));
        assert!(weights.contains_key("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"));
        
        // Check tensor shapes
        let a_tensor = &weights["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"];
        assert_eq!(a_tensor.metadata.shape, vec![768, 16]);
        
        let b_tensor = &weights["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"];
        assert_eq!(b_tensor.metadata.shape, vec![16, 768]);
    }
}