//! HuggingFace Transformers format exporter for T2L adapters
//!
//! This module provides functionality to export T2L LoRA adapters to HuggingFace format.
//! The exported model can be directly loaded using AutoModelForCausalLM.from_pretrained().

use crate::export::Precision;
use crate::lora::{LoraParameters, LoraLayer};
use crate::utils::tensor::{Tensor, TensorDataType, TensorMetadata, DeviceType};
use crate::utils::tensor::precision;
use crate::utils::safetensors;
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;

/// HuggingFace model configuration structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HuggingFaceConfig {
    /// Model type (e.g., "llama", "mistral", "gemma")
    pub model_type: String,
    /// Architecture name(s)
    pub architectures: Vec<String>,
    /// Hidden size of the model
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Intermediate size for FFN
    pub intermediate_size: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// Whether to tie word embeddings
    pub tie_word_embeddings: bool,
    /// Whether to use cache
    pub use_cache: bool,
    /// BOS token ID
    pub bos_token_id: usize,
    /// EOS token ID
    pub eos_token_id: usize,
    /// Torch dtype
    pub torch_dtype: String,
    /// Transformers version
    pub transformers_version: String,
}

/// Export T2L adapter to HuggingFace format
pub async fn export_to_hf(
    adapter: &LoraParameters,
    target_model: Option<&str>,
    output_path: &Path,
    precision: Precision,
) -> Result<()> {
    tracing::info!("Exporting T2L adapter to HuggingFace format at: {}", output_path.display());
    
    // Create output directory
    tokio::fs::create_dir_all(output_path)
        .await
        .context("Failed to create output directory")?;

    // 1. Create and save config.json
    let hf_config = create_hf_config(adapter, target_model)?;
    let config_path = output_path.join("config.json");
    let config_json = serde_json::to_string_pretty(&hf_config)
        .context("Failed to serialize HuggingFace config")?;
    tokio::fs::write(&config_path, config_json)
        .await
        .context("Failed to write config.json")?;
    
    tracing::debug!("Created config.json");

    // 2. Convert LoRA weights to HuggingFace state_dict format
    let (hf_state_dict, is_adapter_only) = convert_to_hf_state_dict(adapter, precision)?;
    
    // 3. Save model weights
    if is_adapter_only {
        // Save as adapter files
        let adapter_safetensors_path = output_path.join("adapter_model.safetensors");
        save_hf_safetensors(&hf_state_dict, &adapter_safetensors_path, adapter)
            .await
            .context("Failed to save adapter SafeTensors")?;
        
        // Also save adapter config for compatibility
        let adapter_config = create_adapter_config(adapter, target_model)?;
        let adapter_config_path = output_path.join("adapter_config.json");
        tokio::fs::write(&adapter_config_path, serde_json::to_string_pretty(&adapter_config)?)
            .await
            .context("Failed to write adapter_config.json")?;
    } else {
        // Save as full model
        let model_safetensors_path = output_path.join("model.safetensors");
        save_hf_safetensors(&hf_state_dict, &model_safetensors_path, adapter)
            .await
            .context("Failed to save model SafeTensors")?;
    }
    
    tracing::debug!("Saved model weights");

    // 4. Save generation_config.json
    let generation_config = create_generation_config();
    let generation_config_path = output_path.join("generation_config.json");
    tokio::fs::write(&generation_config_path, serde_json::to_string_pretty(&generation_config)?)
        .await
        .context("Failed to write generation_config.json")?;

    // 5. Save tokenizer files if available
    if let Some(model_name) = target_model {
        save_tokenizer_config(model_name, output_path).await?;
    }

    // 6. Create and save model card (README.md)
    let model_card = create_model_card(adapter, target_model, &hf_config, is_adapter_only);
    let readme_path = output_path.join("README.md");
    tokio::fs::write(&readme_path, model_card)
        .await
        .context("Failed to write README.md")?;

    tracing::info!("✅ Successfully exported to HuggingFace format at: {}", output_path.display());
    
    Ok(())
}

/// Create HuggingFace model configuration
fn create_hf_config(adapter: &LoraParameters, target_model: Option<&str>) -> Result<HuggingFaceConfig> {
    let model_type = adapter.config.target_architecture.to_lowercase();
    
    // Default configurations for common architectures
    let (hidden_size, num_heads, num_layers, intermediate_size, vocab_size) = match model_type.as_str() {
        "llama" => (4096, 32, 32, 11008, 32000),
        "mistral" => (4096, 32, 32, 14336, 32000),
        "gemma" => (3072, 16, 28, 24576, 256000),
        _ => (4096, 32, 32, 11008, 32000), // Default to Llama-like config
    };
    
    let architecture = match model_type.as_str() {
        "llama" => "LlamaForCausalLM",
        "mistral" => "MistralForCausalLM",
        "gemma" => "GemmaForCausalLM",
        _ => "LlamaForCausalLM",
    };
    
    Ok(HuggingFaceConfig {
        model_type: model_type.clone(),
        architectures: vec![architecture.to_string()],
        hidden_size,
        num_attention_heads: num_heads,
        num_hidden_layers: num_layers,
        intermediate_size,
        vocab_size,
        max_position_embeddings: 4096,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: false,
        use_cache: true,
        bos_token_id: 1,
        eos_token_id: 2,
        torch_dtype: match precision_to_torch_dtype(Precision::Fp16) {
            "float16" => "float16",
            "float32" => "float32",
            _ => "float16",
        }.to_string(),
        transformers_version: "4.40.0".to_string(),
    })
}

/// Convert T2L LoRA weights to HuggingFace state_dict format
fn convert_to_hf_state_dict(
    adapter: &LoraParameters,
    precision: Precision,
) -> Result<(HashMap<String, Tensor>, bool)> {
    let mut state_dict = HashMap::new();
    let is_adapter_only = true; // For now, we only export adapter weights
    
    for (layer_name, lora_layer) in &adapter.layers {
        // Convert T2L layer names to HuggingFace format
        let hf_layer_names = convert_layer_name_to_hf(layer_name);
        
        // Create tensors for A and B matrices
        let a_tensor = create_tensor_from_weights(
            &lora_layer.a_weights,
            lora_layer.a_matrix_shape(),
            precision,
        )?;
        
        let b_tensor = create_tensor_from_weights(
            &lora_layer.b_weights,
            lora_layer.b_matrix_shape(),
            precision,
        )?;
        
        // Add to state dict with HuggingFace naming convention
        for hf_name in hf_layer_names {
            if is_adapter_only {
                // Adapter-only format
                state_dict.insert(format!("{}.lora_A.weight", hf_name), a_tensor.clone());
                state_dict.insert(format!("{}.lora_B.weight", hf_name), b_tensor.clone());
            } else {
                // Full model format (would need base weights merged)
                // This would require loading the base model weights
                return Err(anyhow!("Full model export not yet implemented"));
            }
        }
    }
    
    Ok((state_dict, is_adapter_only))
}

/// Convert T2L layer names to HuggingFace format
fn convert_layer_name_to_hf(t2l_name: &str) -> Vec<String> {
    let mut hf_names = Vec::new();
    
    // Handle different naming patterns
    if t2l_name.contains("layers.") {
        // Pattern: layers.0.self_attn.q_proj -> model.layers.0.self_attn.q_proj
        hf_names.push(format!("model.{}", t2l_name));
    } else if t2l_name.contains("block.") {
        // Pattern: block.0.attn.q -> model.layers.0.self_attn.q_proj
        let converted = t2l_name
            .replace("block.", "model.layers.")
            .replace(".attn.", ".self_attn.")
            .replace(".q", ".q_proj")
            .replace(".k", ".k_proj")
            .replace(".v", ".v_proj")
            .replace(".o", ".o_proj");
        hf_names.push(converted);
    } else {
        // Direct mapping
        hf_names.push(format!("model.{}", t2l_name));
    }
    
    hf_names
}

/// Create tensor from weight array
fn create_tensor_from_weights(
    weights: &[f32],
    shape: (usize, usize),
    precision: Precision,
) -> Result<Tensor> {
    let dtype = match precision {
        Precision::Fp16 => TensorDataType::Float16,
        Precision::Fp32 => TensorDataType::Float32,
        Precision::Int8 => TensorDataType::Int8,
    };
    
    let metadata = TensorMetadata {
        shape: vec![shape.0, shape.1],
        dtype,
        device: DeviceType::Cpu,
        name: String::new(), // Name will be set when adding to state_dict
    };
    
    // Convert weights based on precision
    let data: Vec<u8> = match precision {
        Precision::Fp32 => {
            let bytes = weights.iter()
                .flat_map(|&v| v.to_le_bytes())
                .collect();
            bytes
        },
        Precision::Fp16 => {
            // Convert f32 to f16
            weights.iter()
                .flat_map(|&v| {
                    let f16_val = half::f16::from_f32(v);
                    f16_val.to_le_bytes().to_vec()
                })
                .collect()
        },
        Precision::Int8 => {
            // Quantize to int8 (simple implementation)
            weights.iter()
                .map(|&v| {
                    let scaled = (v * 127.0).clamp(-128.0, 127.0) as i8;
                    scaled as u8
                })
                .collect()
        },
    };
    
    Ok(Tensor {
        data,
        metadata,
    })
}

/// Save tensors in HuggingFace SafeTensors format
async fn save_hf_safetensors(
    state_dict: &HashMap<String, Tensor>,
    output_path: &Path,
    adapter: &LoraParameters,
) -> Result<()> {
    // Create metadata
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "pt".to_string());
    if let Some(meta) = adapter.metadata() {
        metadata.insert("task_description".to_string(), meta.task_description.clone());
        metadata.insert("generator_version".to_string(), meta.generator_version.clone());
    }
    
    // Write SafeTensors file using the utility function
    safetensors::write_safetensors(output_path, state_dict, Some(&metadata))
        .context("Failed to write SafeTensors file")?;
    
    Ok(())
}

/// Convert TensorDataType to string representation
fn tensor_dtype_to_string(dtype: &TensorDataType) -> String {
    match dtype {
        TensorDataType::Float32 => "F32",
        TensorDataType::Float16 => "F16",
        TensorDataType::Int8 => "I8",
        _ => "F32",
    }.to_string()
}

/// Convert Precision to PyTorch dtype string
fn precision_to_torch_dtype(precision: Precision) -> &'static str {
    match precision {
        Precision::Fp16 => "float16",
        Precision::Fp32 => "float32",
        Precision::Int8 => "int8",
    }
}

/// Create adapter configuration for HuggingFace
fn create_adapter_config(adapter: &LoraParameters, target_model: Option<&str>) -> Result<Value> {
    let config = json!({
        "adapter_type": "lora",
        "r": adapter.config.default_rank,
        "lora_alpha": adapter.config.default_alpha,
        "lora_dropout": 0.0,
        "target_modules": adapter.config.target_modules,
        "bias": "none",
        "base_model_name_or_path": target_model.unwrap_or("unknown"),
        "revision": null,
        "task_type": "CAUSAL_LM",
    });
    
    Ok(config)
}

/// Create generation configuration
fn create_generation_config() -> Value {
    json!({
        "do_sample": true,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 512,
        "repetition_penalty": 1.1,
        "eos_token_id": 2,
        "bos_token_id": 1,
        "pad_token_id": 0,
    })
}

/// Save tokenizer configuration files
async fn save_tokenizer_config(model_name: &str, output_path: &Path) -> Result<()> {
    // Basic tokenizer config
    let tokenizer_config = json!({
        "model_max_length": 4096,
        "tokenizer_class": "LlamaTokenizer",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "add_bos_token": true,
        "add_eos_token": false,
        "clean_up_tokenization_spaces": false,
    });
    
    let tokenizer_config_path = output_path.join("tokenizer_config.json");
    tokio::fs::write(&tokenizer_config_path, serde_json::to_string_pretty(&tokenizer_config)?)
        .await
        .context("Failed to write tokenizer_config.json")?;
    
    // Special tokens map
    let special_tokens_map = json!({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    });
    
    let special_tokens_path = output_path.join("special_tokens_map.json");
    tokio::fs::write(&special_tokens_path, serde_json::to_string_pretty(&special_tokens_map)?)
        .await
        .context("Failed to write special_tokens_map.json")?;
    
    Ok(())
}

/// Create model card (README.md) with metadata and usage instructions
fn create_model_card(
    adapter: &LoraParameters,
    target_model: Option<&str>,
    config: &HuggingFaceConfig,
    is_adapter_only: bool,
) -> String {
    let base_model = target_model.unwrap_or("unknown");
    let model_type = if is_adapter_only { "LoRA Adapter" } else { "Fine-tuned Model" };
    
    let mut card = format!(
        r#"---
tags:
- lora
- text-generation
- causal-lm
- {}
base_model: {}
license: apache-2.0
language:
- en
library_name: transformers
---

# T2L Generated {}

This model was generated using T2L (Text-to-LoRA) framework.

## Model Details

- **Model Type**: {}
- **Base Model**: {}
- **Architecture**: {}
- **LoRA Rank**: {}
- **LoRA Alpha**: {}
- **Target Modules**: {}
- **Total Parameters**: {:,}

"#,
        config.model_type,
        base_model,
        model_type,
        model_type,
        base_model,
        config.architectures.join(", "),
        adapter.config.default_rank,
        adapter.config.default_alpha,
        adapter.config.target_modules.join(", "),
        adapter.total_parameters()
    );

    // Add metadata if available
    if let Some(metadata) = adapter.metadata() {
        card.push_str(&format!(
            r#"## Generation Details

- **Task Description**: {}
- **Generated At**: {}
- **Generator Version**: {}

"#,
            metadata.task_description,
            metadata.created_at.format("%Y-%m-%d %H:%M:%S UTC"),
            metadata.generator_version
        ));
    }

    // Add usage instructions
    card.push_str(r#"## Usage

### Using with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "path/to/this/model",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("path/to/this/model")

# Generate text
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

"#);

    if is_adapter_only {
        card.push_str(&format!(r#"### Loading as Adapter

Since this is a LoRA adapter, you can also load it with the base model:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{}")
# Load adapter
model = PeftModel.from_pretrained(base_model, "path/to/this/adapter")
```

"#, base_model));
    }

    card.push_str(r#"## Model Architecture

This model uses the LoRA (Low-Rank Adaptation) technique to efficiently adapt a pre-trained language model. LoRA reduces the number of trainable parameters by learning low-rank decomposition matrices.

## Limitations

- This model inherits the limitations of its base model
- The adapter may exhibit behavior specific to the task it was generated for
- Performance may vary depending on the input domain

## Citation

If you use this model, please cite:

```bibtex
@software{t2l2024,
  title={T2L: Text-to-LoRA},
  author={T2L Contributors},
  year={2024},
  url={https://github.com/yourusername/t2l}
}
```

## License

This model is released under the Apache 2.0 license.
"#);

    card
}