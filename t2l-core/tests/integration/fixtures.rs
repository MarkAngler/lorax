//! Test fixtures and utilities for integration tests

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use t2l_core::lora::{LoraParameters, LoraLayer, LoraParameterConfig, ParameterMetadata};
use t2l_core::Result;
use serde_json::json;
use std::fs;
use serde_json;

/// Create a temporary directory for test outputs
pub fn create_test_dir() -> (TempDir, PathBuf) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().to_path_buf();
    (temp_dir, path)
}

/// Create a minimal test adapter for a given architecture
pub fn create_test_adapter(architecture: &str, rank: usize) -> LoraParameters {
    let mut layers = HashMap::new();
    
    // Define layer configurations based on architecture
    let layer_configs = match architecture {
        "llama" => vec![
            ("model.layers.0.self_attn.q_proj", 4096, 4096),
            ("model.layers.0.self_attn.k_proj", 4096, 4096),
            ("model.layers.0.self_attn.v_proj", 4096, 4096),
            ("model.layers.0.self_attn.o_proj", 4096, 4096),
            ("model.layers.0.mlp.gate_proj", 4096, 11008),
            ("model.layers.0.mlp.up_proj", 4096, 11008),
            ("model.layers.0.mlp.down_proj", 11008, 4096),
        ],
        "mistral" => vec![
            ("model.layers.0.self_attn.q_proj", 4096, 4096),
            ("model.layers.0.self_attn.k_proj", 4096, 1024),
            ("model.layers.0.self_attn.v_proj", 4096, 1024),
            ("model.layers.0.self_attn.o_proj", 4096, 4096),
            ("model.layers.0.mlp.gate_proj", 4096, 14336),
            ("model.layers.0.mlp.up_proj", 4096, 14336),
            ("model.layers.0.mlp.down_proj", 14336, 4096),
        ],
        "gemma" => vec![
            ("model.layers.0.self_attn.q_proj", 3072, 3072),
            ("model.layers.0.self_attn.k_proj", 3072, 256),
            ("model.layers.0.self_attn.v_proj", 3072, 256),
            ("model.layers.0.self_attn.o_proj", 3072, 3072),
            ("model.layers.0.mlp.gate_proj", 3072, 24576),
            ("model.layers.0.mlp.up_proj", 3072, 24576),
            ("model.layers.0.mlp.down_proj", 24576, 3072),
        ],
        _ => vec![
            ("layers.0.self_attn.q_proj", 768, 768),
            ("layers.0.self_attn.v_proj", 768, 768),
        ],
    };
    
    let alpha = rank as f32 * 2.0;
    
    for (name, input_dim, output_dim) in layer_configs {
        let mut layer = LoraLayer::new(
            name.to_string(),
            input_dim,
            output_dim,
            rank,
            alpha,
        );
        
        // Initialize with deterministic test data
        layer.randomize_weights();
        layers.insert(layer.name.clone(), layer);
    }
    
    let target_modules = match architecture {
        "llama" | "mistral" | "gemma" => vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
            "gate_proj".to_string(),
            "up_proj".to_string(),
            "down_proj".to_string(),
        ],
        _ => vec!["q_proj".to_string(), "v_proj".to_string()],
    };
    
    let config = LoraParameterConfig {
        target_architecture: architecture.to_string(),
        default_rank: rank,
        default_alpha: alpha,
        target_modules,
        merge_weights: false,
    };
    
    LoraParameters {
        layers,
        config,
        metadata: Some(HashMap::from([
            ("task_description".to_string(), "Test task for integration testing".to_string()),
            ("model_type".to_string(), architecture.to_string()),
            ("created_by".to_string(), "t2l-core-test".to_string()),
        ])),
    }
}

/// Create a mock base model configuration file
pub fn create_mock_model_config(path: &Path, architecture: &str) -> Result<()> {
    let config = match architecture {
        "llama" => json!({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32000,
            "model_type": "llama",
        }),
        "mistral" => json!({
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "rms_norm_eps": 1e-5,
            "vocab_size": 32000,
            "model_type": "mistral",
        }),
        "gemma" => json!({
            "architectures": ["GemmaForCausalLM"],
            "hidden_size": 3072,
            "intermediate_size": 24576,
            "num_attention_heads": 16,
            "num_key_value_heads": 1,
            "num_hidden_layers": 28,
            "rms_norm_eps": 1e-6,
            "vocab_size": 256000,
            "model_type": "gemma",
        }),
        _ => json!({
            "architectures": ["GPT2LMHeadModel"],
            "hidden_size": 768,
            "n_head": 12,
            "n_layer": 12,
            "vocab_size": 50257,
            "model_type": "gpt2",
        }),
    };
    
    std::fs::write(path.join("config.json"), config.to_string())?;
    Ok(())
}

/// Create mock model weights (minimal for testing)
pub fn create_mock_model_weights(path: &Path, architecture: &str) -> Result<()> {
    use safetensors::{serialize, SafeTensors};
    use ndarray::Array2;
    
    let mut tensors = HashMap::new();
    
    // Add minimal tensors based on architecture
    match architecture {
        "llama" => {
            // Add embedding and output weights
            let embed = Array2::<f32>::zeros((32000, 4096));
            let output = Array2::<f32>::zeros((32000, 4096));
            tensors.insert("model.embed_tokens.weight".to_string(), embed);
            tensors.insert("lm_head.weight".to_string(), output);
        }
        "mistral" => {
            let embed = Array2::<f32>::zeros((32000, 4096));
            let output = Array2::<f32>::zeros((32000, 4096));
            tensors.insert("model.embed_tokens.weight".to_string(), embed);
            tensors.insert("lm_head.weight".to_string(), output);
        }
        "gemma" => {
            let embed = Array2::<f32>::zeros((256000, 3072));
            let output = Array2::<f32>::zeros((256000, 3072));
            tensors.insert("model.embed_tokens.weight".to_string(), embed);
            tensors.insert("lm_head.weight".to_string(), output);
        }
        _ => {
            let embed = Array2::<f32>::zeros((50257, 768));
            let output = Array2::<f32>::zeros((50257, 768));
            tensors.insert("wte.weight".to_string(), embed);
            tensors.insert("lm_head.weight".to_string(), output);
        }
    }
    
    // Note: This is a simplified version. In real implementation,
    // we'd use safetensors::serialize properly
    std::fs::create_dir_all(path)?;
    std::fs::write(path.join("model.safetensors"), b"mock_weights")?;
    
    Ok(())
}

/// Create a complete mock model directory
pub fn create_mock_model(path: &Path, architecture: &str) -> Result<()> {
    std::fs::create_dir_all(path)?;
    create_mock_model_config(path, architecture)?;
    create_mock_model_weights(path, architecture)?;
    
    // Create tokenizer files
    let tokenizer_config = json!({
        "tokenizer_class": match architecture {
            "llama" | "mistral" => "LlamaTokenizer",
            "gemma" => "GemmaTokenizer",
            _ => "GPT2Tokenizer",
        },
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "bos_token": "<s>",
    });
    
    std::fs::write(path.join("tokenizer_config.json"), tokenizer_config.to_string())?;
    std::fs::write(path.join("tokenizer.json"), "{}")?; // Minimal tokenizer
    
    Ok(())
}

/// Sample task descriptions for testing
pub const TEST_TASKS: &[&str] = &[
    "Summarize news articles into bullet points",
    "Translate English text to French",
    "Generate Python code from natural language descriptions",
    "Answer questions about scientific papers",
    "Correct grammar and spelling errors",
    "Extract key information from business documents",
    "Generate creative stories based on prompts",
    "Classify sentiment in customer reviews",
];

/// Test model IDs for different architectures
pub const TEST_MODEL_IDS: &[(&str, &str)] = &[
    ("llama", "meta-llama/Llama-2-7b-hf"),
    ("mistral", "mistralai/Mistral-7B-v0.1"),
    ("gemma", "google/gemma-2b"),
    ("gpt2", "gpt2"),
];

/// Create a test configuration file
pub fn create_test_config(path: &Path, config_type: &str) -> Result<()> {
    let config = match config_type {
        "encoder" => json!({
            "encoder": {
                "type": "sentence-transformers",
                "model_name": "all-MiniLM-L6-v2",
                "embedding_dim": 384,
                "max_length": 512,
                "cache_embeddings": true,
            }
        }),
        "hypernetwork" => json!({
            "hypernetwork": {
                "architecture": "mlp",
                "input_dim": 384,
                "hidden_dims": [768, 1024],
                "output_dim": 2048,
                "activation": "gelu",
                "dropout": 0.1,
                "use_layer_norm": true,
            }
        }),
        "lora" => json!({
            "lora": {
                "default_rank": 16,
                "default_alpha": 32.0,
                "target_modules": ["q_proj", "v_proj"],
                "merge_weights": false,
            }
        }),
        _ => json!({
            "encoder": {
                "type": "sentence-transformers",
                "model_name": "all-MiniLM-L6-v2",
                "embedding_dim": 384,
            },
            "hypernetwork": {
                "architecture": "transformer",
                "input_dim": 384,
                "hidden_dims": [512],
                "output_dim": 1024,
            },
            "lora": {
                "default_rank": 8,
                "default_alpha": 16.0,
            }
        }),
    };
    
    std::fs::write(path, config.to_string())?;
    Ok(())
}

/// Verify that a file exists and has non-zero size
pub fn verify_file_exists(path: &Path) -> Result<()> {
    assert!(path.exists(), "File does not exist: {:?}", path);
    let metadata = std::fs::metadata(path)?;
    assert!(metadata.len() > 0, "File is empty: {:?}", path);
    Ok(())
}

/// Verify JSON file content
pub fn verify_json_file(path: &Path) -> Result<serde_json::Value> {
    let content = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;
    Ok(json)
}

/// Save LoraParameters to a directory (helper for tests)
pub fn save_lora_parameters(params: &LoraParameters, path: &Path) -> Result<()> {
    std::fs::create_dir_all(path)?;
    
    // Save config
    let config_path = path.join("adapter_config.json");
    let config_json = serde_json::to_string_pretty(&params.config)?;
    std::fs::write(config_path, config_json)?;
    
    // Save layers
    let layers_path = path.join("adapter_model.json");
    let layers_json = serde_json::to_string_pretty(&params.layers)?;
    std::fs::write(layers_path, layers_json)?;
    
    // Save metadata if present
    if let Some(metadata) = &params.metadata {
        let metadata_path = path.join("adapter_metadata.json");
        let metadata_json = serde_json::to_string_pretty(metadata)?;
        std::fs::write(metadata_path, metadata_json)?;
    }
    
    Ok(())
}

/// Load LoraParameters from a directory (helper for tests)
pub fn load_lora_parameters(path: &Path) -> Result<LoraParameters> {
    // Load config
    let config_path = path.join("adapter_config.json");
    let config_json = std::fs::read_to_string(config_path)?;
    let config: LoraParameterConfig = serde_json::from_str(&config_json)?;
    
    // Load layers
    let layers_path = path.join("adapter_model.json");
    let layers_json = std::fs::read_to_string(layers_path)?;
    let layers: HashMap<String, LoraLayer> = serde_json::from_str(&layers_json)?;
    
    // Load metadata if present
    let metadata = if path.join("adapter_metadata.json").exists() {
        let metadata_json = std::fs::read_to_string(path.join("adapter_metadata.json"))?;
        Some(serde_json::from_str(&metadata_json)?)
    } else {
        None
    };
    
    Ok(LoraParameters {
        layers,
        config,
        metadata,
    })
}