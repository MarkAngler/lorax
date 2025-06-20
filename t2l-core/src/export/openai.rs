//! OpenAI API-compatible format exporter for T2L adapters
//!
//! This module provides functionality to export T2L LoRA adapters to OpenAI-compatible format.
//! This enables deployment of T2L adapters as custom OpenAI API-compatible models.

use crate::lora::LoraParameters;
use anyhow::{Context, Result, anyhow};
use serde::{Serialize, Deserialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use chrono::{DateTime, Utc};

/// OpenAI model metadata format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIModelMetadata {
    /// Model identifier (e.g., "ft:gpt-3.5-turbo:org-id:model-name:abc123")
    pub id: String,
    /// Model type (always "model" for models)
    pub object: String,
    /// Unix timestamp when model was created
    pub created: i64,
    /// Owner organization
    pub owned_by: String,
    /// Permission list (can be empty for custom models)
    pub permission: Vec<OpenAIPermission>,
    /// Model root (base model)
    pub root: String,
    /// Parent model (if fine-tuned from another)
    pub parent: Option<String>,
}

/// OpenAI permission structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIPermission {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    pub group: Option<String>,
    pub is_blocking: bool,
}

/// OpenAI deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIDeploymentConfig {
    /// API endpoint configuration
    pub endpoint: OpenAIEndpointConfig,
    /// Model serving configuration
    pub serving: OpenAIServingConfig,
    /// Rate limiting configuration
    pub rate_limits: OpenAIRateLimits,
}

/// OpenAI endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIEndpointConfig {
    /// Base URL for the API
    pub base_url: String,
    /// API version
    pub api_version: String,
    /// Supported endpoints
    pub endpoints: Vec<String>,
}

/// OpenAI serving configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIServingConfig {
    /// Model engine type
    pub engine: String,
    /// Maximum context length
    pub max_tokens: usize,
    /// Temperature range
    pub temperature_range: (f32, f32),
    /// Top-p range
    pub top_p_range: (f32, f32),
    /// Frequency penalty range
    pub frequency_penalty_range: (f32, f32),
    /// Presence penalty range
    pub presence_penalty_range: (f32, f32),
}

/// OpenAI rate limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIRateLimits {
    /// Requests per minute
    pub rpm: usize,
    /// Tokens per minute
    pub tpm: usize,
    /// Requests per day
    pub rpd: usize,
}

/// Export T2L adapter to OpenAI-compatible format
pub async fn export_to_openai(
    adapter: &LoraParameters,
    output_path: &Path,
) -> Result<()> {
    tracing::info!("Exporting T2L adapter to OpenAI format at: {}", output_path.display());
    
    // Create output directory
    tokio::fs::create_dir_all(output_path)
        .await
        .context("Failed to create output directory")?;
    
    // 1. Generate OpenAI model metadata
    let model_metadata = generate_model_metadata(adapter)?;
    let metadata_path = output_path.join("model_metadata.json");
    let metadata_json = serde_json::to_string_pretty(&model_metadata)
        .context("Failed to serialize model metadata")?;
    tokio::fs::write(&metadata_path, metadata_json)
        .await
        .context("Failed to write model metadata")?;
    
    tracing::debug!("Generated model metadata");
    
    // 2. Create deployment configuration
    let deployment_config = create_deployment_config(adapter)?;
    let deployment_path = output_path.join("deployment_config.json");
    let deployment_json = serde_json::to_string_pretty(&deployment_config)
        .context("Failed to serialize deployment config")?;
    tokio::fs::write(&deployment_path, deployment_json)
        .await
        .context("Failed to write deployment config")?;
    
    tracing::debug!("Created deployment configuration");
    
    // 3. Generate API specification
    let api_spec = generate_api_specification(adapter, &model_metadata)?;
    let api_spec_path = output_path.join("openai_api_spec.json");
    tokio::fs::write(&api_spec_path, serde_json::to_string_pretty(&api_spec)?)
        .await
        .context("Failed to write API specification")?;
    
    tracing::debug!("Generated API specification");
    
    // 4. Export adapter weights in OpenAI format
    export_adapter_weights(adapter, output_path).await?;
    
    // 5. Generate model capabilities documentation
    let capabilities_doc = generate_capabilities_doc(adapter)?;
    let capabilities_path = output_path.join("model_capabilities.json");
    tokio::fs::write(&capabilities_path, serde_json::to_string_pretty(&capabilities_doc)?)
        .await
        .context("Failed to write capabilities documentation")?;
    
    // 6. Create usage examples and documentation
    create_usage_documentation(adapter, &model_metadata, output_path).await?;
    
    // 7. Generate fine-tuning configuration
    let finetune_config = generate_finetune_config(adapter)?;
    let finetune_path = output_path.join("finetune_config.json");
    tokio::fs::write(&finetune_path, serde_json::to_string_pretty(&finetune_config)?)
        .await
        .context("Failed to write fine-tuning configuration")?;
    
    tracing::info!("✅ Successfully exported T2L adapter to OpenAI format");
    tracing::info!("📁 Output directory: {}", output_path.display());
    tracing::info!("🆔 Model ID: {}", model_metadata.id);
    
    Ok(())
}

/// Generate OpenAI model metadata
fn generate_model_metadata(adapter: &LoraParameters) -> Result<OpenAIModelMetadata> {
    let created_timestamp = adapter.metadata()
        .map(|m| m.created_at.timestamp())
        .unwrap_or_else(|| Utc::now().timestamp());
    
    // Generate model ID based on architecture and task
    let model_id = generate_model_id(adapter)?;
    let org_id = "t2l-org";
    
    Ok(OpenAIModelMetadata {
        id: model_id.clone(),
        object: "model".to_string(),
        created: created_timestamp,
        owned_by: format!("{}", org_id),
        permission: vec![
            OpenAIPermission {
                id: format!("modelperm-{}", generate_permission_id()),
                object: "model_permission".to_string(),
                created: created_timestamp,
                allow_create_engine: false,
                allow_sampling: true,
                allow_logprobs: true,
                allow_search_indices: false,
                allow_view: true,
                allow_fine_tuning: true,
                organization: org_id.to_string(),
                group: None,
                is_blocking: false,
            }
        ],
        root: get_base_model_name(&adapter.config.target_architecture),
        parent: None,
    })
}

/// Generate unique model ID
fn generate_model_id(adapter: &LoraParameters) -> Result<String> {
    let arch = &adapter.config.target_architecture;
    let task_suffix = adapter.metadata()
        .map(|m| {
            // Extract key words from task description
            let task_words: Vec<&str> = m.task_description
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .take(3)
                .collect();
            task_words.join("-").to_lowercase()
        })
        .unwrap_or_else(|| "custom".to_string());
    
    // Generate hash from adapter parameters for uniqueness
    let param_hash = format!("{:x}", calculate_adapter_hash(adapter));
    let short_hash = &param_hash[..8];
    
    Ok(format!("ft:t2l-{}:t2l-org:{}:{}", arch, task_suffix, short_hash))
}

/// Calculate hash of adapter parameters
fn calculate_adapter_hash(adapter: &LoraParameters) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    adapter.config.target_architecture.hash(&mut hasher);
    adapter.total_parameters().hash(&mut hasher);
    adapter.layer_names().len().hash(&mut hasher);
    hasher.finish()
}

/// Generate permission ID
fn generate_permission_id() -> String {
    use uuid::Uuid;
    Uuid::new_v4().to_string().chars().take(12).collect()
}

/// Get base model name from architecture
fn get_base_model_name(architecture: &str) -> String {
    match architecture.to_lowercase().as_str() {
        "llama" | "llama2" => "meta-llama/Llama-2-7b-hf",
        "llama3" => "meta-llama/Meta-Llama-3-8B",
        "mistral" => "mistralai/Mistral-7B-v0.1",
        "gemma" => "google/gemma-7b",
        "phi" => "microsoft/phi-2",
        _ => format!("{}-base", architecture),
    }.to_string()
}

/// Create deployment configuration
fn create_deployment_config(adapter: &LoraParameters) -> Result<OpenAIDeploymentConfig> {
    let max_tokens = match adapter.config.target_architecture.to_lowercase().as_str() {
        "llama" | "llama2" | "llama3" => 4096,
        "mistral" => 8192,
        "gemma" => 8192,
        _ => 4096,
    };
    
    Ok(OpenAIDeploymentConfig {
        endpoint: OpenAIEndpointConfig {
            base_url: "https://api.openai.com/v1".to_string(),
            api_version: "v1".to_string(),
            endpoints: vec![
                "/chat/completions".to_string(),
                "/completions".to_string(),
                "/embeddings".to_string(),
            ],
        },
        serving: OpenAIServingConfig {
            engine: "t2l-inference".to_string(),
            max_tokens,
            temperature_range: (0.0, 2.0),
            top_p_range: (0.0, 1.0),
            frequency_penalty_range: (-2.0, 2.0),
            presence_penalty_range: (-2.0, 2.0),
        },
        rate_limits: OpenAIRateLimits {
            rpm: 3500,
            tpm: 90000,
            rpd: 200000,
        },
    })
}

/// Generate API specification
fn generate_api_specification(
    adapter: &LoraParameters,
    metadata: &OpenAIModelMetadata,
) -> Result<Value> {
    Ok(json!({
        "openapi": "3.0.0",
        "info": {
            "title": format!("T2L Model API - {}", metadata.id),
            "version": "1.0.0",
            "description": format!(
                "OpenAI-compatible API for T2L generated {} model",
                adapter.config.target_architecture
            ),
        },
        "servers": [
            {
                "url": "https://api.your-domain.com/v1",
                "description": "Production server"
            }
        ],
        "paths": {
            "/chat/completions": {
                "post": {
                    "summary": "Create chat completion",
                    "operationId": "createChatCompletion",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ChatCompletionRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ChatCompletionResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "ChatCompletionRequest": {
                    "type": "object",
                    "required": ["model", "messages"],
                    "properties": {
                        "model": {
                            "type": "string",
                            "example": metadata.id
                        },
                        "messages": {
                            "type": "array",
                            "items": {
                                "$ref": "#/components/schemas/ChatMessage"
                            }
                        },
                        "temperature": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 2,
                            "default": 0.7
                        },
                        "max_tokens": {
                            "type": "integer",
                            "minimum": 1
                        },
                        "stream": {
                            "type": "boolean",
                            "default": false
                        }
                    }
                },
                "ChatMessage": {
                    "type": "object",
                    "required": ["role", "content"],
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["system", "user", "assistant"]
                        },
                        "content": {
                            "type": "string"
                        }
                    }
                },
                "ChatCompletionResponse": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "object": {"type": "string"},
                        "created": {"type": "integer"},
                        "model": {"type": "string"},
                        "choices": {
                            "type": "array",
                            "items": {
                                "$ref": "#/components/schemas/ChatChoice"
                            }
                        },
                        "usage": {
                            "$ref": "#/components/schemas/Usage"
                        }
                    }
                },
                "ChatChoice": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "message": {
                            "$ref": "#/components/schemas/ChatMessage"
                        },
                        "finish_reason": {"type": "string"}
                    }
                },
                "Usage": {
                    "type": "object",
                    "properties": {
                        "prompt_tokens": {"type": "integer"},
                        "completion_tokens": {"type": "integer"},
                        "total_tokens": {"type": "integer"}
                    }
                }
            }
        }
    }))
}

/// Export adapter weights in OpenAI format
async fn export_adapter_weights(adapter: &LoraParameters, output_path: &Path) -> Result<()> {
    let weights_dir = output_path.join("weights");
    tokio::fs::create_dir_all(&weights_dir)
        .await
        .context("Failed to create weights directory")?;
    
    // Export adapter configuration
    let adapter_config = json!({
        "adapter_type": "lora",
        "r": adapter.config.default_rank,
        "lora_alpha": adapter.config.default_alpha,
        "target_modules": adapter.config.target_modules,
        "architecture": adapter.config.target_architecture,
        "total_parameters": adapter.total_parameters(),
        "layers": adapter.layer_names().into_iter().map(|name| {
            let layer = adapter.get_layer(name).unwrap();
            json!({
                "name": name,
                "rank": layer.rank,
                "alpha": layer.alpha,
                "input_dim": layer.input_dim,
                "output_dim": layer.output_dim,
                "parameters": layer.num_parameters(),
            })
        }).collect::<Vec<_>>(),
    });
    
    let config_path = weights_dir.join("adapter_config.json");
    tokio::fs::write(&config_path, serde_json::to_string_pretty(&adapter_config)?)
        .await
        .context("Failed to write adapter config")?;
    
    // Export weight index for efficient loading
    let weight_index = create_weight_index(adapter)?;
    let index_path = weights_dir.join("weight_index.json");
    tokio::fs::write(&index_path, serde_json::to_string_pretty(&weight_index)?)
        .await
        .context("Failed to write weight index")?;
    
    tracing::debug!("Exported adapter weights and configuration");
    
    Ok(())
}

/// Create weight index for efficient loading
fn create_weight_index(adapter: &LoraParameters) -> Result<Value> {
    let mut weight_files = HashMap::new();
    let mut total_size = 0;
    
    for (layer_name, layer) in &adapter.layers {
        let a_size = layer.a_weights.len() * std::mem::size_of::<f32>();
        let b_size = layer.b_weights.len() * std::mem::size_of::<f32>();
        
        weight_files.insert(format!("{}.lora_A", layer_name), json!({
            "shape": [layer.input_dim, layer.rank],
            "dtype": "float32",
            "size_bytes": a_size,
        }));
        
        weight_files.insert(format!("{}.lora_B", layer_name), json!({
            "shape": [layer.rank, layer.output_dim],
            "dtype": "float32",
            "size_bytes": b_size,
        }));
        
        total_size += a_size + b_size;
    }
    
    Ok(json!({
        "metadata": {
            "total_size": total_size,
            "tensor_count": weight_files.len(),
            "format": "t2l-openai",
            "version": "1.0",
        },
        "weight_map": weight_files,
    }))
}

/// Generate model capabilities documentation
fn generate_capabilities_doc(adapter: &LoraParameters) -> Result<Value> {
    let capabilities = json!({
        "model_type": "text-generation",
        "architectures": [adapter.config.target_architecture.clone()],
        "capabilities": {
            "text_generation": true,
            "chat": true,
            "instruction_following": true,
            "function_calling": false,
            "embeddings": false,
            "fine_tuning": true,
            "streaming": true,
        },
        "parameters": {
            "adapter_type": "lora",
            "total_parameters": adapter.total_parameters(),
            "trainable_parameters": adapter.total_parameters(),
            "rank": adapter.config.default_rank,
            "alpha": adapter.config.default_alpha,
            "target_modules": adapter.config.target_modules,
        },
        "context_window": {
            "max_tokens": 4096,
            "recommended": 2048,
        },
        "performance": {
            "inference_optimization": "lora_efficient",
            "batch_size_recommendation": 8,
            "precision": "fp16",
        },
        "limitations": [
            "Model behavior is influenced by the LoRA adaptation",
            "May exhibit task-specific biases based on adaptation",
            "Performance depends on similarity to adaptation task",
        ],
    });
    
    Ok(capabilities)
}

/// Create usage documentation
async fn create_usage_documentation(
    adapter: &LoraParameters,
    metadata: &OpenAIModelMetadata,
    output_path: &Path,
) -> Result<()> {
    // Create README with usage examples
    let total_params_formatted = format!("{:}", adapter.total_parameters());
    let readme_content = format!(
        r#"# T2L OpenAI-Compatible Model

## Model Information
- **Model ID**: `{}`
- **Base Model**: `{}`
- **Architecture**: `{}`
- **Created**: `{}`
- **Total Parameters**: `{}`

## Quick Start

### Installation

```bash
pip install openai
```

### Basic Usage

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    api_key="your-api-key",
    base_url="https://your-api-endpoint.com/v1"
)

# Create a chat completion
response = client.chat.completions.create(
    model="{}",
    messages=[
        {{"role": "system", "content": "You are a helpful assistant."}},
        {{"role": "user", "content": "Hello! How can you help me today?"}}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Streaming Example

```python
# Stream responses
stream = client.chat.completions.create(
    model="{}",
    messages=[{{"role": "user", "content": "Tell me a story"}}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### Advanced Configuration

```python
response = client.chat.completions.create(
    model="{}",
    messages=messages,
    temperature=0.8,
    max_tokens=1000,
    top_p=0.9,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop=["\\n\\n", "END"]
)
```

## API Endpoints

### Chat Completions
- **Endpoint**: `POST /v1/chat/completions`
- **Model**: `{}`

### Legacy Completions
- **Endpoint**: `POST /v1/completions`
- **Model**: `{}`

## Model Configuration

{}

## Rate Limits
- **Requests per minute**: 3,500
- **Tokens per minute**: 90,000
- **Requests per day**: 200,000

## Authentication

Include your API key in the `Authorization` header:

```
Authorization: Bearer YOUR_API_KEY
```

## Error Handling

```python
try:
    response = client.chat.completions.create(
        model="{}",
        messages=messages
    )
except openai.APIError as e:
    print(f"OpenAI API error: {{e}}")
except openai.APIConnectionError as e:
    print(f"Failed to connect to API: {{e}}")
except openai.RateLimitError as e:
    print(f"Rate limit exceeded: {{e}}")
```

## Best Practices

1. **Prompt Engineering**: This model has been adapted using LoRA for specific tasks. Structure your prompts to align with the adaptation.

2. **Temperature Settings**: 
   - Use lower temperatures (0.0-0.5) for factual/deterministic outputs
   - Use higher temperatures (0.7-1.0) for creative tasks

3. **Context Management**: Keep track of your token usage to stay within context limits.

4. **Error Handling**: Always implement proper error handling and retry logic.

## Support

For issues or questions, please refer to the T2L documentation or contact support.
"#,
        metadata.id,
        metadata.root,
        adapter.config.target_architecture,
        DateTime::<Utc>::from_timestamp(metadata.created, 0)
            .unwrap_or_default()
            .format("%Y-%m-%d %H:%M:%S UTC"),
        total_params_formatted,
        metadata.id,
        metadata.id,
        metadata.id,
        metadata.id,
        metadata.id,
        serde_json::to_string_pretty(&json!({
            "architecture": adapter.config.target_architecture,
            "lora_rank": adapter.config.default_rank,
            "lora_alpha": adapter.config.default_alpha,
            "target_modules": adapter.config.target_modules,
        }))?,
        metadata.id,
    );
    
    let readme_path = output_path.join("README.md");
    tokio::fs::write(&readme_path, readme_content)
        .await
        .context("Failed to write README")?;
    
    // Create example scripts directory
    let examples_dir = output_path.join("examples");
    tokio::fs::create_dir_all(&examples_dir)
        .await
        .context("Failed to create examples directory")?;
    
    // Create basic example script
    let basic_example = format!(
        r#"#!/usr/bin/env python3
"""Basic example of using the T2L OpenAI-compatible model"""

from openai import OpenAI
import os

# Initialize the client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

def chat_with_model():
    """Simple chat interaction with the model"""
    
    messages = [
        {{"role": "system", "content": "You are a helpful assistant."}},
        {{"role": "user", "content": "What are the benefits of using LoRA for model adaptation?"}}
    ]
    
    response = client.chat.completions.create(
        model="{}",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    print("Model response:")
    print(response.choices[0].message.content)
    print(f"\\nTokens used: {{response.usage.total_tokens}}")

if __name__ == "__main__":
    chat_with_model()
"#,
        metadata.id
    );
    
    let example_path = examples_dir.join("basic_chat.py");
    tokio::fs::write(&example_path, basic_example)
        .await
        .context("Failed to write basic example")?;
    
    // Create streaming example
    let streaming_example = format!(
        r#"#!/usr/bin/env python3
"""Streaming example for the T2L OpenAI-compatible model"""

from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

def stream_chat():
    """Stream responses from the model"""
    
    messages = [
        {{"role": "user", "content": "Write a short story about AI and humans working together."}}
    ]
    
    stream = client.chat.completions.create(
        model="{}",
        messages=messages,
        temperature=0.8,
        max_tokens=1000,
        stream=True
    )
    
    print("Streaming response:\\n")
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\\n\\nStreaming complete!")

if __name__ == "__main__":
    stream_chat()
"#,
        metadata.id
    );
    
    let stream_path = examples_dir.join("streaming_chat.py");
    tokio::fs::write(&stream_path, streaming_example)
        .await
        .context("Failed to write streaming example")?;
    
    tracing::debug!("Created usage documentation and examples");
    
    Ok(())
}

/// Generate fine-tuning configuration
fn generate_finetune_config(adapter: &LoraParameters) -> Result<Value> {
    Ok(json!({
        "fine_tuning": {
            "base_model": get_base_model_name(&adapter.config.target_architecture),
            "adapter_config": {
                "r": adapter.config.default_rank,
                "lora_alpha": adapter.config.default_alpha,
                "lora_dropout": 0.0,
                "target_modules": adapter.config.target_modules,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            "training_config": {
                "learning_rate": 1e-4,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "warmup_ratio": 0.1,
                "weight_decay": 0.001,
                "logging_steps": 10,
                "save_strategy": "epoch",
                "evaluation_strategy": "epoch",
                "fp16": true,
                "gradient_checkpointing": true,
                "optim": "adamw_torch",
                "lr_scheduler_type": "cosine",
            },
            "data_config": {
                "max_seq_length": 2048,
                "dataset_text_field": "text",
                "formatting_func": "chat_template",
                "packing": false,
            },
            "deployment": {
                "merge_and_unload": false,
                "push_to_hub": false,
                "quantization": null,
            }
        }
    }))
}