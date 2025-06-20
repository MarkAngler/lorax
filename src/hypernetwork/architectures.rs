//! Target model architecture handling

use super::lora::{LayerConfig, LayerType};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported target model architectures
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TargetArchitecture {
    /// GPT-style decoder-only transformer
    GPT {
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
    },
    /// BERT-style encoder transformer
    BERT {
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
    },
    /// T5-style encoder-decoder transformer
    T5 {
        hidden_size: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        num_heads: usize,
    },
    /// LLaMA architecture
    LLaMA {
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_size: usize,
    },
    /// Vision Transformer
    ViT {
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        patch_size: usize,
    },
    /// Custom architecture with flexible configuration
    Custom {
        config: HashMap<String, usize>,
    },
}

impl TargetArchitecture {
    /// Get a human-readable name for the architecture
    pub fn name(&self) -> &str {
        match self {
            TargetArchitecture::GPT { .. } => "GPT",
            TargetArchitecture::BERT { .. } => "BERT",
            TargetArchitecture::T5 { .. } => "T5",
            TargetArchitecture::LLaMA { .. } => "LLaMA",
            TargetArchitecture::ViT { .. } => "ViT",
            TargetArchitecture::Custom { .. } => "Custom",
        }
    }
}

/// Handler for different model architectures
pub struct ArchitectureHandler {
    /// Predefined layer patterns for common architectures
    patterns: HashMap<String, Vec<LayerPattern>>,
}

#[derive(Debug, Clone)]
struct LayerPattern {
    name_template: String,
    layer_type: LayerType,
    in_features_fn: Box<dyn Fn(usize) -> usize>,
    out_features_fn: Box<dyn Fn(usize) -> usize>,
}

impl ArchitectureHandler {
    pub fn new() -> Self {
        let mut handler = Self {
            patterns: HashMap::new(),
        };
        
        handler.register_default_patterns();
        handler
    }
    
    /// Register default patterns for common architectures
    fn register_default_patterns(&mut self) {
        // GPT patterns
        self.patterns.insert("GPT".to_string(), vec![
            LayerPattern {
                name_template: "transformer.h.{}.attn.c_attn".to_string(),
                layer_type: LayerType::Attention,
                in_features_fn: Box::new(|h| h),
                out_features_fn: Box::new(|h| h * 3), // Q, K, V
            },
            LayerPattern {
                name_template: "transformer.h.{}.attn.c_proj".to_string(),
                layer_type: LayerType::Attention,
                in_features_fn: Box::new(|h| h),
                out_features_fn: Box::new(|h| h),
            },
            LayerPattern {
                name_template: "transformer.h.{}.mlp.c_fc".to_string(),
                layer_type: LayerType::MLP,
                in_features_fn: Box::new(|h| h),
                out_features_fn: Box::new(|h| h * 4),
            },
            LayerPattern {
                name_template: "transformer.h.{}.mlp.c_proj".to_string(),
                layer_type: LayerType::MLP,
                in_features_fn: Box::new(|h| h * 4),
                out_features_fn: Box::new(|h| h),
            },
        ]);
        
        // BERT patterns
        self.patterns.insert("BERT".to_string(), vec![
            LayerPattern {
                name_template: "bert.encoder.layer.{}.attention.self.query".to_string(),
                layer_type: LayerType::Attention,
                in_features_fn: Box::new(|h| h),
                out_features_fn: Box::new(|h| h),
            },
            LayerPattern {
                name_template: "bert.encoder.layer.{}.attention.self.key".to_string(),
                layer_type: LayerType::Attention,
                in_features_fn: Box::new(|h| h),
                out_features_fn: Box::new(|h| h),
            },
            LayerPattern {
                name_template: "bert.encoder.layer.{}.attention.self.value".to_string(),
                layer_type: LayerType::Attention,
                in_features_fn: Box::new(|h| h),
                out_features_fn: Box::new(|h| h),
            },
            LayerPattern {
                name_template: "bert.encoder.layer.{}.intermediate.dense".to_string(),
                layer_type: LayerType::MLP,
                in_features_fn: Box::new(|h| h),
                out_features_fn: Box::new(|h| h * 4),
            },
        ]);
        
        // LLaMA patterns
        self.patterns.insert("LLaMA".to_string(), vec![
            LayerPattern {
                name_template: "model.layers.{}.self_attn.q_proj".to_string(),
                layer_type: LayerType::Attention,
                in_features_fn: Box::new(|h| h),
                out_features_fn: Box::new(|h| h),
            },
            LayerPattern {
                name_template: "model.layers.{}.self_attn.k_proj".to_string(),
                layer_type: LayerType::Attention,
                in_features_fn: Box::new(|h| h),
                out_features_fn: Box::new(|h| h),
            },
            LayerPattern {
                name_template: "model.layers.{}.self_attn.v_proj".to_string(),
                layer_type: LayerType::Attention,
                in_features_fn: Box::new(|h| h),
                out_features_fn: Box::new(|h| h),
            },
            LayerPattern {
                name_template: "model.layers.{}.self_attn.o_proj".to_string(),
                layer_type: LayerType::Attention,
                in_features_fn: Box::new(|h| h),
                out_features_fn: Box::new(|h| h),
            },
            LayerPattern {
                name_template: "model.layers.{}.mlp.gate_proj".to_string(),
                layer_type: LayerType::MLP,
                in_features_fn: Box::new(|h| h),
                out_features_fn: Box::new(|_| 0), // Will be set from intermediate_size
            },
            LayerPattern {
                name_template: "model.layers.{}.mlp.up_proj".to_string(),
                layer_type: LayerType::MLP,
                in_features_fn: Box::new(|h| h),
                out_features_fn: Box::new(|_| 0), // Will be set from intermediate_size
            },
            LayerPattern {
                name_template: "model.layers.{}.mlp.down_proj".to_string(),
                layer_type: LayerType::MLP,
                in_features_fn: Box::new(|_| 0), // Will be set from intermediate_size
                out_features_fn: Box::new(|h| h),
            },
        ]);
    }
    
    /// Get layer configurations for a target architecture
    pub fn get_layer_configs(&self, arch: &TargetArchitecture) -> Result<Vec<LayerConfig>> {
        match arch {
            TargetArchitecture::GPT { hidden_size, num_layers, .. } => {
                self.generate_configs("GPT", *hidden_size, *num_layers, None)
            }
            TargetArchitecture::BERT { hidden_size, num_layers, .. } => {
                self.generate_configs("BERT", *hidden_size, *num_layers, None)
            }
            TargetArchitecture::LLaMA { hidden_size, num_layers, intermediate_size, .. } => {
                self.generate_configs("LLaMA", *hidden_size, *num_layers, Some(*intermediate_size))
            }
            TargetArchitecture::T5 { .. } => {
                // T5 is more complex with encoder and decoder
                Err(anyhow!("T5 architecture support not yet implemented"))
            }
            TargetArchitecture::ViT { .. } => {
                Err(anyhow!("ViT architecture support not yet implemented"))
            }
            TargetArchitecture::Custom { .. } => {
                Err(anyhow!("Custom architecture requires manual configuration"))
            }
        }
    }
    
    /// Generate layer configurations from patterns
    fn generate_configs(
        &self,
        arch_name: &str,
        hidden_size: usize,
        num_layers: usize,
        intermediate_size: Option<usize>,
    ) -> Result<Vec<LayerConfig>> {
        let patterns = self.patterns.get(arch_name)
            .ok_or_else(|| anyhow!("Unknown architecture: {}", arch_name))?;
        
        let mut configs = Vec::new();
        
        for layer_idx in 0..num_layers {
            for pattern in patterns {
                let name = pattern.name_template.replace("{}", &layer_idx.to_string());
                
                let mut in_features = (pattern.in_features_fn)(hidden_size);
                let mut out_features = (pattern.out_features_fn)(hidden_size);
                
                // Handle special cases like LLaMA intermediate size
                if arch_name == "LLaMA" && pattern.layer_type == LayerType::MLP {
                    if let Some(inter_size) = intermediate_size {
                        if out_features == 0 {
                            out_features = inter_size;
                        }
                        if in_features == 0 {
                            in_features = inter_size;
                        }
                    }
                }
                
                configs.push(LayerConfig {
                    name,
                    in_features,
                    out_features,
                    layer_type: pattern.layer_type,
                });
            }
        }
        
        Ok(configs)
    }
    
    /// Add a custom architecture pattern
    pub fn add_custom_pattern(
        &mut self,
        arch_name: String,
        patterns: Vec<LayerPattern>,
    ) {
        self.patterns.insert(arch_name, patterns);
    }
}

// Helper to create LayerPattern more easily
impl LayerPattern {
    pub fn new(
        name_template: String,
        layer_type: LayerType,
        in_scale: f32,
        out_scale: f32,
    ) -> Self {
        Self {
            name_template,
            layer_type,
            in_features_fn: Box::new(move |h| (h as f32 * in_scale) as usize),
            out_features_fn: Box::new(move |h| (h as f32 * out_scale) as usize),
        }
    }
}