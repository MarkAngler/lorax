//! LoRA configuration types and utilities

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main LoRA configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoraConfig {
    /// LoRA rank (size of the low-rank matrices)
    pub rank: usize,
    /// LoRA alpha (scaling factor)
    pub alpha: f32,
    /// Dropout probability for LoRA layers
    pub dropout: f32,
    /// Target modules to apply LoRA to
    pub target_modules: Vec<String>,
    /// Bias configuration
    pub bias: BiasType,
    /// Task type for specialized configurations
    pub task_type: Option<TaskType>,
    /// Fan-in-fan-out configuration
    pub fan_in_fan_out: bool,
    /// Whether to merge weights for inference
    pub merge_weights: bool,
}

/// Types of bias handling in LoRA
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BiasType {
    /// No bias adaptation
    None,
    /// Adapt all bias terms
    All,
    /// Only adapt LoRA bias terms
    LoraOnly,
}

/// Task types for specialized LoRA configurations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskType {
    /// Text generation tasks
    Generation,
    /// Classification tasks
    Classification,
    /// Question answering
    QuestionAnswering,
    /// Summarization
    Summarization,
    /// Translation
    Translation,
    /// Code generation
    CodeGeneration,
    /// Mathematical reasoning
    MathReasoning,
    /// General purpose
    General,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            bias: BiasType::None,
            task_type: Some(TaskType::General),
            fan_in_fan_out: false,
            merge_weights: false,
        }
    }
}

impl LoraConfig {
    /// Create configuration optimized for a specific task type
    pub fn for_task(task_type: TaskType) -> Self {
        let mut config = Self::default();
        config.task_type = Some(task_type.clone());
        
        // Adjust parameters based on task type
        match task_type {
            TaskType::Generation => {
                config.rank = 16;
                config.alpha = 32.0;
                config.dropout = 0.05;
            }
            TaskType::Classification => {
                config.rank = 8;
                config.alpha = 16.0;
                config.dropout = 0.1;
                config.target_modules = vec![
                    "q_proj".to_string(),
                    "v_proj".to_string(),
                    "o_proj".to_string(),
                ];
            }
            TaskType::QuestionAnswering => {
                config.rank = 32;
                config.alpha = 64.0;
                config.dropout = 0.05;
            }
            TaskType::Summarization => {
                config.rank = 16;
                config.alpha = 32.0;
                config.dropout = 0.1;
            }
            TaskType::Translation => {
                config.rank = 32;
                config.alpha = 64.0;
                config.dropout = 0.05;
                config.target_modules.extend(vec![
                    "k_proj".to_string(),
                    "v_proj".to_string(),
                ]);
            }
            TaskType::CodeGeneration => {
                config.rank = 64;
                config.alpha = 128.0;
                config.dropout = 0.05;
            }
            TaskType::MathReasoning => {
                config.rank = 64;
                config.alpha = 128.0;
                config.dropout = 0.05;
                config.bias = BiasType::All;
            }
            TaskType::General => {
                // Use default values
            }
        }
        
        config
    }
    
    /// Create configuration for specific model architecture
    pub fn for_architecture(architecture: &str) -> Self {
        let mut config = Self::default();
        
        match architecture.to_lowercase().as_str() {
            "llama" | "llama2" | "llama3" => {
                config.target_modules = vec![
                    "q_proj".to_string(),
                    "k_proj".to_string(),
                    "v_proj".to_string(),
                    "o_proj".to_string(),
                    "gate_proj".to_string(),
                    "up_proj".to_string(),
                    "down_proj".to_string(),
                ];
            }
            "mistral" => {
                config.target_modules = vec![
                    "q_proj".to_string(),
                    "k_proj".to_string(),
                    "v_proj".to_string(),
                    "o_proj".to_string(),
                    "gate_proj".to_string(),
                    "up_proj".to_string(),
                    "down_proj".to_string(),
                ];
                config.rank = 32; // Mistral often benefits from higher rank
            }
            "gemma" => {
                config.target_modules = vec![
                    "q_proj".to_string(),
                    "k_proj".to_string(),
                    "v_proj".to_string(),
                    "o_proj".to_string(),
                    "gate_proj".to_string(),
                    "up_proj".to_string(),
                    "down_proj".to_string(),
                ];
            }
            "phi" => {
                config.target_modules = vec![
                    "q_proj".to_string(),
                    "k_proj".to_string(),
                    "v_proj".to_string(),
                    "dense".to_string(),
                    "fc1".to_string(),
                    "fc2".to_string(),
                ];
            }
            "qwen" => {
                config.target_modules = vec![
                    "c_attn".to_string(),
                    "c_proj".to_string(),
                    "w1".to_string(),
                    "w2".to_string(),
                ];
            }
            _ => {
                // Use default LLaMA-style configuration
            }
        }
        
        config
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), crate::error::Error> {
        if self.rank == 0 {
            return Err(crate::error::Error::InvalidInput("LoRA rank must be greater than 0".to_string()));
        }
        
        if self.rank > 512 {
            return Err(crate::error::Error::InvalidInput("LoRA rank should not exceed 512 for efficiency".to_string()));
        }
        
        if self.alpha <= 0.0 {
            return Err(crate::error::Error::InvalidInput("LoRA alpha must be positive".to_string()));
        }
        
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(crate::error::Error::InvalidInput("Dropout must be between 0.0 and 1.0".to_string()));
        }
        
        if self.target_modules.is_empty() {
            return Err(crate::error::Error::InvalidInput("At least one target module must be specified".to_string()));
        }
        
        // Check for duplicate target modules
        let mut unique_modules = std::collections::HashSet::new();
        for module in &self.target_modules {
            if !unique_modules.insert(module) {
                return Err(crate::error::Error::InvalidInput(format!("Duplicate target module: {}", module)));
            }
        }
        
        Ok(())
    }
    
    /// Get effective scaling factor (alpha / rank)
    pub fn scaling_factor(&self) -> f32 {
        self.alpha / self.rank as f32
    }
    
    /// Get estimated parameter overhead compared to full fine-tuning
    pub fn parameter_efficiency(&self, model_params: usize, target_layers: usize) -> f32 {
        // Estimate LoRA parameters
        let avg_layer_size = model_params / target_layers;
        let lora_params_per_layer = 2 * self.rank * (avg_layer_size as f32).sqrt() as usize;
        let total_lora_params = lora_params_per_layer * self.target_modules.len();
        
        total_lora_params as f32 / model_params as f32
    }
    
    /// Create a copy with different rank
    pub fn with_rank(&self, rank: usize) -> Self {
        let mut config = self.clone();
        config.rank = rank;
        config
    }
    
    /// Create a copy with different alpha
    pub fn with_alpha(&self, alpha: f32) -> Self {
        let mut config = self.clone();
        config.alpha = alpha;
        config
    }
    
    /// Create a copy with different target modules
    pub fn with_target_modules(&self, modules: Vec<String>) -> Self {
        let mut config = self.clone();
        config.target_modules = modules;
        config
    }
    
    /// Add target module
    pub fn add_target_module(&mut self, module: String) {
        if !self.target_modules.contains(&module) {
            self.target_modules.push(module);
        }
    }
    
    /// Remove target module
    pub fn remove_target_module(&mut self, module: &str) {
        self.target_modules.retain(|m| m != module);
    }
    
    /// Get configuration summary
    pub fn summary(&self) -> ConfigSummary {
        ConfigSummary {
            rank: self.rank,
            alpha: self.alpha,
            scaling_factor: self.scaling_factor(),
            dropout: self.dropout,
            num_target_modules: self.target_modules.len(),
            bias_type: self.bias,
            task_type: self.task_type.clone(),
        }
    }
}

/// Summary of LoRA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSummary {
    pub rank: usize,
    pub alpha: f32,
    pub scaling_factor: f32,
    pub dropout: f32,
    pub num_target_modules: usize,
    pub bias_type: BiasType,
    pub task_type: Option<TaskType>,
}

/// Preset configurations for common use cases
pub struct LoraPresets;

impl LoraPresets {
    /// Ultra low-rank configuration for maximum efficiency
    pub fn ultra_efficient() -> LoraConfig {
        LoraConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.1,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            ..Default::default()
        }
    }
    
    /// Balanced configuration for general use
    pub fn balanced() -> LoraConfig {
        LoraConfig::default()
    }
    
    /// High-capacity configuration for complex tasks
    pub fn high_capacity() -> LoraConfig {
        LoraConfig {
            rank: 64,
            alpha: 128.0,
            dropout: 0.05,
            bias: BiasType::All,
            ..Default::default()
        }
    }
    
    /// Configuration for fine-grained control
    pub fn fine_grained() -> LoraConfig {
        LoraConfig {
            rank: 32,
            alpha: 64.0,
            dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
                "embed_tokens".to_string(),
                "lm_head".to_string(),
            ],
            bias: BiasType::All,
            ..Default::default()
        }
    }
}

/// Builder for creating LoRA configurations
pub struct LoraConfigBuilder {
    config: LoraConfig,
}

impl LoraConfigBuilder {
    /// Create new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: LoraConfig::default(),
        }
    }
    
    /// Set rank
    pub fn rank(mut self, rank: usize) -> Self {
        self.config.rank = rank;
        self
    }
    
    /// Set alpha
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.config.alpha = alpha;
        self
    }
    
    /// Set dropout
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.config.dropout = dropout;
        self
    }
    
    /// Set target modules
    pub fn target_modules(mut self, modules: Vec<String>) -> Self {
        self.config.target_modules = modules;
        self
    }
    
    /// Set bias type
    pub fn bias(mut self, bias: BiasType) -> Self {
        self.config.bias = bias;
        self
    }
    
    /// Set task type
    pub fn task_type(mut self, task_type: TaskType) -> Self {
        self.config.task_type = Some(task_type);
        self
    }
    
    /// Enable weight merging
    pub fn merge_weights(mut self, merge: bool) -> Self {
        self.config.merge_weights = merge;
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> Result<LoraConfig, crate::error::Error> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for LoraConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}