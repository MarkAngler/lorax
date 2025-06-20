//! Direct inference module
//!
//! This module provides functionality for direct inference with base models + adapters,
//! enabling immediate testing and deployment of T2L-generated adapters.

pub mod generator;
pub mod tokenizer;
pub mod sampling;

use crate::lora::LoraParameters;
use anyhow::Result;
use candle_core::Device;
use std::path::Path;

/// Generation request parameters
#[derive(Debug)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub stream: bool,
    pub system: Option<String>,
}

/// Generation response
#[derive(Debug)]
pub struct GenerationResponse {
    pub text: String,
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
    pub usage: TokenUsage,
}

/// Reason for generation completion
#[derive(Debug)]
pub enum FinishReason {
    MaxTokens,
    EosToken,
    StopSequence,
}

/// Token usage statistics
#[derive(Debug)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Output format for inference results
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
    Markdown,
}

/// Inference engine for running generation with adapted models
pub struct InferenceEngine {
    model: generator::AdaptedModel,
    tokenizer: Box<dyn tokenizer::Tokenizer>,
    generator: generator::TextGenerator,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub async fn new(
        base_model_path: &str,
        adapter: &LoraParameters,
        device: &Device,
    ) -> Result<Self> {
        tracing::info!("Initializing inference engine with model: {}", base_model_path);
        
        // 1. Load and adapt model
        let mut base_model = crate::apply::loader::load_base_model(base_model_path, device).await?;
        base_model.apply_lora(adapter)?;
        let model = generator::AdaptedModel::new(base_model);

        // 2. Load tokenizer
        let tokenizer = tokenizer::load_tokenizer(base_model_path).await?;

        // 3. Create generator
        let generator = generator::TextGenerator::new(device.clone());

        Ok(Self {
            model,
            tokenizer,
            generator,
        })
    }

    /// Generate text response
    pub async fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationResponse> {
        tracing::info!("Generating response for prompt: {}", request.prompt);
        
        // 1. Tokenize input
        let input_tokens = self.tokenizer.encode(&request.prompt, true)?;
        
        // 2. Generate tokens
        let output_tokens = if request.stream {
            self.generate_streaming(&input_tokens, request).await?
        } else {
            self.generate_batch(&input_tokens, request).await?
        };

        // 3. Decode output
        let output_text = self.tokenizer.decode(&output_tokens, true)?;

        Ok(GenerationResponse {
            text: output_text,
            tokens: output_tokens,
            finish_reason: FinishReason::MaxTokens,
            usage: TokenUsage {
                prompt_tokens: input_tokens.len(),
                completion_tokens: output_tokens.len(),
                total_tokens: input_tokens.len() + output_tokens.len(),
            },
        })
    }

    async fn generate_batch(
        &mut self,
        input_tokens: &[u32],
        request: &GenerationRequest,
    ) -> Result<Vec<u32>> {
        let mut generated_tokens = Vec::new();
        let mut current_tokens = input_tokens.to_vec();

        for _ in 0..request.max_tokens {
            // Forward pass
            let logits = self.model.forward(&current_tokens)?;
            
            // Sample next token
            let next_token = self.generator.sample(
                &logits,
                request.temperature,
                request.top_p,
            )?;
            
            // Check for EOS
            if self.tokenizer.is_eos_token(next_token) {
                break;
            }
            
            generated_tokens.push(next_token);
            current_tokens.push(next_token);
        }

        Ok(generated_tokens)
    }

    async fn generate_streaming(
        &mut self,
        input_tokens: &[u32],
        request: &GenerationRequest,
    ) -> Result<Vec<u32>> {
        let mut generated_tokens = Vec::new();
        let mut current_tokens = input_tokens.to_vec();

        for _ in 0..request.max_tokens {
            // Forward pass
            let logits = self.model.forward(&current_tokens)?;
            
            // Sample next token
            let next_token = self.generator.sample(
                &logits,
                request.temperature,
                request.top_p,
            )?;
            
            // Stream partial decode
            if let Ok(partial_text) = self.tokenizer.decode(&[next_token], false) {
                print!("{}", partial_text);
                use std::io::Write;
                std::io::stdout().flush()?;
            }
            
            // Check for EOS
            if self.tokenizer.is_eos_token(next_token) {
                break;
            }
            
            generated_tokens.push(next_token);
            current_tokens.push(next_token);
        }

        println!(); // New line after streaming
        Ok(generated_tokens)
    }
}