//! Task encoder module for converting text to embeddings

use async_trait::async_trait;
use std::sync::Arc;

use crate::config::EncoderConfig;
use crate::error::Result;

mod bert;
mod tokenizer;

pub use bert::BertEncoder;
pub use tokenizer::{BertTokenizer, TokenizerConfig};

/// Task embedding representation
#[derive(Debug, Clone)]
pub struct TaskEmbedding {
    /// Embedding vector
    pub vector: Vec<f32>,
    /// Embedding dimension
    pub dim: usize,
    /// Source text (optional)
    pub text: Option<String>,
}

impl TaskEmbedding {
    /// Create new task embedding
    pub fn new(vector: Vec<f32>) -> Self {
        let dim = vector.len();
        Self {
            vector,
            dim,
            text: None,
        }
    }
    
    /// Create with source text
    pub fn with_text(vector: Vec<f32>, text: String) -> Self {
        let dim = vector.len();
        Self {
            vector,
            dim,
            text: Some(text),
        }
    }
    
    /// Get vector reference
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }
    
    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Pooling strategies for sequence embeddings
#[derive(Debug, Clone, Copy)]
pub enum PoolingStrategy {
    /// Mean pooling over sequence
    Mean,
    /// Max pooling over sequence
    Max,
    /// Use [CLS] token embedding
    Cls,
    /// Weighted mean pooling
    WeightedMean,
}

/// Pooling layer for combining token embeddings
#[derive(Debug, Clone)]
pub struct PoolingLayer {
    strategy: PoolingStrategy,
}

impl PoolingLayer {
    /// Create new pooling layer
    pub fn new(strategy: PoolingStrategy) -> Self {
        Self { strategy }
    }
    
    /// Apply pooling to token embeddings
    pub fn pool(&self, embeddings: &[Vec<f32>], attention_mask: &[u32]) -> Vec<f32> {
        match self.strategy {
            PoolingStrategy::Mean => self.mean_pool(embeddings, attention_mask),
            PoolingStrategy::Max => self.max_pool(embeddings),
            PoolingStrategy::Cls => embeddings.first().unwrap_or(&vec![]).clone(),
            PoolingStrategy::WeightedMean => self.weighted_mean_pool(embeddings, attention_mask),
        }
    }
    
    fn mean_pool(&self, embeddings: &[Vec<f32>], attention_mask: &[u32]) -> Vec<f32> {
        if embeddings.is_empty() {
            return vec![];
        }
        
        let dim = embeddings[0].len();
        let mut result = vec![0.0; dim];
        let mut count = 0;
        
        for (i, embedding) in embeddings.iter().enumerate() {
            if i < attention_mask.len() && attention_mask[i] == 1 {
                for (j, &val) in embedding.iter().enumerate() {
                    result[j] += val;
                }
                count += 1;
            }
        }
        
        if count > 0 {
            for val in &mut result {
                *val /= count as f32;
            }
        }
        
        result
    }
    
    fn max_pool(&self, embeddings: &[Vec<f32>]) -> Vec<f32> {
        if embeddings.is_empty() {
            return vec![];
        }
        
        let dim = embeddings[0].len();
        let mut result = vec![f32::NEG_INFINITY; dim];
        
        for embedding in embeddings {
            for (j, &val) in embedding.iter().enumerate() {
                result[j] = result[j].max(val);
            }
        }
        
        result
    }
    
    fn weighted_mean_pool(&self, embeddings: &[Vec<f32>], attention_mask: &[u32]) -> Vec<f32> {
        // For now, just use regular mean pooling
        // TODO: Implement actual weighted pooling
        self.mean_pool(embeddings, attention_mask)
    }
}

/// Trait for task encoders
#[async_trait]
pub trait TaskEncoder: Send + Sync {
    /// Encode a single task description
    async fn encode(&self, text: &str) -> Result<Vec<f32>>;
    
    /// Encode multiple task descriptions (batched)
    async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    
    /// Get the output dimension
    fn output_dim(&self) -> usize;
}

/// Generic encoder interface
pub trait Encoder: Send + Sync {
    /// Initialize the encoder
    fn initialize(&mut self) -> Result<()>;
    
    /// Process input text
    fn process(&self, text: &str) -> Result<Vec<f32>>;
    
    /// Process batch of texts
    fn process_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// Create an encoder based on configuration
pub fn create_encoder(config: &EncoderConfig) -> Result<Arc<dyn TaskEncoder>> {
    match &config.encoder_type {
        crate::config::EncoderType::Bert => {
            let encoder = BertEncoder::new(config.clone())?;
            Ok(Arc::new(encoder))
        }
        crate::config::EncoderType::Roberta => {
            // TODO: Implement RoBERTa encoder
            Err(crate::error::Error::encoder("RoBERTa encoder not yet implemented"))
        }
        crate::config::EncoderType::Distilbert => {
            // TODO: Implement DistilBERT encoder
            Err(crate::error::Error::encoder("DistilBERT encoder not yet implemented"))
        }
        crate::config::EncoderType::Custom(name) => {
            Err(crate::error::Error::encoder(format!("Custom encoder '{}' not found", name)))
        }
    }
}