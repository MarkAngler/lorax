//! Embedding configuration and dimensions

use serde::{Deserialize, Serialize};

/// Embedding dimension configurations for different model sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingDimension {
    /// Small: 384 dimensions (e.g., MiniLM)
    Small,
    /// Base: 768 dimensions (e.g., BERT-base)
    Base,
    /// Large: 1024 dimensions (e.g., BERT-large, GTE-large)
    Large,
    /// XLarge: 1536 dimensions (e.g., specialized embeddings)
    XLarge,
    /// Custom dimension
    Custom(usize),
}

impl EmbeddingDimension {
    /// Convert to actual dimension size
    pub fn as_usize(&self) -> usize {
        match self {
            Self::Small => 384,
            Self::Base => 768,
            Self::Large => 1024,
            Self::XLarge => 1536,
            Self::Custom(dim) => *dim,
        }
    }
    
    /// Create from dimension size
    pub fn from_size(size: usize) -> Self {
        match size {
            384 => Self::Small,
            768 => Self::Base,
            1024 => Self::Large,
            1536 => Self::XLarge,
            _ => Self::Custom(size),
        }
    }
}

/// Configuration for embedding models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model variant to use
    pub model_variant: ModelVariant,
    /// Cache directory for models
    pub cache_dir: Option<String>,
    /// Whether to use quantized models
    pub use_quantized: bool,
    /// Batch size for inference
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_variant: ModelVariant::GTELarge,
            cache_dir: None,
            use_quantized: false,
            batch_size: 1,
        }
    }
}

/// Supported embedding model variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelVariant {
    /// Alibaba GTE-large-en-v1.5
    GTELarge,
    /// Sentence-BERT all-MiniLM-L6-v2
    MiniLM,
    /// BERT-base-uncased
    BertBase,
    /// BERT-large-uncased
    BertLarge,
    /// Custom model path
    Custom(String),
}

impl ModelVariant {
    /// Get the model identifier
    pub fn model_id(&self) -> &str {
        match self {
            Self::GTELarge => "Alibaba-NLP/gte-large-en-v1.5",
            Self::MiniLM => "sentence-transformers/all-MiniLM-L6-v2",
            Self::BertBase => "bert-base-uncased",
            Self::BertLarge => "bert-large-uncased",
            Self::Custom(path) => path,
        }
    }
    
    /// Get expected embedding dimension
    pub fn embedding_dim(&self) -> EmbeddingDimension {
        match self {
            Self::GTELarge => EmbeddingDimension::Large,
            Self::MiniLM => EmbeddingDimension::Small,
            Self::BertBase => EmbeddingDimension::Base,
            Self::BertLarge => EmbeddingDimension::Large,
            Self::Custom(_) => EmbeddingDimension::Base, // Default assumption
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_embedding_dimensions() {
        assert_eq!(EmbeddingDimension::Small.as_usize(), 384);
        assert_eq!(EmbeddingDimension::Base.as_usize(), 768);
        assert_eq!(EmbeddingDimension::Large.as_usize(), 1024);
        assert_eq!(EmbeddingDimension::XLarge.as_usize(), 1536);
        assert_eq!(EmbeddingDimension::Custom(2048).as_usize(), 2048);
    }
    
    #[test]
    fn test_dimension_from_size() {
        assert_eq!(EmbeddingDimension::from_size(768), EmbeddingDimension::Base);
        assert_eq!(EmbeddingDimension::from_size(1024), EmbeddingDimension::Large);
        assert!(matches!(EmbeddingDimension::from_size(512), EmbeddingDimension::Custom(512)));
    }
    
    #[test]
    fn test_model_variants() {
        let gte = ModelVariant::GTELarge;
        assert_eq!(gte.model_id(), "Alibaba-NLP/gte-large-en-v1.5");
        assert_eq!(gte.embedding_dim(), EmbeddingDimension::Large);
    }
}