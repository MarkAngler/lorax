//! Tokenizer implementations for text encoding

use anyhow::Result;
use tokenizers::{Tokenizer, Encoding};
use std::collections::HashMap;

/// Configuration for tokenizer
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    pub model_name: String,
    pub max_length: usize,
    pub padding: bool,
    pub truncation: bool,
    pub add_special_tokens: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            model_name: "bert-base-uncased".to_string(),
            max_length: 512,
            padding: true,
            truncation: true,
            add_special_tokens: true,
        }
    }
}

/// BERT tokenizer wrapper
pub struct BertTokenizer {
    tokenizer: Tokenizer,
    config: TokenizerConfig,
}

impl BertTokenizer {
    /// Create tokenizer from pretrained model
    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = TokenizerConfig {
            model_name: model_name.to_string(),
            ..Default::default()
        };
        
        // TODO: Load actual tokenizer from HuggingFace
        // For now, create a placeholder
        let tokenizer = Tokenizer::from_file("tokenizer.json")
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        Ok(Self { tokenizer, config })
    }
    
    /// Encode text to tokens
    pub fn encode(&self, text: &str) -> Result<Encoding> {
        let encoding = self.tokenizer
            .encode(text, self.config.add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
        
        Ok(encoding)
    }
    
    /// Encode batch of texts
    pub fn encode_batch(&self, texts: &[String]) -> Result<Vec<Encoding>> {
        let encodings = self.tokenizer
            .encode_batch(texts.iter().map(|s| s.as_str()).collect(), self.config.add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Batch encoding failed: {}", e))?;
        
        Ok(encodings)
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }
    
    /// Get configuration
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }
}

/// Token IDs and attention masks
#[derive(Debug, Clone)]
pub struct TokenizedInput {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub token_type_ids: Option<Vec<u32>>,
}

impl TokenizedInput {
    /// Create from encoding
    pub fn from_encoding(encoding: &Encoding) -> Self {
        Self {
            input_ids: encoding.get_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().to_vec(),
            token_type_ids: encoding.get_type_ids().map(|ids| ids.to_vec()),
        }
    }
    
    /// Pad to target length
    pub fn pad_to_length(&mut self, target_length: usize, pad_token_id: u32) {
        while self.input_ids.len() < target_length {
            self.input_ids.push(pad_token_id);
            self.attention_mask.push(0);
            if let Some(ref mut type_ids) = self.token_type_ids {
                type_ids.push(0);
            }
        }
    }
    
    /// Truncate to target length
    pub fn truncate_to_length(&mut self, target_length: usize) {
        if self.input_ids.len() > target_length {
            self.input_ids.truncate(target_length);
            self.attention_mask.truncate(target_length);
            if let Some(ref mut type_ids) = self.token_type_ids {
                type_ids.truncate(target_length);
            }
        }
    }
}