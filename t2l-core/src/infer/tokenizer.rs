//! Tokenizer integration for T2L inference
//!
//! This module provides tokenizer functionality for text generation, including:
//! - Tokenizer trait for different implementations
//! - HuggingFace tokenizer integration
//! - Automatic tokenizer loading from model paths
//! - Special token handling and EOS detection
//! - Batch encoding/decoding support

use anyhow::{anyhow, Context, Result};
use hf_hub::api::tokio::Api;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::{EncodeInput, Encoding, InputSequence};
use tracing::{debug, info, warn};

/// Tokenizer trait for different implementations
pub trait Tokenizer: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    
    /// Decode token IDs to text
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;
    
    /// Batch encode multiple texts
    fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<u32>>>;
    
    /// Batch decode multiple token sequences
    fn decode_batch(&self, token_ids: &[Vec<u32>], skip_special_tokens: bool) -> Result<Vec<String>>;
    
    /// Check if a token is the EOS token
    fn is_eos_token(&self, token_id: u32) -> bool;
    
    /// Get the EOS token ID
    fn eos_token_id(&self) -> Option<u32>;
    
    /// Get the BOS token ID
    fn bos_token_id(&self) -> Option<u32>;
    
    /// Get the PAD token ID
    fn pad_token_id(&self) -> Option<u32>;
    
    /// Get vocabulary size
    fn vocab_size(&self) -> usize;
    
    /// Convert token ID to string
    fn id_to_token(&self, token_id: u32) -> Option<String>;
    
    /// Convert token string to ID
    fn token_to_id(&self, token: &str) -> Option<u32>;
    
    /// Get all special tokens
    fn special_tokens(&self) -> Vec<String>;
    
    /// Get tokenizer name/type
    fn name(&self) -> &str;
}

/// HuggingFace tokenizer implementation
pub struct HuggingFaceTokenizer {
    tokenizer: HFTokenizer,
    special_tokens: SpecialTokens,
    name: String,
}

/// Special tokens configuration
#[derive(Debug, Clone)]
struct SpecialTokens {
    eos_token_id: Option<u32>,
    bos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
    unk_token_id: Option<u32>,
    sep_token_id: Option<u32>,
    cls_token_id: Option<u32>,
    mask_token_id: Option<u32>,
    additional_special_tokens: Vec<(String, u32)>,
}

impl HuggingFaceTokenizer {
    /// Create a new HuggingFace tokenizer
    pub fn new(tokenizer: HFTokenizer, name: String) -> Result<Self> {
        let special_tokens = Self::extract_special_tokens(&tokenizer)?;
        
        Ok(Self {
            tokenizer,
            special_tokens,
            name,
        })
    }
    
    /// Load tokenizer from file
    pub fn from_file<P: AsRef<Path>>(path: P, name: String) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
        
        Self::new(tokenizer, name)
    }
    
    /// Load tokenizer from pretrained model
    pub async fn from_pretrained(model_id: &str) -> Result<Self> {
        info!("Loading tokenizer from HuggingFace Hub: {}", model_id);
        
        let api = Api::new()?;
        let repo = api.model(model_id.to_string());
        
        // Try to download tokenizer.json
        let tokenizer_path = repo.get("tokenizer.json").await
            .context("Failed to download tokenizer.json from HuggingFace Hub")?;
        
        Self::from_file(tokenizer_path, model_id.to_string())
    }
    
    /// Extract special tokens from tokenizer
    fn extract_special_tokens(tokenizer: &HFTokenizer) -> Result<SpecialTokens> {
        let mut special_tokens = SpecialTokens {
            eos_token_id: None,
            bos_token_id: None,
            pad_token_id: None,
            unk_token_id: None,
            sep_token_id: None,
            cls_token_id: None,
            mask_token_id: None,
            additional_special_tokens: Vec::new(),
        };
        
        // Common special token patterns
        let special_token_patterns = vec![
            ("</s>", "eos"),
            ("<s>", "bos"),
            ("<pad>", "pad"),
            ("<unk>", "unk"),
            ("[SEP]", "sep"),
            ("[CLS]", "cls"),
            ("[MASK]", "mask"),
            ("<|endoftext|>", "eos"),
            ("<|startoftext|>", "bos"),
        ];
        
        // Try to find special tokens
        for (token_str, token_type) in special_token_patterns {
            if let Some(token_id) = tokenizer.token_to_id(token_str) {
                match token_type {
                    "eos" => special_tokens.eos_token_id = Some(token_id),
                    "bos" => special_tokens.bos_token_id = Some(token_id),
                    "pad" => special_tokens.pad_token_id = Some(token_id),
                    "unk" => special_tokens.unk_token_id = Some(token_id),
                    "sep" => special_tokens.sep_token_id = Some(token_id),
                    "cls" => special_tokens.cls_token_id = Some(token_id),
                    "mask" => special_tokens.mask_token_id = Some(token_id),
                    _ => {}
                }
            }
        }
        
        // If no EOS token found, try common IDs
        if special_tokens.eos_token_id.is_none() {
            // Common EOS token IDs
            for &eos_id in &[0, 1, 2, 50256] {
                if let Some(token) = tokenizer.id_to_token(eos_id) {
                    if token.contains("eos") || token.contains("</s>") || token.contains("endoftext") {
                        special_tokens.eos_token_id = Some(eos_id);
                        break;
                    }
                }
            }
        }
        
        debug!("Detected special tokens: {:?}", special_tokens);
        
        Ok(special_tokens)
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        
        Ok(encoding.get_ids().to_vec())
    }
    
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| anyhow!("Decoding failed: {}", e))
    }
    
    fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<u32>>> {
        let encodings = self.tokenizer
            .encode_batch(texts.to_vec(), add_special_tokens)
            .map_err(|e| anyhow!("Batch tokenization failed: {}", e))?;
        
        Ok(encodings.into_iter()
            .map(|enc| enc.get_ids().to_vec())
            .collect())
    }
    
    fn decode_batch(&self, token_ids: &[Vec<u32>], skip_special_tokens: bool) -> Result<Vec<String>> {
        token_ids.iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }
    
    fn is_eos_token(&self, token_id: u32) -> bool {
        self.special_tokens.eos_token_id == Some(token_id)
    }
    
    fn eos_token_id(&self) -> Option<u32> {
        self.special_tokens.eos_token_id
    }
    
    fn bos_token_id(&self) -> Option<u32> {
        self.special_tokens.bos_token_id
    }
    
    fn pad_token_id(&self) -> Option<u32> {
        self.special_tokens.pad_token_id
    }
    
    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
    
    fn id_to_token(&self, token_id: u32) -> Option<String> {
        self.tokenizer.id_to_token(token_id)
    }
    
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }
    
    fn special_tokens(&self) -> Vec<String> {
        let mut tokens = Vec::new();
        
        if let Some(id) = self.special_tokens.eos_token_id {
            if let Some(token) = self.id_to_token(id) {
                tokens.push(token);
            }
        }
        
        if let Some(id) = self.special_tokens.bos_token_id {
            if let Some(token) = self.id_to_token(id) {
                tokens.push(token);
            }
        }
        
        if let Some(id) = self.special_tokens.pad_token_id {
            if let Some(token) = self.id_to_token(id) {
                tokens.push(token);
            }
        }
        
        for (token, _) in &self.special_tokens.additional_special_tokens {
            tokens.push(token.clone());
        }
        
        tokens
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Tokenizer factory for automatic loading
pub struct TokenizerFactory;

impl TokenizerFactory {
    /// Detect tokenizer type from model config
    pub async fn detect_tokenizer_type(model_path: &str) -> Result<String> {
        // Check if it's a local path
        let path = Path::new(model_path);
        if path.exists() && path.is_dir() {
            return Self::detect_from_local_path(path).await;
        }
        
        // Otherwise, assume it's a HuggingFace model ID
        Self::detect_from_huggingface(model_path).await
    }
    
    /// Detect tokenizer from local model directory
    async fn detect_from_local_path(path: &Path) -> Result<String> {
        // Check for tokenizer config files
        let tokenizer_config_path = path.join("tokenizer_config.json");
        if tokenizer_config_path.exists() {
            let config_str = tokio::fs::read_to_string(&tokenizer_config_path).await?;
            let config: serde_json::Value = serde_json::from_str(&config_str)?;
            
            // Extract tokenizer class
            if let Some(tokenizer_class) = config.get("tokenizer_class").and_then(|v| v.as_str()) {
                return Ok(tokenizer_class.to_string());
            }
        }
        
        // Check for tokenizer.json (indicates HuggingFace Fast tokenizer)
        if path.join("tokenizer.json").exists() {
            return Ok("PreTrainedTokenizerFast".to_string());
        }
        
        // Default to generic tokenizer
        Ok("AutoTokenizer".to_string())
    }
    
    /// Detect tokenizer from HuggingFace model ID
    async fn detect_from_huggingface(model_id: &str) -> Result<String> {
        let api = Api::new()?;
        let repo = api.model(model_id.to_string());
        
        // Try to download tokenizer_config.json
        if let Ok(config_path) = repo.get("tokenizer_config.json").await {
            let config_str = tokio::fs::read_to_string(&config_path).await?;
            let config: serde_json::Value = serde_json::from_str(&config_str)?;
            
            if let Some(tokenizer_class) = config.get("tokenizer_class").and_then(|v| v.as_str()) {
                return Ok(tokenizer_class.to_string());
            }
        }
        
        // Default to AutoTokenizer
        Ok("AutoTokenizer".to_string())
    }
}

/// Load tokenizer automatically from model path
pub async fn load_tokenizer(model_path: &str) -> Result<Box<dyn Tokenizer>> {
    info!("Loading tokenizer for model: {}", model_path);
    
    // Check if it's a local path
    let path = Path::new(model_path);
    if path.exists() && path.is_dir() {
        // Try to load from local tokenizer.json
        let tokenizer_json_path = path.join("tokenizer.json");
        if tokenizer_json_path.exists() {
            info!("Loading tokenizer from local file: {}", tokenizer_json_path.display());
            let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_json_path, model_path.to_string())?;
            return Ok(Box::new(tokenizer));
        }
        
        // Try other tokenizer formats
        let tokenizer_files = vec![
            "tokenizer.model",
            "spiece.model",
            "vocab.json",
            "merges.txt",
        ];
        
        for file in tokenizer_files {
            if path.join(file).exists() {
                warn!("Found {} but it requires special handling. Using default tokenizer.", file);
                // In a full implementation, we would handle different tokenizer formats
                // For now, we'll return an error
                return Err(anyhow!("Tokenizer format {} not yet supported. Please provide tokenizer.json", file));
            }
        }
        
        return Err(anyhow!("No supported tokenizer files found in {}", model_path));
    }
    
    // Load from HuggingFace Hub
    let tokenizer = HuggingFaceTokenizer::from_pretrained(model_path).await?;
    Ok(Box::new(tokenizer))
}

/// Simple tokenizer for testing
#[cfg(test)]
pub struct MockTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    eos_token_id: u32,
}

#[cfg(test)]
impl MockTokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();
        
        // Simple vocabulary
        let tokens = vec![
            ("<pad>", 0),
            ("<s>", 1),
            ("</s>", 2),
            ("<unk>", 3),
            ("hello", 4),
            ("world", 5),
            ("the", 6),
            ("a", 7),
            ("an", 8),
            (".", 9),
            (",", 10),
            ("!", 11),
            ("?", 12),
        ];
        
        for (token, id) in tokens {
            vocab.insert(token.to_string(), id);
            reverse_vocab.insert(id, token.to_string());
        }
        
        Self {
            vocab,
            reverse_vocab,
            eos_token_id: 2,
        }
    }
}

#[cfg(test)]
impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        
        if add_special_tokens {
            tokens.push(1); // BOS
        }
        
        // Simple word-based tokenization
        for word in text.to_lowercase().split_whitespace() {
            if let Some(&token_id) = self.vocab.get(word) {
                tokens.push(token_id);
            } else {
                tokens.push(3); // UNK
            }
        }
        
        if add_special_tokens {
            tokens.push(2); // EOS
        }
        
        Ok(tokens)
    }
    
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut words = Vec::new();
        
        for &token_id in token_ids {
            if let Some(token) = self.reverse_vocab.get(&token_id) {
                if skip_special_tokens && (token == "<s>" || token == "</s>" || token == "<pad>") {
                    continue;
                }
                words.push(token.clone());
            }
        }
        
        Ok(words.join(" "))
    }
    
    fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<u32>>> {
        texts.iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }
    
    fn decode_batch(&self, token_ids: &[Vec<u32>], skip_special_tokens: bool) -> Result<Vec<String>> {
        token_ids.iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }
    
    fn is_eos_token(&self, token_id: u32) -> bool {
        token_id == self.eos_token_id
    }
    
    fn eos_token_id(&self) -> Option<u32> {
        Some(self.eos_token_id)
    }
    
    fn bos_token_id(&self) -> Option<u32> {
        Some(1)
    }
    
    fn pad_token_id(&self) -> Option<u32> {
        Some(0)
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    fn id_to_token(&self, token_id: u32) -> Option<String> {
        self.reverse_vocab.get(&token_id).cloned()
    }
    
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }
    
    fn special_tokens(&self) -> Vec<String> {
        vec!["<pad>".to_string(), "<s>".to_string(), "</s>".to_string(), "<unk>".to_string()]
    }
    
    fn name(&self) -> &str {
        "MockTokenizer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mock_tokenizer() {
        let tokenizer = MockTokenizer::new();
        
        // Test encoding
        let text = "hello world";
        let tokens = tokenizer.encode(text, true).unwrap();
        assert_eq!(tokens, vec![1, 4, 5, 2]); // <s> hello world </s>
        
        let tokens_no_special = tokenizer.encode(text, false).unwrap();
        assert_eq!(tokens_no_special, vec![4, 5]); // hello world
        
        // Test decoding
        let decoded = tokenizer.decode(&tokens, false).unwrap();
        assert_eq!(decoded, "<s> hello world </s>");
        
        let decoded_skip_special = tokenizer.decode(&tokens, true).unwrap();
        assert_eq!(decoded_skip_special, "hello world");
        
        // Test EOS detection
        assert!(tokenizer.is_eos_token(2));
        assert!(!tokenizer.is_eos_token(1));
        
        // Test special tokens
        assert_eq!(tokenizer.eos_token_id(), Some(2));
        assert_eq!(tokenizer.bos_token_id(), Some(1));
        assert_eq!(tokenizer.pad_token_id(), Some(0));
    }
    
    #[test]
    fn test_batch_operations() {
        let tokenizer = MockTokenizer::new();
        
        let texts = vec!["hello", "world", "hello world"];
        let batch_tokens = tokenizer.encode_batch(&texts, true).unwrap();
        
        assert_eq!(batch_tokens.len(), 3);
        assert_eq!(batch_tokens[0], vec![1, 4, 2]); // <s> hello </s>
        assert_eq!(batch_tokens[1], vec![1, 5, 2]); // <s> world </s>
        assert_eq!(batch_tokens[2], vec![1, 4, 5, 2]); // <s> hello world </s>
        
        let decoded_batch = tokenizer.decode_batch(&batch_tokens, true).unwrap();
        assert_eq!(decoded_batch, vec!["hello", "world", "hello world"]);
    }
    
    #[test]
    fn test_unknown_tokens() {
        let tokenizer = MockTokenizer::new();
        
        let text = "hello unknown world";
        let tokens = tokenizer.encode(text, false).unwrap();
        assert_eq!(tokens, vec![4, 3, 5]); // hello <unk> world
        
        let decoded = tokenizer.decode(&tokens, false).unwrap();
        assert_eq!(decoded, "hello <unk> world");
    }
    
    #[test]
    fn test_vocab_operations() {
        let tokenizer = MockTokenizer::new();
        
        assert_eq!(tokenizer.vocab_size(), 13);
        assert_eq!(tokenizer.id_to_token(4), Some("hello".to_string()));
        assert_eq!(tokenizer.token_to_id("world"), Some(5));
        assert_eq!(tokenizer.token_to_id("nonexistent"), None);
        
        let special_tokens = tokenizer.special_tokens();
        assert!(special_tokens.contains(&"<pad>".to_string()));
        assert!(special_tokens.contains(&"</s>".to_string()));
    }
}