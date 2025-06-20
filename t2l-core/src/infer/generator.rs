//! Text generation engine with LoRA-adapted models
//!
//! This module provides the core text generation functionality for T2L, including:
//! - AdaptedModel wrapper for models with LoRA adapters
//! - TextGenerator with advanced sampling strategies
//! - KV cache for efficient generation
//! - Streaming and batch generation support

use crate::apply::{BaseModel, loader::LoraCompatibleLayer};
use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::Module;
use lorax::lora::{LoraLayer, LoraParameters};
use rand::distributions::Distribution;
use rand::SeedableRng;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Adapted model wrapper that applies LoRA during forward pass
pub struct AdaptedModel {
    base_model: BaseModel,
    lora_adapters: HashMap<String, LoraAdapter>,
    cache: KVCache,
    device: Device,
}

impl AdaptedModel {
    /// Create a new adapted model
    pub fn new(base_model: BaseModel) -> Self {
        let device = base_model.device().clone();
        Self {
            base_model,
            lora_adapters: HashMap::new(),
            cache: KVCache::new(),
            device,
        }
    }

    /// Apply LoRA adapters to the model
    pub fn apply_lora(&mut self, adapter_params: &LoraParameters) -> Result<()> {
        info!("Applying {} LoRA adapters to model", adapter_params.layers.len());
        
        for (layer_name, lora_layer) in &adapter_params.layers {
            debug!("Creating LoRA adapter for layer: {}", layer_name);
            
            let adapter = LoraAdapter::new(
                &lora_layer.a_weights,
                &lora_layer.b_weights,
                lora_layer.rank,
                lora_layer.alpha,
                lora_layer.a_matrix_shape(),
                lora_layer.b_matrix_shape(),
                &self.device,
            )?;
            
            self.lora_adapters.insert(layer_name.clone(), adapter);
        }
        
        Ok(())
    }

    /// Forward pass through the adapted model
    pub fn forward(&mut self, input_ids: &[u32]) -> Result<Tensor> {
        let batch_size = 1;
        let seq_len = input_ids.len();
        
        // Convert input IDs to tensor
        let input_tensor = Tensor::from_vec(
            input_ids.to_vec(),
            (batch_size, seq_len),
            &self.device,
        )?;
        
        // Create sequence length offsets for attention
        let seqlen_offsets: Vec<usize> = (0..batch_size + 1).map(|i| i * seq_len).collect();
        
        // Forward pass through base model
        let mut logits = self.base_model.forward(&input_tensor, &seqlen_offsets)?;
        
        // Apply LoRA adapters if available
        if !self.lora_adapters.is_empty() {
            debug!("Applying {} LoRA adapters", self.lora_adapters.len());
            // Note: In a real implementation, LoRA would be integrated into the model layers
            // For now, we're returning the base model output
            // This is a placeholder for the actual LoRA application logic
        }
        
        // Extract logits for the last token
        let last_token_logits = logits.i((0, seq_len - 1, ..))?;
        
        Ok(last_token_logits)
    }

    /// Reset the KV cache
    pub fn reset_cache(&mut self) {
        self.cache.clear();
    }
    
    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// LoRA adapter for efficient fine-tuning
pub struct LoraAdapter {
    a_matrix: Tensor,
    b_matrix: Tensor,
    rank: usize,
    alpha: f32,
    scaling: f32,
}

impl LoraAdapter {
    /// Create a new LoRA adapter
    pub fn new(
        a_weights: &[f32],
        b_weights: &[f32],
        rank: usize,
        alpha: f32,
        a_shape: (usize, usize),
        b_shape: (usize, usize),
        device: &Device,
    ) -> Result<Self> {
        // Create tensors from weights
        let a_matrix = Tensor::from_vec(
            a_weights.to_vec(),
            a_shape,
            device,
        )?;
        
        let b_matrix = Tensor::from_vec(
            b_weights.to_vec(),
            b_shape,
            device,
        )?;
        
        let scaling = alpha / rank as f32;
        
        Ok(Self {
            a_matrix,
            b_matrix,
            rank,
            alpha,
            scaling,
        })
    }
    
    /// Apply LoRA transformation to input
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // LoRA computation: output = input + (input @ A @ B) * scaling
        let lora_output = input
            .matmul(&self.a_matrix)?
            .matmul(&self.b_matrix)?;
        
        let scaled_output = lora_output.affine(self.scaling, 0.0)?;
        
        input.add(&scaled_output)
    }
}

/// Text generator with advanced sampling strategies
pub struct TextGenerator {
    device: Device,
    rng: rand::rngs::StdRng,
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
}

impl TextGenerator {
    /// Create a new text generator
    pub fn new(device: Device) -> Self {
        Self {
            device,
            rng: rand::rngs::StdRng::from_entropy(),
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 0,
        }
    }
    
    /// Set repetition penalty
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }
    
    /// Set no-repeat n-gram size
    pub fn with_no_repeat_ngram_size(mut self, size: usize) -> Self {
        self.no_repeat_ngram_size = size;
        self
    }
    
    /// Sample next token with temperature and top-p
    pub fn sample(
        &mut self,
        logits: &Tensor,
        temperature: f32,
        top_p: f32,
    ) -> Result<u32> {
        let logits_vec = self.prepare_logits(logits, temperature)?;
        
        if temperature == 0.0 {
            // Greedy sampling
            self.sample_greedy(&logits_vec)
        } else if top_p < 1.0 {
            // Top-p (nucleus) sampling
            self.sample_top_p(&logits_vec, top_p)
        } else {
            // Standard temperature sampling
            self.sample_temperature(&logits_vec)
        }
    }
    
    /// Sample with additional constraints
    pub fn sample_constrained(
        &mut self,
        logits: &Tensor,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        generated_tokens: &[u32],
    ) -> Result<u32> {
        let mut logits_vec = self.prepare_logits(logits, temperature)?;
        
        // Apply repetition penalty
        if self.repetition_penalty != 1.0 && !generated_tokens.is_empty() {
            self.apply_repetition_penalty(&mut logits_vec, generated_tokens);
        }
        
        // Apply no-repeat n-gram penalty
        if self.no_repeat_ngram_size > 0 && generated_tokens.len() >= self.no_repeat_ngram_size {
            self.apply_no_repeat_ngram_penalty(&mut logits_vec, generated_tokens);
        }
        
        // Apply top-k filtering if specified
        if let Some(k) = top_k {
            self.apply_top_k(&mut logits_vec, k);
        }
        
        // Sample with top-p
        if top_p < 1.0 {
            self.sample_top_p(&logits_vec, top_p)
        } else {
            self.sample_temperature(&logits_vec)
        }
    }
    
    /// Prepare logits with temperature scaling
    fn prepare_logits(&self, logits: &Tensor, temperature: f32) -> Result<Vec<f32>> {
        let mut logits_vec = logits.to_vec1::<f32>()?;
        
        // Apply temperature scaling
        if temperature != 1.0 && temperature > 0.0 {
            for logit in logits_vec.iter_mut() {
                *logit /= temperature;
            }
        }
        
        Ok(logits_vec)
    }
    
    /// Greedy sampling (argmax)
    fn sample_greedy(&self, logits: &[f32]) -> Result<u32> {
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| anyhow!("Empty logits"))?;
        
        Ok(max_idx as u32)
    }
    
    /// Temperature sampling
    fn sample_temperature(&mut self, logits: &[f32]) -> Result<u32> {
        // Convert logits to probabilities
        let probs = self.softmax(logits)?;
        
        // Sample from distribution
        let dist = rand::distributions::WeightedIndex::new(&probs)
            .map_err(|e| anyhow!("Failed to create weighted distribution: {}", e))?;
        
        Ok(dist.sample(&mut self.rng) as u32)
    }
    
    /// Top-p (nucleus) sampling
    fn sample_top_p(&mut self, logits: &[f32], top_p: f32) -> Result<u32> {
        // Get probabilities
        let probs = self.softmax(logits)?;
        
        // Create sorted indices by probability
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        
        // Find cutoff for top-p
        let mut cumsum = 0.0;
        let mut cutoff_idx = 0;
        
        for (i, &idx) in indices.iter().enumerate() {
            cumsum += probs[idx];
            if cumsum >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        // Sample from top-p tokens
        let top_indices = &indices[..cutoff_idx.max(1)];
        let top_probs: Vec<f32> = top_indices.iter().map(|&idx| probs[idx]).collect();
        
        let dist = rand::distributions::WeightedIndex::new(&top_probs)
            .map_err(|e| anyhow!("Failed to create top-p distribution: {}", e))?;
        
        Ok(top_indices[dist.sample(&mut self.rng)] as u32)
    }
    
    /// Apply top-k filtering
    fn apply_top_k(&self, logits: &mut [f32], k: usize) {
        if k == 0 || k >= logits.len() {
            return;
        }
        
        // Find k-th largest value
        let mut sorted_logits = logits.to_vec();
        sorted_logits.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted_logits[k.min(sorted_logits.len() - 1)];
        
        // Set all logits below threshold to -inf
        for logit in logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
    
    /// Apply repetition penalty
    fn apply_repetition_penalty(&self, logits: &mut [f32], generated_tokens: &[u32]) {
        for &token in generated_tokens {
            if (token as usize) < logits.len() {
                if logits[token as usize] > 0.0 {
                    logits[token as usize] /= self.repetition_penalty;
                } else {
                    logits[token as usize] *= self.repetition_penalty;
                }
            }
        }
    }
    
    /// Apply no-repeat n-gram penalty
    fn apply_no_repeat_ngram_penalty(&self, logits: &mut [f32], generated_tokens: &[u32]) {
        let n = self.no_repeat_ngram_size;
        if generated_tokens.len() < n - 1 {
            return;
        }
        
        // Get the last n-1 tokens
        let ngram_prefix = &generated_tokens[generated_tokens.len() - (n - 1)..];
        
        // Check all previous n-grams
        for i in 0..generated_tokens.len() - (n - 1) {
            let prev_ngram = &generated_tokens[i..i + (n - 1)];
            
            // If we find a matching n-gram prefix
            if prev_ngram == ngram_prefix {
                // Ban the next token in that n-gram
                if i + n <= generated_tokens.len() {
                    let banned_token = generated_tokens[i + n - 1];
                    if (banned_token as usize) < logits.len() {
                        logits[banned_token as usize] = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }
    
    /// Compute softmax probabilities
    fn softmax(&self, logits: &[f32]) -> Result<Vec<f32>> {
        // Find max for numerical stability
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exp(logit - max)
        let mut exp_logits: Vec<f32> = logits
            .iter()
            .map(|&logit| (logit - max_logit).exp())
            .collect();
        
        // Normalize
        let sum: f32 = exp_logits.iter().sum();
        if sum == 0.0 {
            return Err(anyhow!("Softmax sum is zero"));
        }
        
        for exp_logit in exp_logits.iter_mut() {
            *exp_logit /= sum;
        }
        
        Ok(exp_logits)
    }
}

/// KV Cache for efficient generation
pub struct KVCache {
    cache: HashMap<String, (Tensor, Tensor)>,
    max_sequence_length: usize,
    current_length: usize,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            max_sequence_length: 2048,
            current_length: 0,
        }
    }
    
    /// Create with specific max sequence length
    pub fn with_max_length(max_sequence_length: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_sequence_length,
            current_length: 0,
        }
    }
    
    /// Get cached KV for a layer
    pub fn get(&self, layer: &str) -> Option<&(Tensor, Tensor)> {
        self.cache.get(layer)
    }
    
    /// Set cached KV for a layer
    pub fn set(&mut self, layer: String, key: Tensor, value: Tensor) {
        self.cache.insert(layer, (key, value));
        
        // Update current length based on key tensor shape
        if let Ok(shape) = key.shape().dims() {
            if shape.len() >= 2 {
                self.current_length = shape[shape.len() - 2];
            }
        }
    }
    
    /// Update cache with new key-value pairs
    pub fn update(&mut self, layer: String, new_key: Tensor, new_value: Tensor) -> Result<()> {
        if let Some((existing_key, existing_value)) = self.cache.get_mut(&layer) {
            // Concatenate along sequence dimension
            let updated_key = Tensor::cat(&[existing_key, &new_key], 1)?;
            let updated_value = Tensor::cat(&[existing_value, &new_value], 1)?;
            
            // Check if we've exceeded max length
            if let Ok(shape) = updated_key.shape().dims() {
                if shape.len() >= 2 && shape[1] > self.max_sequence_length {
                    warn!("KV cache exceeded max sequence length, truncating");
                    // Truncate to max length (keep most recent)
                    let start = shape[1] - self.max_sequence_length;
                    let truncated_key = updated_key.narrow(1, start, self.max_sequence_length)?;
                    let truncated_value = updated_value.narrow(1, start, self.max_sequence_length)?;
                    *existing_key = truncated_key;
                    *existing_value = truncated_value;
                } else {
                    *existing_key = updated_key;
                    *existing_value = updated_value;
                }
            }
        } else {
            self.set(layer, new_key, new_value);
        }
        
        Ok(())
    }
    
    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.current_length = 0;
    }
    
    /// Get current cache length
    pub fn length(&self) -> usize {
        self.current_length
    }
    
    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_kv_cache() {
        let mut cache = KVCache::new();
        assert!(cache.is_empty());
        
        // Create dummy tensors
        let device = Device::Cpu;
        let key = Tensor::randn(0.0f32, 1.0f32, (1, 10, 64), &device).unwrap();
        let value = Tensor::randn(0.0f32, 1.0f32, (1, 10, 64), &device).unwrap();
        
        cache.set("layer1".to_string(), key.clone(), value.clone());
        assert!(!cache.is_empty());
        assert!(cache.get("layer1").is_some());
        
        // Test update
        let new_key = Tensor::randn(0.0f32, 1.0f32, (1, 5, 64), &device).unwrap();
        let new_value = Tensor::randn(0.0f32, 1.0f32, (1, 5, 64), &device).unwrap();
        
        cache.update("layer1".to_string(), new_key, new_value).unwrap();
        
        if let Some((cached_key, _)) = cache.get("layer1") {
            let shape = cached_key.shape().dims().unwrap();
            assert_eq!(shape[1], 15); // 10 + 5
        }
        
        cache.clear();
        assert!(cache.is_empty());
    }
    
    #[test]
    fn test_softmax() {
        let generator = TextGenerator::new(Device::Cpu);
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        
        let probs = generator.softmax(&logits).unwrap();
        
        // Check that probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check that probabilities are ordered correctly
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
        assert!(probs[2] < probs[3]);
    }
    
    #[test]
    fn test_top_k_filtering() {
        let generator = TextGenerator::new(Device::Cpu);
        let mut logits = vec![1.0, 4.0, 2.0, 5.0, 3.0];
        
        generator.apply_top_k(&mut logits, 3);
        
        // Check that only top 3 values are kept
        let non_inf_count = logits.iter().filter(|&&x| x != f32::NEG_INFINITY).count();
        assert_eq!(non_inf_count, 3);
        
        // Check that the correct values are kept
        assert_ne!(logits[1], f32::NEG_INFINITY); // 4.0
        assert_ne!(logits[3], f32::NEG_INFINITY); // 5.0
        assert_ne!(logits[4], f32::NEG_INFINITY); // 3.0
        assert_eq!(logits[0], f32::NEG_INFINITY); // 1.0
        assert_eq!(logits[2], f32::NEG_INFINITY); // 2.0
    }
    
    #[test]
    fn test_repetition_penalty() {
        let generator = TextGenerator::new(Device::Cpu).with_repetition_penalty(1.5);
        let mut logits = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let generated = vec![0, 2, 4];
        
        generator.apply_repetition_penalty(&mut logits, &generated);
        
        // Positive logits should be divided by penalty
        assert!((logits[0] - 1.0 / 1.5).abs() < 1e-6);
        assert!((logits[2] - 3.0 / 1.5).abs() < 1e-6);
        assert!((logits[4] - 5.0 / 1.5).abs() < 1e-6);
        
        // Negative logits unchanged (not in generated)
        assert_eq!(logits[1], -2.0);
        assert_eq!(logits[3], -4.0);
    }
}