//! Advanced sampling strategies for text generation
//!
//! This module provides various sampling methods for language model generation:
//! - Beam search with diverse beam groups
//! - Constrained generation with token filters
//! - Guided generation with logit processors
//! - Length penalty and repetition control
//! - Structured generation support

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use std::collections::{HashMap, HashSet, VecDeque};
use tracing::{debug, trace};

/// Beam search configuration
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of beams
    pub num_beams: usize,
    /// Number of beam groups for diverse beam search
    pub num_beam_groups: usize,
    /// Diversity penalty for beam groups
    pub diversity_penalty: f32,
    /// Length penalty (alpha in length normalization)
    pub length_penalty: f32,
    /// Early stopping
    pub early_stopping: bool,
    /// Maximum length difference between beams
    pub max_length_diff: usize,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            num_beams: 4,
            num_beam_groups: 1,
            diversity_penalty: 0.0,
            length_penalty: 1.0,
            early_stopping: true,
            max_length_diff: 5,
        }
    }
}

/// Beam for beam search
#[derive(Debug, Clone)]
struct Beam {
    /// Token sequence
    tokens: Vec<u32>,
    /// Log probability of the sequence
    score: f32,
    /// Whether this beam is finished (hit EOS)
    finished: bool,
}

impl Beam {
    fn new() -> Self {
        Self {
            tokens: Vec::new(),
            score: 0.0,
            finished: false,
        }
    }
    
    fn length_normalized_score(&self, length_penalty: f32) -> f32 {
        let length = self.tokens.len() as f32;
        self.score / length.powf(length_penalty)
    }
}

/// Beam search sampler
pub struct BeamSearchSampler {
    config: BeamSearchConfig,
    device: Device,
}

impl BeamSearchSampler {
    /// Create a new beam search sampler
    pub fn new(config: BeamSearchConfig, device: Device) -> Self {
        Self { config, device }
    }
    
    /// Run beam search
    pub fn search<F>(
        &self,
        initial_tokens: &[u32],
        max_length: usize,
        eos_token_id: Option<u32>,
        mut forward_fn: F,
    ) -> Result<Vec<Vec<u32>>>
    where
        F: FnMut(&[u32]) -> Result<Tensor>,
    {
        let mut beams = vec![Beam::new(); self.config.num_beams];
        
        // Initialize beams with initial tokens
        for beam in &mut beams {
            beam.tokens.extend_from_slice(initial_tokens);
        }
        
        let mut finished_beams = Vec::new();
        
        // Generate tokens
        for step in 0..max_length {
            trace!("Beam search step {}", step);
            
            // Collect candidates from all beams
            let mut all_candidates = Vec::new();
            
            for (beam_idx, beam) in beams.iter().enumerate() {
                if beam.finished {
                    continue;
                }
                
                // Get logits for current beam
                let logits = forward_fn(&beam.tokens)?;
                let log_probs = self.log_softmax(&logits)?;
                
                // Get top-k tokens
                let top_k = self.config.num_beams * 2;
                let (top_scores, top_indices) = self.top_k(&log_probs, top_k)?;
                
                // Create candidates
                for (score, token_id) in top_scores.iter().zip(top_indices.iter()) {
                    let mut candidate = beam.clone();
                    candidate.tokens.push(*token_id);
                    candidate.score += score;
                    
                    // Check if finished
                    if Some(*token_id) == eos_token_id {
                        candidate.finished = true;
                        finished_beams.push(candidate.clone());
                    }
                    
                    all_candidates.push((candidate, beam_idx));
                }
            }
            
            // Apply diversity penalty if using beam groups
            if self.config.num_beam_groups > 1 {
                self.apply_diversity_penalty(&mut all_candidates)?;
            }
            
            // Select top beams
            all_candidates.sort_by(|a, b| {
                b.0.length_normalized_score(self.config.length_penalty)
                    .partial_cmp(&a.0.length_normalized_score(self.config.length_penalty))
                    .unwrap()
            });
            
            // Update beams
            let mut new_beams = Vec::new();
            let mut seen_prefixes = HashSet::new();
            
            for (candidate, _) in all_candidates.iter() {
                if new_beams.len() >= self.config.num_beams {
                    break;
                }
                
                // Avoid duplicate prefixes
                let prefix = candidate.tokens.clone();
                if seen_prefixes.insert(prefix) {
                    new_beams.push(candidate.clone());
                }
            }
            
            beams = new_beams;
            
            // Check early stopping
            if self.config.early_stopping && self.should_stop_early(&beams, &finished_beams) {
                break;
            }
        }
        
        // Collect final results
        let mut results: Vec<_> = beams.into_iter()
            .chain(finished_beams)
            .collect();
        
        results.sort_by(|a, b| {
            b.length_normalized_score(self.config.length_penalty)
                .partial_cmp(&a.length_normalized_score(self.config.length_penalty))
                .unwrap()
        });
        
        Ok(results.into_iter()
            .take(self.config.num_beams)
            .map(|beam| beam.tokens)
            .collect())
    }
    
    /// Apply diversity penalty for diverse beam search
    fn apply_diversity_penalty(&self, candidates: &mut [(Beam, usize)]) -> Result<()> {
        let beams_per_group = self.config.num_beams / self.config.num_beam_groups;
        
        for group_idx in 0..self.config.num_beam_groups {
            if group_idx == 0 {
                continue; // First group has no penalty
            }
            
            let group_start = group_idx * beams_per_group;
            let group_end = (group_idx + 1) * beams_per_group;
            
            // Get tokens from previous groups
            let mut previous_tokens = HashSet::new();
            for i in 0..group_start {
                if i < candidates.len() {
                    if let Some(&last_token) = candidates[i].0.tokens.last() {
                        previous_tokens.insert(last_token);
                    }
                }
            }
            
            // Apply penalty
            for i in group_start..group_end.min(candidates.len()) {
                if let Some(&last_token) = candidates[i].0.tokens.last() {
                    if previous_tokens.contains(&last_token) {
                        candidates[i].0.score -= self.config.diversity_penalty;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if we should stop early
    fn should_stop_early(&self, active_beams: &[Beam], finished_beams: &[Beam]) -> bool {
        if finished_beams.is_empty() {
            return false;
        }
        
        // Find best finished score
        let best_finished_score = finished_beams.iter()
            .map(|b| b.length_normalized_score(self.config.length_penalty))
            .fold(f32::NEG_INFINITY, f32::max);
        
        // Check if all active beams are worse than best finished
        active_beams.iter().all(|beam| {
            beam.length_normalized_score(self.config.length_penalty) < best_finished_score
        })
    }
    
    /// Compute log softmax
    fn log_softmax(&self, logits: &Tensor) -> Result<Tensor> {
        let max_logit = logits.max(0)?;
        let shifted = logits.broadcast_sub(&max_logit)?;
        let exp_shifted = shifted.exp()?;
        let sum_exp = exp_shifted.sum(0)?;
        let log_sum_exp = sum_exp.log()?;
        
        shifted.broadcast_sub(&log_sum_exp)
    }
    
    /// Get top-k values and indices
    fn top_k(&self, tensor: &Tensor, k: usize) -> Result<(Vec<f32>, Vec<u32>)> {
        let values = tensor.to_vec1::<f32>()?;
        
        let mut indexed_values: Vec<(usize, f32)> = values.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        
        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let top_k_values: Vec<f32> = indexed_values.iter()
            .take(k)
            .map(|(_, v)| *v)
            .collect();
        
        let top_k_indices: Vec<u32> = indexed_values.iter()
            .take(k)
            .map(|(i, _)| *i as u32)
            .collect();
        
        Ok((top_k_values, top_k_indices))
    }
}

/// Token filter for constrained generation
pub trait TokenFilter: Send + Sync {
    /// Filter token probabilities
    fn filter(&self, token_probs: &mut [f32], generated_tokens: &[u32]) -> Result<()>;
    
    /// Get name of the filter
    fn name(&self) -> &str;
}

/// Logit processor for guided generation
pub trait LogitProcessor: Send + Sync {
    /// Process logits before sampling
    fn process(&self, logits: &mut Tensor, generated_tokens: &[u32]) -> Result<()>;
    
    /// Get name of the processor
    fn name(&self) -> &str;
}

/// Constrained generation with allowed/forbidden tokens
pub struct ConstrainedSampler {
    allowed_tokens: Option<HashSet<u32>>,
    forbidden_tokens: Option<HashSet<u32>>,
    prefix_allowed_tokens_fn: Option<Box<dyn Fn(&[u32]) -> Vec<u32> + Send + Sync>>,
}

impl ConstrainedSampler {
    /// Create a new constrained sampler
    pub fn new() -> Self {
        Self {
            allowed_tokens: None,
            forbidden_tokens: None,
            prefix_allowed_tokens_fn: None,
        }
    }
    
    /// Set allowed tokens
    pub fn with_allowed_tokens(mut self, tokens: HashSet<u32>) -> Self {
        self.allowed_tokens = Some(tokens);
        self
    }
    
    /// Set forbidden tokens
    pub fn with_forbidden_tokens(mut self, tokens: HashSet<u32>) -> Self {
        self.forbidden_tokens = Some(tokens);
        self
    }
    
    /// Set prefix-based allowed tokens function
    pub fn with_prefix_allowed_tokens_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&[u32]) -> Vec<u32> + Send + Sync + 'static,
    {
        self.prefix_allowed_tokens_fn = Some(Box::new(f));
        self
    }
}

impl TokenFilter for ConstrainedSampler {
    fn filter(&self, token_probs: &mut [f32], generated_tokens: &[u32]) -> Result<()> {
        // Apply prefix-based constraints first
        if let Some(ref prefix_fn) = self.prefix_allowed_tokens_fn {
            let allowed = prefix_fn(generated_tokens);
            let allowed_set: HashSet<u32> = allowed.into_iter().collect();
            
            for (idx, prob) in token_probs.iter_mut().enumerate() {
                if !allowed_set.contains(&(idx as u32)) {
                    *prob = 0.0;
                }
            }
        }
        
        // Apply static allowed tokens
        if let Some(ref allowed) = self.allowed_tokens {
            for (idx, prob) in token_probs.iter_mut().enumerate() {
                if !allowed.contains(&(idx as u32)) {
                    *prob = 0.0;
                }
            }
        }
        
        // Apply forbidden tokens
        if let Some(ref forbidden) = self.forbidden_tokens {
            for &token_id in forbidden {
                if (token_id as usize) < token_probs.len() {
                    token_probs[token_id as usize] = 0.0;
                }
            }
        }
        
        // Renormalize
        let sum: f32 = token_probs.iter().sum();
        if sum > 0.0 {
            for prob in token_probs.iter_mut() {
                *prob /= sum;
            }
        } else {
            return Err(anyhow!("All tokens filtered out"));
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "ConstrainedSampler"
    }
}

/// Temperature warping processor
pub struct TemperatureWarper {
    temperature: f32,
}

impl TemperatureWarper {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl LogitProcessor for TemperatureWarper {
    fn process(&self, logits: &mut Tensor, _generated_tokens: &[u32]) -> Result<()> {
        if self.temperature != 1.0 && self.temperature > 0.0 {
            *logits = logits.affine(1.0 / self.temperature, 0.0)?;
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "TemperatureWarper"
    }
}

/// Top-k filtering processor
pub struct TopKWarper {
    top_k: usize,
    filter_value: f32,
    min_tokens_to_keep: usize,
}

impl TopKWarper {
    pub fn new(top_k: usize) -> Self {
        Self {
            top_k,
            filter_value: f32::NEG_INFINITY,
            min_tokens_to_keep: 1,
        }
    }
}

impl LogitProcessor for TopKWarper {
    fn process(&self, logits: &mut Tensor, _generated_tokens: &[u32]) -> Result<()> {
        if self.top_k == 0 {
            return Ok(());
        }
        
        let vocab_size = logits.dims()[0];
        let k = self.top_k.min(vocab_size).max(self.min_tokens_to_keep);
        
        // Get values as vector
        let mut values = logits.to_vec1::<f32>()?;
        
        // Find k-th largest value
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted_values[k - 1];
        
        // Filter values below threshold
        for (i, value) in values.iter_mut().enumerate() {
            if *value < threshold {
                *value = self.filter_value;
            }
        }
        
        // Update tensor
        *logits = Tensor::from_vec(values, vocab_size, logits.device())?;
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "TopKWarper"
    }
}

/// Structured generation support
pub struct StructuredGenerator {
    /// JSON schema for structured output
    schema: Option<serde_json::Value>,
    /// Current parsing state
    state: StructuredState,
}

#[derive(Debug, Clone)]
enum StructuredState {
    Start,
    InObject(Vec<String>),
    InArray(usize),
    InString,
    InNumber,
    Complete,
}

impl StructuredGenerator {
    /// Create a new structured generator
    pub fn new(schema: Option<serde_json::Value>) -> Self {
        Self {
            schema,
            state: StructuredState::Start,
        }
    }
    
    /// Get allowed tokens for current state
    pub fn get_allowed_tokens(&self, vocab: &HashMap<String, u32>) -> Vec<u32> {
        match &self.state {
            StructuredState::Start => {
                // Allow object or array start
                let mut tokens = Vec::new();
                if let Some(token_id) = vocab.get("{") {
                    tokens.push(*token_id);
                }
                if let Some(token_id) = vocab.get("[") {
                    tokens.push(*token_id);
                }
                tokens
            }
            StructuredState::InObject(_) => {
                // Allow object keys, closing brace
                let mut tokens = Vec::new();
                if let Some(token_id) = vocab.get("\"") {
                    tokens.push(*token_id);
                }
                if let Some(token_id) = vocab.get("}") {
                    tokens.push(*token_id);
                }
                tokens
            }
            StructuredState::InArray(_) => {
                // Allow array values, closing bracket
                self.get_value_tokens(vocab)
            }
            StructuredState::InString => {
                // Allow any token except unescaped quotes
                vocab.iter()
                    .filter(|(k, _)| !k.contains('"') || k == "\\\"")
                    .map(|(_, v)| *v)
                    .collect()
            }
            StructuredState::InNumber => {
                // Allow digits, decimal point, scientific notation
                vocab.iter()
                    .filter(|(k, _)| {
                        k.chars().all(|c| c.is_numeric() || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+')
                    })
                    .map(|(_, v)| *v)
                    .collect()
            }
            StructuredState::Complete => Vec::new(),
        }
    }
    
    /// Get tokens allowed for JSON values
    fn get_value_tokens(&self, vocab: &HashMap<String, u32>) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        // Primitives
        for literal in &["true", "false", "null"] {
            if let Some(token_id) = vocab.get(*literal) {
                tokens.push(*token_id);
            }
        }
        
        // String start
        if let Some(token_id) = vocab.get("\"") {
            tokens.push(*token_id);
        }
        
        // Number start (digits and minus)
        for digit in &["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"] {
            if let Some(token_id) = vocab.get(*digit) {
                tokens.push(*token_id);
            }
        }
        
        // Object/array start
        if let Some(token_id) = vocab.get("{") {
            tokens.push(*token_id);
        }
        if let Some(token_id) = vocab.get("[") {
            tokens.push(*token_id);
        }
        
        tokens
    }
    
    /// Update state based on generated token
    pub fn update_state(&mut self, token: &str) {
        match (&self.state, token) {
            (StructuredState::Start, "{") => {
                self.state = StructuredState::InObject(Vec::new());
            }
            (StructuredState::Start, "[") => {
                self.state = StructuredState::InArray(0);
            }
            (StructuredState::InObject(_), "}") => {
                self.state = StructuredState::Complete;
            }
            (StructuredState::InArray(_), "]") => {
                self.state = StructuredState::Complete;
            }
            (StructuredState::InObject(_), "\"") => {
                self.state = StructuredState::InString;
            }
            (StructuredState::InString, "\"") => {
                // TODO: Handle escaped quotes properly
                self.state = StructuredState::InObject(Vec::new());
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_beam_creation() {
        let beam = Beam::new();
        assert!(beam.tokens.is_empty());
        assert_eq!(beam.score, 0.0);
        assert!(!beam.finished);
    }
    
    #[test]
    fn test_length_normalized_score() {
        let mut beam = Beam::new();
        beam.tokens = vec![1, 2, 3, 4];
        beam.score = -8.0;
        
        // With length penalty 1.0
        let normalized = beam.length_normalized_score(1.0);
        assert_eq!(normalized, -2.0); // -8.0 / 4
        
        // With length penalty 0.5
        let normalized = beam.length_normalized_score(0.5);
        assert_eq!(normalized, -4.0); // -8.0 / 2
    }
    
    #[test]
    fn test_constrained_sampler() {
        let mut sampler = ConstrainedSampler::new();
        
        // Test with allowed tokens
        let allowed: HashSet<u32> = vec![0, 1, 2].into_iter().collect();
        sampler = sampler.with_allowed_tokens(allowed);
        
        let mut probs = vec![0.25, 0.25, 0.25, 0.25];
        sampler.filter(&mut probs, &[]).unwrap();
        
        // Only first 3 tokens should have non-zero probability
        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
        assert!(probs[2] > 0.0);
        assert_eq!(probs[3], 0.0);
        
        // Check normalization
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_structured_generator() {
        let mut vocab = HashMap::new();
        vocab.insert("{".to_string(), 0);
        vocab.insert("}".to_string(), 1);
        vocab.insert("[".to_string(), 2);
        vocab.insert("]".to_string(), 3);
        vocab.insert("\"".to_string(), 4);
        
        let generator = StructuredGenerator::new(None);
        let allowed = generator.get_allowed_tokens(&vocab);
        
        // At start, should allow { or [
        assert!(allowed.contains(&0));
        assert!(allowed.contains(&2));
        assert_eq!(allowed.len(), 2);
    }
}