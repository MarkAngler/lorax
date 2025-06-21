//! Utility functions and types for the T2L system

use anyhow::Result;
use std::path::Path;
use std::fs;

/// Mathematical utilities
pub mod math {
    /// Calculate Xavier/Glorot initialization bound
    pub fn xavier_bound(fan_in: usize, fan_out: usize) -> f64 {
        (6.0 / (fan_in + fan_out) as f64).sqrt()
    }
    
    /// Calculate He initialization standard deviation
    pub fn he_std(fan_in: usize) -> f64 {
        (2.0 / fan_in as f64).sqrt()
    }
    
    /// Softmax function
    pub fn softmax(x: &[f32]) -> Vec<f32> {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_values: Vec<f32> = x.iter().map(|&val| (val - max_val).exp()).collect();
        let sum: f32 = exp_values.iter().sum();
        exp_values.iter().map(|&val| val / sum).collect()
    }
    
    /// Layer normalization
    pub fn layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let variance = x.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / x.len() as f32;
        let std = (variance + eps).sqrt();
        x.iter().map(|&val| (val - mean) / std).collect()
    }
}

/// File I/O utilities
pub mod io {
    use super::*;
    
    /// Ensure directory exists
    pub fn ensure_dir_exists<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            fs::create_dir_all(path)?;
        }
        Ok(())
    }
    
    /// Check if file exists and is readable
    pub fn is_file_readable<P: AsRef<Path>>(path: P) -> bool {
        let path = path.as_ref();
        path.exists() && path.is_file() && fs::metadata(path).is_ok()
    }
    
    /// Get file size in bytes
    pub fn file_size<P: AsRef<Path>>(path: P) -> Result<u64> {
        let metadata = fs::metadata(path)?;
        Ok(metadata.len())
    }
    
    /// Safe file write with atomic operation
    pub fn write_file_atomic<P: AsRef<Path>>(path: P, content: &[u8]) -> Result<()> {
        let path = path.as_ref();
        let temp_path = path.with_extension("tmp");
        
        fs::write(&temp_path, content)?;
        fs::rename(temp_path, path)?;
        
        Ok(())
    }
}

/// Memory utilities
pub mod memory {
    use super::Result;
    
    /// Get system memory usage information
    #[derive(Debug, Clone)]
    pub struct MemoryInfo {
        pub total_mb: u64,
        pub available_mb: u64,
        pub used_mb: u64,
        pub usage_percent: f64,
    }
    
    impl MemoryInfo {
        /// Get current memory information (placeholder implementation)
        pub fn current() -> Self {
            // TODO: Implement actual memory monitoring
            Self {
                total_mb: 8192,
                available_mb: 4096,
                used_mb: 4096,
                usage_percent: 50.0,
            }
        }
        
        /// Check if memory usage is high
        pub fn is_high_usage(&self, threshold: f64) -> bool {
            self.usage_percent > threshold
        }
    }
    
    /// Memory pool for efficient tensor allocation
    pub struct MemoryPool {
        allocated_bytes: usize,
        max_bytes: usize,
    }
    
    impl MemoryPool {
        /// Create new memory pool with size limit
        pub fn new(max_bytes: usize) -> Self {
            Self {
                allocated_bytes: 0,
                max_bytes,
            }
        }
        
        /// Try to allocate memory
        pub fn allocate(&mut self, bytes: usize) -> Result<bool> {
            if self.allocated_bytes + bytes <= self.max_bytes {
                self.allocated_bytes += bytes;
                Ok(true)
            } else {
                Ok(false)
            }
        }
        
        /// Free allocated memory
        pub fn deallocate(&mut self, bytes: usize) {
            self.allocated_bytes = self.allocated_bytes.saturating_sub(bytes);
        }
        
        /// Get current usage
        pub fn usage(&self) -> (usize, usize) {
            (self.allocated_bytes, self.max_bytes)
        }
        
        /// Get usage percentage
        pub fn usage_percent(&self) -> f64 {
            if self.max_bytes == 0 {
                0.0
            } else {
                (self.allocated_bytes as f64 / self.max_bytes as f64) * 100.0
            }
        }
    }
}

/// Performance timing utilities
pub mod timing {
    use std::time::{Duration, Instant};
    
    /// Simple timer for measuring performance
    pub struct Timer {
        start: Instant,
        name: String,
    }
    
    impl Timer {
        /// Start a new timer
        pub fn new(name: impl Into<String>) -> Self {
            Self {
                start: Instant::now(),
                name: name.into(),
            }
        }
        
        /// Get elapsed time
        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }
        
        /// Get elapsed time in milliseconds
        pub fn elapsed_ms(&self) -> f64 {
            self.elapsed().as_secs_f64() * 1000.0
        }
        
        /// Stop timer and log result
        pub fn stop(self) -> Duration {
            let elapsed = self.elapsed();
            tracing::debug!("Timer '{}' elapsed: {:.2}ms", self.name, self.elapsed_ms());
            elapsed
        }
    }
    
    /// Performance measurement collector
    #[derive(Debug, Default)]
    pub struct PerfCollector {
        measurements: std::collections::HashMap<String, Vec<Duration>>,
    }
    
    impl PerfCollector {
        /// Create new performance collector
        pub fn new() -> Self {
            Self::default()
        }
        
        /// Add measurement
        pub fn add_measurement(&mut self, name: impl Into<String>, duration: Duration) {
            self.measurements
                .entry(name.into())
                .or_insert_with(Vec::new)
                .push(duration);
        }
        
        /// Get average duration for a measurement
        pub fn average(&self, name: &str) -> Option<Duration> {
            let measurements = self.measurements.get(name)?;
            if measurements.is_empty() {
                return None;
            }
            
            let total: Duration = measurements.iter().sum();
            Some(total / measurements.len() as u32)
        }
        
        /// Get all measurements
        pub fn measurements(&self) -> &std::collections::HashMap<String, Vec<Duration>> {
            &self.measurements
        }
        
        /// Clear all measurements
        pub fn clear(&mut self) {
            self.measurements.clear();
        }
    }
}

/// String processing utilities
pub mod text {
    /// Clean and normalize task description
    pub fn clean_task_description(text: &str) -> String {
        text.trim()
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    /// Truncate text to maximum length
    pub fn truncate_text(text: &str, max_length: usize) -> String {
        if text.len() <= max_length {
            text.to_string()
        } else {
            let mut truncated = text.chars().take(max_length - 3).collect::<String>();
            truncated.push_str("...");
            truncated
        }
    }
    
    /// Extract keywords from text (simple implementation)
    pub fn extract_keywords(text: &str, max_keywords: usize) -> Vec<String> {
        let stop_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"];
        
        text.to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 2 && !stop_words.contains(word))
            .take(max_keywords)
            .map(|s| s.to_string())
            .collect()
    }
}

/// Configuration utilities
pub mod config {
    use serde::{Deserialize, Serialize};
    use std::path::Path;
    use super::Result;
    
    /// Load configuration from file
    pub fn load_config<T, P>(path: P) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
        P: AsRef<Path>,
    {
        let content = std::fs::read_to_string(path.as_ref())?;
        
        // Try different formats based on extension
        let config = if path.as_ref().extension().map_or(false, |ext| ext == "json") {
            serde_json::from_str(&content)?
        } else if path.as_ref().extension().map_or(false, |ext| ext == "yaml" || ext == "yml") {
            serde_yaml::from_str(&content)?
        } else {
            // Default to JSON
            serde_json::from_str(&content)?
        };
        
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_config<T, P>(config: &T, path: P) -> Result<()>
    where
        T: Serialize,
        P: AsRef<Path>,
    {
        super::io::ensure_dir_exists(path.as_ref().parent().unwrap_or(Path::new(".")))?;
        
        let content = if path.as_ref().extension().map_or(false, |ext| ext == "yaml" || ext == "yml") {
            serde_yaml::to_string(config)?
        } else {
            serde_json::to_string_pretty(config)?
        };
        
        super::io::write_file_atomic(path, content.as_bytes())?;
        Ok(())
    }
}

/// Validation utilities
pub mod validation {
    use anyhow::anyhow;
    use super::Result;
    
    /// Validate LoRA rank
    pub fn validate_lora_rank(rank: usize) -> Result<()> {
        if rank == 0 {
            return Err(anyhow!("LoRA rank must be greater than 0"));
        }
        if rank > 512 {
            return Err(anyhow!("LoRA rank should not exceed 512 for efficiency"));
        }
        Ok(())
    }
    
    /// Validate alpha value
    pub fn validate_alpha(alpha: f32) -> Result<()> {
        if alpha <= 0.0 {
            return Err(anyhow!("Alpha must be positive"));
        }
        if alpha > 1000.0 {
            return Err(anyhow!("Alpha value seems too large (>1000)"));
        }
        Ok(())
    }
    
    /// Validate dropout probability
    pub fn validate_dropout(dropout: f32) -> Result<()> {
        if dropout < 0.0 || dropout > 1.0 {
            return Err(anyhow!("Dropout must be between 0.0 and 1.0"));
        }
        Ok(())
    }
    
    /// Validate dimensions
    pub fn validate_dimensions(input_dim: usize, output_dim: usize) -> Result<()> {
        if input_dim == 0 {
            return Err(anyhow!("Input dimension must be greater than 0"));
        }
        if output_dim == 0 {
            return Err(anyhow!("Output dimension must be greater than 0"));
        }
        Ok(())
    }
}

/// Random number generation utilities
pub mod random {
    use rand::Rng;
    
    /// Generate random weights with Xavier initialization
    pub fn xavier_weights(fan_in: usize, fan_out: usize, size: usize) -> Vec<f32> {
        let bound = super::math::xavier_bound(fan_in, fan_out) as f32;
        let mut rng = rand::thread_rng();
        
        (0..size)
            .map(|_| rng.gen_range(-bound..bound))
            .collect()
    }
    
    /// Generate random weights with He initialization
    pub fn he_weights(fan_in: usize, size: usize) -> Vec<f32> {
        let std = super::math::he_std(fan_in) as f32;
        let mut rng = rand::thread_rng();
        
        (0..size)
            .map(|_| rng.sample::<f32, _>(rand_distr::StandardNormal) * std)
            .collect()
    }
    
    /// Generate zero-initialized weights
    pub fn zero_weights(size: usize) -> Vec<f32> {
        vec![0.0; size]
    }
}