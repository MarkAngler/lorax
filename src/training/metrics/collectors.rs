//! Metrics collectors for different system components
//!
//! This module provides various collectors that gather metrics from different
//! sources including training losses, system resources, model parameters,
//! and performance statistics.

use std::time::{Duration, Instant};
use async_trait::async_trait;
use anyhow::Result;

use super::{TrainingMetrics, LossMetrics, PerformanceMetrics, ModelMetrics, SystemMetrics};

/// Trait for metrics collectors
#[async_trait]
pub trait MetricsCollector {
    /// Collector name
    fn name(&self) -> &str;
    
    /// Collect metrics
    async fn collect(&mut self) -> Result<TrainingMetrics>;
    
    /// Reset collector state
    async fn reset(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Check if collector is enabled
    fn is_enabled(&self) -> bool {
        true
    }
}

/// Loss metrics collector
pub struct LossCollector {
    name: String,
    last_collection: Option<Instant>,
}

impl LossCollector {
    pub fn new() -> Self {
        Self {
            name: "loss_collector".to_string(),
            last_collection: None,
        }
    }
}

#[async_trait]
impl MetricsCollector for LossCollector {
    fn name(&self) -> &str {
        &self.name
    }
    
    async fn collect(&mut self) -> Result<TrainingMetrics> {
        self.last_collection = Some(Instant::now());
        
        // In a real implementation, this would collect actual loss values
        // from the training context
        let mut metrics = TrainingMetrics::default();
        metrics.loss = LossMetrics::default();
        
        Ok(metrics)
    }
}

/// Performance metrics collector
pub struct PerformanceCollector {
    name: String,
    last_collection: Option<Instant>,
    step_times: Vec<Duration>,
}

impl PerformanceCollector {
    pub fn new() -> Self {
        Self {
            name: "performance_collector".to_string(),
            last_collection: None,
            step_times: Vec::new(),
        }
    }
    
    pub fn record_step_time(&mut self, duration: Duration) {
        self.step_times.push(duration);
        // Keep only last 100 step times
        if self.step_times.len() > 100 {
            self.step_times.remove(0);
        }
    }
}

#[async_trait]
impl MetricsCollector for PerformanceCollector {
    fn name(&self) -> &str {
        &self.name
    }
    
    async fn collect(&mut self) -> Result<TrainingMetrics> {
        self.last_collection = Some(Instant::now());
        
        let mut metrics = TrainingMetrics::default();
        
        if !self.step_times.is_empty() {
            let avg_step_time = self.step_times.iter().sum::<Duration>() / self.step_times.len() as u32;
            metrics.performance.avg_step_time_ms = avg_step_time.as_millis() as f64;
            
            // Calculate steps per second
            if avg_step_time.as_secs_f64() > 0.0 {
                metrics.performance.steps_per_second = 1.0 / avg_step_time.as_secs_f64();
            }
        }
        
        Ok(metrics)
    }
}

/// Memory usage collector
pub struct MemoryCollector {
    name: String,
    last_collection: Option<Instant>,
}

impl MemoryCollector {
    pub fn new() -> Self {
        Self {
            name: "memory_collector".to_string(),
            last_collection: None,
        }
    }
}

#[async_trait]
impl MetricsCollector for MemoryCollector {
    fn name(&self) -> &str {
        &self.name
    }
    
    async fn collect(&mut self) -> Result<TrainingMetrics> {
        self.last_collection = Some(Instant::now());
        
        let mut metrics = TrainingMetrics::default();
        
        // Collect memory usage (simplified implementation)
        #[cfg(target_os = "linux")]
        {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                metrics.system.memory_usage_bytes = kb * 1024;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(metrics)
    }
}

/// System metrics collector
pub struct SystemCollector {
    name: String,
    last_collection: Option<Instant>,
    last_cpu_time: Option<Duration>,
}

impl SystemCollector {
    pub fn new() -> Self {
        Self {
            name: "system_collector".to_string(),
            last_collection: None,
            last_cpu_time: None,
        }
    }
}

#[async_trait]
impl MetricsCollector for SystemCollector {
    fn name(&self) -> &str {
        &self.name
    }
    
    async fn collect(&mut self) -> Result<TrainingMetrics> {
        self.last_collection = Some(Instant::now());
        
        let mut metrics = TrainingMetrics::default();
        
        // Collect CPU utilization (simplified)
        #[cfg(target_os = "linux")]
        {
            if let Ok(stat) = std::fs::read_to_string("/proc/stat") {
                if let Some(cpu_line) = stat.lines().next() {
                    let fields: Vec<&str> = cpu_line.split_whitespace().collect();
                    if fields.len() >= 5 {
                        if let (Ok(user), Ok(nice), Ok(system), Ok(idle)) = (
                            fields[1].parse::<u64>(),
                            fields[2].parse::<u64>(),
                            fields[3].parse::<u64>(),
                            fields[4].parse::<u64>(),
                        ) {
                            let total = user + nice + system + idle;
                            let active = user + nice + system;
                            if total > 0 {
                                metrics.system.cpu_utilization = (active as f64 / total as f64) * 100.0;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(metrics)
    }
}

/// Model metrics collector
pub struct ModelCollector {
    name: String,
    last_collection: Option<Instant>,
}

impl ModelCollector {
    pub fn new() -> Self {
        Self {
            name: "model_collector".to_string(),
            last_collection: None,
        }
    }
}

#[async_trait]
impl MetricsCollector for ModelCollector {
    fn name(&self) -> &str {
        &self.name
    }
    
    async fn collect(&mut self) -> Result<TrainingMetrics> {
        self.last_collection = Some(Instant::now());
        
        let mut metrics = TrainingMetrics::default();
        metrics.model = ModelMetrics::default();
        
        // In a real implementation, this would collect actual model metrics
        // from the model state (gradients, weights, etc.)
        
        Ok(metrics)
    }
}