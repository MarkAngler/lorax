//! Metrics tracking and monitoring system for T2L training
//!
//! This module provides comprehensive metrics collection, real-time monitoring,
//! and visualization support for training progress, performance, and model
//! quality metrics.

pub mod tracker;
pub mod collectors;
pub mod exporters;
pub mod aggregators;

pub use tracker::MetricsTracker;
pub use collectors::{
    MetricsCollector, LossCollector, PerformanceCollector, 
    MemoryCollector, SystemCollector, ModelCollector
};
pub use exporters::{
    MetricsExporter, WandbExporter, TensorBoardExporter, 
    PrometheusExporter, JsonExporter
};
pub use aggregators::{MetricsAggregator, TimeWindowAggregator, RollingAverageAggregator};

use std::collections::HashMap;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Core training metrics structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss metrics
    pub loss: LossMetrics,
    
    /// Performance metrics
    pub performance: PerformanceMetrics,
    
    /// Model-specific metrics
    pub model: ModelMetrics,
    
    /// System resource metrics
    pub system: SystemMetrics,
    
    /// Learning rate and optimization metrics
    pub optimization: OptimizationMetrics,
    
    /// Validation metrics
    pub validation: Option<ValidationMetrics>,
    
    /// Custom metrics
    pub custom: HashMap<String, MetricValue>,
    
    /// Timestamp of metrics collection
    pub timestamp: DateTime<Utc>,
    
    /// Training step when metrics were collected
    pub step: usize,
    
    /// Training epoch when metrics were collected
    pub epoch: usize,
}

/// Loss-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossMetrics {
    /// Current training loss
    pub train_loss: f64,
    
    /// Validation loss
    pub val_loss: Option<f64>,
    
    /// Reconstruction loss (for T2L training)
    pub reconstruction_loss: Option<f64>,
    
    /// Supervised loss
    pub supervised_loss: Option<f64>,
    
    /// Regularization loss
    pub regularization_loss: Option<f64>,
    
    /// Total loss (sum of all components)
    pub total_loss: f64,
    
    /// Loss history for trend analysis
    pub loss_history: Vec<f64>,
    
    /// Loss smoothed over time window
    pub smoothed_loss: f64,
    
    /// Loss gradient (rate of change)
    pub loss_gradient: f64,
}

/// Performance-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Samples processed per second
    pub samples_per_second: f64,
    
    /// Training steps per second
    pub steps_per_second: f64,
    
    /// Tokens processed per second
    pub tokens_per_second: f64,
    
    /// Average time per training step
    pub avg_step_time_ms: f64,
    
    /// Average time per epoch
    pub avg_epoch_time_ms: f64,
    
    /// Forward pass time
    pub forward_time_ms: f64,
    
    /// Backward pass time
    pub backward_time_ms: f64,
    
    /// Optimizer step time
    pub optimizer_time_ms: f64,
    
    /// Data loading time
    pub data_loading_time_ms: f64,
    
    /// Model throughput in FLOPS
    pub model_flops: f64,
    
    /// Training efficiency percentage
    pub efficiency_percent: f64,
}

/// Model-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Gradient norm
    pub gradient_norm: f64,
    
    /// Parameter norm
    pub parameter_norm: f64,
    
    /// Weight updates norm
    pub weight_update_norm: f64,
    
    /// Model accuracy (if applicable)
    pub accuracy: Option<f64>,
    
    /// Model perplexity
    pub perplexity: Option<f64>,
    
    /// BLEU score (for text generation)
    pub bleu_score: Option<f64>,
    
    /// LoRA adaptation metrics
    pub lora_metrics: Option<LoRAMetrics>,
    
    /// Layer-wise statistics
    pub layer_stats: HashMap<String, LayerMetrics>,
    
    /// Activation statistics
    pub activation_stats: ActivationMetrics,
}

/// LoRA-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAMetrics {
    /// LoRA weight magnitudes
    pub weight_magnitudes: HashMap<String, f64>,
    
    /// LoRA rank utilization
    pub rank_utilization: f64,
    
    /// Adaptation strength
    pub adaptation_strength: f64,
    
    /// LoRA efficiency
    pub efficiency: f64,
}

/// Layer-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMetrics {
    /// Layer activation mean
    pub activation_mean: f64,
    
    /// Layer activation standard deviation
    pub activation_std: f64,
    
    /// Gradient magnitude
    pub gradient_magnitude: f64,
    
    /// Weight magnitude
    pub weight_magnitude: f64,
    
    /// Layer-specific loss contribution
    pub loss_contribution: Option<f64>,
}

/// Activation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationMetrics {
    /// Dead neuron percentage
    pub dead_neurons_percent: f64,
    
    /// Activation sparsity
    pub sparsity: f64,
    
    /// Activation distribution entropy
    pub entropy: f64,
    
    /// Saturation percentage
    pub saturation_percent: f64,
}

/// System resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    
    /// Memory utilization percentage
    pub memory_utilization: f64,
    
    /// GPU metrics (if available)
    pub gpu: Option<GpuMetrics>,
    
    /// Disk I/O metrics
    pub disk_io: DiskIOMetrics,
    
    /// Network I/O metrics
    pub network_io: NetworkIOMetrics,
    
    /// Temperature metrics
    pub temperature: Option<TemperatureMetrics>,
}

/// GPU-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// GPU utilization percentage
    pub utilization: f64,
    
    /// GPU memory usage in bytes
    pub memory_used_bytes: u64,
    
    /// GPU memory utilization percentage
    pub memory_utilization: f64,
    
    /// GPU temperature in Celsius
    pub temperature_celsius: f64,
    
    /// GPU power usage in watts
    pub power_usage_watts: f64,
    
    /// GPU clock speeds
    pub clock_speeds: GpuClockSpeeds,
}

/// GPU clock speeds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuClockSpeeds {
    /// Core clock speed in MHz
    pub core_mhz: f64,
    
    /// Memory clock speed in MHz
    pub memory_mhz: f64,
}

/// Disk I/O metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIOMetrics {
    /// Read rate in bytes per second
    pub read_bytes_per_sec: f64,
    
    /// Write rate in bytes per second
    pub write_bytes_per_sec: f64,
    
    /// Read operations per second
    pub read_ops_per_sec: f64,
    
    /// Write operations per second
    pub write_ops_per_sec: f64,
}

/// Network I/O metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIOMetrics {
    /// Bytes received per second
    pub bytes_received_per_sec: f64,
    
    /// Bytes sent per second
    pub bytes_sent_per_sec: f64,
    
    /// Packets received per second
    pub packets_received_per_sec: f64,
    
    /// Packets sent per second
    pub packets_sent_per_sec: f64,
}

/// Temperature metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureMetrics {
    /// CPU temperature in Celsius
    pub cpu_celsius: f64,
    
    /// GPU temperature in Celsius
    pub gpu_celsius: Option<f64>,
    
    /// System temperature in Celsius
    pub system_celsius: f64,
}

/// Optimization-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationMetrics {
    /// Current learning rate
    pub learning_rate: f64,
    
    /// Learning rate history
    pub lr_history: Vec<f64>,
    
    /// Gradient clipping ratio
    pub gradient_clip_ratio: f64,
    
    /// Optimizer momentum (if applicable)
    pub momentum: Option<f64>,
    
    /// AdamW beta parameters
    pub beta1: Option<f64>,
    pub beta2: Option<f64>,
    
    /// Weight decay value
    pub weight_decay: f64,
    
    /// Epsilon value
    pub epsilon: f64,
    
    /// Gradient accumulation steps
    pub grad_accum_steps: usize,
    
    /// Effective batch size
    pub effective_batch_size: usize,
}

/// Validation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    /// Validation accuracy
    pub accuracy: f64,
    
    /// Validation F1 score
    pub f1_score: f64,
    
    /// Validation precision
    pub precision: f64,
    
    /// Validation recall
    pub recall: f64,
    
    /// Validation AUC
    pub auc: Option<f64>,
    
    /// Task-specific metrics
    pub task_metrics: HashMap<String, f64>,
    
    /// Confusion matrix (flattened)
    pub confusion_matrix: Option<Vec<f64>>,
    
    /// Per-class metrics
    pub per_class_metrics: HashMap<String, ClassMetrics>,
}

/// Per-class validation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetrics {
    /// Class precision
    pub precision: f64,
    
    /// Class recall
    pub recall: f64,
    
    /// Class F1 score
    pub f1_score: f64,
    
    /// Class support (number of samples)
    pub support: usize,
}

/// Generic metric value that can hold different types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Float(f64),
    Integer(i64),
    Boolean(bool),
    String(String),
    Array(Vec<f64>),
    Histogram { bins: Vec<f64>, counts: Vec<u64> },
}

/// Metric aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    /// Time window for aggregation
    pub window_size: Duration,
    
    /// Aggregation method
    pub method: AggregationMethod,
    
    /// Update frequency
    pub update_frequency: Duration,
    
    /// Maximum number of data points to keep
    pub max_points: usize,
}

/// Aggregation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    Mean,
    Median,
    Min,
    Max,
    Sum,
    Count,
    StdDev,
    Percentile(f64),
    ExponentialMovingAverage { alpha: f64 },
    WeightedAverage { weights: Vec<f64> },
}

impl TrainingMetrics {
    /// Create new training metrics with default values
    pub fn new(step: usize, epoch: usize) -> Self {
        Self {
            loss: LossMetrics::default(),
            performance: PerformanceMetrics::default(),
            model: ModelMetrics::default(),
            system: SystemMetrics::default(),
            optimization: OptimizationMetrics::default(),
            validation: None,
            custom: HashMap::new(),
            timestamp: Utc::now(),
            step,
            epoch,
        }
    }
    
    /// Add a custom metric
    pub fn add_custom_metric(&mut self, name: String, value: MetricValue) {
        self.custom.insert(name, value);
    }
    
    /// Get a metric value by name
    pub fn get(&self, name: &str) -> Option<f64> {
        match name {
            "train_loss" => Some(self.loss.train_loss),
            "val_loss" => self.loss.val_loss,
            "learning_rate" => Some(self.optimization.learning_rate),
            "gradient_norm" => Some(self.model.gradient_norm),
            "samples_per_second" => Some(self.performance.samples_per_second),
            "cpu_utilization" => Some(self.system.cpu_utilization),
            "memory_utilization" => Some(self.system.memory_utilization),
            _ => self.custom.get(name).and_then(|v| match v {
                MetricValue::Float(f) => Some(*f),
                MetricValue::Integer(i) => Some(*i as f64),
                _ => None,
            }),
        }
    }
    
    /// Calculate training efficiency
    pub fn calculate_efficiency(&self) -> f64 {
        // Simple efficiency calculation based on multiple factors
        let loss_factor = 1.0 / (1.0 + self.loss.train_loss);
        let performance_factor = self.performance.efficiency_percent / 100.0;
        let resource_factor = 1.0 - (self.system.cpu_utilization / 100.0);
        
        (loss_factor + performance_factor + resource_factor) / 3.0
    }
    
    /// Get trend analysis for a metric
    pub fn get_trend(&self, metric_name: &str, history: &[TrainingMetrics]) -> MetricTrend {
        let values: Vec<f64> = history.iter()
            .filter_map(|m| m.get(metric_name))
            .collect();
        
        if values.len() < 2 {
            return MetricTrend::Stable;
        }
        
        let recent_avg = values[values.len().saturating_sub(5)..].iter().sum::<f64>() / 5.0;
        let older_avg = values[..values.len().saturating_sub(5)].iter().sum::<f64>() / (values.len() - 5) as f64;
        
        let change_percent = ((recent_avg - older_avg) / older_avg.abs()) * 100.0;
        
        if change_percent > 5.0 {
            MetricTrend::Increasing
        } else if change_percent < -5.0 {
            MetricTrend::Decreasing
        } else {
            MetricTrend::Stable
        }
    }
}

/// Metric trend analysis
#[derive(Debug, Clone, PartialEq)]
pub enum MetricTrend {
    Increasing,
    Decreasing,
    Stable,
}

// Default implementations for all metric types
impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl Default for LossMetrics {
    fn default() -> Self {
        Self {
            train_loss: 0.0,
            val_loss: None,
            reconstruction_loss: None,
            supervised_loss: None,
            regularization_loss: None,
            total_loss: 0.0,
            loss_history: Vec::new(),
            smoothed_loss: 0.0,
            loss_gradient: 0.0,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            samples_per_second: 0.0,
            steps_per_second: 0.0,
            tokens_per_second: 0.0,
            avg_step_time_ms: 0.0,
            avg_epoch_time_ms: 0.0,
            forward_time_ms: 0.0,
            backward_time_ms: 0.0,
            optimizer_time_ms: 0.0,
            data_loading_time_ms: 0.0,
            model_flops: 0.0,
            efficiency_percent: 0.0,
        }
    }
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            gradient_norm: 0.0,
            parameter_norm: 0.0,
            weight_update_norm: 0.0,
            accuracy: None,
            perplexity: None,
            bleu_score: None,
            lora_metrics: None,
            layer_stats: HashMap::new(),
            activation_stats: ActivationMetrics::default(),
        }
    }
}

impl Default for ActivationMetrics {
    fn default() -> Self {
        Self {
            dead_neurons_percent: 0.0,
            sparsity: 0.0,
            entropy: 0.0,
            saturation_percent: 0.0,
        }
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_usage_bytes: 0,
            memory_utilization: 0.0,
            gpu: None,
            disk_io: DiskIOMetrics::default(),
            network_io: NetworkIOMetrics::default(),
            temperature: None,
        }
    }
}

impl Default for DiskIOMetrics {
    fn default() -> Self {
        Self {
            read_bytes_per_sec: 0.0,
            write_bytes_per_sec: 0.0,
            read_ops_per_sec: 0.0,
            write_ops_per_sec: 0.0,
        }
    }
}

impl Default for NetworkIOMetrics {
    fn default() -> Self {
        Self {
            bytes_received_per_sec: 0.0,
            bytes_sent_per_sec: 0.0,
            packets_received_per_sec: 0.0,
            packets_sent_per_sec: 0.0,
        }
    }
}

impl Default for OptimizationMetrics {
    fn default() -> Self {
        Self {
            learning_rate: 0.0,
            lr_history: Vec::new(),
            gradient_clip_ratio: 0.0,
            momentum: None,
            beta1: None,
            beta2: None,
            weight_decay: 0.0,
            epsilon: 1e-8,
            grad_accum_steps: 1,
            effective_batch_size: 0,
        }
    }
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            window_size: Duration::from_secs(60),
            method: AggregationMethod::Mean,
            update_frequency: Duration::from_secs(10),
            max_points: 1000,
        }
    }
}

impl MetricValue {
    /// Convert metric value to f64 if possible
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetricValue::Float(f) => Some(*f),
            MetricValue::Integer(i) => Some(*i as f64),
            MetricValue::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
    
    /// Check if metric value is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(self, MetricValue::Float(_) | MetricValue::Integer(_))
    }
}