//! Metrics tracker for comprehensive training monitoring
//!
//! This module provides the main MetricsTracker class that coordinates
//! metrics collection, aggregation, and export across all system components.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tracing::{debug, info, warn, error};

use crate::training::config::LoggingConfig;
use super::{
    AggregationConfig, AggregationMethod,
    MetricsCollector, MetricsExporter, MetricsAggregator
};
use super::{TrainingMetrics, MetricValue};
use super::collectors::{
    LossCollector, PerformanceCollector, MemoryCollector, 
    SystemCollector, ModelCollector
};
use super::exporters::{
    WandbExporter, TensorBoardExporter, PrometheusExporter, JsonExporter
};
use super::aggregators::{TimeWindowAggregator, RollingAverageAggregator};

/// Main metrics tracking coordinator
pub struct MetricsTracker {
    /// Current metrics state
    current_metrics: Arc<RwLock<TrainingMetrics>>,
    
    /// Historical metrics storage
    metrics_history: Arc<RwLock<VecDeque<TrainingMetrics>>>,
    
    /// Metrics collectors
    collectors: Vec<Box<dyn MetricsCollector + Send + Sync>>,
    
    /// Metrics exporters
    exporters: Vec<Box<dyn MetricsExporter + Send + Sync>>,
    
    /// Metrics aggregators
    aggregators: HashMap<String, Box<dyn MetricsAggregator + Send + Sync>>,
    
    /// Custom metrics registry
    custom_metrics: Arc<RwLock<HashMap<String, MetricValue>>>,
    
    /// Metrics collection configuration
    config: MetricsConfig,
    
    /// Collection interval timer
    collection_timer: Option<tokio::time::Interval>,
    
    /// Event channel for real-time updates
    event_tx: Option<mpsc::UnboundedSender<MetricsEvent>>,
    
    /// Performance tracking
    collection_stats: CollectionStats,
    
    /// Metrics collection start time
    start_time: Instant,
}

/// Metrics configuration
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Collection interval
    pub collection_interval: Duration,
    
    /// Maximum history size
    pub max_history_size: usize,
    
    /// Enable real-time collection
    pub real_time_collection: bool,
    
    /// Buffer size for batch collection
    pub buffer_size: usize,
    
    /// Aggregation configurations
    pub aggregations: HashMap<String, AggregationConfig>,
    
    /// Export interval
    pub export_interval: Duration,
    
    /// Enable performance profiling
    pub enable_profiling: bool,
    
    /// Custom metric definitions
    pub custom_metrics: HashMap<String, CustomMetricConfig>,
}

/// Custom metric configuration
#[derive(Debug, Clone)]
pub struct CustomMetricConfig {
    /// Metric name
    pub name: String,
    
    /// Metric description
    pub description: String,
    
    /// Metric unit
    pub unit: String,
    
    /// Aggregation method
    pub aggregation: AggregationMethod,
    
    /// Collection frequency
    pub frequency: Duration,
    
    /// Whether to export this metric
    pub export: bool,
}

/// Metrics events for real-time monitoring
#[derive(Debug, Clone)]
pub enum MetricsEvent {
    /// New metrics collected
    MetricsUpdated { metrics: TrainingMetrics },
    
    /// Metric threshold crossed
    ThresholdCrossed { metric: String, value: f64, threshold: f64 },
    
    /// Collection error occurred
    CollectionError { collector: String, error: String },
    
    /// Export completed
    ExportCompleted { exporter: String, duration: Duration },
    
    /// Aggregation completed
    AggregationCompleted { aggregator: String, metrics_count: usize },
}

/// Collection performance statistics
#[derive(Debug, Clone)]
pub struct CollectionStats {
    /// Total collections performed
    pub total_collections: u64,
    
    /// Average collection time
    pub avg_collection_time: Duration,
    
    /// Peak collection time
    pub peak_collection_time: Duration,
    
    /// Collection errors
    pub collection_errors: u64,
    
    /// Export stats per exporter
    pub export_stats: HashMap<String, ExportStats>,
    
    /// Memory usage for metrics storage
    pub memory_usage_bytes: usize,
}

/// Export statistics
#[derive(Debug, Clone)]
pub struct ExportStats {
    /// Total exports
    pub total_exports: u64,
    
    /// Average export time
    pub avg_export_time: Duration,
    
    /// Export errors
    pub export_errors: u64,
    
    /// Last export timestamp
    pub last_export: Option<Instant>,
}

impl MetricsTracker {
    /// Create a new metrics tracker
    pub fn new(logging_config: LoggingConfig) -> Result<Self> {
        let config = MetricsConfig::from_logging_config(logging_config.clone());
        
        let mut tracker = Self {
            current_metrics: Arc::new(RwLock::new(TrainingMetrics::default())),
            metrics_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.max_history_size))),
            collectors: Vec::new(),
            exporters: Vec::new(),
            aggregators: HashMap::new(),
            custom_metrics: Arc::new(RwLock::new(HashMap::new())),
            config,
            collection_timer: None,
            event_tx: None,
            collection_stats: CollectionStats::new(),
            start_time: Instant::now(),
        };
        
        // Initialize default collectors
        tracker.setup_default_collectors()?;
        
        // Initialize exporters based on configuration
        tracker.setup_exporters(&logging_config)?;
        
        // Initialize aggregators
        tracker.setup_aggregators()?;
        
        Ok(tracker)
    }
    
    /// Start metrics collection
    pub async fn start_collection(&mut self) -> Result<()> {
        info!("Starting metrics collection");
        
        if self.config.real_time_collection {
            let mut interval = tokio::time::interval(self.config.collection_interval);
            self.collection_timer = Some(interval);
            
            // Start collection loop
            self.run_collection_loop().await?;
        }
        
        Ok(())
    }
    
    /// Stop metrics collection
    pub async fn stop_collection(&mut self) -> Result<()> {
        info!("Stopping metrics collection");
        
        self.collection_timer = None;
        
        // Final export
        self.export_all_metrics().await?;
        
        Ok(())
    }
    
    /// Collect metrics manually (single collection)
    pub async fn collect_metrics(&mut self, step: usize, epoch: usize) -> Result<TrainingMetrics> {
        let collection_start = Instant::now();
        
        let mut metrics = TrainingMetrics::new(step, epoch);
        
        // Collect from all collectors
        for collector in &mut self.collectors {
            match collector.collect().await {
                Ok(collector_metrics) => {
                    Self::merge_metrics(&mut metrics, collector_metrics);
                }
                Err(e) => {
                    warn!("Collector {} failed: {}", collector.name(), e);
                    self.collection_stats.collection_errors += 1;
                    
                    if let Some(tx) = &self.event_tx {
                        let _ = tx.send(MetricsEvent::CollectionError {
                            collector: collector.name().to_string(),
                            error: e.to_string(),
                        });
                    }
                }
            }
        }
        
        // Add custom metrics
        {
            let custom_metrics = self.custom_metrics.read();
            for (name, value) in custom_metrics.iter() {
                metrics.add_custom_metric(name.clone(), value.clone());
            }
        }
        
        // Update current metrics
        {
            let mut current = self.current_metrics.write();
            *current = metrics.clone();
        }
        
        // Add to history
        {
            let mut history = self.metrics_history.write();
            history.push_back(metrics.clone());
            
            // Trim history if needed
            while history.len() > self.config.max_history_size {
                history.pop_front();
            }
        }
        
        // Update collection stats
        let collection_time = collection_start.elapsed();
        self.collection_stats.total_collections += 1;
        self.collection_stats.avg_collection_time = Duration::from_secs_f64(
            (self.collection_stats.avg_collection_time.as_secs_f64() * (self.collection_stats.total_collections - 1) as f64 + collection_time.as_secs_f64()) 
            / self.collection_stats.total_collections as f64);
        
        if collection_time > self.collection_stats.peak_collection_time {
            self.collection_stats.peak_collection_time = collection_time;
        }
        
        // Send event
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(MetricsEvent::MetricsUpdated { 
                metrics: metrics.clone() 
            });
        }
        
        // Run aggregations
        self.run_aggregations(&metrics).await?;
        
        // Export if needed
        if self.should_export() {
            self.export_all_metrics().await?;
        }
        
        debug!("Metrics collected in {:?}", collection_time);
        
        Ok(metrics)
    }
    
    /// Get current metrics
    pub fn current_metrics(&self) -> TrainingMetrics {
        self.current_metrics.read().clone()
    }
    
    /// Get metrics history
    pub fn metrics_history(&self) -> Vec<TrainingMetrics> {
        self.metrics_history.read().iter().cloned().collect()
    }
    
    /// Add a custom metric
    pub fn add_custom_metric(&self, name: String, value: MetricValue) {
        let mut custom_metrics = self.custom_metrics.write();
        custom_metrics.insert(name, value);
    }
    
    /// Remove a custom metric
    pub fn remove_custom_metric(&self, name: &str) {
        let mut custom_metrics = self.custom_metrics.write();
        custom_metrics.remove(name);
    }
    
    /// Get metric value by name
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.current_metrics.read().get(name)
    }
    
    /// Get metric history for a specific metric
    pub fn get_metric_history(&self, name: &str) -> Vec<f64> {
        self.metrics_history.read()
            .iter()
            .filter_map(|m| m.get(name))
            .collect()
    }
    
    /// Set event channel for real-time updates
    pub fn set_event_channel(&mut self, tx: mpsc::UnboundedSender<MetricsEvent>) {
        self.event_tx = Some(tx);
    }
    
    /// Get collection statistics
    pub fn collection_stats(&self) -> &CollectionStats {
        &self.collection_stats
    }
    
    /// Setup default collectors
    fn setup_default_collectors(&mut self) -> Result<()> {
        self.collectors.push(Box::new(LossCollector::new()));
        self.collectors.push(Box::new(PerformanceCollector::new()));
        self.collectors.push(Box::new(MemoryCollector::new()));
        self.collectors.push(Box::new(SystemCollector::new()));
        self.collectors.push(Box::new(ModelCollector::new()));
        
        info!("Initialized {} default collectors", self.collectors.len());
        Ok(())
    }
    
    /// Setup exporters based on configuration
    fn setup_exporters(&mut self, logging_config: &LoggingConfig) -> Result<()> {
        // Weights & Biases exporter
        if logging_config.wandb.enabled {
            self.exporters.push(Box::new(WandbExporter::new(logging_config.wandb.clone())?));
        }
        
        // TensorBoard exporter
        if logging_config.tensorboard.enabled {
            self.exporters.push(Box::new(TensorBoardExporter::new(logging_config.tensorboard.clone())?));
        }
        
        // Prometheus exporter
        if logging_config.prometheus.enabled {
            self.exporters.push(Box::new(PrometheusExporter::new(logging_config.prometheus.clone())?));
        }
        
        // Always add JSON exporter for local storage
        self.exporters.push(Box::new(JsonExporter::new("./metrics".into())?));
        
        info!("Initialized {} exporters", self.exporters.len());
        Ok(())
    }
    
    /// Setup aggregators
    fn setup_aggregators(&mut self) -> Result<()> {
        // Time window aggregators for key metrics
        let time_window_metrics = vec![
            "train_loss", "val_loss", "learning_rate", 
            "gradient_norm", "samples_per_second"
        ];
        
        for metric in time_window_metrics {
            let aggregator = TimeWindowAggregator::new(
                Duration::from_secs(300), // 5-minute window
                100, // max points
            );
            self.aggregators.insert(
                format!("{}_5min", metric),
                Box::new(aggregator)
            );
        }
        
        // Rolling average aggregators
        let rolling_avg_metrics = vec![
            "train_loss", "samples_per_second", "cpu_utilization"
        ];
        
        for metric in rolling_avg_metrics {
            let aggregator = RollingAverageAggregator::new(50); // 50-point rolling average
            self.aggregators.insert(
                format!("{}_rolling", metric),
                Box::new(aggregator)
            );
        }
        
        info!("Initialized {} aggregators", self.aggregators.len());
        Ok(())
    }
    
    /// Run collection loop for real-time metrics
    async fn run_collection_loop(&mut self) -> Result<()> {
        // Take the interval out temporarily to avoid double mutable borrow
        if let Some(mut interval) = self.collection_timer.take() {
            loop {
                interval.tick().await;
                
                // Collect metrics (use current step/epoch from state)
                let current = self.current_metrics.read().clone();
                let _ = self.collect_metrics(current.step, current.epoch).await;
            }
        }
        
        Ok(())
    }
    
    /// Merge metrics from collector into main metrics
    fn merge_metrics(target: &mut TrainingMetrics, source: TrainingMetrics) {
        // This is a simplified merge - in practice, you'd want more sophisticated merging logic
        if source.loss.train_loss != 0.0 {
            target.loss.train_loss = source.loss.train_loss;
        }
        
        if source.performance.samples_per_second != 0.0 {
            target.performance.samples_per_second = source.performance.samples_per_second;
        }
        
        if source.model.gradient_norm != 0.0 {
            target.model.gradient_norm = source.model.gradient_norm;
        }
        
        if source.system.cpu_utilization != 0.0 {
            target.system.cpu_utilization = source.system.cpu_utilization;
        }
        
        // Merge custom metrics
        for (key, value) in source.custom {
            target.custom.insert(key, value);
        }
    }
    
    /// Run aggregations on collected metrics
    async fn run_aggregations(&mut self, metrics: &TrainingMetrics) -> Result<()> {
        for (name, aggregator) in &mut self.aggregators {
            match aggregator.aggregate(metrics).await {
                Ok(aggregated_count) => {
                    if let Some(tx) = &self.event_tx {
                        let _ = tx.send(MetricsEvent::AggregationCompleted {
                            aggregator: name.clone(),
                            metrics_count: aggregated_count,
                        });
                    }
                }
                Err(e) => {
                    warn!("Aggregator {} failed: {}", name, e);
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if metrics should be exported
    fn should_export(&self) -> bool {
        self.start_time.elapsed() >= self.config.export_interval
    }
    
    /// Export metrics to all configured exporters
    async fn export_all_metrics(&mut self) -> Result<()> {
        let metrics = self.current_metrics.read().clone();
        let history: Vec<TrainingMetrics> = self.metrics_history.read().iter().cloned().collect();
        
        for exporter in &mut self.exporters {
            let export_start = Instant::now();
            
            match exporter.export(&metrics, &history).await {
                Ok(_) => {
                    let export_time = export_start.elapsed();
                    
                    // Update export stats
                    let exporter_name = exporter.name().to_string();
                    let stats = self.collection_stats.export_stats
                        .entry(exporter_name.clone())
                        .or_insert_with(ExportStats::new);
                    
                    stats.total_exports += 1;
                    stats.avg_export_time = Duration::from_secs_f64(
                        (stats.avg_export_time.as_secs_f64() * (stats.total_exports - 1) as f64 + export_time.as_secs_f64()) 
                        / stats.total_exports as f64);
                    stats.last_export = Some(export_start);
                    
                    if let Some(tx) = &self.event_tx {
                        let _ = tx.send(MetricsEvent::ExportCompleted {
                            exporter: exporter_name,
                            duration: export_time,
                        });
                    }
                }
                Err(e) => {
                    warn!("Exporter {} failed: {}", exporter.name(), e);
                    
                    let exporter_name = exporter.name().to_string();
                    let stats = self.collection_stats.export_stats
                        .entry(exporter_name)
                        .or_insert_with(ExportStats::new);
                    stats.export_errors += 1;
                }
            }
        }
        
        Ok(())
    }
}

impl MetricsConfig {
    /// Create configuration from logging config
    fn from_logging_config(logging_config: LoggingConfig) -> Self {
        Self {
            collection_interval: Duration::from_millis(100),
            max_history_size: 10000,
            real_time_collection: true,
            buffer_size: 100,
            aggregations: HashMap::new(),
            export_interval: Duration::from_secs(60),
            enable_profiling: true,
            custom_metrics: HashMap::new(),
        }
    }
}

impl CollectionStats {
    /// Create new collection stats
    fn new() -> Self {
        Self {
            total_collections: 0,
            avg_collection_time: Duration::from_millis(0),
            peak_collection_time: Duration::from_millis(0),
            collection_errors: 0,
            export_stats: HashMap::new(),
            memory_usage_bytes: 0,
        }
    }
}

impl ExportStats {
    /// Create new export stats
    fn new() -> Self {
        Self {
            total_exports: 0,
            avg_export_time: Duration::from_millis(0),
            export_errors: 0,
            last_export: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::config::LoggingConfig;

    #[tokio::test]
    async fn test_metrics_tracker_creation() {
        let logging_config = LoggingConfig::default();
        let tracker = MetricsTracker::new(logging_config);
        assert!(tracker.is_ok());
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let logging_config = LoggingConfig::default();
        let mut tracker = MetricsTracker::new(logging_config).unwrap();
        
        let metrics = tracker.collect_metrics(1, 0).await.unwrap();
        assert_eq!(metrics.step, 1);
        assert_eq!(metrics.epoch, 0);
    }

    #[test]
    fn test_custom_metrics() {
        let logging_config = LoggingConfig::default();
        let tracker = MetricsTracker::new(logging_config).unwrap();
        
        tracker.add_custom_metric("test_metric".to_string(), MetricValue::Float(42.0));
        
        let custom_metrics = tracker.custom_metrics.read();
        assert!(custom_metrics.contains_key("test_metric"));
    }
}