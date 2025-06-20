//! Metrics exporters for different platforms and formats
//!
//! This module provides exporters for sending metrics to various monitoring
//! and logging platforms including Weights & Biases, TensorBoard, Prometheus,
//! and local file storage.

use std::collections::VecDeque;
use std::path::PathBuf;
use async_trait::async_trait;
use anyhow::Result;
use serde_json;

use crate::training::config::{WandbConfig, TensorBoardConfig, PrometheusConfig};
use super::TrainingMetrics;

/// Trait for metrics exporters
#[async_trait]
pub trait MetricsExporter {
    /// Exporter name
    fn name(&self) -> &str;
    
    /// Export metrics
    async fn export(&mut self, current: &TrainingMetrics, history: &[TrainingMetrics]) -> Result<()>;
    
    /// Check if exporter is enabled
    fn is_enabled(&self) -> bool {
        true
    }
    
    /// Get export format
    fn format(&self) -> &str {
        "json"
    }
}

/// Weights & Biases exporter
pub struct WandbExporter {
    name: String,
    config: WandbConfig,
}

impl WandbExporter {
    pub fn new(config: WandbConfig) -> Result<Self> {
        Ok(Self {
            name: "wandb_exporter".to_string(),
            config,
        })
    }
}

#[async_trait]
impl MetricsExporter for WandbExporter {
    fn name(&self) -> &str {
        &self.name
    }
    
    async fn export(&mut self, current: &TrainingMetrics, _history: &[TrainingMetrics]) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        
        // In a real implementation, this would use the wandb API
        // For now, we'll just log that we would export
        tracing::debug!("Would export to W&B: step={}, loss={:.4}", 
                       current.step, current.loss.train_loss);
        
        Ok(())
    }
    
    fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

/// TensorBoard exporter
pub struct TensorBoardExporter {
    name: String,
    config: TensorBoardConfig,
}

impl TensorBoardExporter {
    pub fn new(config: TensorBoardConfig) -> Result<Self> {
        Ok(Self {
            name: "tensorboard_exporter".to_string(),
            config,
        })
    }
}

#[async_trait]
impl MetricsExporter for TensorBoardExporter {
    fn name(&self) -> &str {
        &self.name
    }
    
    async fn export(&mut self, current: &TrainingMetrics, _history: &[TrainingMetrics]) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        
        // In a real implementation, this would write TensorBoard logs
        tracing::debug!("Would export to TensorBoard: step={}, loss={:.4}", 
                       current.step, current.loss.train_loss);
        
        Ok(())
    }
    
    fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

/// Prometheus exporter
pub struct PrometheusExporter {
    name: String,
    config: PrometheusConfig,
}

impl PrometheusExporter {
    pub fn new(config: PrometheusConfig) -> Result<Self> {
        Ok(Self {
            name: "prometheus_exporter".to_string(),
            config,
        })
    }
}

#[async_trait]
impl MetricsExporter for PrometheusExporter {
    fn name(&self) -> &str {
        &self.name
    }
    
    async fn export(&mut self, current: &TrainingMetrics, _history: &[TrainingMetrics]) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        
        // In a real implementation, this would update Prometheus metrics
        tracing::debug!("Would export to Prometheus: step={}, loss={:.4}", 
                       current.step, current.loss.train_loss);
        
        Ok(())
    }
    
    fn is_enabled(&self) -> bool {
        self.config.enabled
    }
    
    fn format(&self) -> &str {
        "prometheus"
    }
}

/// JSON file exporter
pub struct JsonExporter {
    name: String,
    output_dir: PathBuf,
}

impl JsonExporter {
    pub fn new(output_dir: PathBuf) -> Result<Self> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&output_dir)?;
        
        Ok(Self {
            name: "json_exporter".to_string(),
            output_dir,
        })
    }
}

#[async_trait]
impl MetricsExporter for JsonExporter {
    fn name(&self) -> &str {
        &self.name
    }
    
    async fn export(&mut self, current: &TrainingMetrics, history: &[TrainingMetrics]) -> Result<()> {
        // Export current metrics
        let current_file = self.output_dir.join(format!("metrics_step_{:08}.json", current.step));
        let current_json = serde_json::to_string_pretty(current)?;
        tokio::fs::write(current_file, current_json).await?;
        
        // Export history summary
        if !history.is_empty() {
            let history_file = self.output_dir.join("metrics_history.json");
            let history_json = serde_json::to_string_pretty(history)?;
            tokio::fs::write(history_file, history_json).await?;
        }
        
        Ok(())
    }
    
    fn format(&self) -> &str {
        "json"
    }
}