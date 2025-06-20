//! Metrics aggregators for time-series analysis and smoothing
//!
//! This module provides various aggregation strategies for metrics including
//! time-window aggregation, rolling averages, and statistical summaries.

use std::collections::VecDeque;
use std::time::{Duration, Instant};
use async_trait::async_trait;
use anyhow::Result;

use super::{TrainingMetrics, AggregationMethod};

/// Trait for metrics aggregators
#[async_trait]
pub trait MetricsAggregator {
    /// Aggregator name
    fn name(&self) -> &str;
    
    /// Add metrics to aggregation
    async fn aggregate(&mut self, metrics: &TrainingMetrics) -> Result<usize>;
    
    /// Get aggregated result
    async fn get_result(&self) -> Result<Option<f64>>;
    
    /// Reset aggregator state
    async fn reset(&mut self) -> Result<()>;
    
    /// Check if aggregator has enough data
    fn has_data(&self) -> bool;
}

/// Time window aggregator
pub struct TimeWindowAggregator {
    name: String,
    window_size: Duration,
    max_points: usize,
    data_points: VecDeque<(Instant, f64)>,
    method: AggregationMethod,
}

impl TimeWindowAggregator {
    pub fn new(window_size: Duration, max_points: usize) -> Self {
        Self {
            name: "time_window_aggregator".to_string(),
            window_size,
            max_points,
            data_points: VecDeque::with_capacity(max_points),
            method: AggregationMethod::Mean,
        }
    }
    
    pub fn with_method(mut self, method: AggregationMethod) -> Self {
        self.method = method;
        self
    }
    
    fn cleanup_old_data(&mut self) {
        let cutoff = Instant::now() - self.window_size;
        
        while let Some(&(timestamp, _)) = self.data_points.front() {
            if timestamp < cutoff {
                self.data_points.pop_front();
            } else {
                break;
            }
        }
        
        // Also respect max_points limit
        while self.data_points.len() > self.max_points {
            self.data_points.pop_front();
        }
    }
    
    fn calculate_aggregate(&self) -> Option<f64> {
        if self.data_points.is_empty() {
            return None;
        }
        
        let values: Vec<f64> = self.data_points.iter().map(|(_, v)| *v).collect();
        
        match &self.method {
            AggregationMethod::Mean => {
                Some(values.iter().sum::<f64>() / values.len() as f64)
            }
            AggregationMethod::Median => {
                let mut sorted = values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = sorted.len() / 2;
                if sorted.len() % 2 == 0 {
                    Some((sorted[mid - 1] + sorted[mid]) / 2.0)
                } else {
                    Some(sorted[mid])
                }
            }
            AggregationMethod::Min => {
                values.iter().fold(f64::INFINITY, |a, &b| a.min(b)).into()
            }
            AggregationMethod::Max => {
                values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)).into()
            }
            AggregationMethod::Sum => {
                Some(values.iter().sum())
            }
            AggregationMethod::Count => {
                Some(values.len() as f64)
            }
            AggregationMethod::StdDev => {
                if values.len() < 2 {
                    return Some(0.0);
                }
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>() / (values.len() - 1) as f64;
                Some(variance.sqrt())
            }
            AggregationMethod::Percentile(p) => {
                let mut sorted = values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let index = ((*p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
                sorted.get(index).copied()
            }
            AggregationMethod::ExponentialMovingAverage { alpha } => {
                if values.is_empty() {
                    return None;
                }
                
                let mut ema = values[0];
                for &value in &values[1..] {
                    ema = alpha * value + (1.0 - alpha) * ema;
                }
                Some(ema)
            }
            AggregationMethod::WeightedAverage { weights } => {
                if weights.len() != values.len() {
                    return Some(values.iter().sum::<f64>() / values.len() as f64);
                }
                
                let weighted_sum: f64 = values.iter().zip(weights.iter())
                    .map(|(v, w)| v * w)
                    .sum();
                let weight_sum: f64 = weights.iter().sum();
                
                if weight_sum != 0.0 {
                    Some(weighted_sum / weight_sum)
                } else {
                    None
                }
            }
        }
    }
}

#[async_trait]
impl MetricsAggregator for TimeWindowAggregator {
    fn name(&self) -> &str {
        &self.name
    }
    
    async fn aggregate(&mut self, metrics: &TrainingMetrics) -> Result<usize> {
        // For this example, we'll aggregate the training loss
        // In practice, you'd configure which metric to aggregate
        let value = metrics.loss.train_loss;
        let timestamp = Instant::now();
        
        self.data_points.push_back((timestamp, value));
        self.cleanup_old_data();
        
        Ok(self.data_points.len())
    }
    
    async fn get_result(&self) -> Result<Option<f64>> {
        Ok(self.calculate_aggregate())
    }
    
    async fn reset(&mut self) -> Result<()> {
        self.data_points.clear();
        Ok(())
    }
    
    fn has_data(&self) -> bool {
        !self.data_points.is_empty()
    }
}

/// Rolling average aggregator
pub struct RollingAverageAggregator {
    name: String,
    window_size: usize,
    values: VecDeque<f64>,
    sum: f64,
}

impl RollingAverageAggregator {
    pub fn new(window_size: usize) -> Self {
        Self {
            name: "rolling_average_aggregator".to_string(),
            window_size,
            values: VecDeque::with_capacity(window_size),
            sum: 0.0,
        }
    }
}

#[async_trait]
impl MetricsAggregator for RollingAverageAggregator {
    fn name(&self) -> &str {
        &self.name
    }
    
    async fn aggregate(&mut self, metrics: &TrainingMetrics) -> Result<usize> {
        let value = metrics.loss.train_loss;
        
        // Add new value
        self.values.push_back(value);
        self.sum += value;
        
        // Remove old value if window is full
        if self.values.len() > self.window_size {
            if let Some(old_value) = self.values.pop_front() {
                self.sum -= old_value;
            }
        }
        
        Ok(self.values.len())
    }
    
    async fn get_result(&self) -> Result<Option<f64>> {
        if self.values.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.sum / self.values.len() as f64))
        }
    }
    
    async fn reset(&mut self) -> Result<()> {
        self.values.clear();
        self.sum = 0.0;
        Ok(())
    }
    
    fn has_data(&self) -> bool {
        !self.values.is_empty()
    }
}

/// Statistical summary aggregator
pub struct StatisticalAggregator {
    name: String,
    values: Vec<f64>,
    max_values: usize,
}

impl StatisticalAggregator {
    pub fn new(max_values: usize) -> Self {
        Self {
            name: "statistical_aggregator".to_string(),
            values: Vec::with_capacity(max_values),
            max_values,
        }
    }
    
    pub fn get_statistics(&self) -> StatisticalSummary {
        if self.values.is_empty() {
            return StatisticalSummary::default();
        }
        
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let count = sorted.len() as f64;
        let sum = sorted.iter().sum::<f64>();
        let mean = sum / count;
        
        let variance = if count > 1.0 {
            sorted.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / (count - 1.0)
        } else {
            0.0
        };
        
        let std_dev = variance.sqrt();
        
        let median = if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        
        StatisticalSummary {
            count: count as usize,
            sum,
            mean,
            median,
            std_dev,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            q25: sorted[(0.25 * (sorted.len() - 1) as f64).round() as usize],
            q75: sorted[(0.75 * (sorted.len() - 1) as f64).round() as usize],
        }
    }
}

#[async_trait]
impl MetricsAggregator for StatisticalAggregator {
    fn name(&self) -> &str {
        &self.name
    }
    
    async fn aggregate(&mut self, metrics: &TrainingMetrics) -> Result<usize> {
        let value = metrics.loss.train_loss;
        
        self.values.push(value);
        
        // Keep only the most recent max_values
        if self.values.len() > self.max_values {
            self.values.remove(0);
        }
        
        Ok(self.values.len())
    }
    
    async fn get_result(&self) -> Result<Option<f64>> {
        if self.values.is_empty() {
            Ok(None)
        } else {
            let stats = self.get_statistics();
            Ok(Some(stats.mean))
        }
    }
    
    async fn reset(&mut self) -> Result<()> {
        self.values.clear();
        Ok(())
    }
    
    fn has_data(&self) -> bool {
        !self.values.is_empty()
    }
}

/// Statistical summary result
#[derive(Debug, Clone, Default)]
pub struct StatisticalSummary {
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub q25: f64,
    pub q75: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rolling_average_aggregator() {
        let mut aggregator = RollingAverageAggregator::new(3);
        
        // Add some test metrics
        for i in 1..=5 {
            let mut metrics = TrainingMetrics::default();
            metrics.loss.train_loss = i as f64;
            
            let count = aggregator.aggregate(&metrics).await.unwrap();
            assert!(count <= 3);
        }
        
        let result = aggregator.get_result().await.unwrap();
        assert!(result.is_some());
        
        // Should be average of last 3 values: (3 + 4 + 5) / 3 = 4.0
        assert!((result.unwrap() - 4.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_time_window_aggregator() {
        let mut aggregator = TimeWindowAggregator::new(Duration::from_secs(1), 10);
        
        let mut metrics = TrainingMetrics::default();
        metrics.loss.train_loss = 5.0;
        
        aggregator.aggregate(&metrics).await.unwrap();
        
        let result = aggregator.get_result().await.unwrap();
        assert_eq!(result, Some(5.0));
    }

    #[tokio::test]
    async fn test_statistical_aggregator() {
        let mut aggregator = StatisticalAggregator::new(100);
        
        // Add test data
        for i in 1..=10 {
            let mut metrics = TrainingMetrics::default();
            metrics.loss.train_loss = i as f64;
            aggregator.aggregate(&metrics).await.unwrap();
        }
        
        let stats = aggregator.get_statistics();
        assert_eq!(stats.count, 10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
        assert_eq!(stats.mean, 5.5);
    }
}