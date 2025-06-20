//! Learning rate schedulers for training optimization
//!
//! This module provides various learning rate scheduling strategies including
//! linear, cosine, exponential, and step-based schedules with warmup support.

use std::collections::HashMap;
use anyhow::Result;

use super::{Scheduler, SchedulerStateDictLegacy};

/// Linear learning rate scheduler
pub struct LinearScheduler {
    /// Base learning rate
    base_lr: f64,
    
    /// Minimum learning rate
    min_lr: f64,
    
    /// Total training steps
    total_steps: usize,
    
    /// Warmup steps
    warmup_steps: usize,
    
    /// Current step
    current_step: usize,
    
    /// Current learning rate
    current_lr: f64,
}

impl LinearScheduler {
    /// Create a new linear scheduler
    pub fn new(base_lr: f64, min_lr: f64, total_steps: usize, warmup_steps: usize) -> Self {
        Self {
            base_lr,
            min_lr,
            total_steps,
            warmup_steps,
            current_step: 0,
            current_lr: base_lr,
        }
    }
    
    /// Calculate learning rate for current step
    fn calculate_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else {
            // Linear decay
            let decay_steps = self.total_steps - self.warmup_steps;
            let decay_progress = (self.current_step - self.warmup_steps) as f64 / decay_steps as f64;
            let decay_factor = 1.0 - decay_progress.min(1.0);
            
            self.min_lr + (self.base_lr - self.min_lr) * decay_factor
        }
    }
}

impl Scheduler for LinearScheduler {
    fn name(&self) -> &str {
        "linear"
    }
    
    fn step(&mut self, _metric: Option<f64>) {
        self.current_step += 1;
        self.current_lr = self.calculate_lr();
    }
    
    fn get_lr(&self) -> f64 {
        self.current_lr
    }
    
    fn state_dict(&self) -> Result<SchedulerStateDictLegacy> {
        let mut state = HashMap::new();
        state.insert("total_steps".to_string(), self.total_steps as f64);
        state.insert("warmup_steps".to_string(), self.warmup_steps as f64);
        
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("min_lr".to_string(), self.min_lr);
        
        Ok(SchedulerStateDictLegacy {
            scheduler_type: "linear".to_string(),
            step_count: self.current_step,
            current_lr: self.current_lr,
            base_lr: self.base_lr,
            state,
            hyperparameters,
        })
    }
    
    fn load_state_dict(&mut self, state_dict: SchedulerStateDictLegacy) -> Result<()> {
        self.current_step = state_dict.step_count;
        self.current_lr = state_dict.current_lr;
        self.base_lr = state_dict.base_lr;
        
        if let Some(&min_lr) = state_dict.hyperparameters.get("min_lr") {
            self.min_lr = min_lr;
        }
        
        if let Some(&total_steps) = state_dict.state.get("total_steps") {
            self.total_steps = total_steps as usize;
        }
        
        if let Some(&warmup_steps) = state_dict.state.get("warmup_steps") {
            self.warmup_steps = warmup_steps as usize;
        }
        
        Ok(())
    }
    
    fn is_done(&self) -> bool {
        self.current_step >= self.total_steps
    }
    
    fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = self.base_lr;
    }
}

/// Cosine annealing learning rate scheduler
pub struct CosineScheduler {
    /// Base learning rate
    base_lr: f64,
    
    /// Minimum learning rate
    min_lr: f64,
    
    /// Total training steps
    total_steps: usize,
    
    /// Warmup steps
    warmup_steps: usize,
    
    /// Current step
    current_step: usize,
    
    /// Current learning rate
    current_lr: f64,
}

impl CosineScheduler {
    /// Create a new cosine scheduler
    pub fn new(base_lr: f64, min_lr: f64, total_steps: usize, warmup_steps: usize) -> Self {
        Self {
            base_lr,
            min_lr,
            total_steps,
            warmup_steps,
            current_step: 0,
            current_lr: base_lr,
        }
    }
    
    /// Calculate learning rate for current step
    fn calculate_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else {
            // Cosine annealing
            let decay_steps = self.total_steps - self.warmup_steps;
            let decay_progress = (self.current_step - self.warmup_steps) as f64 / decay_steps as f64;
            let cosine_factor = 0.5 * (1.0 + (std::f64::consts::PI * decay_progress.min(1.0)).cos());
            
            self.min_lr + (self.base_lr - self.min_lr) * cosine_factor
        }
    }
}

impl Scheduler for CosineScheduler {
    fn name(&self) -> &str {
        "cosine"
    }
    
    fn step(&mut self, _metric: Option<f64>) {
        self.current_step += 1;
        self.current_lr = self.calculate_lr();
    }
    
    fn get_lr(&self) -> f64 {
        self.current_lr
    }
    
    fn state_dict(&self) -> Result<SchedulerStateDictLegacy> {
        let mut state = HashMap::new();
        state.insert("total_steps".to_string(), self.total_steps as f64);
        state.insert("warmup_steps".to_string(), self.warmup_steps as f64);
        
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("min_lr".to_string(), self.min_lr);
        
        Ok(SchedulerStateDictLegacy {
            scheduler_type: "cosine".to_string(),
            step_count: self.current_step,
            current_lr: self.current_lr,
            base_lr: self.base_lr,
            state,
            hyperparameters,
        })
    }
    
    fn load_state_dict(&mut self, state_dict: SchedulerStateDictLegacy) -> Result<()> {
        self.current_step = state_dict.step_count;
        self.current_lr = state_dict.current_lr;
        self.base_lr = state_dict.base_lr;
        
        if let Some(&min_lr) = state_dict.hyperparameters.get("min_lr") {
            self.min_lr = min_lr;
        }
        
        if let Some(&total_steps) = state_dict.state.get("total_steps") {
            self.total_steps = total_steps as usize;
        }
        
        if let Some(&warmup_steps) = state_dict.state.get("warmup_steps") {
            self.warmup_steps = warmup_steps as usize;
        }
        
        Ok(())
    }
    
    fn is_done(&self) -> bool {
        self.current_step >= self.total_steps
    }
    
    fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = self.base_lr;
    }
}

/// Exponential learning rate scheduler
pub struct ExponentialScheduler {
    /// Base learning rate
    base_lr: f64,
    
    /// Decay factor
    decay_factor: f64,
    
    /// Warmup steps
    warmup_steps: usize,
    
    /// Current step
    current_step: usize,
    
    /// Current learning rate
    current_lr: f64,
}

impl ExponentialScheduler {
    /// Create a new exponential scheduler
    pub fn new(base_lr: f64, decay_factor: f64, warmup_steps: usize) -> Self {
        Self {
            base_lr,
            decay_factor,
            warmup_steps,
            current_step: 0,
            current_lr: base_lr,
        }
    }
    
    /// Calculate learning rate for current step
    fn calculate_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else {
            // Exponential decay
            let decay_steps = self.current_step - self.warmup_steps;
            self.base_lr * self.decay_factor.powi(decay_steps as i32)
        }
    }
}

impl Scheduler for ExponentialScheduler {
    fn name(&self) -> &str {
        "exponential"
    }
    
    fn step(&mut self, _metric: Option<f64>) {
        self.current_step += 1;
        self.current_lr = self.calculate_lr();
    }
    
    fn get_lr(&self) -> f64 {
        self.current_lr
    }
    
    fn state_dict(&self) -> Result<SchedulerStateDictLegacy> {
        let mut state = HashMap::new();
        state.insert("warmup_steps".to_string(), self.warmup_steps as f64);
        
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("decay_factor".to_string(), self.decay_factor);
        
        Ok(SchedulerStateDictLegacy {
            scheduler_type: "exponential".to_string(),
            step_count: self.current_step,
            current_lr: self.current_lr,
            base_lr: self.base_lr,
            state,
            hyperparameters,
        })
    }
    
    fn load_state_dict(&mut self, state_dict: SchedulerStateDictLegacy) -> Result<()> {
        self.current_step = state_dict.step_count;
        self.current_lr = state_dict.current_lr;
        self.base_lr = state_dict.base_lr;
        
        if let Some(&decay_factor) = state_dict.hyperparameters.get("decay_factor") {
            self.decay_factor = decay_factor;
        }
        
        if let Some(&warmup_steps) = state_dict.state.get("warmup_steps") {
            self.warmup_steps = warmup_steps as usize;
        }
        
        Ok(())
    }
    
    fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = self.base_lr;
    }
}

/// Step learning rate scheduler
pub struct StepScheduler {
    /// Base learning rate
    base_lr: f64,
    
    /// Step size (steps between decay)
    step_size: usize,
    
    /// Decay factor
    decay_factor: f64,
    
    /// Warmup steps
    warmup_steps: usize,
    
    /// Current step
    current_step: usize,
    
    /// Current learning rate
    current_lr: f64,
}

impl StepScheduler {
    /// Create a new step scheduler
    pub fn new(base_lr: f64, step_size: usize, decay_factor: f64, warmup_steps: usize) -> Self {
        Self {
            base_lr,
            step_size,
            decay_factor,
            warmup_steps,
            current_step: 0,
            current_lr: base_lr,
        }
    }
    
    /// Calculate learning rate for current step
    fn calculate_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else {
            // Step decay
            let decay_steps = self.current_step - self.warmup_steps;
            let decay_epochs = decay_steps / self.step_size;
            self.base_lr * self.decay_factor.powi(decay_epochs as i32)
        }
    }
}

impl Scheduler for StepScheduler {
    fn name(&self) -> &str {
        "step"
    }
    
    fn step(&mut self, _metric: Option<f64>) {
        self.current_step += 1;
        self.current_lr = self.calculate_lr();
    }
    
    fn get_lr(&self) -> f64 {
        self.current_lr
    }
    
    fn state_dict(&self) -> Result<SchedulerStateDictLegacy> {
        let mut state = HashMap::new();
        state.insert("step_size".to_string(), self.step_size as f64);
        state.insert("warmup_steps".to_string(), self.warmup_steps as f64);
        
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("decay_factor".to_string(), self.decay_factor);
        
        Ok(SchedulerStateDictLegacy {
            scheduler_type: "step".to_string(),
            step_count: self.current_step,
            current_lr: self.current_lr,
            base_lr: self.base_lr,
            state,
            hyperparameters,
        })
    }
    
    fn load_state_dict(&mut self, state_dict: SchedulerStateDictLegacy) -> Result<()> {
        self.current_step = state_dict.step_count;
        self.current_lr = state_dict.current_lr;
        self.base_lr = state_dict.base_lr;
        
        if let Some(&decay_factor) = state_dict.hyperparameters.get("decay_factor") {
            self.decay_factor = decay_factor;
        }
        
        if let Some(&step_size) = state_dict.state.get("step_size") {
            self.step_size = step_size as usize;
        }
        
        if let Some(&warmup_steps) = state_dict.state.get("warmup_steps") {
            self.warmup_steps = warmup_steps as usize;
        }
        
        Ok(())
    }
    
    fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = self.base_lr;
    }
}

/// Constant learning rate scheduler (no decay)
pub struct ConstantScheduler {
    /// Base learning rate
    base_lr: f64,
    
    /// Current step
    current_step: usize,
}

impl ConstantScheduler {
    /// Create a new constant scheduler
    pub fn new(base_lr: f64) -> Self {
        Self {
            base_lr,
            current_step: 0,
        }
    }
}

impl Scheduler for ConstantScheduler {
    fn name(&self) -> &str {
        "constant"
    }
    
    fn step(&mut self, _metric: Option<f64>) {
        self.current_step += 1;
    }
    
    fn get_lr(&self) -> f64 {
        self.base_lr
    }
    
    fn state_dict(&self) -> Result<SchedulerStateDictLegacy> {
        Ok(SchedulerStateDictLegacy {
            scheduler_type: "constant".to_string(),
            step_count: self.current_step,
            current_lr: self.base_lr,
            base_lr: self.base_lr,
            state: HashMap::new(),
            hyperparameters: HashMap::new(),
        })
    }
    
    fn load_state_dict(&mut self, state_dict: SchedulerStateDictLegacy) -> Result<()> {
        self.current_step = state_dict.step_count;
        self.base_lr = state_dict.base_lr;
        Ok(())
    }
    
    fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// Learning rate scheduler trait object factory
pub fn create_scheduler(
    scheduler_type: &str,
    base_lr: f64,
    config: HashMap<String, f64>,
) -> Result<Box<dyn Scheduler + Send + Sync>> {
    match scheduler_type {
        "linear" => {
            let total_steps = config.get("total_steps").copied().unwrap_or(1000.0) as usize;
            let min_lr = config.get("min_lr").copied().unwrap_or(0.0);
            let warmup_steps = config.get("warmup_steps").copied().unwrap_or(0.0) as usize;
            
            Ok(Box::new(LinearScheduler::new(base_lr, min_lr, total_steps, warmup_steps)))
        }
        "cosine" => {
            let total_steps = config.get("total_steps").copied().unwrap_or(1000.0) as usize;
            let min_lr = config.get("min_lr").copied().unwrap_or(0.0);
            let warmup_steps = config.get("warmup_steps").copied().unwrap_or(0.0) as usize;
            
            Ok(Box::new(CosineScheduler::new(base_lr, min_lr, total_steps, warmup_steps)))
        }
        "exponential" => {
            let decay_factor = config.get("decay_factor").copied().unwrap_or(0.95);
            let warmup_steps = config.get("warmup_steps").copied().unwrap_or(0.0) as usize;
            
            Ok(Box::new(ExponentialScheduler::new(base_lr, decay_factor, warmup_steps)))
        }
        "step" => {
            let step_size = config.get("step_size").copied().unwrap_or(100.0) as usize;
            let decay_factor = config.get("decay_factor").copied().unwrap_or(0.1);
            let warmup_steps = config.get("warmup_steps").copied().unwrap_or(0.0) as usize;
            
            Ok(Box::new(StepScheduler::new(base_lr, step_size, decay_factor, warmup_steps)))
        }
        "constant" => {
            Ok(Box::new(ConstantScheduler::new(base_lr)))
        }
        _ => Err(anyhow::anyhow!("Unknown scheduler type: {}", scheduler_type)),
    }
}

/// Generic learning rate scheduler interface
pub trait LearningRateScheduler: Scheduler {}

impl<T: Scheduler> LearningRateScheduler for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_scheduler() {
        let mut scheduler = LinearScheduler::new(0.001, 0.0, 1000, 100);
        
        assert_eq!(scheduler.name(), "linear");
        assert_eq!(scheduler.get_lr(), 0.001);
        
        // Test warmup
        for _ in 0..50 {
            scheduler.step(None);
        }
        assert!(scheduler.get_lr() < 0.001);
        
        // Test after warmup
        for _ in 50..200 {
            scheduler.step(None);
        }
        assert!(scheduler.get_lr() > 0.0);
    }

    #[test]
    fn test_cosine_scheduler() {
        let mut scheduler = CosineScheduler::new(0.001, 0.0, 1000, 100);
        
        assert_eq!(scheduler.name(), "cosine");
        
        // Test full schedule
        for _ in 0..1000 {
            scheduler.step(None);
        }
        
        assert!(scheduler.is_done());
    }

    #[test]
    fn test_constant_scheduler() {
        let mut scheduler = ConstantScheduler::new(0.001);
        
        assert_eq!(scheduler.name(), "constant");
        assert_eq!(scheduler.get_lr(), 0.001);
        
        // Should remain constant
        for _ in 0..1000 {
            scheduler.step(None);
        }
        assert_eq!(scheduler.get_lr(), 0.001);
    }

    #[test]
    fn test_scheduler_state_dict() {
        let mut scheduler = LinearScheduler::new(0.001, 0.0, 1000, 100);
        
        // Step a few times
        for _ in 0..50 {
            scheduler.step(None);
        }
        
        let state_dict = scheduler.state_dict().unwrap();
        assert_eq!(state_dict.step_count, 50);
        assert_eq!(state_dict.base_lr, 0.001);
        
        // Create new scheduler and load state
        let mut new_scheduler = LinearScheduler::new(0.002, 0.0, 500, 50);
        new_scheduler.load_state_dict(state_dict).unwrap();
        
        assert_eq!(new_scheduler.current_step, 50);
        assert_eq!(new_scheduler.base_lr, 0.001);
    }
}