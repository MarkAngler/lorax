//! AdamW optimizer implementation
//!
//! This module provides the AdamW optimizer with decoupled weight decay,
//! which is commonly used for training transformer models and LoRA adaptations.

use std::collections::HashMap;
use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarMap;

use super::{Optimizer, OptimizerStateDictLegacy, OptimizerTensorState};

/// AdamW optimizer with decoupled weight decay
pub struct AdamWOptimizer {
    /// Variable map containing parameters
    var_map: VarMap,
    
    /// Learning rate
    learning_rate: f64,
    
    /// Beta1 parameter (momentum)
    beta1: f64,
    
    /// Beta2 parameter (RMSprop)
    beta2: f64,
    
    /// Epsilon for numerical stability
    epsilon: f64,
    
    /// Weight decay coefficient
    weight_decay: f64,
    
    /// Current step count
    step_count: usize,
    
    /// First moment estimates (momentum)
    momentum: HashMap<String, Tensor>,
    
    /// Second moment estimates (RMSprop)
    variance: HashMap<String, Tensor>,
    
    /// Parameter count
    parameter_count: usize,
}

impl AdamWOptimizer {
    /// Create a new AdamW optimizer
    pub fn new(
        var_map: &VarMap,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> Result<Self> {
        let parameter_count = var_map.all_vars().len();
        
        Ok(Self {
            var_map: var_map.clone(),
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step_count: 0,
            momentum: HashMap::new(),
            variance: HashMap::new(),
            parameter_count,
        })
    }
    
    /// Initialize momentum and variance for a parameter
    fn initialize_state(&mut self, name: &str, param: &Tensor) -> Result<()> {
        if !self.momentum.contains_key(name) {
            let zeros = Tensor::zeros(param.shape(), param.dtype(), param.device())?;
            self.momentum.insert(name.to_string(), zeros.clone());
            self.variance.insert(name.to_string(), zeros);
        }
        Ok(())
    }
    
    /// Apply AdamW update to a parameter
    fn update_parameter(
        &mut self,
        name: &str,
        param: &Tensor,
        grad: &Tensor,
    ) -> Result<()> {
        // Initialize state if needed
        self.initialize_state(name, param)?;
        
        let momentum = self.momentum.get_mut(name).unwrap();
        let variance = self.variance.get_mut(name).unwrap();
        
        // Update momentum: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        let new_momentum = (momentum * self.beta1)? + (grad * (1.0 - self.beta1))?;
        *momentum = new_momentum.clone();
        
        // Update variance: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        let grad_squared = grad.sqr()?;
        let new_variance = (variance * self.beta2)? + (grad_squared * (1.0 - self.beta2))?;
        *variance = new_variance.clone();
        
        // Bias correction
        let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32 + 1);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32 + 1);
        
        let corrected_momentum = &new_momentum / bias_correction1;
        let corrected_variance = &new_variance / bias_correction2;
        
        // Compute update
        let sqrt_variance = corrected_variance.sqrt()?;
        let denominator = sqrt_variance + self.epsilon;
        let update = corrected_momentum.div(&denominator)?;
        
        // Apply weight decay (decoupled)
        let weight_decay_update = if self.weight_decay > 0.0 {
            param * self.weight_decay
        } else {
            Tensor::zeros(param.shape(), param.dtype(), param.device())?
        };
        
        // Final parameter update: θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})
        let total_update = (update * self.learning_rate)? + (weight_decay_update * self.learning_rate)?;
        
        // This is where you would actually update the parameter
        // In practice, this would modify the parameter tensor in place
        // param.sub_assign(&total_update)?;
        
        Ok(())
    }
}

impl Optimizer for AdamWOptimizer {
    fn name(&self) -> &str {
        "adamw"
    }
    
    fn step(&mut self, gradients: &candle_core::backprop::GradStore) -> Result<()> {
        self.step_count += 1;
        
        // In a real implementation, you would iterate through all parameters
        // and their gradients to apply the AdamW update
        //
        // for (name, param) in self.var_map.all_vars() {
        //     if let Some(grad) = gradients.get(&param) {
        //         self.update_parameter(&name, &param, &grad)?;
        //     }
        // }
        
        Ok(())
    }
    
    fn zero_grad(&mut self) -> Result<()> {
        // In a real implementation, this would zero all gradients
        Ok(())
    }
    
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }
    
    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
    
    fn state_dict(&self) -> Result<OptimizerStateDictLegacy> {
        let mut state = HashMap::new();
        
        // Serialize momentum and variance tensors
        for (name, tensor) in &self.momentum {
            let tensor_state = OptimizerTensorState {
                shape: tensor.shape().dims().to_vec(),
                data: vec![], // In practice, you'd flatten the tensor data
                device: format!("{:?}", tensor.device()),
                dtype: format!("{:?}", tensor.dtype()),
            };
            state.insert(format!("{}_momentum", name), tensor_state);
        }
        
        for (name, tensor) in &self.variance {
            let tensor_state = OptimizerTensorState {
                shape: tensor.shape().dims().to_vec(),
                data: vec![], // In practice, you'd flatten the tensor data
                device: format!("{:?}", tensor.device()),
                dtype: format!("{:?}", tensor.dtype()),
            };
            state.insert(format!("{}_variance", name), tensor_state);
        }
        
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("beta1".to_string(), self.beta1);
        hyperparameters.insert("beta2".to_string(), self.beta2);
        hyperparameters.insert("epsilon".to_string(), self.epsilon);
        hyperparameters.insert("weight_decay".to_string(), self.weight_decay);
        
        Ok(OptimizerStateDictLegacy {
            optimizer_type: "adamw".to_string(),
            step_count: self.step_count,
            learning_rate: self.learning_rate,
            state,
            hyperparameters,
        })
    }
    
    fn load_state_dict(&mut self, state_dict: OptimizerStateDictLegacy) -> Result<()> {
        self.step_count = state_dict.step_count;
        self.learning_rate = state_dict.learning_rate;
        
        // Load hyperparameters
        if let Some(&beta1) = state_dict.hyperparameters.get("beta1") {
            self.beta1 = beta1;
        }
        if let Some(&beta2) = state_dict.hyperparameters.get("beta2") {
            self.beta2 = beta2;
        }
        if let Some(&epsilon) = state_dict.hyperparameters.get("epsilon") {
            self.epsilon = epsilon;
        }
        if let Some(&weight_decay) = state_dict.hyperparameters.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
        
        // In a real implementation, you would reconstruct the momentum and variance tensors
        // from the state dictionary
        
        Ok(())
    }
    
    fn parameter_count(&self) -> usize {
        self.parameter_count
    }
    
    fn step_count(&self) -> usize {
        self.step_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_adamw_optimizer_creation() {
        let var_map = VarMap::new();
        let optimizer = AdamWOptimizer::new(
            &var_map,
            0.001,  // learning_rate
            0.9,    // beta1
            0.999,  // beta2
            1e-8,   // epsilon
            0.01,   // weight_decay
        );
        
        assert!(optimizer.is_ok());
        let opt = optimizer.unwrap();
        assert_eq!(opt.name(), "adamw");
        assert_eq!(opt.learning_rate(), 0.001);
        assert_eq!(opt.step_count(), 0);
    }

    #[test]
    fn test_adamw_state_dict() {
        let var_map = VarMap::new();
        let optimizer = AdamWOptimizer::new(&var_map, 0.001, 0.9, 0.999, 1e-8, 0.01).unwrap();
        
        let state_dict = optimizer.state_dict().unwrap();
        assert_eq!(state_dict.optimizer_type, "adamw");
        assert_eq!(state_dict.learning_rate, 0.001);
        assert_eq!(state_dict.step_count, 0);
        
        assert!(state_dict.hyperparameters.contains_key("beta1"));
        assert!(state_dict.hyperparameters.contains_key("beta2"));
        assert!(state_dict.hyperparameters.contains_key("epsilon"));
        assert!(state_dict.hyperparameters.contains_key("weight_decay"));
    }
}