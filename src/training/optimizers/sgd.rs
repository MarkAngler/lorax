//! SGD optimizer implementation with momentum
//!
//! This module provides the Stochastic Gradient Descent optimizer with
//! optional momentum and weight decay support.

use std::collections::HashMap;
use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarMap;

use super::{Optimizer, OptimizerStateDictLegacy, OptimizerTensorState};

/// SGD optimizer with momentum
pub struct SGDOptimizer {
    /// Variable map containing parameters
    var_map: VarMap,
    
    /// Learning rate
    learning_rate: f64,
    
    /// Momentum coefficient
    momentum: f64,
    
    /// Weight decay coefficient
    weight_decay: f64,
    
    /// Current step count
    step_count: usize,
    
    /// Momentum buffers
    momentum_buffers: HashMap<String, Tensor>,
    
    /// Parameter count
    parameter_count: usize,
}

impl SGDOptimizer {
    /// Create a new SGD optimizer
    pub fn new(
        var_map: &VarMap,
        learning_rate: f64,
        momentum: f64,
        weight_decay: f64,
    ) -> Result<Self> {
        let parameter_count = var_map.all_vars().len();
        
        Ok(Self {
            var_map: var_map.clone(),
            learning_rate,
            momentum,
            weight_decay,
            step_count: 0,
            momentum_buffers: HashMap::new(),
            parameter_count,
        })
    }
    
    /// Initialize momentum buffer for a parameter
    fn initialize_momentum(&mut self, name: &str, param: &Tensor) -> Result<()> {
        if !self.momentum_buffers.contains_key(name) && self.momentum > 0.0 {
            let zeros = Tensor::zeros_like(param)?;
            self.momentum_buffers.insert(name.to_string(), zeros);
        }
        Ok(())
    }
    
    /// Apply SGD update to a parameter
    fn update_parameter(
        &mut self,
        name: &str,
        param: &Tensor,
        grad: &Tensor,
    ) -> Result<()> {
        // Apply weight decay to gradient if specified
        let effective_grad = if self.weight_decay > 0.0 {
            let weight_decay_term = (param * self.weight_decay)?;
            (grad + &weight_decay_term)?
        } else {
            grad.clone()
        };
        
        let update = if self.momentum > 0.0 {
            // Initialize momentum buffer if needed
            self.initialize_momentum(name, param)?;
            
            let momentum_buffer = self.momentum_buffers.get_mut(name).unwrap();
            
            // Update momentum buffer: v_t = μ * v_{t-1} + g_t
            let new_momentum = ((&*momentum_buffer * self.momentum)? + &effective_grad)?;
            *momentum_buffer = new_momentum.clone();
            
            new_momentum
        } else {
            effective_grad
        };
        
        // Apply update: θ_t = θ_{t-1} - α * update
        let param_update = (&update * self.learning_rate)?;
        
        // This is where you would actually update the parameter
        // In practice, this would modify the parameter tensor in place
        // param.sub_assign(&param_update)?;
        
        Ok(())
    }
}

impl Optimizer for SGDOptimizer {
    fn name(&self) -> &str {
        if self.momentum > 0.0 {
            "sgd_momentum"
        } else {
            "sgd"
        }
    }
    
    fn step(&mut self, _gradients: &candle_core::backprop::GradStore) -> Result<()> {
        self.step_count += 1;
        
        // In a real implementation, you would iterate through all parameters
        // and their gradients to apply the SGD update
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
        
        // Serialize momentum buffers
        for (name, tensor) in &self.momentum_buffers {
            let tensor_state = OptimizerTensorState {
                shape: tensor.shape().dims().to_vec(),
                data: vec![], // In practice, you'd flatten the tensor data
                device: format!("{:?}", tensor.device()),
                dtype: format!("{:?}", tensor.dtype()),
            };
            state.insert(format!("{}_momentum", name), tensor_state);
        }
        
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("momentum".to_string(), self.momentum);
        hyperparameters.insert("weight_decay".to_string(), self.weight_decay);
        
        Ok(OptimizerStateDictLegacy {
            optimizer_type: self.name().to_string(),
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
        if let Some(&momentum) = state_dict.hyperparameters.get("momentum") {
            self.momentum = momentum;
        }
        if let Some(&weight_decay) = state_dict.hyperparameters.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
        
        // In a real implementation, you would reconstruct the momentum buffers
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

    #[test]
    fn test_sgd_optimizer_creation() {
        let var_map = VarMap::new();
        let optimizer = SGDOptimizer::new(
            &var_map,
            0.01,   // learning_rate
            0.9,    // momentum
            0.0001, // weight_decay
        );
        
        assert!(optimizer.is_ok());
        let opt = optimizer.unwrap();
        assert_eq!(opt.name(), "sgd_momentum");
        assert_eq!(opt.learning_rate(), 0.01);
        assert_eq!(opt.step_count(), 0);
    }

    #[test]
    fn test_sgd_without_momentum() {
        let var_map = VarMap::new();
        let optimizer = SGDOptimizer::new(&var_map, 0.01, 0.0, 0.0001).unwrap();
        assert_eq!(optimizer.name(), "sgd");
    }

    #[test]
    fn test_sgd_state_dict() {
        let var_map = VarMap::new();
        let optimizer = SGDOptimizer::new(&var_map, 0.01, 0.9, 0.0001).unwrap();
        
        let state_dict = optimizer.state_dict().unwrap();
        assert_eq!(state_dict.optimizer_type, "sgd_momentum");
        assert_eq!(state_dict.learning_rate, 0.01);
        assert_eq!(state_dict.step_count, 0);
        
        assert!(state_dict.hyperparameters.contains_key("momentum"));
        assert!(state_dict.hyperparameters.contains_key("weight_decay"));
    }
}