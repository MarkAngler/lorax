//! Tests for specialized trainers

#[cfg(test)]
mod supervised_tests {
    use super::super::*;
    use crate::hypernetwork::{HyperNetwork, HypernetworkConfig};
    use crate::models::{create_base_model, ModelType, BaseModelContainer};
    use crate::training::{DataLoader, DataLoaderConfig, SupervisedDataset};
    use candle_core::{Device, Tensor, DType};
    use std::sync::Arc;
    use parking_lot::RwLock;
    use tempfile::TempDir;
    
    fn create_test_hypernetwork() -> Arc<RwLock<HyperNetwork>> {
        let config = HypernetworkConfig {
            input_dim: 768,
            hidden_dims: vec![512, 256],
            output_strategy: crate::hypernetwork::OutputStrategy::TaskSpecific {
                heads: std::collections::HashMap::new(),
            },
            activation: crate::hypernetwork::Activation::ReLU,
            dropout: 0.1,
            use_layer_norm: true,
            use_residual: false,
            initialization: crate::hypernetwork::Initialization::Xavier,
        };
        
        let hypernetwork = HyperNetwork::new(config).unwrap();
        Arc::new(RwLock::new(hypernetwork))
    }
    
    fn create_test_base_model(device: &Device) -> BaseModelContainer {
        let model = create_base_model(
            ModelType::BERT,
            "bert-base",
            device,
            None
        ).unwrap();
        
        BaseModelContainer {
            model,
            config: crate::models::ModelConfig::default(),
            tokenizer: None,
            original_state: None,
        }
    }
    
    #[test]
    fn test_supervised_trainer_creation() {
        let device = Device::Cpu;
        let config = SupervisedTrainerConfig::default();
        let hypernetwork = create_test_hypernetwork();
        let base_model = create_test_base_model(&device);
        
        // Create dummy data loaders
        let train_loader = DataLoader::new(
            DataLoaderConfig::default(),
            vec![],
            device.clone(),
        ).unwrap();
        
        let trainer = SupervisedTrainer::new(
            config,
            hypernetwork,
            base_model,
            train_loader,
            None,
            device,
        );
        
        assert!(trainer.is_ok());
    }
    
    #[test]
    fn test_tokenizer_config() {
        let config = TokenizerConfig {
            tokenizer_path: None,
            padding: PaddingStrategy::Max,
            truncation: true,
            add_special_tokens: true,
            return_token_type_ids: true,
        };
        
        match config.padding {
            PaddingStrategy::Max => assert!(true),
            _ => panic!("Wrong padding strategy"),
        }
    }
    
    #[test]
    fn test_lora_adaptation_config() {
        let config = LoraAdaptationConfig {
            rank: 16,
            alpha: 32.0,
            dropout: 0.1,
            target_modules: vec![".*attention.*".to_string()],
            merge_weights: false,
            init_scale: 0.01,
            dynamic_rank: None,
        };
        
        assert_eq!(config.rank, 16);
        assert_eq!(config.alpha, 32.0);
        assert_eq!(config.target_modules.len(), 1);
    }
}

#[cfg(test)]
mod reconstruction_tests {
    use super::super::*;
    
    #[test]
    fn test_reconstruction_trainer_config() {
        let config = ReconstructionTrainerConfig::default();
        assert!(matches!(
            config.reconstruction.layer_weighting,
            reconstruction::LayerWeightingStrategy::Uniform
        ));
        assert_eq!(config.validation.best_model_metric, "eval_loss");
    }
}