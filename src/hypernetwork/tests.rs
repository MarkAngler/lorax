#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypernetwork::*;
    use ndarray::Array1;
    use approx::assert_relative_eq;

    #[test]
    fn test_model_sizes() {
        assert_eq!(ModelSize::Large.num_layers(), 2);
        assert_eq!(ModelSize::Large.hidden_dim(), 256);
        
        assert_eq!(ModelSize::Medium.num_layers(), 4);
        assert_eq!(ModelSize::Medium.hidden_dim(), 512);
        
        assert_eq!(ModelSize::Small.num_layers(), 8);
        assert_eq!(ModelSize::Small.hidden_dim(), 1024);
    }

    #[test]
    fn test_hypernetwork_creation() {
        let configs = vec![
            HyperNetworkConfig {
                model_size: ModelSize::Large,
                ..Default::default()
            },
            HyperNetworkConfig {
                model_size: ModelSize::Medium,
                ..Default::default()
            },
            HyperNetworkConfig {
                model_size: ModelSize::Small,
                ..Default::default()
            },
        ];

        for config in configs {
            let hypernetwork = HyperNetwork::new(config.clone());
            assert!(hypernetwork.is_ok());
            let hypernetwork = hypernetwork.unwrap();
            assert_eq!(hypernetwork.config().model_size.num_layers(), config.model_size.num_layers());
        }
    }

    #[test]
    fn test_forward_pass() {
        let config = HyperNetworkConfig::default();
        let hypernetwork = HyperNetwork::new(config).unwrap();
        
        // Create random input
        let input = Array1::from_vec(vec![0.1; 768]);
        
        // Test with GPT architecture
        let arch = TargetArchitecture::GPT {
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
        };
        
        let result = hypernetwork.generate_lora_params(&input, arch);
        assert!(result.is_ok());
        
        let lora_params = result.unwrap();
        assert_eq!(lora_params.rank, 16);
        assert!(!lora_params.layers.is_empty());
    }

    #[test]
    fn test_lora_generation_for_different_architectures() {
        let config = HyperNetworkConfig::default();
        let hypernetwork = HyperNetwork::new(config).unwrap();
        let input = Array1::from_vec(vec![0.1; 768]);
        
        // Test GPT
        let gpt_arch = TargetArchitecture::GPT {
            hidden_size: 768,
            num_layers: 2,
            num_heads: 12,
        };
        let gpt_result = hypernetwork.generate_lora_params(&input, gpt_arch);
        assert!(gpt_result.is_ok());
        
        // Test BERT
        let bert_arch = TargetArchitecture::BERT {
            hidden_size: 768,
            num_layers: 2,
            num_heads: 12,
        };
        let bert_result = hypernetwork.generate_lora_params(&input, bert_arch);
        assert!(bert_result.is_ok());
        
        // Test LLaMA
        let llama_arch = TargetArchitecture::LLaMA {
            hidden_size: 768,
            num_layers: 2,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let llama_result = hypernetwork.generate_lora_params(&input, llama_arch);
        assert!(llama_result.is_ok());
    }

    #[test]
    fn test_lora_application() {
        let weight = Array2::from_shape_vec((768, 768), vec![0.1; 768 * 768]).unwrap();
        let lora_params = LoRALayerParams {
            matrix_a: Array2::from_shape_vec((768, 16), vec![0.01; 768 * 16]).unwrap(),
            matrix_b: Array2::from_shape_vec((16, 768), vec![0.01; 16 * 768]).unwrap(),
            alpha: 32.0,
        };
        
        let modified_weight = apply_lora(&weight, &lora_params);
        
        // Check that weight has been modified
        assert_ne!(weight, modified_weight);
        
        // Check dimensions are preserved
        assert_eq!(weight.shape(), modified_weight.shape());
    }

    #[test]
    fn test_activation_functions() {
        let config = HyperNetworkConfig {
            activation: ActivationType::ReLU,
            ..Default::default()
        };
        
        let model = HyperNetworkModel::new(&config).unwrap();
        let input = Array1::from_vec(vec![-1.0, 0.0, 1.0, 2.0]);
        
        // Test ReLU
        let relu_out = model.apply_activation(&input);
        assert_eq!(relu_out[0], 0.0);
        assert_eq!(relu_out[1], 0.0);
        assert_eq!(relu_out[2], 1.0);
        assert_eq!(relu_out[3], 2.0);
    }

    #[test]
    fn test_layer_configs_generation() {
        let handler = ArchitectureHandler::new();
        
        let arch = TargetArchitecture::GPT {
            hidden_size: 768,
            num_layers: 2,
            num_heads: 12,
        };
        
        let configs = handler.get_layer_configs(&arch).unwrap();
        
        // GPT has 4 patterns per layer
        assert_eq!(configs.len(), 8);
        
        // Check first attention layer
        assert_eq!(configs[0].name, "transformer.h.0.attn.c_attn");
        assert_eq!(configs[0].in_features, 768);
        assert_eq!(configs[0].out_features, 2304); // 768 * 3 for Q,K,V
    }
}