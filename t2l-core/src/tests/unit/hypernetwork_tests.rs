//! Unit tests for T2L Hypernetwork components

use crate::hypernetwork::{HyperNetwork, HyperNetConfig, Architecture};
use crate::adapter::{LoRAAdapter, AdapterConfig};
use candle_core::{Device, Tensor, DType};
use rstest::*;
use proptest::prelude::*;
use assert_approx_eq::assert_approx_eq;

#[fixture]
fn device() -> Device {
    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
}

#[fixture]
fn config_l() -> HyperNetConfig {
    HyperNetConfig {
        architecture: Architecture::L,
        input_dim: 1024,
        hidden_dim: 4096,
        output_dim: 32,
        num_layers: 8,
        dropout: 0.1,
        use_layer_norm: true,
        activation: "gelu".to_string(),
    }
}

#[fixture]
fn config_m() -> HyperNetConfig {
    HyperNetConfig {
        architecture: Architecture::M,
        input_dim: 768,
        hidden_dim: 2048,
        output_dim: 16,
        num_layers: 6,
        dropout: 0.1,
        use_layer_norm: true,
        activation: "relu".to_string(),
    }
}

#[fixture]
fn config_s() -> HyperNetConfig {
    HyperNetConfig {
        architecture: Architecture::S,
        input_dim: 512,
        hidden_dim: 1024,
        output_dim: 8,
        num_layers: 4,
        dropout: 0.05,
        use_layer_norm: true,
        activation: "tanh".to_string(),
    }
}

#[rstest]
#[case(config_l())]
#[case(config_m())]
#[case(config_s())]
fn test_hypernetwork_initialization(
    #[case] config: HyperNetConfig,
    device: Device,
) {
    let hypernetwork = HyperNetwork::new(&config, &device).unwrap();
    
    // Verify architecture
    assert_eq!(hypernetwork.config().architecture, config.architecture);
    assert_eq!(hypernetwork.config().num_layers, config.num_layers);
    assert_eq!(hypernetwork.config().hidden_dim, config.hidden_dim);
}

#[rstest]
fn test_hypernetwork_forward_shape(
    config_l: HyperNetConfig,
    device: Device,
) {
    let hypernetwork = HyperNetwork::new(&config_l, &device).unwrap();
    let batch_size = 4;
    let input = Tensor::randn(
        0f32, 1f32, 
        &[batch_size, config_l.input_dim], 
        &device
    ).unwrap();
    
    let output = hypernetwork.forward(&input).unwrap();
    
    // Check output shape
    let expected_shape = vec![batch_size, config_l.output_dim];
    assert_eq!(output.shape().dims(), &expected_shape);
}

#[rstest]
fn test_hypernetwork_deterministic_output(
    config_m: HyperNetConfig,
    device: Device,
) {
    let hypernetwork = HyperNetwork::new(&config_m, &device).unwrap();
    hypernetwork.eval(); // Set to evaluation mode
    
    let input = Tensor::ones(&[1, config_m.input_dim], DType::F32, &device).unwrap();
    
    let output1 = hypernetwork.forward(&input).unwrap();
    let output2 = hypernetwork.forward(&input).unwrap();
    
    // Outputs should be identical in eval mode
    let diff = (&output1 - &output2).unwrap().abs().unwrap().max_all().unwrap();
    assert!(diff.to_scalar::<f32>().unwrap() < 1e-6);
}

#[rstest]
#[case(Architecture::L, 130_000_000)] // ~130M parameters
#[case(Architecture::M, 50_000_000)]   // ~50M parameters
#[case(Architecture::S, 10_000_000)]   // ~10M parameters
fn test_hypernetwork_parameter_count(
    #[case] architecture: Architecture,
    #[case] expected_range: usize,
    device: Device,
) {
    let config = HyperNetConfig {
        architecture,
        input_dim: 1024,
        hidden_dim: match architecture {
            Architecture::L => 4096,
            Architecture::M => 2048,
            Architecture::S => 1024,
        },
        output_dim: 32,
        num_layers: match architecture {
            Architecture::L => 8,
            Architecture::M => 6,
            Architecture::S => 4,
        },
        dropout: 0.1,
        use_layer_norm: true,
        activation: "gelu".to_string(),
    };
    
    let hypernetwork = HyperNetwork::new(&config, &device).unwrap();
    let param_count = hypernetwork.total_parameters();
    
    // Check parameter count is within expected range (±20%)
    let lower_bound = (expected_range as f64 * 0.8) as usize;
    let upper_bound = (expected_range as f64 * 1.2) as usize;
    assert!(
        param_count >= lower_bound && param_count <= upper_bound,
        "Parameter count {} not in range [{}, {}]",
        param_count, lower_bound, upper_bound
    );
}

#[rstest]
fn test_lora_generation_shapes(
    config_l: HyperNetConfig,
    device: Device,
) {
    let hypernetwork = HyperNetwork::new(&config_l, &device).unwrap();
    let task_embedding = Tensor::randn(
        0f32, 1f32,
        &[1, config_l.input_dim],
        &device
    ).unwrap();
    
    let adapter_config = AdapterConfig {
        rank: 16,
        alpha: 32.0,
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        dropout: 0.0,
    };
    
    let lora_params = hypernetwork.generate_lora_parameters(
        &task_embedding,
        &adapter_config
    ).unwrap();
    
    // Verify LoRA parameter shapes
    for (module_name, (lora_a, lora_b)) in &lora_params {
        assert!(adapter_config.target_modules.contains(module_name));
        
        // Check shape compatibility
        let a_shape = lora_a.shape().dims();
        let b_shape = lora_b.shape().dims();
        
        assert_eq!(a_shape[1], adapter_config.rank);
        assert_eq!(b_shape[0], adapter_config.rank);
    }
}

#[rstest]
fn test_gradient_flow(
    config_s: HyperNetConfig,
    device: Device,
) {
    let mut hypernetwork = HyperNetwork::new(&config_s, &device).unwrap();
    hypernetwork.train();
    
    let input = Tensor::randn(
        0f32, 1f32,
        &[2, config_s.input_dim],
        &device
    ).unwrap();
    
    let output = hypernetwork.forward(&input).unwrap();
    let loss = output.mean_all().unwrap();
    
    // Compute gradients
    let grads = loss.backward().unwrap();
    
    // Check that gradients exist and are non-zero
    for (name, param) in hypernetwork.named_parameters() {
        let grad = grads.get(&param).unwrap();
        let grad_norm = grad.sqr().unwrap().sum_all().unwrap().sqrt().unwrap();
        let grad_norm_scalar = grad_norm.to_scalar::<f32>().unwrap();
        
        assert!(
            grad_norm_scalar > 0.0,
            "Zero gradient for parameter {}",
            name
        );
    }
}

proptest! {
    #[test]
    fn test_hypernetwork_numerical_stability(
        batch_size in 1usize..16,
        scale in 0.1f32..10.0,
    ) {
        let device = Device::Cpu;
        let config = HyperNetConfig {
            architecture: Architecture::S,
            input_dim: 64,
            hidden_dim: 128,
            output_dim: 8,
            num_layers: 2,
            dropout: 0.0,
            use_layer_norm: true,
            activation: "relu".to_string(),
        };
        
        let hypernetwork = HyperNetwork::new(&config, &device).unwrap();
        hypernetwork.eval();
        
        // Create scaled input
        let input = Tensor::randn(
            0f32, scale,
            &[batch_size, config.input_dim],
            &device
        ).unwrap();
        
        let output = hypernetwork.forward(&input).unwrap();
        
        // Check for NaN or Inf
        let output_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for val in output_vec {
            prop_assert!(val.is_finite(), "Output contains NaN or Inf");
        }
    }
}

#[test]
fn test_hypernetwork_save_load() {
    let device = Device::Cpu;
    let config = HyperNetConfig {
        architecture: Architecture::M,
        input_dim: 128,
        hidden_dim: 256,
        output_dim: 16,
        num_layers: 3,
        dropout: 0.1,
        use_layer_norm: true,
        activation: "gelu".to_string(),
    };
    
    let hypernetwork = HyperNetwork::new(&config, &device).unwrap();
    
    // Generate output with original model
    let input = Tensor::ones(&[1, config.input_dim], DType::F32, &device).unwrap();
    let output_original = hypernetwork.forward(&input).unwrap();
    
    // Save model
    let temp_dir = tempfile::tempdir().unwrap();
    let save_path = temp_dir.path().join("hypernetwork.safetensors");
    hypernetwork.save(&save_path).unwrap();
    
    // Load model
    let loaded_hypernetwork = HyperNetwork::load(&save_path, &config, &device).unwrap();
    let output_loaded = loaded_hypernetwork.forward(&input).unwrap();
    
    // Compare outputs
    let diff = (&output_original - &output_loaded).unwrap()
        .abs().unwrap()
        .max_all().unwrap()
        .to_scalar::<f32>().unwrap();
    
    assert!(diff < 1e-6, "Loaded model output differs from original");
}

#[rstest]
#[case("gelu")]
#[case("relu")]
#[case("tanh")]
#[case("swish")]
fn test_activation_functions(
    #[case] activation: &str,
    device: Device,
) {
    let config = HyperNetConfig {
        architecture: Architecture::S,
        input_dim: 32,
        hidden_dim: 64,
        output_dim: 8,
        num_layers: 2,
        dropout: 0.0,
        use_layer_norm: false,
        activation: activation.to_string(),
    };
    
    let hypernetwork = HyperNetwork::new(&config, &device).unwrap();
    let input = Tensor::randn(0f32, 1f32, &[1, config.input_dim], &device).unwrap();
    
    // Should not panic
    let output = hypernetwork.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, config.output_dim]);
}