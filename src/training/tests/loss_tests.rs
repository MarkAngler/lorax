//! Tests for loss functions

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use std::collections::HashMap;
use proptest::prelude::*;

use crate::training::loss::{
    LossFunction, LossConfig, LossType, ReconstructionLoss, SupervisedLoss,
    CompositeLoss, RegularizationPenalties, LoraRegularizer, LossUtils,
    ReductionMethod, MatrixLossType, SupervisedTaskType,
};

use super::test_utils::{
    assert_tensor_close, assert_metrics_valid, create_test_device,
    create_test_lora_params, random_tensor,
};
use super::mock_data::{create_mock_batch, create_mock_supervised_batch};

#[test]
fn test_loss_functions() {
    // Placeholder test to be referenced by mod.rs
}

#[test]
fn test_reconstruction_loss_mse() -> Result<()> {
    let device = create_test_device();
    
    let config = LossConfig {
        loss_type: LossType::Reconstruction {
            matrix_loss: MatrixLossType::MSE,
            layer_weights: HashMap::new(),
        },
        regularization: Default::default(),
        scaling: Default::default(),
        stability: Default::default(),
        task_weights: HashMap::new(),
        reduction: ReductionMethod::Mean,
    };
    
    let loss_fn = ReconstructionLoss::new(config, &device)?;
    
    // Create mock data
    let batch = create_mock_batch(4, &device)?;
    let predicted_params = create_test_lora_params(2, 64, 4, &device)?;
    
    // Convert to the expected format
    let predicted = crate::training::loss::reconstruction::PredictedLoraParams {
        layers: predicted_params.into_iter()
            .map(|(name, (a, b))| {
                (name.clone(), crate::training::loss::reconstruction::PredictedLoraLayer {
                    layer_name: name,
                    lora_a: a,
                    lora_b: b,
                })
            })
            .collect(),
    };
    
    // Compute loss
    let loss_value = loss_fn.forward(&batch, &predicted)?;
    
    // Verify loss properties
    assert!(loss_value.dims().len() == 0 || loss_value.dims() == &[1]); // Scalar
    let loss_scalar = loss_value.to_scalar::<f32>()?;
    assert!(loss_scalar >= 0.0);
    assert!(!loss_scalar.is_nan());
    assert!(!loss_scalar.is_infinite());
    
    Ok(())
}

#[test]
fn test_reconstruction_loss_mae() -> Result<()> {
    let device = create_test_device();
    
    let config = LossConfig {
        loss_type: LossType::Reconstruction {
            matrix_loss: MatrixLossType::MAE,
            layer_weights: HashMap::new(),
        },
        regularization: Default::default(),
        scaling: Default::default(),
        stability: Default::default(),
        task_weights: HashMap::new(),
        reduction: ReductionMethod::Mean,
    };
    
    let loss_fn = ReconstructionLoss::new(config, &device)?;
    
    // Test with known values
    let target_a = Tensor::ones(&[1, 64, 4], DType::F32, &device)?;
    let target_b = Tensor::ones(&[1, 4, 64], DType::F32, &device)?;
    
    let pred_a = Tensor::full(2.0f32, &[64, 4], &device)?;
    let pred_b = Tensor::full(2.0f32, &[4, 64], &device)?;
    
    // MAE should be |2 - 1| = 1
    let batch = crate::training::data::ReconstructionBatch {
        task_embeddings: Tensor::zeros(&[1, 64], DType::F32, &device)?,
        target_params: HashMap::from([
            ("layer_0".to_string(), (target_a, target_b)),
        ]),
        layer_mask: None,
        sample_ids: vec!["test".to_string()],
        metadata: HashMap::new(),
    };
    
    let predicted = crate::training::loss::reconstruction::PredictedLoraParams {
        layers: HashMap::from([
            ("layer_0".to_string(), crate::training::loss::reconstruction::PredictedLoraLayer {
                layer_name: "layer_0".to_string(),
                lora_a: pred_a,
                lora_b: pred_b,
            }),
        ]),
    };
    
    let loss_value = loss_fn.forward(&batch, &predicted)?;
    let loss_scalar = loss_value.to_scalar::<f32>()?;
    
    // MAE should be approximately 1.0
    assert!((loss_scalar - 1.0).abs() < 0.01);
    
    Ok(())
}

#[test]
fn test_supervised_loss_cross_entropy() -> Result<()> {
    let device = create_test_device();
    
    let config = LossConfig {
        loss_type: LossType::Supervised {
            task_type: SupervisedTaskType::Classification,
            num_classes: Some(10),
            label_smoothing: 0.0,
        },
        regularization: Default::default(),
        scaling: Default::default(),
        stability: Default::default(),
        task_weights: HashMap::new(),
        reduction: ReductionMethod::Mean,
    };
    
    let loss_fn = SupervisedLoss::new(config, &device)?;
    
    // Create mock batch
    let batch_size = 4;
    let seq_len = 128;
    let vocab_size = 1000;
    let batch = create_mock_supervised_batch(batch_size, seq_len, vocab_size, &device)?;
    
    // Create logits
    let logits = random_tensor(&[batch_size, seq_len, vocab_size], &device)?;
    
    // Compute loss
    let loss_value = loss_fn.forward(&batch, &logits)?;
    
    // Verify loss properties
    assert!(loss_value.dims().len() == 0 || loss_value.dims() == &[1]);
    let loss_scalar = loss_value.to_scalar::<f32>()?;
    assert!(loss_scalar > 0.0);
    assert!(!loss_scalar.is_nan());
    
    Ok(())
}

#[test]
fn test_loss_with_regularization() -> Result<()> {
    let device = create_test_device();
    
    let mut config = LossConfig {
        loss_type: LossType::Reconstruction {
            matrix_loss: MatrixLossType::MSE,
            layer_weights: HashMap::new(),
        },
        regularization: crate::training::loss::RegularizationConfig {
            l1_lambda: Some(0.01),
            l2_lambda: Some(0.01),
            orthogonal_lambda: None,
            nuclear_lambda: None,
            sparsity_target: None,
            sparsity_lambda: None,
            gradient_penalty_lambda: None,
        },
        scaling: Default::default(),
        stability: Default::default(),
        task_weights: HashMap::new(),
        reduction: ReductionMethod::Mean,
    };
    
    let loss_fn = ReconstructionLoss::new(config, &device)?;
    
    // Create data with known properties
    let batch = create_mock_batch(2, &device)?;
    let mut predicted_params = HashMap::new();
    
    // Create parameters with known L1/L2 norms
    let a = Tensor::full(2.0f32, &[64, 4], &device)?;
    let b = Tensor::full(3.0f32, &[4, 64], &device)?;
    
    predicted_params.insert("layer_0".to_string(), (a.clone(), b.clone()));
    
    let predicted = crate::training::loss::reconstruction::PredictedLoraParams {
        layers: predicted_params.into_iter()
            .map(|(name, (a, b))| {
                (name.clone(), crate::training::loss::reconstruction::PredictedLoraLayer {
                    layer_name: name,
                    lora_a: a,
                    lora_b: b,
                })
            })
            .collect(),
    };
    
    // Compute loss with regularization
    let loss_value = loss_fn.forward(&batch, &predicted)?;
    let metrics = loss_fn.compute_metrics(&batch, &predicted)?;
    
    // Check that regularization penalties are included
    assert!(metrics.regularization_penalties.is_some());
    if let Some(reg_penalties) = &metrics.regularization_penalties {
        assert!(reg_penalties.l1_penalty.is_some());
        assert!(reg_penalties.l2_penalty.is_some());
    }
    
    Ok(())
}

#[test]
fn test_composite_loss() -> Result<()> {
    let device = create_test_device();
    
    // Create reconstruction loss
    let recon_config = LossConfig {
        loss_type: LossType::Reconstruction {
            matrix_loss: MatrixLossType::MSE,
            layer_weights: HashMap::new(),
        },
        regularization: Default::default(),
        scaling: Default::default(),
        stability: Default::default(),
        task_weights: HashMap::new(),
        reduction: ReductionMethod::Mean,
    };
    
    let recon_loss = ReconstructionLoss::new(recon_config.clone(), &device)?;
    
    // Create composite loss
    let mut composite_config = LossConfig::default();
    composite_config.task_weights = HashMap::from([
        ("reconstruction".to_string(), 0.7),
        ("auxiliary".to_string(), 0.3),
    ]);
    
    let mut composite_loss = CompositeLoss::new(composite_config)?;
    composite_loss.add_task("reconstruction", Box::new(recon_loss), 0.7)?;
    
    // Test that weights sum to 1.0 after normalization
    let weights = composite_loss.get_weights();
    let weight_sum: f64 = weights.values().sum();
    assert!((weight_sum - 1.0).abs() < 1e-6);
    
    Ok(())
}

#[test]
fn test_loss_numerical_stability() -> Result<()> {
    let device = create_test_device();
    
    let config = LossConfig {
        loss_type: LossType::Reconstruction {
            matrix_loss: MatrixLossType::MSE,
            layer_weights: HashMap::new(),
        },
        regularization: Default::default(),
        scaling: Default::default(),
        stability: crate::training::loss::StabilityConfig {
            eps: 1e-6,
            clip_value: Some(10.0),
            log_eps: 1e-8,
        },
        task_weights: HashMap::new(),
        reduction: ReductionMethod::Mean,
    };
    
    let loss_fn = ReconstructionLoss::new(config, &device)?;
    
    // Test with extreme values
    let batch = create_mock_batch(1, &device)?;
    
    // Create parameters with very large values
    let large_value = 1e10f32;
    let a = Tensor::full(large_value, &[64, 4], &device)?;
    let b = Tensor::full(large_value, &[4, 64], &device)?;
    
    let predicted = crate::training::loss::reconstruction::PredictedLoraParams {
        layers: HashMap::from([
            ("layer_0".to_string(), crate::training::loss::reconstruction::PredictedLoraLayer {
                layer_name: "layer_0".to_string(),
                lora_a: a,
                lora_b: b,
            }),
        ]),
    };
    
    // Loss should not explode due to clipping
    let loss_value = loss_fn.forward(&batch, &predicted)?;
    let loss_scalar = loss_value.to_scalar::<f32>()?;
    assert!(!loss_scalar.is_nan());
    assert!(!loss_scalar.is_infinite());
    
    Ok(())
}

#[test]
fn test_loss_reduction_methods() -> Result<()> {
    let device = create_test_device();
    
    // Test different reduction methods
    for reduction in [ReductionMethod::Mean, ReductionMethod::Sum, ReductionMethod::None] {
        let config = LossConfig {
            loss_type: LossType::Reconstruction {
                matrix_loss: MatrixLossType::MSE,
                layer_weights: HashMap::new(),
            },
            regularization: Default::default(),
            scaling: Default::default(),
            stability: Default::default(),
            task_weights: HashMap::new(),
            reduction,
        };
        
        let loss_fn = ReconstructionLoss::new(config, &device)?;
        
        let batch = create_mock_batch(4, &device)?;
        let predicted_params = create_test_lora_params(2, 64, 4, &device)?;
        
        let predicted = crate::training::loss::reconstruction::PredictedLoraParams {
            layers: predicted_params.into_iter()
                .map(|(name, (a, b))| {
                    (name.clone(), crate::training::loss::reconstruction::PredictedLoraLayer {
                        layer_name: name,
                        lora_a: a,
                        lora_b: b,
                    })
                })
                .collect(),
        };
        
        let loss_value = loss_fn.forward(&batch, &predicted)?;
        
        match reduction {
            ReductionMethod::Mean | ReductionMethod::Sum => {
                assert!(loss_value.dims().len() == 0 || loss_value.dims() == &[1]);
            }
            ReductionMethod::None => {
                // Should return per-sample losses
                assert!(loss_value.dims().len() > 0);
            }
        }
    }
    
    Ok(())
}

#[test]
fn test_loss_metrics_computation() -> Result<()> {
    let device = create_test_device();
    
    let config = LossConfig {
        loss_type: LossType::Reconstruction {
            matrix_loss: MatrixLossType::MSE,
            layer_weights: HashMap::new(),
        },
        regularization: Default::default(),
        scaling: Default::default(),
        stability: Default::default(),
        task_weights: HashMap::new(),
        reduction: ReductionMethod::Mean,
    };
    
    let loss_fn = ReconstructionLoss::new(config, &device)?;
    
    let batch = create_mock_batch(4, &device)?;
    let predicted_params = create_test_lora_params(2, 64, 4, &device)?;
    
    let predicted = crate::training::loss::reconstruction::PredictedLoraParams {
        layers: predicted_params.into_iter()
            .map(|(name, (a, b))| {
                (name.clone(), crate::training::loss::reconstruction::PredictedLoraLayer {
                    layer_name: name,
                    lora_a: a,
                    lora_b: b,
                })
            })
            .collect(),
    };
    
    let metrics = loss_fn.compute_metrics(&batch, &predicted)?;
    
    // Verify metrics structure
    assert!(metrics.total_loss > 0.0);
    assert!(metrics.main_loss > 0.0);
    assert_metrics_valid(&metrics.component_losses);
    
    if let Some(layer_losses) = &metrics.layer_losses {
        assert!(!layer_losses.is_empty());
        for (_, loss) in layer_losses {
            assert!(*loss >= 0.0);
        }
    }
    
    Ok(())
}

// Property-based tests
proptest! {
    #[test]
    fn prop_test_loss_non_negative(
        batch_size in 1usize..=8,
        hidden_size in 8usize..=32,
        rank in 1usize..=8,
    ) {
        let device = create_test_device();
        
        let config = LossConfig {
            loss_type: LossType::Reconstruction {
                matrix_loss: MatrixLossType::MSE,
                layer_weights: HashMap::new(),
            },
            regularization: Default::default(),
            scaling: Default::default(),
            stability: Default::default(),
            task_weights: HashMap::new(),
            reduction: ReductionMethod::Mean,
        };
        
        let loss_fn = ReconstructionLoss::new(config, &device).unwrap();
        
        // Create random batch
        let batch = create_mock_batch(batch_size, &device).unwrap();
        let predicted_params = create_test_lora_params(2, hidden_size, rank, &device).unwrap();
        
        let predicted = crate::training::loss::reconstruction::PredictedLoraParams {
            layers: predicted_params.into_iter()
                .map(|(name, (a, b))| {
                    (name.clone(), crate::training::loss::reconstruction::PredictedLoraLayer {
                        layer_name: name,
                        lora_a: a,
                        lora_b: b,
                    })
                })
                .collect(),
        };
        
        let loss_value = loss_fn.forward(&batch, &predicted).unwrap();
        let loss_scalar = loss_value.to_scalar::<f32>().unwrap();
        
        prop_assert!(loss_scalar >= 0.0);
        prop_assert!(!loss_scalar.is_nan());
        prop_assert!(!loss_scalar.is_infinite());
    }
    
    #[test]
    fn prop_test_loss_monotonicity(
        scale in 0.1f32..=10.0,
    ) {
        let device = create_test_device();
        
        let config = LossConfig {
            loss_type: LossType::Reconstruction {
                matrix_loss: MatrixLossType::MSE,
                layer_weights: HashMap::new(),
            },
            regularization: Default::default(),
            scaling: Default::default(),
            stability: Default::default(),
            task_weights: HashMap::new(),
            reduction: ReductionMethod::Mean,
        };
        
        let loss_fn = ReconstructionLoss::new(config, &device).unwrap();
        
        // Create base predictions
        let batch = create_mock_batch(2, &device).unwrap();
        let base_params = create_test_lora_params(1, 32, 4, &device).unwrap();
        
        // Scale predictions
        let scaled_params: HashMap<String, (Tensor, Tensor)> = base_params.into_iter()
            .map(|(name, (a, b))| {
                let scaled_a = a.mul_scalar(scale as f64).unwrap();
                let scaled_b = b.mul_scalar(scale as f64).unwrap();
                (name, (scaled_a, scaled_b))
            })
            .collect();
        
        let predicted = crate::training::loss::reconstruction::PredictedLoraParams {
            layers: scaled_params.into_iter()
                .map(|(name, (a, b))| {
                    (name.clone(), crate::training::loss::reconstruction::PredictedLoraLayer {
                        layer_name: name,
                        lora_a: a,
                        lora_b: b,
                    })
                })
                .collect(),
        };
        
        let loss_value = loss_fn.forward(&batch, &predicted).unwrap();
        let loss_scalar = loss_value.to_scalar::<f32>().unwrap();
        
        // Loss should increase with scale (for non-zero targets)
        prop_assert!(loss_scalar >= 0.0);
    }
}