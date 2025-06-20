//! Example of exporting T2L adapters to GGML format for llama.cpp
//!
//! This example demonstrates how to:
//! 1. Load a T2L adapter
//! 2. Export it to GGML format with different precision options
//! 3. Use the exported file with llama.cpp

use t2l_core::export::{ggml, Precision};
use t2l_core::lora::{LoraLayer, LoraParameters, LoraParameterConfig};
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();
    
    // Create or load your LoRA adapter
    let adapter = create_example_adapter();
    
    // Export to GGML format with FP16 precision (recommended for llama.cpp)
    println!("Exporting adapter to GGML format...");
    
    let output_path = Path::new("./my_adapter.ggml");
    ggml::export_to_ggml(&adapter, output_path, Precision::Fp16).await?;
    
    println!("✅ Successfully exported to: {}", output_path.display());
    println!("\nYou can now use this adapter with llama.cpp:");
    println!("  ./main -m base_model.gguf --lora my_adapter.ggml");
    
    // Example: Export with different precisions
    println!("\nExporting with different precisions:");
    
    // FP32 - Highest precision, largest file
    ggml::export_to_ggml(
        &adapter,
        Path::new("./my_adapter_fp32.ggml"),
        Precision::Fp32
    ).await?;
    println!("  - FP32: my_adapter_fp32.ggml");
    
    // INT8 - Quantized, smallest file
    ggml::export_to_ggml(
        &adapter,
        Path::new("./my_adapter_int8.ggml"),
        Precision::Int8
    ).await?;
    println!("  - INT8: my_adapter_int8.ggml");
    
    Ok(())
}

/// Create an example LoRA adapter
/// In practice, you would load this from a .safetensors file
fn create_example_adapter() -> LoraParameters {
    let config = LoraParameterConfig {
        target_architecture: "llama".to_string(),
        default_rank: 32,
        default_alpha: 64.0,
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
            "gate_proj".to_string(),
            "up_proj".to_string(),
            "down_proj".to_string(),
        ],
        merge_weights: false,
    };
    
    let mut params = LoraParameters::new(config);
    
    // Add layers for a small model (e.g., 32 transformer blocks)
    for block_idx in 0..32 {
        // Attention layers
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let layer_name = format!("layers.{}.self_attn.{}", block_idx, proj);
            let layer = LoraLayer::with_random_weights(
                layer_name,
                4096, // hidden_size
                4096, // hidden_size
                32,   // rank
                64.0, // alpha
            );
            params.add_layer(layer).unwrap();
        }
        
        // MLP layers
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let (in_dim, out_dim) = match proj {
                "gate_proj" | "up_proj" => (4096, 11008), // hidden -> intermediate
                "down_proj" => (11008, 4096), // intermediate -> hidden
                _ => unreachable!(),
            };
            
            let layer_name = format!("layers.{}.mlp.{}", block_idx, proj);
            let layer = LoraLayer::with_random_weights(
                layer_name,
                in_dim,
                out_dim,
                32,   // rank
                64.0, // alpha
            );
            params.add_layer(layer).unwrap();
        }
    }
    
    println!("Created adapter with {} layers", params.layers.len());
    println!("Total parameters: {}", params.total_parameters());
    
    params
}