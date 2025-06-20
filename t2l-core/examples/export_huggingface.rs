//! Example showing how to export T2L LoRA adapters to HuggingFace format
//!
//! This example demonstrates:
//! - Loading a T2L adapter
//! - Exporting it to HuggingFace format
//! - Different precision options
//!
//! Usage:
//! ```bash
//! cargo run --example export_huggingface -- --adapter path/to/adapter.safetensors --output ./hf_model/
//! ```

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use t2l_core::{
    export::{ExportEngine, ExportFormat, Precision},
    lora::LoraParameters,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to T2L adapter file
    #[arg(short, long)]
    adapter: PathBuf,

    /// Output directory for HuggingFace format
    #[arg(short, long)]
    output: PathBuf,

    /// Target model name (e.g., "meta-llama/Llama-2-7b-hf")
    #[arg(short, long)]
    target_model: Option<String>,

    /// Weight precision
    #[arg(long, default_value = "fp16")]
    precision: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    println!("🚀 T2L HuggingFace Export Example");
    println!("================================");
    println!();

    // Load adapter
    println!("📂 Loading adapter from: {}", args.adapter.display());
    let adapter = load_adapter(&args.adapter).await?;
    
    // Display adapter info
    let summary = adapter.summary();
    println!("📊 Adapter Summary:");
    println!("   - Architecture: {}", summary.target_architecture);
    println!("   - Total parameters: {:,}", summary.total_parameters);
    println!("   - Number of layers: {}", summary.num_layers);
    println!("   - Average rank: {:.1}", summary.avg_rank);
    println!("   - Average alpha: {:.1}", summary.avg_alpha);
    println!();

    // Parse precision
    let precision = match args.precision.as_str() {
        "fp16" => Precision::Fp16,
        "fp32" => Precision::Fp32,
        "int8" => Precision::Int8,
        _ => {
            eprintln!("❌ Invalid precision: {}. Using fp16.", args.precision);
            Precision::Fp16
        }
    };

    // Create export engine
    let export_engine = ExportEngine::new(precision, false);

    // Export to HuggingFace format
    println!("🔄 Exporting to HuggingFace format...");
    export_engine
        .export(
            &adapter,
            ExportFormat::Hf,
            args.target_model.as_deref(),
            &args.output,
        )
        .await?;

    println!();
    println!("✅ Export complete!");
    println!();
    println!("📁 Output files:");
    
    // List expected output files
    let output_files = vec![
        "config.json",
        "adapter_model.safetensors",
        "adapter_config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "README.md",
    ];

    for file in output_files {
        let file_path = args.output.join(file);
        if file_path.exists() {
            println!("   ✓ {}", file);
        }
    }

    println!();
    println!("🎯 To use with HuggingFace Transformers:");
    println!();
    println!("   from transformers import AutoModelForCausalLM, AutoTokenizer");
    println!("   ");
    println!("   model = AutoModelForCausalLM.from_pretrained(");
    println!("       \"{}\",", args.output.display());
    println!("       torch_dtype=torch.float16,");
    println!("       device_map=\"auto\"");
    println!("   )");
    println!("   tokenizer = AutoTokenizer.from_pretrained(\"{}\");", args.output.display());
    println!();

    Ok(())
}

/// Load adapter from file (mock implementation for example)
async fn load_adapter(path: &PathBuf) -> Result<LoraParameters> {
    // In a real implementation, this would load from the file
    // For this example, we'll create a mock adapter
    use t2l_core::lora::{LoraLayer, LoraParameterConfig, LoraParameters, ParameterMetadata};
    use std::collections::HashMap;

    let config = LoraParameterConfig {
        target_architecture: "llama".to_string(),
        default_rank: 16,
        default_alpha: 32.0,
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
        ],
        merge_weights: false,
    };

    let mut adapter = LoraParameters::new(config);

    // Add some mock layers
    for i in 0..32 {
        for module in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let layer_name = format!("layers.{}.self_attn.{}", i, module);
            let layer = LoraLayer::with_random_weights(
                layer_name.clone(),
                4096,  // input_dim
                4096,  // output_dim
                16,    // rank
                32.0,  // alpha
            );
            adapter.add_layer(layer)?;
        }
    }

    // Add metadata
    let metadata = ParameterMetadata {
        task_description: "Example LoRA adapter for HuggingFace export".to_string(),
        created_at: chrono::Utc::now(),
        generator_version: "t2l-v1.0.0".to_string(),
        hyperparameters: HashMap::new(),
        metrics: None,
    };
    adapter.set_metadata(metadata);

    Ok(adapter)
}