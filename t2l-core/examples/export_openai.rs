//! Example of exporting T2L adapter to OpenAI-compatible format

use anyhow::Result;
use t2l_core::lora::{LoraParameters, LoraParameterConfig, LoraLayer, ParameterMetadata};
use t2l_core::export::openai::export_to_openai;
use std::collections::HashMap;
use chrono::Utc;
use std::path::PathBuf;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "export-openai")]
#[command(about = "Export T2L adapter to OpenAI-compatible format")]
struct Args {
    /// Path to the T2L adapter file (JSON format)
    #[arg(short, long)]
    adapter: Option<PathBuf>,
    
    /// Output directory for OpenAI format files
    #[arg(short, long, default_value = "./openai_export")]
    output: PathBuf,
    
    /// Target architecture (llama, mistral, gemma, etc.)
    #[arg(short = 't', long, default_value = "llama")]
    architecture: String,
    
    /// LoRA rank
    #[arg(short = 'r', long, default_value = "16")]
    rank: usize,
    
    /// LoRA alpha
    #[arg(short = 'a', long, default_value = "32.0")]
    alpha: f32,
    
    /// Task description for the model
    #[arg(short = 'd', long, default_value = "General purpose AI assistant")]
    description: String,
}

/// Create example LoRA parameters
fn create_example_lora_params(args: &Args) -> Result<LoraParameters> {
    let config = LoraParameterConfig {
        target_architecture: args.architecture.clone(),
        default_rank: args.rank,
        default_alpha: args.alpha,
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
    
    // Add example layers for demonstration
    // In a real scenario, these would be loaded from the adapter file
    let layer_configs = vec![
        ("model.layers.0.self_attn.q_proj", 4096, 4096),
        ("model.layers.0.self_attn.k_proj", 4096, 4096),
        ("model.layers.0.self_attn.v_proj", 4096, 4096),
        ("model.layers.0.self_attn.o_proj", 4096, 4096),
        ("model.layers.0.mlp.gate_proj", 4096, 11008),
        ("model.layers.0.mlp.up_proj", 4096, 11008),
        ("model.layers.0.mlp.down_proj", 11008, 4096),
    ];
    
    for (name, input_dim, output_dim) in layer_configs {
        let layer = LoraLayer::with_random_weights(
            name.to_string(),
            input_dim,
            output_dim,
            args.rank,
            args.alpha,
        );
        params.add_layer(layer)?;
    }
    
    // Add metadata
    let mut hyperparameters = HashMap::new();
    hyperparameters.insert("learning_rate".to_string(), serde_json::json!(1e-4));
    hyperparameters.insert("batch_size".to_string(), serde_json::json!(4));
    hyperparameters.insert("num_epochs".to_string(), serde_json::json!(3));
    
    let metadata = ParameterMetadata {
        task_description: args.description.clone(),
        created_at: Utc::now(),
        generator_version: "t2l-v0.1.0".to_string(),
        hyperparameters,
        metrics: Some({
            let mut metrics = HashMap::new();
            metrics.insert("final_loss".to_string(), 0.023);
            metrics.insert("perplexity".to_string(), 1.45);
            metrics
        }),
    };
    params.set_metadata(metadata);
    
    Ok(params)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    
    let args = Args::parse();
    
    println!("🚀 T2L OpenAI Format Export Example");
    println!("====================================");
    
    // Load or create adapter
    let adapter = if let Some(adapter_path) = &args.adapter {
        println!("📄 Loading adapter from: {}", adapter_path.display());
        // In a real implementation, you would deserialize from the file
        // For this example, we'll create a demo adapter
        create_example_lora_params(&args)?
    } else {
        println!("🔨 Creating example adapter with:");
        println!("   Architecture: {}", args.architecture);
        println!("   Rank: {}", args.rank);
        println!("   Alpha: {}", args.alpha);
        create_example_lora_params(&args)?
    };
    
    // Display adapter info
    let summary = adapter.summary();
    println!("\n📊 Adapter Summary:");
    println!("   Total Parameters: {:,}", summary.total_parameters);
    println!("   Number of Layers: {}", summary.num_layers);
    println!("   Average Rank: {:.1}", summary.avg_rank);
    println!("   Average Alpha: {:.1}", summary.avg_alpha);
    
    // Export to OpenAI format
    println!("\n📦 Exporting to OpenAI format...");
    export_to_openai(&adapter, &args.output).await?;
    
    println!("\n✅ Export completed successfully!");
    println!("\n📁 Output files:");
    println!("   {}/", args.output.display());
    println!("   ├── model_metadata.json     - OpenAI model metadata");
    println!("   ├── deployment_config.json  - Deployment configuration");
    println!("   ├── openai_api_spec.json   - OpenAPI specification");
    println!("   ├── model_capabilities.json - Model capabilities");
    println!("   ├── finetune_config.json   - Fine-tuning configuration");
    println!("   ├── README.md              - Usage documentation");
    println!("   ├── weights/");
    println!("   │   ├── adapter_config.json - Adapter configuration");
    println!("   │   └── weight_index.json   - Weight file index");
    println!("   └── examples/");
    println!("       ├── basic_chat.py      - Basic usage example");
    println!("       └── streaming_chat.py  - Streaming example");
    
    // Display usage instructions
    println!("\n🎯 Next Steps:");
    println!("1. Deploy the model using your OpenAI-compatible API server");
    println!("2. Update the base_url in the example scripts");
    println!("3. Set your API key as an environment variable");
    println!("4. Run the examples:");
    println!("   python {}/examples/basic_chat.py", args.output.display());
    
    Ok(())
}