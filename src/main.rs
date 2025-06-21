use anyhow::{Result, Context};
use clap::{Parser, Subcommand};
use lorax::{TextToLora, Config};
use std::path::PathBuf;
use std::fs;
use tracing::{info, warn, error};

#[derive(Parser)]
#[command(name = "lorax")]
#[command(about = "LoRA eXtensions - Production-ready Text-to-LoRA implementation", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate LoRA parameters from text description
    Generate {
        /// Path to text file containing task description
        #[arg(short, long)]
        input: PathBuf,
        
        /// Path to save generated LoRA parameters
        #[arg(short, long)]
        output: PathBuf,
        
        /// Configuration file path
        #[arg(short, long, default_value = "config.json")]
        config: PathBuf,
    },
    
    /// Validate configuration file
    Config {
        /// Configuration file to validate
        #[arg(short, long)]
        file: PathBuf,
    },
    
    /// Show system information
    Info,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Generate { input, output, config } => {
            generate_lora(input, output, config).await?;
        }
        
        Commands::Config { file } => {
            validate_config(file)?;
        }
        
        Commands::Info => {
            show_system_info();
        }
    }
    
    Ok(())
}

async fn generate_lora(input: PathBuf, output: PathBuf, config_path: PathBuf) -> Result<()> {
    info!("Starting LoRA parameter generation");
    
    // Load and validate configuration
    let config = Config::from_file(&config_path)
        .context("Failed to load configuration file")?;
    
    info!("Configuration loaded successfully");
    
    // Load task description
    let task_description = fs::read_to_string(&input)
        .context("Failed to read task description file")?;
    
    info!("Task description: {}", task_description.trim());
    
    // Initialize T2L system
    info!("Initializing Text-to-LoRA system...");
    let t2l = TextToLora::new(config).await
        .context("Failed to initialize Text-to-LoRA system")?;
    
    info!("System initialized successfully");
    
    // Generate LoRA parameters
    info!("Generating LoRA parameters...");
    let lora_params = t2l.generate(&task_description).await
        .context("Failed to generate LoRA parameters")?;
    
    // Ensure output directory exists
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .context("Failed to create output directory")?;
    }
    
    // Save LoRA parameters (simplified serialization)
    let serialized = serde_json::to_string_pretty(&lora_params)
        .context("Failed to serialize LoRA parameters")?;
    
    fs::write(&output, serialized)
        .context("Failed to write LoRA parameters")?;
    
    info!("LoRA parameters successfully saved to: {}", output.display());
    info!("Generation complete!");
    
    Ok(())
}

fn validate_config(config_path: PathBuf) -> Result<()> {
    info!("Validating configuration file: {}", config_path.display());
    
    let config = Config::from_file(&config_path)
        .context("Failed to load configuration file")?;
    
    // The Config::from_file already calls validate() internally
    info!("✅ Configuration is valid!");
    info!("Configuration summary:");
    info!("  - Encoder: {} ({}D embeddings)", config.encoder.model_name, config.encoder.embedding_dim);
    info!("  - Hypernetwork: {:?} model", config.hypernetwork.model_size);
    info!("  - LoRA rank: {}", config.lora.rank);
    info!("  - Target modules: {:?}", config.lora.target_modules);
    
    Ok(())
}

fn show_system_info() {
    println!("🦀 LoRAX - Production Text-to-LoRA Implementation");
    println!();
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!("Built with Rust: {}", option_env!("CARGO_PKG_RUST_VERSION").unwrap_or("unknown"));
    println!();
    println!("Features:");
    println!("  ✅ Text-to-LoRA generation");
    println!("  ✅ Hypernetwork-based parameter synthesis");
    println!("  ✅ Multiple model architectures (BERT, GPT, LLaMA, T5, ViT)");
    println!("  ✅ Configurable LoRA parameters");
    println!("  ✅ Production-ready Rust implementation");
    println!();
    println!("Hardware support:");
    
    #[cfg(feature = "cuda")]
    println!("  ✅ NVIDIA CUDA GPU acceleration");
    #[cfg(not(feature = "cuda"))]
    println!("  ❌ CUDA support (not compiled)");
    
    #[cfg(feature = "metal")]
    println!("  ✅ Apple Metal GPU acceleration");
    #[cfg(not(feature = "metal"))]
    println!("  ❌ Metal support (not compiled)");
    
    #[cfg(feature = "accelerate")]
    println!("  ✅ Apple Accelerate framework");
    #[cfg(not(feature = "accelerate"))]
    println!("  ❌ Accelerate support (not compiled)");
    
    println!("  ✅ CPU inference");
    println!();
    println!("Usage:");
    println!("  lorax generate -i task.txt -o output.json -c config.json");
    println!("  lorax config -f config.json  # Validate configuration");
    println!("  lorax info                   # Show this information");
    println!();
    println!("For more information, see: https://github.com/lorax-project/lorax");
}