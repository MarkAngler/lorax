use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Generator, Shell};
use std::io;
use t2l_core::cli::{self, commands::*};

#[derive(Parser)]
#[command(
    name = "t2l",
    version,
    about = "Text-to-LoRA: Generate task-specific adapters from natural language descriptions",
    long_about = "T2L (Text-to-LoRA) enables instant adaptation of large language models through \
                  natural language task descriptions. Generate LoRA adapters without traditional fine-tuning."
)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Set the verbosity level (can be repeated for more verbose output)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
    
    /// Silence all output except errors
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    quiet: bool,
    
    /// Use JSON output format
    #[arg(long, global = true)]
    json: bool,
    
    /// Configuration file path
    #[arg(short, long, global = true, env = "T2L_CONFIG")]
    config: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a T2L hypernetwork
    Train(TrainCommand),
    
    /// Generate a LoRA adapter from a task description
    Generate(GenerateCommand),
    
    /// Apply a LoRA adapter to a base model
    Apply(ApplyCommand),
    
    /// Export a LoRA adapter to different formats
    Export(ExportCommand),
    
    /// Evaluate model performance on benchmarks
    Evaluate(EvaluateCommand),
    
    /// Run inference with base model + adapter
    Infer(InferCommand),
    
    /// Start the T2L API server
    Serve(ServeCommand),
    
    /// Manage T2L configuration
    Config {
        #[command(subcommand)]
        subcommand: ConfigSubcommand,
    },
    
    /// Generate shell completions
    Completions {
        /// The shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[derive(Subcommand)]
enum ConfigSubcommand {
    /// Show current configuration
    Show,
    
    /// Initialize configuration file
    Init {
        /// Force overwrite existing configuration
        #[arg(short, long)]
        force: bool,
    },
    
    /// Get a configuration value
    Get {
        /// Configuration key to get
        key: String,
    },
    
    /// Set a configuration value
    Set {
        /// Configuration key to set
        key: String,
        /// Value to set
        value: String,
    },
}

fn print_completions<G: Generator>(gen: G, cmd: &mut clap::Command) {
    generate(gen, cmd, cmd.get_name().to_string(), &mut io::stdout());
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging based on verbosity
    cli::logging::init_logging(cli.verbose, cli.quiet, cli.json)?;
    
    // Load configuration
    let config = cli::config::load_config(cli.config.as_deref())?;
    
    match cli.command {
        Commands::Train(cmd) => cli::commands::train::execute(cmd, config).await,
        Commands::Generate(cmd) => cli::commands::generate::execute(cmd, config).await,
        Commands::Apply(cmd) => cli::commands::apply::execute(cmd, config).await,
        Commands::Export(cmd) => cli::commands::export::execute(cmd, config).await,
        Commands::Evaluate(cmd) => cli::commands::evaluate::execute(cmd, config).await,
        Commands::Infer(cmd) => cli::commands::infer::execute(cmd, config).await,
        Commands::Serve(cmd) => cli::commands::serve::execute(cmd, config).await,
        Commands::Config { subcommand } => handle_config(subcommand, config),
        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            print_completions(shell, &mut cmd);
            Ok(())
        }
    }
}

fn handle_config(subcommand: ConfigSubcommand, config: cli::config::Config) -> Result<()> {
    match subcommand {
        ConfigSubcommand::Show => cli::config::show_config(&config),
        ConfigSubcommand::Init { force } => cli::config::init_config(force),
        ConfigSubcommand::Get { key } => cli::config::get_config_value(&config, &key),
        ConfigSubcommand::Set { key, value } => cli::config::set_config_value(&key, &value),
    }
}