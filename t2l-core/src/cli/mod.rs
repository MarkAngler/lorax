pub mod commands;
pub mod config;
pub mod error;
pub mod logging;
pub mod progress;

// Re-export command structures
pub use commands::{
    apply::ApplyCommand,
    evaluate::EvaluateCommand,
    generate::GenerateCommand,
    serve::ServeCommand,
    train::TrainCommand,
};

// Re-export error types
pub use error::{CliError, CliResult};