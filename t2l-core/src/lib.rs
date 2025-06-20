//! T2L Core Library
//!
//! This library provides the core functionality for T2L (Text-to-LoRA), including:
//! - LoRA adapter application to base models
//! - Format conversion (PEFT, GGML, HuggingFace)
//! - Direct inference with adapted models
//! - CLI interface and configuration management

pub mod cli;
pub mod apply;
pub mod export;
pub mod infer;
pub mod utils;

// Re-export from the main lorax crate
pub use lorax::lora;
pub use lorax::TextToLora;

#[cfg(test)]
mod tests;

/// Current version of T2L Core
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// T2L Core result type
pub type Result<T> = anyhow::Result<T>;