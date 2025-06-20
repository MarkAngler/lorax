use crate::cli::{config::Config, error::CliResult, progress::ProgressReporter};
use crate::infer::{GenerationRequest, InferenceEngine, OutputFormat as InferOutputFormat};
use crate::lora::LoraParameters;
use anyhow::{Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf, time::Instant};
use tracing::{info, warn};

#[derive(Args, Debug)]
pub struct InferCommand {
    /// Path to base model (local path or HuggingFace model ID)
    #[arg(short, long, help = "Base model path or HuggingFace ID")]
    pub base_model: String,

    /// Path to T2L adapter file
    #[arg(short, long, help = "Path to T2L adapter file")]
    pub adapter: PathBuf,

    /// Input prompt for generation
    #[arg(short, long, help = "Input prompt for generation")]
    pub prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value = "100", help = "Maximum tokens to generate")]
    pub max_tokens: usize,

    /// Sampling temperature (0.0 = deterministic, higher = more random)
    #[arg(long, default_value = "0.7", help = "Sampling temperature")]
    pub temperature: f32,

    /// Top-p nucleus sampling (0.0-1.0, lower = more focused)
    #[arg(long, default_value = "0.9", help = "Top-p nucleus sampling")]
    pub top_p: f32,

    /// Target device for inference
    #[arg(long, default_value = "auto", help = "Target device (auto, cuda, cpu)")]
    pub device: DeviceType,

    /// Enable streaming output (print tokens as generated)
    #[arg(long, help = "Enable streaming output")]
    pub stream: bool,

    /// Output format for results
    #[arg(long, default_value = "text", help = "Output format")]
    pub output_format: InferOutputFormat,

    /// System prompt (optional)
    #[arg(long, help = "System prompt to prepend")]
    pub system: Option<String>,

    /// Conversation history file (JSON format)
    #[arg(long, help = "JSON file with conversation history")]
    pub history: Option<PathBuf>,

    /// Save output to file
    #[arg(long, help = "Path to save generation results")]
    pub save_output: Option<PathBuf>,

    /// Batch inference from multiple prompts
    #[arg(long, conflicts_with = "prompt", help = "JSON file with multiple prompts")]
    pub batch_file: Option<PathBuf>,

    /// Repetition penalty (1.0 = no penalty, >1.0 = penalize repetition)
    #[arg(long, default_value = "1.0", help = "Repetition penalty")]
    pub repetition_penalty: f32,

    /// Random seed for reproducible generation
    #[arg(long, help = "Random seed for reproducible generation")]
    pub seed: Option<u64>,

    /// Stop sequences (generation stops when any sequence is encountered)
    #[arg(long, help = "Stop sequences for generation", value_delimiter = ',')]
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum DeviceType {
    #[value(name = "auto")]
    Auto,
    #[value(name = "cuda")]
    Cuda,
    #[value(name = "cpu")]
    Cpu,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConversationHistory {
    pub messages: Vec<ConversationMessage>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String, // "user", "assistant", "system"
    pub content: String,
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchPrompt {
    pub id: String,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub system: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceResult {
    pub id: Option<String>,
    pub prompt: String,
    pub generated_text: String,
    pub total_tokens: usize,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    pub generation_time_ms: u64,
    pub tokens_per_second: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub async fn execute(cmd: InferCommand, config: Config) -> CliResult<()> {
    info!("Starting T2L inference");

    // Validate command inputs
    validate_command(&cmd)?;

    // Handle batch inference if requested
    if let Some(batch_file) = &cmd.batch_file {
        return execute_batch_inference(cmd, config, batch_file).await;
    }

    // Single prompt inference
    execute_single_inference(cmd, config).await
}

async fn execute_single_inference(cmd: InferCommand, config: Config) -> CliResult<()> {
    let start_time = Instant::now();
    
    // Initialize progress reporter
    let progress = ProgressReporter::new("Running inference")?;
    
    progress.set_message("Loading adapter...");
    
    // Load LoRA adapter
    let adapter = load_adapter(&cmd.adapter)?;
    
    progress.advance("Initializing inference engine...");
    
    // Determine device
    let device = determine_device(&cmd.device)?;
    
    // Initialize inference engine
    let mut engine = InferenceEngine::new(&cmd.base_model, &adapter, &device)
        .await
        .context("Failed to initialize inference engine")?;
    
    progress.advance("Preparing generation request...");
    
    // Build generation request
    let request = build_generation_request(&cmd).await?;
    
    progress.advance("Generating response...");
    
    // Run inference
    let response = engine.generate(&request)
        .await
        .context("Failed to generate response")?;
    
    let generation_time = start_time.elapsed();
    let tokens_per_second = response.usage.completion_tokens as f64 / generation_time.as_secs_f64();
    
    progress.finish("Inference completed!");
    
    // Create inference result
    let result = InferenceResult {
        id: None,
        prompt: cmd.prompt.clone(),
        generated_text: response.text.clone(),
        total_tokens: response.usage.total_tokens,
        prompt_tokens: response.usage.prompt_tokens,
        completion_tokens: response.usage.completion_tokens,
        finish_reason: format!("{:?}", response.finish_reason),
        generation_time_ms: generation_time.as_millis() as u64,
        tokens_per_second,
        timestamp: chrono::Utc::now(),
    };
    
    // Output results
    output_result(&result, &cmd)?;
    
    // Save output if requested
    if let Some(output_path) = &cmd.save_output {
        save_result(&result, output_path)?;
    }
    
    info!("Generated {} tokens in {:.2}s ({:.1} tokens/sec)", 
          response.usage.completion_tokens, 
          generation_time.as_secs_f64(), 
          tokens_per_second);
    
    Ok(())
}

async fn execute_batch_inference(
    cmd: InferCommand,
    config: Config,
    batch_file: &PathBuf,
) -> CliResult<()> {
    info!("Starting batch inference");
    
    // Load batch prompts
    let batch_prompts = load_batch_prompts(batch_file)?;
    info!("Loaded {} prompts for batch inference", batch_prompts.len());
    
    // Initialize inference engine once
    let progress = ProgressReporter::new("Initializing for batch inference")?;
    
    progress.set_message("Loading adapter...");
    let adapter = load_adapter(&cmd.adapter)?;
    
    progress.advance("Initializing inference engine...");
    let device = determine_device(&cmd.device)?;
    let mut engine = InferenceEngine::new(&cmd.base_model, &adapter, &device)
        .await
        .context("Failed to initialize inference engine")?;
    
    progress.finish("Engine initialized");
    
    // Process each prompt
    let batch_progress = ProgressReporter::new_with_total("Batch inference", batch_prompts.len() as u64)?;
    let mut results = Vec::new();
    
    for (i, batch_prompt) in batch_prompts.iter().enumerate() {
        let start_time = Instant::now();
        
        // Build request for this prompt
        let mut request = build_generation_request(&cmd).await?;
        request.prompt = batch_prompt.prompt.clone();
        
        // Override parameters if specified in batch
        if let Some(max_tokens) = batch_prompt.max_tokens {
            request.max_tokens = max_tokens;
        }
        if let Some(temperature) = batch_prompt.temperature {
            request.temperature = temperature;
        }
        if let Some(top_p) = batch_prompt.top_p {
            request.top_p = top_p;
        }
        if let Some(system) = &batch_prompt.system {
            request.system = Some(system.clone());
        }
        
        // Generate response
        let response = engine.generate(&request)
            .await
            .context(format!("Failed to generate response for prompt {}", i + 1))?;
        
        let generation_time = start_time.elapsed();
        let tokens_per_second = response.usage.completion_tokens as f64 / generation_time.as_secs_f64();
        
        // Create result
        let result = InferenceResult {
            id: Some(batch_prompt.id.clone()),
            prompt: batch_prompt.prompt.clone(),
            generated_text: response.text,
            total_tokens: response.usage.total_tokens,
            prompt_tokens: response.usage.prompt_tokens,
            completion_tokens: response.usage.completion_tokens,
            finish_reason: format!("{:?}", response.finish_reason),
            generation_time_ms: generation_time.as_millis() as u64,
            tokens_per_second,
            timestamp: chrono::Utc::now(),
        };
        
        results.push(result);
        batch_progress.advance(&format!("Completed prompt {}/{}", i + 1, batch_prompts.len()));
    }
    
    batch_progress.finish("Batch inference completed!");
    
    // Output all results
    for result in &results {
        output_result(result, &cmd)?;
        println!("---"); // Separator between results
    }
    
    // Save batch results if requested
    if let Some(output_path) = &cmd.save_output {
        save_batch_results(&results, output_path)?;
    }
    
    info!("Completed batch inference for {} prompts", results.len());
    
    Ok(())
}

fn validate_command(cmd: &InferCommand) -> CliResult<()> {
    // Check adapter file exists
    if !cmd.adapter.exists() {
        return Err(crate::cli::error::CliError::AdapterNotFound(cmd.adapter.clone()));
    }
    
    // Validate temperature range
    if cmd.temperature < 0.0 || cmd.temperature > 2.0 {
        return Err(crate::cli::error::CliError::InvalidTemperature(cmd.temperature));
    }
    
    // Validate top_p range
    if cmd.top_p <= 0.0 || cmd.top_p > 1.0 {
        return Err(crate::cli::error::CliError::InvalidTopP(cmd.top_p));
    }
    
    // Validate max_tokens
    if cmd.max_tokens == 0 || cmd.max_tokens > 4096 {
        return Err(crate::cli::error::CliError::InvalidMaxTokens(cmd.max_tokens));
    }
    
    // Validate repetition penalty
    if cmd.repetition_penalty < 0.0 || cmd.repetition_penalty > 2.0 {
        return Err(crate::cli::error::CliError::InvalidRepetitionPenalty(cmd.repetition_penalty));
    }
    
    // Check history file exists if specified
    if let Some(history_file) = &cmd.history {
        if !history_file.exists() {
            return Err(crate::cli::error::CliError::HistoryFileNotFound(history_file.clone()));
        }
    }
    
    // Check batch file exists if specified
    if let Some(batch_file) = &cmd.batch_file {
        if !batch_file.exists() {
            return Err(crate::cli::error::CliError::BatchFileNotFound(batch_file.clone()));
        }
    }
    
    Ok(())
}

fn load_adapter(adapter_path: &PathBuf) -> CliResult<LoraParameters> {
    // TODO: Implement adapter loading based on file extension
    // This would handle .safetensors, .json, .bin formats
    info!("Loading adapter from: {}", adapter_path.display());
    
    let _content = fs::read(adapter_path)
        .context("Failed to read adapter file")?;
    
    // Placeholder implementation - would need proper deserialization based on file format
    // For .json files: serde_json::from_slice(&content)?
    // For .safetensors files: custom safetensors deserializer
    // For .bin files: custom binary format deserializer
    warn!("Adapter loading not yet fully implemented - using placeholder");
    Ok(create_placeholder_lora_parameters())
}

fn determine_device(device_type: &DeviceType) -> CliResult<candle_core::Device> {
    use candle_core::Device;
    
    match device_type {
        DeviceType::Auto => {
            // Try CUDA first, fallback to CPU
            // TODO: Replace with actual CUDA availability check when candle integration is complete
            info!("Auto-selected CPU device (CUDA support pending)");
            Ok(Device::Cpu)
        }
        DeviceType::Cuda => {
            // TODO: Replace with actual CUDA availability check when candle integration is complete
            warn!("CUDA device requested but support not yet implemented, using CPU");
            Ok(Device::Cpu)
        }
        DeviceType::Cpu => {
            info!("Using CPU device");
            Ok(Device::Cpu)
        }
    }
}

async fn build_generation_request(cmd: &InferCommand) -> CliResult<GenerationRequest> {
    let mut prompt = cmd.prompt.clone();
    
    // Add system prompt if specified
    if let Some(system) = &cmd.system {
        prompt = format!("{}\n\n{}", system, prompt);
    }
    
    // Load conversation history if specified
    if let Some(history_file) = &cmd.history {
        let history = load_conversation_history(history_file)?;
        prompt = format_with_history(&history, &prompt)?;
    }
    
    Ok(GenerationRequest {
        prompt,
        max_tokens: cmd.max_tokens,
        temperature: cmd.temperature,
        top_p: cmd.top_p,
        stream: cmd.stream,
        system: cmd.system.clone(),
    })
}

fn load_conversation_history(history_file: &PathBuf) -> CliResult<ConversationHistory> {
    let content = fs::read_to_string(history_file)
        .context("Failed to read conversation history file")?;
    
    let history: ConversationHistory = serde_json::from_str(&content)
        .context("Failed to parse conversation history JSON")?;
    
    Ok(history)
}

fn format_with_history(history: &ConversationHistory, current_prompt: &str) -> CliResult<String> {
    let mut formatted = String::new();
    
    for message in &history.messages {
        match message.role.as_str() {
            "system" => formatted.push_str(&format!("System: {}\n\n", message.content)),
            "user" => formatted.push_str(&format!("User: {}\n\n", message.content)),
            "assistant" => formatted.push_str(&format!("Assistant: {}\n\n", message.content)),
            _ => warn!("Unknown conversation role: {}", message.role),
        }
    }
    
    formatted.push_str(&format!("User: {}\n\nAssistant:", current_prompt));
    
    Ok(formatted)
}

fn load_batch_prompts(batch_file: &PathBuf) -> CliResult<Vec<BatchPrompt>> {
    let content = fs::read_to_string(batch_file)
        .context("Failed to read batch file")?;
    
    // Try JSON format first
    if let Ok(prompts) = serde_json::from_str::<Vec<BatchPrompt>>(&content) {
        return Ok(prompts);
    }
    
    // Fallback to simple text format (one prompt per line)
    let prompts: Vec<BatchPrompt> = content
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(i, line)| BatchPrompt {
            id: format!("prompt_{}", i + 1),
            prompt: line.trim().to_string(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            system: None,
        })
        .collect();
    
    if prompts.is_empty() {
        return Err(crate::cli::error::CliError::EmptyBatchFile(batch_file.clone()));
    }
    
    Ok(prompts)
}

fn output_result(result: &InferenceResult, cmd: &InferCommand) -> CliResult<()> {
    match cmd.output_format {
        InferOutputFormat::Text => {
            if !cmd.stream {
                println!("{}", result.generated_text);
            }
        }
        InferOutputFormat::Json => {
            let json_output = serde_json::to_string_pretty(result)?;
            println!("{}", json_output);
        }
        InferOutputFormat::Markdown => {
            println!("## Generated Response\n");
            println!("**Prompt:** {}\n", result.prompt);
            println!("**Response:**\n{}\n", result.generated_text);
            println!("**Stats:** {} tokens in {:.2}s ({:.1} tokens/sec)",
                    result.completion_tokens,
                    result.generation_time_ms as f64 / 1000.0,
                    result.tokens_per_second);
        }
    }
    
    Ok(())
}

fn save_result(result: &InferenceResult, output_path: &PathBuf) -> CliResult<()> {
    // Create parent directory if needed
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    
    let json_output = serde_json::to_string_pretty(result)?;
    fs::write(output_path, json_output)?;
    
    info!("Saved inference result to: {}", output_path.display());
    Ok(())
}

fn save_batch_results(results: &[InferenceResult], output_path: &PathBuf) -> CliResult<()> {
    // Create parent directory if needed
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    
    let json_output = serde_json::to_string_pretty(results)?;
    fs::write(output_path, json_output)?;
    
    info!("Saved batch inference results to: {}", output_path.display());
    Ok(())
}

// Helper function to create placeholder LoRA parameters
fn create_placeholder_lora_parameters() -> LoraParameters {
    // TODO: Remove this when proper LoRA loading is implemented
    warn!("Using placeholder LoRA parameters");
    LoraParameters::new(crate::lora::LoraParameterConfig::default())
}