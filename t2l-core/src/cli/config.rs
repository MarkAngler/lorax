use crate::cli::error::{CliError, CliResult};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::PathBuf};
use directories::ProjectDirs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub t2l: T2LConfig,
    pub server: ServerConfig,
    pub evaluation: EvaluationConfig,
    pub training: TrainingConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T2LConfig {
    pub encoder: EncoderConfig,
    pub hypernetwork: HypernetworkConfig,
    pub lora: LoraConfig,
    pub projection: ProjectionConfig,
    pub device: DeviceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    pub model_name: String,
    pub embedding_dim: usize,
    pub max_length: usize,
    pub cache_dir: Option<PathBuf>,
    pub trust_remote_code: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypernetworkConfig {
    pub model_size: ModelSize,
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub output_dim: usize,
    pub activation: ActivationType,
    pub dropout: f32,
    pub batch_norm: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub bias: BiasType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionConfig {
    pub activation: ActivationType,
    pub dropout: f32,
    pub normalize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    pub device_type: DeviceType,
    pub gpu_ids: Option<Vec<usize>>,
    pub mixed_precision: bool,
    pub memory_fraction: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
    pub max_concurrent: usize,
    pub timeout_seconds: u64,
    pub cors_enabled: bool,
    pub metrics_enabled: bool,
    pub auth: AuthConfig,
    pub rate_limiting: RateLimitConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    pub enabled: bool,
    pub api_key: Option<String>,
    pub jwt_secret: Option<String>,
    pub token_expiry_hours: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub enabled: bool,
    pub requests_per_minute: u32,
    pub burst_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub default_benchmarks: Vec<String>,
    pub batch_size: usize,
    pub max_samples: Option<usize>,
    pub few_shot_examples: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub max_generation_length: usize,
    pub seed: u64,
    pub save_predictions: bool,
    pub cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub default_epochs: usize,
    pub default_batch_size: usize,
    pub default_learning_rate: f64,
    pub default_weight_decay: f64,
    pub gradient_clip_value: f64,
    pub warmup_steps: usize,
    pub save_every_epochs: usize,
    pub eval_every_epochs: usize,
    pub early_stopping_patience: Option<usize>,
    pub mixed_precision: bool,
    pub data_workers: usize,
    pub checkpoint_dir: PathBuf,
    pub wandb: WandbConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WandbConfig {
    pub enabled: bool,
    pub project: String,
    pub entity: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file: Option<PathBuf>,
    pub json_format: bool,
    pub include_module_path: bool,
    pub include_file_line: bool,
    pub console_colors: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ModelSize {
    Small,
    Medium,
    Large,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    GELU,
    SiLU,
    Tanh,
    Sigmoid,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BiasType {
    None,
    All,
    LoraOnly,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DeviceType {
    Auto,
    CPU,
    CUDA,
    Metal,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            t2l: T2LConfig::default(),
            server: ServerConfig::default(),
            evaluation: EvaluationConfig::default(),
            training: TrainingConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for T2LConfig {
    fn default() -> Self {
        Self {
            encoder: EncoderConfig::default(),
            hypernetwork: HypernetworkConfig::default(),
            lora: LoraConfig::default(),
            projection: ProjectionConfig::default(),
            device: DeviceConfig::default(),
        }
    }
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            embedding_dim: 384,
            max_length: 512,
            cache_dir: None,
            trust_remote_code: false,
        }
    }
}

impl Default for HypernetworkConfig {
    fn default() -> Self {
        Self {
            model_size: ModelSize::Medium,
            input_dim: 384,
            hidden_dims: vec![512, 1024, 512],
            output_dim: 8192, // For rank-16 LoRA with typical layer sizes
            activation: ActivationType::GELU,
            dropout: 0.1,
            batch_norm: true,
        }
    }
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            bias: BiasType::None,
        }
    }
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            activation: ActivationType::GELU,
            dropout: 0.1,
            normalize: true,
        }
    }
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Auto,
            gpu_ids: None,
            mixed_precision: true,
            memory_fraction: None,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            workers: num_cpus::get(),
            max_concurrent: 100,
            timeout_seconds: 30,
            cors_enabled: true,
            metrics_enabled: true,
            auth: AuthConfig::default(),
            rate_limiting: RateLimitConfig::default(),
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_key: None,
            jwt_secret: None,
            token_expiry_hours: 24,
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_minute: 60,
            burst_size: 10,
        }
    }
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            default_benchmarks: vec![
                "gsm8k".to_string(),
                "arc".to_string(),
                "boolq".to_string(),
                "hellaswag".to_string(),
            ],
            batch_size: 8,
            max_samples: None,
            few_shot_examples: 0,
            temperature: 0.0,
            top_p: 1.0,
            max_generation_length: 512,
            seed: 42,
            save_predictions: false,
            cache_dir: None,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            default_epochs: 10,
            default_batch_size: 32,
            default_learning_rate: 1e-4,
            default_weight_decay: 0.01,
            gradient_clip_value: 1.0,
            warmup_steps: 1000,
            save_every_epochs: 1,
            eval_every_epochs: 1,
            early_stopping_patience: Some(5),
            mixed_precision: true,
            data_workers: 4,
            checkpoint_dir: PathBuf::from("./checkpoints"),
            wandb: WandbConfig::default(),
        }
    }
}

impl Default for WandbConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            project: "text-to-lora".to_string(),
            entity: None,
            tags: vec!["t2l".to_string(), "hypernetwork".to_string()],
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            file: None,
            json_format: false,
            include_module_path: true,
            include_file_line: false,
            console_colors: true,
        }
    }
}

// Configuration loading and management functions
pub fn load_config(config_path: Option<&str>) -> CliResult<Config> {
    let config_file = if let Some(path) = config_path {
        PathBuf::from(path)
    } else {
        get_default_config_path()?
    };

    if !config_file.exists() {
        return Ok(Config::default());
    }

    let content = fs::read_to_string(&config_file)
        .map_err(|e| CliError::Config(format!("Failed to read config file: {}", e)))?;

    let config: Config = if config_file.extension().map_or(false, |ext| ext == "yaml" || ext == "yml") {
        serde_yaml::from_str(&content)?
    } else if config_file.extension().map_or(false, |ext| ext == "toml") {
        toml::from_str(&content)?
    } else {
        serde_json::from_str(&content)?
    };

    Ok(config)
}

pub fn save_config(config: &Config, config_path: Option<&str>) -> CliResult<()> {
    let config_file = if let Some(path) = config_path {
        PathBuf::from(path)
    } else {
        get_default_config_path()?
    };

    // Create parent directory if it doesn't exist
    if let Some(parent) = config_file.parent() {
        fs::create_dir_all(parent)?;
    }

    let content = if config_file.extension().map_or(false, |ext| ext == "yaml" || ext == "yml") {
        serde_yaml::to_string(config)?
    } else if config_file.extension().map_or(false, |ext| ext == "toml") {
        toml::to_string(config)
            .map_err(|e| CliError::Config(format!("TOML serialization error: {}", e)))?
    } else {
        serde_json::to_string_pretty(config)?
    };

    fs::write(&config_file, content)
        .map_err(|e| CliError::Config(format!("Failed to write config file: {}", e)))?;

    Ok(())
}

pub fn get_default_config_path() -> CliResult<PathBuf> {
    let proj_dirs = ProjectDirs::from("ai", "anthropic", "t2l")
        .ok_or_else(|| CliError::Config("Failed to determine config directory".to_string()))?;

    let config_dir = proj_dirs.config_dir();
    Ok(config_dir.join("config.yaml"))
}

pub fn show_config(config: &Config) -> CliResult<()> {
    let yaml_content = serde_yaml::to_string(config)?;
    println!("{}", yaml_content);
    Ok(())
}

pub fn init_config(force: bool) -> CliResult<()> {
    let config_path = get_default_config_path()?;

    if config_path.exists() && !force {
        return Err(CliError::Config(format!(
            "Configuration file already exists: {}. Use --force to overwrite.",
            config_path.display()
        )));
    }

    let default_config = Config::default();
    save_config(&default_config, Some(config_path.to_str().unwrap()))?;

    println!("Configuration file created: {}", config_path.display());
    Ok(())
}

pub fn get_config_value(config: &Config, key: &str) -> CliResult<()> {
    let value = get_nested_value(config, key)?;
    println!("{}", value);
    Ok(())
}

pub fn set_config_value(key: &str, value: &str) -> CliResult<()> {
    let config_path = get_default_config_path()?;
    let mut config = load_config(Some(config_path.to_str().unwrap()))?;

    set_nested_value(&mut config, key, value)?;
    save_config(&config, Some(config_path.to_str().unwrap()))?;

    println!("Configuration updated: {} = {}", key, value);
    Ok(())
}

fn get_nested_value(config: &Config, key: &str) -> CliResult<String> {
    // Convert config to a map for easy traversal
    let config_map = config_to_map(config)?;
    
    let keys: Vec<&str> = key.split('.').collect();
    let mut current = &config_map;

    for k in &keys[..keys.len() - 1] {
        current = current
            .get(*k)
            .and_then(|v| v.as_object())
            .ok_or_else(|| CliError::InvalidConfigKey(key.to_string()))?;
    }

    let final_key = keys.last().unwrap();
    let value = current
        .get(*final_key)
        .ok_or_else(|| CliError::InvalidConfigKey(key.to_string()))?;

    Ok(value.to_string())
}

fn set_nested_value(config: &mut Config, key: &str, value: &str) -> CliResult<()> {
    // For simplicity, implement a few common config changes
    match key {
        "server.port" => {
            config.server.port = value.parse()
                .map_err(|_| CliError::InvalidArgument(format!("Invalid port: {}", value)))?;
        }
        "server.host" => {
            config.server.host = value.to_string();
        }
        "t2l.lora.rank" => {
            config.t2l.lora.rank = value.parse()
                .map_err(|_| CliError::InvalidArgument(format!("Invalid rank: {}", value)))?;
        }
        "t2l.lora.alpha" => {
            config.t2l.lora.alpha = value.parse()
                .map_err(|_| CliError::InvalidArgument(format!("Invalid alpha: {}", value)))?;
        }
        "logging.level" => {
            config.logging.level = value.to_string();
        }
        _ => {
            return Err(CliError::InvalidConfigKey(format!(
                "Setting '{}' is not supported. Edit the config file directly.",
                key
            )));
        }
    }

    Ok(())
}

fn config_to_map(config: &Config) -> CliResult<serde_json::Map<String, serde_json::Value>> {
    let json_value = serde_json::to_value(config)?;
    Ok(json_value.as_object().unwrap().clone())
}

impl Config {
    /// Validate the configuration
    pub fn validate(&self) -> CliResult<()> {
        // Validate T2L config
        if self.t2l.lora.rank == 0 || self.t2l.lora.rank > 512 {
            return Err(CliError::ValidationError(
                "LoRA rank must be between 1 and 512".to_string(),
            ));
        }

        if self.t2l.lora.alpha <= 0.0 {
            return Err(CliError::ValidationError(
                "LoRA alpha must be positive".to_string(),
            ));
        }

        if self.t2l.hypernetwork.dropout < 0.0 || self.t2l.hypernetwork.dropout > 1.0 {
            return Err(CliError::ValidationError(
                "Dropout must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Validate server config
        if self.server.port == 0 {
            return Err(CliError::ValidationError(
                "Server port must be non-zero".to_string(),
            ));
        }

        if self.server.max_concurrent == 0 {
            return Err(CliError::ValidationError(
                "Max concurrent requests must be positive".to_string(),
            ));
        }

        // Validate evaluation config
        if self.evaluation.batch_size == 0 {
            return Err(CliError::ValidationError(
                "Evaluation batch size must be positive".to_string(),
            ));
        }

        if self.evaluation.temperature < 0.0 || self.evaluation.temperature > 2.0 {
            return Err(CliError::ValidationError(
                "Temperature must be between 0.0 and 2.0".to_string(),
            ));
        }

        if self.evaluation.top_p <= 0.0 || self.evaluation.top_p > 1.0 {
            return Err(CliError::ValidationError(
                "Top-p must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Validate training config
        if self.training.default_batch_size == 0 {
            return Err(CliError::ValidationError(
                "Training batch size must be positive".to_string(),
            ));
        }

        if self.training.default_learning_rate <= 0.0 || self.training.default_learning_rate > 1.0 {
            return Err(CliError::ValidationError(
                "Learning rate must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }
}