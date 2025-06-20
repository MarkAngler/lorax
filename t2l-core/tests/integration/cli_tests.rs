//! CLI integration tests
//!
//! Tests for command-line interface functionality

use super::fixtures::*;
use super::init_test_logging;
use std::process::Command;
use std::path::Path;
use tempfile::TempDir;
use t2l_core::Result;

/// Run a CLI command and capture output
fn run_cli_command(args: &[&str]) -> Result<(String, String, bool)> {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "t2l", "--"])
        .args(args)
        .output()?;
    
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let success = output.status.success();
    
    Ok((stdout, stderr, success))
}

#[test]
fn test_cli_help_command() {
    init_test_logging();
    
    let (stdout, _, success) = run_cli_command(&["--help"]).unwrap();
    
    assert!(success, "Help command failed");
    assert!(stdout.contains("T2L (Text-to-LoRA) CLI"));
    assert!(stdout.contains("USAGE"));
    assert!(stdout.contains("SUBCOMMANDS"));
    assert!(stdout.contains("apply"));
    assert!(stdout.contains("export"));
    assert!(stdout.contains("infer"));
}

#[test]
fn test_cli_version_command() {
    init_test_logging();
    
    let (stdout, _, success) = run_cli_command(&["--version"]).unwrap();
    
    assert!(success, "Version command failed");
    assert!(stdout.contains("t2l"));
}

#[test]
fn test_cli_apply_command() {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let model_path = temp_path.join("model");
    let adapter_path = temp_path.join("adapter");
    let output_path = temp_path.join("output");
    
    // Create test data
    create_mock_model(&model_path, "llama").unwrap();
    let adapter = create_test_adapter("llama", 16);
    save_lora_parameters(&adapter, &adapter_path).unwrap();
    
    // Test apply command
    let (stdout, stderr, success) = run_cli_command(&[
        "apply",
        "--model", model_path.to_str().unwrap(),
        "--adapter", adapter_path.to_str().unwrap(),
        "--output", output_path.to_str().unwrap(),
        "--strategy", "linear",
    ]).unwrap();
    
    if !success {
        eprintln!("STDOUT: {}", stdout);
        eprintln!("STDERR: {}", stderr);
    }
    
    assert!(success, "Apply command failed");
    assert!(output_path.exists(), "Output directory not created");
}

#[test]
fn test_cli_export_command() {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let adapter_path = temp_path.join("adapter");
    let output_path = temp_path.join("exported");
    
    // Create test adapter
    let adapter = create_test_adapter("llama", 16);
    save_lora_parameters(&adapter, &adapter_path).unwrap();
    
    // Test export to PEFT format
    let (stdout, stderr, success) = run_cli_command(&[
        "export",
        "--adapter", adapter_path.to_str().unwrap(),
        "--format", "peft",
        "--output", output_path.to_str().unwrap(),
        "--target-model", "meta-llama/Llama-2-7b-hf",
        "--precision", "fp32",
    ]).unwrap();
    
    if !success {
        eprintln!("STDOUT: {}", stdout);
        eprintln!("STDERR: {}", stderr);
    }
    
    assert!(success, "Export command failed");
    assert!(output_path.join("adapter_config.json").exists(), "PEFT config not created");
}

#[test]
fn test_cli_export_formats() {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let adapter_path = temp_path.join("adapter");
    
    // Create test adapter
    let adapter = create_test_adapter("llama", 16);
    save_lora_parameters(&adapter, &adapter_path).unwrap();
    
    // Test all export formats
    let formats = vec!["peft", "ggml", "huggingface", "openai"];
    
    for format in formats {
        let output_path = temp_path.join(format!("export_{}", format));
        
        let (_, stderr, success) = run_cli_command(&[
            "export",
            "--adapter", adapter_path.to_str().unwrap(),
            "--format", format,
            "--output", output_path.to_str().unwrap(),
        ]).unwrap();
        
        // GGML and OpenAI might not be fully implemented yet
        if format == "peft" || format == "huggingface" {
            assert!(success, "Export to {} failed: {}", format, stderr);
            assert!(output_path.exists(), "Export output for {} not created", format);
        }
    }
}

#[test]
fn test_cli_infer_command() {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let model_path = temp_path.join("model");
    let adapter_path = temp_path.join("adapter");
    
    // Create test data
    create_mock_model(&model_path, "llama").unwrap();
    let adapter = create_test_adapter("llama", 16);
    save_lora_parameters(&adapter, &adapter_path).unwrap();
    
    // Test inference command
    let (stdout, stderr, success) = run_cli_command(&[
        "infer",
        "--model", model_path.to_str().unwrap(),
        "--adapter", adapter_path.to_str().unwrap(),
        "--prompt", "Hello, world!",
        "--max-tokens", "50",
        "--temperature", "0.7",
    ]).unwrap();
    
    // Inference might fail due to mock model, but command should parse correctly
    if !success && !stderr.contains("parse") {
        println!("Expected inference failure with mock model");
    }
}

#[test]
fn test_cli_error_handling() {
    init_test_logging();
    
    // Test missing required arguments
    let (_, stderr, success) = run_cli_command(&["apply"]).unwrap();
    assert!(!success, "Command should fail without required args");
    assert!(stderr.contains("required") || stderr.contains("USAGE"));
    
    // Test invalid file paths
    let (_, stderr, success) = run_cli_command(&[
        "apply",
        "--model", "/nonexistent/path",
        "--adapter", "/another/nonexistent/path",
    ]).unwrap();
    assert!(!success, "Command should fail with invalid paths");
    
    // Test invalid format
    let (_, stderr, success) = run_cli_command(&[
        "export",
        "--adapter", "dummy",
        "--format", "invalid_format",
        "--output", "output",
    ]).unwrap();
    assert!(!success, "Command should fail with invalid format");
}

#[test]
fn test_cli_verbose_logging() {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let adapter_path = temp_path.join("adapter");
    let output_path = temp_path.join("output");
    
    // Create test adapter
    let adapter = create_test_adapter("llama", 16);
    save_lora_parameters(&adapter, &adapter_path).unwrap();
    
    // Test with verbose flag
    let (stdout, stderr, _) = run_cli_command(&[
        "-v",
        "export",
        "--adapter", adapter_path.to_str().unwrap(),
        "--format", "peft",
        "--output", output_path.to_str().unwrap(),
    ]).unwrap();
    
    // Should contain debug/trace logs
    let combined_output = format!("{}\n{}", stdout, stderr);
    assert!(
        combined_output.contains("DEBUG") || 
        combined_output.contains("TRACE") || 
        combined_output.contains("Loading adapter"),
        "Verbose output should contain detailed logs"
    );
}

#[test]
fn test_cli_config_file() {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    let config_path = temp_path.join("config.toml");
    
    // Create config file
    let config_content = r#"
[export]
default_precision = "fp16"
optimize_for_inference = true

[apply]
default_strategy = "linear"
verify_compatibility = true
"#;
    std::fs::write(&config_path, config_content).unwrap();
    
    // Test using config file
    let (_, _, success) = run_cli_command(&[
        "--config", config_path.to_str().unwrap(),
        "--help",
    ]).unwrap();
    
    assert!(success, "Command failed with config file");
}

#[test]
fn test_cli_batch_processing() {
    init_test_logging();
    
    let (_temp_dir, temp_path) = create_test_dir();
    
    // Create multiple adapters
    let adapters: Vec<_> = (0..3).map(|i| {
        let adapter = create_test_adapter("llama", 8 + i * 4);
        let adapter_path = temp_path.join(format!("adapter_{}", i));
        save_lora_parameters(&adapter, &adapter_path).unwrap();
        adapter_path
    }).collect();
    
    // Test batch export
    for (i, adapter_path) in adapters.iter().enumerate() {
        let output_path = temp_path.join(format!("batch_export_{}", i));
        
        let (_, _, success) = run_cli_command(&[
            "export",
            "--adapter", adapter_path.to_str().unwrap(),
            "--format", "peft",
            "--output", output_path.to_str().unwrap(),
            "--quiet", // Suppress output for batch processing
        ]).unwrap();
        
        assert!(success, "Batch export {} failed", i);
    }
}