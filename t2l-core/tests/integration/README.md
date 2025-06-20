# T2L Integration Tests

This directory contains comprehensive integration tests for the T2L (Text-to-LoRA) system.

## Test Categories

### 1. Full Workflow Tests (`workflow_tests.rs`)
- End-to-end workflows: generate → apply → infer → export
- Multi-architecture support (Llama, Mistral, Gemma)
- Batch processing capabilities
- Memory efficiency testing
- Concurrent operations

### 2. CLI Integration Tests (`cli_tests.rs`)
- Command-line interface validation
- All subcommands: apply, export, infer
- Error handling for invalid inputs
- Configuration file support
- Batch command processing

### 3. Export Format Tests (`export_tests.rs`)
- PEFT format compatibility
- HuggingFace format export
- GGML quantization support
- OpenAI format compatibility
- Precision format testing (FP32, FP16, BF16, INT8)
- Metadata preservation

### 4. Architecture Compatibility Tests (`architecture_tests.rs`)
- Llama architecture variants (7B, 13B, 70B)
- Mistral with grouped-query attention
- Gemma architecture specifics
- Cross-architecture conversion
- Device compatibility (CPU/CUDA)
- Mixed precision support

### 5. Error Handling Tests (`error_tests.rs`)
- Missing model files
- Incompatible architectures
- Corrupted adapter files
- Dimension mismatches
- Out-of-memory scenarios
- Invalid paths
- Concurrent access conflicts
- Recovery from partial operations

### 6. Performance Benchmarks (`performance_tests.rs`)
- Adapter generation performance
- Export operation benchmarks
- Apply operation timing
- Batch processing efficiency
- Memory usage profiling
- Concurrent operation scaling
- Precision impact on performance

## Running the Tests

### Run all integration tests:
```bash
cargo test --test integration -- --test-threads=1
```

### Run specific test module:
```bash
cargo test --test integration workflow_tests
```

### Run with logging:
```bash
RUST_LOG=debug cargo test --test integration -- --nocapture
```

### Run benchmarks:
```bash
cargo test --test integration performance_tests -- --nocapture
```

## Test Data

The tests use mock models and fixtures to simulate real scenarios without requiring actual model downloads. Key fixtures include:

- Mock model configurations for Llama, Mistral, and Gemma
- Test adapters with various rank configurations
- Sample task descriptions
- Mock tokenizer configurations

## Writing New Tests

When adding new integration tests:

1. Use the provided fixtures in `fixtures.rs`
2. Follow the existing test patterns
3. Clean up temporary files using `TempDir`
4. Use meaningful test names that describe the scenario
5. Include both success and failure cases
6. Add performance assertions where relevant

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

- Tests use temporary directories for isolation
- No external dependencies or downloads required
- Deterministic test data generation
- Reasonable timeouts for all operations
- Clear error messages for debugging

## Known Limitations

- CUDA tests require GPU availability
- Some export formats (GGML, OpenAI) may not be fully implemented
- Performance benchmarks use mock memory measurements
- Large-scale tests are limited to prevent CI resource exhaustion