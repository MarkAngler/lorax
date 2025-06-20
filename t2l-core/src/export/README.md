# T2L Export Formats

This module provides format conversion capabilities for T2L LoRA adapters, enabling integration with various ML frameworks.

## Supported Formats

### GGML Format

The GGML format is used by llama.cpp and similar CPU-optimized inference engines. Our implementation provides:

- **Binary format** compatible with llama.cpp LoRA loading
- **Multiple precision options**: FP32, FP16, INT8 quantization
- **Automatic layer name mapping** from T2L to llama.cpp conventions
- **Metadata preservation** including task description and hyperparameters

#### GGML Binary Format Specification

```
Header:
- Magic number: 0x67676d6c ("ggml")
- Version: 1 (u32)
- Tensor count: N (u32)
- Architecture: string (length-prefixed)

Metadata:
- Metadata count: M (u32)
- For each metadata entry:
  - Key: string (length-prefixed)
  - Value: string (length-prefixed)

Tensors:
- For each tensor:
  - Dimensions: ndims (i32)
  - Name: string (length-prefixed)
  - Data type: dtype (u32)
  - Shape: [dim1, dim2, ...] (i32 array)
  - Data: binary tensor data
  - Alpha: scaling factor (f32)

End marker: 0xFFFFFFFF
```

#### Usage Example

```rust
use t2l_core::export::{ggml, Precision};

// Export with FP16 precision (recommended)
ggml::export_to_ggml(
    &adapter,
    Path::new("adapter.ggml"),
    Precision::Fp16
).await?;

// Use with llama.cpp
// ./main -m llama-7b.gguf --lora adapter.ggml
```

#### Layer Name Mapping

T2L uses PyTorch-style layer names, which are automatically mapped to llama.cpp conventions:

| T2L Format | GGML Format |
|------------|-------------|
| `layers.0.self_attn.q_proj` | `blk.0.attn_q` |
| `layers.0.self_attn.k_proj` | `blk.0.attn_k` |
| `layers.0.self_attn.v_proj` | `blk.0.attn_v` |
| `layers.0.self_attn.o_proj` | `blk.0.attn_output` |
| `layers.0.mlp.gate_proj` | `blk.0.ffn_gate` |
| `layers.0.mlp.up_proj` | `blk.0.ffn_up` |
| `layers.0.mlp.down_proj` | `blk.0.ffn_down` |

### HuggingFace Format

The HuggingFace format enables direct integration with the Transformers library. Our implementation provides:

- **Standard HF model structure** with config.json and safetensors weights
- **Direct loading** with `AutoModelForCausalLM.from_pretrained()`
- **Adapter-only export** compatible with PEFT library
- **Automatic tokenizer configuration** for common architectures
- **Model card generation** with usage instructions

#### HuggingFace Export Structure

```
output_directory/
├── config.json              # Model configuration
├── adapter_model.safetensors # LoRA weights
├── adapter_config.json      # Adapter configuration
├── generation_config.json   # Generation parameters
├── tokenizer_config.json    # Tokenizer settings
├── special_tokens_map.json  # Special token mappings
└── README.md               # Model card with usage
```

#### Usage Example

```rust
use t2l_core::export::{huggingface, Precision};

// Export to HuggingFace format
huggingface::export_to_hf(
    &adapter,
    Some("meta-llama/Llama-2-7b-hf"), // Base model
    Path::new("./hf_model/"),
    Precision::Fp16
).await?;

// Use with Transformers
// from transformers import AutoModelForCausalLM
// model = AutoModelForCausalLM.from_pretrained("./hf_model/")
```

#### Supported Architectures

- **LLaMA** (1, 2, 3): Full configuration support
- **Mistral**: Including sliding window attention
- **Gemma**: Google's efficient models
- **Custom**: Configurable for other architectures

### PEFT Format

The PEFT (Parameter-Efficient Fine-Tuning) format is specifically designed for HuggingFace's PEFT library:

- **Minimal overhead** - only stores adapter weights
- **Easy integration** with any supported base model
- **Standard format** recognized by the PEFT ecosystem

#### Usage Example

```rust
use t2l_core::export::{peft, Precision};

// Export to PEFT format
peft::export_to_peft(
    &adapter,
    Some("meta-llama/Llama-2-7b-hf"),
    Path::new("./peft_adapter/"),
    Precision::Fp16
).await?;

// Use with PEFT
// from peft import PeftModel
// model = PeftModel.from_pretrained(base_model, "./peft_adapter/")
```

### OpenAI Format

The OpenAI format provides compatibility with OpenAI's API structure (coming soon).

## Precision Options

- **FP32**: Full 32-bit floating point (largest, highest precision)
- **FP16**: 16-bit floating point (recommended balance)
- **INT8**: 8-bit quantization (smallest, some quality loss)

## Error Handling

All export functions return detailed error messages with context:

```rust
match ggml::export_to_ggml(&adapter, path, precision).await {
    Ok(_) => println!("Export successful"),
    Err(e) => eprintln!("Export failed: {:#}", e),
}
```