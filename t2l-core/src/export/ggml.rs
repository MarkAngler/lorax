//! GGML format exporter for T2L adapters
//!
//! This module provides functionality to export T2L LoRA adapters to GGML format
//! for use with llama.cpp and similar implementations.

use crate::export::Precision;
use crate::lora::LoraParameters;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// GGML file magic number (0x67676d6c -> "ggml")
const GGML_MAGIC: u32 = 0x67676d6c;

/// GGML LoRA file version
const GGML_LORA_VERSION: u32 = 1;

/// GGML data types
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q5_0 = 6,
    // Q5_1 = 7,
    Q8_0 = 8,
    // Additional quantization types can be added here
}

impl GGMLType {
    /// Get the type for a given precision
    fn from_precision(precision: Precision) -> Self {
        match precision {
            Precision::Fp32 => GGMLType::F32,
            Precision::Fp16 => GGMLType::F16,
            Precision::Int8 => GGMLType::Q8_0,
        }
    }
    
    /// Get the size in bytes per element
    fn element_size(&self) -> usize {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 => 2,
            GGMLType::Q4_0 => 18, // 32 values in 18 bytes
            GGMLType::Q4_1 => 20, // 32 values in 20 bytes
            GGMLType::Q8_0 => 34, // 32 values in 34 bytes
        }
    }
}

/// Export T2L adapter to GGML format
pub async fn export_to_ggml(
    adapter: &LoraParameters,
    output_path: &Path,
    precision: Precision,
) -> Result<()> {
    tracing::info!("Exporting T2L adapter to GGML format at: {}", output_path.display());
    
    // Ensure output directory exists
    if let Some(parent) = output_path.parent() {
        tokio::fs::create_dir_all(parent).await
            .context("Failed to create output directory")?;
    }
    
    // Create output file with .ggml extension
    let output_file = if output_path.extension().is_some() {
        output_path.to_path_buf()
    } else {
        output_path.with_extension("ggml")
    };
    
    // Create GGML writer and export
    let mut writer = GGMLWriter::new(
        File::create(&output_file).context("Failed to create output file")?,
        precision,
    );
    
    writer.write_header(adapter)
        .context("Failed to write GGML header")?;
    
    writer.write_metadata(adapter)
        .context("Failed to write metadata")?;
    
    writer.write_tensors(adapter)
        .context("Failed to write tensors")?;
    
    writer.finalize()
        .context("Failed to finalize GGML file")?;
    
    tracing::info!("✅ Successfully exported to GGML format: {}", output_file.display());
    
    Ok(())
}

/// GGML binary format writer
struct GGMLWriter {
    writer: BufWriter<File>,
    precision: Precision,
    tensor_count: u32,
}

impl GGMLWriter {
    /// Create a new GGML writer
    fn new(file: File, precision: Precision) -> Self {
        Self {
            writer: BufWriter::new(file),
            precision,
            tensor_count: 0,
        }
    }
    
    /// Write GGML header
    fn write_header(&mut self, adapter: &LoraParameters) -> Result<()> {
        // Write magic number
        self.writer.write_u32::<LittleEndian>(GGML_MAGIC)
            .context("Failed to write magic number")?;
        
        // Write version
        self.writer.write_u32::<LittleEndian>(GGML_LORA_VERSION)
            .context("Failed to write version")?;
        
        // Count total tensors (2 per layer: A and B matrices)
        self.tensor_count = (adapter.layers.len() * 2) as u32;
        self.writer.write_u32::<LittleEndian>(self.tensor_count)
            .context("Failed to write tensor count")?;
        
        // Write target architecture
        let arch_bytes = adapter.config.target_architecture.as_bytes();
        self.writer.write_u32::<LittleEndian>(arch_bytes.len() as u32)
            .context("Failed to write architecture length")?;
        self.writer.write_all(arch_bytes)
            .context("Failed to write architecture")?;
        
        Ok(())
    }
    
    /// Write metadata section
    fn write_metadata(&mut self, adapter: &LoraParameters) -> Result<()> {
        // Write metadata count
        let mut metadata = HashMap::new();
        
        // Add standard metadata
        metadata.insert("lora.type", "lora".to_string());
        metadata.insert("lora.alpha", adapter.config.default_alpha.to_string());
        metadata.insert("lora.rank", adapter.config.default_rank.to_string());
        
        // Add target modules
        let target_modules = adapter.config.target_modules.join(",");
        metadata.insert("lora.target_modules", target_modules);
        
        // Add optional metadata
        if let Some(param_metadata) = &adapter.metadata {
            metadata.insert("lora.task", param_metadata.task_description.clone());
            metadata.insert("lora.created_at", param_metadata.created_at.to_rfc3339());
            metadata.insert("lora.generator_version", param_metadata.generator_version.clone());
        }
        
        // Write metadata count
        self.writer.write_u32::<LittleEndian>(metadata.len() as u32)
            .context("Failed to write metadata count")?;
        
        // Write each metadata entry
        for (key, value) in metadata {
            // Write key
            let key_bytes = key.as_bytes();
            self.writer.write_u32::<LittleEndian>(key_bytes.len() as u32)?;
            self.writer.write_all(key_bytes)?;
            
            // Write value
            let value_bytes = value.as_bytes();
            self.writer.write_u32::<LittleEndian>(value_bytes.len() as u32)?;
            self.writer.write_all(value_bytes)?;
        }
        
        Ok(())
    }
    
    /// Write all tensors
    fn write_tensors(&mut self, adapter: &LoraParameters) -> Result<()> {
        // Sort layers by name for consistent output
        let mut sorted_layers: Vec<_> = adapter.layers.iter().collect();
        sorted_layers.sort_by_key(|(name, _)| name.as_str());
        
        for (layer_name, lora_layer) in sorted_layers {
            // Write A matrix
            self.write_tensor(
                &format!("{}.lora_A", map_layer_name_to_ggml(layer_name)),
                &lora_layer.a_weights,
                &[lora_layer.input_dim as i32, lora_layer.rank as i32],
                lora_layer.alpha,
            ).context(format!("Failed to write A matrix for layer {}", layer_name))?;
            
            // Write B matrix
            self.write_tensor(
                &format!("{}.lora_B", map_layer_name_to_ggml(layer_name)),
                &lora_layer.b_weights,
                &[lora_layer.rank as i32, lora_layer.output_dim as i32],
                lora_layer.alpha,
            ).context(format!("Failed to write B matrix for layer {}", layer_name))?;
        }
        
        Ok(())
    }
    
    /// Write a single tensor
    fn write_tensor(
        &mut self,
        name: &str,
        data: &[f32],
        shape: &[i32],
        alpha: f32,
    ) -> Result<()> {
        // Write tensor header
        self.write_tensor_header(name, shape)?;
        
        // Write tensor data based on precision
        match self.precision {
            Precision::Fp32 => self.write_f32_data(data),
            Precision::Fp16 => self.write_f16_data(data),
            Precision::Int8 => self.write_q8_0_data(data),
        }?;
        
        // Write scaling factor (alpha) as additional metadata
        self.writer.write_f32::<LittleEndian>(alpha)
            .context("Failed to write alpha")?;
        
        Ok(())
    }
    
    /// Write tensor header
    fn write_tensor_header(&mut self, name: &str, shape: &[i32]) -> Result<()> {
        // Write number of dimensions
        let n_dims = shape.len() as i32;
        self.writer.write_i32::<LittleEndian>(n_dims)
            .context("Failed to write dimension count")?;
        
        // Write name length and name
        let name_bytes = name.as_bytes();
        self.writer.write_i32::<LittleEndian>(name_bytes.len() as i32)
            .context("Failed to write name length")?;
        self.writer.write_all(name_bytes)
            .context("Failed to write tensor name")?;
        
        // Write data type
        let dtype = GGMLType::from_precision(self.precision);
        self.writer.write_u32::<LittleEndian>(dtype as u32)
            .context("Failed to write data type")?;
        
        // Write shape
        for &dim in shape {
            self.writer.write_i32::<LittleEndian>(dim)
                .context("Failed to write dimension")?;
        }
        
        Ok(())
    }
    
    /// Write f32 data
    fn write_f32_data(&mut self, data: &[f32]) -> Result<()> {
        for &value in data {
            self.writer.write_f32::<LittleEndian>(value)
                .context("Failed to write f32 value")?;
        }
        Ok(())
    }
    
    /// Write f16 data
    fn write_f16_data(&mut self, data: &[f32]) -> Result<()> {
        use half::f16;
        
        for &value in data {
            let f16_value = f16::from_f32(value);
            self.writer.write_u16::<LittleEndian>(f16_value.to_bits())
                .context("Failed to write f16 value")?;
        }
        Ok(())
    }
    
    /// Write quantized int8 data (Q8_0 format)
    fn write_q8_0_data(&mut self, data: &[f32]) -> Result<()> {
        // Q8_0 quantization: blocks of 32 values
        const BLOCK_SIZE: usize = 32;
        
        for chunk in data.chunks(BLOCK_SIZE) {
            // Find min and max for this block
            let min = chunk.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            
            // Calculate scale
            let scale = if max != min {
                255.0 / (max - min)
            } else {
                0.0
            };
            
            // Write scale as f16
            use half::f16;
            let scale_f16 = f16::from_f32(scale);
            self.writer.write_u16::<LittleEndian>(scale_f16.to_bits())?;
            
            // Quantize and write values
            let mut quantized = vec![0u8; BLOCK_SIZE];
            for (i, &value) in chunk.iter().enumerate() {
                let normalized = if scale != 0.0 {
                    ((value - min) * scale).round() as i32
                } else {
                    0
                };
                quantized[i] = normalized.clamp(0, 255) as u8;
            }
            
            // Pad with zeros if chunk is smaller than block size
            for i in chunk.len()..BLOCK_SIZE {
                quantized[i] = 0;
            }
            
            self.writer.write_all(&quantized)?;
        }
        
        Ok(())
    }
    
    /// Finalize the GGML file
    fn finalize(mut self) -> Result<()> {
        // Write end marker
        self.writer.write_u32::<LittleEndian>(0xFFFFFFFF)
            .context("Failed to write end marker")?;
        
        // Flush buffer
        self.writer.flush()
            .context("Failed to flush writer")?;
        
        Ok(())
    }
}

/// Map T2L layer names to GGML/llama.cpp compatible names
fn map_layer_name_to_ggml(t2l_name: &str) -> String {
    // T2L format: "layers.0.self_attn.q_proj"
    // GGML format: "blk.0.attn_q"
    
    if let Some(layer_match) = parse_layer_name(t2l_name) {
        let block_num = layer_match.block;
        let component = map_component_name(&layer_match.component);
        
        format!("blk.{}.{}", block_num, component)
    } else {
        // Fallback: use original name
        t2l_name.to_string()
    }
}

struct LayerMatch {
    block: usize,
    component: String,
}

fn parse_layer_name(name: &str) -> Option<LayerMatch> {
    // Parse patterns like "layers.0.self_attn.q_proj"
    let parts: Vec<&str> = name.split('.').collect();
    
    if parts.len() >= 4 && parts[0] == "layers" {
        if let Ok(block) = parts[1].parse::<usize>() {
            let component = parts[3..].join(".");
            return Some(LayerMatch { block, component });
        }
    }
    
    None
}

fn map_component_name(component: &str) -> &str {
    match component {
        "q_proj" => "attn_q",
        "k_proj" => "attn_k", 
        "v_proj" => "attn_v",
        "o_proj" => "attn_output",
        "gate_proj" => "ffn_gate",
        "up_proj" => "ffn_up",
        "down_proj" => "ffn_down",
        _ => component,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_layer_name_mapping() {
        assert_eq!(
            map_layer_name_to_ggml("layers.0.self_attn.q_proj"),
            "blk.0.attn_q"
        );
        assert_eq!(
            map_layer_name_to_ggml("layers.10.mlp.gate_proj"),
            "blk.10.ffn_gate"
        );
        assert_eq!(
            map_layer_name_to_ggml("unknown_format"),
            "unknown_format"
        );
    }
    
    #[test]
    fn test_ggml_type_sizes() {
        assert_eq!(GGMLType::F32.element_size(), 4);
        assert_eq!(GGMLType::F16.element_size(), 2);
        assert_eq!(GGMLType::Q8_0.element_size(), 34);
    }
}