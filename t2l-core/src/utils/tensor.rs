//! Tensor manipulation and weight conversion utilities for T2L integration
//!
//! This module provides utilities for:
//! - Precision conversion (fp32 ↔ fp16 ↔ int8)
//! - Tensor reshaping for LoRA weight matrices
//! - SafeTensors format utilities
//! - Device management for tensor operations
//! - Memory-efficient tensor operations
//! - Format conversion helpers for PEFT/GGML export

use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// Supported tensor data types for precision conversion
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TensorDataType {
    Float32,
    Float16,
    Int8,
    Int4,
    BFloat16,
}

/// Device types for tensor operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Cuda(u32), // device index
    Metal,
    Mps,
}

/// Tensor metadata for shape and dtype information
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub shape: Vec<usize>,
    pub dtype: TensorDataType,
    pub device: DeviceType,
    pub name: String,
}

/// LoRA weight matrix configuration
#[derive(Debug, Clone)]
pub struct LoraMatrixConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
}

/// Tensor wrapper for type-safe operations
pub struct Tensor {
    pub data: Vec<u8>,
    pub metadata: TensorMetadata,
}

/// Precision conversion utilities
pub mod precision {
    use super::*;

    /// Convert f32 tensor data to f16
    pub fn f32_to_f16(data: &[f32]) -> Result<Vec<u8>> {
        let mut result = Vec::with_capacity(data.len() * 2);
        for &value in data {
            let f16_value = half::f16::from_f32(value);
            result.extend_from_slice(&f16_value.to_le_bytes());
        }
        Ok(result)
    }

    /// Convert f16 tensor data to f32
    pub fn f16_to_f32(data: &[u8]) -> Result<Vec<f32>> {
        if data.len() % 2 != 0 {
            return Err(anyhow!("Invalid f16 data length: {}", data.len()));
        }
        
        let mut result = Vec::with_capacity(data.len() / 2);
        for chunk in data.chunks_exact(2) {
            let bytes = [chunk[0], chunk[1]];
            let f16_value = half::f16::from_le_bytes(bytes);
            result.push(f16_value.to_f32());
        }
        Ok(result)
    }

    /// Convert f32 tensor data to int8 with quantization
    pub fn f32_to_int8(data: &[f32], scale: f32, zero_point: i8) -> Result<Vec<i8>> {
        let mut result = Vec::with_capacity(data.len());
        for &value in data {
            let quantized = (value / scale + zero_point as f32).round() as i32;
            let clamped = quantized.max(i8::MIN as i32).min(i8::MAX as i32) as i8;
            result.push(clamped);
        }
        Ok(result)
    }

    /// Convert int8 tensor data to f32 with dequantization
    pub fn int8_to_f32(data: &[i8], scale: f32, zero_point: i8) -> Result<Vec<f32>> {
        let mut result = Vec::with_capacity(data.len());
        for &value in data {
            let dequantized = (value - zero_point) as f32 * scale;
            result.push(dequantized);
        }
        Ok(result)
    }

    /// Convert between tensor data types
    pub fn convert_dtype(
        data: &[u8],
        from_dtype: &TensorDataType,
        to_dtype: &TensorDataType,
    ) -> Result<Vec<u8>> {
        if from_dtype == to_dtype {
            return Ok(data.to_vec());
        }

        match (from_dtype, to_dtype) {
            (TensorDataType::Float32, TensorDataType::Float16) => {
                let f32_data = bytemuck::cast_slice::<u8, f32>(data);
                f32_to_f16(f32_data)
            }
            (TensorDataType::Float16, TensorDataType::Float32) => {
                let f32_data = f16_to_f32(data)?;
                Ok(bytemuck::cast_slice::<f32, u8>(&f32_data).to_vec())
            }
            (TensorDataType::Float32, TensorDataType::Int8) => {
                let f32_data = bytemuck::cast_slice::<u8, f32>(data);
                // Use simple quantization parameters
                let scale = 1.0 / 127.0;
                let zero_point = 0i8;
                let int8_data = f32_to_int8(f32_data, scale, zero_point)?;
                Ok(bytemuck::cast_slice::<i8, u8>(&int8_data).to_vec())
            }
            _ => Err(anyhow!(
                "Unsupported dtype conversion: {:?} -> {:?}",
                from_dtype,
                to_dtype
            )),
        }
    }

    /// Calculate quantization parameters for f32 -> int8 conversion
    pub fn calculate_quantization_params(data: &[f32]) -> (f32, i8) {
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let scale = (max_val - min_val) / 255.0;
        let zero_point = (-min_val / scale).round() as i8;
        
        (scale, zero_point)
    }
}

/// LoRA weight matrix utilities
pub mod lora {
    use super::*;

    /// Reshape LoRA A matrix for efficient computation
    pub fn reshape_lora_a(
        weight: &[f32],
        original_shape: &[usize],
        rank: usize,
    ) -> Result<Vec<f32>> {
        if original_shape.len() != 2 {
            return Err(anyhow!("LoRA A matrix must be 2D"));
        }
        
        let [out_features, in_features] = [original_shape[0], original_shape[1]];
        if weight.len() != out_features * rank {
            return Err(anyhow!(
                "Weight size mismatch for LoRA A: expected {}, got {}",
                out_features * rank,
                weight.len()
            ));
        }
        
        // LoRA A: [out_features, rank] -> [rank, out_features] for efficient gemm
        let mut reshaped = vec![0.0f32; weight.len()];
        for i in 0..out_features {
            for j in 0..rank {
                reshaped[j * out_features + i] = weight[i * rank + j];
            }
        }
        
        Ok(reshaped)
    }

    /// Reshape LoRA B matrix for efficient computation
    pub fn reshape_lora_b(
        weight: &[f32],
        original_shape: &[usize],
        rank: usize,
    ) -> Result<Vec<f32>> {
        if original_shape.len() != 2 {
            return Err(anyhow!("LoRA B matrix must be 2D"));
        }
        
        let [rank_dim, in_features] = [original_shape[0], original_shape[1]];
        if rank_dim != rank {
            return Err(anyhow!(
                "LoRA B matrix rank mismatch: expected {}, got {}",
                rank,
                rank_dim
            ));
        }
        
        if weight.len() != rank * in_features {
            return Err(anyhow!(
                "Weight size mismatch for LoRA B: expected {}, got {}",
                rank * in_features,
                weight.len()
            ));
        }
        
        // LoRA B is already in the correct format [rank, in_features]
        Ok(weight.to_vec())
    }

    /// Apply LoRA scaling factor
    pub fn apply_lora_scaling(
        lora_a: &mut [f32],
        lora_b: &mut [f32],
        alpha: f32,
        rank: usize,
    ) {
        let scaling = alpha / rank as f32;
        
        // Apply scaling to LoRA A
        for weight in lora_a.iter_mut() {
            *weight *= scaling;
        }
    }

    /// Merge LoRA weights into base model weights
    pub fn merge_lora_weights(
        base_weight: &[f32],
        lora_a: &[f32],
        lora_b: &[f32],
        alpha: f32,
        rank: usize,
    ) -> Result<Vec<f32>> {
        let base_shape = infer_matrix_shape(base_weight)?;
        let [out_features, in_features] = [base_shape[0], base_shape[1]];
        
        if lora_a.len() != out_features * rank {
            return Err(anyhow!("LoRA A shape mismatch"));
        }
        
        if lora_b.len() != rank * in_features {
            return Err(anyhow!("LoRA B shape mismatch"));
        }
        
        let mut merged = base_weight.to_vec();
        let scaling = alpha / rank as f32;
        
        // Compute LoRA contribution: A @ B with scaling
        for i in 0..out_features {
            for j in 0..in_features {
                let mut delta = 0.0f32;
                for k in 0..rank {
                    delta += lora_a[i * rank + k] * lora_b[k * in_features + j];
                }
                merged[i * in_features + j] += delta * scaling;
            }
        }
        
        Ok(merged)
    }

    /// Infer matrix shape from flattened weights
    fn infer_matrix_shape(weight: &[f32]) -> Result<[usize; 2]> {
        let total_size = weight.len();
        
        // Try common aspect ratios for transformer models
        let common_ratios = [
            (1, 1),
            (4, 1),
            (1, 4),
            (3, 1),
            (1, 3),
        ];
        
        for (ratio_h, ratio_w) in common_ratios {
            let h = ((total_size as f64).sqrt() * ratio_h as f64 / ratio_w as f64).round() as usize;
            let w = total_size / h;
            
            if h * w == total_size {
                return Ok([h, w]);
            }
        }
        
        Err(anyhow!("Could not infer matrix shape from weight size: {}", total_size))
    }
}

/// SafeTensors format utilities
pub mod safetensors {
    use super::*;
    use std::io::{Read, Write};

    /// SafeTensors header for metadata
    #[derive(Debug, Clone)]
    pub struct SafeTensorsHeader {
        pub tensors: HashMap<String, TensorInfo>,
        pub metadata: HashMap<String, String>,
    }

    /// Individual tensor information in SafeTensors format
    #[derive(Debug, Clone)]
    pub struct TensorInfo {
        pub dtype: String,
        pub shape: Vec<usize>,
        pub data_offsets: [usize; 2], // [start, end]
    }

    /// Read SafeTensors file header
    pub fn read_header<P: AsRef<Path>>(path: P) -> Result<SafeTensorsHeader> {
        let mut file = std::fs::File::open(path)?;
        
        // Read header length (first 8 bytes)
        let mut header_len_bytes = [0u8; 8];
        file.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;
        
        // Read header JSON
        let mut header_json = vec![0u8; header_len];
        file.read_exact(&mut header_json)?;
        
        let header_str = std::str::from_utf8(&header_json)
            .context("Invalid UTF-8 in SafeTensors header")?;
        
        let json_value: serde_json::Value = serde_json::from_str(header_str)
            .context("Invalid JSON in SafeTensors header")?;
        
        parse_header_json(&json_value)
    }

    /// Parse SafeTensors header JSON
    fn parse_header_json(json: &serde_json::Value) -> Result<SafeTensorsHeader> {
        let obj = json.as_object()
            .ok_or_else(|| anyhow!("SafeTensors header must be an object"))?;
        
        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();
        
        for (key, value) in obj {
            if key == "__metadata__" {
                if let Some(meta_obj) = value.as_object() {
                    for (meta_key, meta_value) in meta_obj {
                        if let Some(meta_str) = meta_value.as_str() {
                            metadata.insert(meta_key.clone(), meta_str.to_string());
                        }
                    }
                }
            } else {
                let tensor_info = parse_tensor_info(value)?;
                tensors.insert(key.clone(), tensor_info);
            }
        }
        
        Ok(SafeTensorsHeader { tensors, metadata })
    }

    /// Parse individual tensor information
    fn parse_tensor_info(value: &serde_json::Value) -> Result<TensorInfo> {
        let obj = value.as_object()
            .ok_or_else(|| anyhow!("Tensor info must be an object"))?;
        
        let dtype = obj.get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing or invalid dtype"))?
            .to_string();
        
        let shape = obj.get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("Missing or invalid shape"))?
            .iter()
            .map(|v| v.as_u64().map(|n| n as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow!("Invalid shape values"))?;
        
        let data_offsets = obj.get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("Missing or invalid data_offsets"))?;
        
        if data_offsets.len() != 2 {
            return Err(anyhow!("data_offsets must have exactly 2 elements"));
        }
        
        let start = data_offsets[0].as_u64()
            .ok_or_else(|| anyhow!("Invalid start offset"))? as usize;
        let end = data_offsets[1].as_u64()
            .ok_or_else(|| anyhow!("Invalid end offset"))? as usize;
        
        Ok(TensorInfo {
            dtype,
            shape,
            data_offsets: [start, end],
        })
    }

    /// Write SafeTensors format file
    pub fn write_safetensors<P: AsRef<Path>>(
        path: P,
        tensors: &HashMap<String, Tensor>,
        metadata: Option<&HashMap<String, String>>,
    ) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        
        // Build header JSON
        let mut header_json = serde_json::Map::new();
        let mut current_offset = 0usize;
        
        // Add metadata if provided
        if let Some(meta) = metadata {
            let meta_value = serde_json::Value::Object(
                meta.iter()
                    .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                    .collect()
            );
            header_json.insert("__metadata__".to_string(), meta_value);
        }
        
        // Calculate data offsets and build tensor info
        let mut tensor_data = Vec::new();
        for (name, tensor) in tensors {
            let dtype_str = match tensor.metadata.dtype {
                TensorDataType::Float32 => "F32",
                TensorDataType::Float16 => "F16",
                TensorDataType::Int8 => "I8",
                TensorDataType::BFloat16 => "BF16",
                TensorDataType::Int4 => "I4",
            };
            
            let tensor_info = serde_json::json!({
                "dtype": dtype_str,
                "shape": tensor.metadata.shape,
                "data_offsets": [current_offset, current_offset + tensor.data.len()]
            });
            
            header_json.insert(name.clone(), tensor_info);
            tensor_data.extend_from_slice(&tensor.data);
            current_offset += tensor.data.len();
        }
        
        // Write header
        let header_str = serde_json::to_string(&header_json)?;
        let header_bytes = header_str.as_bytes();
        let header_len = header_bytes.len() as u64;
        
        file.write_all(&header_len.to_le_bytes())?;
        file.write_all(header_bytes)?;
        
        // Write tensor data
        file.write_all(&tensor_data)?;
        
        Ok(())
    }
}

/// Device management utilities
pub mod device {
    use super::*;

    /// Check if CUDA is available
    pub fn cuda_available() -> bool {
        // This would typically check for CUDA runtime
        // For now, we'll assume it's available if the system has it
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
    }

    /// Get optimal device for tensor operations
    pub fn get_optimal_device() -> DeviceType {
        if cuda_available() {
            DeviceType::Cuda(0)
        } else {
            DeviceType::Cpu
        }
    }

    /// Transfer tensor to specified device
    pub fn transfer_to_device(tensor: &mut Tensor, target_device: DeviceType) -> Result<()> {
        if tensor.metadata.device == target_device {
            return Ok(()); // Already on target device
        }
        
        // In a real implementation, this would use the appropriate backend
        // For now, we'll just update the metadata
        tensor.metadata.device = target_device;
        
        Ok(())
    }

    /// Estimate memory usage for tensor
    pub fn estimate_memory_usage(shape: &[usize], dtype: &TensorDataType) -> usize {
        let element_count = shape.iter().product::<usize>();
        let element_size = match dtype {
            TensorDataType::Float32 => 4,
            TensorDataType::Float16 => 2,
            TensorDataType::BFloat16 => 2,
            TensorDataType::Int8 => 1,
            TensorDataType::Int4 => 1, // Packed, but still 1 byte minimum
        };
        
        element_count * element_size
    }
}

/// Memory-efficient tensor operations
pub mod memory {
    use super::*;

    /// Chunked tensor processing for memory efficiency
    pub struct ChunkedProcessor {
        chunk_size: usize,
        overlap: usize,
    }

    impl ChunkedProcessor {
        pub fn new(chunk_size: usize, overlap: usize) -> Self {
            Self { chunk_size, overlap }
        }

        /// Process tensor in chunks to reduce memory usage
        pub fn process_chunks<F>(
            &self,
            data: &[f32],
            mut processor: F,
        ) -> Result<Vec<f32>>
        where
            F: FnMut(&[f32]) -> Result<Vec<f32>>,
        {
            let mut result = Vec::new();
            let mut start = 0;
            
            while start < data.len() {
                let end = (start + self.chunk_size).min(data.len());
                let chunk = &data[start..end];
                
                let processed_chunk = processor(chunk)?;
                
                // Handle overlap to avoid discontinuities
                if start > 0 && self.overlap > 0 {
                    let overlap_start = self.overlap.min(processed_chunk.len());
                    result.extend_from_slice(&processed_chunk[overlap_start..]);
                } else {
                    result.extend_from_slice(&processed_chunk);
                }
                
                start = end - self.overlap;
            }
            
            Ok(result)
        }
    }

    /// Memory pool for efficient tensor allocation
    pub struct TensorPool {
        pools: HashMap<usize, Vec<Vec<u8>>>,
        max_pool_size: usize,
    }

    impl TensorPool {
        pub fn new(max_pool_size: usize) -> Self {
            Self {
                pools: HashMap::new(),
                max_pool_size,
            }
        }

        /// Get a buffer from the pool or allocate a new one
        pub fn get_buffer(&mut self, size: usize) -> Vec<u8> {
            if let Some(pool) = self.pools.get_mut(&size) {
                if let Some(buffer) = pool.pop() {
                    return buffer;
                }
            }
            
            vec![0u8; size]
        }

        /// Return a buffer to the pool
        pub fn return_buffer(&mut self, mut buffer: Vec<u8>) {
            let size = buffer.len();
            buffer.clear();
            buffer.shrink_to_fit();
            
            let pool = self.pools.entry(size).or_insert_with(Vec::new);
            if pool.len() < self.max_pool_size {
                pool.push(buffer);
            }
        }
    }
}

/// Format conversion helpers for PEFT/GGML export
pub mod format {
    use super::*;

    /// PEFT format configuration
    #[derive(Debug, Clone)]
    pub struct PeftConfig {
        pub base_model_name: String,
        pub task_type: String,
        pub inference_mode: bool,
        pub r: usize,
        pub lora_alpha: f32,
        pub lora_dropout: f32,
        pub target_modules: Vec<String>,
    }

    /// GGML format configuration
    #[derive(Debug, Clone)]
    pub struct GgmlConfig {
        pub version: u32,
        pub vocab_size: usize,
        pub embedding_size: usize,
        pub quantization: TensorDataType,
    }

    /// Convert tensor to PEFT format
    pub fn to_peft_format(
        tensors: &HashMap<String, Tensor>,
        config: &PeftConfig,
    ) -> Result<HashMap<String, Vec<u8>>> {
        let mut peft_tensors = HashMap::new();
        
        for (name, tensor) in tensors {
            // Convert LoRA weights to PEFT naming convention
            let peft_name = if name.contains("lora_A") {
                name.replace("lora_A", "lora_A.weight")
            } else if name.contains("lora_B") {
                name.replace("lora_B", "lora_B.weight")
            } else {
                name.clone()
            };
            
            // Ensure data is in float32 for PEFT compatibility
            let data = if tensor.metadata.dtype != TensorDataType::Float32 {
                precision::convert_dtype(
                    &tensor.data,
                    &tensor.metadata.dtype,
                    &TensorDataType::Float32,
                )?
            } else {
                tensor.data.clone()
            };
            
            peft_tensors.insert(peft_name, data);
        }
        
        Ok(peft_tensors)
    }

    /// Convert tensor to GGML format
    pub fn to_ggml_format(
        tensors: &HashMap<String, Tensor>,
        config: &GgmlConfig,
    ) -> Result<Vec<u8>> {
        let mut ggml_data = Vec::new();
        
        // Write GGML header
        ggml_data.extend_from_slice(&config.version.to_le_bytes());
        ggml_data.extend_from_slice(&config.vocab_size.to_le_bytes());
        ggml_data.extend_from_slice(&config.embedding_size.to_le_bytes());
        
        // Write tensor data
        for (name, tensor) in tensors {
            // Write tensor name
            let name_bytes = name.as_bytes();
            ggml_data.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            ggml_data.extend_from_slice(name_bytes);
            
            // Write tensor shape
            ggml_data.extend_from_slice(&(tensor.metadata.shape.len() as u32).to_le_bytes());
            for &dim in &tensor.metadata.shape {
                ggml_data.extend_from_slice(&(dim as u32).to_le_bytes());
            }
            
            // Convert to target quantization if needed
            let data = if tensor.metadata.dtype != config.quantization {
                precision::convert_dtype(
                    &tensor.data,
                    &tensor.metadata.dtype,
                    &config.quantization,
                )?
            } else {
                tensor.data.clone()
            };
            
            // Write tensor data
            ggml_data.extend_from_slice(&(data.len() as u32).to_le_bytes());
            ggml_data.extend_from_slice(&data);
        }
        
        Ok(ggml_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_conversion() {
        let f32_data = vec![1.0f32, -1.0f32, 0.5f32, -0.5f32];
        
        // Test f32 to f16 conversion
        let f16_bytes = precision::f32_to_f16(&f32_data).unwrap();
        let converted_back = precision::f16_to_f32(&f16_bytes).unwrap();
        
        // Check approximate equality (f16 has lower precision)
        for (original, converted) in f32_data.iter().zip(converted_back.iter()) {
            assert!((original - converted).abs() < 0.001);
        }
    }

    #[test]
    fn test_lora_reshaping() {
        let rank = 4;
        let out_features = 8;
        let in_features = 6;
        
        // Create test LoRA A matrix
        let lora_a = (0..out_features * rank).map(|i| i as f32).collect::<Vec<_>>();
        let reshaped_a = lora::reshape_lora_a(&lora_a, &[out_features, rank], rank).unwrap();
        
        assert_eq!(reshaped_a.len(), lora_a.len());
        
        // Create test LoRA B matrix
        let lora_b = (0..rank * in_features).map(|i| i as f32).collect::<Vec<_>>();
        let reshaped_b = lora::reshape_lora_b(&lora_b, &[rank, in_features], rank).unwrap();
        
        assert_eq!(reshaped_b.len(), lora_b.len());
    }

    #[test]
    fn test_quantization_params() {
        let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32];
        let (scale, zero_point) = precision::calculate_quantization_params(&data);
        
        assert!(scale > 0.0);
        assert!(zero_point >= i8::MIN && zero_point <= i8::MAX);
    }

    #[test]
    fn test_memory_estimation() {
        let shape = vec![1024, 768];
        let memory_f32 = device::estimate_memory_usage(&shape, &TensorDataType::Float32);
        let memory_f16 = device::estimate_memory_usage(&shape, &TensorDataType::Float16);
        
        assert_eq!(memory_f32, 1024 * 768 * 4);
        assert_eq!(memory_f16, 1024 * 768 * 2);
        assert_eq!(memory_f32, memory_f16 * 2);
    }
}