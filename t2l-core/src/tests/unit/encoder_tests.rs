//! Unit tests for T2L Task Encoder components

use crate::encoder::{TaskEncoder, EncoderConfig, EncoderType};
use crate::embedding::{TaskEmbedding, EmbeddingCache};
use candle_core::{Device, Tensor, DType};
use tokenizers::Tokenizer;
use rstest::*;
use proptest::prelude::*;
use std::sync::Arc;
use parking_lot::RwLock;

#[fixture]
fn device() -> Device {
    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
}

#[fixture]
fn encoder_config_transformer() -> EncoderConfig {
    EncoderConfig {
        encoder_type: EncoderType::Transformer,
        model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        embedding_dim: 384,
        max_seq_length: 512,
        pooling_strategy: "mean".to_string(),
        normalize_embeddings: true,
        cache_size: 1000,
        use_fp16: false,
    }
}

#[fixture]
fn encoder_config_bi_encoder() -> EncoderConfig {
    EncoderConfig {
        encoder_type: EncoderType::BiEncoder,
        model_name: "bert-base-uncased".to_string(),
        embedding_dim: 768,
        max_seq_length: 256,
        pooling_strategy: "cls".to_string(),
        normalize_embeddings: true,
        cache_size: 500,
        use_fp16: false,
    }
}

#[fixture]
fn encoder_config_contrastive() -> EncoderConfig {
    EncoderConfig {
        encoder_type: EncoderType::Contrastive,
        model_name: "BAAI/bge-small-en-v1.5".to_string(),
        embedding_dim: 384,
        max_seq_length: 512,
        pooling_strategy: "mean".to_string(),
        normalize_embeddings: true,
        cache_size: 2000,
        use_fp16: true,
    }
}

#[rstest]
#[case(encoder_config_transformer())]
#[case(encoder_config_bi_encoder())]
#[case(encoder_config_contrastive())]
fn test_encoder_initialization(
    #[case] config: EncoderConfig,
    device: Device,
) {
    let encoder = TaskEncoder::new(&config, &device).unwrap();
    
    assert_eq!(encoder.config().encoder_type, config.encoder_type);
    assert_eq!(encoder.config().embedding_dim, config.embedding_dim);
    assert_eq!(encoder.config().max_seq_length, config.max_seq_length);
}

#[rstest]
fn test_single_task_encoding(
    encoder_config_transformer: EncoderConfig,
    device: Device,
) {
    let encoder = TaskEncoder::new(&encoder_config_transformer, &device).unwrap();
    let task_description = "Summarize the given text in one paragraph";
    
    let embedding = encoder.encode(task_description).unwrap();
    
    // Check embedding shape
    assert_eq!(embedding.shape().dims(), &[encoder_config_transformer.embedding_dim]);
    
    // Check normalization if enabled
    if encoder_config_transformer.normalize_embeddings {
        let norm = embedding.sqr().unwrap().sum_all().unwrap().sqrt().unwrap();
        let norm_scalar = norm.to_scalar::<f32>().unwrap();
        assert_approx_eq!(norm_scalar, 1.0, 1e-4);
    }
}

#[rstest]
fn test_batch_encoding(
    encoder_config_bi_encoder: EncoderConfig,
    device: Device,
) {
    let encoder = TaskEncoder::new(&encoder_config_bi_encoder, &device).unwrap();
    let task_descriptions = vec![
        "Translate English to French",
        "Answer questions based on context",
        "Generate creative stories",
        "Solve mathematical problems",
    ];
    
    let embeddings = encoder.encode_batch(&task_descriptions).unwrap();
    
    // Check batch dimensions
    assert_eq!(
        embeddings.shape().dims(),
        &[task_descriptions.len(), encoder_config_bi_encoder.embedding_dim]
    );
    
    // Verify all embeddings are different
    for i in 0..task_descriptions.len() {
        for j in (i + 1)..task_descriptions.len() {
            let emb_i = embeddings.i(i).unwrap();
            let emb_j = embeddings.i(j).unwrap();
            let diff = (&emb_i - &emb_j).unwrap().abs().unwrap().sum_all().unwrap();
            let diff_scalar = diff.to_scalar::<f32>().unwrap();
            assert!(diff_scalar > 0.1, "Embeddings {} and {} are too similar", i, j);
        }
    }
}

#[rstest]
fn test_encoder_caching(
    encoder_config_transformer: EncoderConfig,
    device: Device,
) {
    let encoder = TaskEncoder::new(&encoder_config_transformer, &device).unwrap();
    let task = "Classify sentiment of reviews";
    
    // First encoding (cache miss)
    let start = std::time::Instant::now();
    let embedding1 = encoder.encode(task).unwrap();
    let first_duration = start.elapsed();
    
    // Second encoding (cache hit)
    let start = std::time::Instant::now();
    let embedding2 = encoder.encode(task).unwrap();
    let second_duration = start.elapsed();
    
    // Cache hit should be much faster
    assert!(
        second_duration < first_duration / 10,
        "Cache hit not significantly faster: {:?} vs {:?}",
        second_duration, first_duration
    );
    
    // Embeddings should be identical
    let diff = (&embedding1 - &embedding2).unwrap().abs().unwrap().max_all().unwrap();
    assert_eq!(diff.to_scalar::<f32>().unwrap(), 0.0);
}

#[rstest]
fn test_long_input_truncation(
    encoder_config_transformer: EncoderConfig,
    device: Device,
) {
    let encoder = TaskEncoder::new(&encoder_config_transformer, &device).unwrap();
    
    // Create a very long task description
    let long_task = "Analyze the following document and ".repeat(100);
    
    // Should not panic, should truncate
    let embedding = encoder.encode(&long_task).unwrap();
    assert_eq!(embedding.shape().dims(), &[encoder_config_transformer.embedding_dim]);
}

#[rstest]
fn test_special_characters_handling(
    encoder_config_contrastive: EncoderConfig,
    device: Device,
) {
    let encoder = TaskEncoder::new(&encoder_config_contrastive, &device).unwrap();
    
    let special_tasks = vec![
        "Handle UTF-8: résum√© naïve café",
        "Emojis: Translate 🌍 to 🗺️",
        "Math symbols: Solve ∑(x²) where x ∈ ℝ",
        "Code: Parse `fn main() { println!(\"Hello\"); }`",
    ];
    
    for task in special_tasks {
        // Should not panic
        let embedding = encoder.encode(task).unwrap();
        assert_eq!(embedding.shape().dims(), &[encoder_config_contrastive.embedding_dim]);
    }
}

#[rstest]
fn test_semantic_similarity(
    encoder_config_transformer: EncoderConfig,
    device: Device,
) {
    let encoder = TaskEncoder::new(&encoder_config_transformer, &device).unwrap();
    
    // Similar tasks
    let task1 = "Translate text from English to Spanish";
    let task2 = "Convert English sentences to Spanish language";
    
    // Different task
    let task3 = "Generate Python code for data analysis";
    
    let emb1 = encoder.encode(task1).unwrap();
    let emb2 = encoder.encode(task2).unwrap();
    let emb3 = encoder.encode(task3).unwrap();
    
    // Compute cosine similarities
    let sim_12 = cosine_similarity(&emb1, &emb2).unwrap();
    let sim_13 = cosine_similarity(&emb1, &emb3).unwrap();
    let sim_23 = cosine_similarity(&emb2, &emb3).unwrap();
    
    // Similar tasks should have higher similarity
    assert!(sim_12 > sim_13, "Similar tasks should have higher similarity");
    assert!(sim_12 > sim_23, "Similar tasks should have higher similarity");
}

#[rstest]
fn test_fp16_encoding(
    mut encoder_config_contrastive: EncoderConfig,
    device: Device,
) {
    encoder_config_contrastive.use_fp16 = true;
    let encoder = TaskEncoder::new(&encoder_config_contrastive, &device).unwrap();
    
    let task = "Process data using half precision";
    let embedding = encoder.encode(task).unwrap();
    
    // Check that dtype is F16 if supported
    if device.supports_fp16() {
        assert_eq!(embedding.dtype(), DType::F16);
    }
    
    // Verify embedding is still valid
    assert_eq!(embedding.shape().dims(), &[encoder_config_contrastive.embedding_dim]);
}

proptest! {
    #[test]
    fn test_encoder_consistency(
        seed: u64,
        batch_size in 1usize..8,
    ) {
        let device = Device::Cpu;
        let config = EncoderConfig {
            encoder_type: EncoderType::Transformer,
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            embedding_dim: 384,
            max_seq_length: 128,
            pooling_strategy: "mean".to_string(),
            normalize_embeddings: true,
            cache_size: 100,
            use_fp16: false,
        };
        
        let encoder = TaskEncoder::new(&config, &device).unwrap();
        
        // Generate random tasks
        let mut rng = proptest::test_runner::TestRng::from_seed(
            proptest::test_runner::RngAlgorithm::ChaCha,
            &seed.to_le_bytes()
        );
        
        let tasks: Vec<String> = (0..batch_size)
            .map(|i| format!("Task {}: Process data with seed {}", i, seed))
            .collect();
        
        // Encode twice
        let embeddings1 = encoder.encode_batch(&tasks).unwrap();
        let embeddings2 = encoder.encode_batch(&tasks).unwrap();
        
        // Should be deterministic
        let diff = (&embeddings1 - &embeddings2).unwrap()
            .abs().unwrap()
            .max_all().unwrap()
            .to_scalar::<f32>().unwrap();
        
        prop_assert!(diff < 1e-5, "Embeddings not consistent");
    }
}

#[test]
fn test_cache_eviction() {
    let device = Device::Cpu;
    let config = EncoderConfig {
        encoder_type: EncoderType::Transformer,
        model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        embedding_dim: 384,
        max_seq_length: 512,
        pooling_strategy: "mean".to_string(),
        normalize_embeddings: true,
        cache_size: 10, // Small cache for testing
        use_fp16: false,
    };
    
    let encoder = TaskEncoder::new(&config, &device).unwrap();
    
    // Fill cache beyond capacity
    for i in 0..20 {
        let task = format!("Task number {}", i);
        encoder.encode(&task).unwrap();
    }
    
    // Cache should not exceed configured size
    assert!(encoder.cache_size() <= config.cache_size);
}

#[rstest]
fn test_pooling_strategies(
    device: Device,
) {
    let pooling_strategies = vec!["mean", "max", "cls", "last"];
    
    for strategy in pooling_strategies {
        let config = EncoderConfig {
            encoder_type: EncoderType::Transformer,
            model_name: "bert-base-uncased".to_string(),
            embedding_dim: 768,
            max_seq_length: 128,
            pooling_strategy: strategy.to_string(),
            normalize_embeddings: false,
            cache_size: 100,
            use_fp16: false,
        };
        
        let encoder = TaskEncoder::new(&config, &device).unwrap();
        let embedding = encoder.encode("Test pooling strategy").unwrap();
        
        assert_eq!(
            embedding.shape().dims(),
            &[config.embedding_dim],
            "Failed for pooling strategy: {}",
            strategy
        );
    }
}

#[test]
fn test_thread_safety() {
    use std::thread;
    use std::sync::Arc;
    
    let device = Device::Cpu;
    let config = EncoderConfig {
        encoder_type: EncoderType::Transformer,
        model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        embedding_dim: 384,
        max_seq_length: 512,
        pooling_strategy: "mean".to_string(),
        normalize_embeddings: true,
        cache_size: 1000,
        use_fp16: false,
    };
    
    let encoder = Arc::new(TaskEncoder::new(&config, &device).unwrap());
    let mut handles = vec![];
    
    // Spawn multiple threads
    for i in 0..4 {
        let encoder_clone = Arc::clone(&encoder);
        let handle = thread::spawn(move || {
            for j in 0..10 {
                let task = format!("Thread {} task {}", i, j);
                encoder_clone.encode(&task).unwrap();
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Should complete without panics or deadlocks
}

// Helper function for cosine similarity
fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32, candle_core::Error> {
    let dot_product = (a * b)?.sum_all()?;
    let norm_a = a.sqr()?.sum_all()?.sqrt()?;
    let norm_b = b.sqr()?.sum_all()?.sqrt()?;
    let similarity = dot_product / (norm_a * norm_b)?;
    similarity.to_scalar::<f32>()
}