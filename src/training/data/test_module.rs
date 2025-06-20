//! Test module to verify training data infrastructure compiles independently

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;
    use tempfile::TempDir;
    use std::fs;
    
    #[test]
    fn test_data_structures() {
        // Test creating LoraDatasetConfig
        let mut layer_dims = HashMap::new();
        layer_dims.insert("layer1".to_string(), (512, 512));
        
        let lora_config = LoraDatasetConfig {
            layer_names: vec!["layer1".to_string()],
            rank: 16,
            alpha: 32.0,
            layer_dims,
        };
        
        assert_eq!(lora_config.rank, 16);
        assert_eq!(lora_config.alpha, 32.0);
        
        // Test TaskInfo
        let task_info = TaskInfo {
            task_type: "classification".to_string(),
            num_classes: Some(10),
            vocab_size: Some(50000),
            max_seq_len: Some(512),
        };
        
        assert_eq!(task_info.task_type, "classification");
        assert_eq!(task_info.num_classes, Some(10));
        
        // Test DatasetMetadata
        let metadata = DatasetMetadata {
            name: "test_dataset".to_string(),
            num_samples: 100,
            lora_config: Some(lora_config),
            task_info: Some(task_info),
        };
        
        assert_eq!(metadata.name, "test_dataset");
        assert_eq!(metadata.num_samples, 100);
        assert!(metadata.lora_config.is_some());
        assert!(metadata.task_info.is_some());
    }
    
    #[test]
    fn test_data_sample_creation() {
        let device = Device::Cpu;
        
        // Test SampleData::Supervised
        let supervised_data = SampleData::Supervised {
            input_text: "Test input".to_string(),
            target_text: Some("Test target".to_string()),
            labels: None,
        };
        
        let sample = DataSample {
            id: "test_sample".to_string(),
            task_description: "Test task".to_string(),
            task_embeddings: None,
            data: supervised_data,
        };
        
        assert_eq!(sample.id, "test_sample");
        assert_eq!(sample.task_description, "Test task");
        
        match &sample.data {
            SampleData::Supervised { input_text, .. } => {
                assert_eq!(input_text, "Test input");
            },
            _ => panic!("Expected supervised data"),
        }
    }
    
    #[test]
    fn test_data_error() {
        let error = DataError::InvalidIndex {
            index: 5,
            dataset_size: 3,
        };
        
        assert!(error.to_string().contains("Invalid sample index"));
        
        let error2 = DataError::DatasetNotFound {
            path: "/nonexistent/path".to_string(),
        };
        
        assert!(error2.to_string().contains("Dataset not found"));
    }
    
    #[test] 
    fn test_dataloader_config() {
        let config = DataLoaderConfig {
            batch_size: 32,
            shuffle: true,
            num_workers: 4,
            buffer_size: 100,
            drop_last: false,
            prefetch_factor: 2,
        };
        
        assert_eq!(config.batch_size, 32);
        assert!(config.shuffle);
        assert_eq!(config.num_workers, 4);
        
        // Test default
        let default_config = DataLoaderConfig::default();
        assert_eq!(default_config.batch_size, 32);
        assert!(default_config.shuffle);
    }
}