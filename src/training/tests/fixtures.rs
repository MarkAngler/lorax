//! Test fixtures for setting up common test scenarios

use anyhow::Result;
use std::path::PathBuf;
use tempfile::TempDir;
use std::sync::Arc;
use parking_lot::RwLock;
use candle_core::Device;
use candle_nn::VarMap;

use crate::training::{
    TrainingConfig, T2LTrainer, TrainingState,
    CheckpointManager, MetricsTracker,
};
use crate::training::data::{DataLoader, Dataset};
use crate::training::optimizers::{OptimizerState, SchedulerState};

use super::test_utils::{create_test_config, create_test_device, create_test_model};
use super::mock_data::{create_mock_dataset, DatasetType};

/// Base test fixture trait
pub trait TestFixture {
    fn setup(&mut self) -> Result<()>;
    fn teardown(&mut self) -> Result<()>;
}

/// Training test fixture with all necessary components
pub struct TrainingTestFixture {
    pub config: TrainingConfig,
    pub device: Device,
    pub model: VarMap,
    pub train_loader: Arc<DataLoader<dyn Dataset>>,
    pub val_loader: Option<Arc<DataLoader<dyn Dataset>>>,
    pub temp_dir: TempDir,
    pub checkpoint_dir: PathBuf,
}

impl TrainingTestFixture {
    pub fn new() -> Result<Self> {
        let config = create_test_config();
        let device = create_test_device();
        let model = create_test_model(&device)?;
        
        // Create temporary directories
        let temp_dir = tempfile::tempdir()?;
        let checkpoint_dir = temp_dir.path().join("checkpoints");
        std::fs::create_dir_all(&checkpoint_dir)?;
        
        // Create mock data loaders
        let train_dataset = create_mock_dataset(100, DatasetType::Reconstruction);
        let val_dataset = create_mock_dataset(20, DatasetType::Reconstruction);
        
        // TODO: Create data loaders once DataLoader API is stable
        // For now, we'll use placeholders
        let train_loader: Arc<DataLoader<dyn Dataset>> = todo!("DataLoader not yet implemented");
        let val_loader: Option<Arc<DataLoader<dyn Dataset>>> = None;
        
        Ok(Self {
            config,
            device,
            model,
            train_loader,
            val_loader,
            temp_dir,
            checkpoint_dir,
        })
    }
    
    pub fn create_trainer(&self) -> Result<T2LTrainer> {
        // This would create the actual trainer once it's implemented
        // For now, we'll return an error
        anyhow::bail!("Trainer creation not yet implemented")
    }
}

impl TestFixture for TrainingTestFixture {
    fn setup(&mut self) -> Result<()> {
        // Additional setup if needed
        Ok(())
    }
    
    fn teardown(&mut self) -> Result<()> {
        // Cleanup is automatic with TempDir
        Ok(())
    }
}

/// Data loading test fixture
pub struct DataTestFixture {
    pub device: Device,
    pub temp_dir: TempDir,
    pub data_dir: PathBuf,
    pub reconstruction_dataset: Arc<dyn crate::training::data::Dataset>,
    pub supervised_dataset: Arc<dyn crate::training::data::Dataset>,
}

impl DataTestFixture {
    pub fn new() -> Result<Self> {
        let device = create_test_device();
        let temp_dir = tempfile::tempdir()?;
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir)?;
        
        let reconstruction_dataset = create_mock_dataset(50, DatasetType::Reconstruction);
        let supervised_dataset = create_mock_dataset(50, DatasetType::Supervised);
        
        Ok(Self {
            device,
            temp_dir,
            data_dir,
            reconstruction_dataset,
            supervised_dataset,
        })
    }
    
    pub fn create_hdf5_dataset(&self, name: &str, num_samples: usize) -> Result<PathBuf> {
        // Create a mock HDF5 file for testing
        let file_path = self.data_dir.join(format!("{}.h5", name));
        
        use hdf5::File;
        let file = File::create(&file_path)?;
        
        // Create groups and datasets
        let group = file.create_group("data")?;
        
        // Add task descriptions
        let task_descriptions: Vec<String> = (0..num_samples)
            .map(|i| format!("Task {}", i))
            .collect();
        let task_desc_dataset = group.create_dataset::<hdf5::types::VarLenUnicode>("task_descriptions")?;
        task_desc_dataset.write(&task_descriptions)?;
        
        // Add LoRA parameters
        let lora_group = group.create_group("lora_params")?;
        for i in 0..num_samples {
            let sample_group = lora_group.create_group(&format!("sample_{}", i))?;
            
            // Create mock LoRA A and B matrices
            let a_data: Vec<f32> = vec![0.1; 64 * 4];
            let b_data: Vec<f32> = vec![0.1; 4 * 64];
            
            sample_group.new_dataset::<f32>()
                .shape([64, 4])
                .create("layer_0_A")?
                .write(&a_data)?;
            
            sample_group.new_dataset::<f32>()
                .shape([4, 64])
                .create("layer_0_B")?
                .write(&b_data)?;
        }
        
        // Add metadata
        file.new_attr::<u32>()
            .create("num_samples")?
            .write_scalar(&(num_samples as u32))?;
        
        Ok(file_path)
    }
}

impl TestFixture for DataTestFixture {
    fn setup(&mut self) -> Result<()> {
        Ok(())
    }
    
    fn teardown(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Checkpoint test fixture
pub struct CheckpointTestFixture {
    pub device: Device,
    pub temp_dir: TempDir,
    pub checkpoint_dir: PathBuf,
    pub model: VarMap,
    pub config: TrainingConfig,
}

impl CheckpointTestFixture {
    pub fn new() -> Result<Self> {
        let device = create_test_device();
        let temp_dir = tempfile::tempdir()?;
        let checkpoint_dir = temp_dir.path().join("checkpoints");
        std::fs::create_dir_all(&checkpoint_dir)?;
        
        let model = create_test_model(&device)?;
        let mut config = create_test_config();
        config.checkpointing.output_dir = checkpoint_dir.to_string_lossy().to_string();
        
        Ok(Self {
            device,
            temp_dir,
            checkpoint_dir,
            model,
            config,
        })
    }
    
    pub fn create_checkpoint_manager(&self) -> Result<CheckpointManager> {
        CheckpointManager::new(
            &self.checkpoint_dir,
            self.config.checkpointing.save_total_limit,
            self.config.checkpointing.checkpoint_compression,
        )
    }
    
    pub fn create_test_state(&self) -> TrainingState {
        TrainingState {
            epoch: 1,
            step: 100,
            global_step: 100,
            best_score: Some(0.95),
            steps_since_best: 10,
            start_time: chrono::Utc::now(),
            last_eval_time: Some(chrono::Utc::now()),
            status: crate::training::TrainingStatus::Running,
            memory_usage: Default::default(),
        }
    }
}

impl TestFixture for CheckpointTestFixture {
    fn setup(&mut self) -> Result<()> {
        Ok(())
    }
    
    fn teardown(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Create a test fixture with sensible defaults
pub fn create_default_fixture() -> Result<TrainingTestFixture> {
    TrainingTestFixture::new()
}

/// Macro for running tests with fixtures
#[macro_export]
macro_rules! with_fixture {
    ($fixture_type:ty, $test_fn:expr) => {{
        let mut fixture = <$fixture_type>::new()?;
        fixture.setup()?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            $test_fn(&mut fixture)
        }));
        fixture.teardown()?;
        result.unwrap()
    }};
}