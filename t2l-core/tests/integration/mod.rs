//! Integration tests for T2L Core functionality
//!
//! This test suite validates end-to-end workflows, command-line interface,
//! export format compatibility, and performance characteristics.

mod fixtures;
mod workflow_tests;
mod cli_tests;
mod export_tests;
mod architecture_tests;
mod error_tests;
mod performance_tests;

use t2l_core::Result;

/// Common test initialization
pub fn init_test_logging() {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_env_filter("t2l_core=debug")
        .try_init();
}

/// Utility to run a test with proper initialization
pub async fn run_test<F, T>(test_fn: F) -> Result<T>
where
    F: FnOnce() -> T,
{
    init_test_logging();
    Ok(test_fn())
}