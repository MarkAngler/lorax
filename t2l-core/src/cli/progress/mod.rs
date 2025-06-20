use crate::cli::error::CliResult;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use std::sync::Arc;
use std::time::Duration;

pub struct ProgressReporter {
    bar: ProgressBar,
    multi: Option<Arc<MultiProgress>>,
}

impl ProgressReporter {
    /// Create a new progress reporter with indeterminate progress
    pub fn new(message: &str) -> CliResult<Self> {
        let bar = ProgressBar::new_spinner();
        bar.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        bar.set_message(message.to_string());
        bar.enable_steady_tick(Duration::from_millis(100));

        Ok(Self {
            bar,
            multi: None,
        })
    }

    /// Create a new progress reporter with a known total
    pub fn new_with_total(message: &str, total: u64) -> CliResult<Self> {
        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.cyan} {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("█▇▆▅▄▃▂▁  "),
        );
        bar.set_message(message.to_string());

        Ok(Self {
            bar,
            multi: None,
        })
    }

    /// Create a multi-progress reporter for managing multiple concurrent progress bars
    pub fn new_multi() -> Arc<MultiProgress> {
        Arc::new(MultiProgress::new())
    }

    /// Add this progress reporter to a multi-progress instance
    pub fn with_multi(mut self, multi: Arc<MultiProgress>) -> Self {
        let bar = multi.add(self.bar.clone());
        self.bar = bar;
        self.multi = Some(multi);
        self
    }

    /// Update the progress message
    pub fn set_message(&self, message: &str) {
        self.bar.set_message(message.to_string());
    }

    /// Advance progress by 1 and optionally update message
    pub fn advance(&self, message: &str) {
        if message.is_empty() {
            self.bar.inc(1);
        } else {
            self.bar.set_message(message.to_string());
            self.bar.inc(1);
        }
    }

    /// Advance progress by a specific amount
    pub fn advance_by(&self, amount: u64) {
        self.bar.inc(amount);
    }

    /// Set absolute progress position
    pub fn set_position(&self, position: u64) {
        self.bar.set_position(position);
    }

    /// Set the total length (useful for dynamic totals)
    pub fn set_length(&self, length: u64) {
        self.bar.set_length(length);
    }

    /// Finish the progress bar with a completion message
    pub fn finish(&self, message: &str) {
        self.bar.finish_with_message(message.to_string());
    }

    /// Finish the progress bar and clear it
    pub fn finish_and_clear(&self) {
        self.bar.finish_and_clear();
    }

    /// Get the underlying progress bar for advanced operations
    pub fn bar(&self) -> &ProgressBar {
        &self.bar
    }

    /// Create a child progress bar for nested operations
    pub fn create_child(&self, message: &str, total: Option<u64>) -> CliResult<Self> {
        if let Some(multi) = &self.multi {
            let child_bar = if let Some(total) = total {
                ProgressBar::new(total)
            } else {
                ProgressBar::new_spinner()
            };

            child_bar.set_style(
                ProgressStyle::default_spinner()
                    .template("  {spinner:.yellow} {msg}")
                    .unwrap()
                    .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
            );
            child_bar.set_message(message.to_string());

            let child_bar = multi.add(child_bar);
            child_bar.enable_steady_tick(Duration::from_millis(120));

            Ok(Self {
                bar: child_bar,
                multi: Some(multi.clone()),
            })
        } else {
            // If no multi-progress, create a standalone child
            Self::new(message)
        }
    }
}

impl Clone for ProgressReporter {
    fn clone(&self) -> Self {
        Self {
            bar: self.bar.clone(),
            multi: self.multi.clone(),
        }
    }
}

/// Progress reporter for training operations
pub struct TrainingProgress {
    epoch_bar: ProgressBar,
    batch_bar: ProgressBar,
    multi: Arc<MultiProgress>,
}

impl TrainingProgress {
    pub fn new(total_epochs: u64, batches_per_epoch: u64) -> Self {
        let multi = Arc::new(MultiProgress::new());

        let epoch_bar = ProgressBar::new(total_epochs);
        epoch_bar.set_style(
            ProgressStyle::default_bar()
                .template("Epochs: [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▇▆▅▄▃▂▁  "),
        );

        let batch_bar = ProgressBar::new(batches_per_epoch);
        batch_bar.set_style(
            ProgressStyle::default_bar()
                .template("Batches: [{bar:40.green/yellow}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▇▆▅▄▃▂▁  "),
        );

        let epoch_bar = multi.add(epoch_bar);
        let batch_bar = multi.add(batch_bar);

        Self {
            epoch_bar,
            batch_bar,
            multi,
        }
    }

    pub fn start_epoch(&self, epoch: u64, message: &str) {
        self.epoch_bar.set_position(epoch);
        self.epoch_bar.set_message(message.to_string());
        self.batch_bar.reset();
    }

    pub fn advance_batch(&self, loss: f64, lr: f64) {
        self.batch_bar.inc(1);
        self.batch_bar.set_message(format!("loss: {:.4}, lr: {:.2e}", loss, lr));
    }

    pub fn finish_epoch(&self, message: &str) {
        self.batch_bar.finish_and_clear();
        self.epoch_bar.set_message(message.to_string());
        self.epoch_bar.inc(1);
    }

    pub fn finish(&self, message: &str) {
        self.batch_bar.finish_and_clear();
        self.epoch_bar.finish_with_message(message.to_string());
    }

    pub fn multi(&self) -> Arc<MultiProgress> {
        self.multi.clone()
    }
}

/// Progress reporter for evaluation operations
pub struct EvaluationProgress {
    benchmark_bar: ProgressBar,
    sample_bar: ProgressBar,
    multi: Arc<MultiProgress>,
}

impl EvaluationProgress {
    pub fn new(total_benchmarks: u64) -> Self {
        let multi = Arc::new(MultiProgress::new());

        let benchmark_bar = ProgressBar::new(total_benchmarks);
        benchmark_bar.set_style(
            ProgressStyle::default_bar()
                .template("Benchmarks: [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▇▆▅▄▃▂▁  "),
        );

        let sample_bar = ProgressBar::new(0);
        sample_bar.set_style(
            ProgressStyle::default_bar()
                .template("Samples: [{bar:40.green/yellow}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▇▆▅▄▃▂▁  "),
        );

        let benchmark_bar = multi.add(benchmark_bar);
        let sample_bar = multi.add(sample_bar);

        Self {
            benchmark_bar,
            sample_bar,
            multi,
        }
    }

    pub fn start_benchmark(&self, benchmark: &str, total_samples: u64) {
        self.benchmark_bar.set_message(format!("Evaluating {}", benchmark));
        self.sample_bar.reset();
        self.sample_bar.set_length(total_samples);
    }

    pub fn advance_sample(&self, accuracy: f64) {
        self.sample_bar.inc(1);
        self.sample_bar.set_message(format!("acc: {:.3}", accuracy));
    }

    pub fn finish_benchmark(&self, benchmark: &str, accuracy: f64) {
        self.sample_bar.finish_and_clear();
        self.benchmark_bar.set_message(format!("{}: {:.1}%", benchmark, accuracy * 100.0));
        self.benchmark_bar.inc(1);
    }

    pub fn finish(&self, overall_accuracy: f64) {
        self.sample_bar.finish_and_clear();
        self.benchmark_bar.finish_with_message(format!("Overall: {:.1}%", overall_accuracy * 100.0));
    }

    pub fn multi(&self) -> Arc<MultiProgress> {
        self.multi.clone()
    }
}

/// Progress reporter for batch generation operations
pub struct BatchProgress {
    task_bar: ProgressBar,
    multi: Arc<MultiProgress>,
}

impl BatchProgress {
    pub fn new(total_tasks: u64) -> Self {
        let multi = Arc::new(MultiProgress::new());

        let task_bar = ProgressBar::new(total_tasks);
        task_bar.set_style(
            ProgressStyle::default_bar()
                .template("Tasks: [{bar:40.cyan/blue}] {pos}/{len} {msg} ({eta})")
                .unwrap()
                .progress_chars("█▇▆▅▄▃▂▁  "),
        );

        let task_bar = multi.add(task_bar);

        Self {
            task_bar,
            multi,
        }
    }

    pub fn advance_task(&self, task_name: &str, generation_time_ms: u64) {
        self.task_bar.inc(1);
        self.task_bar.set_message(format!("{} ({}ms)", task_name, generation_time_ms));
    }

    pub fn finish(&self, total_time_ms: u64, success_rate: f64) {
        self.task_bar.finish_with_message(
            format!("Completed in {}ms ({}% success)", total_time_ms, success_rate * 100.0)
        );
    }

    pub fn multi(&self) -> Arc<MultiProgress> {
        self.multi.clone()
    }
}