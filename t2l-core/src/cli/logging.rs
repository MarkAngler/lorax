use anyhow::Result;
use colored::*;
use std::io::{self, IsTerminal};
use tracing::{Level, Metadata};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer,
};

pub fn init_logging(verbosity: u8, quiet: bool, json_output: bool) -> Result<()> {
    if quiet {
        // Only show errors
        std::env::set_var("RUST_LOG", "error");
    } else {
        // Set log level based on verbosity
        let log_level = match verbosity {
            0 => "t2l=info,warn",
            1 => "t2l=debug,info",
            2 => "t2l=trace,debug",
            _ => "trace",
        };
        
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", log_level);
        }
    }
    
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    
    let is_terminal = io::stdout().is_terminal();
    
    if json_output {
        // JSON output for machine parsing
        let json_layer = fmt::layer()
            .json()
            .with_current_span(true)
            .with_span_list(true)
            .with_filter(env_filter);
            
        tracing_subscriber::registry()
            .with(json_layer)
            .init();
    } else if is_terminal && !quiet {
        // Pretty terminal output
        let fmt_layer = fmt::layer()
            .with_target(false)
            .with_thread_ids(false)
            .with_thread_names(false)
            .with_ansi(true)
            .with_span_events(FmtSpan::CLOSE)
            .event_format(ColoredFormatter)
            .with_filter(env_filter);
            
        tracing_subscriber::registry()
            .with(fmt_layer)
            .init();
    } else {
        // Plain output for non-terminal or quiet mode
        let fmt_layer = fmt::layer()
            .with_target(false)
            .with_ansi(false)
            .without_time()
            .with_filter(env_filter);
            
        tracing_subscriber::registry()
            .with(fmt_layer)
            .init();
    }
    
    Ok(())
}

/// Custom formatter for colored terminal output
struct ColoredFormatter;

impl<S, N> fmt::FormatEvent<S, N> for ColoredFormatter
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'a> fmt::FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &fmt::FmtContext<'_, S, N>,
        mut writer: fmt::format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        let metadata = event.metadata();
        let level = metadata.level();
        
        // Format timestamp
        let now = chrono::Local::now();
        write!(
            writer,
            "{} ",
            now.format("%H:%M:%S").to_string().dimmed()
        )?;
        
        // Format level with colors
        let level_str = match *level {
            Level::ERROR => "ERROR".red().bold(),
            Level::WARN => "WARN".yellow().bold(),
            Level::INFO => "INFO".green().bold(),
            Level::DEBUG => "DEBUG".blue().bold(),
            Level::TRACE => "TRACE".purple().bold(),
        };
        write!(writer, "{} ", level_str)?;
        
        // Format the message
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        
        writeln!(writer)
    }
}

/// Log a success message with green checkmark
pub fn success(message: &str) {
    if io::stdout().is_terminal() {
        println!("{} {}", "✓".green().bold(), message);
    } else {
        println!("SUCCESS: {}", message);
    }
}

/// Log an info message with blue info icon
pub fn info(message: &str) {
    if io::stdout().is_terminal() {
        println!("{} {}", "ℹ".blue().bold(), message);
    } else {
        println!("INFO: {}", message);
    }
}

/// Log a warning message with yellow warning icon
pub fn warning(message: &str) {
    if io::stdout().is_terminal() {
        eprintln!("{} {}", "⚠".yellow().bold(), message);
    } else {
        eprintln!("WARNING: {}", message);
    }
}

/// Log an error message with red X
pub fn error(message: &str) {
    if io::stdout().is_terminal() {
        eprintln!("{} {}", "✗".red().bold(), message);
    } else {
        eprintln!("ERROR: {}", message);
    }
}