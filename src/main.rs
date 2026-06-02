//! Command line entry point. Parses the experiment config and runs the full
//! per-label `SMARTS` evolution sweep over the selected dataset.

use std::path::PathBuf;

use clap::Parser;
use smarts_evolution::FileLogConfig;

use chemtax_smarts::{ExperimentConfig, run_experiment};

fn init_slow_smarts_log(config: &ExperimentConfig) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let slow_smarts_log_path = config.output_dir.join("slow-smarts.log");
    FileLogConfig::new(slow_smarts_log_path.clone())
        .append(false)
        .init()?;
    Ok(slow_smarts_log_path)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ExperimentConfig::parse();

    // Distributed modes (NATS feature): merge logs, seed the queue, or run as a
    // worker. Each returns before the single-machine path.
    #[cfg(feature = "nats")]
    {
        use chemtax_smarts::experiment::distributed;
        if config.merge_logs.is_some() {
            distributed::merge_logs(&config)?;
            return Ok(());
        }
        if config.seed_queue {
            distributed::seed_queue(&config).await?;
            return Ok(());
        }
        if config.nats_url.is_some() {
            std::fs::create_dir_all(&config.output_dir)?;
            init_slow_smarts_log(&config)?;
            Box::pin(distributed::run_worker(&config)).await?;
            return Ok(());
        }
    }

    std::fs::create_dir_all(&config.output_dir)?;
    let slow_smarts_log_path = init_slow_smarts_log(&config)?;
    let summary = run_experiment(&config).await?;
    println!(
        "completed {} tasks, skipped {} tasks | results={} | summary={} | slow_smarts={}",
        summary.completed_tasks,
        summary.skipped_tasks,
        summary.results_path.display(),
        summary.output_dir.join("summary.json").display(),
        slow_smarts_log_path.display(),
    );
    Ok(())
}
