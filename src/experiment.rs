use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;

use clap::Parser;
use indicatif::{MultiProgress, ProgressBar};
use serde::{Deserialize, Serialize};
use smarts_evolution::{
    EvolutionConfig as SmartsEvolutionConfig, EvolutionError, EvolutionTask, FoldData,
    IndicatifEvolutionProgress, RankedSmarts, SeedCorpus, SmartsEvaluator, SmartsGenome,
    TaskResult, TuiEvolutionDashboard, TuiEvolutionError,
};
use thiserror::Error;
use zenodo_rs::ZenodoError;

use crate::dataset::{DatasetSplit, FoldSelectionCounts, LabelHead, Vocabulary};
use crate::download::{DatasetName, DownloadedDatasetFile, ensure_dataset};
use crate::util::{progress_style, usize_to_u64};

const ALL_POSITIVES_PER_LABEL: usize = usize::MAX;

/// Anything that can go wrong while running the experiment.
#[derive(Debug, Error)]
pub enum ExperimentError {
    /// Filesystem or other I/O failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// `JSON` serialization or deserialization failure.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Parquet read failure.
    #[error(transparent)]
    Parquet(#[from] parquet::errors::ParquetError),
    /// `Zenodo` download failure.
    #[error(transparent)]
    Zenodo(#[from] ZenodoError),
    /// Failure inside the `smarts-evolution` engine.
    #[error(transparent)]
    Evolution(#[from] EvolutionError),
    /// The native `TUI` dashboard could not start or run.
    #[error("evolution dashboard failed: {0}")]
    Dashboard(String),
    /// A named split parsed to zero rows.
    #[error("split {0} did not contain any rows")]
    EmptySplit(String),
    /// The dataset is structurally invalid or inconsistent.
    #[error("invalid dataset: {0}")]
    InvalidDataset(String),
    /// The shard selection (`--shard-index` / `--shard-count`) is inconsistent.
    #[error("invalid shard configuration: {0}")]
    InvalidShard(String),
    /// A parquet split is missing an expected column.
    #[error("missing parquet column {column} in split {split}")]
    MissingParquetColumn {
        /// Name of the split that was being read.
        split: String,
        /// Name of the expected column that was absent.
        column: String,
    },
    /// A split row carries `SMILES` that could not be parsed.
    #[error("split {split} contains an invalid SMILES row for CID {cid} ({smiles}): {message}")]
    InvalidSmiles {
        /// Name of the split holding the bad row.
        split: String,
        /// `PubChem` CID of the offending row.
        cid: i64,
        /// The `SMILES` string that failed to parse.
        smiles: String,
        /// Underlying parser message.
        message: String,
    },
    /// An evolved `SMARTS` pattern could not be parsed back for scoring.
    #[error("evolved SMARTS '{smarts}' for task {task_id} could not be parsed: {message}")]
    InvalidSmarts {
        /// Identifier of the task that produced the pattern.
        task_id: String,
        /// The `SMARTS` string that failed to parse.
        smarts: String,
        /// Underlying parser message.
        message: String,
    },
}

/// Whether the native `TUI` dashboard drives the evolution phase: yes when stdout
/// is an interactive terminal, otherwise the indicatif progress bars are used.
fn use_tui() -> bool {
    std::io::stdout().is_terminal()
}

/// Command line configuration for one experiment run.
// The bool fields are independent CLI flags, not a state machine, so the
// excessive-bools lint does not apply.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Parser, Serialize)]
pub struct ExperimentConfig {
    /// Which published dataset to evolve against. Required.
    #[arg(long, value_enum)]
    pub dataset: DatasetName,
    /// Directory holding the downloaded dataset files.
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,
    /// Directory where run artifacts are written.
    #[arg(long, default_value = "artifacts")]
    pub output_dir: PathBuf,
    /// Zero-based index of this shard when the label task plan is distributed
    /// across machines. Must be less than `--shard-count`. The sorted plan is
    /// striped by `task_index % shard_count == shard_index`, so each shard gets a
    /// disjoint, size-balanced slice. Give every shard its own `--output-dir`.
    #[arg(long, default_value_t = 0)]
    pub shard_index: usize,
    /// Number of shards the label task plan is split into. Defaults to 1, meaning
    /// no sharding (one process evolves every label).
    #[arg(long, default_value_t = 1)]
    pub shard_count: usize,
    /// Download the dataset files into `--data-dir` and exit without loading the
    /// splits or evolving any labels. Use once to warm a shared data directory
    /// before launching the shard array, so the workers all skip the download.
    #[arg(long)]
    pub download_only: bool,
    /// Ignore any existing `results.jsonl` and restart from scratch. By default a
    /// run resumes, skipping the labels already recorded in `results.jsonl`.
    #[arg(long)]
    pub fresh: bool,
    /// Optional cap on labels evolved per head.
    #[arg(long)]
    pub max_labels_per_head: Option<usize>,
    /// Minimum training positives a label needs to be evolved.
    #[arg(long, default_value_t = 10)]
    pub min_train_positives: usize,
    /// Minimum test positives a label needs to be evolved.
    #[arg(long, default_value_t = 1)]
    pub min_test_positives: usize,
    /// Cap on sampled negatives per stratification bucket (the trained head's
    /// other labels). The total negatives scale with the head's label count, so
    /// the default is per dataset: 4096 for `npclassifier`, 256 for `classyfire`.
    #[arg(long)]
    pub max_negatives_per_label: Option<usize>,
    /// Number of top `SMARTS` kept on each label's leaderboard.
    #[arg(long, default_value_t = 32)]
    pub leaderboard_size: usize,
    /// Genetic algorithm population size.
    #[arg(long, default_value_t = 512)]
    pub population_size: usize,
    /// Maximum generations per label.
    #[arg(long, default_value_t = 300)]
    pub generation_limit: u64,
    /// Per-individual mutation probability.
    #[arg(long, default_value_t = 0.85)]
    pub mutation_rate: f64,
    /// Per-pair crossover probability.
    #[arg(long, default_value_t = 0.70)]
    pub crossover_rate: f64,
    /// Fraction of the population entering the mating pool.
    #[arg(long, default_value_t = 0.50)]
    pub selection_ratio: f64,
    /// Number of individuals per selection tournament.
    #[arg(long, default_value_t = 3)]
    pub tournament_size: usize,
    /// Number of top individuals carried over unchanged each generation.
    #[arg(long, default_value_t = 4)]
    pub elite_count: usize,
    /// Fraction of each generation replaced by random immigrants.
    #[arg(long, default_value_t = 0.10)]
    pub random_immigrant_ratio: f64,
    /// Generations without improvement before a label stops early.
    #[arg(long, default_value_t = 30)]
    pub stagnation_limit: u64,
    /// Optional seed for deterministic runs.
    #[arg(long)]
    pub rng_seed: Option<u64>,
    /// Capacity of the per-label fitness cache.
    #[arg(long, default_value_t = 500_000)]
    pub fitness_cache_capacity: usize,
    /// Optional cap on the length of `SMARTS` evaluated during evolution.
    #[arg(long)]
    pub max_evaluation_smarts_len: Option<usize>,
    /// Per-match time budget in milliseconds.
    #[arg(long, default_value_t = 1_000)]
    pub match_time_limit_millis: u64,
    /// Disable the per-match time limit entirely.
    #[arg(long)]
    pub disable_match_time_limit: bool,
    /// Log evaluations slower than this many milliseconds.
    #[arg(long, default_value_t = 30_000)]
    pub slow_evaluation_log_threshold_millis: u64,
    /// Disable slow-evaluation logging entirely.
    #[arg(long)]
    pub disable_slow_evaluation_logging: bool,
}

impl ExperimentConfig {
    /// Negative-sampling cap to use, falling back to the dataset-specific default
    /// when `--max-negatives-per-label` is not set.
    fn negatives_per_label(&self) -> usize {
        self.max_negatives_per_label.unwrap_or(match self.dataset {
            DatasetName::Npclassifier => 4_096,
            DatasetName::Classyfire => 256,
        })
    }

    /// Reject a shard selection that names a non-existent shard.
    fn validate_sharding(&self) -> Result<(), ExperimentError> {
        if self.shard_count == 0 {
            return Err(ExperimentError::InvalidShard(
                "--shard-count must be at least 1".to_owned(),
            ));
        }
        if self.shard_index >= self.shard_count {
            return Err(ExperimentError::InvalidShard(format!(
                "--shard-index {} is out of range for --shard-count {}",
                self.shard_index, self.shard_count
            )));
        }
        Ok(())
    }

    /// Convert the CLI-facing knobs into one validated evolution config.
    ///
    /// # Errors
    ///
    /// Returns an error if the configured probabilities, limits, or counts are
    /// internally inconsistent for `smarts-evolution`.
    pub fn evolution_config(&self) -> Result<SmartsEvolutionConfig, ExperimentError> {
        let mut builder = SmartsEvolutionConfig::builder()
            .population_size(self.population_size)
            .generation_limit(self.generation_limit)
            .mutation_rate(self.mutation_rate)
            .crossover_rate(self.crossover_rate)
            .selection_ratio(self.selection_ratio)
            .tournament_size(self.tournament_size)
            .elite_count(self.elite_count)
            .random_immigrant_ratio(self.random_immigrant_ratio)
            .stagnation_limit(self.stagnation_limit)
            .fitness_cache_capacity(self.fitness_cache_capacity)
            .pubchem_compatible_smarts(true);
        if let Some(seed) = self.rng_seed {
            builder = builder.rng_seed(seed);
        }
        if let Some(max_len) = self.max_evaluation_smarts_len {
            builder = builder.max_evaluation_smarts_len(max_len);
        }
        if self.disable_match_time_limit || self.match_time_limit_millis == 0 {
            builder = builder.disable_match_time_limit();
        } else {
            builder = builder.match_time_limit(Duration::from_millis(self.match_time_limit_millis));
        }
        if self.disable_slow_evaluation_logging || self.slow_evaluation_log_threshold_millis == 0 {
            builder = builder.disable_slow_evaluation_logging();
        } else {
            builder = builder.slow_evaluation_log_threshold(Duration::from_millis(
                self.slow_evaluation_log_threshold_millis,
            ));
        }

        builder
            .build()
            .map_err(|message| ExperimentError::InvalidDataset(message.clone()))
    }
}

/// Row counts for one split after sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitCounts {
    /// Total sampled rows.
    pub rows: usize,
    /// Rows matching the trained label.
    pub positives: usize,
    /// Rows not matching the trained label.
    pub negatives: usize,
}

/// Scores for one candidate `SMARTS` on both splits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateScore {
    /// The candidate `SMARTS` pattern.
    pub smarts: String,
    /// Length of the `SMARTS` pattern.
    pub smarts_len: usize,
    /// Matthews correlation coefficient on the training split.
    pub training_mcc: f64,
    /// Coverage metric on the training split.
    pub training_coverage_score: f64,
    /// Matthews correlation coefficient on the test split.
    pub test_mcc: f64,
    /// Coverage metric on the test split.
    pub test_coverage_score: f64,
}

/// Outcome of a label whose evolution finished.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedTaskReport {
    /// Head name the label belongs to, such as `class` or `pathway`.
    pub head: String,
    /// Numeric label identifier within the head.
    pub label_id: u16,
    /// Human readable label name.
    pub label_name: String,
    /// Generations the evolution ran.
    pub generations: u64,
    /// Sampled counts for the training split.
    pub training_counts: SplitCounts,
    /// Sampled counts for the test split.
    pub test_counts: SplitCounts,
    /// Best training individual's `SMARTS`.
    pub training_best_smarts: String,
    /// Best training individual's Matthews correlation coefficient.
    pub training_best_mcc: f64,
    /// Best training individual's coverage metric.
    pub training_best_coverage_score: f64,
    /// Chosen candidate's `SMARTS`.
    pub selected_smarts: String,
    /// Length of the chosen candidate's `SMARTS`.
    pub selected_smarts_len: usize,
    /// Chosen candidate's training Matthews correlation coefficient.
    pub selected_training_mcc: f64,
    /// Chosen candidate's training coverage metric.
    pub selected_training_coverage_score: f64,
    /// Chosen candidate's test Matthews correlation coefficient.
    pub selected_test_mcc: f64,
    /// Chosen candidate's test coverage metric.
    pub selected_test_coverage_score: f64,
    /// All scored leaderboard candidates.
    pub candidates: Vec<CandidateScore>,
}

/// Outcome of a label that was skipped before or during evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkippedTaskReport {
    /// Head name the label belongs to, such as `class` or `pathway`.
    pub head: String,
    /// Numeric label identifier within the head.
    pub label_id: u16,
    /// Human readable label name.
    pub label_name: String,
    /// Why the label was skipped.
    pub reason: String,
    /// Sampled counts for the training split.
    pub training_counts: SplitCounts,
    /// Sampled counts for the test split.
    pub test_counts: SplitCounts,
}

/// One line of the results log, serialized as `JSON`.
#[derive(Debug, Clone, Serialize)]
pub enum TaskLogEntry {
    /// A finished label.
    Completed(CompletedTaskReport),
    /// A skipped label.
    Skipped(SkippedTaskReport),
}

/// Result of running a single label task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskOutcome {
    /// A finished label.
    Completed(CompletedTaskReport),
    /// A skipped label.
    Skipped(SkippedTaskReport),
}

/// Summary of a whole experiment run.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentSummary {
    /// `Zenodo` record id of the dataset.
    pub dataset_record_id: u64,
    /// `DOI` of the dataset.
    pub dataset_doi: String,
    /// Configuration the run used.
    pub config: ExperimentConfig,
    /// Files downloaded for the run.
    pub downloaded_files: Vec<DownloadedDatasetFile>,
    /// Number of labels that finished.
    pub completed_tasks: usize,
    /// Number of labels that were skipped.
    pub skipped_tasks: usize,
    /// Directory artifacts were written to.
    pub output_dir: PathBuf,
    /// Path of the per-label results log.
    pub results_path: PathBuf,
    /// Outcome of every label task in run order.
    pub outcomes: Vec<TaskOutcome>,
}

struct LoadedInputs {
    downloaded_files: Vec<DownloadedDatasetFile>,
    vocabulary: Vocabulary,
    training: DatasetSplit,
    test: DatasetSplit,
}

struct InputLoadProgress {
    overall_bar: ProgressBar,
    split_bar: ProgressBar,
}

impl InputLoadProgress {
    fn new() -> Self {
        let multi_progress = MultiProgress::new();
        multi_progress.set_move_cursor(true);

        let overall_bar = multi_progress.add(ProgressBar::new(4));
        overall_bar.set_style(progress_style(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} inputs | {msg}",
        ));
        overall_bar.set_message("starting input preparation");

        let split_bar = multi_progress.add(ProgressBar::new(1));
        split_bar.set_style(progress_style(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.magenta/blue}] {pos}/{len} rows | {msg}",
        ));
        split_bar.enable_steady_tick(Duration::from_millis(100));
        split_bar.set_message("waiting for first split");

        Self {
            overall_bar,
            split_bar,
        }
    }

    fn load_vocabulary(&self, path: &Path) -> Result<Vocabulary, ExperimentError> {
        self.overall_bar.set_message("vocabulary".to_owned());
        self.split_bar.set_length(1);
        self.split_bar.set_position(0);
        self.split_bar
            .set_message("vocabulary | loading labels".to_owned());
        self.split_bar.tick();

        let vocabulary = Vocabulary::load(path)?;
        self.overall_bar.inc(1);
        Ok(vocabulary)
    }

    fn load_split(
        &self,
        path: &Path,
        name: &str,
        head_names: &[String],
    ) -> Result<DatasetSplit, ExperimentError> {
        self.overall_bar.set_message(name.to_owned());

        let split = DatasetSplit::load_with_progress(path, name, head_names, &self.split_bar)?;
        self.overall_bar
            .println(format!("[done] {name} | rows={}", split.len()));
        self.overall_bar.inc(1);
        Ok(split)
    }

    fn finish(&self) {
        self.split_bar.finish_and_clear();
        self.overall_bar
            .finish_with_message("inputs ready".to_owned());
    }
}

struct ExperimentProgress {
    multi_progress: MultiProgress,
    overall_bar: ProgressBar,
    task_bar: ProgressBar,
    use_tui: bool,
}

impl ExperimentProgress {
    fn new(total_tasks: usize, use_tui: bool) -> Self {
        let multi_progress = MultiProgress::new();
        multi_progress.set_move_cursor(true);

        let overall_bar = multi_progress.add(ProgressBar::new(usize_to_u64(total_tasks)));
        overall_bar.set_style(progress_style(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} labels | {msg}",
        ));
        overall_bar.set_message("starting label sweep");

        let task_bar = multi_progress.add(ProgressBar::new(1));
        task_bar.set_style(progress_style(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.yellow/red}] {pos}/{len} steps | {msg}",
        ));
        task_bar.enable_steady_tick(Duration::from_millis(100));
        task_bar.set_message("waiting for first label");

        Self {
            multi_progress,
            overall_bar,
            task_bar,
            use_tui,
        }
    }

    fn start_task(&self, task_name: &str, training_len: usize, test_len: usize) {
        self.overall_bar.set_message(task_name.to_owned());
        self.task_bar.set_length(1);
        self.task_bar.set_position(0);
        self.task_bar.set_message(format!(
            "{task_name} | preparing sampled task sets | training={training_len} test={test_len}"
        ));
        self.task_bar.tick();
    }

    fn set_task_phase(&self, task_name: &str, steps: usize, message: String) {
        self.overall_bar.set_message(task_name.to_owned());
        self.task_bar.set_length(usize_to_u64(steps.max(1)));
        self.task_bar.set_position(0);
        self.task_bar.set_message(message);
        self.task_bar.tick();
    }

    fn log_skip(&self, task_name: &str, reason: &str) {
        self.log_line(format!("[skip] {task_name} | {reason}"));
        self.overall_bar.inc(1);
    }

    fn log_done(&self, report: &CompletedTaskReport) {
        self.log_line(format!(
            "[done] {}:{}:{} | selected={} | training_mcc={:.4} training_coverage={:.4} test_mcc={:.4} test_coverage={:.4}",
            report.head,
            report.label_id,
            report.label_name,
            report.selected_smarts,
            report.selected_training_mcc,
            report.selected_training_coverage_score,
            report.selected_test_mcc,
            report.selected_test_coverage_score
        ));
        self.overall_bar.inc(1);
    }

    fn finish(&self, completed_tasks: usize, skipped_tasks: usize) {
        self.task_bar.finish_and_clear();
        self.overall_bar.finish_with_message(format!(
            "experiment complete | completed={completed_tasks} skipped={skipped_tasks}"
        ));
    }

    fn log_line(&self, message: String) {
        self.overall_bar.println(message);
    }
}

struct TaskRunContext<'a> {
    config: &'a ExperimentConfig,
    evolution_config: &'a SmartsEvolutionConfig,
    seed_corpus: &'a SeedCorpus,
    progress: &'a ExperimentProgress,
    inputs: &'a LoadedInputs,
}

#[derive(Debug, Clone)]
struct TaskSplitCounts {
    training: SplitCounts,
    test: SplitCounts,
}

#[derive(Debug, Clone)]
struct PlannedLabelTask {
    ordinal: usize,
    head: LabelHead,
    head_name: String,
    label_id: u16,
    label_name: String,
    training_positives: usize,
    total_positives: usize,
}

impl PlannedLabelTask {
    fn task_name(&self) -> String {
        format!("{}:{}:{}", self.head_name, self.label_id, self.label_name)
    }
}

/// Run the full end-to-end experiment over the selected dataset's splits.
///
/// # Errors
///
/// Returns an error if the dataset cannot be downloaded or decoded, if the
/// output artifacts cannot be written, or if SMARTS evolution fails for a
/// non-skipped label task.
pub async fn run_experiment(
    config: &ExperimentConfig,
) -> Result<ExperimentSummary, ExperimentError> {
    config.validate_sharding()?;
    fs::create_dir_all(&config.output_dir)?;

    if config.download_only {
        let downloaded_files = ensure_dataset(&config.data_dir, config.dataset.spec()).await?;
        return Ok(download_only_summary(config, downloaded_files));
    }

    let inputs = load_inputs(config).await?;
    persist_run_metadata(config, &inputs.downloaded_files)?;

    let results_path = config.output_dir.join("results.jsonl");
    let resumed = initialize_results(&results_path, config.fresh)?;
    let outcomes = run_all_tasks(config, &inputs, &results_path, resumed)?;
    let (completed_tasks, skipped_tasks) = count_outcomes(&outcomes);

    let spec = config.dataset.spec();
    let summary = ExperimentSummary {
        dataset_record_id: spec.record_id,
        dataset_doi: spec.doi.to_owned(),
        config: config.clone(),
        downloaded_files: inputs.downloaded_files,
        completed_tasks,
        skipped_tasks,
        output_dir: config.output_dir.clone(),
        results_path: results_path.clone(),
        outcomes,
    };
    write_json_pretty(&config.output_dir.join("summary.json"), &summary)?;
    Ok(summary)
}

/// Build the summary returned by a `--download-only` run: the dataset is fetched
/// but no splits are loaded and no labels are planned.
fn download_only_summary(
    config: &ExperimentConfig,
    downloaded_files: Vec<DownloadedDatasetFile>,
) -> ExperimentSummary {
    let spec = config.dataset.spec();
    ExperimentSummary {
        dataset_record_id: spec.record_id,
        dataset_doi: spec.doi.to_owned(),
        config: config.clone(),
        downloaded_files,
        completed_tasks: 0,
        skipped_tasks: 0,
        output_dir: config.output_dir.clone(),
        results_path: config.output_dir.join("results.jsonl"),
        outcomes: Vec::new(),
    }
}

async fn load_inputs(config: &ExperimentConfig) -> Result<LoadedInputs, ExperimentError> {
    let downloaded_files = ensure_dataset(&config.data_dir, config.dataset.spec()).await?;
    let loading_progress = InputLoadProgress::new();
    let vocabulary = loading_progress.load_vocabulary(&config.data_dir.join("vocabulary.json"))?;
    check_dataset_matches(config, &vocabulary)?;
    let head_names = vocabulary.head_names();
    let train = loading_progress.load_split(
        &config.data_dir.join("train.parquet"),
        "train",
        &head_names,
    )?;
    let validation = loading_progress.load_split(
        &config.data_dir.join("validation.parquet"),
        "validation",
        &head_names,
    )?;
    let training = DatasetSplit::concatenate("training", vec![train, validation]);
    let test =
        loading_progress.load_split(&config.data_dir.join("test.parquet"), "test", &head_names)?;
    loading_progress.finish();
    Ok(LoadedInputs {
        downloaded_files,
        vocabulary,
        training,
        test,
    })
}

/// Guard against a `--data-dir` that holds a different dataset. Both datasets ship
/// the same file names, so a mismatched directory would silently load the wrong
/// data. Compare the loaded vocabulary's heads against the dataset's expected set.
fn check_dataset_matches(
    config: &ExperimentConfig,
    vocabulary: &Vocabulary,
) -> Result<(), ExperimentError> {
    let mut expected: Vec<&str> = config.dataset.spec().heads.to_vec();
    expected.sort_unstable();
    let mut loaded: Vec<String> = vocabulary.head_names();
    loaded.sort();
    let loaded_refs: Vec<&str> = loaded.iter().map(String::as_str).collect();
    if loaded_refs == expected {
        return Ok(());
    }
    Err(ExperimentError::InvalidDataset(format!(
        "vocabulary in {} declares heads {loaded:?}, but the {:?} dataset expects {expected:?}. Point --data-dir at the right dataset's files.",
        config.data_dir.display(),
        config.dataset,
    )))
}

fn persist_run_metadata(
    config: &ExperimentConfig,
    downloaded_files: &[DownloadedDatasetFile],
) -> Result<(), ExperimentError> {
    write_json_pretty(&config.output_dir.join("experiment-config.json"), config)?;
    write_json_pretty(
        &config.output_dir.join("downloaded-files.json"),
        downloaded_files,
    )?;
    Ok(())
}

/// The label identity used to decide what a resumed run can skip.
fn outcome_key(outcome: &TaskOutcome) -> (String, u16) {
    match outcome {
        TaskOutcome::Completed(report) => (report.head.clone(), report.label_id),
        TaskOutcome::Skipped(report) => (report.head.clone(), report.label_id),
    }
}

/// Read the outcomes already recorded in a `results.jsonl`. Lines that do not
/// parse (for example a partial final line from an interrupted run) are dropped,
/// so their labels are re-run.
fn load_recorded_outcomes(path: &Path) -> Result<Vec<TaskOutcome>, ExperimentError> {
    let text = fs::read_to_string(path)?;
    let mut outcomes = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(outcome) = serde_json::from_str::<TaskOutcome>(line) {
            outcomes.push(outcome);
        }
    }
    Ok(outcomes)
}

/// Prepare `results.jsonl` for the run and return the outcomes to resume from.
///
/// With `fresh` (or no existing log) the file is truncated and an empty list is
/// returned. Otherwise the existing log is read, rewritten cleanly (dropping any
/// partial trailing line), and its outcomes are returned so `run_all_tasks` can
/// skip those labels.
fn initialize_results(
    results_path: &Path,
    fresh: bool,
) -> Result<Vec<TaskOutcome>, ExperimentError> {
    let resumed = if fresh || !results_path.exists() {
        Vec::new()
    } else {
        load_recorded_outcomes(results_path)?
    };
    File::create(results_path)?;
    for outcome in &resumed {
        append_task_log_entry(results_path, outcome)?;
    }
    Ok(resumed)
}

/// The seed corpus, parsed once per process: the built-in fragments plus the
/// curated SMARTS embedded from `seeds/corpus.json`. Parsing all ~9k seeds is not
/// free, so it is cached rather than rebuilt for every run.
static SEED_CORPUS: LazyLock<Result<SeedCorpus, String>> = LazyLock::new(|| {
    const CORPUS_JSON: &str = include_str!("../seeds/corpus.json");
    let seeds: Vec<String> =
        serde_json::from_str(CORPUS_JSON).map_err(|error| error.to_string())?;
    let mut corpus = SeedCorpus::builtin();
    corpus.extend_from_smarts(seeds.iter().map(String::as_str))?;
    Ok(corpus)
});

/// Borrow the cached seed corpus.
///
/// # Errors
///
/// Returns an error if the embedded corpus JSON cannot be parsed or if any seed
/// SMARTS is invalid for `SmartsGenome`.
fn build_seed_corpus() -> Result<&'static SeedCorpus, ExperimentError> {
    match &*SEED_CORPUS {
        Ok(corpus) => Ok(corpus),
        Err(error) => Err(ExperimentError::InvalidDataset(error.clone())),
    }
}

fn run_all_tasks(
    config: &ExperimentConfig,
    inputs: &LoadedInputs,
    results_path: &Path,
    resumed: Vec<TaskOutcome>,
) -> Result<Vec<TaskOutcome>, ExperimentError> {
    let evolution_config = config.evolution_config()?;
    let seed_corpus = build_seed_corpus()?;
    let task_plan = sorted_task_plan(config, inputs)?;
    let shard_plan = shard_tasks(&task_plan, config.shard_index, config.shard_count);

    if config.shard_count > 1 {
        eprintln!(
            "shard {}/{} | {} of {} planned labels assigned to this shard",
            config.shard_index,
            config.shard_count,
            shard_plan.len(),
            task_plan.len()
        );
    }

    let done: HashSet<(String, u16)> = resumed.iter().map(outcome_key).collect();
    let remaining: Vec<&PlannedLabelTask> = shard_plan
        .into_iter()
        .filter(|task| !done.contains(&(task.head_name.clone(), task.label_id)))
        .collect();

    if !resumed.is_empty() {
        eprintln!(
            "resuming | {} labels already in results.jsonl, {} remaining",
            resumed.len(),
            remaining.len()
        );
    }
    let progress = ExperimentProgress::new(remaining.len(), use_tui());
    let task_context = TaskRunContext {
        config,
        evolution_config: &evolution_config,
        seed_corpus,
        progress: &progress,
        inputs,
    };
    let mut outcomes = resumed;

    for task in remaining {
        let outcome = run_label_task(&task_context, task)?;
        append_task_log_entry(results_path, &outcome)?;
        outcomes.push(outcome);
    }

    let (completed_tasks, skipped_tasks) = count_outcomes(&outcomes);
    progress.finish(completed_tasks, skipped_tasks);
    Ok(outcomes)
}

fn skipped_report(
    task: &PlannedLabelTask,
    reason: String,
    counts: TaskSplitCounts,
) -> SkippedTaskReport {
    SkippedTaskReport {
        head: task.head_name.clone(),
        label_id: task.label_id,
        label_name: task.label_name.clone(),
        reason,
        training_counts: counts.training,
        test_counts: counts.test,
    }
}

fn run_label_task(
    task_context: &TaskRunContext<'_>,
    task: &PlannedLabelTask,
) -> Result<TaskOutcome, ExperimentError> {
    let TaskRunContext {
        config,
        evolution_config,
        seed_corpus,
        progress,
        inputs,
    } = task_context;
    let task_name = task.task_name();
    progress.start_task(&task_name, inputs.training.len(), inputs.test.len());

    let counts = sampled_counts_for_task(
        task_context,
        &task_name,
        task.head,
        &task.head_name,
        task.label_id,
    );

    if let Some(reason) = skip_reason(config, &counts.training, &counts.test) {
        let skipped = skipped_report(task, reason, counts);
        progress.log_skip(&task_name, &skipped.reason);
        return Ok(TaskOutcome::Skipped(skipped));
    }

    let training_fold = inputs.training.build_sampled_fold_with_progress(
        task.head,
        &task.head_name,
        task.label_id,
        ALL_POSITIVES_PER_LABEL,
        config.negatives_per_label(),
        &progress.task_bar,
    )?;
    let Some(result) = evolve_fold_with_progress(
        &task_name,
        training_fold.fold,
        evolution_config,
        seed_corpus,
        config.leaderboard_size,
        task_context.progress,
    )?
    else {
        let skipped = skipped_report(
            task,
            "stopped from the dashboard before completing".to_owned(),
            counts,
        );
        progress.log_skip(&task_name, &skipped.reason);
        return Ok(TaskOutcome::Skipped(skipped));
    };

    let test_evaluator =
        build_test_evaluator(task_context, task.head, &task.head_name, task.label_id)?;
    let candidates = evaluate_candidates(
        &task_name,
        result.leaders(),
        &test_evaluator,
        task_context.progress,
    )?;
    let selected = select_candidate(&task_name, &result, &test_evaluator)?;
    let selected_training_mcc = selected.training_mcc;
    let selected_training_coverage_score = selected.training_coverage_score;
    let selected_test_coverage_score = selected.test_coverage_score;

    let report = CompletedTaskReport {
        head: task.head_name.clone(),
        label_id: task.label_id,
        label_name: task.label_name.clone(),
        generations: result.generations(),
        training_counts: counts.training,
        test_counts: counts.test,
        training_best_smarts: result.best_smarts().to_owned(),
        training_best_mcc: result.best_mcc(),
        training_best_coverage_score: result.best_coverage_score(),
        selected_smarts: selected.smarts.clone(),
        selected_smarts_len: selected.smarts_len,
        selected_training_mcc,
        selected_training_coverage_score,
        selected_test_mcc: selected.test_mcc,
        selected_test_coverage_score,
        candidates,
    };
    progress.log_done(&report);

    Ok(TaskOutcome::Completed(report))
}

fn sorted_task_plan(
    config: &ExperimentConfig,
    inputs: &LoadedInputs,
) -> Result<Vec<PlannedLabelTask>, ExperimentError> {
    let vocabulary = &inputs.vocabulary;
    let mut tasks = Vec::with_capacity(total_task_count(config, vocabulary));

    for head in vocabulary.heads() {
        let head_name = vocabulary.head_name(head);
        let labels = vocabulary.labels(head);
        let max_labels = config.max_labels_per_head.unwrap_or(labels.len());
        let training_positives = inputs.training.label_positive_counts(head, labels.len());
        let test_positives = inputs.test.label_positive_counts(head, labels.len());

        for (label_index, label_name) in labels.iter().enumerate().take(max_labels) {
            let label_id = u16::try_from(label_index).map_err(|error| {
                ExperimentError::InvalidDataset(format!(
                    "label index {label_index} overflowed u16 for {head_name}: {error}"
                ))
            })?;
            let training_count = training_positives[label_index];
            if training_count < config.min_train_positives {
                continue;
            }
            tasks.push(PlannedLabelTask {
                ordinal: tasks.len(),
                head,
                head_name: head_name.to_owned(),
                label_id,
                label_name: label_name.clone(),
                training_positives: training_count,
                total_positives: training_count + test_positives[label_index],
            });
        }
    }

    sort_task_plan(&mut tasks);
    Ok(tasks)
}

fn sort_task_plan(tasks: &mut [PlannedLabelTask]) {
    tasks.sort_by(compare_planned_tasks);
}

/// Select the tasks belonging to one shard. The sorted plan is striped by
/// position, so consecutive (similarly sized) tasks land in different shards and
/// each shard sees a balanced mix of cheap and expensive labels. The selection is
/// a pure function of the plan and the shard parameters, so it is stable across
/// resumes and identical on every machine.
fn shard_tasks(
    tasks: &[PlannedLabelTask],
    shard_index: usize,
    shard_count: usize,
) -> Vec<&PlannedLabelTask> {
    tasks
        .iter()
        .enumerate()
        .filter(|(position, _)| position % shard_count == shard_index)
        .map(|(_, task)| task)
        .collect()
}

fn compare_planned_tasks(left: &PlannedLabelTask, right: &PlannedLabelTask) -> Ordering {
    left.training_positives
        .cmp(&right.training_positives)
        .then_with(|| left.total_positives.cmp(&right.total_positives))
        .then_with(|| left.ordinal.cmp(&right.ordinal))
}

fn sampled_counts_for_task(
    task_context: &TaskRunContext<'_>,
    task_name: &str,
    head: LabelHead,
    head_name: &str,
    label_id: u16,
) -> TaskSplitCounts {
    let inputs = task_context.inputs;
    TaskSplitCounts {
        training: sampled_counts_from_split(
            task_context,
            task_name,
            &inputs.training,
            head,
            head_name,
            label_id,
        ),
        test: sampled_counts_from_split(
            task_context,
            task_name,
            &inputs.test,
            head,
            head_name,
            label_id,
        ),
    }
}

fn sampled_counts_from_split(
    task_context: &TaskRunContext<'_>,
    task_name: &str,
    split: &DatasetSplit,
    head: LabelHead,
    head_name: &str,
    label_id: u16,
) -> SplitCounts {
    task_context.progress.set_task_phase(
        task_name,
        split.len(),
        format!(
            "{} | counting sampled {head_name}:{label_id} rows",
            split.name(),
        ),
    );
    let counts = split.sampled_counts_with_progress(
        head,
        head_name,
        label_id,
        ALL_POSITIVES_PER_LABEL,
        task_context.config.negatives_per_label(),
        &task_context.progress.task_bar,
    );
    counts_from_selection_counts(counts)
}

fn build_test_evaluator(
    task_context: &TaskRunContext<'_>,
    head: LabelHead,
    head_name: &str,
    label_id: u16,
) -> Result<SmartsEvaluator, ExperimentError> {
    let config = task_context.config;
    let progress = &task_context.progress.task_bar;
    let test_fold = task_context.inputs.test.build_sampled_fold_with_progress(
        head,
        head_name,
        label_id,
        ALL_POSITIVES_PER_LABEL,
        config.negatives_per_label(),
        progress,
    )?;
    Ok(SmartsEvaluator::new(vec![test_fold.fold]))
}

/// Evolve one label's training fold.
///
/// Returns `Ok(None)` when an interactive run is stopped from the dashboard, so
/// the caller can skip the label and continue the sweep. Non-interactive runs
/// always return `Ok(Some(_))` or an error.
fn evolve_fold_with_progress(
    task_name: &str,
    training_fold: FoldData,
    evolution_config: &SmartsEvolutionConfig,
    seed_corpus: &SeedCorpus,
    leaderboard_size: usize,
    progress: &ExperimentProgress,
) -> Result<Option<TaskResult>, ExperimentError> {
    let task = EvolutionTask::new(task_name.to_owned(), vec![training_fold]);
    if progress.use_tui {
        return run_evolution_with_tui(
            task,
            evolution_config,
            seed_corpus,
            leaderboard_size,
            progress,
        );
    }
    progress.set_task_phase(
        task_name,
        usize::try_from(evolution_config.generation_limit()).unwrap_or(usize::MAX),
        format!("{task_name} | evolution progress from smarts-evolution"),
    );
    let evolution_progress = IndicatifEvolutionProgress::attach_to(&progress.multi_progress)
        .with_best_smarts_width(72)
        .clear_on_finish(true);
    let result = task.evolve_owned_with_indicatif_progress(
        evolution_config,
        seed_corpus,
        leaderboard_size,
        evolution_progress,
    )?;
    Ok(Some(result))
}

/// Drive one label's evolution with the native TUI dashboard.
///
/// The indicatif bars are suspended while the dashboard owns the terminal so
/// their steady-tick redraws do not corrupt the ratatui surface. Pressing stop
/// in the dashboard returns `Ok(None)`, which the caller treats as a skip.
fn run_evolution_with_tui(
    task: EvolutionTask,
    evolution_config: &SmartsEvolutionConfig,
    seed_corpus: &SeedCorpus,
    leaderboard_size: usize,
    progress: &ExperimentProgress,
) -> Result<Option<TaskResult>, ExperimentError> {
    let dashboard = TuiEvolutionDashboard::new().with_best_smarts_width(72);
    let outcome = progress.multi_progress.suspend(|| {
        task.evolve_owned_with_tui_dashboard(
            evolution_config,
            seed_corpus,
            leaderboard_size,
            dashboard,
        )
    });
    match outcome {
        Ok(result) => Ok(Some(result)),
        Err(TuiEvolutionError::Stopped) => Ok(None),
        Err(TuiEvolutionError::Evolution(error)) => Err(ExperimentError::Evolution(error)),
        Err(TuiEvolutionError::Terminal(error)) => Err(ExperimentError::Dashboard(format!(
            "could not start or drive the TUI dashboard ({error}). Redirect or pipe stdout to fall back to progress bars"
        ))),
        Err(
            error @ (TuiEvolutionError::WorkerDisconnected | TuiEvolutionError::WorkerPanicked),
        ) => Err(ExperimentError::Dashboard(error.to_string())),
    }
}

fn append_task_log_entry(path: &Path, outcome: &TaskOutcome) -> Result<(), ExperimentError> {
    let log_entry = match outcome {
        TaskOutcome::Completed(report) => TaskLogEntry::Completed(report.clone()),
        TaskOutcome::Skipped(report) => TaskLogEntry::Skipped(report.clone()),
    };
    append_json_line(path, &log_entry)
}

fn count_outcomes(outcomes: &[TaskOutcome]) -> (usize, usize) {
    outcomes.iter().fold(
        (0usize, 0usize),
        |(completed, skipped), outcome| match outcome {
            TaskOutcome::Completed(_) => (completed + 1, skipped),
            TaskOutcome::Skipped(_) => (completed, skipped + 1),
        },
    )
}

fn counts_from_selection_counts(counts: FoldSelectionCounts) -> SplitCounts {
    SplitCounts {
        rows: counts.positive_count + counts.negative_count,
        positives: counts.positive_count,
        negatives: counts.negative_count,
    }
}

fn skip_reason(
    config: &ExperimentConfig,
    training: &SplitCounts,
    test: &SplitCounts,
) -> Option<String> {
    if training.positives < config.min_train_positives {
        return Some(format!(
            "training positives {} < required {}",
            training.positives, config.min_train_positives
        ));
    }
    if test.positives < config.min_test_positives {
        return Some(format!(
            "test positives {} < required {}",
            test.positives, config.min_test_positives
        ));
    }
    if training.negatives == 0 {
        return Some("training split has no negatives".to_owned());
    }
    if test.negatives == 0 {
        return Some("test split has no negatives".to_owned());
    }
    None
}

fn total_task_count(config: &ExperimentConfig, vocabulary: &Vocabulary) -> usize {
    vocabulary
        .heads()
        .map(|head| {
            let label_count = vocabulary.labels(head).len();
            config
                .max_labels_per_head
                .map_or(label_count, |limit| limit.min(label_count))
        })
        .sum()
}

fn evaluate_candidates(
    task_id: &str,
    leaders: &[RankedSmarts],
    test_evaluator: &SmartsEvaluator,
    progress: &ExperimentProgress,
) -> Result<Vec<CandidateScore>, ExperimentError> {
    let mut candidates = Vec::with_capacity(leaders.len());
    let total_steps = leaders.len();
    progress.set_task_phase(
        task_id,
        total_steps,
        format!("{task_id} | scoring {} leader SMARTS", leaders.len()),
    );
    let mut completed_steps = 0usize;
    for leader in leaders {
        progress.task_bar.set_message(format!(
            "{task_id} | scoring test | {}/{}",
            completed_steps,
            total_steps.max(1)
        ));
        let test_score = evaluate_smarts(task_id, leader.smarts(), test_evaluator)?;
        completed_steps += 1;
        progress
            .task_bar
            .set_position(usize_to_u64(completed_steps));
        candidates.push(candidate_score(
            leader.smarts(),
            leader.smarts_len(),
            leader.mcc(),
            leader.coverage_score(),
            &test_score,
        ));
    }
    candidates.sort_by(compare_candidates);
    Ok(candidates)
}

fn candidate_score(
    smarts: &str,
    smarts_len: usize,
    training_mcc: f64,
    training_coverage_score: f64,
    test_score: &EvaluationScore,
) -> CandidateScore {
    CandidateScore {
        smarts: smarts.to_owned(),
        smarts_len,
        training_mcc,
        training_coverage_score,
        test_mcc: test_score.mcc,
        test_coverage_score: test_score.coverage_score,
    }
}

fn select_candidate(
    task_id: &str,
    result: &smarts_evolution::TaskResult,
    test_evaluator: &SmartsEvaluator,
) -> Result<CandidateScore, ExperimentError> {
    let test_score = evaluate_smarts(task_id, result.best_smarts(), test_evaluator)?;
    Ok(candidate_score(
        result.best_smarts(),
        result.best_smarts_len(),
        result.best_mcc(),
        result.best_coverage_score(),
        &test_score,
    ))
}

fn compare_candidates(left: &CandidateScore, right: &CandidateScore) -> Ordering {
    right
        .training_mcc
        .total_cmp(&left.training_mcc)
        .then_with(|| {
            right
                .training_coverage_score
                .total_cmp(&left.training_coverage_score)
        })
        .then_with(|| left.smarts_len.cmp(&right.smarts_len))
        .then_with(|| left.smarts.cmp(&right.smarts))
}

struct EvaluationScore {
    mcc: f64,
    coverage_score: f64,
}

fn evaluate_smarts(
    task_id: &str,
    smarts: &str,
    evaluator: &SmartsEvaluator,
) -> Result<EvaluationScore, ExperimentError> {
    let genome =
        SmartsGenome::from_smarts(smarts).map_err(|message| ExperimentError::InvalidSmarts {
            task_id: task_id.to_owned(),
            smarts: smarts.to_owned(),
            message,
        })?;
    let evaluation = evaluator.evaluate(&genome);
    Ok(EvaluationScore {
        mcc: evaluation.fitness().mcc(),
        coverage_score: evaluation.coverage_score(),
    })
}

fn append_json_line(path: &Path, value: &(impl Serialize + ?Sized)) -> Result<(), ExperimentError> {
    write_json(OpenOptions::new().append(true).open(path)?, false, value)
}

fn write_json_pretty(
    path: &Path,
    value: &(impl Serialize + ?Sized),
) -> Result<(), ExperimentError> {
    write_json(File::create(path)?, true, value)
}

fn write_json(
    mut handle: File,
    pretty: bool,
    value: &(impl Serialize + ?Sized),
) -> Result<(), ExperimentError> {
    if pretty {
        serde_json::to_writer_pretty(&mut handle, value)?;
    } else {
        serde_json::to_writer(&mut handle, value)?;
    }
    handle.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download::{CLASSYFIRE_DATASET, DatasetSpec, NPCLASSIFIER_DATASET};
    use crate::test_support::{
        self, NPC_HEADS, TestSplitRow, ok, populate_dataset_dir, slice_parquet, temp_dir,
        touch_missing_spec_files,
    };

    type TestClassRow = (&'static str, i64, Vec<u16>);

    fn npc_head_names() -> Vec<String> {
        test_support::head_names(&NPC_HEADS)
    }

    fn baseline_config() -> ExperimentConfig {
        ExperimentConfig {
            dataset: DatasetName::Npclassifier,
            data_dir: PathBuf::from("data"),
            output_dir: PathBuf::from("artifacts"),
            shard_index: 0,
            shard_count: 1,
            download_only: false,
            fresh: false,
            max_labels_per_head: None,
            min_train_positives: 10,
            min_test_positives: 1,
            max_negatives_per_label: None,
            leaderboard_size: 32,
            population_size: 512,
            generation_limit: 300,
            mutation_rate: 0.85,
            crossover_rate: 0.70,
            selection_ratio: 0.50,
            tournament_size: 3,
            elite_count: 4,
            random_immigrant_ratio: 0.10,
            stagnation_limit: 30,
            rng_seed: None,
            fitness_cache_capacity: 500_000,
            max_evaluation_smarts_len: None,
            match_time_limit_millis: 1_000,
            disable_match_time_limit: false,
            slow_evaluation_log_threshold_millis: 30_000,
            disable_slow_evaluation_logging: false,
        }
    }

    fn hidden_experiment_progress() -> ExperimentProgress {
        let multi_progress =
            MultiProgress::with_draw_target(indicatif::ProgressDrawTarget::hidden());
        let overall_bar = multi_progress.add(ProgressBar::hidden());
        let task_bar = multi_progress.add(ProgressBar::hidden());
        ExperimentProgress {
            multi_progress,
            overall_bar,
            task_bar,
            use_tui: false,
        }
    }

    /// Write a 3-head `NPClassifier` split with only `class` labels populated,
    /// delegating the arrow work to the shared fixture.
    fn write_split_parquet(path: &Path, rows: &[TestClassRow]) {
        let generic: Vec<test_support::TestSplitRow> = rows
            .iter()
            .map(|(smiles, cid, class_ids)| {
                (
                    *smiles,
                    *cid,
                    vec![Vec::new(), Vec::new(), class_ids.clone()],
                )
            })
            .collect();
        test_support::write_split_parquet(path, &NPC_HEADS, &generic);
    }

    // Write vocabulary.json plus three split parquets, load them, and fold
    // train+validation into a single "training" split.
    fn build_inputs(
        root: &Path,
        vocabulary_json: &str,
        train: &[TestClassRow],
        validation: &[TestClassRow],
        test: &[TestClassRow],
    ) -> LoadedInputs {
        let vocabulary_path = root.join("vocabulary.json");
        ok(std::fs::write(&vocabulary_path, vocabulary_json));

        write_split_parquet(&root.join("train.parquet"), train);
        write_split_parquet(&root.join("validation.parquet"), validation);
        write_split_parquet(&root.join("test.parquet"), test);

        let vocabulary = ok(Vocabulary::load(&vocabulary_path));
        let heads = npc_head_names();
        let train = ok(DatasetSplit::load(
            &root.join("train.parquet"),
            "train",
            &heads,
        ));
        let validation = ok(DatasetSplit::load(
            &root.join("validation.parquet"),
            "validation",
            &heads,
        ));
        let test = ok(DatasetSplit::load(
            &root.join("test.parquet"),
            "test",
            &heads,
        ));

        LoadedInputs {
            downloaded_files: Vec::new(),
            vocabulary,
            training: DatasetSplit::concatenate("training", vec![train, validation]),
            test,
        }
    }

    fn load_inputs_for_class_task(root: &Path) -> LoadedInputs {
        build_inputs(
            root,
            "{\n  \"pathway\": [],\n  \"superclass\": [],\n  \"class\": [\"amine\"]\n}\n",
            &[
                ("CCN", 1, vec![0]),
                ("N", 2, vec![0]),
                ("CCO", 3, vec![]),
                ("O", 4, vec![]),
            ],
            &[("CN", 5, vec![0]), ("CO", 6, vec![])],
            &[("CCN", 7, vec![0]), ("CCO", 8, vec![])],
        )
    }

    fn planned_task(
        ordinal: usize,
        label_id: u16,
        label_name: &str,
        training_positives: usize,
        test_positives: usize,
    ) -> PlannedLabelTask {
        PlannedLabelTask {
            ordinal,
            head: LabelHead::new(2),
            head_name: String::from("class"),
            label_id,
            label_name: label_name.to_owned(),
            training_positives,
            total_positives: training_positives + test_positives,
        }
    }

    fn planned_class_task(
        task_context: &TaskRunContext<'_>,
        label_id: u16,
        label_name: &str,
    ) -> PlannedLabelTask {
        let label_index = usize::from(label_id);
        let label_count = label_index + 1;
        let class = LabelHead::new(2);
        let training_positives = task_context
            .inputs
            .training
            .label_positive_counts(class, label_count)[label_index];
        let test_positives = task_context
            .inputs
            .test
            .label_positive_counts(class, label_count)[label_index];
        PlannedLabelTask {
            ordinal: 0,
            head: class,
            head_name: String::from("class"),
            label_id,
            label_name: label_name.to_owned(),
            training_positives,
            total_positives: training_positives + test_positives,
        }
    }

    #[test]
    fn negatives_per_label_defaults_per_dataset() {
        let mut config = baseline_config();
        config.max_negatives_per_label = None;
        config.dataset = DatasetName::Npclassifier;
        assert_eq!(config.negatives_per_label(), 4_096);
        config.dataset = DatasetName::Classyfire;
        assert_eq!(config.negatives_per_label(), 256);
        config.max_negatives_per_label = Some(99);
        assert_eq!(config.negatives_per_label(), 99);
    }

    #[test]
    fn total_task_count_respects_head_limits() {
        let vocabulary = Vocabulary::from_pairs([
            ("pathway", vec!["p0", "p1"]),
            ("superclass", vec!["s0"]),
            ("class", vec!["c0", "c1", "c2"]),
        ]);
        let mut config = baseline_config();
        config.max_labels_per_head = Some(2);
        assert_eq!(total_task_count(&config, &vocabulary), 5);
    }

    #[test]
    fn task_plan_filters_labels_with_too_few_training_examples() {
        let temp_dir = temp_dir("task-plan");

        let inputs = build_inputs(
            &temp_dir,
            "{\n  \"pathway\": [],\n  \"superclass\": [],\n  \"class\": [\"rare\", \"kept\"]\n}\n",
            &[
                ("CN", 1, vec![0]),
                ("CCN", 2, vec![1]),
                ("N", 3, vec![1]),
                ("CCO", 4, vec![]),
            ],
            &[("CN", 5, vec![0]), ("CCN", 6, vec![1])],
            &[("CN", 7, vec![0]), ("CCN", 8, vec![1])],
        );
        let mut config = baseline_config();
        config.min_train_positives = 3;

        let task_plan = ok(sorted_task_plan(&config, &inputs));
        assert_eq!(task_plan.len(), 1);
        assert_eq!(task_plan[0].label_name, "kept");

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn task_plan_sort_starts_with_fewest_training_examples() {
        let mut tasks = vec![
            planned_task(0, 0, "many", 10, 1),
            planned_task(1, 1, "few", 2, 4),
            planned_task(2, 2, "same_training_fewer_total", 2, 1),
        ];

        sort_task_plan(&mut tasks);

        assert_eq!(tasks[0].label_name, "same_training_fewer_total");
        assert_eq!(tasks[1].label_name, "few");
        assert_eq!(tasks[2].label_name, "many");
    }

    #[test]
    fn evolution_config_uses_tuned_defaults() {
        let config = baseline_config();
        let built = ok(config.evolution_config());
        assert_eq!(built.population_size(), 512);
        assert_eq!(built.generation_limit(), 300);
        assert_eq!(built.stagnation_limit(), 30);
        assert_eq!(built.tournament_size(), 3);
        assert_eq!(built.elite_count(), 4);
        assert_eq!(built.fitness_cache_capacity(), 500_000);
        assert!(built.pubchem_compatible_smarts());
        assert_eq!(built.match_time_limit(), Some(Duration::from_secs(1)));
        assert_eq!(
            built.slow_evaluation_log_threshold(),
            Some(Duration::from_secs(30))
        );
        assert_eq!(built.max_evaluation_smarts_len(), None);
        assert!((built.mutation_rate() - 0.85).abs() < f64::EPSILON);
        assert!((built.crossover_rate() - 0.70).abs() < f64::EPSILON);
        assert!((built.selection_ratio() - 0.50).abs() < f64::EPSILON);
        assert!((built.random_immigrant_ratio() - 0.10).abs() < f64::EPSILON);
    }

    #[test]
    fn evolution_config_forwards_slow_smarts_controls() {
        let mut config = baseline_config();
        config.max_evaluation_smarts_len = Some(96);
        config.match_time_limit_millis = 250;
        config.slow_evaluation_log_threshold_millis = 250;

        let built = ok(config.evolution_config());
        assert_eq!(built.max_evaluation_smarts_len(), Some(96));
        assert_eq!(built.match_time_limit(), Some(Duration::from_millis(250)));
        assert_eq!(
            built.slow_evaluation_log_threshold(),
            Some(Duration::from_millis(250))
        );
    }

    #[test]
    fn evolution_config_can_disable_match_time_limit() {
        let mut config = baseline_config();
        config.disable_match_time_limit = true;

        let built = ok(config.evolution_config());
        assert_eq!(built.match_time_limit(), None);
    }

    #[test]
    fn evolution_config_can_disable_slow_smarts_logging() {
        let mut config = baseline_config();
        config.disable_slow_evaluation_logging = true;

        let built = ok(config.evolution_config());
        assert_eq!(built.slow_evaluation_log_threshold(), None);
    }

    #[test]
    fn count_outcomes_splits_completed_and_skipped() {
        let completed = CompletedTaskReport {
            head: String::from("class"),
            label_id: 0,
            label_name: String::from("test"),
            generations: 1,
            training_counts: SplitCounts {
                rows: 2,
                positives: 1,
                negatives: 1,
            },
            test_counts: SplitCounts {
                rows: 2,
                positives: 1,
                negatives: 1,
            },
            training_best_smarts: String::from("[#6]"),
            training_best_mcc: 1.0,
            training_best_coverage_score: 1.0,
            selected_smarts: String::from("[#6]"),
            selected_smarts_len: 4,
            selected_training_mcc: 1.0,
            selected_training_coverage_score: 1.0,
            selected_test_mcc: 1.0,
            selected_test_coverage_score: 1.0,
            candidates: Vec::new(),
        };
        let skipped = SkippedTaskReport {
            head: String::from("pathway"),
            label_id: 1,
            label_name: String::from("skip"),
            reason: String::from("not enough positives"),
            training_counts: SplitCounts {
                rows: 1,
                positives: 0,
                negatives: 1,
            },
            test_counts: SplitCounts {
                rows: 1,
                positives: 0,
                negatives: 1,
            },
        };

        let outcomes = vec![
            TaskOutcome::Completed(completed),
            TaskOutcome::Skipped(skipped),
        ];
        assert_eq!(count_outcomes(&outcomes), (1, 1));
    }

    #[test]
    fn skip_reason_reports_too_few_training_positives() {
        let mut config = baseline_config();
        config.min_train_positives = 2;
        let reason = skip_reason(
            &config,
            &SplitCounts {
                rows: 1,
                positives: 1,
                negatives: 0,
            },
            &SplitCounts {
                rows: 2,
                positives: 1,
                negatives: 1,
            },
        );
        assert_eq!(reason.as_deref(), Some("training positives 1 < required 2"));
    }

    #[test]
    fn compare_candidates_prefers_training_then_coverage_then_simplicity() {
        let mut candidates = [
            CandidateScore {
                smarts: String::from("[#6]~[#7]"),
                smarts_len: 9,
                training_mcc: 0.8,
                training_coverage_score: 0.9,
                test_mcc: 0.7,
                test_coverage_score: 0.6,
            },
            CandidateScore {
                smarts: String::from("[#7]"),
                smarts_len: 4,
                training_mcc: 0.8,
                training_coverage_score: 0.8,
                test_mcc: 0.7,
                test_coverage_score: 0.7,
            },
            CandidateScore {
                smarts: String::from("[#8]"),
                smarts_len: 4,
                training_mcc: 0.6,
                training_coverage_score: 1.0,
                test_mcc: 0.6,
                test_coverage_score: 0.6,
            },
        ];
        candidates.sort_by(compare_candidates);
        assert_eq!(candidates[0].smarts, "[#6]~[#7]");
        assert_eq!(candidates[1].smarts, "[#7]");
    }

    #[test]
    fn skip_reason_reports_missing_negatives() {
        let mut config = baseline_config();
        config.min_train_positives = 1;
        let reason = skip_reason(
            &config,
            &SplitCounts {
                rows: 2,
                positives: 2,
                negatives: 0,
            },
            &SplitCounts {
                rows: 2,
                positives: 1,
                negatives: 1,
            },
        );
        assert_eq!(reason.as_deref(), Some("training split has no negatives"));
    }

    #[test]
    fn json_helpers_write_expected_payloads() {
        let temp_dir = temp_dir("json-helpers");

        let jsonl_path = temp_dir.join("results.jsonl");
        ok(File::create(&jsonl_path));
        let json_path = temp_dir.join("summary.json");

        ok(append_json_line(
            &jsonl_path,
            &SplitCounts {
                rows: 3,
                positives: 1,
                negatives: 2,
            },
        ));
        ok(write_json_pretty(
            &json_path,
            &SplitCounts {
                rows: 4,
                positives: 2,
                negatives: 2,
            },
        ));

        let jsonl_payload = ok(std::fs::read_to_string(&jsonl_path));
        assert_eq!(
            jsonl_payload,
            "{\"rows\":3,\"positives\":1,\"negatives\":2}\n"
        );

        let summary_payload = ok(std::fs::read_to_string(&json_path));
        assert!(summary_payload.contains("\"rows\": 4"));
        assert!(summary_payload.ends_with('\n'));

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn run_label_task_completes_on_small_class_problem() {
        let temp_dir = temp_dir("run-label-task");

        let inputs = load_inputs_for_class_task(&temp_dir);
        let mut config = baseline_config();
        config.min_train_positives = 1;
        config.population_size = 16;
        config.generation_limit = 6;
        config.stagnation_limit = 3;
        config.leaderboard_size = 4;
        config.rng_seed = Some(7);
        let evolution_config = ok(config.evolution_config());
        let progress = hidden_experiment_progress();
        let seed_corpus = SeedCorpus::builtin();
        let task_context = TaskRunContext {
            config: &config,
            evolution_config: &evolution_config,
            seed_corpus: &seed_corpus,
            progress: &progress,
            inputs: &inputs,
        };
        let task = planned_class_task(&task_context, 0, "amine");
        let outcome = run_label_task(&task_context, &task);
        let TaskOutcome::Completed(report) = ok(outcome) else {
            unreachable!();
        };
        assert_eq!(report.head, "class");
        assert_eq!(report.label_id, 0);
        assert_eq!(report.training_counts.positives, 3);
        assert_eq!(report.test_counts.positives, 1);
        assert!(!report.selected_smarts.is_empty());
        assert!(!report.candidates.is_empty());

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn run_label_task_returns_skipped_when_thresholds_are_not_met() {
        let temp_dir = temp_dir("run-label-skip");

        let inputs = load_inputs_for_class_task(&temp_dir);
        let mut config = baseline_config();
        config.min_train_positives = 10;
        let evolution_config = ok(config.evolution_config());
        let progress = hidden_experiment_progress();
        let seed_corpus = SeedCorpus::builtin();
        let task_context = TaskRunContext {
            config: &config,
            evolution_config: &evolution_config,
            seed_corpus: &seed_corpus,
            progress: &progress,
            inputs: &inputs,
        };
        let task = planned_class_task(&task_context, 0, "amine");
        let outcome = run_label_task(&task_context, &task);
        assert!(matches!(outcome, Ok(TaskOutcome::Skipped(_))));

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn seed_corpus_embeds_curated_smarts() {
        let builtin = SeedCorpus::builtin().len();
        let corpus = ok(build_seed_corpus());
        assert!(
            corpus.len() > builtin + 9_000,
            "corpus {} should add the curated seeds to builtin {builtin}",
            corpus.len()
        );
    }

    fn outcome_head(outcome: &TaskOutcome) -> &str {
        match outcome {
            TaskOutcome::Completed(report) => report.head.as_str(),
            TaskOutcome::Skipped(report) => report.head.as_str(),
        }
    }

    fn outcome_label_name(outcome: &TaskOutcome) -> &str {
        match outcome {
            TaskOutcome::Completed(report) => report.label_name.as_str(),
            TaskOutcome::Skipped(report) => report.label_name.as_str(),
        }
    }

    /// Run the real `run_experiment` against a prepared data dir and assert the
    /// dataset-agnostic invariants: artifacts written, every download skipped,
    /// outcomes and the results log line up, and at least one task completes.
    async fn run_pipeline(tag: &str, data_dir: &Path, dataset: DatasetName) -> ExperimentSummary {
        let output_dir = temp_dir(tag);
        let mut config = baseline_config();
        config.dataset = dataset;
        config.data_dir = data_dir.to_path_buf();
        config.output_dir = output_dir.clone();
        config.max_labels_per_head = Some(3);
        config.min_train_positives = 1;
        config.min_test_positives = 1;
        config.max_negatives_per_label = Some(256);
        config.population_size = 16;
        config.generation_limit = 6;
        config.stagnation_limit = 3;
        config.leaderboard_size = 4;
        config.rng_seed = Some(7);

        let summary = ok(run_experiment(&config).await);

        assert_eq!(
            summary.completed_tasks + summary.skipped_tasks,
            summary.outcomes.len()
        );
        assert!(summary.completed_tasks >= 1);
        for file in &summary.downloaded_files {
            assert!(file.skipped);
        }
        for name in [
            "summary.json",
            "results.jsonl",
            "experiment-config.json",
            "downloaded-files.json",
        ] {
            let metadata = ok(std::fs::metadata(output_dir.join(name)));
            assert!(metadata.len() > 0);
        }
        let results = ok(std::fs::read_to_string(output_dir.join("results.jsonl")));
        let line_count = results.lines().filter(|line| !line.is_empty()).count();
        assert_eq!(line_count, summary.outcomes.len());
        let summary_text = ok(std::fs::read_to_string(output_dir.join("summary.json")));
        let value: serde_json::Value = ok(serde_json::from_str(&summary_text));
        assert!(value.get("completed_tasks").is_some());
        assert!(value.get("outcomes").is_some());

        let _ = std::fs::remove_dir_all(&output_dir);
        summary
    }

    #[tokio::test]
    async fn smoke_run_npclassifier_end_to_end() {
        let data_dir = temp_dir("smoke-npc-data");
        let vocabulary = "{\"pathway\":[\"p0\"],\"superclass\":[\"s0\"],\"class\":[\"c0\",\"c1\"]}";
        let train: &[TestSplitRow] = &[
            ("CCN", 1, vec![vec![0], vec![0], vec![0]]),
            ("CCO", 2, vec![vec![0], vec![0], vec![0]]),
            ("CCC", 3, vec![vec![], vec![], vec![0]]),
            ("CCCC", 4, vec![vec![], vec![], vec![0]]),
            ("c1ccccc1", 5, vec![vec![], vec![], vec![1]]),
            ("O", 6, vec![vec![], vec![], vec![]]),
            ("N", 7, vec![vec![], vec![], vec![]]),
            ("CO", 8, vec![vec![], vec![], vec![]]),
        ];
        let validation: &[TestSplitRow] = &[
            ("CN", 9, vec![vec![0], vec![0], vec![0]]),
            ("CC", 10, vec![vec![], vec![], vec![]]),
        ];
        let test: &[TestSplitRow] = &[
            ("CCN", 11, vec![vec![0], vec![0], vec![0]]),
            ("CCO", 12, vec![vec![], vec![], vec![]]),
        ];
        populate_dataset_dir(
            &data_dir,
            &NPCLASSIFIER_DATASET,
            &NPC_HEADS,
            vocabulary,
            train,
            validation,
            test,
        );

        let summary = run_pipeline("smoke-npc-out", &data_dir, DatasetName::Npclassifier).await;
        assert_eq!(summary.dataset_record_id, 19_701_295);
        for outcome in &summary.outcomes {
            assert!(NPC_HEADS.contains(&outcome_head(outcome)));
        }

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[tokio::test]
    async fn smoke_run_classyfire_end_to_end() {
        const HEADS: [&str; 9] = [
            "kingdom",
            "superclass",
            "class",
            "subclass",
            "direct_parent",
            "intermediate_nodes",
            "alternative_parents",
            "substituents",
            "mapped_features",
        ];
        let data_dir = temp_dir("smoke-cf-data");
        let vocabulary = "{\"kingdom\":[\"k0\",\"k1\"],\"superclass\":[\"sc0\"],\"class\":[\"cl0\"],\"subclass\":[],\"direct_parent\":[\"d0\"],\"intermediate_nodes\":[\"i0\"],\"alternative_parents\":[\"a0\",\"a1\"],\"substituents\":[\"su0\"],\"mapped_features\":[\"m0\"]}";
        // kingdom (index 0), direct_parent (4), alternative_parents (6, multi-label).
        // The remaining heads stay empty so they plan no tasks.
        let row = |smiles: &'static str,
                   cid: i64,
                   kingdom: Vec<u16>,
                   direct_parent: Vec<u16>,
                   alternative_parents: Vec<u16>|
         -> TestSplitRow {
            let mut ids = vec![Vec::new(); HEADS.len()];
            ids[0] = kingdom;
            ids[4] = direct_parent;
            ids[6] = alternative_parents;
            (smiles, cid, ids)
        };
        let train = [
            row("CCN", 1, vec![0], vec![0], vec![0]),
            row("CCO", 2, vec![0], vec![0], vec![0, 1]),
            row("CCC", 3, vec![0], vec![0], vec![1]),
            row("CN", 4, vec![1], vec![], vec![]),
            row("O", 5, vec![], vec![], vec![]),
            row("N", 6, vec![], vec![], vec![]),
        ];
        let validation = [
            row("CO", 7, vec![0], vec![0], vec![0]),
            row("CC", 8, vec![], vec![], vec![]),
        ];
        let test = [
            row("CCN", 9, vec![0], vec![0], vec![0]),
            row("CCO", 10, vec![], vec![], vec![]),
        ];
        populate_dataset_dir(
            &data_dir,
            &CLASSYFIRE_DATASET,
            &HEADS,
            vocabulary,
            &train,
            &validation,
            &test,
        );

        let summary = run_pipeline("smoke-cf-out", &data_dir, DatasetName::Classyfire).await;
        assert_eq!(summary.dataset_record_id, 20_472_700);
        let distinct: std::collections::HashSet<&str> =
            summary.outcomes.iter().map(outcome_head).collect();
        assert!(distinct.len() >= 2);

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[tokio::test]
    async fn run_aborts_when_data_dir_holds_a_different_dataset() {
        // An NPClassifier-shaped vocabulary on disk while `--dataset classyfire`
        // is requested: the head guard must abort before loading any splits.
        let data_dir = temp_dir("wrong-dataset-data");
        let output_dir = temp_dir("wrong-dataset-out");
        let vocabulary = "{\"pathway\":[\"p0\"],\"superclass\":[\"s0\"],\"class\":[\"c0\"]}";
        let rows: &[TestSplitRow] = &[("CCN", 1, vec![vec![0], vec![0], vec![0]])];
        populate_dataset_dir(
            &data_dir,
            &CLASSYFIRE_DATASET,
            &NPC_HEADS,
            vocabulary,
            rows,
            rows,
            rows,
        );

        let mut config = baseline_config();
        config.dataset = DatasetName::Classyfire;
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();

        let result = run_experiment(&config).await;
        assert!(matches!(result, Err(ExperimentError::InvalidDataset(_))));

        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&output_dir);
    }

    #[tokio::test]
    async fn resume_skips_labels_already_in_results_log() {
        let data_dir = temp_dir("resume-data");
        let output_dir = temp_dir("resume-out");
        let vocabulary = "{\"pathway\":[\"p0\"],\"superclass\":[\"s0\"],\"class\":[\"c0\",\"c1\"]}";
        let train: &[TestSplitRow] = &[
            ("CCN", 1, vec![vec![0], vec![0], vec![0]]),
            ("CCO", 2, vec![vec![0], vec![0], vec![0]]),
            ("CCC", 3, vec![vec![], vec![], vec![0]]),
            ("CCCC", 4, vec![vec![], vec![], vec![0]]),
            ("c1ccccc1", 5, vec![vec![], vec![], vec![1]]),
            ("O", 6, vec![vec![], vec![], vec![]]),
            ("N", 7, vec![vec![], vec![], vec![]]),
            ("CO", 8, vec![vec![], vec![], vec![]]),
        ];
        let validation: &[TestSplitRow] = &[
            ("CN", 9, vec![vec![0], vec![0], vec![0]]),
            ("CC", 10, vec![vec![], vec![], vec![]]),
        ];
        let test: &[TestSplitRow] = &[
            ("CCN", 11, vec![vec![0], vec![0], vec![0]]),
            ("CCO", 12, vec![vec![], vec![], vec![]]),
        ];
        populate_dataset_dir(
            &data_dir,
            &NPCLASSIFIER_DATASET,
            &NPC_HEADS,
            vocabulary,
            train,
            validation,
            test,
        );

        let mut config = baseline_config();
        config.dataset = DatasetName::Npclassifier;
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.max_labels_per_head = Some(3);
        config.min_train_positives = 1;
        config.min_test_positives = 1;
        config.max_negatives_per_label = Some(256);
        config.population_size = 12;
        config.generation_limit = 2;
        config.stagnation_limit = 2;
        config.leaderboard_size = 4;
        config.rng_seed = Some(7);

        let results_path = output_dir.join("results.jsonl");

        // First run records every planned label.
        let first = ok(run_experiment(&config).await);
        let total = first.outcomes.len();
        assert!(total >= 2);

        // Rewrite the log keeping all but the last label, and rename the kept ones
        // to a sentinel. A resumed (loaded) outcome keeps the sentinel name, while a
        // re-evolved one would carry the real vocabulary name, so this distinguishes
        // a real skip from an accidental re-run.
        let recorded = ok(std::fs::read_to_string(&results_path));
        let mut lines: Vec<&str> = recorded
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect();
        lines.pop();
        let mut tampered = String::new();
        for line in &lines {
            let mut value: serde_json::Value = ok(serde_json::from_str(line));
            if let Some(object) = value.as_object_mut() {
                for inner in object.values_mut() {
                    if let Some(report) = inner.as_object_mut() {
                        report.insert(
                            "label_name".to_owned(),
                            serde_json::Value::String("RESUMED_MARKER".to_owned()),
                        );
                    }
                }
            }
            tampered.push_str(&ok(serde_json::to_string(&value)));
            tampered.push('\n');
        }
        assert!(std::fs::write(&results_path, &tampered).is_ok());

        // Resume: the recorded labels are skipped (sentinel survives), the one
        // dropped label is re-run (real name), and the total is unchanged.
        let second = ok(run_experiment(&config).await);
        assert_eq!(second.outcomes.len(), total);
        let resumed = second
            .outcomes
            .iter()
            .filter(|outcome| outcome_label_name(outcome) == "RESUMED_MARKER")
            .count();
        let rerun = second
            .outcomes
            .iter()
            .filter(|outcome| outcome_label_name(outcome) != "RESUMED_MARKER")
            .count();
        assert_eq!(resumed, total - 1);
        assert_eq!(rerun, 1);

        // A fresh run ignores the log and re-evolves everything, so no sentinel
        // survives.
        config.fresh = true;
        let third = ok(run_experiment(&config).await);
        assert_eq!(third.outcomes.len(), total);
        assert!(
            third
                .outcomes
                .iter()
                .all(|outcome| outcome_label_name(outcome) != "RESUMED_MARKER")
        );

        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&output_dir);
    }

    #[test]
    fn shard_tasks_partition_the_plan_without_overlap() {
        let plan: Vec<PlannedLabelTask> = (0..10)
            .map(|index| {
                planned_task(
                    index,
                    u16::try_from(index).unwrap_or(0),
                    "label",
                    index + 1,
                    0,
                )
            })
            .collect();

        let shard_count = 3;
        let mut seen: Vec<u16> = Vec::new();
        for shard_index in 0..shard_count {
            let shard = shard_tasks(&plan, shard_index, shard_count);
            for task in &shard {
                seen.push(task.label_id);
            }
        }
        seen.sort_unstable();

        let expected: Vec<u16> = (0u16..10).collect();
        assert_eq!(seen, expected, "shards must cover every task exactly once");

        // A single shard sees the whole plan; counts stay balanced (4/3/3 here).
        assert_eq!(shard_tasks(&plan, 0, 1).len(), 10);
        assert_eq!(shard_tasks(&plan, 0, shard_count).len(), 4);
        assert_eq!(shard_tasks(&plan, 1, shard_count).len(), 3);
        assert_eq!(shard_tasks(&plan, 2, shard_count).len(), 3);
    }

    #[test]
    fn rejects_an_out_of_range_or_empty_shard() {
        let mut config = baseline_config();

        config.shard_count = 4;
        config.shard_index = 4;
        assert!(matches!(
            config.validate_sharding(),
            Err(ExperimentError::InvalidShard(_))
        ));

        config.shard_count = 0;
        config.shard_index = 0;
        assert!(matches!(
            config.validate_sharding(),
            Err(ExperimentError::InvalidShard(_))
        ));

        config.shard_count = 4;
        config.shard_index = 3;
        assert!(config.validate_sharding().is_ok());
    }

    #[tokio::test]
    async fn download_only_skips_planning_and_writes_no_results() {
        // All spec files are present, so `ensure_dataset` is a pure skip with no
        // network access. `--download-only` must return before loading splits.
        let data_dir = temp_dir("download-only-data");
        let output_dir = temp_dir("download-only-out");
        let vocabulary = "{\"pathway\":[\"p0\"],\"superclass\":[\"s0\"],\"class\":[\"c0\"]}";
        let rows: &[TestSplitRow] = &[("CCN", 1, vec![vec![0], vec![0], vec![0]])];
        populate_dataset_dir(
            &data_dir,
            &NPCLASSIFIER_DATASET,
            &NPC_HEADS,
            vocabulary,
            rows,
            rows,
            rows,
        );

        let mut config = baseline_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.download_only = true;

        let summary = ok(run_experiment(&config).await);
        assert_eq!(summary.completed_tasks, 0);
        assert!(summary.outcomes.is_empty());
        assert!(summary.downloaded_files.iter().all(|file| file.skipped));
        assert!(!output_dir.join("results.jsonl").exists());

        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&output_dir);
    }

    #[tokio::test]
    async fn sharded_runs_cover_the_same_labels_as_an_unsharded_run() {
        let data_dir = temp_dir("shard-data");
        let vocabulary = "{\"pathway\":[\"p0\"],\"superclass\":[\"s0\"],\"class\":[\"c0\",\"c1\"]}";
        let train: &[TestSplitRow] = &[
            ("CCN", 1, vec![vec![0], vec![0], vec![0]]),
            ("CCO", 2, vec![vec![0], vec![0], vec![0]]),
            ("CCC", 3, vec![vec![], vec![], vec![0]]),
            ("CCCC", 4, vec![vec![], vec![], vec![0]]),
            ("c1ccccc1", 5, vec![vec![], vec![], vec![1]]),
            ("O", 6, vec![vec![], vec![], vec![]]),
            ("N", 7, vec![vec![], vec![], vec![]]),
            ("CO", 8, vec![vec![], vec![], vec![]]),
        ];
        let validation: &[TestSplitRow] = &[
            ("CN", 9, vec![vec![0], vec![0], vec![0]]),
            ("CC", 10, vec![vec![], vec![], vec![]]),
        ];
        let test: &[TestSplitRow] = &[
            ("CCN", 11, vec![vec![0], vec![0], vec![0]]),
            ("CCO", 12, vec![vec![], vec![], vec![]]),
        ];
        populate_dataset_dir(
            &data_dir,
            &NPCLASSIFIER_DATASET,
            &NPC_HEADS,
            vocabulary,
            train,
            validation,
            test,
        );

        let run = |tag: &'static str, shard_index: usize, shard_count: usize| {
            let data_dir = data_dir.clone();
            async move {
                let output_dir = temp_dir(tag);
                let mut config = baseline_config();
                config.data_dir = data_dir;
                config.output_dir = output_dir.clone();
                config.shard_index = shard_index;
                config.shard_count = shard_count;
                config.max_labels_per_head = Some(3);
                config.min_train_positives = 1;
                config.min_test_positives = 1;
                config.max_negatives_per_label = Some(256);
                config.population_size = 12;
                config.generation_limit = 2;
                config.stagnation_limit = 2;
                config.leaderboard_size = 4;
                config.rng_seed = Some(7);
                let summary = ok(run_experiment(&config).await);
                let _ = std::fs::remove_dir_all(&output_dir);
                summary
            }
        };

        let full = run("shard-full", 0, 1).await;
        let shard0 = run("shard-0", 0, 2).await;
        let shard1 = run("shard-1", 1, 2).await;

        let keys = |summary: &ExperimentSummary| -> Vec<(String, u16)> {
            summary.outcomes.iter().map(outcome_key).collect()
        };
        let full_keys: HashSet<(String, u16)> = keys(&full).into_iter().collect();
        let shard0_keys: HashSet<(String, u16)> = keys(&shard0).into_iter().collect();
        let shard1_keys: HashSet<(String, u16)> = keys(&shard1).into_iter().collect();

        assert!(full.outcomes.len() >= 2, "need at least two tasks to split");
        assert!(
            shard0_keys.is_disjoint(&shard1_keys),
            "shards must not share a label"
        );
        let union: HashSet<(String, u16)> = shard0_keys.union(&shard1_keys).cloned().collect();
        assert_eq!(union, full_keys, "shards together cover the whole plan");

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[tokio::test]
    #[ignore = "requires local dataset files. Run with `cargo test -- --ignored` before release"]
    async fn real_data_smoke_run() {
        real_data_case(
            "data/classyfire-splits",
            &CLASSYFIRE_DATASET,
            DatasetName::Classyfire,
            20_472_700,
        )
        .await;
        real_data_case(
            "data",
            &NPCLASSIFIER_DATASET,
            DatasetName::Npclassifier,
            19_701_295,
        )
        .await;
    }

    /// Slice a subset of one real dataset's local files into a temp dir and run
    /// the pipeline against the real schema. Skips quietly if the files are
    /// absent so the test is a no-op on machines without the data.
    async fn real_data_case(
        src_dir: &str,
        spec: &DatasetSpec,
        dataset: DatasetName,
        record_id: u64,
    ) {
        let src = Path::new(src_dir);
        let required = [
            "vocabulary.json",
            "train.parquet",
            "validation.parquet",
            "test.parquet",
        ];
        if required.iter().any(|name| !src.join(name).exists()) {
            eprintln!("real_data_smoke_run: skipping {src_dir} (dataset files absent)");
            return;
        }
        let data_dir = temp_dir(&format!("smoke-real-{record_id}"));
        assert!(
            std::fs::copy(
                src.join("vocabulary.json"),
                data_dir.join("vocabulary.json")
            )
            .is_ok()
        );
        slice_parquet(
            &src.join("train.parquet"),
            &data_dir.join("train.parquet"),
            40_000,
        );
        slice_parquet(
            &src.join("validation.parquet"),
            &data_dir.join("validation.parquet"),
            8_000,
        );
        slice_parquet(
            &src.join("test.parquet"),
            &data_dir.join("test.parquet"),
            8_000,
        );
        touch_missing_spec_files(&data_dir, spec);

        let summary =
            run_pipeline(&format!("smoke-real-out-{record_id}"), &data_dir, dataset).await;
        assert_eq!(summary.dataset_record_id, record_id);

        let _ = std::fs::remove_dir_all(&data_dir);
    }
}
