#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

/// Parquet split loading, the dynamic label-head vocabulary, and fold sampling.
pub mod dataset;
/// Zenodo dataset registry and download of the published split files.
pub mod download;
/// End-to-end experiment: planning per-label tasks and evolving SMARTS for each.
pub mod experiment;
mod util;

#[cfg(test)]
mod test_support;

pub use dataset::{DatasetSplit, LabelHead, Vocabulary};
pub use download::{
    CLASSYFIRE_DATASET, DatasetName, DatasetSpec, DownloadedDatasetFile, NPCLASSIFIER_DATASET,
    ensure_dataset, missing_dataset_files,
};
pub use experiment::{
    ExperimentConfig, ExperimentError, ExperimentSummary, TaskLogEntry, TaskOutcome, run_experiment,
};
