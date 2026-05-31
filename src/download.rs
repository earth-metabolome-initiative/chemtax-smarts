use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::ValueEnum;
use indicatif::ProgressBar;
use serde::Serialize;
use zenodo_rs::{Auth, RecordId, ZenodoClient, ZenodoError};

use crate::util::{progress_style, usize_to_u64};

/// One published dataset: its `Zenodo` record and the files to fetch into the data
/// directory. The loader only consumes `vocabulary.json` and the three parquet
/// splits, so a spec lists exactly those plus whatever else the record ships.
pub struct DatasetSpec {
    /// `Zenodo` record id holding the dataset files.
    pub record_id: u64,
    /// `DOI` string for the published record.
    pub doi: &'static str,
    /// File keys to download from the record.
    pub files: &'static [&'static str],
    /// Head names the dataset's `vocabulary.json` is expected to declare. The
    /// loader checks the loaded vocabulary against these to catch a `--data-dir`
    /// that points at a different dataset (both datasets share file names).
    pub heads: &'static [&'static str],
}

/// The 3-head `NPClassifier` distilled teacher splits.
pub const NPCLASSIFIER_DATASET: DatasetSpec = DatasetSpec {
    record_id: 19_701_295,
    doi: "10.5281/zenodo.19701295",
    files: &[
        "README.md",
        "LICENSE",
        "manifest.json",
        "SHA256SUMS.txt",
        "vocabulary.json",
        "train.parquet",
        "train.pathway-vectors.f16.zst",
        "train.superclass-vectors.f16.zst",
        "train.class-vectors.f16.zst",
        "validation.parquet",
        "validation.pathway-vectors.f16.zst",
        "validation.superclass-vectors.f16.zst",
        "validation.class-vectors.f16.zst",
        "test.parquet",
        "test.pathway-vectors.f16.zst",
        "test.superclass-vectors.f16.zst",
        "test.class-vectors.f16.zst",
    ],
    heads: &["pathway", "superclass", "class"],
};

/// The 9-head `ClassyFire` (`ChemOnt`) splits, published as the
/// "InChIKey-Deduplicated ClassyFire/ChemOnt Label Collection". The record also
/// ships the raw source files. The loader downloads only the four artifacts below.
pub const CLASSYFIRE_DATASET: DatasetSpec = DatasetSpec {
    record_id: 20_472_700,
    doi: "10.5281/zenodo.20472700",
    files: &[
        "vocabulary.json",
        "train.parquet",
        "validation.parquet",
        "test.parquet",
    ],
    heads: &[
        "kingdom",
        "superclass",
        "class",
        "subclass",
        "direct_parent",
        "intermediate_nodes",
        "alternative_parents",
        "substituents",
        "mapped_features",
    ],
};

/// Which published dataset a run uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetName {
    /// The `NPClassifier` distilled teacher dataset.
    Npclassifier,
    /// The `ClassyFire` (`ChemOnt`) dataset.
    Classyfire,
}

impl DatasetName {
    /// Registry entry (record id, `DOI`, file list) for this dataset.
    #[must_use]
    pub const fn spec(self) -> &'static DatasetSpec {
        match self {
            Self::Npclassifier => &NPCLASSIFIER_DATASET,
            Self::Classyfire => &CLASSYFIRE_DATASET,
        }
    }
}

/// Outcome of processing one file from a dataset spec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DownloadedDatasetFile {
    /// File key within the record.
    pub key: String,
    /// Final path of the file on disk.
    pub path: PathBuf,
    /// Number of bytes written, zero when skipped.
    pub bytes_written: u64,
    /// Whether the file was already present and left untouched.
    pub skipped: bool,
}

/// Return the dataset file keys that are absent from `data_dir`.
#[must_use]
pub fn missing_dataset_files(data_dir: &Path, spec: &DatasetSpec) -> Vec<&'static str> {
    spec.files
        .iter()
        .copied()
        .filter(|key| !data_dir.join(key).exists())
        .collect()
}

/// Download the dataset's split files into `data_dir`.
///
/// # Errors
///
/// Returns an error if the dataset directory cannot be created, the Zenodo
/// client cannot be initialized, or any required file cannot be downloaded and
/// atomically moved into place.
pub async fn ensure_dataset(
    data_dir: &Path,
    spec: &DatasetSpec,
) -> Result<Vec<DownloadedDatasetFile>, ZenodoError> {
    fs::create_dir_all(data_dir)?;
    let client = ZenodoClient::builder(Auth::new(
        std::env::var(Auth::TOKEN_ENV_VAR).unwrap_or_default(),
    ))
    .user_agent("chemtax-smarts/dataset")
    .build()?;

    let progress_bar = download_progress_bar(spec.files.len());
    let mut downloaded = Vec::with_capacity(spec.files.len());
    let mut downloaded_count = 0usize;
    let mut skipped_count = 0usize;
    for key in spec.files {
        let final_path = data_dir.join(key);
        if final_path.exists() {
            skipped_count += 1;
            downloaded.push(record_file(key, final_path, 0, true));
            progress_bar.set_message(format!("dataset | skipping {key}"));
            progress_bar.inc(1);
            progress_bar.println(format!("[skip] {key} | already present"));
            continue;
        }

        progress_bar.set_message(format!("dataset | downloading {key}"));
        let part_path = temporary_download_path(&final_path);
        if part_path.exists() {
            fs::remove_file(&part_path)?;
        }
        let resolved = client
            .download_record_file_by_key_to_path(RecordId(spec.record_id), key, &part_path)
            .await?;
        fs::rename(&part_path, &final_path)?;
        downloaded_count += 1;
        downloaded.push(record_file(key, final_path, resolved.bytes_written, false));
        progress_bar.inc(1);
        progress_bar.println(format!(
            "[download] {key} | {} bytes",
            resolved.bytes_written
        ));
    }

    progress_bar.finish_with_message(format!(
        "dataset ready | downloaded={downloaded_count} skipped={skipped_count}"
    ));
    Ok(downloaded)
}

fn temporary_download_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("download");
    path.with_file_name(format!("{file_name}.part"))
}

fn record_file(
    key: &str,
    path: PathBuf,
    bytes_written: u64,
    skipped: bool,
) -> DownloadedDatasetFile {
    DownloadedDatasetFile {
        key: key.to_owned(),
        path,
        bytes_written,
        skipped,
    }
}

fn download_progress_bar(total_files: usize) -> ProgressBar {
    let progress_bar = ProgressBar::new(usize_to_u64(total_files));
    progress_bar.set_style(progress_style(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} {msg}",
    ));
    progress_bar.enable_steady_tick(Duration::from_millis(100));
    progress_bar.set_message("dataset | checking required files");
    progress_bar
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{ok, temp_dir};

    #[test]
    fn missing_files_only_reports_absent_dataset_entries() {
        let temp_dir = temp_dir("download-test");

        ok(fs::write(temp_dir.join("manifest.json"), "{}\n"));

        let missing = missing_dataset_files(&temp_dir, &NPCLASSIFIER_DATASET);
        assert!(!missing.contains(&"manifest.json"));
        assert!(missing.contains(&"train.parquet"));
        assert!(missing.contains(&"test.class-vectors.f16.zst"));

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn temporary_download_path_uses_part_suffix() {
        let path = Path::new("data/train.parquet");
        assert_eq!(
            temporary_download_path(path),
            PathBuf::from("data/train.parquet.part")
        );
    }
}
