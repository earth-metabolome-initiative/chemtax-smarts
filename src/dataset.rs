use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use arrow_array::cast::{as_list_array, as_primitive_array, as_string_array};
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch, UInt16Array};
use indexmap::IndexMap;
use indicatif::ProgressBar;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smarts_evolution::{FoldData, FoldSample};
use smarts_rs::PreparedTarget;
use smiles_parser::Smiles;

use crate::experiment::ExperimentError;
use crate::util::usize_to_u64;

const FOLD_PROGRESS_GRANULARITY: usize = 8_192;

fn sample_limit_label(limit: usize) -> String {
    if limit == usize::MAX {
        "all".to_owned()
    } else {
        limit.to_string()
    }
}

/// A handle to one label head, identified by its position in the dataset's
/// ordered head set (the key order of `vocabulary.json`). Head names and labels
/// live in [`Vocabulary`], so the handle itself is a `Copy` index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LabelHead(usize);

impl LabelHead {
    /// Build a head handle from its position in the canonical head order.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Position of this head in the canonical head order.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Ordered label vocabulary loaded from the dataset's `vocabulary.json`: an object
/// mapping each head name to its ordered list of label names. The key order is the
/// canonical head order that [`LabelHead`] indexes into.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Vocabulary {
    heads: IndexMap<String, Vec<String>>,
}

impl Vocabulary {
    /// Load the label vocabulary JSON emitted with the published dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if the JSON payload is not
    /// an object mapping head names to label-name arrays.
    pub fn load(path: &Path) -> Result<Self, ExperimentError> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// Build a vocabulary from ordered (head name, labels) pairs.
    #[must_use]
    pub fn from_pairs<I, S, L>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (S, L)>,
        S: Into<String>,
        L: IntoIterator,
        L::Item: Into<String>,
    {
        let heads = pairs
            .into_iter()
            .map(|(name, labels)| (name.into(), labels.into_iter().map(Into::into).collect()))
            .collect();
        Self { heads }
    }

    /// Number of heads the dataset declares.
    #[must_use]
    pub fn head_count(&self) -> usize {
        self.heads.len()
    }

    /// Iterate the head handles in canonical order.
    pub fn heads(&self) -> impl Iterator<Item = LabelHead> {
        (0..self.heads.len()).map(LabelHead::new)
    }

    /// Ordered head names, used to resolve the `{head}_ids` parquet columns.
    #[must_use]
    pub fn head_names(&self) -> Vec<String> {
        self.heads.keys().cloned().collect()
    }

    /// Name of one head, or `""` if the handle is out of range.
    #[must_use]
    pub fn head_name(&self, head: LabelHead) -> &str {
        self.heads
            .get_index(head.index())
            .map_or("", |(name, _)| name.as_str())
    }

    /// Ordered label names of one head, or an empty slice if the handle is out of range.
    #[must_use]
    pub fn labels(&self, head: LabelHead) -> &[String] {
        self.heads
            .get_index(head.index())
            .map_or(&[], |(_, labels)| labels.as_slice())
    }
}

/// One loaded dataset row: a molecule and its label ids per head.
#[derive(Clone)]
pub struct SplitRow {
    /// `PubChem` compound id of the molecule.
    pub cid: i64,
    /// `SMILES` string of the molecule.
    pub smiles: String,
    /// Label ids per head, indexed by [`LabelHead::index`]. The outer order
    /// matches the dataset's [`Vocabulary`] head order.
    pub head_ids: Vec<Vec<u16>>,
}

impl SplitRow {
    /// Label ids for one head, or an empty slice if the handle is out of range.
    #[must_use]
    pub fn labels(&self, head: LabelHead) -> &[u16] {
        self.head_ids.get(head.index()).map_or(&[], Vec::as_slice)
    }
}

/// One named dataset split (train, validation, or test) holding its loaded rows.
#[derive(Clone)]
pub struct DatasetSplit {
    name: String,
    rows: Vec<SplitRow>,
}

impl DatasetSplit {
    /// Load one published parquet split without preparing molecule match data.
    ///
    /// # Errors
    ///
    /// Returns an error if the parquet file cannot be read, if required columns
    /// are missing, if labels are malformed, or if any SMILES row cannot be
    /// parsed into the in-memory row representation.
    pub fn load(
        path: &Path,
        name: impl Into<String>,
        head_names: &[String],
    ) -> Result<Self, ExperimentError> {
        let progress_bar = ProgressBar::hidden();
        Self::load_with_progress(path, name, head_names, &progress_bar)
    }

    /// Load one published parquet split while updating a progress bar by row.
    ///
    /// # Errors
    ///
    /// Returns an error if the parquet file cannot be read, if required columns
    /// are missing, if labels are malformed, or if any SMILES row cannot be
    /// parsed into the in-memory row representation.
    pub fn load_with_progress(
        path: &Path,
        name: impl Into<String>,
        head_names: &[String],
        progress_bar: &ProgressBar,
    ) -> Result<Self, ExperimentError> {
        let name = name.into();
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let total_rows =
            usize::try_from(builder.metadata().file_metadata().num_rows()).map_err(|error| {
                ExperimentError::InvalidDataset(format!(
                    "split {name} reported an invalid row count: {error}"
                ))
            })?;
        progress_bar.set_length(usize_to_u64(total_rows));
        progress_bar.set_position(0);
        progress_bar.set_message(format!("{name} | loading parquet batches"));
        let reader = builder.build()?;

        let mut rows = Vec::with_capacity(total_rows);
        for batch in reader {
            let batch =
                batch.map_err(|error| ExperimentError::InvalidDataset(error.to_string()))?;
            progress_bar.set_message(format!("{name} | loading raw rows"));
            let prepared_rows = prepare_batch_rows(&batch, &name, head_names)?;
            progress_bar.inc(usize_to_u64(prepared_rows.len()));
            rows.extend(prepared_rows);
        }

        if rows.is_empty() {
            return Err(ExperimentError::EmptySplit(name));
        }

        Ok(Self { name, rows })
    }

    /// Merge several splits into one named split, preserving row order.
    #[must_use]
    pub fn concatenate(name: impl Into<String>, splits: Vec<Self>) -> Self {
        let row_count = splits.iter().map(Self::len).sum();
        let mut rows = Vec::with_capacity(row_count);
        for split in splits {
            rows.extend(split.rows);
        }
        Self {
            name: name.into(),
            rows,
        }
    }

    /// Name of this split.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Loaded rows of this split.
    #[must_use]
    pub fn rows(&self) -> &[SplitRow] {
        &self.rows
    }

    /// Number of rows in this split.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether this split has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Count rows carrying each label id of one head, indexed by label id.
    #[must_use]
    pub fn label_positive_counts(&self, head: LabelHead, label_count: usize) -> Vec<usize> {
        let mut counts = vec![0usize; label_count];
        for row in &self.rows {
            for &label_id in row.labels(head) {
                if let Some(count) = counts.get_mut(usize::from(label_id)) {
                    *count += 1;
                }
            }
        }
        counts
    }

    /// Build an unsampled labeled evaluation set.
    ///
    /// # Errors
    ///
    /// Returns an error if any selected SMILES cannot be parsed.
    pub fn build_fold(
        &self,
        head: LabelHead,
        head_name: &str,
        label_id: u16,
    ) -> Result<LabeledFold, ExperimentError> {
        let progress_bar = ProgressBar::hidden();
        self.build_fold_with_progress(head, head_name, label_id, &progress_bar)
    }

    /// Build an unsampled labeled evaluation set with progress.
    ///
    /// # Errors
    ///
    /// Returns an error if any selected SMILES cannot be parsed.
    pub fn build_fold_with_progress(
        &self,
        head: LabelHead,
        head_name: &str,
        label_id: u16,
        progress_bar: &ProgressBar,
    ) -> Result<LabeledFold, ExperimentError> {
        self.build_sampled_fold_with_progress(
            head,
            head_name,
            label_id,
            usize::MAX,
            usize::MAX,
            progress_bar,
        )
    }

    /// Count the positives and negatives a sampled fold would select, without
    /// parsing any SMILES.
    #[must_use]
    pub fn sampled_counts_with_progress(
        &self,
        head: LabelHead,
        head_name: &str,
        label_id: u16,
        max_positives_per_class: usize,
        max_negatives_per_class: usize,
        progress_bar: &ProgressBar,
    ) -> FoldSelectionCounts {
        let selection = self.select_sample_indices(
            head,
            head_name,
            label_id,
            max_positives_per_class,
            max_negatives_per_class,
            progress_bar,
        );
        FoldSelectionCounts {
            positive_count: selection.positive_count,
            negative_count: selection.negative_count,
        }
    }

    /// Build a sampled labeled evaluation set with progress.
    ///
    /// # Errors
    ///
    /// Returns an error if any selected SMILES cannot be parsed.
    pub fn build_sampled_fold_with_progress(
        &self,
        head: LabelHead,
        head_name: &str,
        label_id: u16,
        max_positives_per_class: usize,
        max_negatives_per_class: usize,
        progress_bar: &ProgressBar,
    ) -> Result<LabeledFold, ExperimentError> {
        let SampleSelection {
            indices,
            positive_count,
            negative_count,
        } = self.select_sample_indices(
            head,
            head_name,
            label_id,
            max_positives_per_class,
            max_negatives_per_class,
            progress_bar,
        );
        progress_bar.set_length(usize_to_u64(indices.len()));
        progress_bar.set_position(0);
        progress_bar.set_message(format!(
            "{} | preparing {} selected {head_name}:{label_id} targets",
            self.name,
            indices.len(),
        ));

        let completed_count = AtomicUsize::new(0);
        let selected_count = indices.len();
        let samples = indices
            .par_iter()
            .map(|&row_index| {
                let row = &self.rows[row_index];
                let sample = prepare_fold_sample(row, &self.name, head, label_id);
                let completed = completed_count.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                if completed.is_multiple_of(FOLD_PROGRESS_GRANULARITY)
                    || completed == selected_count
                {
                    progress_bar.set_position(usize_to_u64(completed));
                }
                sample
            })
            .collect::<Result<Vec<_>, _>>()?;
        drop(indices);

        progress_bar.set_message(format!(
            "{} | indexing {} selected {head_name}:{label_id} targets",
            self.name,
            samples.len(),
        ));
        progress_bar.tick();
        Ok(LabeledFold {
            fold: FoldData::new(samples),
            positive_count,
            negative_count,
        })
    }

    fn select_sample_indices(
        &self,
        head: LabelHead,
        head_name: &str,
        label_id: u16,
        max_positives_per_class: usize,
        max_negatives_per_class: usize,
        progress_bar: &ProgressBar,
    ) -> SampleSelection {
        let row_count = self.rows.len();
        progress_bar.set_length(usize_to_u64(row_count));
        progress_bar.set_position(0);
        let max_positives_label = sample_limit_label(max_positives_per_class);
        let max_negatives_label = sample_limit_label(max_negatives_per_class);
        progress_bar.set_message(format!(
            "{} | selecting {head_name}:{label_id} evaluation rows | max_positives_per_class={max_positives_label} max_negatives_per_class={max_negatives_label}",
            self.name,
        ));

        let mut all_positive_indices = Vec::new();
        let mut all_negative_indices = Vec::new();
        let mut positive_buckets: HashMap<Option<u16>, BinaryHeap<RankedIndex>> = HashMap::new();
        let mut negative_buckets: HashMap<Option<u16>, BinaryHeap<RankedIndex>> = HashMap::new();

        for (row_index, row) in self.rows.iter().enumerate() {
            if row.labels(head).contains(&label_id) {
                if max_positives_per_class == usize::MAX {
                    all_positive_indices.push(row_index);
                } else if max_positives_per_class > 0 {
                    push_sample_candidates(
                        &mut positive_buckets,
                        row,
                        head,
                        row_index,
                        max_positives_per_class,
                    );
                }
            } else if max_negatives_per_class == usize::MAX {
                all_negative_indices.push(row_index);
            } else if max_negatives_per_class > 0 {
                push_sample_candidates(
                    &mut negative_buckets,
                    row,
                    head,
                    row_index,
                    max_negatives_per_class,
                );
            }

            let completed = row_index + 1;
            if completed.is_multiple_of(FOLD_PROGRESS_GRANULARITY) || completed == row_count {
                progress_bar.set_position(usize_to_u64(completed));
            }
        }

        let mut positive_indices = if max_positives_per_class == usize::MAX {
            all_positive_indices
        } else {
            sampled_indices(positive_buckets)
        };
        positive_indices.sort_unstable();
        positive_indices.dedup();

        let mut negative_indices = if max_negatives_per_class == usize::MAX {
            all_negative_indices
        } else {
            sampled_indices(negative_buckets)
        };
        negative_indices.sort_unstable();
        negative_indices.dedup();

        let positive_count = positive_indices.len();
        let negative_count = negative_indices.len();
        positive_indices.extend(negative_indices);
        positive_indices.sort_unstable();

        SampleSelection {
            indices: positive_indices,
            positive_count,
            negative_count,
        }
    }
}

/// A prepared evaluation fold with its selected positive and negative counts.
#[derive(Clone)]
pub struct LabeledFold {
    /// Prepared match targets for every selected row.
    pub fold: FoldData,
    /// Number of selected rows carrying the target label.
    pub positive_count: usize,
    /// Number of selected rows not carrying the target label.
    pub negative_count: usize,
}

/// Positive and negative row counts a fold selection produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldSelectionCounts {
    /// Number of selected rows carrying the target label.
    pub positive_count: usize,
    /// Number of selected rows not carrying the target label.
    pub negative_count: usize,
}

struct SampleSelection {
    indices: Vec<usize>,
    positive_count: usize,
    negative_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RankedIndex {
    score: u64,
    index: usize,
}

impl Ord for RankedIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for RankedIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Negatives (and capped positives) are stratified by the label ids of the head
// currently being trained, so the per-class cap spreads the sample across that
// head's labels. Rows with no label in this head fall into a single `None`
// bucket.
fn push_sample_candidates(
    buckets: &mut HashMap<Option<u16>, BinaryHeap<RankedIndex>>,
    row: &SplitRow,
    head: LabelHead,
    row_index: usize,
    max_per_class: usize,
) {
    let ids = row.labels(head);
    if ids.is_empty() {
        push_sample_candidate(
            buckets,
            None,
            RankedIndex {
                score: sample_score(row, row_index, None),
                index: row_index,
            },
            max_per_class,
        );
        return;
    }

    for &id in ids {
        push_sample_candidate(
            buckets,
            Some(id),
            RankedIndex {
                score: sample_score(row, row_index, Some(id)),
                index: row_index,
            },
            max_per_class,
        );
    }
}

fn push_sample_candidate(
    buckets: &mut HashMap<Option<u16>, BinaryHeap<RankedIndex>>,
    class_id: Option<u16>,
    candidate: RankedIndex,
    max_per_class: usize,
) {
    let bucket = buckets.entry(class_id).or_default();
    if bucket.len() < max_per_class {
        bucket.push(candidate);
    } else if bucket.peek().is_some_and(|worst| candidate < *worst) {
        bucket.pop();
        bucket.push(candidate);
    }
}

fn sampled_indices(buckets: HashMap<Option<u16>, BinaryHeap<RankedIndex>>) -> Vec<usize> {
    buckets
        .into_values()
        .flat_map(BinaryHeap::into_iter)
        .map(|candidate| candidate.index)
        .collect()
}

fn sample_score(row: &SplitRow, row_index: usize, class_id: Option<u16>) -> u64 {
    let cid_bits = u64::from_ne_bytes(row.cid.to_ne_bytes());
    let row_bits = u64::try_from(row_index).unwrap_or(u64::MAX);
    let class_bits = class_id.map_or(u64::MAX, u64::from);
    mix64(cid_bits ^ row_bits.rotate_left(17) ^ class_bits.rotate_left(41))
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn column<'a>(
    batch: &'a RecordBatch,
    split: &str,
    name: &str,
) -> Result<&'a dyn Array, ExperimentError> {
    batch
        .column_by_name(name)
        .map(std::convert::AsRef::as_ref)
        .ok_or_else(|| ExperimentError::MissingParquetColumn {
            split: split.to_owned(),
            column: name.to_owned(),
        })
}

fn label_column<'a>(
    batch: &'a RecordBatch,
    split: &str,
    name: &str,
) -> Result<LabelColumnView<'a>, ExperimentError> {
    let list = as_list_array(column(batch, split, name)?);
    let values = list
        .values()
        .as_any()
        .downcast_ref::<UInt16Array>()
        .ok_or_else(|| {
            ExperimentError::InvalidDataset(format!(
                "list column {name} did not contain uint16 values"
            ))
        })?;
    Ok(LabelColumnView { list, values })
}

struct LabelColumnView<'a> {
    list: &'a arrow_array::ListArray,
    values: &'a UInt16Array,
}

impl LabelColumnView<'_> {
    fn values(&self, row_index: usize) -> Result<Vec<u16>, ExperimentError> {
        let offsets = self.list.value_offsets();
        let start = usize::try_from(offsets[row_index]).map_err(|error| {
            ExperimentError::InvalidDataset(format!(
                "row offset for index {row_index} overflowed usize: {error}"
            ))
        })?;
        let end = usize::try_from(offsets[row_index + 1]).map_err(|error| {
            ExperimentError::InvalidDataset(format!(
                "row offset for index {} overflowed usize: {error}",
                row_index + 1
            ))
        })?;
        Ok(self.values.values()[start..end].to_vec())
    }
}

struct RawSplitRow {
    cid: i64,
    smiles: String,
    head_ids: Vec<Vec<u16>>,
}

fn prepare_batch_rows(
    batch: &RecordBatch,
    split: &str,
    head_names: &[String],
) -> Result<Vec<SplitRow>, ExperimentError> {
    let smiles_array = as_string_array(column(batch, split, "smiles")?);
    let cid_array = as_primitive_array::<Int64Type>(column(batch, split, "cid")?);
    let head_views = head_names
        .iter()
        .map(|name| label_column(batch, split, &format!("{name}_ids")))
        .collect::<Result<Vec<_>, ExperimentError>>()?;

    let raw_rows = (0..batch.num_rows())
        .map(|row_index| {
            let head_ids = head_views
                .iter()
                .map(|view| view.values(row_index))
                .collect::<Result<Vec<_>, ExperimentError>>()?;
            Ok(RawSplitRow {
                cid: cid_array.value(row_index),
                smiles: smiles_array.value(row_index).to_owned(),
                head_ids,
            })
        })
        .collect::<Result<Vec<_>, ExperimentError>>()?;

    Ok(raw_rows.into_iter().map(prepare_raw_row).collect())
}

fn prepare_raw_row(row: RawSplitRow) -> SplitRow {
    let RawSplitRow {
        cid,
        smiles,
        head_ids,
    } = row;
    SplitRow {
        cid,
        smiles,
        head_ids,
    }
}

fn prepare_fold_sample(
    row: &SplitRow,
    split: &str,
    head: LabelHead,
    label_id: u16,
) -> Result<FoldSample, ExperimentError> {
    let parsed = row
        .smiles
        .parse::<Smiles>()
        .map_err(|error| ExperimentError::InvalidSmiles {
            split: split.to_owned(),
            cid: row.cid,
            smiles: row.smiles.clone(),
            message: error.to_string(),
        })?;
    let target = PreparedTarget::new(parsed);
    if row.labels(head).contains(&label_id) {
        Ok(FoldSample::positive(target))
    } else {
        Ok(FoldSample::negative(target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        NPC_HEADS, TestSplitRow, head_names, ok, temp_dir, write_split_parquet,
    };

    #[test]
    fn dataset_split_loads_rows_and_builds_binary_fold() {
        let temp_dir = temp_dir("dataset-load-test");

        let split_path = temp_dir.join("train.parquet");
        write_split_parquet(
            &split_path,
            &NPC_HEADS,
            &[
                ("CCN", 1, vec![vec![0], vec![1], vec![2]]),
                ("CCO", 2, vec![vec![], vec![], vec![]]),
                ("NC", 3, vec![vec![0], vec![1], vec![2]]),
            ],
        );

        let pathway = LabelHead::new(0);
        let class = LabelHead::new(2);
        let loaded = ok(DatasetSplit::load(
            &split_path,
            "train",
            &head_names(&NPC_HEADS),
        ));
        assert_eq!(loaded.name(), "train");
        assert_eq!(loaded.len(), 3);
        assert!(!loaded.is_empty());
        assert_eq!(loaded.rows()[0].cid, 1);
        assert_eq!(loaded.rows()[0].smiles, "CCN");
        assert_eq!(loaded.rows()[0].labels(class), &[2]);
        assert!(loaded.rows()[1].labels(pathway).is_empty());
        assert_eq!(loaded.label_positive_counts(class, 4), vec![0, 0, 2, 0]);
        assert_eq!(loaded.label_positive_counts(pathway, 1), vec![2]);

        let fold = ok(loaded.build_fold(class, "class", 2));
        assert_eq!(fold.positive_count, 2);
        assert_eq!(fold.negative_count, 1);
        assert_eq!(fold.fold.len(), 3);

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn sampled_fold_caps_positives_and_negatives_by_class() {
        let temp_dir = temp_dir("dataset-sampling-test");

        let split_path = temp_dir.join("train.parquet");
        write_split_parquet(
            &split_path,
            &NPC_HEADS,
            &[
                ("CCN", 1, vec![vec![], vec![], vec![0]]),
                ("CN", 2, vec![vec![], vec![], vec![0]]),
                ("CCO", 3, vec![vec![], vec![], vec![1]]),
                ("CO", 4, vec![vec![], vec![], vec![1]]),
                ("CCC", 5, vec![vec![], vec![], vec![2]]),
                ("CC", 6, vec![vec![], vec![], vec![2]]),
                ("O", 7, vec![vec![], vec![], vec![]]),
                ("N", 8, vec![vec![], vec![], vec![]]),
            ],
        );

        let class = LabelHead::new(2);
        let loaded = ok(DatasetSplit::load(
            &split_path,
            "train",
            &head_names(&NPC_HEADS),
        ));
        let progress_bar = ProgressBar::hidden();
        let counts = loaded.sampled_counts_with_progress(class, "class", 0, 1, 1, &progress_bar);
        assert_eq!(
            counts,
            FoldSelectionCounts {
                positive_count: 1,
                negative_count: 3,
            }
        );

        let fold =
            ok(loaded.build_sampled_fold_with_progress(class, "class", 0, 1, 1, &progress_bar));
        assert_eq!(fold.positive_count, 1);
        assert_eq!(fold.negative_count, 3);
        assert_eq!(fold.fold.len(), 4);

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn vocabulary_load_preserves_head_order_and_labels() {
        let temp_dir = temp_dir("vocabulary-test");

        let vocabulary_path = temp_dir.join("vocabulary.json");
        ok(std::fs::write(
            &vocabulary_path,
            "{\n  \"pathway\": [\"p0\"],\n  \"superclass\": [\"s0\", \"s1\"],\n  \"class\": [\"c0\", \"c1\", \"c2\"]\n}\n",
        ));

        let loaded = ok(Vocabulary::load(&vocabulary_path));
        assert_eq!(loaded.head_count(), 3);
        assert_eq!(loaded.head_names(), vec!["pathway", "superclass", "class"]);
        assert_eq!(loaded.head_name(LabelHead::new(0)), "pathway");
        assert_eq!(loaded.labels(LabelHead::new(0)), ["p0"]);
        assert_eq!(loaded.labels(LabelHead::new(1)), ["s0", "s1"]);
        assert_eq!(loaded.labels(LabelHead::new(2)), ["c0", "c1", "c2"]);

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn from_pairs_round_trips_head_order() {
        let vocabulary =
            Vocabulary::from_pairs([("pathway", vec!["p0"]), ("superclass", vec!["s0", "s1"])]);
        assert_eq!(vocabulary.head_count(), 2);
        let heads: Vec<&str> = vocabulary
            .heads()
            .map(|h| vocabulary.head_name(h))
            .collect();
        assert_eq!(heads, vec!["pathway", "superclass"]);
    }

    #[test]
    fn nine_head_classyfire_shape_loads_and_builds_folds() {
        let temp_dir = temp_dir("classyfire-shape");

        let heads = [
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
        // direct_parent is head index 4: a non-class head, to prove sampling and
        // folds work off any head, not just class.
        let direct_parent = LabelHead::new(4);
        let row = |smiles, cid, dp: u16| -> TestSplitRow {
            let mut ids = vec![Vec::new(); heads.len()];
            ids[0] = vec![0]; // kingdom
            ids[4] = vec![dp]; // direct_parent
            (smiles, cid, ids)
        };
        let split_path = temp_dir.join("train.parquet");
        write_split_parquet(
            &split_path,
            &heads,
            &[
                row("CCN", 1, 0),
                row("CN", 2, 0),
                row("CCO", 3, 1),
                row("O", 4, 1),
            ],
        );

        let loaded = ok(DatasetSplit::load(
            &split_path,
            "train",
            &head_names(&heads),
        ));
        assert_eq!(loaded.len(), 4);
        assert_eq!(loaded.rows()[0].head_ids.len(), 9);
        assert_eq!(loaded.rows()[0].labels(LabelHead::new(0)), &[0]);
        assert_eq!(loaded.rows()[0].labels(direct_parent), &[0]);
        assert!(loaded.rows()[0].labels(LabelHead::new(2)).is_empty());
        assert_eq!(loaded.label_positive_counts(direct_parent, 2), vec![2, 2]);

        let fold = ok(loaded.build_fold(direct_parent, "direct_parent", 0));
        assert_eq!(fold.positive_count, 2);
        assert_eq!(fold.negative_count, 2);

        let _ = std::fs::remove_dir_all(temp_dir);
    }
}
