//! Fixtures shared by the unit tests across modules.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, ListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::download::DatasetSpec;

/// The 3-head `NPClassifier` head names, in their conventional order.
pub(crate) const NPC_HEADS: [&str; 3] = ["pathway", "superclass", "class"];

/// One test row: `SMILES`, `CID`, and one id list per head (aligned to the head
/// name slice passed to [`write_split_parquet`]).
pub(crate) type TestSplitRow = (&'static str, i64, Vec<Vec<u16>>);

/// Unwrap a `Result` in a test, failing the test if it is `Err`. Replaces the
/// `assert!(r.is_ok()); let Ok(x) = r else { unreachable!() };` idiom that the
/// no-`unwrap`/`expect` lint config would otherwise force at every call site.
pub(crate) fn ok<T, E>(result: Result<T, E>) -> T {
    let Ok(value) = result else {
        unreachable!("expected Ok in test");
    };
    value
}

/// Create a fresh, empty temp directory tagged with the test name and the pid.
pub(crate) fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("chemtax-smarts-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let created = std::fs::create_dir_all(&dir);
    assert!(created.is_ok());
    dir
}

/// Owned head names from string slices, as the dataset loader expects.
pub(crate) fn head_names(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

/// Write a parquet split with an arbitrary head set: `cid`, `smiles`, then one
/// `List<UInt16>` column per `{head}_ids`.
pub(crate) fn write_split_parquet(path: &Path, heads: &[&str], rows: &[TestSplitRow]) {
    let mut fields = vec![
        Field::new("cid", DataType::Int64, false),
        Field::new("smiles", DataType::Utf8, false),
    ];
    for name in heads {
        fields.push(Field::new(
            format!("{name}_ids"),
            DataType::List(Arc::new(Field::new("item", DataType::UInt16, true))),
            false,
        ));
    }
    let schema = Arc::new(Schema::new(fields));

    let cids = rows.iter().map(|(_, cid, _)| *cid).collect::<Int64Array>();
    let smiles = StringArray::from_iter_values(rows.iter().map(|(smiles, _, _)| *smiles));
    let mut columns: Vec<ArrayRef> = vec![Arc::new(cids), Arc::new(smiles)];
    for head_index in 0..heads.len() {
        let list = ListArray::from_iter_primitive::<arrow_array::types::UInt16Type, _, _>(
            rows.iter()
                .map(|(_, _, head_ids)| Some(head_ids[head_index].iter().copied().map(Some))),
        );
        columns.push(Arc::new(list));
    }
    let batch_result = RecordBatch::try_new(Arc::clone(&schema), columns);
    assert!(batch_result.is_ok());

    let file_result = File::create(path);
    assert!(file_result.is_ok());
    let Ok(file) = file_result else {
        unreachable!()
    };
    let writer_result = ArrowWriter::try_new(file, Arc::clone(&schema), None);
    assert!(writer_result.is_ok());
    let Ok(mut writer) = writer_result else {
        unreachable!()
    };
    let Ok(batch) = batch_result else {
        unreachable!()
    };
    let batch_write_result = writer.write(&batch);
    assert!(batch_write_result.is_ok());
    let close_result = writer.close();
    assert!(close_result.is_ok());
}

/// Create an empty placeholder for every `spec.files` entry not already present,
/// so `ensure_distillation_dataset` sees the dataset as fully downloaded and
/// never reaches the network. The loader only reads `vocabulary.json` and the
/// three split parquets, so empty placeholders for the rest are harmless.
pub(crate) fn touch_missing_spec_files(dir: &Path, spec: &DatasetSpec) {
    for key in spec.files {
        let path = dir.join(key);
        if !path.exists() {
            assert!(File::create(&path).is_ok());
        }
    }
}

/// Lay out a full dataset directory: a real `vocabulary.json` and the three split
/// parquets, plus empty placeholders for any other files the spec lists.
pub(crate) fn populate_dataset_dir(
    dir: &Path,
    spec: &DatasetSpec,
    heads: &[&str],
    vocabulary_json: &str,
    train: &[TestSplitRow],
    validation: &[TestSplitRow],
    test: &[TestSplitRow],
) {
    assert!(std::fs::write(dir.join("vocabulary.json"), vocabulary_json).is_ok());
    write_split_parquet(&dir.join("train.parquet"), heads, train);
    write_split_parquet(&dir.join("validation.parquet"), heads, validation);
    write_split_parquet(&dir.join("test.parquet"), heads, test);
    touch_missing_spec_files(dir, spec);
}

/// Copy the first `max_rows` rows of a parquet file into a new uncompressed
/// parquet, preserving the full schema. Lets the ignored real-data smoke test
/// subset the published splits without loading all of them.
pub(crate) fn slice_parquet(src: &Path, dst: &Path, max_rows: usize) {
    let file = ok(File::open(src));
    let builder = ok(ParquetRecordBatchReaderBuilder::try_new(file)).with_batch_size(max_rows);
    let mut reader = ok(builder.build());
    let Some(batch) = reader.next() else {
        unreachable!("source parquet has no row groups");
    };
    let batch = ok(batch);
    let out = ok(File::create(dst));
    let mut writer = ok(ArrowWriter::try_new(out, batch.schema(), None));
    assert!(writer.write(&batch).is_ok());
    assert!(writer.close().is_ok());
}
