# chemtax-smarts

[![CI](https://github.com/earth-metabolome-initiative/chemtax-smarts/actions/workflows/ci.yml/badge.svg)](https://github.com/earth-metabolome-initiative/chemtax-smarts/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/earth-metabolome-initiative/chemtax-smarts/graph/badge.svg)](https://codecov.io/gh/earth-metabolome-initiative/chemtax-smarts)
[![NPClassifier DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19701295.svg)](https://doi.org/10.5281/zenodo.19701295)
[![ClassyFire DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20472700.svg)](https://doi.org/10.5281/zenodo.20472700)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`chemtax-smarts` downloads a chemical-taxonomy dataset from Zenodo and evolves one-vs-rest SMARTS for every label of every classification head the dataset declares, with fixed GA defaults. The head set comes from the dataset's `vocabulary.json`, so the same run handles the 3-head `NPClassifier` splits (pathway, superclass, class) and the 9-head `ClassyFire` (`ChemOnt`) splits. The published train and validation splits are merged into one training pool, and the test split is held out for reporting.

## Run

`--dataset` is required and selects one taxonomy per run.

```bash
# NPClassifier (Zenodo 19701295)
RUSTFLAGS="-C target-cpu=native" cargo run --release -- \
  --dataset npclassifier --output-dir artifacts-npclassifier

# ClassyFire / ChemOnt (Zenodo 20472700)
RUSTFLAGS="-C target-cpu=native" cargo run --release -- \
  --dataset classyfire --output-dir artifacts-classyfire
```

Give each dataset its own `--output-dir`. Both declare `class` and `superclass` heads, and a run resumes by skipping the `(head, label)` pairs already in `<output-dir>/results.jsonl`, so a shared directory would make one dataset skip the other's labels.

The dataset is downloaded from Zenodo into `--data-dir` (default `data/`) on first run and reused afterwards. A run resumes by default. Pass `--fresh` to ignore an existing `results.jsonl` and start over.

When stdout is an interactive terminal, each label's evolution runs in the native full-screen dashboard from `smarts-evolution`, with live MCC and coverage plots, the current best SMARTS, and pause, stop, and help controls. Stopping a label from the dashboard skips that label and moves on to the next one. When the output is redirected or piped (for example in CI), the run falls back to the indicatif progress bars instead.

By default, each training/test task set includes all positives and samples up to 4096 negatives per label of the head being trained. Override negative sampling with `--max-negatives-per-label`.

Labels with fewer than 10 training examples are filtered out by default. Override with `--min-train-positives`.

The default GA evaluates 512 SMARTS per generation for up to 300 generations, with early stopping after 30 stagnant generations.

Results report both MCC and match coverage scores for the merged training pool and held-out test split.

Generated SMARTS are restricted to the conservative PubChem-compatible subset provided by `smarts-evolution`.

Slow SMARTS warnings are logged by default after 30 seconds. Each SMARTS evaluation also has a cooperative 1 second time limit by default, and SMARTS length can be capped before evaluation. Use `--slow-evaluation-log-threshold-millis`, `--match-time-limit-millis`, and `--max-evaluation-smarts-len` to tune evaluation guardrails. The run writes these warnings to `artifacts/slow-smarts.log` without colliding with the progress bars.
