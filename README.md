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
  --dataset npclassifier --data-dir data/npclassifier --output-dir artifacts-npclassifier

# ClassyFire / ChemOnt (Zenodo 20472700)
RUSTFLAGS="-C target-cpu=native" cargo run --release -- \
  --dataset classyfire --data-dir data/classyfire --output-dir artifacts-classyfire
```

Give each dataset its own `--data-dir` and `--output-dir`. Both datasets use the same file names, so a shared `--data-dir` would load whichever dataset's files are already there. The run aborts if the directory's `vocabulary.json` heads do not match `--dataset`. Both also declare `class` and `superclass` heads, so a shared `--output-dir` would make a resumed run skip the other dataset's labels.

The dataset is downloaded from Zenodo into `--data-dir` on first run and reused afterwards (default `data/`, but give each dataset a distinct path as above). Runs resume from `<output-dir>/results.jsonl` by default. Pass `--fresh` to start over.

On an interactive terminal, each label evolves in the native `smarts-evolution` dashboard (live MCC and coverage plots, the current best SMARTS, and pause/stop/help). When stdout is piped or redirected, it falls back to progress bars.

By default, each training/test task set includes all positives and samples up to 4096 negatives per label of the head being trained. Override negative sampling with `--max-negatives-per-label`.

Labels with fewer than 10 training examples are filtered out by default. Override with `--min-train-positives`. At this cutoff, 752 of 770 `NPClassifier` labels and 9,835 of 10,947 `ClassyFire` labels clear the bar and are evolved. Per head (passing / total):

| head | `NPClassifier` | `ClassyFire` |
| --- | --- | --- |
| `pathway` | 7 / 7 | - |
| `kingdom` | - | 2 / 2 |
| `superclass` | 75 / 76 | 29 / 31 |
| `class` | 670 / 687 | 604 / 641 |
| `subclass` | - | 1,644 / 1,921 |
| `direct_parent` | - | 3,220 / 3,617 |
| `intermediate_nodes` | - | 715 / 765 |
| `alternative_parents` | - | 2,870 / 3,172 |
| `substituents` | - | 372 / 396 |
| `mapped_features` | - | 379 / 402 |
| total | 752 / 770 | 9,835 / 10,947 |

The default GA evaluates 512 SMARTS per generation for up to 300 generations, with early stopping after 30 stagnant generations.

Results report both MCC and match coverage scores for the merged training pool and held-out test split.

Generated SMARTS are restricted to the conservative PubChem-compatible subset provided by `smarts-evolution`.

Each SMARTS evaluation has a 1 second time limit, and evaluations slower than 30 seconds are logged to `<output-dir>/slow-smarts.log`. Tune these with `--match-time-limit-millis`, `--slow-evaluation-log-threshold-millis`, and `--max-evaluation-smarts-len`.
