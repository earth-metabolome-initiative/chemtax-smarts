# Lawrencium shard runs

These wrappers distribute a full `chemtax-smarts` run over LBL's Lawrencium cluster under SLURM. The work is one-vs-rest SMARTS evolution per label, which is embarrassingly parallel: thousands of independent labels (752 for NPClassifier, 9,835 for ClassyFire at the default cutoff). The binary's deterministic, sorted task plan is split into `SHARDS` shards by `task_index % shard_count`, so each array task evolves a disjoint, size-balanced slice and writes to its own directory.

## One-time setup

Set your allocation once. The scripts ship with a placeholder (`xxxxxxxxxxxxxxxx`), so this must be set before submitting:

```bash
export CHEMTAX_LRC_ACCOUNT=pc_yourpi
export CHEMTAX_REPO_DIR=$HOME/chemtax-smarts        # default
export CHEMTAX_SCRATCH_ROOT=/global/scratch/users/$USER/chemtax-smarts  # default
```

Build the release binary on a compute node so `target-cpu=native` matches the nodes the shards run on:

```bash
salloc --partition=lr6 --qos=lr_normal --nodes=1 --time=1:00:00
bash slurm/lrc/setup_env.sh
exit
```

## Run

```bash
bash slurm/lrc/prefetch.sh classyfire        # warm the shared data dir once
bash slurm/lrc/submit.sh classyfire          # 512 shards by default
bash slurm/lrc/status.sh classyfire 60       # watch the queue
bash slurm/lrc/merge.sh classyfire           # combine per-shard results when done
```

NPClassifier is the same with `npclassifier` (64 shards by default).

Check what would be submitted first with `--dry-run`, or run a single shard on the debug partition with `--debug`. Tune the split and scheduling with `--shards`, `--concurrency`, `--time`, `--partition`, and `--qos`. See `submit.sh -h`.

## Layout

Everything lives under `$CHEMTAX_SCRATCH_ROOT`:

```text
data/<dataset>/                 downloaded splits (shared, read-only at run time)
results/<dataset>/shards/shard-00000/   per-shard results.jsonl, summary.json, ...
results/<dataset>/results.jsonl         merged log written by merge.sh
logs/worker_<dataset>_<jobid>_<task>.{out,err}
```

## Restartability

Each shard resumes from its own `results.jsonl`, so a requeued, preempted, or re-submitted shard skips the labels it already finished. Re-running `submit.sh` with the same `--shards` is safe: the shard-to-label assignment is a pure function of the plan and `--shards`, so a shard always owns the same labels.

To reuse work from a prior run, pass `submit.sh --resume-from=<results.jsonl>`. Every shard treats that log's labels as done and skips them without copying the entries into its own output, so a local run's `results.jsonl` carries over to the cluster with no recompute and no duplication. The seed log must use the same dataset vocabulary. The final result is then the seed log plus the merged shard logs, which are disjoint.

Give each dataset its own data and output directories. Both datasets ship identical file names, and the run aborts if a data dir's `vocabulary.json` heads do not match `--dataset`. The defaults above already separate them by dataset.
