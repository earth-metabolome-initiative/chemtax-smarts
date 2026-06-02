# Distributing over workstations with NATS JetStream

These wrappers distribute a `chemtax-smarts` run across several workstations on one Tailscale network, coordinated by a NATS JetStream work queue. The per-label work is embarrassingly parallel and the task list is derivable, so a worker only needs to be told which label to claim. The queue hands out labels dynamically: faster machines pull more, a machine can join or leave mid-run, and a label whose worker dies is redelivered to another.

This is the lighter-weight alternative to `slurm/lrc/` for a handful of machines rather than a scheduled cluster.

## How it works

- **Broker**: one `nats-server -js` on a Tailscale node.
- **Seeder** (run once): builds the same sorted task plan every worker would, drops anything already done, and publishes one `{head, label_id}` message per remaining label onto a WorkQueue stream.
- **Workers** (one per machine): pull one label at a time, evolve it, append to that machine's own `results.jsonl`, and ack. One label in flight per machine; rayon uses all of that machine's cores within the label.
- **Merge**: collect the per-machine logs onto one host and dedup them by `(head, label_id)`.

## Prerequisites

- `nats-server` on the broker node (https://github.com/nats-io/nats-server/releases).
- The binary built with the NATS feature on every machine: `cargo build --release --features nats`.
- Each machine needs its own local dataset copy and enough RAM to hold the splits (GBs for ClassyFire). The first worker run downloads that machine's copy from Zenodo automatically. NATS only carries label ids, never the data.

Optional environment overrides: `CHEMTAX_REPO_DIR` (default `$HOME/chemtax-smarts`), `CHEMTAX_DATA_DIR`, `CHEMTAX_OUTPUT_DIR`, and the broker's `CHEMTAX_NATS_ADDR` / `CHEMTAX_NATS_PORT` / `CHEMTAX_NATS_STORE`.

## Run

On the broker node:

```bash
bash dist/nats/start-broker.sh        # leave running; note this node's Tailscale IP
```

Once, from any machine (URL is the broker's Tailscale IP):

```bash
bash dist/nats/seed.sh classyfire nats://100.x.y.z:4222
```

On each workstation:

```bash
bash dist/nats/worker.sh classyfire nats://100.x.y.z:4222
```

When the workers report the queue drained, gather and merge:

```bash
bash dist/nats/collect-merge.sh classyfire workstation-a workstation-b workstation-c
```

## Notes

- No `--shards`/`--shard-index` assignment is needed here. The queue balances dynamically; that machinery is for the static SLURM path.
- Restart is safe. A requeued or restarted worker resumes from its own `results.jsonl`, and an unacked label (crashed worker) is redelivered after `--nats-ack-wait-secs` (default 1800). A label longer than that is kept alive by progress pings, so the timeout only bites on a real crash.
- Re-seeding purges and rebuilds the stream, so it never double-enqueues. Pass `seed.sh ... --resume-from <merged-log>` to skip labels already finished in a prior round.
- Cross-worker duplicates only happen on crash-before-ack, so `collect-merge.sh` dedups by `(head, label_id)` rather than plain concatenation. The final log equals a single-machine run's label set.
- NPClassifier and ClassyFire get their own stream names (`chemtax-<dataset>`), so the two can run at once against the same broker. Override with `--nats-stream`.
