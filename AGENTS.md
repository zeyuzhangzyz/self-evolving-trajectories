# Project Notes

## Repository Scope

This repository contains the public Ser-FOX / self-evolving trajectory training and evaluation code. Keep code, scripts, configs, and project documents synchronized through Git/GitHub before using them on remote machines.

## Ser-FOX Conventions

- `Ser-FOX/serfox_eval.py` defaults serialized-AR evaluation to `--pad_eos_last true`. This keeps PAD/EOS value tokens masked until all regular target positions are filled. Use `--pad_eos_last false` only to reproduce the legacy AR decode path.
- `--norepeat true` controls index-token deduplication. With `--pad_eos_last true`, `--norepeat false` disables only the dedup mask while keeping the PAD/EOS-last value-slot constraint.
- `Ser-FOX/serfox_soft_index.py` defines `logit_margin` as the ground-truth value logit minus the best non-ground-truth value logit. This avoids treating wrong-but-confident value predictions as high-readiness positions.
- If a mixed soft-index batch lacks a cache for any non-canonical source, return `soft_index_targets=None` so training falls back to dynamic target construction. Do not fill missing soft targets with all-zero distributions.
- In `Ser-FOX/serfox_train.py`, online regeneration is part of the main source in the `[Main, Prev, Canonical]` mix. Turning on `--online_regen_sample` must not bypass `--mix_ratios`; source 0 is sampled online while sources 1 and 2 remain prev and canonical according to the configured mix.
- `Ser-FOX/serfox_train.py` defaults `--shuffle_order true`: round 1 shuffles all sampled rows, while round 2+ shuffles only canonical mix rows and keeps regenerated main/prev rows in confident-first order.
- `--regen_copies_per_sample N` is the offline pseudo-online regen path. `N=1` is the deterministic argmax baseline and writes `train_<round>.bin`; `N>1` samples N trajectories per base sample from the soft-index position distribution, writes `train_<round>_xN.bin`, and trains by uniformly sampling rows from that enlarged pool. Do not reuse a higher-N pool to materialize the N=1 argmax baseline.
- Canonical mix rows supervise values only for index positions in round 2+. They should not add hard or soft supervision for one specific canonical index order.

## Remote Run Discipline

- Write formal training, evaluation, or batch commands into bash scripts in the repository before running them remotely.
- Commit and push the script and related code first, then run the Git-synchronized script on the server.
- Before starting resource-heavy jobs, report dataset, checkpoint/model, key hyperparameters, machine, GPU/CPU plan, output directory, command/script path, overwrite behavior, and checkpoint/provenance behavior.
