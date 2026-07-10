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

## Local Diagnostic Experiments

- `scripts/experiments/run_compare_index_gradient_estimators.sh` is the CPU-only launcher for comparing direct all-index weighting with the current batch-shared random-frontier estimator.
- The experiment uses tiny synthetic serialized trajectories and `Ser-FOX/serfox_model.py`; it writes raw CSV, exact expectation checks, JSON/Markdown summaries, a log, and `provenance.txt` under `results/index_weighting_sampling/<run_tag>/`.
- The launcher refuses to overwrite an existing output directory and explicitly disables CUDA.

## amax-77 Diagnostic Environment

- SSH alias: `amax-77`.
- Isolated repository for the index-gradient diagnostic: `/home/amax/experiments/self-evolving-trajectories-9fc8d76`, cloned through Git and detached at commit `9fc8d76e1f4905c600540b18a5e31e929f104a10`.
- Isolated Python environment: `/home/amax/.virtualenvs/serfox-index-estimator-9fc8d76`; it is a venv based on `/home/amax/miniforge3/envs/ml` with system-site packages, Python 3.11.13, PyTorch 2.6.0+cu124, and NumPy 2.1.3.
- Known environment pitfall: `/usr/bin/python3 -m venv` fails because system `ensurepip` is unavailable. For this diagnostic, create the isolated venv with `/home/amax/miniforge3/envs/ml/bin/python -m venv --system-site-packages <venv-path>` instead.
- The diagnostic is CPU-only (`CUDA_VISIBLE_DEVICES=""`, two CPU threads). Its outputs are under the isolated repository's `results/index_weighting_sampling/<run_tag>/` directory.
- Continue syncing code to this server only through Git/GitHub; do not edit the isolated checkout directly.
