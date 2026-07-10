# Runs

## 2026-07-10 14:47 - user_reported_serfox_4-8-512_round4

Status: running (user-reported; four rounds completed)

Task type: training run; exact CPU/GPU allocation pending

Machine: pending user provenance

GPU: pending user provenance

Branch: pending user provenance

Commit: pending user provenance

Script: pending user provenance

Command: pending user provenance

Input: pending user provenance

Output: pending user provenance

Log: pending user provenance

Checkpoint: pending user provenance

Provenance file: pending user provenance

Key config: Ser-FOX backbone `4-8-512`; grouped random-frontier training described as changing one-step `T -> T-1` value supervision into same-prefix supervision of all remaining positions (`T -> 0` shorthand)

Key results:

| Round | AR | PI | PI - AR | AR delta | PI delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.262 | 0.359 | +0.097 | N/A | N/A |
| 2 | 0.461 | 0.530 | +0.069 | +0.199 | +0.171 |
| 3 | 0.512 | 0.556 | +0.044 | +0.051 | +0.026 |
| 4 | 0.549 | 0.594 | +0.045 | +0.037 | +0.038 |

Notes: PI is monotonic and remains above AR through round 4. The user reports that AR 0.549 exceeds the prior PDF maximum for the same model size. DOG PI reference is 0.788, leaving a 0.194 absolute gap. These are single-run observations until seeds and full provenance are available.

Next action: collect later rounds and provenance; run a matched ablation isolating remaining-value coverage, teacherless isolation, shared-frontier positions, and value reweighting.

## 2026-07-10 14:38 - direct_vs_batch_shared_k4_16_81_v1

Status: completed

Task type: cpu_only

Machine: `amax-77`

GPU: N/A; `CUDA_VISIBLE_DEVICES=""`. All eight GPUs remained at 18 MiB and 0% utilization before and after the run.

Branch: detached checkout from `codex/attention-mask-slides`

Commit: `9fc8d76e1f4905c600540b18a5e31e929f104a10`

Script: `scripts/experiments/run_compare_index_gradient_estimators.sh`

Command: `CUDA_VISIBLE_DEVICES='' /home/amax/.virtualenvs/serfox-index-estimator-9fc8d76/bin/python scripts/experiments/compare_index_gradient_estimators.py --k-values 4,16,81 --draw-counts 100,1000,10000,100000 --problem-seeds 3 --sampling-seeds 20 --batch-size 8 --quiz-size 2 --value-vocab-size 11 --n-layer 1 --n-head 1 --n-embd 16 --threads 2 --dtype float32 --device cpu`

Input: synthetic serialized trajectories generated from recorded seeds

Output: `/home/amax/experiments/self-evolving-trajectories-9fc8d76/results/index_weighting_sampling/direct_vs_batch_shared_k4_16_81_v1/`

Log: `/home/amax/experiments/self-evolving-trajectories-9fc8d76/results/index_weighting_sampling/direct_vs_batch_shared_k4_16_81_v1/run.log`

Checkpoint: N/A

Provenance file: `/home/amax/experiments/self-evolving-trajectories-9fc8d76/results/index_weighting_sampling/direct_vs_batch_shared_k4_16_81_v1/provenance.txt`

Key config: tiny Ser-FOX, one layer, one head, embedding size 16, dropout 0, batch size 8, `K=4/16/81`, three problem seeds, twenty sampling seeds, 100 to 100,000 batch-shared frontier draws

Key results:

- Exact enumerated expectation vs direct weighting: maximum loss absolute error `1.1921e-7`; maximum gradient relative L2 `3.2221e-7`.
- At 100 draws, mean gradient relative L2 was `0.1230` (`K=4`), `0.2821` (`K=16`), and `0.9515` (`K=81`).
- At 100,000 draws, mean gradient relative L2 was `0.003309`, `0.009115`, and `0.029580`; corresponding cosine similarities were `0.9999937`, `0.9999578`, and `0.9995613`.
- Gradient-error log-log slopes were approximately `-0.5`, confirming `N^-1/2` convergence.

Notes: isolated repository `/home/amax/experiments/self-evolving-trajectories-9fc8d76`; isolated venv `/home/amax/.virtualenvs/serfox-index-estimator-9fc8d76`; 720 raw rows checked finite; no checkpoint or GPU resource used.

Next action: decide whether to test batch-shared `kk`, per-sample `kk`, and direct all-frontier weighting in a controlled full-training ablation.
