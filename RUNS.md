# Runs

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
