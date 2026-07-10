# Progress

## 2026-07-10 11:54 - Compare Siwei and mine model/trainer snapshots

Status: done

Branch: `codex/attention-mask-slides`

Machine: local Windows workspace (CPU-only diagnostic)

Related files:

- `compare/siwei/serfox_model.py`
- `compare/siwei/serfox_train.py`
- `compare/serfox_model_mine noRope_noGroupIndex.py`
- `compare/serfox_train_mine.py`

Summary:

- Confirmed that both model snapshots have the same parameterized GPT backbone, learned absolute position embeddings, and checkpoint tensor layout.
- Identified the main semantic difference: the Siwei pair samples a random frontier and jointly supervises the next index plus all remaining values, while the mine pair uses full serialized autoregressive training.
- Confirmed with a minimal CPU probe that the mine model's `_get_ste_visible_msk` definition does not match its `_get_ste_visible_mask` call, breaking direct non-cached parallel-index scoring.
- Found that the Siwei compressed batch layout is not aligned with the existing fixed-position soft-index and value-only logic.
- No source code, training configuration, remote state, or formal experiment result was changed.

Next:

- Decide which objective should be tested and fix the selected pair's interface issues before any formal run.

## 2026-07-10 12:43 - Add detailed visual explanation of training objectives

Status: done

Branch: `codex/attention-mask-slides`

Machine: local Windows workspace (CPU-only diagram rendering)

Related files:

- `figures/serfox-training-objective-comparison.mmd`
- `figures/serfox-training-objective-comparison.md`
- `figures/serfox-training-objective-comparison.png`
- `figures/serfox-k4-frontier-example.mmd`
- `figures/serfox-k4-frontier-example.md`
- `figures/serfox-k4-frontier-example.png`

Summary:

- Added an overview diagram showing that both variants share the same parameterized GPT backbone but use different batch contexts and losses.
- Added a concrete `Q=2`, `K=4`, `kk=2` example with a non-canonical trajectory order, explicit hidden-to-target alignment, shared-frontier positions, isolation, and loss weights.
- Rendered both Mermaid sources to PNG with Mermaid CLI and visually reviewed arrow direction, labels, completeness, and readability; both diagrams passed the required review threshold.
- No model, trainer, configuration, dataset, remote state, or formal experiment result was changed.

Next:

- Use the diagrams to choose whether grouped-frontier training should become a controlled ablation or remain separate from the serialized-AR path.

## 2026-07-10 14:23 - Prepare index-gradient estimator experiment

Status: planned

Branch: `codex/attention-mask-slides`

Machine: local Windows workspace; planned CPU-only execution

Related files:

- `scripts/experiments/compare_index_gradient_estimators.py`
- `scripts/experiments/run_compare_index_gradient_estimators.sh`
- `AGENTS.md`
- `ROADMAP.md`
- `TODO.md`

Summary:

- Prepared a tiny-Ser-FOX experiment comparing the direct index objective `1/(2K)` per frontier with the current batch-shared sampled estimator `1/2` on one uniformly sampled frontier.
- The experiment checks exact enumerated loss/gradient expectation and Monte Carlo convergence for `K=4,16,81` using multiple problem and sampling seeds.
- The planned run is CPU-only, disables CUDA, refuses to overwrite results, and records raw tables, summaries, logs, Git/environment provenance, and the original command.
- No experiment has been launched; execution is waiting for the required user confirmation.

Next:

- After confirmation, check CPU load, memory, and disk; run the committed launcher; analyze and record the results.

## 2026-07-10 14:40 - Complete amax-77 index-gradient estimator diagnostic

Status: done

Branch: detached remote checkout from `codex/attention-mask-slides`

Commit: `9fc8d76e1f4905c600540b18a5e31e929f104a10`

Machine: `amax-77`, CPU-only, two threads, `CUDA_VISIBLE_DEVICES=""`

Related files:

- `scripts/experiments/compare_index_gradient_estimators.py`
- `scripts/experiments/run_compare_index_gradient_estimators.sh`
- Remote output: `/home/amax/experiments/self-evolving-trajectories-9fc8d76/results/index_weighting_sampling/direct_vs_batch_shared_k4_16_81_v1/`

Summary:

- Created a dedicated Git checkout and venv on amax-77; no existing Conda environment or project checkout was modified.
- Verified 720 raw Monte Carlo rows are finite and complete across `K=4,16,81`, four draw counts, three problem seeds, and twenty sampling seeds.
- Exact enumeration matched direct weighting within `1.20e-7` absolute loss error and `3.23e-7` relative gradient L2, confirming that the sampled index estimator is unbiased to float32 precision.
- Finite-sample gradient error was material: at 100 draws it was `12.3%`, `28.2%`, and `95.2%` for `K=4,16,81`; at 100,000 draws it fell to `0.331%`, `0.912%`, and `2.96%`.
- Log-log gradient-error slopes were `-0.522`, `-0.496`, and `-0.501`, consistent with Monte Carlo `N^-1/2` convergence.
- Pre/post checks confirmed all GPUs stayed at 0% utilization and 18 MiB idle memory.

Next:

- Treat direct weighting and batch-shared sampling as expectation-equivalent only for the index branch; design a separate full-training ablation for value-context and finite-gradient-variance effects.

## 2026-07-10 14:47 - Analyze user-reported 4-8-512 round metrics

Status: in_progress

Branch: unknown; user-run provenance pending

Machine: unknown; user-reported training run

Related run: `user_reported_serfox_4-8-512_round4`

Summary:

- Recorded four completed round-end accuracies: `(AR, PI)=(0.262,0.359)`, `(0.461,0.530)`, `(0.512,0.556)`, and `(0.549,0.594)`.
- AR improved monotonically by `+0.199`, `+0.051`, and `+0.037`; PI improved by `+0.171`, `+0.026`, and `+0.038`.
- The PI-minus-AR gap remained positive at `+0.097`, `+0.069`, `+0.044`, and `+0.045`, so PI has not shown the previous multi-round collapse through round 4.
- From round 1 to round 4, AR gained `0.287` absolute (`+109.5%` relative) and PI gained `0.235` absolute (`+65.5%` relative).
- Current PI `0.594` reaches `75.4%` of the reported DOG `0.788`, leaving a `0.194` absolute gap; it has closed `54.8%` of the original round-1-to-DOG gap.
- Mechanistic hypothesis: original AR supervises only the diagonal state/candidate value pairs, whereas grouped-frontier training supervises the upper triangle of all remaining candidates under the same prefix. This directly targets the off-policy value/confidence estimates used by PI and trajectory regeneration.
- If the response has `K` positions (`K` is not the 512 embedding size), a sampled frontier supervises `(K+1)/2` remaining values on average. Relative to uniform one-value supervision, the expected weight multiplier for trajectory value `j` is `sum_{n=K-j}^K 1/n`, shifting weight from early/easy values toward late/hard values while keeping total value-loss mass fixed.
- Causality is not established from one run and four rounds; the old-PDF comparison also requires matched seed, data, optimizer, trajectory mix, evaluation, and checkpoint selection.

Next:

- Continue tracking later rounds and run controlled coverage/context ablations with confidence-calibration heatmaps.
