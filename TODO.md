# TODO

## Doing

- [ ] Await user confirmation to run the prepared CPU-only direct-weighting versus batch-shared-frontier gradient experiment. Planned launcher: `scripts/experiments/run_compare_index_gradient_estimators.sh`; planned output: `results/index_weighting_sampling/direct_vs_batch_shared_k4_16_81_v1/`.

## Next

- [ ] Analyze `sampling_results.csv` and `exact_expectation_checks.csv` after the confirmed run, reporting mean +/- std for loss error, gradient relative L2, and gradient cosine.
- [ ] Decide whether to evaluate the Siwei grouped-frontier objective as an ablation against full serialized AR.
- [ ] Before running the mine snapshot, explicitly pair the trainer with the intended model file and fix the `_msk` / `_mask` mismatch.
- [ ] If retaining grouped-frontier training, redesign or disable incompatible soft-index and value-only paths and add shape/position tests.

## Blocked

- None.

## Done

- [x] Completed a static and minimal-runtime comparison of the four files on 2026-07-10; no training job was launched.
- [x] Added and verified two Mermaid/PNG diagrams explaining the objective-level difference and a concrete `K=4`, `kk=2` frontier example.
