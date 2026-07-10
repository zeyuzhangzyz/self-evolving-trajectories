# TODO

## Doing

- None.

## Next

- [ ] Decide whether the next training ablation should compare batch-shared `kk` against per-sample `kk` or direct all-frontier weighting, especially for large `K`.
- [ ] Decide whether to evaluate the Siwei grouped-frontier objective as an ablation against full serialized AR.
- [ ] Before running the mine snapshot, explicitly pair the trainer with the intended model file and fix the `_msk` / `_mask` mismatch.
- [ ] If retaining grouped-frontier training, redesign or disable incompatible soft-index and value-only paths and add shape/position tests.

## Blocked

- None.

## Done

- [x] Ran and analyzed the isolated amax-77 CPU diagnostic at commit `9fc8d76`; direct and enumerated sampled index gradients matched within `3.23e-7` relative L2, while Monte Carlo gradient error followed the expected `N^-1/2` decay.
- [x] Completed a static and minimal-runtime comparison of the four files on 2026-07-10; no training job was launched.
- [x] Added and verified two Mermaid/PNG diagrams explaining the objective-level difference and a concrete `K=4`, `kk=2` frontier example.
