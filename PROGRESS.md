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
