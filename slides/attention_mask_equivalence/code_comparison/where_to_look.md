# Where to Look for the Code Comparison

This directory extracts the two paths you should compare side by side:

- `old_c9f16e2_recompute_excerpt.py`
- `current_head_kv_cache_excerpt.py`

The excerpts are intentionally not standalone runnable files. They are focused reading aids for the attention-mask/KV-cache discussion.

## Recommended Reading Order

1. Start with the old path:
   - `old_c9f16e2_recompute_excerpt.py`
   - Read `build_ste_visible_mask`.
   - Then read `CausalSelfAttention.forward(..., num_parallel_indices)`.
   - Then read `GPT.score_parallel_indices`.

2. Then read the current path:
   - `current_head_kv_cache_excerpt.py`
   - Confirm `build_ste_visible_mask` is the same single-index visibility rule.
   - Read `forward_new_tokens_with_kv`.
   - Pay attention to `causal_new_tokens=False`:
     `visible = torch.eye(T)` is the cached PI candidate-isolation mask.
   - Read `build_prefix_kv_cache`.
   - Read `score_parallel_indices_cached`.
   - Read `append_to_kv_cache`.

3. Finally compare usage:
   - Old: each PI step calls `score_parallel_indices(prefix + all_index_tokens, R)`.
   - Current: build prefix cache once, call `score_parallel_indices_cached(all_index_tokens, cache)`, then append only selected `[index, value]`.

## Original Source References

Old version:

- Commit: `c9f16e2`
- File: `Ser-FOX/serfox_model.py`
- Key lines:
  - `build_ste_visible_mask`: old lines 42-71
  - `CausalSelfAttention.forward`: old lines 116-154
  - `_run_transformer`: old lines 304-308
  - `score_parallel_indices`: old lines 324-348

Current version:

- Branch: `codex/attention-mask-slides`
- File: `Ser-FOX/serfox_model.py`
- Key lines:
  - `build_ste_visible_mask`: lines 26-55
  - `forward_new_tokens_with_kv`: lines 199-232
  - `build_prefix_kv_cache`: lines 613-636
  - `append_to_kv_cache`: lines 638-681
  - `score_parallel_indices`: lines 728-764
  - `score_parallel_indices_cached`: lines 814-853
  - `generate_parallel_index`: lines 1014-1080

Current training usage:

- File: `Ser-FOX/serfox_train.py`
- Key lines:
  - build cache: line 1602
  - recompute branch: lines 1607-1611
  - cached branch: line 1613
  - append committed pair: line 1651

## Core Comparison Point

Old recompute path computes:

```text
score_parallel_indices(prefix_t + all_index_tokens, R)
```

Current cached path computes:

```text
score_parallel_indices_cached(all_index_tokens, cache(prefix_t))
```

Both use the same candidate visibility semantics:

```text
candidate i sees prefix
candidate i sees itself
candidate i does not see candidate j where i != j
```

The speedup comes from avoiding repeated encoding of `prefix_t`; it does not come from changing PI semantics.
