# Ser-FOX Attention Mask Equivalence and KV-cache Notes

Date: 2026-07-07
Repository: `self-evolving-trajectories`
Branch checked: `feat/serfox-self-evolving-trajectories`
Key commits checked: `c9f16e2`, `6a4da49`, `d2874fc`

## Slide 1 - Title

Title: Ser-FOX Attention Mask Equivalence

Subtitle: Why the current KV-cache implementation is semantically the same as the original PI scorer, and where the speedup comes from.

Speaker notes:
- The discussion is only about Transformer attention visibility in Ser-FOX PI scoring.
- Decode-time logit masks such as `norepeat` and `pad_eos_last` are separate.
- The short answer is: the PI mask did not change. The speedup comes from not re-encoding the accumulated prefix.

## Slide 2 - Main Claim

Claim:
- Old recompute path and current recompute path are equivalent.
- Current KV-cache path is equivalent to recompute up to floating point roundoff.
- KV-cache does not change PI. It only caches the serialized prefix and appends committed tokens incrementally.

Evidence:
- `old/current recompute max_abs_diff = 0`
- `current recompute/cache max_abs_diff = 4.77e-07`
- finite masks are identical in both comparisons.

## Slide 3 - What PI Does Each Step

At each PI step:

```text
prefix_t = [quiz][I_a, y_a][I_b, y_b]...
candidates = [I_0, I_1, ..., I_{R-1}]

score(prefix_t + candidates)
choose one unresolved position k
commit [I_k, y_k]
prefix_{t+1} = prefix_t + [I_k, y_k]
```

Important:
- All candidate index tokens are scored in parallel.
- Candidate index tokens share the same logical frontier position.
- Candidate index tokens cannot attend to each other.

## Slide 4 - The Original Recompute Path

Original / non-cache logic:

```text
for each step:
    idx_app = concat(prefix_t, all_index_tokens)
    logits = score_parallel_indices(idx_app, R)
    commit one [index, value]
```

Cost profile:
- Re-encodes the whole growing prefix every step.
- Also encodes the full candidate index block every step.
- The prefix part grows as decoding/regeneration progresses.

Code:
- `Ser-FOX/serfox_model.py:728` - `score_parallel_indices`
- `Ser-FOX/serfox_train.py:1607-1613` - branch between recompute and cache.

## Slide 5 - The Current KV-cache Path

Current cache logic:

```text
cache = build_prefix_kv_cache(prefix_0)

for each step:
    logits = score_parallel_indices_cached(all_index_tokens, cache)
    commit one [index, value]
    cache = append_to_kv_cache(cache, [index, value])
```

Cost profile:
- Prefix K/V is built once.
- Each step scores only the candidate index block against cached prefix K/V.
- After committing, only the new two tokens `[index, value]` are appended to the cache.

Code:
- `Ser-FOX/serfox_model.py:613` - `build_prefix_kv_cache`
- `Ser-FOX/serfox_model.py:638` - `append_to_kv_cache`
- `Ser-FOX/serfox_model.py:814` - `score_parallel_indices_cached`
- `Ser-FOX/serfox_model.py:1014` - `generate_parallel_index`
- `Ser-FOX/serfox_train.py:1599` - regeneration cache branch.

## Slide 6 - Same Mask Invariant

For a sequence:

```text
[prefix length P][R candidate index tokens]
```

Visibility matrix:

```text
prefix -> prefix: standard causal lower triangle
candidate i -> prefix: visible
candidate i -> candidate i: visible
candidate i -> candidate j, i != j: hidden
```

Source:
- `Ser-FOX/serfox_model.py:26` - `build_ste_visible_mask`
- `Ser-FOX/serfox_model.py:199` - `forward_new_tokens_with_kv`

Key line:

```python
visible = torch.eye(T, dtype=torch.bool, device=x.device)
```

This is used when `causal_new_tokens=False`, which is exactly the cached PI scoring mode.

## Slide 7 - Equivalence Calculation

Let prefix K/V be:

```text
K_p, V_p = TransformerPrefix(prefix_t)
```

For candidate block tokens:

```text
Q_c, K_c, V_c = Project(candidates)
```

Recompute attention for a candidate row:

```text
softmax([Q_c K_p^T, Q_c K_c^T masked-to-self]) [V_p, V_c]
```

Cached attention computes the same expression:

```text
prefix part: Q_c K_p^T using cached K_p,V_p
candidate part: Q_c K_c^T with identity visibility
concat logits -> same softmax -> same output
```

Therefore:

```text
score_parallel_indices(prefix_t + candidates)
==
score_parallel_indices_cached(candidates, cache(prefix_t))
```

up to floating point roundoff.

## Slide 8 - Why the Speedup Happens

The speedup is not because PI became smaller or less parallel.

It happens because:

```text
old: each step recomputes Transformer(prefix_t + all_candidates)
new: each step computes Transformer(new_candidates | cached prefix)
     then appends only committed [index, value]
```

The accumulated prefix is the repeated work.

Important wording:

> PI still scores all candidate positions. KV-cache removes repeated prefix computation and makes accumulation incremental.

## Slide 9 - Simple Local Validation

Local CPU benchmark configuration:

```text
B=16
layers=3
heads=4
embd=128
quiz=16
response=36
threads=4
```

Numerical results:

```text
old vs current recompute:
  max_abs_diff = 0
  finite masks equal = True

current recompute vs current KV-cache:
  max_abs_diff = 4.77e-07
  finite masks equal = True
```

Interpretation:
- Old and current recompute paths are exactly equal in this sanity test.
- Cache path is equal within normal floating point roundoff.

## Slide 10 - Simple Speed Comparison

CPU timing median over five runs:

```text
old recompute:        1166.1 ms
current recompute:    1168.0 ms
current KV-cache:      937.7 ms
```

Speedups:

```text
current recompute vs old recompute: 1.00x
current KV-cache vs current recompute: 1.25x
current KV-cache vs old recompute: 1.24x
```

Interpretation:
- Recompute did not become faster by itself.
- Cache path is faster because it avoids re-encoding the growing prefix.
- CPU toy results are directional; GPU and larger tasks should be benchmarked separately.

## Slide 11 - What Changed vs What Did Not

Changed:
- Added `build_prefix_kv_cache`.
- Added `append_to_kv_cache`.
- Added `score_parallel_indices_cached`.
- Generation/regeneration can choose cache path.

Not changed:
- The single-index Ser-FOX attention visibility rule.
- The parallel candidate isolation rule.
- The position-choice semantics of PI.
- The value prediction surface for candidate index tokens.

## Slide 12 - Practical Explanation to Use in Discussion

Short phrasing:

> Our current code is an incremental implementation of the same PI scorer. The original code recomputed the entire serialized prefix plus all candidate index tokens at every step. The new code caches the prefix K/V once, scores the same candidate index block against that cache using the same index-isolating visibility mask, and after selecting one position appends only the committed `[index, value]` pair. Therefore the mathematical scorer is the same; the speedup comes from eliminating repeated prefix computation.

Potential caveats:
- `use_rope=True` is a new optional positional encoding path, so it is not a strict old-code comparison.
- `score_parallel_digit_groups` is a new grouped-index feature, not part of the original single-index equivalence claim.
- `norepeat` and `pad_eos_last` are decode-time logit masks, not attention masks.
