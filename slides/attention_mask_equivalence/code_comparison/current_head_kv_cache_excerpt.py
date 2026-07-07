"""
Current Ser-FOX KV-cache path excerpt.

Source:
  commit: current branch HEAD when extracted
  file:   Ser-FOX/serfox_model.py
  usage:  Ser-FOX/serfox_train.py

Purpose:
  This file is not standalone runnable code. It extracts the minimal current
  path needed to compare against old_c9f16e2_recompute_excerpt.py.

Read in this order:
  1. build_ste_visible_mask
  2. CausalSelfAttention.forward_new_tokens_with_kv
  3. GPT.build_prefix_kv_cache
  4. GPT.score_parallel_indices_cached
  5. GPT.append_to_kv_cache
  6. regeneration_loop_excerpt
"""


def build_ste_visible_mask(seq_len, num_indices, device):
    """
    Same single-index Ser-FOX visibility mask as the old version:
    prefix is causal; appended candidates see prefix and only themselves.
    """
    if num_indices <= 0:
        raise ValueError(f"num_indices must be positive, got {num_indices}")
    if num_indices > seq_len:
        raise ValueError(f"num_indices ({num_indices}) cannot exceed seq_len ({seq_len})")

    prefix_len = seq_len - num_indices
    visible = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    if prefix_len > 0:
        visible[:prefix_len, :prefix_len] = torch.tril(
            torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=device)
        )
        visible[prefix_len:, :prefix_len] = True

    visible[prefix_len:, prefix_len:] = torch.eye(num_indices, dtype=torch.bool, device=device)
    return visible


class CausalSelfAttention:
    def forward_new_tokens_with_kv(self, x, prefix_k, prefix_v, causal_new_tokens):
        """
        Run attention for newly appended tokens against cached prefix K/V.

        causal_new_tokens=False is the cached PI scorer case:
        new candidate index tokens can see prefix K/V and themselves, but not
        other candidate index tokens.
        """
        B, T, C = x.size()
        q, k, v = self._project_qkv(x)
        scale = 1.0 / math.sqrt(k.size(-1))
        pieces = []
        values = []

        if prefix_k is not None and prefix_k.size(2) > 0:
            pieces.append((q @ prefix_k.transpose(-2, -1)) * scale)
            values.append(prefix_v)

        new_att = (q @ k.transpose(-2, -1)) * scale
        if causal_new_tokens:
            # Used when the selected [index, value] pair becomes real prefix.
            visible = torch.tril(torch.ones((T, T), dtype=torch.bool, device=x.device))
        else:
            # Used by cached PI scoring. This is the candidate isolation rule.
            visible = torch.eye(T, dtype=torch.bool, device=x.device)
        new_att = new_att.masked_fill(~visible.view(1, 1, T, T), float("-inf"))
        pieces.append(new_att)
        values.append(v)

        att = torch.cat(pieces, dim=-1)
        all_v = torch.cat(values, dim=2)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ all_v
        return self._finish_attention(y, B, T, C), k, v


class GPT:
    def build_prefix_kv_cache(self, idx):
        """Build a layer-wise K/V cache for a serialized causal prefix."""
        pos = self._build_position_ids(idx)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        layers = []
        for block in self.transformer.h:
            x, k, v = block.forward_causal_with_kv(x)
            layers.append({"k": k, "v": v})

        return {"layers": layers, "seq_len": idx.size(1)}

    def score_parallel_indices_cached(self, index_tokens, cache):
        """Phase 1 scorer using a cached serialized prefix."""
        num_indices = index_tokens.size(1)
        if num_indices != self.config.num_index_tokens:
            raise ValueError(
                f"Expected {self.config.num_index_tokens} index tokens, got {num_indices}"
            )

        prefix_len = cache["seq_len"]

        # Same shared-frontier position as recompute path, but only for new
        # candidate index tokens. Prefix K/V is already cached.
        pos = torch.full((1, num_indices), prefix_len, dtype=torch.long, device=index_tokens.device)
        tok_emb = self.transformer.wte(index_tokens)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for layer_cache, block in zip(cache["layers"], self.transformer.h):
            x, _, _ = block.forward_new_tokens_with_kv(
                x,
                layer_cache["k"],
                layer_cache["v"],
                causal_new_tokens=False,
            )

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x).clone()
        logits[..., self.config.index_token_start:] = float("-inf")
        return logits

    def append_to_kv_cache(self, cache, idx_new, return_hidden=False):
        """Append committed serialized prefix tokens to an existing causal K/V cache."""
        start = cache["seq_len"]
        end = start + idx_new.size(1)

        pos = torch.arange(start, end, dtype=torch.long, device=idx_new.device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx_new)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for layer_cache, block in zip(cache["layers"], self.transformer.h):
            x, k, v = block.forward_new_tokens_with_kv(
                x,
                layer_cache["k"],
                layer_cache["v"],
                causal_new_tokens=True,
            )
            layer_cache["k"] = torch.cat([layer_cache["k"], k], dim=2)
            layer_cache["v"] = torch.cat([layer_cache["v"], v], dim=2)

        cache["seq_len"] = end
        if return_hidden:
            return self.transformer.ln_f(x), cache
        return cache


def regeneration_loop_excerpt(scoring_model, z_basic, index_tokens, append_tokens):
    """
    Usage pattern from Ser-FOX/serfox_train.py.

    With cache:
      - build prefix cache once
      - score all candidate index tokens against cached prefix
      - append only the selected [index, value] pair
    """
    kv_cache = scoring_model.build_prefix_kv_cache(z_basic)

    index_logits = scoring_model.score_parallel_indices_cached(index_tokens, kv_cache)

    # After choosing a position, the committed pair becomes part of the prefix.
    kv_cache = scoring_model.append_to_kv_cache(kv_cache, append_tokens)
    return index_logits, kv_cache
