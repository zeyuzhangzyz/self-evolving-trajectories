"""
Old Ser-FOX recompute path excerpt.

Source:
  commit: c9f16e2 Initial commit
  file:   Ser-FOX/serfox_model.py

Purpose:
  This file is not standalone runnable code. It extracts the minimal old-version
  path needed to compare the original PI scorer against the current KV-cache
  implementation.

Read in this order:
  1. build_ste_visible_mask
  2. CausalSelfAttention.forward(..., num_parallel_indices)
  3. GPT.score_parallel_indices
"""


def build_ste_visible_mask(seq_len, num_indices, device):
    """
    Build the Ser-FOX visibility mask for a sequence of the form:

        [serialized_prefix][appended_index_block]

    Visibility rules:
    - tokens inside the serialized prefix use standard causal visibility
    - each appended index token can attend to the full prefix
    - appended index tokens are isolated from one another and can only
      attend to themselves inside the appended block
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
    def forward(self, x, num_parallel_indices=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Standard AR path: use ordinary causal attention.
        if self.flash and num_parallel_indices is None:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if num_parallel_indices is None:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            else:
                # Original PI path: build the full [prefix][candidate block] mask.
                visible = build_ste_visible_mask(T, num_parallel_indices, att.device)
                att = att.masked_fill(~visible.view(1, 1, T, T), float("-inf"))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class GPT:
    def _run_transformer(self, x, num_parallel_indices=None):
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, num_parallel_indices)
        return self.transformer.ln_f(x)

    def score_parallel_indices(self, idx, num_indices):
        """Phase 1: score the appended index block under the Ser-FOX mask."""
        pos = self._build_position_ids(idx)
        _, t = idx.size()
        if num_indices <= 0:
            raise ValueError(f"num_indices must be positive, got {num_indices}")
        if num_indices > t:
            raise ValueError(f"num_indices ({num_indices}) cannot exceed sequence length ({t})")

        prefix_len = t - num_indices

        # All appended index tokens share the same logical frontier position
        # while remaining token-distinct through their learned identities.
        tok_emb = self.transformer.wte(idx)
        prefix_pos_emb = self.transformer.wpe(pos[:, :prefix_len])
        frontier_pos_emb = self.transformer.wpe(
            pos[:, prefix_len:prefix_len + 1]
        ).repeat(1, num_indices, 1)
        pos_emb = torch.cat([prefix_pos_emb, frontier_pos_emb], dim=1)

        # This re-encodes the entire growing prefix plus all candidate indices.
        x = self._run_transformer(tok_emb + pos_emb, num_parallel_indices=num_indices)

        logits = self.lm_head(x[:, -num_indices:, :]).clone()
        logits[..., self.config.index_token_start:] = float("-inf")
        return logits
