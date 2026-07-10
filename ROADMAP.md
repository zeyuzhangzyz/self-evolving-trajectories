# Roadmap

## Project goal

Develop and validate Ser-FOX self-evolving trajectory training and evaluation while keeping training objectives, decoding behavior, and experiment provenance reproducible.

## Current focus

- Compare serialized autoregressive training with frontier-grouped parallel-value supervision.
- Keep hard-index, soft-index, value-only, and parallel-index paths internally consistent before running controlled experiments.

## Open questions

- Whether the Siwei random-frontier grouped objective should be retained as an ablation or integrated into the main training path.
- How grouped-frontier batches should represent soft-index and value-only targets without fixed-position misalignment.

## Next milestone

- Completed: a tiny CPU Ser-FOX diagnostic verified that direct all-index weighting and batch-shared random-frontier sampling have identical expected index loss and gradients to float32 precision; finite-sample gradient variance grows strongly with K and decays as approximately `N^-1/2`.
- Next, decide whether a controlled training ablation should use direct all-frontier weighting, batch-shared `kk`, or per-sample `kk`, then add focused interface tests and compare full AR/PI behavior.
