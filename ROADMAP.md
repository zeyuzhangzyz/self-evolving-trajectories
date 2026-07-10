# Roadmap

## Project goal

Develop and validate Ser-FOX self-evolving trajectory training and evaluation while keeping training objectives, decoding behavior, and experiment provenance reproducible.

## Current focus

- Compare serialized autoregressive training with frontier-grouped parallel-value supervision.
- Keep hard-index, soft-index, value-only, and parallel-index paths internally consistent before running controlled experiments.

## Open questions

- Whether the Siwei random-frontier grouped objective should be retained as an ablation or integrated into the main training path.
- How grouped-frontier batches should represent soft-index and value-only targets without fixed-position misalignment.
- Whether the observed four-round PI stability comes primarily from expanding value supervision from the diagonal state/candidate pairs to the full remaining-position upper triangle, from teacherless candidate isolation, or from the induced late-position reweighting.

## Next milestone

- Completed: a tiny CPU Ser-FOX diagnostic verified that direct all-index weighting and batch-shared random-frontier sampling have identical expected index loss and gradients to float32 precision; finite-sample gradient variance grows strongly with K and decays as approximately `N^-1/2`.
- Current evidence: a user-reported 4-8-512 grouped-frontier run improved monotonically through round 4 from `(AR, PI)=(0.262,0.359)` to `(0.549,0.594)` without the previous PI collapse.
- Next, run a matched supervision-coverage ablation and measure a prefix-depth × remaining-position value-accuracy/confidence heatmap before attributing the gain to the `T -> all remaining values` change.
