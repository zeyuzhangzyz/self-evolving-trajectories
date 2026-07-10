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

- First verify on a tiny CPU Ser-FOX model that direct all-index weighting and batch-shared random-frontier sampling have matching expected loss and gradients, while measuring Monte Carlo variance as K grows.
- Then select the intended model/trainer pairing, add focused interface tests, and define a controlled AR/PI comparison run before launching full training experiments.
