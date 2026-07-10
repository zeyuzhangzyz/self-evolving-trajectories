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

- Select the intended model/trainer pairing, add focused interface tests, and define a controlled AR/PI comparison run before launching experiments.
