#!/usr/bin/env python3
"""Compare direct index weighting with batch-shared frontier sampling.

The direct index component is

    L_direct = (1 / (2 K)) * sum_k mean_b CE(b, k).

The Siwei-style estimator samples one frontier k for the whole batch and uses

    L_sample(k) = (1 / 2) * mean_b CE(b, k).

This script verifies the exact enumerated expectation and then measures how
quickly Monte Carlo averages of the sampled loss and gradient approach the
direct objective on a tiny Ser-FOX model.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F


def comma_separated_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"Expected positive comma-separated integers, got {value!r}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare direct all-index gradients with batch-shared sampled-frontier gradients."
    )
    parser.add_argument("--k-values", type=comma_separated_ints, default=[4, 16, 81])
    parser.add_argument("--draw-counts", type=comma_separated_ints, default=[100, 1000, 10000, 100000])
    parser.add_argument("--problem-seeds", type=int, default=3)
    parser.add_argument("--sampling-seeds", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--quiz-size", type=int, default=2)
    parser.add_argument("--value-vocab-size", type=int, default=11)
    parser.add_argument("--n-layer", type=int, default=1)
    parser.add_argument("--n-head", type=int, default=1)
    parser.add_argument("--n-embd", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.stdout.strip()


def load_serfox_model(repo_root: Path):
    model_path = repo_root / "Ser-FOX" / "serfox_model.py"
    spec = importlib.util.spec_from_file_location("serfox_model_gradient_experiment", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load model module from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_serialized_batch(
    *,
    batch_size: int,
    quiz_size: int,
    k: int,
    value_vocab_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    prompt = torch.randint(
        value_vocab_size,
        (batch_size, quiz_size),
        generator=generator,
        dtype=torch.long,
    )
    values_by_coordinate = torch.randint(
        value_vocab_size,
        (batch_size, k),
        generator=generator,
        dtype=torch.long,
    )
    permutations = torch.stack([torch.randperm(k, generator=generator) for _ in range(batch_size)])
    index_tokens = permutations + value_vocab_size
    ordered_values = values_by_coordinate.gather(1, permutations)
    pairs = torch.stack((index_tokens, ordered_values), dim=-1).reshape(batch_size, 2 * k)
    z = torch.cat((prompt, pairs), dim=1)
    x = z[:, :-1].contiguous()
    y = z[:, 1:].contiguous()
    index_target_positions = torch.arange(k, dtype=torch.long) * 2 + (quiz_size - 1)
    if not torch.equal(y[:, index_target_positions], index_tokens):
        raise AssertionError("Serialized index targets are not aligned with the expected AR positions")
    return x, y, index_target_positions


def flatten_gradients(
    gradients: Iterable[torch.Tensor | None],
    parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    pieces = []
    for gradient, parameter in zip(gradients, parameters):
        pieces.append(torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1))
    return torch.cat(pieces).detach().cpu()


def gradients_for_problem(
    *,
    model_module,
    args: argparse.Namespace,
    k: int,
    problem_seed: int,
) -> dict:
    torch.manual_seed(problem_seed)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    config = model_module.GPTConfig(
        vocab_size=args.value_vocab_size + k,
        value_vocab_size=args.value_vocab_size,
        block_size=args.quiz_size + 3 * k,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        bias=True,
        quiz_size=args.quiz_size,
        response_size=k,
        use_rope=False,
    )
    model = model_module.GPT(config).to(device=args.device, dtype=dtype)
    model.eval()
    x, y, index_positions = build_serialized_batch(
        batch_size=args.batch_size,
        quiz_size=args.quiz_size,
        k=k,
        value_vocab_size=args.value_vocab_size,
        seed=100_000 + problem_seed,
    )
    x = x.to(args.device)
    y = y.to(args.device)
    index_positions = index_positions.to(args.device)

    logits, _ = model.forward_ar(x)
    index_logits = logits[:, index_positions, :]
    index_targets = y[:, index_positions]
    loss_matrix = F.cross_entropy(
        index_logits.reshape(-1, index_logits.size(-1)),
        index_targets.reshape(-1),
        reduction="none",
    ).reshape(args.batch_size, k)

    # The index component occupies half of the full index+value token objective.
    direct_loss = 0.5 * loss_matrix.mean()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    direct_gradient = flatten_gradients(
        torch.autograd.grad(direct_loss, parameters, retain_graph=True, allow_unused=True),
        parameters,
    )

    frontier_losses = []
    frontier_gradients = []
    for frontier in range(k):
        sampled_loss = 0.5 * loss_matrix[:, frontier].mean()
        gradients = torch.autograd.grad(
            sampled_loss,
            parameters,
            retain_graph=frontier < k - 1,
            allow_unused=True,
        )
        frontier_losses.append(float(sampled_loss.detach().cpu()))
        frontier_gradients.append(flatten_gradients(gradients, parameters))

    frontier_losses_tensor = torch.tensor(frontier_losses, dtype=torch.float64)
    frontier_gradients_tensor = torch.stack(frontier_gradients).to(torch.float64)
    direct_gradient64 = direct_gradient.to(torch.float64)
    exact_loss = float(frontier_losses_tensor.mean())
    exact_gradient = frontier_gradients_tensor.mean(dim=0)
    gradient_norm = float(torch.linalg.vector_norm(direct_gradient64))
    exact_gradient_error = float(
        torch.linalg.vector_norm(exact_gradient - direct_gradient64) / max(gradient_norm, 1e-30)
    )

    return {
        "direct_loss": float(direct_loss.detach().cpu()),
        "direct_gradient": direct_gradient64,
        "frontier_losses": frontier_losses_tensor,
        "frontier_gradients": frontier_gradients_tensor,
        "gradient_norm": gradient_norm,
        "exact_loss_abs_error": abs(exact_loss - float(direct_loss.detach().cpu())),
        "exact_gradient_rel_l2": exact_gradient_error,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "sequence_length": x.size(1),
    }


def monte_carlo_rows(
    *,
    problem: dict,
    k: int,
    problem_seed: int,
    draw_counts: list[int],
    sampling_seeds: int,
) -> list[dict]:
    rows = []
    direct_loss = problem["direct_loss"]
    direct_gradient = problem["direct_gradient"]
    direct_gradient_norm = problem["gradient_norm"]
    probabilities = np.full(k, 1.0 / k, dtype=np.float64)

    for draws in draw_counts:
        for sampling_seed in range(sampling_seeds):
            rng_seed = 1_000_000 * problem_seed + 10_000 * k + 100 * draws + sampling_seed
            rng = np.random.default_rng(rng_seed)
            counts = rng.multinomial(draws, probabilities)
            empirical = torch.from_numpy(counts.astype(np.float64) / draws)
            estimated_loss = float(empirical @ problem["frontier_losses"])
            estimated_gradient = empirical @ problem["frontier_gradients"]
            gradient_error = torch.linalg.vector_norm(estimated_gradient - direct_gradient)
            grad_rel_l2 = float(gradient_error / max(direct_gradient_norm, 1e-30))
            denominator = float(
                torch.linalg.vector_norm(estimated_gradient) * torch.linalg.vector_norm(direct_gradient)
            )
            grad_cosine = (
                float(torch.dot(estimated_gradient, direct_gradient) / denominator)
                if denominator > 0
                else float("nan")
            )
            exact_coefficient = 1.0 / (2.0 * k)
            empirical_coefficients = counts.astype(np.float64) / (2.0 * draws)
            rows.append(
                {
                    "k": k,
                    "problem_seed": problem_seed,
                    "draws": draws,
                    "sampling_seed": sampling_seed,
                    "direct_loss": direct_loss,
                    "estimated_loss": estimated_loss,
                    "loss_abs_error": abs(estimated_loss - direct_loss),
                    "loss_rel_error": abs(estimated_loss - direct_loss) / max(abs(direct_loss), 1e-30),
                    "grad_rel_l2": grad_rel_l2,
                    "grad_cosine": grad_cosine,
                    "coefficient_linf": float(np.max(np.abs(empirical_coefficients - exact_coefficient))),
                    "min_frontier_count": int(counts.min()),
                    "max_frontier_count": int(counts.max()),
                }
            )
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["k"], row["draws"])].append(row)

    summaries = []
    for (k, draws), group in sorted(grouped.items()):
        record = {"k": k, "draws": draws, "replicates": len(group)}
        for metric in ("loss_abs_error", "loss_rel_error", "grad_rel_l2", "grad_cosine", "coefficient_linf"):
            mean, std = mean_std([float(row[metric]) for row in group])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
        summaries.append(record)
    return summaries


def coefficient_theory(k_values: list[int], batch_size: int) -> list[dict]:
    records = []
    for k in k_values:
        mean = 1.0 / (2.0 * k)
        shared_variance = (k - 1.0) / (4.0 * k * k)
        records.append(
            {
                "k": k,
                "expected_per_index_weight": mean,
                "batch_shared_variance_per_draw": shared_variance,
                "batch_shared_cv": math.sqrt(k - 1.0),
                "per_sample_independent_variance_of_batch_mean": shared_variance / batch_size,
                "per_sample_independent_cv_of_batch_mean": math.sqrt((k - 1.0) / batch_size),
            }
        )
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(
    path: Path,
    exact_checks: list[dict],
    summaries: list[dict],
    theory: list[dict],
) -> None:
    lines = [
        "# Direct Weighting vs Batch-Shared Sampling",
        "",
        "## Exact enumerated expectation",
        "",
        "| K | Problem seed | Direct loss | Exact loss abs. error | Exact gradient rel. L2 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in exact_checks:
        lines.append(
            f"| {row['k']} | {row['problem_seed']} | {row['direct_loss']:.8g} | "
            f"{row['exact_loss_abs_error']:.3e} | {row['exact_gradient_rel_l2']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Monte Carlo convergence",
            "",
            "| K | Draws | Replicates | Loss rel. error | Gradient rel. L2 | Gradient cosine | Coefficient L-inf |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['k']} | {row['draws']} | {row['replicates']} | "
            f"{row['loss_rel_error_mean']:.3e} +/- {row['loss_rel_error_std']:.3e} | "
            f"{row['grad_rel_l2_mean']:.3e} +/- {row['grad_rel_l2_std']:.3e} | "
            f"{row['grad_cosine_mean']:.8f} +/- {row['grad_cosine_std']:.3e} | "
            f"{row['coefficient_linf_mean']:.3e} +/- {row['coefficient_linf_std']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Frontier coefficient theory",
            "",
            "| K | Expected weight per index | Shared-k CV | Per-sample-k batch-mean CV |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in theory:
        lines.append(
            f"| {row['k']} | {row['expected_per_index_weight']:.8g} | "
            f"{row['batch_shared_cv']:.6g} | "
            f"{row['per_sample_independent_cv_of_batch_mean']:.6g} |"
        )
    lines.extend(
        [
            "",
            "The current Siwei implementation samples one frontier for the whole batch. "
            "If each sample instead drew its own frontier independently, the frontier-selection "
            "coefficient variance of a batch mean would be reduced by approximately the batch size.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [entry for entry in path.iterdir() if entry.name != "run.log"]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use a new RUN_TAG or pass --overwrite explicitly."
        )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config = {
        "k_values": args.k_values,
        "draw_counts": args.draw_counts,
        "problem_seeds": args.problem_seeds,
        "sampling_seeds": args.sampling_seeds,
        "batch_size": args.batch_size,
        "quiz_size": args.quiz_size,
        "value_vocab_size": args.value_vocab_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "threads": args.threads,
        "dtype": args.dtype,
        "device": args.device,
        "output_dir": str(args.output_dir.resolve()),
    }
    if args.problem_seeds < 1 or args.sampling_seeds < 1:
        raise ValueError("problem_seeds and sampling_seeds must both be >= 1")
    if args.batch_size < 1 or args.quiz_size < 1 or args.threads < 1:
        raise ValueError("batch_size, quiz_size, and threads must all be >= 1")
    if args.n_embd % args.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return

    prepare_output_dir(args.output_dir, args.overwrite)
    torch.set_num_threads(args.threads)
    model_module = load_serfox_model(repo_root)
    all_rows: list[dict] = []
    exact_checks: list[dict] = []

    for k in args.k_values:
        for problem_seed in range(args.problem_seeds):
            print(f"Computing exact frontier gradients: K={k}, problem_seed={problem_seed}", flush=True)
            problem = gradients_for_problem(
                model_module=model_module,
                args=args,
                k=k,
                problem_seed=problem_seed,
            )
            exact_checks.append(
                {
                    "k": k,
                    "problem_seed": problem_seed,
                    "direct_loss": problem["direct_loss"],
                    "exact_loss_abs_error": problem["exact_loss_abs_error"],
                    "exact_gradient_rel_l2": problem["exact_gradient_rel_l2"],
                    "parameter_count": problem["parameter_count"],
                    "sequence_length": problem["sequence_length"],
                }
            )
            all_rows.extend(
                monte_carlo_rows(
                    problem=problem,
                    k=k,
                    problem_seed=problem_seed,
                    draw_counts=args.draw_counts,
                    sampling_seeds=args.sampling_seeds,
                )
            )

    summaries = aggregate_rows(all_rows)
    theory = coefficient_theory(args.k_values, args.batch_size)
    write_csv(args.output_dir / "sampling_results.csv", all_rows)
    write_csv(args.output_dir / "exact_expectation_checks.csv", exact_checks)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": config,
                "exact_checks": exact_checks,
                "summaries": summaries,
                "coefficient_theory": theory,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary_markdown(args.output_dir / "summary.md", exact_checks, summaries, theory)

    command = os.environ.get("RUN_COMMAND_ORIGINAL") or " ".join(
        shlex.quote(part) for part in [sys.executable, *sys.argv]
    )
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_type": "cpu_only",
        "machine": platform.node(),
        "gpu": "N/A",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "script": str(Path(__file__).resolve()),
        "command": command,
        "input": "synthetic serialized trajectories generated from recorded seeds",
        "output": str(args.output_dir.resolve()),
        "checkpoint": "N/A",
        "git_branch": run_git(repo_root, "branch", "--show-current"),
        "git_commit": run_git(repo_root, "rev-parse", "HEAD"),
        "git_status_porcelain": run_git(repo_root, "status", "--porcelain"),
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "config": config,
    }
    (args.output_dir / "provenance.txt").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
