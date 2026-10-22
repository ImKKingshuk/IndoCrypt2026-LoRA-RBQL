from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from lowrank_extraction.endpoint_observability import (
    invert_endpoint_hidden_states,
    invert_hidden_states_from_log_probs,
    invert_hidden_states_from_shifted_logits,
    unbiased_log_probs_from_biased_token_log_probs,
)
from lowrank_extraction.metrics import relative_frobenius_error


DEFAULT_PROMPTS = [
    "Cryptographic protocols often expose only carefully restricted oracle outputs.",
    "A low rank adapter can concentrate task specific behavior in a small subspace.",
    "The reviewer asked whether endpoint logits reveal hidden representations.",
    "Language model APIs sometimes return token log probabilities for debugging.",
    "The public output embedding maps hidden states to vocabulary logits.",
    "A defender can remove scores while keeping the predicted token unchanged.",
    "Matrix rank determines whether a public linear head can be inverted.",
    "Chosen queries are a standard tool in cryptanalytic model extraction.",
    "The theorem separates observable recovery from quotient non-identifiability.",
    "Top k censoring hides most token scores in a distributional API.",
    "Logit bias can force a requested token into the returned candidate set.",
    "Adapter extraction should distinguish parameters from functional behavior.",
    "Endpoint recovery is possible only when enough linear signal is exposed.",
    "The artifact records query counts and numerical reconstruction errors.",
    "Transformer symmetries create equivalent parameterizations of the same map.",
    "A final vocabulary head can have far more rows than hidden dimensions.",
    "The attack reconstructs a hidden representation before solving the adapter.",
    "Hard labels remove scale information from the underlying score function.",
    "Noise and rounding can preserve utility while degrading parameter recovery.",
    "The experiment uses public base features and an unknown low rank update.",
    "Reviewers expect clear limitations and reproducible evidence.",
    "A full vocabulary endpoint is different from a two class classifier head.",
    "Rank deficient heads expose only a quotient of the hidden state.",
    "The strongest claim should match the actual oracle model.",
    "Public model weights make the base hidden representation available.",
    "An adapted hidden state changes the next token logit vector.",
    "The recovery algorithm solves a linear least squares system.",
    "The low rank update is evaluated on held out prompts.",
    "A top one log probability interface can be probed token by token.",
    "The query transcript contains enough equations only under observability.",
    "We test exact logits and forceable restricted log probabilities.",
    "The paper should not claim arbitrary hidden transformer recovery.",
    "A language modeling head provides a realistic full rank output map.",
    "The adapter acts on a public low dimensional projection of hidden states.",
    "Endpoint inversion converts output probabilities into representation access.",
    "A stable local block can then be treated as an affine map.",
    "Exact recovery fails when the output map has a nontrivial kernel.",
    "The defense sweep measures utility against parameter leakage.",
    "Synthetic claims need at least one real pretrained model sanity check.",
    "The old title is defensible only with explicit recovery boundaries.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover a low-rank final-representation adapter through a real full-vocabulary LM endpoint head."
    )
    parser.add_argument("--outdir", type=Path, default=Path("results") / "lm_endpoint_inversion")
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--adapter-feature-dim", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument("--rank-tol", type=float, default=1e-3)
    parser.add_argument("--query-prompts", type=int, default=28)
    parser.add_argument("--eval-prompts", type=int, default=12)
    parser.add_argument("--adapter-scale", type=float, default=0.03)
    parser.add_argument("--bias-strength", type=float, default=8.0)
    parser.add_argument("--index-bias-range", type=float, default=80.0)
    parser.add_argument("--index-binary-steps", type=int, default=24)
    parser.add_argument("--row-budgets", type=int, nargs="+", default=[128, 256, 512, 768, 769, 1024])
    parser.add_argument("--max-length", type=int, default=64)
    return parser.parse_args()


def _require_deps() -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "The LM endpoint inversion experiment needs optional heavy dependencies. "
            "Install them with: uv sync --extra dev --extra lora"
        ) from exc
    return {"torch": torch, "AutoModelForCausalLM": AutoModelForCausalLM, "AutoTokenizer": AutoTokenizer}


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def _load_lm_features(
    *,
    deps: dict[str, Any],
    model_name: str,
    prompts: list[str],
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    torch = deps["torch"]
    tokenizer = deps["AutoTokenizer"].from_pretrained(model_name)
    model = deps["AutoModelForCausalLM"].from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True, return_dict=True)
    lengths = enc["attention_mask"].sum(dim=1).to(torch.long) - 1
    hidden = outputs.hidden_states[-1][torch.arange(len(prompts)), lengths].detach().cpu().numpy().astype(np.float64)
    weight = model.get_output_embeddings().weight.detach().cpu().numpy().astype(np.float64)
    return hidden, weight, model.config._name_or_path


def _make_adapter(
    *,
    hidden_dim: int,
    adapter_feature_dim: int,
    rank: int,
    adapter_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if adapter_feature_dim >= hidden_dim:
        raise ValueError("adapter_feature_dim must be smaller than the hidden dimension")
    if rank > min(hidden_dim, adapter_feature_dim):
        raise ValueError("rank must not exceed min(hidden_dim, adapter_feature_dim)")
    rng = np.random.default_rng(seed)
    raw_projection = rng.normal(0.0, 1.0, size=(hidden_dim, adapter_feature_dim))
    projection, _ = np.linalg.qr(raw_projection)
    adapter_left = rng.normal(0.0, adapter_scale / np.sqrt(rank), size=(hidden_dim, rank))
    adapter_right = rng.normal(0.0, adapter_scale / np.sqrt(rank), size=(adapter_feature_dim, rank))
    delta = adapter_left @ adapter_right.T
    return projection.T.astype(np.float64), delta.astype(np.float64)


def _head_spectrum(weight: np.ndarray, *, augmented: bool = False) -> dict[str, float | int]:
    matrix = np.column_stack([weight, -np.ones(weight.shape[0], dtype=np.float64)]) if augmented else weight
    eigvals = np.linalg.eigvalsh(matrix.T @ matrix)
    positive = eigvals[eigvals > 1e-10]
    rank = int(positive.size)
    min_singular = float(np.sqrt(np.min(positive))) if positive.size else 0.0
    max_singular = float(np.sqrt(np.max(positive))) if positive.size else 0.0
    prefix = "softmax_augmented" if augmented else "output_head"
    return {
        f"{prefix}_rank": rank,
        f"{prefix}_min_singular": min_singular,
        f"{prefix}_condition_number": max_singular / max(min_singular, 1e-300),
    }


def _recover_delta(public_features: np.ndarray, base_hidden: np.ndarray, recovered_adapted_hidden: np.ndarray) -> np.ndarray:
    update = recovered_adapted_hidden - base_hidden
    theta, *_ = np.linalg.lstsq(public_features, update, rcond=None)
    return theta.T


def _rank_diagnostics(
    recovered_delta: np.ndarray,
    true_delta: np.ndarray,
    *,
    max_rank: int,
    rank_tol: float,
) -> dict[str, float | int]:
    if max_rank < 1:
        raise ValueError("max_rank must be positive")
    if rank_tol <= 0:
        raise ValueError("rank_tol must be positive")
    u, singular_values, vt = np.linalg.svd(recovered_delta, full_matrices=False)
    if not singular_values.size:
        return {
            "rank_estimate": 0,
            "rank_spectral_gap": float("inf"),
            "rank_projected_delta_relerr": 1.0,
        }
    threshold = max(float(singular_values[0]) * rank_tol, 1e-12)
    estimated_rank = int(np.sum(singular_values > threshold))
    estimated_rank = min(estimated_rank, max_rank, singular_values.size)
    if estimated_rank == 0:
        projected = np.zeros_like(recovered_delta)
        spectral_gap = float("inf")
    else:
        projected = (u[:, :estimated_rank] * singular_values[:estimated_rank]) @ vt[:estimated_rank]
        next_singular = float(singular_values[estimated_rank]) if estimated_rank < singular_values.size else 0.0
        spectral_gap = float(singular_values[estimated_rank - 1] / max(next_singular, 1e-300))
    return {
        "rank_estimate": int(estimated_rank),
        "rank_spectral_gap": spectral_gap,
        "rank_projected_delta_relerr": relative_frobenius_error(projected, true_delta),
    }


def _forced_top1_biased_token_log_probs(
    *,
    logits: np.ndarray,
    bias_strength: float,
) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    total = np.sum(exp_shifted, axis=1)
    boosted = np.exp(bias_strength)
    observed = np.empty_like(logits)
    for token_idx in range(logits.shape[1]):
        token_exp = exp_shifted[:, token_idx]
        biased_denominator = total + (boosted - 1.0) * token_exp
        observed[:, token_idx] = shifted[:, token_idx] + bias_strength - np.log(biased_denominator)
    return observed


def _forced_top1_biased_token_log_probs_subset(
    *,
    logits: np.ndarray,
    indices: np.ndarray,
    bias_strength: float,
) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    total = np.sum(exp_shifted, axis=1)
    boosted = np.exp(bias_strength)
    observed = np.empty((logits.shape[0], indices.size), dtype=np.float64)
    for output_col, token_idx in enumerate(indices):
        token_exp = exp_shifted[:, token_idx]
        biased_denominator = total + (boosted - 1.0) * token_exp
        observed[:, output_col] = shifted[:, token_idx] + bias_strength - np.log(biased_denominator)
    return observed


def _top1_bias_hidden_recovery(
    *,
    output_weight: np.ndarray,
    logits: np.ndarray,
    bias_strength: float,
) -> np.ndarray:
    biased_token_log_probs = _forced_top1_biased_token_log_probs(logits=logits, bias_strength=bias_strength)
    recovered_log_probs = unbiased_log_probs_from_biased_token_log_probs(
        biased_token_log_probs,
        bias=bias_strength,
    )
    return invert_hidden_states_from_log_probs(
        output_weight,
        np.zeros(output_weight.shape[0], dtype=np.float64),
        recovered_log_probs,
    )


def _top1_bias_subset_hidden_recovery(
    *,
    output_weight: np.ndarray,
    logits: np.ndarray,
    indices: np.ndarray,
    bias_strength: float,
) -> np.ndarray:
    biased_token_log_probs = _forced_top1_biased_token_log_probs_subset(
        logits=logits,
        indices=indices,
        bias_strength=bias_strength,
    )
    recovered_log_probs = unbiased_log_probs_from_biased_token_log_probs(
        biased_token_log_probs,
        bias=bias_strength,
    )
    return invert_hidden_states_from_log_probs(
        output_weight[indices],
        np.zeros(indices.size, dtype=np.float64),
        recovered_log_probs,
    )


def _top1_index_bias_shifted_logits(
    *,
    logits: np.ndarray,
    bias_range: float,
    binary_steps: int,
) -> tuple[np.ndarray, int, float]:
    if bias_range <= 0:
        raise ValueError("bias_range must be positive")
    if binary_steps <= 0:
        raise ValueError("binary_steps must be positive")
    base_indices = np.argmax(logits, axis=1)
    rows = np.arange(logits.shape[0])
    base_logits = logits[rows, base_indices]
    true_gaps = base_logits[:, None] - logits
    recoverable = true_gaps <= bias_range
    low = np.zeros_like(logits)
    high = np.full_like(logits, bias_range)
    for _ in range(binary_steps):
        mid = 0.5 * (low + high)
        forced_top1 = logits + mid >= base_logits[:, None]
        high = np.where(forced_top1, mid, high)
        low = np.where(forced_top1, low, mid)
    thresholds = high
    thresholds[rows, base_indices] = 0.0
    shifted_logits = -thresholds
    endpoint_calls = int(logits.shape[0] * (logits.shape[1] - 1) * binary_steps)
    uncovered_fraction = float(1.0 - np.mean(recoverable))
    return shifted_logits, endpoint_calls, uncovered_fraction


def _top1_index_bias_shifted_logits_subset(
    *,
    logits: np.ndarray,
    indices: np.ndarray,
    bias_range: float,
    binary_steps: int,
) -> tuple[np.ndarray, int, float]:
    if bias_range <= 0:
        raise ValueError("bias_range must be positive")
    if binary_steps <= 0:
        raise ValueError("binary_steps must be positive")
    base_logits = np.max(logits, axis=1)
    true_gaps = base_logits[:, None] - logits[:, indices]
    recoverable = true_gaps <= bias_range
    low = np.zeros_like(true_gaps)
    high = np.full_like(true_gaps, bias_range)
    selected_logits = logits[:, indices]
    for _ in range(binary_steps):
        mid = 0.5 * (low + high)
        forced_top1 = selected_logits + mid >= base_logits[:, None]
        high = np.where(forced_top1, mid, high)
        low = np.where(forced_top1, low, mid)
    endpoint_calls = int(logits.shape[0] * indices.size * binary_steps)
    uncovered_fraction = float(1.0 - np.mean(recoverable))
    return -high, endpoint_calls, uncovered_fraction


def _top1_index_bias_hidden_recovery(
    *,
    output_weight: np.ndarray,
    logits: np.ndarray,
    bias_range: float,
    binary_steps: int,
) -> tuple[np.ndarray, int, float]:
    shifted_logits, endpoint_calls, uncovered_fraction = _top1_index_bias_shifted_logits(
        logits=logits,
        bias_range=bias_range,
        binary_steps=binary_steps,
    )
    recovered_hidden = invert_hidden_states_from_shifted_logits(
        output_weight,
        np.zeros(output_weight.shape[0], dtype=np.float64),
        shifted_logits,
    )
    return recovered_hidden, endpoint_calls, uncovered_fraction


def _top1_index_bias_subset_hidden_recovery(
    *,
    output_weight: np.ndarray,
    logits: np.ndarray,
    indices: np.ndarray,
    bias_range: float,
    binary_steps: int,
) -> tuple[np.ndarray, int, float]:
    shifted_logits, endpoint_calls, uncovered_fraction = _top1_index_bias_shifted_logits_subset(
        logits=logits,
        indices=indices,
        bias_range=bias_range,
        binary_steps=binary_steps,
    )
    recovered_hidden = invert_hidden_states_from_shifted_logits(
        output_weight[indices],
        np.zeros(indices.size, dtype=np.float64),
        shifted_logits,
    )
    return recovered_hidden, endpoint_calls, uncovered_fraction


def _condition_number(matrix: np.ndarray, *, tol: float = 1e-10) -> float:
    eigvals = np.linalg.eigvalsh(matrix.T @ matrix)
    positive = eigvals[eigvals > tol]
    if not positive.size:
        return float("inf")
    return float(np.sqrt(positive[-1]) / max(float(np.sqrt(positive[0])), 1e-300))


def _pivot_rows(matrix: np.ndarray, *, rows: int) -> np.ndarray:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if rows > matrix.shape[0]:
        raise ValueError("cannot select more rows than the matrix has")
    try:
        from scipy.linalg import qr
    except ImportError as exc:  # pragma: no cover - scipy is installed through scikit-learn.
        raise RuntimeError("adaptive coordinate selection requires scipy.linalg.qr") from exc
    _, _, pivots = qr(matrix.T, pivoting=True, mode="economic")
    return np.asarray(pivots[:rows], dtype=np.int64)


def _adaptive_coordinate_recovery(
    *,
    output_weight: np.ndarray,
    logits: np.ndarray,
    true_hidden: np.ndarray,
    base_hidden: np.ndarray,
    public_features: np.ndarray,
    delta: np.ndarray,
    bias_strength: float,
    bias_range: float,
    binary_steps: int,
) -> dict[str, float | int]:
    hidden_dim = output_weight.shape[1]
    exact_indices = _pivot_rows(output_weight, rows=hidden_dim)
    shifted_matrix = np.column_stack([output_weight, -np.ones(output_weight.shape[0], dtype=np.float64)])
    shifted_indices = _pivot_rows(shifted_matrix, rows=hidden_dim + 1)

    recovered_exact = invert_endpoint_hidden_states(
        output_weight[exact_indices],
        np.zeros(exact_indices.size, dtype=np.float64),
        logits[:, exact_indices],
    )
    recovered_top1 = _top1_bias_subset_hidden_recovery(
        output_weight=output_weight,
        logits=logits,
        indices=shifted_indices,
        bias_strength=bias_strength,
    )
    recovered_index, index_calls, index_uncovered = _top1_index_bias_subset_hidden_recovery(
        output_weight=output_weight,
        logits=logits,
        indices=shifted_indices,
        bias_range=bias_range,
        binary_steps=binary_steps,
    )
    exact_delta = _recover_delta(public_features, base_hidden, recovered_exact)
    top1_delta = _recover_delta(public_features, base_hidden, recovered_top1)
    index_delta = _recover_delta(public_features, base_hidden, recovered_index)
    return {
        "adaptive_exact_rows": int(exact_indices.size),
        "adaptive_shifted_rows": int(shifted_indices.size),
        "adaptive_exact_condition": _condition_number(output_weight[exact_indices]),
        "adaptive_shifted_condition": _condition_number(shifted_matrix[shifted_indices]),
        "adaptive_exact_hidden_relerr": relative_frobenius_error(recovered_exact, true_hidden),
        "adaptive_top1_bias_hidden_relerr": relative_frobenius_error(recovered_top1, true_hidden),
        "adaptive_top1_index_bias_hidden_relerr": relative_frobenius_error(recovered_index, true_hidden),
        "adaptive_exact_delta_relerr": relative_frobenius_error(exact_delta, delta),
        "adaptive_top1_bias_delta_relerr": relative_frobenius_error(top1_delta, delta),
        "adaptive_top1_index_bias_delta_relerr": relative_frobenius_error(index_delta, delta),
        "adaptive_exact_endpoint_calls": int(logits.shape[0] * exact_indices.size),
        "adaptive_top1_bias_endpoint_calls": int(logits.shape[0] * shifted_indices.size),
        "adaptive_top1_index_bias_endpoint_calls": int(index_calls),
        "adaptive_top1_index_bias_uncovered_fraction": index_uncovered,
    }


def _coordinate_exposure_sweep(
    *,
    output_weight: np.ndarray,
    logits: np.ndarray,
    true_hidden: np.ndarray,
    row_budgets: list[int],
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    vocab_size, hidden_dim = output_weight.shape
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    result: dict[str, float | int] = {
        "coordinate_exact_min_full_rank_budget": 0,
        "coordinate_shifted_min_full_rank_budget": 0,
        "coordinate_exact_hidden_relerr_at_min": float("nan"),
        "coordinate_shifted_hidden_relerr_at_min": float("nan"),
        "coordinate_exact_condition_at_min": float("nan"),
        "coordinate_shifted_condition_at_min": float("nan"),
    }
    for budget in sorted({budget for budget in row_budgets if 0 < budget <= vocab_size}):
        indices = rng.choice(vocab_size, size=budget, replace=False)
        subset_weight = output_weight[indices]
        exact_eigvals = np.linalg.eigvalsh(subset_weight.T @ subset_weight)
        exact_positive = exact_eigvals[exact_eigvals > 1e-10]
        exact_rank = int(exact_positive.size)
        if exact_rank >= hidden_dim and int(result["coordinate_exact_min_full_rank_budget"]) == 0:
            recovered = invert_endpoint_hidden_states(
                subset_weight,
                np.zeros(budget, dtype=np.float64),
                logits[:, indices],
            )
            result["coordinate_exact_min_full_rank_budget"] = int(budget)
            result["coordinate_exact_hidden_relerr_at_min"] = relative_frobenius_error(recovered, true_hidden)
            result["coordinate_exact_condition_at_min"] = float(np.sqrt(exact_positive[-1]) / max(float(np.sqrt(exact_positive[0])), 1e-300))

        augmented = np.column_stack([subset_weight, -np.ones(budget, dtype=np.float64)])
        shifted_eigvals = np.linalg.eigvalsh(augmented.T @ augmented)
        shifted_positive = shifted_eigvals[shifted_eigvals > 1e-10]
        shifted_rank = int(shifted_positive.size)
        if shifted_rank >= hidden_dim + 1 and int(result["coordinate_shifted_min_full_rank_budget"]) == 0:
            recovered = invert_hidden_states_from_shifted_logits(
                subset_weight,
                np.zeros(budget, dtype=np.float64),
                shifted_logits[:, indices],
            )
            result["coordinate_shifted_min_full_rank_budget"] = int(budget)
            result["coordinate_shifted_hidden_relerr_at_min"] = relative_frobenius_error(recovered, true_hidden)
            result["coordinate_shifted_condition_at_min"] = float(
                np.sqrt(shifted_positive[-1]) / max(float(np.sqrt(shifted_positive[0])), 1e-300)
            )
    return result


def run_seed(args: argparse.Namespace, deps: dict[str, Any], seed: int) -> dict[str, float | int | str]:
    prompt_count = args.query_prompts + args.eval_prompts
    if prompt_count > len(DEFAULT_PROMPTS):
        raise ValueError(f"prompt budget {prompt_count} exceeds embedded prompt count {len(DEFAULT_PROMPTS)}")
    rng = np.random.default_rng(seed)
    prompt_ids = rng.permutation(len(DEFAULT_PROMPTS))[:prompt_count]
    prompts = [DEFAULT_PROMPTS[int(idx)] for idx in prompt_ids]
    hidden, output_weight, resolved_model = _load_lm_features(
        deps=deps,
        model_name=args.model,
        prompts=prompts,
        max_length=args.max_length,
    )
    hidden_dim = int(hidden.shape[1])
    vocab_size = int(output_weight.shape[0])
    projection, delta = _make_adapter(
        hidden_dim=hidden_dim,
        adapter_feature_dim=args.adapter_feature_dim,
        rank=args.rank,
        adapter_scale=args.adapter_scale,
        seed=seed + 50_000,
    )
    public_features = hidden @ projection.T
    adapted_hidden = hidden + public_features @ delta.T
    logits = adapted_hidden @ output_weight.T

    query = slice(0, args.query_prompts)
    evaluate = slice(args.query_prompts, prompt_count)
    recovered_hidden_logits = invert_endpoint_hidden_states(
        output_weight,
        np.zeros(vocab_size, dtype=np.float64),
        logits[query],
    )
    recovered_hidden_top1 = _top1_bias_hidden_recovery(
        output_weight=output_weight,
        logits=logits[query],
        bias_strength=args.bias_strength,
    )
    recovered_hidden_index, index_calls, index_uncovered_fraction = _top1_index_bias_hidden_recovery(
        output_weight=output_weight,
        logits=logits[query],
        bias_range=args.index_bias_range,
        binary_steps=args.index_binary_steps,
    )
    coordinate_sweep = _coordinate_exposure_sweep(
        output_weight=output_weight,
        logits=logits[query],
        true_hidden=adapted_hidden[query],
        row_budgets=args.row_budgets,
        seed=seed + 70_000,
    )
    adaptive_coordinate = _adaptive_coordinate_recovery(
        output_weight=output_weight,
        logits=logits[query],
        true_hidden=adapted_hidden[query],
        base_hidden=hidden[query],
        public_features=public_features[query],
        delta=delta,
        bias_strength=args.bias_strength,
        bias_range=args.index_bias_range,
        binary_steps=args.index_binary_steps,
    )
    recovered_delta_logits = _recover_delta(public_features[query], hidden[query], recovered_hidden_logits)
    recovered_delta_top1 = _recover_delta(public_features[query], hidden[query], recovered_hidden_top1)
    recovered_delta_index = _recover_delta(public_features[query], hidden[query], recovered_hidden_index)
    max_rank = args.max_rank if args.max_rank is not None else max(args.rank + 1, 2 * args.rank)
    full_logit_rank = _rank_diagnostics(
        recovered_delta_logits,
        delta,
        max_rank=max_rank,
        rank_tol=args.rank_tol,
    )
    top1_rank = _rank_diagnostics(
        recovered_delta_top1,
        delta,
        max_rank=max_rank,
        rank_tol=args.rank_tol,
    )
    index_rank = _rank_diagnostics(
        recovered_delta_index,
        delta,
        max_rank=max_rank,
        rank_tol=args.rank_tol,
    )

    predicted_hidden_logits = hidden[evaluate] + public_features[evaluate] @ recovered_delta_logits.T
    predicted_hidden_top1 = hidden[evaluate] + public_features[evaluate] @ recovered_delta_top1.T
    predicted_hidden_index = hidden[evaluate] + public_features[evaluate] @ recovered_delta_index.T
    true_eval_logits = logits[evaluate]
    predicted_logits = predicted_hidden_logits @ output_weight.T
    predicted_top1_logits = predicted_hidden_top1 @ output_weight.T
    predicted_index_logits = predicted_hidden_index @ output_weight.T
    feature_rank = int(np.linalg.matrix_rank(public_features[query], tol=1e-10))
    spectrum = _head_spectrum(output_weight)
    augmented_spectrum = _head_spectrum(output_weight, augmented=True)
    return {
        "model": resolved_model,
        "seed": seed,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        **spectrum,
        **augmented_spectrum,
        "adapter_feature_dim": args.adapter_feature_dim,
        "rank": args.rank,
        "unknown_rank_max_rank": int(max_rank),
        "unknown_rank_tol": float(args.rank_tol),
        "query_prompts": args.query_prompts,
        "eval_prompts": args.eval_prompts,
        "public_feature_rank": feature_rank,
        "full_logit_hidden_relerr": relative_frobenius_error(recovered_hidden_logits, adapted_hidden[query]),
        "top1_bias_hidden_relerr": relative_frobenius_error(recovered_hidden_top1, adapted_hidden[query]),
        "top1_index_bias_hidden_relerr": relative_frobenius_error(recovered_hidden_index, adapted_hidden[query]),
        "full_logit_delta_relerr": relative_frobenius_error(recovered_delta_logits, delta),
        "top1_bias_delta_relerr": relative_frobenius_error(recovered_delta_top1, delta),
        "top1_index_bias_delta_relerr": relative_frobenius_error(recovered_delta_index, delta),
        "full_logit_unknown_rank_estimate": full_logit_rank["rank_estimate"],
        "top1_bias_unknown_rank_estimate": top1_rank["rank_estimate"],
        "top1_index_bias_unknown_rank_estimate": index_rank["rank_estimate"],
        "full_logit_unknown_rank_spectral_gap": full_logit_rank["rank_spectral_gap"],
        "top1_bias_unknown_rank_spectral_gap": top1_rank["rank_spectral_gap"],
        "top1_index_bias_unknown_rank_spectral_gap": index_rank["rank_spectral_gap"],
        "full_logit_rank_projected_delta_relerr": full_logit_rank["rank_projected_delta_relerr"],
        "top1_bias_rank_projected_delta_relerr": top1_rank["rank_projected_delta_relerr"],
        "top1_index_bias_rank_projected_delta_relerr": index_rank["rank_projected_delta_relerr"],
        "full_logit_eval_max_logit_error": float(np.max(np.abs(predicted_logits - true_eval_logits))),
        "top1_bias_eval_max_logit_error": float(np.max(np.abs(predicted_top1_logits - true_eval_logits))),
        "top1_index_bias_eval_max_logit_error": float(np.max(np.abs(predicted_index_logits - true_eval_logits))),
        "full_logit_eval_top1_agreement": float(np.mean(np.argmax(predicted_logits, axis=1) == np.argmax(true_eval_logits, axis=1))),
        "top1_bias_eval_top1_agreement": float(np.mean(np.argmax(predicted_top1_logits, axis=1) == np.argmax(true_eval_logits, axis=1))),
        "top1_index_bias_eval_top1_agreement": float(np.mean(np.argmax(predicted_index_logits, axis=1) == np.argmax(true_eval_logits, axis=1))),
        "full_logit_endpoint_queries": args.query_prompts,
        "top1_bias_endpoint_calls": args.query_prompts * vocab_size,
        "top1_index_bias_endpoint_calls": index_calls,
        "top1_index_bias_uncovered_fraction": index_uncovered_fraction,
        "top1_bias_probe_mode": "per-token-forced-row-loop",
        "top1_index_bias_probe_mode": "temperature-zero-binary-search",
        **coordinate_sweep,
        **adaptive_coordinate,
    }


def summarize_rows(rows: list[dict[str, float | int | str]]) -> str:
    if not rows:
        return "# LM Endpoint Inversion Summary\n\nNo rows.\n"
    numeric_keys = [
        ("output_head_rank", "Output-head rank"),
        ("output_head_min_singular", "Output-head min singular value"),
        ("output_head_condition_number", "Output-head condition number"),
        ("softmax_augmented_rank", "Softmax-augmented rank"),
        ("softmax_augmented_condition_number", "Softmax-augmented condition number"),
        ("public_feature_rank", "Public adapter-feature rank"),
        ("full_logit_hidden_relerr", "Full-logit hidden relerr"),
        ("top1_bias_hidden_relerr", "Top-1/logit-bias hidden relerr"),
        ("top1_index_bias_hidden_relerr", "Top-1/index-bias hidden relerr"),
        ("full_logit_delta_relerr", "Full-logit adapter relerr"),
        ("top1_bias_delta_relerr", "Top-1/logit-bias adapter relerr"),
        ("top1_index_bias_delta_relerr", "Top-1/index-bias adapter relerr"),
        ("full_logit_unknown_rank_estimate", "Full-logit unknown-rank estimate"),
        ("top1_bias_unknown_rank_estimate", "Top-1/logit-bias unknown-rank estimate"),
        ("top1_index_bias_unknown_rank_estimate", "Top-1/index-bias unknown-rank estimate"),
        ("full_logit_unknown_rank_spectral_gap", "Full-logit unknown-rank spectral gap"),
        ("top1_bias_unknown_rank_spectral_gap", "Top-1/logit-bias unknown-rank spectral gap"),
        ("top1_index_bias_unknown_rank_spectral_gap", "Top-1/index-bias unknown-rank spectral gap"),
        ("full_logit_rank_projected_delta_relerr", "Full-logit rank-projected adapter relerr"),
        ("top1_bias_rank_projected_delta_relerr", "Top-1/logit-bias rank-projected adapter relerr"),
        ("top1_index_bias_rank_projected_delta_relerr", "Top-1/index-bias rank-projected adapter relerr"),
        ("full_logit_eval_max_logit_error", "Full-logit held-out max logit error"),
        ("top1_bias_eval_max_logit_error", "Top-1/logit-bias held-out max logit error"),
        ("top1_index_bias_eval_max_logit_error", "Top-1/index-bias held-out max logit error"),
        ("full_logit_eval_top1_agreement", "Full-logit held-out top-1 agreement"),
        ("top1_bias_eval_top1_agreement", "Top-1/logit-bias held-out top-1 agreement"),
        ("top1_index_bias_eval_top1_agreement", "Top-1/index-bias held-out top-1 agreement"),
        ("full_logit_endpoint_queries", "Full-logit endpoint queries"),
        ("top1_bias_endpoint_calls", "Top-1/logit-bias endpoint calls"),
        ("top1_index_bias_endpoint_calls", "Top-1/index-bias endpoint calls"),
        ("top1_index_bias_uncovered_fraction", "Top-1/index-bias uncovered fraction"),
        ("coordinate_exact_min_full_rank_budget", "Exact subset min full-rank rows"),
        ("coordinate_shifted_min_full_rank_budget", "Shifted subset min full-rank rows"),
        ("coordinate_exact_hidden_relerr_at_min", "Exact subset hidden relerr at threshold"),
        ("coordinate_shifted_hidden_relerr_at_min", "Shifted subset hidden relerr at threshold"),
        ("coordinate_exact_condition_at_min", "Exact subset condition at threshold"),
        ("coordinate_shifted_condition_at_min", "Shifted subset condition at threshold"),
        ("adaptive_exact_rows", "Adaptive exact rows"),
        ("adaptive_shifted_rows", "Adaptive shifted rows"),
        ("adaptive_exact_condition", "Adaptive exact condition"),
        ("adaptive_shifted_condition", "Adaptive shifted condition"),
        ("adaptive_exact_hidden_relerr", "Adaptive exact hidden relerr"),
        ("adaptive_top1_bias_hidden_relerr", "Adaptive top-1/logit-bias hidden relerr"),
        ("adaptive_top1_index_bias_hidden_relerr", "Adaptive top-1/index-bias hidden relerr"),
        ("adaptive_exact_delta_relerr", "Adaptive exact adapter relerr"),
        ("adaptive_top1_bias_delta_relerr", "Adaptive top-1/logit-bias adapter relerr"),
        ("adaptive_top1_index_bias_delta_relerr", "Adaptive top-1/index-bias adapter relerr"),
        ("adaptive_exact_endpoint_calls", "Adaptive exact endpoint calls"),
        ("adaptive_top1_bias_endpoint_calls", "Adaptive top-1/logit-bias endpoint calls"),
        ("adaptive_top1_index_bias_endpoint_calls", "Adaptive top-1/index-bias endpoint calls"),
        ("adaptive_top1_index_bias_uncovered_fraction", "Adaptive top-1/index-bias uncovered fraction"),
    ]
    lines = [
        "# LM Endpoint Inversion Summary",
        "",
        f"Base causal LM: `{rows[0]['model']}`",
        f"Hidden dimension: `{rows[0]['hidden_dim']}`; vocabulary rows: `{rows[0]['vocab_size']}`",
        f"Adapter feature dimension: `{rows[0]['adapter_feature_dim']}`; rank: `{rows[0]['rank']}`; rank upper bound: `{rows[0]['unknown_rank_max_rank']}`",
        "",
        "| metric | mean | std |",
        "|---|---:|---:|",
    ]
    for key, label in numeric_keys:
        values = [float(row[key]) for row in rows]
        lines.append(f"| {label} | {mean(values):.4g} | {(pstdev(values) if len(values) > 1 else 0.0):.4g} |")
    lines.extend(
        [
            "",
            "Interpretation: this is a real pretrained full-vocabulary endpoint-head experiment. The public LM head is used as the endpoint map. Exact full logits recover the adapted final representation; a restricted top-1/logit-bias log-prob interface recovers the same representation by forcing each vocabulary row into view. Once the hidden representation is recovered, a low-rank final-representation adapter over public base features is solved by linear equations.",
            "",
            "The unknown-rank rows estimate the numerical rank from the recovered adapter spectrum using the declared threshold, then report the error after projecting to that estimated rank. They test that the recovery transcript does not need the true rank as side information when the singular spectrum is separated.",
            "",
            "The top-1/logit-bias numbers are computed by an explicit per-token forced-row oracle loop over the local model logits, giving the same transcript an API would return if it exposed log-probabilities for forced tokens.",
            "",
            "The top-1/index-bias numbers use only the temperature-zero winning token under per-token logit-bias binary search. This scoreless oracle reconstructs relative logits up to an additive shift when every token is within the configured bias range.",
            "",
            "The coordinate-exposure rows report the smallest sampled vocabulary-row budget whose public output submatrix is full rank. Exact subset logits need hidden_dim independent rows; shifted logits from log-probability or top-1/index-bias style oracles need hidden_dim+1 augmented rows because the additive shift is unknown.",
            "",
            "The adaptive-coordinate rows select vocabulary rows by rank-revealing QR on the public endpoint head before any oracle call. They show that a public full-column-rank vocabulary head does not require full-vocabulary probing: hidden_dim exact rows, or hidden_dim+1 shifted rows, are sufficient and necessary for exact endpoint inversion under the stated coordinate-access model.",
            "",
            "Scope: this supports recovery under injective full-vocabulary endpoint heads. It is not a claim that low-dimensional classifier endpoints or arbitrary hidden transformer LoRA modules are identifiable from final logits.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    deps = _require_deps()
    args.outdir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    rows = [run_seed(args, deps, seed) for seed in seeds]

    json_path = args.outdir / "lm_endpoint_inversion.json"
    csv_path = args.outdir / "lm_endpoint_inversion.csv"
    md_path = args.outdir / "summary.md"
    payload: dict[str, float | int | str] | list[dict[str, float | int | str]] = rows[0] if len(rows) == 1 else rows
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_rows(rows)
    md_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
