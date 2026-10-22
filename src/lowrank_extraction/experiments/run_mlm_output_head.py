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
    unbiased_log_probs_from_biased_token_log_probs,
)
from lowrank_extraction.metrics import relative_frobenius_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover a low-rank full-vocabulary MLM output-head adapter from endpoint logits."
    )
    parser.add_argument("--outdir", type=Path, default=Path("results") / "mlm_output_head")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--query-size", type=int, default=900)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-rank", type=int, default=8)
    parser.add_argument("--rank-tol", type=float, default=1e-3)
    parser.add_argument("--feature-adapter-dim", type=int, default=16)
    parser.add_argument("--adapter-scale", type=float, default=0.02)
    parser.add_argument("--feature-adapter-scale", type=float, default=0.03)
    parser.add_argument("--bias-strength", type=float, default=8.0)
    parser.add_argument("--block-size", type=int, default=2048)
    return parser.parse_args()


def _require_deps() -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "The MLM output-head experiment needs optional heavy dependencies. "
            "Install them with: uv sync --extra dev --extra lora"
        ) from exc
    return {"torch": torch, "AutoModelForMaskedLM": AutoModelForMaskedLM, "AutoTokenizer": AutoTokenizer}


def _masked_contexts(count: int, *, seed: int) -> list[str]:
    subjects = [
        "cryptographic protocol",
        "low rank adapter",
        "language model endpoint",
        "masked token predictor",
        "public feature map",
        "oracle transcript",
        "transformer layer",
        "security reviewer",
        "model extraction attack",
        "defensive API",
        "top k interface",
        "hidden representation",
    ]
    verbs = [
        "reveals",
        "hides",
        "recovers",
        "tests",
        "measures",
        "bounds",
        "compares",
        "protects",
        "queries",
        "estimates",
        "separates",
        "verifies",
    ]
    objects = [
        "the private update",
        "a quotient space",
        "the output head",
        "the adapter row space",
        "a decision boundary",
        "the score vector",
        "the vocabulary distribution",
        "a rank certificate",
        "the public baseline",
        "the recovery theorem",
        "the endpoint signal",
        "a reproducible artifact",
    ]
    rng = np.random.default_rng(seed)
    contexts = []
    for idx in range(count):
        subject = subjects[int(rng.integers(len(subjects)))]
        verb = verbs[int(rng.integers(len(verbs)))]
        obj = objects[int(rng.integers(len(objects)))]
        contexts.append(f"The {subject} {verb} [MASK] for {obj} in experiment {idx}.")
    return contexts


def _mlm_features_and_head(
    *,
    deps: dict[str, Any],
    model_name: str,
    texts: list[str],
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    torch = deps["torch"]
    tokenizer = deps["AutoTokenizer"].from_pretrained(model_name)
    model = deps["AutoModelForMaskedLM"].from_pretrained(model_name)
    model.eval()
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("tokenizer does not define a mask token")
    features = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            outputs = model.bert(**enc, return_dict=True)
            mask_positions = (enc["input_ids"] == mask_token_id).nonzero(as_tuple=False)
            if int(mask_positions.shape[0]) != len(batch_texts):
                raise RuntimeError("each generated context must contain exactly one mask token")
            sequence_hidden = outputs.last_hidden_state[mask_positions[:, 0], mask_positions[:, 1]]
            transformed = model.cls.predictions.transform(sequence_hidden)
            features.append(transformed.detach().cpu().numpy().astype(np.float64))
    decoder = model.cls.predictions.decoder
    weight = decoder.weight.detach().cpu().numpy().astype(np.float64)
    bias = model.cls.predictions.bias.detach().cpu().numpy().astype(np.float64)
    return np.vstack(features), weight, bias, model.config._name_or_path


def _make_output_adapter(
    *,
    vocab_size: int,
    hidden_dim: int,
    rank: int,
    adapter_scale: float,
    seed: int,
) -> np.ndarray:
    if rank > min(vocab_size, hidden_dim):
        raise ValueError("rank must not exceed min(vocab_size, hidden_dim)")
    rng = np.random.default_rng(seed)
    left = rng.normal(0.0, adapter_scale / np.sqrt(rank), size=(vocab_size, rank))
    right = rng.normal(0.0, adapter_scale / np.sqrt(rank), size=(hidden_dim, rank))
    return (left @ right.T).astype(np.float64)


def _make_feature_adapter(
    *,
    hidden_dim: int,
    feature_adapter_dim: int,
    rank: int,
    adapter_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if feature_adapter_dim >= hidden_dim:
        raise ValueError("feature_adapter_dim must be smaller than hidden_dim")
    if rank > min(hidden_dim, feature_adapter_dim):
        raise ValueError("rank must not exceed min(hidden_dim, feature_adapter_dim)")
    rng = np.random.default_rng(seed)
    raw_projection = rng.normal(0.0, 1.0, size=(hidden_dim, feature_adapter_dim))
    projection, _ = np.linalg.qr(raw_projection)
    left = rng.normal(0.0, adapter_scale / np.sqrt(rank), size=(hidden_dim, rank))
    right = rng.normal(0.0, adapter_scale / np.sqrt(rank), size=(feature_adapter_dim, rank))
    return projection.T.astype(np.float64), (left @ right.T).astype(np.float64)


def _head_spectrum(weight: np.ndarray) -> dict[str, float | int]:
    eigvals = np.linalg.eigvalsh(weight.T @ weight)
    positive = eigvals[eigvals > 1e-10]
    rank = int(positive.size)
    min_singular = float(np.sqrt(np.min(positive))) if positive.size else 0.0
    max_singular = float(np.sqrt(np.max(positive))) if positive.size else 0.0
    return {
        "output_head_rank": rank,
        "output_head_min_singular": min_singular,
        "output_head_condition_number": max_singular / max(min_singular, 1e-300),
    }


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
    except ImportError as exc:  # pragma: no cover - scipy is available through the default scientific stack.
        raise RuntimeError("adaptive coordinate selection requires scipy.linalg.qr") from exc
    _, _, pivots = qr(matrix.T, pivoting=True, mode="economic")
    return np.asarray(pivots[:rows], dtype=np.int64)


def _forced_biased_log_probs_subset(
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
        observed[:, output_col] = shifted[:, token_idx] + bias_strength - np.log(total + (boosted - 1.0) * token_exp)
    return observed


def _recover_feature_delta(public_features: np.ndarray, base_features: np.ndarray, recovered_features: np.ndarray) -> np.ndarray:
    update = recovered_features - base_features
    theta, *_ = np.linalg.lstsq(public_features, update, rcond=None)
    return theta.T


def _recover_feature_adapter_from_endpoint(
    *,
    base_features: np.ndarray,
    output_weight: np.ndarray,
    output_bias: np.ndarray,
    feature_adapter_dim: int,
    rank: int,
    adapter_scale: float,
    bias_strength: float,
    seed: int,
) -> dict[str, float | int]:
    hidden_dim = output_weight.shape[1]
    projection, delta = _make_feature_adapter(
        hidden_dim=hidden_dim,
        feature_adapter_dim=feature_adapter_dim,
        rank=rank,
        adapter_scale=adapter_scale,
        seed=seed,
    )
    public_features = base_features @ projection.T
    adapted_features = base_features + public_features @ delta.T
    logits = adapted_features @ output_weight.T + output_bias
    exact_indices = _pivot_rows(output_weight, rows=hidden_dim)
    shifted_matrix = np.column_stack([output_weight, -np.ones(output_weight.shape[0], dtype=np.float64)])
    shifted_indices = _pivot_rows(shifted_matrix, rows=hidden_dim + 1)

    exact_hidden = invert_endpoint_hidden_states(
        output_weight[exact_indices],
        output_bias[exact_indices],
        logits[:, exact_indices],
    )
    biased_log_probs = _forced_biased_log_probs_subset(
        logits=logits,
        indices=shifted_indices,
        bias_strength=bias_strength,
    )
    recovered_log_probs = unbiased_log_probs_from_biased_token_log_probs(
        biased_log_probs,
        bias=bias_strength,
    )
    shifted_hidden = invert_hidden_states_from_log_probs(
        output_weight[shifted_indices],
        output_bias[shifted_indices],
        recovered_log_probs,
    )
    exact_delta = _recover_feature_delta(public_features, base_features, exact_hidden)
    shifted_delta = _recover_feature_delta(public_features, base_features, shifted_hidden)
    return {
        "feature_adapter_dim": int(feature_adapter_dim),
        "feature_adapter_rank": int(rank),
        "adaptive_exact_rows": int(exact_indices.size),
        "adaptive_shifted_rows": int(shifted_indices.size),
        "adaptive_exact_condition": _condition_number(output_weight[exact_indices]),
        "adaptive_shifted_condition": _condition_number(shifted_matrix[shifted_indices]),
        "adaptive_exact_hidden_relerr": relative_frobenius_error(exact_hidden, adapted_features),
        "adaptive_shifted_hidden_relerr": relative_frobenius_error(shifted_hidden, adapted_features),
        "adaptive_exact_delta_relerr": relative_frobenius_error(exact_delta, delta),
        "adaptive_shifted_delta_relerr": relative_frobenius_error(shifted_delta, delta),
        "adaptive_exact_endpoint_calls": int(base_features.shape[0] * exact_indices.size),
        "adaptive_shifted_endpoint_calls": int(base_features.shape[0] * shifted_indices.size),
    }


def _recover_output_head_blockwise(
    *,
    z_query: np.ndarray,
    z_test: np.ndarray,
    base_weight: np.ndarray,
    base_bias: np.ndarray,
    delta: np.ndarray,
    block_size: int,
) -> dict[str, float | int]:
    design = np.hstack([z_query, np.ones((z_query.shape[0], 1), dtype=np.float64)])
    feature_rank = int(np.linalg.matrix_rank(design, tol=1e-10))
    if feature_rank < design.shape[1]:
        raise RuntimeError(
            f"query features have rank {feature_rank}, but output-head recovery needs {design.shape[1]}"
        )
    design_pinv = np.linalg.pinv(design)
    true_weight = base_weight + delta
    logits_query = z_query @ true_weight.T + base_bias
    true_test_logits = z_test @ true_weight.T + base_bias
    predicted_test_logits = np.empty_like(true_test_logits)

    delta_error_sq = 0.0
    delta_norm_sq = 0.0
    bias_error_sq = 0.0
    bias_norm_sq = float(np.sum(base_bias**2))
    max_logit_error = 0.0
    for start in range(0, base_weight.shape[0], block_size):
        end = min(start + block_size, base_weight.shape[0])
        theta = design_pinv @ logits_query[:, start:end]
        recovered_weight = theta[:-1].T
        recovered_bias = theta[-1]
        recovered_delta = recovered_weight - base_weight[start:end]
        delta_error_sq += float(np.sum((recovered_delta - delta[start:end]) ** 2))
        delta_norm_sq += float(np.sum(delta[start:end] ** 2))
        bias_error_sq += float(np.sum((recovered_bias - base_bias[start:end]) ** 2))
        predicted_block = z_test @ recovered_weight.T + recovered_bias
        predicted_test_logits[:, start:end] = predicted_block
        max_logit_error = max(max_logit_error, float(np.max(np.abs(predicted_block - true_test_logits[:, start:end]))))

    return {
        "augmented_feature_rank": feature_rank,
        "output_delta_relerr": float(np.sqrt(delta_error_sq) / max(np.sqrt(delta_norm_sq), 1e-12)),
        "output_bias_relerr": float(np.sqrt(bias_error_sq) / max(np.sqrt(bias_norm_sq), 1e-12)),
        "heldout_max_logit_error": max_logit_error,
        "heldout_top1_agreement": float(
            np.mean(np.argmax(predicted_test_logits, axis=1) == np.argmax(true_test_logits, axis=1))
        ),
    }


class SparseCoordinateOutputHeadOracle:
    """Transcript oracle for selected vocabulary-row scores on fixed query features."""

    def __init__(self, z_query: np.ndarray, target_weight: np.ndarray, target_bias: np.ndarray) -> None:
        self._z_query = z_query
        self._target_weight = target_weight
        self._target_bias = target_bias
        self.call_count = 0
        self.scalar_observations = 0

    def query(self, query_indices: np.ndarray, row_indices: np.ndarray) -> np.ndarray:
        query_indices = np.asarray(query_indices, dtype=np.int64)
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if query_indices.ndim != 1 or row_indices.ndim != 1:
            raise ValueError("query_indices and row_indices must be one-dimensional")
        if np.any(query_indices < 0) or np.any(query_indices >= self._z_query.shape[0]):
            raise ValueError("query index out of range")
        if np.any(row_indices < 0) or np.any(row_indices >= self._target_weight.shape[0]):
            raise ValueError("row index out of range")
        self.call_count += 1
        self.scalar_observations += int(query_indices.size * row_indices.size)
        return self._z_query[query_indices] @ self._target_weight[row_indices].T + self._target_bias[row_indices]


def _recover_output_head_lowrank_coordinate(
    *,
    z_query: np.ndarray,
    z_test: np.ndarray,
    base_weight: np.ndarray,
    base_bias: np.ndarray,
    delta: np.ndarray,
    rank: int,
    block_size: int,
) -> dict[str, float | int]:
    hidden_dim = int(base_weight.shape[1])
    if z_query.shape[0] < hidden_dim + 1:
        raise RuntimeError("coordinate low-rank recovery needs at least hidden_dim+1 query features")
    if z_query.shape[0] < rank + 1:
        raise RuntimeError("coordinate low-rank recovery needs at least rank+1 query features")

    full_design = np.hstack([z_query[: hidden_dim + 1], np.ones((hidden_dim + 1, 1), dtype=np.float64)])
    if np.linalg.matrix_rank(full_design, tol=1e-10) < hidden_dim + 1:
        raise RuntimeError("full-row coordinate design is rank deficient")
    full_pinv = np.linalg.pinv(full_design)
    full_query_indices = np.arange(hidden_dim + 1, dtype=np.int64)
    lowrank_query_indices = np.arange(rank + 1, dtype=np.int64)
    oracle = SparseCoordinateOutputHeadOracle(z_query, base_weight + delta, base_bias)

    recovered_basis_rows = []
    basis_indices = []
    basis_attempts = 0
    row_space_rank = 0
    for row_idx in range(base_weight.shape[0]):
        basis_attempts += 1
        scores = oracle.query(full_query_indices, np.asarray([row_idx], dtype=np.int64))[:, 0]
        theta = full_pinv @ scores
        recovered_delta_row = theta[:hidden_dim] - base_weight[row_idx]
        recovered_basis_rows.append(recovered_delta_row)
        basis_indices.append(row_idx)
        singular_values = np.linalg.svd(np.vstack(recovered_basis_rows), compute_uv=False)
        row_space_rank = int(np.sum(singular_values > 1e-10))
        if row_space_rank >= rank:
            break
    if row_space_rank < rank:
        raise RuntimeError("failed to identify enough independent adapter rows")

    _, _, vt = np.linalg.svd(np.vstack(recovered_basis_rows), full_matrices=False)
    row_basis = vt[:rank]
    lowrank_features = z_query[: rank + 1]
    lowrank_design = np.hstack(
        [lowrank_features @ row_basis.T, np.ones((rank + 1, 1), dtype=np.float64)]
    )
    lowrank_design_rank = int(np.linalg.matrix_rank(lowrank_design, tol=1e-10))
    if lowrank_design_rank < rank + 1:
        raise RuntimeError("low-rank coordinate design is rank deficient")
    lowrank_pinv = np.linalg.pinv(lowrank_design)

    true_weight = base_weight + delta
    true_test_logits = z_test @ true_weight.T + base_bias
    predicted_test_logits = np.empty_like(true_test_logits)
    basis_index_set = set(basis_indices)
    delta_error_sq = 0.0
    delta_norm_sq = 0.0
    max_logit_error = 0.0
    for start in range(0, base_weight.shape[0], block_size):
        end = min(start + block_size, base_weight.shape[0])
        recovered_delta_block = np.empty_like(delta[start:end])
        recovered_bias_block = np.empty(end - start, dtype=np.float64)
        for offset, row_idx in enumerate(range(start, end)):
            if row_idx in basis_index_set:
                basis_pos = basis_indices.index(row_idx)
                recovered_delta_block[offset] = recovered_basis_rows[basis_pos]
                recovered_bias_block[offset] = base_bias[row_idx]
                continue
            scores = oracle.query(lowrank_query_indices, np.asarray([row_idx], dtype=np.int64))[:, 0]
            residual_scores = scores - lowrank_features @ base_weight[row_idx]
            theta = lowrank_pinv @ residual_scores
            recovered_delta_block[offset] = theta[:rank] @ row_basis
            recovered_bias_block[offset] = theta[rank]
        delta_error_sq += float(np.sum((recovered_delta_block - delta[start:end]) ** 2))
        delta_norm_sq += float(np.sum(delta[start:end] ** 2))
        predicted_block = z_test @ (base_weight[start:end] + recovered_delta_block).T + recovered_bias_block
        predicted_test_logits[:, start:end] = predicted_block
        max_logit_error = max(max_logit_error, float(np.max(np.abs(predicted_block - true_test_logits[:, start:end]))))

    basis_rows = len(basis_indices)
    full_logit_scalars = z_query.shape[0] * base_weight.shape[0]
    return {
        "coordinate_basis_rows": int(basis_rows),
        "coordinate_basis_attempts": int(basis_attempts),
        "coordinate_row_space_rank": int(row_space_rank),
        "coordinate_basis_queries_per_row": int(hidden_dim + 1),
        "coordinate_lowrank_queries_per_remaining_row": int(rank + 1),
        "coordinate_lowrank_design_rank": int(lowrank_design_rank),
        "coordinate_oracle_calls": int(oracle.call_count),
        "coordinate_total_scalar_observations": int(oracle.scalar_observations),
        "coordinate_full_logit_equivalent_scalars": int(full_logit_scalars),
        "coordinate_observation_reduction": float(full_logit_scalars / max(oracle.scalar_observations, 1)),
        "coordinate_output_delta_relerr": float(np.sqrt(delta_error_sq) / max(np.sqrt(delta_norm_sq), 1e-12)),
        "coordinate_heldout_max_logit_error": max_logit_error,
        "coordinate_heldout_top1_agreement": float(
            np.mean(np.argmax(predicted_test_logits, axis=1) == np.argmax(true_test_logits, axis=1))
        ),
    }


def _recover_output_head_unknown_rank_coordinate(
    *,
    z_query: np.ndarray,
    z_test: np.ndarray,
    base_weight: np.ndarray,
    base_bias: np.ndarray,
    delta: np.ndarray,
    max_rank: int,
    rank_tol: float,
    block_size: int,
) -> dict[str, float | int]:
    hidden_dim = int(base_weight.shape[1])
    if max_rank < 1:
        raise ValueError("max_rank must be positive")
    if z_query.shape[0] < hidden_dim + 1:
        raise RuntimeError("unknown-rank coordinate recovery needs at least hidden_dim+1 query features")
    basis_rows = min(base_weight.shape[0], max_rank + 2)
    full_query_indices = np.arange(hidden_dim + 1, dtype=np.int64)
    full_design = np.hstack([z_query[full_query_indices], np.ones((hidden_dim + 1, 1), dtype=np.float64)])
    if np.linalg.matrix_rank(full_design, tol=1e-10) < hidden_dim + 1:
        raise RuntimeError("full-row coordinate design is rank deficient")
    full_pinv = np.linalg.pinv(full_design)
    oracle = SparseCoordinateOutputHeadOracle(z_query, base_weight + delta, base_bias)

    recovered_basis_rows = []
    for row_idx in range(basis_rows):
        scores = oracle.query(full_query_indices, np.asarray([row_idx], dtype=np.int64))[:, 0]
        theta = full_pinv @ scores
        recovered_basis_rows.append(theta[:hidden_dim] - base_weight[row_idx])
    basis_matrix = np.vstack(recovered_basis_rows)
    _, singular_values, vt = np.linalg.svd(basis_matrix, full_matrices=False)
    threshold = max(float(singular_values[0]) * rank_tol, 1e-12) if singular_values.size else 1e-12
    estimated_rank = int(np.sum(singular_values > threshold))
    if estimated_rank < 1:
        raise RuntimeError("unknown-rank coordinate recovery estimated zero rank")
    if estimated_rank > max_rank:
        raise RuntimeError("unknown-rank coordinate recovery exceeded max_rank")
    if z_query.shape[0] < estimated_rank + 1:
        raise RuntimeError("unknown-rank coordinate recovery needs estimated_rank+1 query features")
    next_singular = float(singular_values[estimated_rank]) if estimated_rank < singular_values.size else 0.0
    spectral_gap = float(singular_values[estimated_rank - 1] / max(next_singular, 1e-300))
    row_basis = vt[:estimated_rank]

    lowrank_query_indices = np.arange(estimated_rank + 1, dtype=np.int64)
    lowrank_features = z_query[lowrank_query_indices]
    lowrank_design = np.hstack(
        [lowrank_features @ row_basis.T, np.ones((estimated_rank + 1, 1), dtype=np.float64)]
    )
    lowrank_design_rank = int(np.linalg.matrix_rank(lowrank_design, tol=1e-10))
    if lowrank_design_rank < estimated_rank + 1:
        raise RuntimeError("unknown-rank low-rank coordinate design is rank deficient")
    lowrank_pinv = np.linalg.pinv(lowrank_design)

    true_weight = base_weight + delta
    true_test_logits = z_test @ true_weight.T + base_bias
    predicted_test_logits = np.empty_like(true_test_logits)
    delta_error_sq = 0.0
    delta_norm_sq = 0.0
    max_logit_error = 0.0
    for start in range(0, base_weight.shape[0], block_size):
        end = min(start + block_size, base_weight.shape[0])
        recovered_delta_block = np.empty_like(delta[start:end])
        recovered_bias_block = np.empty(end - start, dtype=np.float64)
        for offset, row_idx in enumerate(range(start, end)):
            if row_idx < basis_rows:
                recovered_delta_block[offset] = basis_matrix[row_idx]
                recovered_bias_block[offset] = base_bias[row_idx]
                continue
            scores = oracle.query(lowrank_query_indices, np.asarray([row_idx], dtype=np.int64))[:, 0]
            residual_scores = scores - lowrank_features @ base_weight[row_idx]
            theta = lowrank_pinv @ residual_scores
            recovered_delta_block[offset] = theta[:estimated_rank] @ row_basis
            recovered_bias_block[offset] = theta[estimated_rank]
        delta_error_sq += float(np.sum((recovered_delta_block - delta[start:end]) ** 2))
        delta_norm_sq += float(np.sum(delta[start:end] ** 2))
        predicted_block = z_test @ (base_weight[start:end] + recovered_delta_block).T + recovered_bias_block
        predicted_test_logits[:, start:end] = predicted_block
        max_logit_error = max(max_logit_error, float(np.max(np.abs(predicted_block - true_test_logits[:, start:end]))))

    full_logit_scalars = z_query.shape[0] * base_weight.shape[0]
    return {
        "unknown_rank_max_rank": int(max_rank),
        "unknown_rank_estimate": int(estimated_rank),
        "unknown_rank_basis_rows": int(basis_rows),
        "unknown_rank_spectral_gap": spectral_gap,
        "unknown_rank_lowrank_design_rank": int(lowrank_design_rank),
        "unknown_rank_oracle_calls": int(oracle.call_count),
        "unknown_rank_total_scalar_observations": int(oracle.scalar_observations),
        "unknown_rank_observation_reduction": float(full_logit_scalars / max(oracle.scalar_observations, 1)),
        "unknown_rank_output_delta_relerr": float(np.sqrt(delta_error_sq) / max(np.sqrt(delta_norm_sq), 1e-12)),
        "unknown_rank_heldout_max_logit_error": max_logit_error,
        "unknown_rank_heldout_top1_agreement": float(
            np.mean(np.argmax(predicted_test_logits, axis=1) == np.argmax(true_test_logits, axis=1))
        ),
    }


def run_seed(args: argparse.Namespace, deps: dict[str, Any], seed: int) -> dict[str, float | int | str | bool]:
    texts = _masked_contexts(args.query_size + args.test_size, seed=seed + 10_000)
    features, output_weight, output_bias, resolved_model = _mlm_features_and_head(
        deps=deps,
        model_name=args.model,
        texts=texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    hidden_dim = int(features.shape[1])
    vocab_size = int(output_weight.shape[0])
    delta = _make_output_adapter(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        rank=args.rank,
        adapter_scale=args.adapter_scale,
        seed=seed + 20_000,
    )
    spectrum = _head_spectrum(output_weight)
    recovery = _recover_output_head_blockwise(
        z_query=features[: args.query_size],
        z_test=features[args.query_size :],
        base_weight=output_weight,
        base_bias=output_bias,
        delta=delta,
        block_size=args.block_size,
    )
    coordinate_recovery = _recover_output_head_lowrank_coordinate(
        z_query=features[: args.query_size],
        z_test=features[args.query_size :],
        base_weight=output_weight,
        base_bias=output_bias,
        delta=delta,
        rank=args.rank,
        block_size=args.block_size,
    )
    unknown_rank_recovery = _recover_output_head_unknown_rank_coordinate(
        z_query=features[: args.query_size],
        z_test=features[args.query_size :],
        base_weight=output_weight,
        base_bias=output_bias,
        delta=delta,
        max_rank=args.max_rank,
        rank_tol=args.rank_tol,
        block_size=args.block_size,
    )
    feature_recovery = _recover_feature_adapter_from_endpoint(
        base_features=features[: args.query_size],
        output_weight=output_weight,
        output_bias=output_bias,
        feature_adapter_dim=args.feature_adapter_dim,
        rank=args.rank,
        adapter_scale=args.feature_adapter_scale,
        bias_strength=args.bias_strength,
        seed=seed + 30_000,
    )
    return {
        "model": resolved_model,
        "seed": seed,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        **spectrum,
        "output_head_injective": bool(int(spectrum["output_head_rank"]) == hidden_dim),
        "rank": args.rank,
        "query_count": args.query_size,
        "test_count": args.test_size,
        **recovery,
        **coordinate_recovery,
        **unknown_rank_recovery,
        **feature_recovery,
    }


def summarize_rows(rows: list[dict[str, float | int | str | bool]]) -> str:
    if not rows:
        return "# MLM Output-Head Adapter Recovery Summary\n\nNo rows.\n"
    numeric_keys = [
        ("output_head_rank", "Output-head rank"),
        ("output_head_min_singular", "Output-head min singular value"),
        ("output_head_condition_number", "Output-head condition number"),
        ("augmented_feature_rank", "Augmented feature rank"),
        ("output_delta_relerr", "Output-head adapter relerr"),
        ("output_bias_relerr", "Output bias relerr"),
        ("heldout_max_logit_error", "Held-out max logit error"),
        ("heldout_top1_agreement", "Held-out top-1 agreement"),
        ("query_count", "Endpoint full-logit queries"),
        ("coordinate_basis_rows", "Coordinate basis rows"),
        ("coordinate_basis_attempts", "Coordinate basis attempts"),
        ("coordinate_row_space_rank", "Coordinate row-space rank"),
        ("coordinate_basis_queries_per_row", "Coordinate basis queries per row"),
        ("coordinate_lowrank_queries_per_remaining_row", "Coordinate low-rank queries per remaining row"),
        ("coordinate_oracle_calls", "Coordinate oracle calls"),
        ("coordinate_total_scalar_observations", "Coordinate scalar observations"),
        ("coordinate_full_logit_equivalent_scalars", "Full-logit equivalent scalars"),
        ("coordinate_observation_reduction", "Coordinate observation reduction"),
        ("coordinate_output_delta_relerr", "Coordinate output-head adapter relerr"),
        ("coordinate_heldout_max_logit_error", "Coordinate held-out max logit error"),
        ("coordinate_heldout_top1_agreement", "Coordinate held-out top-1 agreement"),
        ("unknown_rank_estimate", "Unknown-rank estimate"),
        ("unknown_rank_basis_rows", "Unknown-rank basis rows"),
        ("unknown_rank_spectral_gap", "Unknown-rank spectral gap"),
        ("unknown_rank_total_scalar_observations", "Unknown-rank scalar observations"),
        ("unknown_rank_observation_reduction", "Unknown-rank observation reduction"),
        ("unknown_rank_output_delta_relerr", "Unknown-rank output-head adapter relerr"),
        ("unknown_rank_heldout_top1_agreement", "Unknown-rank held-out top-1 agreement"),
        ("adaptive_exact_rows", "Adaptive exact rows"),
        ("adaptive_shifted_rows", "Adaptive shifted rows"),
        ("adaptive_exact_condition", "Adaptive exact condition"),
        ("adaptive_shifted_condition", "Adaptive shifted condition"),
        ("adaptive_exact_hidden_relerr", "Adaptive exact hidden relerr"),
        ("adaptive_shifted_hidden_relerr", "Adaptive shifted/log-prob hidden relerr"),
        ("adaptive_exact_delta_relerr", "Adaptive exact feature-adapter relerr"),
        ("adaptive_shifted_delta_relerr", "Adaptive shifted feature-adapter relerr"),
        ("adaptive_exact_endpoint_calls", "Adaptive exact endpoint calls"),
        ("adaptive_shifted_endpoint_calls", "Adaptive shifted endpoint calls"),
    ]
    lines = [
        "# MLM Output-Head Adapter Recovery Summary",
        "",
        f"Base masked LM: `{rows[0]['model']}`",
        f"Hidden dimension: `{rows[0]['hidden_dim']}`; vocabulary rows: `{rows[0]['vocab_size']}`",
        f"Adapter rank: `{rows[0]['rank']}`",
        "",
        "| metric | mean | std |",
        "|---|---:|---:|",
    ]
    for key, label in numeric_keys:
        values = [float(row[key]) for row in rows]
        lines.append(f"| {label} | {mean(values):.4g} | {(pstdev(values) if len(values) > 1 else 0.0):.4g} |")
    injective = all(bool(row["output_head_injective"]) for row in rows)
    lines.extend(
        [
            "",
            f"Output head injective in all runs: `{injective}`.",
            "",
            "Interpretation: this is a real full-vocabulary masked-LM endpoint experiment. The attacker queries masked contexts and receives endpoint MLM logits. Because the transformed MLM feature is public and the unknown adaptation is a low-rank vocabulary-output-head update, full logits give linear equations for every vocabulary row and recover the adapter product.",
            "",
            "Low-rank coordinate output-head recovery: the runner also recovers the same synthetic vocabulary-output update from an explicit sparse coordinate-score oracle. It first solves enough complete rows to identify the rank-r adapter row space, then solves each remaining row from r+1 scored coordinates; the reported reduction is in scalar score observations, not ordinary full-logit endpoint calls.",
            "",
            "Unknown-rank coordinate recovery: the runner also estimates the adapter rank from fully recovered coordinate rows before applying the low-rank coordinate attack. This checks that the sparse-oracle recovery path does not need the true rank as secret side information when a safe rank upper bound is provided.",
            "",
            "Adaptive-coordinate endpoint inversion: the runner also constructs a low-rank final-feature adapter over the same public MLM feature and recovers it from QR-selected output rows. Exact coordinate logits need hidden_dim rows; shifted log-probability coordinates need hidden_dim+1 rows.",
            "",
            "Scope: this strengthens the recovery side for token-distribution endpoints. It is still an observable output-head result, not arbitrary hidden-transformer LoRA recovery from a low-dimensional classifier endpoint.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    deps = _require_deps()
    args.outdir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    rows = [run_seed(args, deps, seed) for seed in seeds]

    json_path = args.outdir / "mlm_output_head.json"
    csv_path = args.outdir / "mlm_output_head.csv"
    md_path = args.outdir / "summary.md"
    payload: dict[str, float | int | str | bool] | list[dict[str, float | int | str | bool]]
    payload = rows[0] if len(rows) == 1 else rows
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
