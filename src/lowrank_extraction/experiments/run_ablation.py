from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from lowrank_extraction.attacks import (
    predict_feature_boundaries,
    predict_multiclass_boundaries,
    predict_ranking_boundaries,
    recover_feature_hard_label_boundaries,
    recover_feature_rank_index_boundaries,
    recover_hidden_lora_block_on_positive_cell,
    recover_lowrank_assisted_feature_head_from_observations,
    recover_binary_hard_label_boundary,
    recover_multiclass_hard_label_boundaries,
    recover_raw_affine,
    recover_raw_feature_head,
    recover_topk_scored_feature_head,
    recover_topk_scored_affine,
)
from lowrank_extraction.endpoint_observability import (
    invert_endpoint_hidden_states,
    invert_hidden_states_from_log_probs,
    unbiased_log_probs_from_biased_token_log_probs,
)
from lowrank_extraction.metrics import (
    binary_boundary_agreement,
    label_agreement,
    relative_frobenius_error,
    singular_value_energy,
)
from lowrank_extraction.models import LinearLowRankTarget, make_hidden_lora_block_target, make_linear_lowrank_target, make_relu_feature_lowrank_target, random_probe_points
from lowrank_extraction.oracles import HardLabelOracle, RawLogitOracle, TopKOracle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a compact reproducibility ablation suite.")
    parser.add_argument("--outdir", type=Path, default=Path("results") / "ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--features", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--feature-hard-label-seeds", type=int, nargs="+", default=[4, 5, 6])
    return parser.parse_args()


def run_case(*, n_features: int, n_classes: int, rank: int, seed: int) -> dict[str, float | int | bool]:
    target = make_linear_lowrank_target(
        n_features=n_features,
        n_classes=n_classes,
        rank=min(rank, n_features, n_classes),
        seed=seed,
    )
    eval_x = random_probe_points(2048, target.n_features, seed=seed + 10_000)

    raw_oracle = RawLogitOracle(target)
    raw = recover_raw_affine(raw_oracle, base_weight=target.base_weight, rank=target.rank)

    topk = max(2, min(3, n_classes))
    topk_oracle = TopKOracle(target, k=topk, include_scores=True)
    topk_recovery = recover_topk_scored_affine(
        topk_oracle,
        base_weight=target.base_weight,
        rank=target.rank,
        random_queries=max(500, n_features * n_classes * 10),
        seed=seed + 20_000,
    )

    binary_target = make_linear_lowrank_target(
        n_features=n_features,
        n_classes=2,
        rank=1,
        seed=seed + 30_000,
    )
    binary_oracle = HardLabelOracle(binary_target)
    binary = recover_binary_hard_label_boundary(
        binary_oracle,
        n_features=n_features,
        seed=seed + 40_000,
        validation_points=2048,
    )
    binary_eval_x = random_probe_points(2048, n_features, seed=seed + 50_000, scale=2.5)

    return {
        "features": n_features,
        "classes": n_classes,
        "rank": target.rank,
        "seed": seed,
        "raw_queries": raw.query_count,
        "raw_delta_relerr": relative_frobenius_error(raw.recovered_delta, target.delta),
        "raw_agreement": label_agreement(raw.recovered_weight, raw.recovered_bias, target.weight, target.bias, eval_x),
        "raw_top_rank_energy": singular_value_energy(raw.singular_values, target.rank),
        "topk": topk,
        "topk_queries": topk_recovery.query_count,
        "topk_solved_classes": int(topk_recovery.solved_classes.sum()),
        "topk_delta_relerr": relative_frobenius_error(topk_recovery.recovered_delta, target.delta),
        "topk_agreement": label_agreement(
            topk_recovery.recovered_weight,
            topk_recovery.recovered_bias,
            target.weight,
            target.bias,
            eval_x,
        ),
        "binary_queries": binary.query_count,
        "binary_solved": binary.solved,
        "binary_agreement": binary_boundary_agreement(
            binary.normal,
            binary.offset,
            binary_target.weight,
            binary_target.bias,
            binary_eval_x,
        )
        if binary.solved
        else 0.0,
    }


def run_multiclass_boundary_case(*, n_features: int, n_classes: int, seed: int) -> dict[str, float | int | bool]:
    target = make_linear_lowrank_target(
        n_features=n_features,
        n_classes=n_classes,
        rank=1,
        seed=seed + 60_000,
    )
    oracle = HardLabelOracle(target)
    result = recover_multiclass_hard_label_boundaries(
        oracle,
        n_features=n_features,
        n_classes=n_classes,
        seed=seed + 70_000,
        max_attempts=100_000,
        validation_points=2048,
    )
    eval_x = random_probe_points(4096, n_features, seed=seed + 80_000, scale=2.5)
    pred = predict_multiclass_boundaries(result, eval_x)
    truth = target.labels(eval_x)
    return {
        "features": n_features,
        "classes": n_classes,
        "seed": seed,
        "query_count": result.query_count,
        "solved_undirected_pairs": int(result.solved_pairs.sum() // 2),
        "total_undirected_pairs": int(n_classes * (n_classes - 1) // 2),
        "validation_agreement": result.validation_agreement,
        "eval_agreement": float((pred == truth).mean()),
    }


def run_relu_feature_case(*, n_features: int, hidden_features: int, n_classes: int, rank: int, seed: int) -> dict[str, float | int | bool]:
    target = make_relu_feature_lowrank_target(
        n_features=n_features,
        hidden_features=hidden_features,
        n_classes=n_classes,
        rank=rank,
        seed=seed + 90_000,
    )
    oracle = RawLogitOracle(target)
    result = recover_raw_feature_head(
        oracle,
        base_head=target.base_head,
        feature_map=target.features,
        input_dim=target.n_features,
        feature_dim=target.hidden_features,
        rank=target.rank,
        seed=seed + 100_000,
    )
    topk_oracle = TopKOracle(target, k=max(2, min(3, n_classes)), include_scores=True)
    topk_result = recover_topk_scored_feature_head(
        topk_oracle,
        base_head=target.base_head,
        feature_map=target.features,
        input_dim=target.n_features,
        feature_dim=target.hidden_features,
        rank=target.rank,
        random_queries=2500,
        seed=seed + 105_000,
    )
    eval_x = random_probe_points(2048, n_features, seed=seed + 110_000)
    eval_z = target.features(eval_x)
    return {
        "input_features": n_features,
        "hidden_features": hidden_features,
        "classes": n_classes,
        "rank": rank,
        "seed": seed,
        "query_count": result.query_count,
        "feature_rank": result.feature_rank,
        "head_delta_relerr": relative_frobenius_error(result.recovered_delta, target.delta),
        "eval_agreement": label_agreement(result.recovered_head, result.recovered_bias, target.head, target.bias, eval_z),
        "topk_query_count": topk_result.query_count,
        "topk_solved_classes": int(topk_result.solved_classes.sum()),
        "topk_head_delta_relerr": relative_frobenius_error(topk_result.recovered_delta, target.delta),
        "topk_eval_agreement": label_agreement(
            topk_result.recovered_head,
            topk_result.recovered_bias,
            target.head,
            target.bias,
            eval_z,
        ),
    }


def run_relu_feature_hard_label_case(*, n_features: int, hidden_features: int, n_classes: int, rank: int, seed: int) -> dict[str, float | int | bool]:
    target = make_relu_feature_lowrank_target(
        n_features=n_features,
        hidden_features=hidden_features,
        n_classes=n_classes,
        rank=rank,
        seed=seed + 120_000,
    )
    oracle = HardLabelOracle(target)
    result = recover_feature_hard_label_boundaries(
        oracle,
        feature_map=target.features,
        input_dim=target.n_features,
        feature_dim=target.hidden_features,
        n_classes=target.n_classes,
        seed=seed + 130_000,
        max_attempts=120_000,
        validation_points=2048,
    )
    eval_x = random_probe_points(4096, n_features, seed=seed + 140_000, scale=2.5)
    pred = predict_feature_boundaries(result, target.features(eval_x))
    truth = target.labels(eval_x)
    return {
        "input_features": n_features,
        "hidden_features": hidden_features,
        "classes": n_classes,
        "rank": rank,
        "seed": seed,
        "query_count": result.query_count,
        "solved_undirected_pairs": int(result.solved_pairs.sum() // 2),
        "total_undirected_pairs": int(n_classes * (n_classes - 1) // 2),
        "validation_agreement": result.validation_agreement,
        "eval_agreement": float((pred == truth).mean()),
    }


def run_relu_feature_ranking_case(*, n_features: int, hidden_features: int, n_classes: int, rank: int, seed: int) -> dict[str, float | int | bool]:
    target = make_relu_feature_lowrank_target(
        n_features=n_features,
        hidden_features=hidden_features,
        n_classes=n_classes,
        rank=rank,
        seed=seed + 150_000,
    )
    oracle = TopKOracle(target, k=n_classes, include_scores=False)
    result = recover_feature_rank_index_boundaries(
        oracle,
        feature_map=target.features,
        input_dim=target.n_features,
        feature_dim=target.hidden_features,
        n_classes=target.n_classes,
        seed=seed + 160_000,
        max_attempts=120_000,
        validation_points=2048,
    )
    eval_x = random_probe_points(4096, n_features, seed=seed + 170_000, scale=2.5)
    pred = predict_ranking_boundaries(result, target.features(eval_x))
    truth = target.labels(eval_x)
    return {
        "input_features": n_features,
        "hidden_features": hidden_features,
        "classes": n_classes,
        "rank": rank,
        "seed": seed,
        "query_count": result.query_count,
        "solved_undirected_pairs": int(result.solved_pairs.sum() // 2),
        "total_undirected_pairs": int(n_classes * (n_classes - 1) // 2),
        "validation_agreement": result.validation_agreement,
        "eval_agreement": float((pred == truth).mean()),
    }


def run_observable_partial_ranking_case(*, seed: int) -> dict[str, float | int | bool]:
    weights = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ],
        dtype=np.float64,
    )
    target = LinearLowRankTarget(
        base_weight=weights,
        adapter_left=np.zeros((3, 1), dtype=np.float64),
        adapter_right=np.zeros((2, 1), dtype=np.float64),
        bias=np.zeros(3, dtype=np.float64),
    )
    result = recover_feature_rank_index_boundaries(
        TopKOracle(target, k=2, include_scores=False),
        feature_map=lambda x: np.atleast_2d(np.asarray(x, dtype=np.float64)),
        input_dim=2,
        feature_dim=2,
        n_classes=3,
        seed=seed + 180_000,
        max_attempts=20_000,
        validation_points=2048,
    )
    eval_x = random_probe_points(4096, 2, seed=seed + 190_000, scale=2.5)
    pred = predict_ranking_boundaries(result, eval_x)
    truth = target.labels(eval_x)
    return {
        "input_features": 2,
        "classes": 3,
        "topk": 2,
        "seed": seed,
        "query_count": result.query_count,
        "solved_undirected_pairs": int(result.solved_pairs.sum() // 2),
        "total_undirected_pairs": 3,
        "validation_agreement": result.validation_agreement,
        "eval_agreement": float((pred == truth).mean()),
    }


def run_lowrank_assisted_topk_case(*, seed: int) -> dict[str, float | int | bool]:
    target = make_relu_feature_lowrank_target(
        n_features=6,
        hidden_features=10,
        n_classes=6,
        rank=2,
        seed=seed + 200_000,
    )
    rng = np.random.default_rng(seed + 210_000)
    query_count = 6000
    x = rng.normal(0.0, 2.0, size=(query_count, target.n_features))
    z = target.features(x)
    oracle = TopKOracle(target, k=3, include_scores=True)
    indices, scores = oracle.query(x)
    if scores is None:
        raise RuntimeError("expected score-bearing top-k transcript")

    full_budget = target.hidden_features + 1
    lowrank_budget = target.rank + 1
    seed_classes = {0, 1}
    budgets = {
        klass: (full_budget if klass in seed_classes else lowrank_budget)
        for klass in range(target.n_classes)
    }
    used = {klass: 0 for klass in range(target.n_classes)}
    observations: list[tuple[int, np.ndarray, float]] = []
    for row_idx in range(query_count):
        for pos, klass_raw in enumerate(indices[row_idx]):
            klass = int(klass_raw)
            if used[klass] >= budgets[klass]:
                continue
            observations.append((klass, z[row_idx], float(scores[row_idx, pos])))
            used[klass] += 1
        if all(used[klass] >= budgets[klass] for klass in range(target.n_classes)):
            break

    result = recover_lowrank_assisted_feature_head_from_observations(
        observations,
        base_head=target.base_head,
        rank=target.rank,
        query_count=oracle.query_count,
    )
    eval_x = random_probe_points(4096, target.n_features, seed=seed + 220_000)
    eval_z = target.features(eval_x)
    return {
        "input_features": target.n_features,
        "hidden_features": target.hidden_features,
        "classes": target.n_classes,
        "rank": target.rank,
        "seed": seed,
        "observation_count": result.observation_count,
        "query_count": result.query_count,
        "row_space_rank": result.row_space_rank,
        "full_row_solved_classes": int(result.full_row_solved_classes.sum()),
        "lowrank_solved_classes": int(result.lowrank_solved_classes.sum()),
        "solved_classes": int(result.solved_classes.sum()),
        "generic_row_solved_classes": int(np.sum(result.observations_per_class >= target.hidden_features + 1)),
        "head_delta_relerr": relative_frobenius_error(result.recovered_delta, target.delta),
        "eval_agreement": label_agreement(result.recovered_head, result.recovered_bias, target.head, target.bias, eval_z),
    }


def run_hidden_lora_block_case(*, seed: int) -> dict[str, float | int | bool]:
    target = make_hidden_lora_block_target(
        n_features=6,
        hidden_features=4,
        n_classes=5,
        rank=2,
        seed=seed + 230_000,
        hidden_bias=4.0,
    )
    inversion_oracle = RawLogitOracle(target)
    query_x = np.vstack([np.zeros(target.n_features, dtype=np.float64), 1e-2 * np.eye(target.n_features, dtype=np.float64)])
    endpoint_logits = inversion_oracle.query(query_x)
    true_hidden = np.maximum(0.0, query_x @ target.hidden_weight.T + target.hidden_bias)
    inverted_hidden = invert_endpoint_hidden_states(target.output_head, target.output_bias, endpoint_logits)
    bias_strength = 8.0
    biased_log_probs = np.empty_like(endpoint_logits)
    for klass in range(target.n_classes):
        biased_logits = endpoint_logits.copy()
        biased_logits[:, klass] += bias_strength
        shifted = biased_logits - np.max(biased_logits, axis=1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        biased_log_probs[:, klass] = log_probs[:, klass]
    reconstructed_log_probs = unbiased_log_probs_from_biased_token_log_probs(biased_log_probs, bias=bias_strength)
    top1_bias_hidden = invert_hidden_states_from_log_probs(
        target.output_head,
        target.output_bias,
        reconstructed_log_probs,
    )
    oracle = RawLogitOracle(target)
    result = recover_hidden_lora_block_on_positive_cell(
        oracle,
        base_hidden_weight=target.base_hidden_weight,
        hidden_bias=target.hidden_bias,
        output_head=target.output_head,
        output_bias=target.output_bias,
        rank=target.rank,
    )
    eval_x = random_probe_points(4096, target.n_features, seed=seed + 240_000, scale=0.05)
    recovered_hidden = np.maximum(0.0, eval_x @ result.recovered_hidden_weight.T + result.recovered_hidden_bias)
    recovered_logits = recovered_hidden @ target.output_head.T + target.output_bias
    true_logits = target.logits(eval_x)
    agreement = float(np.mean(np.argmax(recovered_logits, axis=1) == target.labels(eval_x)))
    return {
        "input_features": target.n_features,
        "hidden_features": target.hidden_features,
        "classes": target.n_classes,
        "rank": target.rank,
        "seed": seed,
        "query_count": result.query_count,
        "output_head_rank": result.output_head_rank,
        "activation_margin": result.activation_margin,
        "endpoint_hidden_reconstruction_relerr": relative_frobenius_error(inverted_hidden, true_hidden),
        "top1_bias_hidden_reconstruction_relerr": relative_frobenius_error(top1_bias_hidden, true_hidden),
        "top1_bias_endpoint_calls": int(query_x.shape[0] * target.n_classes),
        "hidden_bias_max_abs_error": float(np.max(np.abs(result.recovered_hidden_bias - target.hidden_bias))),
        "hidden_delta_relerr": relative_frobenius_error(result.recovered_delta, target.delta),
        "endpoint_logit_max_abs_error": float(np.max(np.abs(recovered_logits - true_logits))),
        "eval_agreement": agreement,
    }


def _estimate_rank_from_singular_values(singular_values: np.ndarray, *, relative_threshold: float = 1e-6) -> int:
    if singular_values.size == 0:
        return 0
    threshold = relative_threshold * max(float(singular_values[0]), 1e-300)
    return int(np.sum(singular_values > threshold))


def run_unknown_rank_case(*, seed: int) -> dict[str, float | int | bool]:
    target = make_linear_lowrank_target(
        n_features=24,
        n_classes=10,
        rank=3,
        seed=seed + 250_000,
    )
    oracle = RawLogitOracle(target)
    recovery = recover_raw_affine(oracle, base_weight=target.base_weight, rank=target.rank)
    exact_singular_values = np.linalg.svd(recovery.recovered_delta, compute_uv=False)
    exact_estimated_rank = _estimate_rank_from_singular_values(exact_singular_values)

    design_x = np.vstack([np.zeros(target.n_features, dtype=np.float64), np.eye(target.n_features, dtype=np.float64)])
    exact_logits = target.logits(design_x)
    rounded_logits = np.round(exact_logits, 3)
    theta, *_ = np.linalg.lstsq(np.hstack([design_x, np.ones((design_x.shape[0], 1))]), rounded_logits, rcond=None)
    rounded_delta = theta[:-1].T - target.base_weight
    rounded_singular_values = np.linalg.svd(rounded_delta, compute_uv=False)
    rounded_estimated_rank = _estimate_rank_from_singular_values(rounded_singular_values)

    rng = np.random.default_rng(seed + 260_000)
    noisy_logits = exact_logits + rng.normal(0.0, 1e-3, size=exact_logits.shape)
    noisy_theta, *_ = np.linalg.lstsq(np.hstack([design_x, np.ones((design_x.shape[0], 1))]), noisy_logits, rcond=None)
    noisy_delta = noisy_theta[:-1].T - target.base_weight
    noisy_singular_values = np.linalg.svd(noisy_delta, compute_uv=False)
    noisy_estimated_rank = _estimate_rank_from_singular_values(noisy_singular_values)

    sigma_r = float(exact_singular_values[target.rank - 1])
    sigma_next = float(exact_singular_values[target.rank]) if target.rank < exact_singular_values.size else 0.0
    return {
        "features": target.n_features,
        "classes": target.n_classes,
        "true_rank": target.rank,
        "seed": seed,
        "exact_estimated_rank": exact_estimated_rank,
        "rounded_estimated_rank": rounded_estimated_rank,
        "noisy_estimated_rank": noisy_estimated_rank,
        "sigma_r": sigma_r,
        "sigma_r_plus_1": sigma_next,
        "spectral_gap": sigma_r / max(sigma_next, 1e-300),
        "exact_delta_relerr": relative_frobenius_error(recovery.recovered_delta, target.delta),
        "rounded_delta_relerr": relative_frobenius_error(rounded_delta, target.delta),
        "noisy_delta_relerr": relative_frobenius_error(noisy_delta, target.delta),
        "exact_rank_success": exact_estimated_rank == target.rank,
        "rounded_rank_success": rounded_estimated_rank == target.rank,
        "noisy_rank_success": noisy_estimated_rank == target.rank,
    }


def summarize(rows: list[dict[str, float | int | bool]]) -> str:
    by_feature: dict[int, list[dict[str, float | int | bool]]] = {}
    for row in rows:
        by_feature.setdefault(int(row["features"]), []).append(row)

    lines = [
        "# Ablation Summary",
        "",
        "| features | cases | raw delta relerr mean | top-k solved mean | top-k agreement mean | binary agreement mean |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n_features, group in sorted(by_feature.items()):
        cases = len(group)
        raw_err = sum(float(row["raw_delta_relerr"]) for row in group) / cases
        solved = sum(float(row["topk_solved_classes"]) for row in group) / cases
        topk_agreement = sum(float(row["topk_agreement"]) for row in group) / cases
        binary_agreement = sum(float(row["binary_agreement"]) for row in group) / cases
        lines.append(
            f"| {n_features} | {cases} | {raw_err:.2e} | {solved:.2f} | "
            f"{topk_agreement:.4f} | {binary_agreement:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: raw-logit recovery is exact to numerical precision; scored top-k recovery succeeds when every class appears in enough independent returned scores; binary hard-label recovery reconstructs the decision boundary up to scale.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize_unknown_rank(rows: list[dict[str, float | int | bool]]) -> str:
    if not rows:
        return ""
    cases = len(rows)
    exact_success = sum(float(row["exact_rank_success"]) for row in rows) / cases
    rounded_success = sum(float(row["rounded_rank_success"]) for row in rows) / cases
    noisy_success = sum(float(row["noisy_rank_success"]) for row in rows) / cases
    exact_rank = sum(float(row["exact_estimated_rank"]) for row in rows) / cases
    rounded_rank = sum(float(row["rounded_estimated_rank"]) for row in rows) / cases
    noisy_rank = sum(float(row["noisy_estimated_rank"]) for row in rows) / cases
    gap = sum(float(row["spectral_gap"]) for row in rows) / cases
    rounded_err = sum(float(row["rounded_delta_relerr"]) for row in rows) / cases
    noisy_err = sum(float(row["noisy_delta_relerr"]) for row in rows) / cases
    return "\n".join(
        [
            "## Unknown-Rank Recovery Diagnostic",
            "",
            "| cases | true rank | exact est. rank | rounded est. rank | noisy est. rank | exact success | rounded success | noisy success | spectral gap |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {cases} | {int(rows[0]['true_rank'])} | {exact_rank:.2f} | {rounded_rank:.2f} | {noisy_rank:.2f} | "
            f"{exact_success:.2f} | {rounded_success:.2f} | {noisy_success:.2f} | {gap:.2e} |",
            "",
            f"Mean rounded/noisy adapter relative errors: {rounded_err:.2e} / {noisy_err:.2e}.",
            "",
            "Interpretation: exact logits reveal the numerical rank through the singular spectrum of the recovered update. Rounding and noise can inflate small singular values, so unknown-rank recovery should be reported with its threshold and failure modes rather than assumed.",
            "",
        ]
    )


def summarize_multiclass_boundary(rows: list[dict[str, float | int | bool]]) -> str:
    if not rows:
        return ""
    cases = len(rows)
    solved = sum(float(row["solved_undirected_pairs"]) for row in rows) / cases
    total = sum(float(row["total_undirected_pairs"]) for row in rows) / cases
    agreement = sum(float(row["eval_agreement"]) for row in rows) / cases
    queries = sum(float(row["query_count"]) for row in rows) / cases
    return "\n".join(
        [
            "## Multiclass Hard-Label Boundary Recovery",
            "",
            "| cases | classes | features | solved pairs mean | eval agreement mean | queries mean |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {cases} | {int(rows[0]['classes'])} | {int(rows[0]['features'])} | "
            f"{solved:.2f}/{total:.2f} | {agreement:.4f} | {queries:.1f} |",
            "",
            "Interpretation: when all pairwise affine boundaries are recovered, pairwise voting reconstructs the multiclass hard-label classifier up to boundary-search precision. This still does not identify logit scale or adapter magnitude.",
            "",
        ]
    )


def summarize_relu_feature(rows: list[dict[str, float | int | bool]]) -> str:
    if not rows:
        return ""
    cases = len(rows)
    relerr = sum(float(row["head_delta_relerr"]) for row in rows) / cases
    agreement = sum(float(row["eval_agreement"]) for row in rows) / cases
    queries = sum(float(row["query_count"]) for row in rows) / cases
    topk_solved = sum(float(row["topk_solved_classes"]) for row in rows) / cases
    topk_relerr = sum(float(row["topk_head_delta_relerr"]) for row in rows) / cases
    topk_agreement = sum(float(row["topk_eval_agreement"]) for row in rows) / cases
    topk_queries = sum(float(row["topk_query_count"]) for row in rows) / cases
    return "\n".join(
        [
            "## ReLU Feature-Head Raw-Logit Recovery",
            "",
            "| cases | input features | hidden features | classes | head relerr mean | eval agreement mean | queries mean |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {cases} | {int(rows[0]['input_features'])} | {int(rows[0]['hidden_features'])} | "
            f"{int(rows[0]['classes'])} | {relerr:.2e} | {agreement:.4f} | {queries:.1f} |",
            "",
            "Interpretation: if the public ReLU feature map can be queried at enough points to span feature space, the low-rank adapted classification head is recovered exactly from raw logits.",
            "",
            "## ReLU Feature-Head Score-Bearing Top-k Recovery",
            "",
            "| cases | top-k solved classes mean | top-k head relerr mean | top-k agreement mean | top-k queries mean |",
            "|---:|---:|---:|---:|---:|",
            f"| {cases} | {topk_solved:.2f} | {topk_relerr:.2e} | {topk_agreement:.4f} | {topk_queries:.1f} |",
            "",
            "Interpretation: score-bearing top-k outputs also recover the public-feature adapted head when every class row appears in enough independent returned-score equations.",
            "",
        ]
    )


def summarize_relu_feature_hard_label(rows: list[dict[str, float | int | bool]]) -> str:
    if not rows:
        return ""
    cases = len(rows)
    solved = sum(float(row["solved_undirected_pairs"]) for row in rows) / cases
    total = sum(float(row["total_undirected_pairs"]) for row in rows) / cases
    agreement = sum(float(row["eval_agreement"]) for row in rows) / cases
    queries = sum(float(row["query_count"]) for row in rows) / cases
    return "\n".join(
        [
            "## ReLU Feature-Head Hard-Label Boundary Recovery",
            "",
            "| cases | input features | hidden features | classes | solved pairs mean | eval agreement mean | queries mean |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {cases} | {int(rows[0]['input_features'])} | {int(rows[0]['hidden_features'])} | "
            f"{int(rows[0]['classes'])} | {solved:.2f}/{total:.2f} | {agreement:.4f} | {queries:.1f} |",
            "",
            "Interpretation: hard labels recover pairwise decision boundaries in public feature space up to scale, giving functional equivalence by pairwise voting when all boundaries are solved.",
            "",
        ]
    )


def summarize_relu_feature_ranking(rows: list[dict[str, float | int | bool]]) -> str:
    if not rows:
        return ""
    cases = len(rows)
    solved = sum(float(row["solved_undirected_pairs"]) for row in rows) / cases
    total = sum(float(row["total_undirected_pairs"]) for row in rows) / cases
    agreement = sum(float(row["eval_agreement"]) for row in rows) / cases
    queries = sum(float(row["query_count"]) for row in rows) / cases
    return "\n".join(
        [
            "## ReLU Feature-Head Index-Only Ranking Boundary Recovery",
            "",
            "| cases | input features | hidden features | classes | solved pairs mean | eval agreement mean | queries mean |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {cases} | {int(rows[0]['input_features'])} | {int(rows[0]['hidden_features'])} | "
            f"{int(rows[0]['classes'])} | {solved:.2f}/{total:.2f} | {agreement:.4f} | {queries:.1f} |",
            "",
            "Interpretation: index-only full rankings recover pairwise decision boundaries in public feature space without score values, by bisection on pairwise rank-order flips.",
            "",
        ]
    )


def summarize_observable_partial_ranking(rows: list[dict[str, float | int | bool]]) -> str:
    if not rows:
        return ""
    cases = len(rows)
    solved = sum(float(row["solved_undirected_pairs"]) for row in rows) / cases
    total = sum(float(row["total_undirected_pairs"]) for row in rows) / cases
    agreement = sum(float(row["eval_agreement"]) for row in rows) / cases
    queries = sum(float(row["query_count"]) for row in rows) / cases
    return "\n".join(
        [
            "## Observable Partial Top-k Index-Only Recovery",
            "",
            "| cases | classes | top-k | solved pairs mean | eval agreement mean | queries mean |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {cases} | {int(rows[0]['classes'])} | {int(rows[0]['topk'])} | "
            f"{solved:.2f}/{total:.2f} | {agreement:.4f} | {queries:.1f} |",
            "",
            "Interpretation: partial top-k index-only recovery succeeds when every pairwise boundary is co-visible in the returned set on enough independent boundary queries.",
            "",
        ]
    )


def summarize_lowrank_assisted_topk(rows: list[dict[str, float | int | bool]]) -> str:
    if not rows:
        return ""
    cases = len(rows)
    generic = sum(float(row["generic_row_solved_classes"]) for row in rows) / cases
    full = sum(float(row["full_row_solved_classes"]) for row in rows) / cases
    lowrank = sum(float(row["lowrank_solved_classes"]) for row in rows) / cases
    solved = sum(float(row["solved_classes"]) for row in rows) / cases
    relerr = sum(float(row["head_delta_relerr"]) for row in rows) / cases
    agreement = sum(float(row["eval_agreement"]) for row in rows) / cases
    observations = sum(float(row["observation_count"]) for row in rows) / cases
    return "\n".join(
        [
            "## Low-Rank-Assisted Score-Bearing Top-k Recovery",
            "",
            "| cases | hidden features | rank | generic rows | full rows | low-rank rows | solved classes | observations | agreement |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {cases} | {int(rows[0]['hidden_features'])} | {int(rows[0]['rank'])} | "
            f"{generic:.2f} | {full:.2f} | {lowrank:.2f} | {solved:.2f} | {observations:.1f} | {agreement:.4f} |",
            "",
            f"Mean relative adapter error: {relerr:.2e}.",
            "",
            "Interpretation: after a few full rows identify the adapter row space, classes with only rank+1 scored observations are recovered; a generic row-wise attack would leave those rows unsolved.",
            "",
        ]
    )


def summarize_hidden_lora_block(rows: list[dict[str, float | int | bool]]) -> str:
    if not rows:
        return ""
    cases = len(rows)
    relerr = sum(float(row["hidden_delta_relerr"]) for row in rows) / cases
    agreement = sum(float(row["eval_agreement"]) for row in rows) / cases
    queries = sum(float(row["query_count"]) for row in rows) / cases
    margin = sum(float(row["activation_margin"]) for row in rows) / cases
    hidden_recon = sum(float(row["endpoint_hidden_reconstruction_relerr"]) for row in rows) / cases
    top1_bias_hidden_recon = sum(float(row["top1_bias_hidden_reconstruction_relerr"]) for row in rows) / cases
    top1_bias_calls = sum(float(row["top1_bias_endpoint_calls"]) for row in rows) / cases
    logit_err = sum(float(row["endpoint_logit_max_abs_error"]) for row in rows) / cases
    return "\n".join(
        [
            "## Endpoint Hidden LoRA Block Recovery on One ReLU Cell",
            "",
            "| cases | input features | hidden features | classes | rank | full-logit hidden recon | top-1-bias hidden recon | adapter relerr | max logit err | agreement | queries | top-1-bias calls |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {cases} | {int(rows[0]['input_features'])} | {int(rows[0]['hidden_features'])} | "
            f"{int(rows[0]['classes'])} | {int(rows[0]['rank'])} | {hidden_recon:.2e} | {top1_bias_hidden_recon:.2e} | "
            f"{relerr:.2e} | {logit_err:.2e} | {agreement:.4f} | {queries:.1f} | {top1_bias_calls:.1f} |",
            "",
            f"Mean all-active margin: {margin:.2f}.",
            "",
            "Interpretation: with a public full-column-rank endpoint output head and queries inside a stable all-active ReLU cell, endpoint logits can be inverted to hidden activations and the hidden low-rank adapter is recovered with d+1 local endpoint queries. The same hidden states are also reconstructed from a restricted top-1/logit-bias log-prob interface by forcing one class at a time. This is a positive endpoint-hidden recovery result under explicit injectivity and local-cell assumptions, not a claim about rank-deficient classifier endpoints.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = [
        run_case(n_features=n_features, n_classes=args.classes, rank=args.rank, seed=seed)
        for n_features in args.features
        for seed in args.seeds
    ]
    multiclass_rows = [
        run_multiclass_boundary_case(n_features=4, n_classes=3, seed=seed)
        for seed in args.seeds
    ]
    relu_feature_rows = [
        run_relu_feature_case(n_features=8, hidden_features=12, n_classes=5, rank=2, seed=seed)
        for seed in args.seeds
    ]
    relu_hard_label_rows = [
        run_relu_feature_hard_label_case(n_features=5, hidden_features=4, n_classes=3, rank=1, seed=seed)
        for seed in args.feature_hard_label_seeds
    ]
    relu_ranking_rows = [
        run_relu_feature_ranking_case(n_features=5, hidden_features=4, n_classes=3, rank=1, seed=seed)
        for seed in args.feature_hard_label_seeds
    ]
    observable_partial_rows = [
        run_observable_partial_ranking_case(seed=seed)
        for seed in args.seeds
    ]
    lowrank_assisted_rows = [
        run_lowrank_assisted_topk_case(seed=seed)
        for seed in args.seeds
    ]
    hidden_lora_rows = [
        run_hidden_lora_block_case(seed=seed)
        for seed in args.seeds
    ]
    unknown_rank_rows = [
        run_unknown_rank_case(seed=seed)
        for seed in args.seeds
    ]

    json_path = args.outdir / "ablation.json"
    csv_path = args.outdir / "ablation.csv"
    md_path = args.outdir / "summary.md"
    multiclass_json_path = args.outdir / "multiclass_boundary.json"
    multiclass_csv_path = args.outdir / "multiclass_boundary.csv"
    relu_json_path = args.outdir / "relu_feature_head.json"
    relu_csv_path = args.outdir / "relu_feature_head.csv"
    relu_hard_label_json_path = args.outdir / "relu_feature_hard_label.json"
    relu_hard_label_csv_path = args.outdir / "relu_feature_hard_label.csv"
    relu_ranking_json_path = args.outdir / "relu_feature_ranking.json"
    relu_ranking_csv_path = args.outdir / "relu_feature_ranking.csv"
    observable_partial_json_path = args.outdir / "observable_partial_ranking.json"
    observable_partial_csv_path = args.outdir / "observable_partial_ranking.csv"
    lowrank_assisted_json_path = args.outdir / "lowrank_assisted_topk.json"
    lowrank_assisted_csv_path = args.outdir / "lowrank_assisted_topk.csv"
    hidden_lora_json_path = args.outdir / "hidden_lora_block.json"
    hidden_lora_csv_path = args.outdir / "hidden_lora_block.csv"
    unknown_rank_json_path = args.outdir / "unknown_rank.json"
    unknown_rank_csv_path = args.outdir / "unknown_rank.csv"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    multiclass_json_path.write_text(json.dumps(multiclass_rows, indent=2), encoding="utf-8")
    relu_json_path.write_text(json.dumps(relu_feature_rows, indent=2), encoding="utf-8")
    relu_hard_label_json_path.write_text(json.dumps(relu_hard_label_rows, indent=2), encoding="utf-8")
    relu_ranking_json_path.write_text(json.dumps(relu_ranking_rows, indent=2), encoding="utf-8")
    observable_partial_json_path.write_text(json.dumps(observable_partial_rows, indent=2), encoding="utf-8")
    lowrank_assisted_json_path.write_text(json.dumps(lowrank_assisted_rows, indent=2), encoding="utf-8")
    hidden_lora_json_path.write_text(json.dumps(hidden_lora_rows, indent=2), encoding="utf-8")
    unknown_rank_json_path.write_text(json.dumps(unknown_rank_rows, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with multiclass_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(multiclass_rows[0].keys()))
        writer.writeheader()
        writer.writerows(multiclass_rows)
    with relu_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(relu_feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(relu_feature_rows)
    with relu_hard_label_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(relu_hard_label_rows[0].keys()))
        writer.writeheader()
        writer.writerows(relu_hard_label_rows)
    with relu_ranking_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(relu_ranking_rows[0].keys()))
        writer.writeheader()
        writer.writerows(relu_ranking_rows)
    with observable_partial_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(observable_partial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(observable_partial_rows)
    with lowrank_assisted_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lowrank_assisted_rows[0].keys()))
        writer.writeheader()
        writer.writerows(lowrank_assisted_rows)
    with hidden_lora_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hidden_lora_rows[0].keys()))
        writer.writeheader()
        writer.writerows(hidden_lora_rows)
    with unknown_rank_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(unknown_rank_rows[0].keys()))
        writer.writeheader()
        writer.writerows(unknown_rank_rows)
    summary = (
        summarize(rows)
        + "\n"
        + summarize_multiclass_boundary(multiclass_rows)
        + "\n"
        + summarize_relu_feature(relu_feature_rows)
        + "\n"
        + summarize_relu_feature_hard_label(relu_hard_label_rows)
        + "\n"
        + summarize_relu_feature_ranking(relu_ranking_rows)
        + "\n"
        + summarize_observable_partial_ranking(observable_partial_rows)
        + "\n"
        + summarize_lowrank_assisted_topk(lowrank_assisted_rows)
        + "\n"
        + summarize_hidden_lora_block(hidden_lora_rows)
        + "\n"
        + summarize_unknown_rank(unknown_rank_rows)
    )
    md_path.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {multiclass_json_path}")
    print(f"Wrote {multiclass_csv_path}")
    print(f"Wrote {relu_json_path}")
    print(f"Wrote {relu_csv_path}")
    print(f"Wrote {relu_hard_label_json_path}")
    print(f"Wrote {relu_hard_label_csv_path}")
    print(f"Wrote {relu_ranking_json_path}")
    print(f"Wrote {relu_ranking_csv_path}")
    print(f"Wrote {observable_partial_json_path}")
    print(f"Wrote {observable_partial_csv_path}")
    print(f"Wrote {lowrank_assisted_json_path}")
    print(f"Wrote {lowrank_assisted_csv_path}")
    print(f"Wrote {hidden_lora_json_path}")
    print(f"Wrote {hidden_lora_csv_path}")
    print(f"Wrote {unknown_rank_json_path}")
    print(f"Wrote {unknown_rank_csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
