from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from lowrank_extraction.attacks import (
    fit_hard_label_surrogate,
    predict_feature_boundaries,
    predict_ranking_boundaries,
    recover_feature_hard_label_boundaries,
    recover_feature_rank_index_boundaries,
    recover_hidden_lora_block_on_positive_cell,
    recover_topk_scored_feature_head,
)
from lowrank_extraction.metrics import label_agreement, relative_frobenius_error
from lowrank_extraction.models import make_hidden_lora_block_target
from lowrank_extraction.oracles import HardLabelOracle, RawLogitOracle, TopKOracle
from lowrank_extraction.realdata import load_public_features, make_digits_adapter_problem


class FeatureHardLabelOracle:
    def __init__(self, head: np.ndarray, bias: np.ndarray) -> None:
        self.head = head
        self.bias = bias
        self.query_count = 0

    def query(self, z: np.ndarray) -> np.ndarray:
        z2 = np.atleast_2d(np.asarray(z, dtype=np.float64))
        self.query_count += int(z2.shape[0])
        return np.argmax(z2 @ self.head.T + self.bias, axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data public-feature adapter extraction experiments.")
    parser.add_argument("--outdir", type=Path, default=Path("results") / "realdata")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--datasets", nargs="+", choices=["digits", "wine"], default=["digits"])
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--topk-queries", type=int, default=800)
    parser.add_argument("--boundary-attempts", type=int, default=10_000)
    return parser.parse_args()


def _topk_feature_recovery_from_pool(problem, *, x_pool: np.ndarray, k: int, include_scores: bool, seed: int, decimals: int | None = None, noise_std: float = 0.0) -> dict[str, float | int | bool]:
    oracle = TopKOracle(
        problem.target,
        k=k,
        include_scores=include_scores,
        decimals=decimals,
        noise_std=noise_std,
        seed=seed,
    )
    z_pool = problem.data.feature_map(x_pool)
    indices, scores = oracle.query(x_pool)
    n_classes = problem.target.n_classes
    feature_dim = problem.data.feature_map.feature_dim
    rows_by_class: list[list[np.ndarray]] = [[] for _ in range(n_classes)]
    y_by_class: list[list[float]] = [[] for _ in range(n_classes)]
    if scores is not None:
        for i in range(x_pool.shape[0]):
            z_aug = np.concatenate([z_pool[i], np.ones(1, dtype=np.float64)])
            for pos, klass in enumerate(indices[i]):
                rows_by_class[int(klass)].append(z_aug)
                y_by_class[int(klass)].append(float(scores[i, pos]))

    recovered_head = np.array(problem.target.base_head, copy=True)
    recovered_bias = np.zeros(n_classes, dtype=np.float64)
    solved = np.zeros(n_classes, dtype=bool)
    if scores is not None:
        for klass in range(n_classes):
            if len(rows_by_class[klass]) < feature_dim + 1:
                continue
            design = np.vstack(rows_by_class[klass])
            if np.linalg.matrix_rank(design) < feature_dim + 1:
                continue
            theta, *_ = np.linalg.lstsq(design, np.asarray(y_by_class[klass]), rcond=None)
            recovered_head[klass] = theta[:feature_dim]
            recovered_bias[klass] = theta[feature_dim]
            solved[klass] = True

    if scores is None:
        return {
            "query_count": oracle.query_count,
            "solved_classes": 0,
            "head_delta_relerr": float("nan"),
            "test_agreement": 0.0,
            "failure": True,
        }

    return {
        "query_count": oracle.query_count,
        "solved_classes": int(solved.sum()),
        "head_delta_relerr": relative_frobenius_error(recovered_head - problem.target.base_head, problem.target.delta),
        "test_agreement": label_agreement(recovered_head, recovered_bias, problem.target.head, problem.target.bias, problem.data.z_test),
        "failure": bool(not solved.all()),
    }


def _raw_feature_recovery_from_pool(problem) -> dict[str, np.ndarray | int]:
    feature_dim = problem.data.feature_map.feature_dim
    design_rows: list[np.ndarray] = []
    chosen_x: list[np.ndarray] = []
    for x, z in zip(problem.data.x_train, problem.data.z_train, strict=True):
        row = np.concatenate([z, np.ones(1, dtype=np.float64)])
        candidate = design_rows + [row]
        if np.linalg.matrix_rank(np.vstack(candidate), tol=1e-10) > len(design_rows):
            design_rows.append(row)
            chosen_x.append(x)
        if len(design_rows) == feature_dim + 1:
            break
    if len(design_rows) < feature_dim + 1:
        raise RuntimeError("training feature pool does not span the adapted head space")

    oracle = RawLogitOracle(problem.target)
    design = np.vstack(design_rows)
    values = oracle.query(np.vstack(chosen_x))
    theta, *_ = np.linalg.lstsq(design, values, rcond=None)
    recovered_head = theta[:feature_dim, :].T
    recovered_bias = theta[feature_dim, :]
    return {
        "recovered_head": recovered_head,
        "recovered_bias": recovered_bias,
        "recovered_delta": recovered_head - problem.target.base_head,
        "query_count": oracle.query_count,
    }


def run_case(seed: int, *, dataset: str = "digits", feature_dim: int = 8, rank: int = 2, topk_queries: int = 800, boundary_attempts: int = 20_000) -> dict[str, float | int | bool | str]:
    problem = make_digits_adapter_problem(seed=seed, feature_dim=feature_dim, rank=rank, dataset=dataset)
    rng = np.random.default_rng(seed + 200_000)
    bisection_steps = 50 if dataset == "digits" else 30

    raw = _raw_feature_recovery_from_pool(problem)

    random_x = rng.normal(0.0, 1.0, size=(topk_queries, problem.data.feature_map.input_dim))
    data_idx = rng.choice(problem.data.x_train.shape[0], size=topk_queries, replace=True)
    data_x = problem.data.x_train[data_idx]

    random_topk = recover_topk_scored_feature_head(
        TopKOracle(problem.target, k=min(2, problem.target.n_classes), include_scores=True),
        base_head=problem.target.base_head,
        feature_map=problem.data.feature_map,
        input_dim=problem.data.feature_map.input_dim,
        feature_dim=problem.data.feature_map.feature_dim,
        rank=problem.target.rank,
        random_queries=topk_queries,
        seed=seed + 220_000,
    )
    data_topk = _topk_feature_recovery_from_pool(problem, x_pool=data_x, k=min(2, problem.target.n_classes), include_scores=True, seed=seed + 230_000)
    rounded_topk = _topk_feature_recovery_from_pool(problem, x_pool=random_x, k=min(2, problem.target.n_classes), include_scores=True, decimals=1, seed=seed + 240_000)
    noisy_topk = _topk_feature_recovery_from_pool(problem, x_pool=random_x, k=min(2, problem.target.n_classes), include_scores=True, noise_std=0.05, seed=seed + 250_000)

    hard_label = recover_feature_hard_label_boundaries(
        HardLabelOracle(problem.target),
        feature_map=problem.data.feature_map,
        input_dim=problem.data.feature_map.input_dim,
        feature_dim=problem.data.feature_map.feature_dim,
        n_classes=problem.target.n_classes,
        seed=seed + 260_000,
        max_attempts=boundary_attempts,
        bisection_steps=bisection_steps,
        validation_points=1024,
    )
    hard_pred = predict_feature_boundaries(hard_label, problem.data.z_test)
    truth = problem.target.labels(problem.data.x_test)

    full_ranking = recover_feature_rank_index_boundaries(
        TopKOracle(problem.target, k=problem.target.n_classes, include_scores=False),
        feature_map=problem.data.feature_map,
        input_dim=problem.data.feature_map.input_dim,
        feature_dim=problem.data.feature_map.feature_dim,
        n_classes=problem.target.n_classes,
        seed=seed + 270_000,
        max_attempts=boundary_attempts,
        bisection_steps=bisection_steps,
        validation_points=1024,
    )
    ranking_pred = predict_ranking_boundaries(full_ranking, problem.data.z_test)

    partial_ranking = recover_feature_rank_index_boundaries(
        TopKOracle(problem.target, k=max(1, problem.target.n_classes - 1), include_scores=False),
        feature_map=problem.data.feature_map,
        input_dim=problem.data.feature_map.input_dim,
        feature_dim=problem.data.feature_map.feature_dim,
        n_classes=problem.target.n_classes,
        seed=seed + 280_000,
        max_attempts=boundary_attempts // 3,
        bisection_steps=bisection_steps,
        validation_points=512,
    )

    surrogate = fit_hard_label_surrogate(
        FeatureHardLabelOracle(problem.target.head, problem.target.bias),
        n_features=problem.data.feature_map.feature_dim,
        n_classes=problem.target.n_classes,
        n_queries=min(800, problem.data.z_train.shape[0]),
        seed=seed + 290_000,
    )

    return {
        "dataset": dataset,
        "seed": seed,
        "feature_dim": feature_dim,
        "rank": rank,
        "classes": problem.target.n_classes,
        "base_test_accuracy": problem.base_test_accuracy,
        "adapted_test_accuracy": problem.adapted_test_accuracy,
        "rank_energy": problem.rank_energy,
        "raw_query_count": int(raw["query_count"]),
        "raw_head_relerr": relative_frobenius_error(raw["recovered_delta"], problem.target.delta),
        "raw_test_agreement": label_agreement(raw["recovered_head"], raw["recovered_bias"], problem.target.head, problem.target.bias, problem.data.z_test),
        "random_topk_query_count": random_topk.query_count,
        "random_topk_solved_classes": int(random_topk.solved_classes.sum()),
        "random_topk_head_relerr": relative_frobenius_error(random_topk.recovered_delta, problem.target.delta),
        "random_topk_test_agreement": label_agreement(random_topk.recovered_head, random_topk.recovered_bias, problem.target.head, problem.target.bias, problem.data.z_test),
        "data_topk_query_count": int(data_topk["query_count"]),
        "data_topk_solved_classes": int(data_topk["solved_classes"]),
        "data_topk_head_relerr": float(data_topk["head_delta_relerr"]),
        "data_topk_test_agreement": float(data_topk["test_agreement"]),
        "rounded_topk_head_relerr": float(rounded_topk["head_delta_relerr"]),
        "rounded_topk_test_agreement": float(rounded_topk["test_agreement"]),
        "noisy_topk_head_relerr": float(noisy_topk["head_delta_relerr"]),
        "noisy_topk_test_agreement": float(noisy_topk["test_agreement"]),
        "hard_label_solved_pairs": int(hard_label.solved_pairs.sum() // 2),
        "hard_label_total_pairs": int(problem.target.n_classes * (problem.target.n_classes - 1) // 2),
        "hard_label_query_count": hard_label.query_count,
        "hard_label_test_agreement": float(np.mean(hard_pred == truth)),
        "full_ranking_solved_pairs": int(full_ranking.solved_pairs.sum() // 2),
        "full_ranking_query_count": full_ranking.query_count,
        "full_ranking_test_agreement": float(np.mean(ranking_pred == truth)),
        "partial_ranking_solved_pairs": int(partial_ranking.solved_pairs.sum() // 2),
        "partial_ranking_failure": bool((partial_ranking.solved_pairs.sum() // 2) < (problem.target.n_classes * (problem.target.n_classes - 1) // 2)),
        "surrogate_train_agreement": surrogate.train_agreement,
        "surrogate_test_agreement": label_agreement(surrogate.weight, surrogate.bias, problem.target.head, problem.target.bias, problem.data.z_test),
    }


def run_hidden_block_realdata_case(seed: int, *, dataset: str = "digits", rank: int = 2) -> dict[str, float | int | bool | str]:
    data = load_public_features(seed=seed, feature_dim=8, dataset=dataset)
    hidden_features = 6
    target = make_hidden_lora_block_target(
        n_features=data.x_train.shape[1],
        hidden_features=hidden_features,
        n_classes=hidden_features + 2,
        rank=rank,
        seed=seed + 310_000,
        hidden_bias=8.0,
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
    recovered_hidden = np.maximum(0.0, data.x_test @ result.recovered_hidden_weight.T + result.recovered_hidden_bias)
    recovered_logits = recovered_hidden @ target.output_head.T + target.output_bias
    all_active_rate = float(np.mean(np.all(target.hidden_preactivations(data.x_test) > 0.0, axis=1)))
    return {
        "dataset": dataset,
        "seed": seed,
        "input_features": int(data.x_train.shape[1]),
        "hidden_features": hidden_features,
        "classes": target.n_classes,
        "rank": target.rank,
        "query_count": result.query_count,
        "output_head_rank": result.output_head_rank,
        "activation_margin": result.activation_margin,
        "test_all_active_rate": all_active_rate,
        "hidden_delta_relerr": relative_frobenius_error(result.recovered_delta, target.delta),
        "test_agreement": float(np.mean(np.argmax(recovered_logits, axis=1) == target.labels(data.x_test))),
    }


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if np.isfinite(value)]


def summarize_metric(rows: list[dict[str, float | int | bool | str]], key: str) -> tuple[float, float]:
    vals = _finite([float(row[key]) for row in rows])
    if not vals:
        return float("nan"), float("nan")
    return mean(vals), pstdev(vals) if len(vals) > 1 else 0.0


def _markdown_summary_group(rows: list[dict[str, float | int | bool | str]], *, heading: str) -> list[str]:
    metrics = [
        ("raw_test_agreement", "Raw logits agreement"),
        ("random_topk_test_agreement", "Random scored top-k agreement"),
        ("data_topk_test_agreement", "Dataset-manifold scored top-k agreement"),
        ("rounded_topk_test_agreement", "Rounded scored top-k agreement"),
        ("noisy_topk_test_agreement", "Noisy scored top-k agreement"),
        ("hard_label_test_agreement", "Hard-label boundary agreement"),
        ("full_ranking_test_agreement", "Full-ranking boundary agreement"),
        ("surrogate_test_agreement", "Hard-label surrogate agreement"),
    ]
    lines = [
        f"## {heading}",
        "",
        f"Seeds: {len(rows)}",
        "",
        "| metric | mean | std |",
        "|---|---:|---:|",
    ]
    for key, label in metrics:
        avg, std = summarize_metric(rows, key)
        lines.append(f"| {label} | {avg:.4f} | {std:.4f} |")

    partial_failures = [1.0 if bool(row["partial_ranking_failure"]) else 0.0 for row in rows]
    partial_fail_rate = mean(partial_failures)
    partial_fail_std = pstdev(partial_failures) if len(partial_failures) > 1 else 0.0
    solved_pairs, solved_pairs_std = summarize_metric(rows, "full_ranking_solved_pairs")
    hard_pairs, hard_pairs_std = summarize_metric(rows, "hard_label_solved_pairs")
    lines.extend(
        [
            "",
            "| diagnostic | mean | std |",
            "|---|---:|---:|",
            f"| Full-ranking solved pairs | {solved_pairs:.2f} | {solved_pairs_std:.2f} |",
            f"| Hard-label solved pairs | {hard_pairs:.2f} | {hard_pairs_std:.2f} |",
            f"| Partial-ranking failure rate | {partial_fail_rate:.2f} | {partial_fail_std:.2f} |",
            "",
            "| query distribution | mean | std | min | max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, label in [
        ("raw_query_count", "Raw logits"),
        ("random_topk_query_count", "Random scored top-k"),
        ("data_topk_query_count", "Dataset-manifold scored top-k"),
        ("hard_label_query_count", "Hard-label boundary"),
        ("full_ranking_query_count", "Full-ranking boundary"),
    ]:
        vals = [float(row[key]) for row in rows]
        lines.append(
            f"| {label} | {mean(vals):.1f} | {(pstdev(vals) if len(vals) > 1 else 0.0):.1f} | {min(vals):.0f} | {max(vals):.0f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: real public features validate the public-feature attack path, while partial top-k index-only output is explicitly tracked as a failure/defense setting when not all pairwise comparisons are observable.",
        ]
    )
    return lines


def markdown_summary(rows: list[dict[str, float | int | bool | str]]) -> str:
    datasets = sorted({str(row["dataset"]) for row in rows})
    lines = ["# Real-Data Adapter Summary", ""]
    for dataset in datasets:
        group = [row for row in rows if row["dataset"] == dataset]
        lines.extend(_markdown_summary_group(group, heading=f"{dataset} public-feature adapted head"))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def hidden_block_markdown_summary(rows: list[dict[str, float | int | bool | str]]) -> str:
    if not rows:
        return ""
    datasets = sorted({str(row["dataset"]) for row in rows})
    lines = ["# Real-Data Hidden LoRA Block Summary", ""]
    for dataset in datasets:
        group = [row for row in rows if row["dataset"] == dataset]
        agreement, agreement_std = summarize_metric(group, "test_agreement")
        relerr, relerr_std = summarize_metric(group, "hidden_delta_relerr")
        active, active_std = summarize_metric(group, "test_all_active_rate")
        queries, queries_std = summarize_metric(group, "query_count")
        lines.extend(
            [
                f"## {dataset} inputs with synthetic hidden LoRA block",
                "",
                f"Seeds: {len(group)}",
                "",
                "| metric | mean | std |",
                "|---|---:|---:|",
                f"| Hidden adapter relative error | {relerr:.2e} | {relerr_std:.2e} |",
                f"| Test functional agreement | {agreement:.4f} | {agreement_std:.4f} |",
                f"| Test all-active rate | {active:.4f} | {active_std:.4f} |",
                f"| Query count | {queries:.1f} | {queries_std:.1f} |",
                "",
                "Interpretation: this is a real-input-distribution hidden-adapter stress test, not a pretrained transformer LoRA checkpoint. It validates the local-cell transfer theorem on held-out dataset inputs when the activation-cell condition holds.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        rows.extend(
            run_case(
                seed,
                dataset=dataset,
                feature_dim=args.feature_dim,
                rank=args.rank,
                topk_queries=args.topk_queries,
                boundary_attempts=args.boundary_attempts,
            )
            for seed in args.seeds
        )
    hidden_rows = [
        run_hidden_block_realdata_case(seed, dataset=dataset, rank=args.rank)
        for dataset in args.datasets
        for seed in args.seeds
    ]

    json_path = args.outdir / "digits_adapter.json"
    csv_path = args.outdir / "digits_adapter.csv"
    md_path = args.outdir / "summary.md"
    hidden_json_path = args.outdir / "hidden_lora_digits.json"
    hidden_csv_path = args.outdir / "hidden_lora_digits.csv"
    hidden_md_path = args.outdir / "hidden_lora_summary.md"
    json_path.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    hidden_json_path.write_text(json.dumps(hidden_rows, indent=2, allow_nan=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with hidden_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hidden_rows[0].keys()))
        writer.writeheader()
        writer.writerows(hidden_rows)
    summary = markdown_summary(rows)
    hidden_summary = hidden_block_markdown_summary(hidden_rows)
    md_path.write_text(summary, encoding="utf-8")
    hidden_md_path.write_text(hidden_summary, encoding="utf-8")
    print(summary)
    print(hidden_summary)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {hidden_json_path}")
    print(f"Wrote {hidden_csv_path}")
    print(f"Wrote {hidden_md_path}")


if __name__ == "__main__":
    main()
