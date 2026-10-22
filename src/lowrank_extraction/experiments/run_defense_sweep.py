from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from lowrank_extraction.experiments.run_realdata import (
    _raw_feature_recovery_from_pool,
    _topk_feature_recovery_from_pool,
)
from lowrank_extraction.metrics import label_agreement, relative_frobenius_error
from lowrank_extraction.realdata import make_digits_adapter_problem


Row = dict[str, float | int | bool | str | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run utility-vs-leakage defense sweeps for restricted adapter-extraction oracles.")
    parser.add_argument("--outdir", type=Path, default=Path("results") / "defense_sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--datasets", nargs="+", choices=["digits", "wine"], default=["digits"])
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--topk-queries", type=int, default=800)
    parser.add_argument("--round-decimals", type=int, nargs="+", default=[3, 2, 1, 0])
    parser.add_argument("--noise-stds", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.05, 0.1])
    parser.add_argument("--endpoint-json", type=Path, default=Path("results") / "endpoint_transformer_bert_base" / "endpoint_transformer.json")
    parser.add_argument(
        "--lora-json",
        type=Path,
        default=Path("results") / "lora_checkpoint_sst2_bert_base_stronger" / "lora_checkpoint.json",
    )
    return parser.parse_args()


def _finite(values: list[float | int | bool | str | None]) -> list[float]:
    output: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if np.isfinite(number):
            output.append(number)
    return output


def _mean_std(rows: list[Row], key: str) -> tuple[float, float]:
    values = _finite([row.get(key) for row in rows])
    if not values:
        return float("nan"), float("nan")
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def _clean_number(value: float | int | bool | str | None) -> float | int | bool | str | None:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _append_public_feature_rows(
    rows: list[Row],
    *,
    seed: int,
    dataset: str,
    feature_dim: int,
    rank: int,
    topk_queries: int,
    round_decimals: list[int],
    noise_stds: list[float],
) -> None:
    problem = make_digits_adapter_problem(seed=seed, dataset=dataset, feature_dim=feature_dim, rank=rank)
    rng = np.random.default_rng(seed + 1_000_000)
    random_x = rng.normal(0.0, 1.0, size=(topk_queries, problem.data.feature_map.input_dim))
    data_idx = rng.choice(problem.data.x_train.shape[0], size=topk_queries, replace=True)
    data_x = problem.data.x_train[data_idx]
    k = min(2, problem.target.n_classes)

    raw = _raw_feature_recovery_from_pool(problem)
    rows.append(
        {
            "setting": f"{dataset}_public_feature_head",
            "dataset": dataset,
            "seed": seed,
            "api": "raw_logits",
            "restriction": "exact_scores",
            "query_count": int(raw["query_count"]),
            "solved_classes": problem.target.n_classes,
            "delta_relerr": relative_frobenius_error(raw["recovered_delta"], problem.target.delta),
            "functional_agreement": label_agreement(raw["recovered_head"], raw["recovered_bias"], problem.target.head, problem.target.bias, problem.data.z_test),
            "api_label_utility": 1.0,
            "observable_fraction": 1.0,
            "failure_rate": 0.0,
            "note": "Exact final scores give full-rank linear equations.",
        }
    )

    exact_random = _topk_feature_recovery_from_pool(problem, x_pool=random_x, k=k, include_scores=True, seed=seed + 1_010_000)
    exact_data = _topk_feature_recovery_from_pool(problem, x_pool=data_x, k=k, include_scores=True, seed=seed + 1_020_000)
    for api, result in [("scored_topk_random_queries", exact_random), ("scored_topk_dataset_queries", exact_data)]:
        rows.append(
            {
                "setting": f"{dataset}_public_feature_head",
                "dataset": dataset,
                "seed": seed,
                "api": api,
                "restriction": f"k={k},exact_scores",
                "query_count": int(result["query_count"]),
                "solved_classes": int(result["solved_classes"]),
                "delta_relerr": _clean_number(float(result["head_delta_relerr"])),
                "functional_agreement": float(result["test_agreement"]),
                "api_label_utility": 1.0,
                "observable_fraction": float(result["solved_classes"]) / problem.target.n_classes,
                "failure_rate": 1.0 if bool(result["failure"]) else 0.0,
                "note": "Score-bearing top-k leaks rows that appear in enough independent returned scores.",
            }
        )

    index_only = _topk_feature_recovery_from_pool(problem, x_pool=random_x, k=k, include_scores=False, seed=seed + 1_030_000)
    rows.append(
        {
            "setting": f"{dataset}_public_feature_head",
            "dataset": dataset,
            "seed": seed,
            "api": "topk_index_only",
            "restriction": f"k={k},no_scores",
            "query_count": int(index_only["query_count"]),
            "solved_classes": 0,
            "delta_relerr": None,
            "functional_agreement": None,
            "api_label_utility": 1.0,
            "observable_fraction": 0.0,
            "failure_rate": 1.0,
            "note": "Top-k labels preserve prediction service while removing score equations.",
        }
    )
    rows.append(
        {
            "setting": f"{dataset}_public_feature_head",
            "dataset": dataset,
            "seed": seed,
            "api": "hard_label",
            "restriction": "top1_index,no_scores",
            "query_count": int(problem.data.x_test.shape[0]),
            "solved_classes": 0,
            "delta_relerr": None,
            "functional_agreement": None,
            "api_label_utility": 1.0,
            "observable_fraction": 0.0,
            "failure_rate": 1.0,
            "note": "Hard labels preserve top-1 utility but erase score scale and exact parameters.",
        }
    )

    for decimals in round_decimals:
        result = _topk_feature_recovery_from_pool(
            problem,
            x_pool=random_x,
            k=k,
            include_scores=True,
            decimals=decimals,
            seed=seed + 1_040_000 + decimals,
        )
        rows.append(
            {
                "setting": f"{dataset}_public_feature_head",
                "dataset": dataset,
                "seed": seed,
                "api": "scored_topk_rounded",
                "restriction": f"k={k},decimals={decimals}",
                "query_count": int(result["query_count"]),
                "solved_classes": int(result["solved_classes"]),
                "delta_relerr": _clean_number(float(result["head_delta_relerr"])),
                "functional_agreement": float(result["test_agreement"]),
                "api_label_utility": 1.0,
                "observable_fraction": float(result["solved_classes"]) / problem.target.n_classes,
                "failure_rate": 1.0 if bool(result["failure"]) else 0.0,
                "note": "Rounding is a utility-preserving score degradation; parameters can degrade before labels.",
            }
        )

    for noise_std in noise_stds:
        result = _topk_feature_recovery_from_pool(
            problem,
            x_pool=random_x,
            k=k,
            include_scores=True,
            noise_std=noise_std,
            seed=seed + 1_050_000 + int(noise_std * 1_000_000),
        )
        rows.append(
            {
                "setting": f"{dataset}_public_feature_head",
                "dataset": dataset,
                "seed": seed,
                "api": "scored_topk_noisy",
                "restriction": f"k={k},sigma={noise_std:g}",
                "query_count": int(result["query_count"]),
                "solved_classes": int(result["solved_classes"]),
                "delta_relerr": _clean_number(float(result["head_delta_relerr"])),
                "functional_agreement": float(result["test_agreement"]),
                "api_label_utility": 1.0,
                "observable_fraction": float(result["solved_classes"]) / problem.target.n_classes,
                "failure_rate": 1.0 if bool(result["failure"]) else 0.0,
                "note": "Additive score noise gives the perturbation-theorem tradeoff.",
            }
        )


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _append_endpoint_rows(rows: list[Row], path: Path) -> None:
    for item in _load_json_rows(path):
        setting = f"{item.get('model', 'endpoint_model')}:{item.get('dataset', 'unknown')}:endpoint_head"
        base = {
            "setting": setting,
            "dataset": str(item.get("dataset", "unknown")),
            "seed": int(item.get("seed", -1)),
            "query_count": int(item.get("query_count", 0)),
            "solved_classes": int(item.get("classes", 0)),
            "api_label_utility": float(item.get("test_accuracy", 0.0)),
            "observable_fraction": 1.0,
            "failure_rate": 0.0,
        }
        rows.append(
            {
                **base,
                "api": "endpoint_logits",
                "restriction": "exact_scores",
                "delta_relerr": float(item["endpoint_delta_relerr"]),
                "functional_agreement": float(item["endpoint_label_agreement"]),
                "note": "Observable final-head adapter is recoverable from endpoint logits.",
            }
        )
        rows.append(
            {
                **base,
                "api": "endpoint_logits_rounded",
                "restriction": f"decimals={int(item.get('rounded_decimals', 0))}",
                "delta_relerr": float(item["rounded_endpoint_delta_relerr"]),
                "functional_agreement": float(item["rounded_endpoint_label_agreement"]),
                "note": "Endpoint rounding preserves labels in this run but increases parameter error.",
            }
        )
        rows.append(
            {
                **base,
                "api": "endpoint_logits_noisy",
                "restriction": f"sigma={float(item.get('noise_std', 0.0)):g}",
                "delta_relerr": float(item["noisy_endpoint_delta_relerr"]),
                "functional_agreement": float(item["noisy_endpoint_label_agreement"]),
                "note": "Endpoint score noise preserves most labels while destroying exact recovery.",
            }
        )


def _append_peft_endpoint_rows(rows: list[Row], path: Path) -> None:
    for item in _load_json_rows(path):
        rows.append(
            {
                "setting": f"{item.get('model', 'peft_model')}:{item.get('dataset', 'unknown')}:hidden_lora",
                "dataset": str(item.get("dataset", "unknown")),
                "seed": int(item.get("seed", -1)),
                "api": "endpoint_logits_hidden_lora",
                "restriction": "final_logits_only",
                "query_count": int(item.get("endpoint_real_examples", 0)),
                "solved_classes": int(item.get("endpoint_classifier_rank", 0)),
                "delta_relerr": None,
                "functional_agreement": 1.0 - float(item.get("endpoint_real_label_flip_rate", 0.0)),
                "api_label_utility": float(item.get("test_accuracy", 0.0)),
                "observable_fraction": float(item.get("endpoint_observable_fraction", 0.0)),
                "failure_rate": 1.0 if bool(item.get("endpoint_nonidentifiable", True)) else 0.0,
                "note": "Endpoint logits preserve prediction utility but hide hidden LoRA components in the classifier nullspace.",
            }
        )
        rows.append(
            {
                "setting": f"{item.get('model', 'peft_model')}:{item.get('dataset', 'unknown')}:module_lora",
                "dataset": str(item.get("dataset", "unknown")),
                "seed": int(item.get("seed", -1)),
                "api": "module_outputs_hidden_lora",
                "restriction": "module_outputs",
                "query_count": int(item.get("module_query_count", 0)),
                "solved_classes": int(item.get("module_out_features", 0)),
                "delta_relerr": float(item.get("module_delta_relerr", 0.0)),
                "functional_agreement": 1.0,
                "api_label_utility": float(item.get("test_accuracy", 0.0)),
                "observable_fraction": 1.0,
                "failure_rate": 0.0,
                "note": "Module-output access recovers the trained LoRA update exactly; this is not endpoint-only access.",
            }
        )


def run_sweep(
    *,
    seeds: list[int],
    datasets: list[str],
    feature_dim: int,
    rank: int,
    topk_queries: int,
    round_decimals: list[int],
    noise_stds: list[float],
    endpoint_json: Path | None = None,
    lora_json: Path | None = None,
) -> list[Row]:
    rows: list[Row] = []
    for dataset in datasets:
        for seed in seeds:
            _append_public_feature_rows(
                rows,
                seed=seed,
                dataset=dataset,
                feature_dim=feature_dim,
                rank=rank,
                topk_queries=topk_queries,
                round_decimals=round_decimals,
                noise_stds=noise_stds,
            )
    if endpoint_json is not None:
        _append_endpoint_rows(rows, endpoint_json)
    if lora_json is not None:
        _append_peft_endpoint_rows(rows, lora_json)
    return rows


def summarize_rows(rows: list[Row]) -> str:
    groups = sorted({(str(row["setting"]), str(row["api"]), str(row["restriction"])) for row in rows})
    lines = [
        "# Utility-vs-Leakage Defense Sweep",
        "",
        "This table reports parameter leakage and prediction utility under increasingly restricted APIs. Low parameter error means the adapter remains recoverable; high functional agreement or API label utility means the API remains useful for prediction.",
        "",
        "| setting | API | restriction | cases | delta relerr mean | functional agreement mean | API label utility mean | observable fraction mean | failure rate mean |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for setting, api, restriction in groups:
        group = [row for row in rows if row["setting"] == setting and row["api"] == api and row["restriction"] == restriction]
        relerr, _ = _mean_std(group, "delta_relerr")
        agreement, _ = _mean_std(group, "functional_agreement")
        utility, _ = _mean_std(group, "api_label_utility")
        observable, _ = _mean_std(group, "observable_fraction")
        failure, _ = _mean_std(group, "failure_rate")
        relerr_text = "n/a" if not np.isfinite(relerr) else f"{relerr:.2e}"
        agreement_text = "n/a" if not np.isfinite(agreement) else f"{agreement:.4f}"
        lines.append(
            f"| {setting} | {api} | {restriction} | {len(group)} | {relerr_text} | {agreement_text} | {utility:.4f} | {observable:.4f} | {failure:.2f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: exact raw or score-bearing APIs expose linear constraints and therefore leak observable adapter parameters. Rounding and noise can preserve labels while increasing parameter error. Scoreless hard-label and top-k-index APIs preserve prediction service but do not support exact linear-system recovery. Endpoint-only PEFT rows report the observable quotient: hidden LoRA effects in the classifier nullspace remain non-identifiable even though endpoint labels are unchanged.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = run_sweep(
        seeds=args.seeds,
        datasets=args.datasets,
        feature_dim=args.feature_dim,
        rank=args.rank,
        topk_queries=args.topk_queries,
        round_decimals=args.round_decimals,
        noise_stds=args.noise_stds,
        endpoint_json=args.endpoint_json,
        lora_json=args.lora_json,
    )

    json_path = args.outdir / "defense_sweep.json"
    csv_path = args.outdir / "defense_sweep.csv"
    md_path = args.outdir / "summary.md"
    json_path.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
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
