from __future__ import annotations

import argparse
import json
from pathlib import Path

from lowrank_extraction.attacks import (
    fit_hard_label_surrogate,
    recover_binary_hard_label_boundary,
    recover_raw_affine,
    recover_topk_scored_affine,
)
from lowrank_extraction.metrics import (
    binary_boundary_agreement,
    label_agreement,
    max_absolute_error,
    relative_frobenius_error,
    singular_value_energy,
)
from lowrank_extraction.models import make_linear_lowrank_target, random_probe_points
from lowrank_extraction.oracles import HardLabelOracle, RawLogitOracle, TopKOracle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic low-rank extraction experiments.")
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hard-label-queries", type=int, default=2000)
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    target = make_linear_lowrank_target(
        n_features=args.features,
        n_classes=args.classes,
        rank=args.rank,
        seed=args.seed,
    )

    raw_oracle = RawLogitOracle(target)
    raw = recover_raw_affine(raw_oracle, base_weight=target.base_weight, rank=target.rank)

    hard_oracle = HardLabelOracle(target)
    surrogate = fit_hard_label_surrogate(
        hard_oracle,
        n_features=target.n_features,
        n_classes=target.n_classes,
        n_queries=args.hard_label_queries,
        seed=args.seed + 1000,
    )

    topk_oracle = TopKOracle(target, k=min(3, target.n_classes), include_scores=True)
    topk_x = random_probe_points(128, target.n_features, seed=args.seed + 2000)
    topk_indices, topk_scores = topk_oracle.query(topk_x)

    topk_recovery_oracle = TopKOracle(target, k=min(3, target.n_classes), include_scores=True)
    topk_recovery = recover_topk_scored_affine(
        topk_recovery_oracle,
        base_weight=target.base_weight,
        rank=target.rank,
        random_queries=max(args.hard_label_queries, args.features * args.classes * 8),
        seed=args.seed + 4000,
    )

    binary_target = make_linear_lowrank_target(
        n_features=args.features,
        n_classes=2,
        rank=min(args.rank, 2),
        seed=args.seed + 5000,
    )
    binary_oracle = HardLabelOracle(binary_target)
    binary_boundary = recover_binary_hard_label_boundary(
        binary_oracle,
        n_features=binary_target.n_features,
        seed=args.seed + 6000,
    )

    eval_x = random_probe_points(4096, target.n_features, seed=args.seed + 3000)
    results = {
        "config": {
            "features": args.features,
            "classes": args.classes,
            "rank": args.rank,
            "seed": args.seed,
        },
        "raw_recovery": {
            "query_count": raw.query_count,
            "weight_relative_error": relative_frobenius_error(raw.recovered_weight, target.weight),
            "delta_relative_error": relative_frobenius_error(raw.recovered_delta, target.delta),
            "rank_projected_delta_relative_error": relative_frobenius_error(raw.rank_projected_delta, target.delta),
            "max_weight_error": max_absolute_error(raw.recovered_weight, target.weight),
            "top_rank_energy": singular_value_energy(raw.singular_values, target.rank),
            "eval_label_agreement": label_agreement(
                raw.recovered_weight,
                raw.recovered_bias,
                target.weight,
                target.bias,
                eval_x,
            ),
        },
        "hard_label_surrogate_baseline": {
            "query_count": surrogate.query_count,
            "train_agreement": surrogate.train_agreement,
            "eval_label_agreement": label_agreement(
                surrogate.weight,
                surrogate.bias,
                target.weight,
                target.bias,
                eval_x,
            ),
            "delta_relative_error_against_truth": relative_frobenius_error(
                surrogate.weight - target.base_weight,
                target.delta,
            ),
        },
        "topk_probe": {
            "query_count": topk_oracle.query_count,
            "k": topk_oracle.k,
            "first_indices": topk_indices[:5].tolist(),
            "first_scores": None if topk_scores is None else topk_scores[:5].tolist(),
        },
        "topk_scored_recovery": {
            "query_count": topk_recovery.query_count,
            "k": topk_recovery_oracle.k,
            "solved_classes": int(topk_recovery.solved_classes.sum()),
            "observations_per_class": topk_recovery.observations_per_class.tolist(),
            "weight_relative_error": relative_frobenius_error(topk_recovery.recovered_weight, target.weight),
            "delta_relative_error": relative_frobenius_error(topk_recovery.recovered_delta, target.delta),
            "rank_projected_delta_relative_error": relative_frobenius_error(
                topk_recovery.rank_projected_delta,
                target.delta,
            ),
            "eval_label_agreement": label_agreement(
                topk_recovery.recovered_weight,
                topk_recovery.recovered_bias,
                target.weight,
                target.bias,
                eval_x,
            ),
        },
        "binary_hard_label_boundary_recovery": {
            "query_count": binary_boundary.query_count,
            "solved": binary_boundary.solved,
            "boundary_points": int(binary_boundary.boundary_points.shape[0]),
            "validation_agreement": binary_boundary.train_agreement,
            "eval_label_agreement": binary_boundary_agreement(
                binary_boundary.normal,
                binary_boundary.offset,
                binary_target.weight,
                binary_target.bias,
                eval_x,
            )
            if binary_boundary.solved
            else 0.0,
            "note": "Hard labels recover the binary decision boundary up to scale, not logits or adapter magnitude.",
        },
    }

    outfile = args.outdir / f"synthetic_f{args.features}_c{args.classes}_r{args.rank}_s{args.seed}.json"
    outfile.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
