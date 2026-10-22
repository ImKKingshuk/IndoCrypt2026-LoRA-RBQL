from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev

from lowrank_extraction.lower_bounds import make_partial_topk_indistinguishable_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run partial top-k index-only indistinguishability certificates.")
    parser.add_argument("--outdir", type=Path, default=Path("results") / "lower_bound")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--domain-radius", type=float, default=1.0)
    return parser.parse_args()


def summarize(rows: list[dict[str, float | int | bool]]) -> str:
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    lines = [
        "# Partial Top-k Index-Only Lower-Bound Certificate",
        "",
        f"Seeds: {len(rows)}",
        "",
        "| metric | mean | std | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("transcript_equal", "Identical transcript rate"),
        ("parameter_distance", "Adapter distance"),
        ("min_hidden_gap_on_queries", "Minimum query gap"),
        ("certified_domain_gap", "Certified bounded-domain gap"),
    ]:
        vals = values(key)
        lines.append(
            f"| {label} | {mean(vals):.4f} | {(pstdev(vals) if len(vals) > 1 else 0.0):.4f} | {min(vals):.4f} | {max(vals):.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: two different rank-one adapters induce identical partial top-k index-only transcripts when the changed class is certifiably outside the returned set. This supports a non-identifiability theorem rather than a heuristic attack failure.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        _, _, _, result = make_partial_topk_indistinguishable_pair(
            n_features=args.features,
            n_classes=args.classes,
            k=args.k,
            n_queries=args.queries,
            domain_radius=args.domain_radius,
            seed=seed,
        )
        row = asdict(result)
        row["seed"] = seed
        rows.append(row)

    json_path = args.outdir / "partial_topk_indistinguishability.json"
    csv_path = args.outdir / "partial_topk_indistinguishability.csv"
    md_path = args.outdir / "summary.md"
    json_path.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    md_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
