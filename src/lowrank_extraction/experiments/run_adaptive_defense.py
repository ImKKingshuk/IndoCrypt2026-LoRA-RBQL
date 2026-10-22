"""Adaptive-attacker evaluation of score-degradation defenses.

The defense sweep measures what a *fixed* attacker recovers when the API
rounds or perturbs scores.  This runner asks the complementary question: how
much of that protection survives an attacker who adapts.  Each defense is
held fixed while the attacker is given the levers allowed by the
stable-recovery bound -- query amplification, D-optimal design selection,
repeated-query averaging, a two-stage intercept probe, and rank projection
onto the declared rank bound.

The evaluation is split into two panels because the two effects have
different causes:

Panel A (raw-logit oracle)
    Every class score is returned, so class coverage is never the binding
    constraint and the estimation levers are isolated.  This is where the
    ``1/A`` and ``1/sqrt(R)`` scaling laws implied by the stable-recovery
    bound can be measured directly.

Panel B (scored top-k oracle)
    Only ``k`` classes are returned per query, so amplification interacts
    with coverage: driving queries to extreme scale collapses the argmax
    into one cell and starves most class rows.  This panel reports the
    resulting failure rates and how much class-coverage-aware selection,
    which is free because the base model is public, repairs them.
"""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from lowrank_extraction.adaptive import AdaptiveStrategy, run_adaptive_topk_recovery
from lowrank_extraction.realdata import make_digits_adapter_problem


Row = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate adaptive attacker strategies against rounding and noise defenses."
    )
    parser.add_argument("--outdir", type=Path, default=Path("results") / "adaptive_defense")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--datasets", nargs="+", choices=["digits", "wine"], default=["digits"])
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--max-rank", type=int, default=4, help="declared rank upper bound used by the attacker")
    parser.add_argument("--pool-size", type=int, default=800)
    parser.add_argument("--design-size", type=int, default=200)
    parser.add_argument("--round-decimals", type=int, nargs="+", default=[1, 0])
    parser.add_argument("--noise-stds", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--amplify", type=float, default=64.0)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--bias-probe", type=int, default=64, help="unamplified probes used for the intercept stage")
    parser.add_argument(
        "--amplification-sweep", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    )
    parser.add_argument("--repeat-sweep", type=int, nargs="+", default=[1, 2, 4, 8, 16, 64, 256])
    parser.add_argument("--sweep-decimals", type=int, default=1)
    parser.add_argument("--sweep-noise", type=float, default=0.1)
    return parser.parse_args()


def build_strategies(
    *, amplify: float, repeats: int, max_rank: int, design_size: int, bias_probe: int
) -> list[AdaptiveStrategy]:
    """Cumulative ablation followed by single-lever controls."""

    return [
        AdaptiveStrategy(name="naive", design_size=design_size),
        AdaptiveStrategy(name="+d_optimal", d_optimal=True, design_size=design_size),
        AdaptiveStrategy(name="+repeat", d_optimal=True, repeats=repeats, design_size=design_size),
        AdaptiveStrategy(
            name="+amplify", d_optimal=True, repeats=repeats, amplify=amplify, design_size=design_size
        ),
        AdaptiveStrategy(
            name="+coverage",
            d_optimal=True,
            coverage_aware=True,
            repeats=repeats,
            amplify=amplify,
            design_size=design_size,
        ),
        AdaptiveStrategy(
            name="+bias_probe",
            d_optimal=True,
            coverage_aware=True,
            repeats=repeats,
            amplify=amplify,
            two_stage_bias=bias_probe,
            design_size=design_size,
        ),
        AdaptiveStrategy(
            name="full_adaptive",
            d_optimal=True,
            coverage_aware=True,
            repeats=repeats,
            amplify=amplify,
            two_stage_bias=bias_probe,
            rank_project=max_rank,
            design_size=design_size,
        ),
    ]


def _defenses(round_decimals: list[int], noise_stds: list[float]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = [("exact", {})]
    out += [(f"round_{d}", {"decimals": d}) for d in round_decimals]
    out += [(f"noise_{s:g}", {"noise_std": s}) for s in noise_stds]
    return out


def _problem_and_pool(seed: int, dataset: str, feature_dim: int, rank: int, pool_size: int):
    problem = make_digits_adapter_problem(seed=seed, dataset=dataset, feature_dim=feature_dim, rank=rank)
    rng = np.random.default_rng(seed + 3_000_000)
    idx = rng.choice(problem.data.x_train.shape[0], size=pool_size, replace=True)
    return problem, problem.data.x_train[idx]


def run_ablation(
    *,
    seeds: list[int],
    datasets: list[str],
    feature_dim: int,
    rank: int,
    max_rank: int,
    pool_size: int,
    design_size: int,
    round_decimals: list[int],
    noise_stds: list[float],
    amplify: float,
    repeats: int,
    bias_probe: int,
    oracle_kind: str,
) -> list[Row]:
    strategies = build_strategies(
        amplify=amplify, repeats=repeats, max_rank=max_rank, design_size=design_size, bias_probe=bias_probe
    )
    rows: list[Row] = []
    for dataset in datasets:
        for seed in seeds:
            problem, x_pool = _problem_and_pool(seed, dataset, feature_dim, rank, pool_size)
            k = min(2, problem.target.n_classes)
            for defense_name, defense_kwargs in _defenses(round_decimals, noise_stds):
                for strategy in strategies:
                    result = run_adaptive_topk_recovery(
                        problem,
                        x_pool=x_pool,
                        k=k,
                        strategy=strategy,
                        # zlib.crc32, not the builtin hash: Python randomises
                        # string hashes per process unless PYTHONHASHSEED is
                        # pinned, so the builtin gave the noise oracle a
                        # different seed on every run and the artifact could not
                        # reproduce its own numbers.
                        seed=seed + 4_000_000 + zlib.crc32(defense_name.encode()) % 10_000,
                        oracle_kind=oracle_kind,
                        **defense_kwargs,
                    )
                    rows.append(
                        {
                            "panel": oracle_kind,
                            "dataset": dataset,
                            "seed": seed,
                            "defense": defense_name,
                            "strategy": result.strategy,
                            "strategy_detail": result.description,
                            "oracle_calls": result.oracle_calls,
                            "distinct_design_points": result.distinct_design_points,
                            "solved_classes": result.solved_classes,
                            "delta_relerr": result.delta_relerr,
                            "bias_abserr": result.bias_abserr,
                            "test_agreement": result.test_agreement,
                            "design_pinv_norm": result.design_pinv_norm,
                            "design_sigma_min": result.design_sigma_min,
                            "mean_feature_norm": result.extra["mean_feature_norm"],
                            "failure": result.failure,
                        }
                    )
    return rows


def run_scaling_sweeps(
    *,
    seeds: list[int],
    dataset: str,
    feature_dim: int,
    rank: int,
    pool_size: int,
    design_size: int,
    amplifications: list[float],
    repeat_counts: list[int],
    round_decimals: int,
    noise_std: float,
) -> list[Row]:
    """Vary one lever at a time on the raw-logit oracle to expose scaling laws.

    Amplification is swept against a fixed absolute rounding grid, where the
    bound predicts a ``1/A`` decay of the linear-head error and no decay of
    the intercept error.  Repetition is swept against additive noise, where
    independent averaging predicts ``1/sqrt(R)``.
    """

    rows: list[Row] = []
    for seed in seeds:
        problem, x_pool = _problem_and_pool(seed, dataset, feature_dim, rank, pool_size)
        k = min(2, problem.target.n_classes)

        for amp in amplifications:
            result = run_adaptive_topk_recovery(
                problem,
                x_pool=x_pool,
                k=k,
                strategy=AdaptiveStrategy(name=f"amp_{amp:g}", amplify=amp, design_size=design_size),
                decimals=round_decimals,
                seed=seed + 5_000_000,
                oracle_kind="raw",
            )
            rows.append(
                {
                    "sweep": "amplification",
                    "seed": seed,
                    "defense": f"round_{round_decimals}",
                    "level": amp,
                    "delta_relerr": result.delta_relerr,
                    "bias_abserr": result.bias_abserr,
                    "test_agreement": result.test_agreement,
                    "mean_feature_norm": result.extra["mean_feature_norm"],
                    "design_pinv_norm": result.design_pinv_norm,
                    "failure": result.failure,
                }
            )

        for rep in repeat_counts:
            result = run_adaptive_topk_recovery(
                problem,
                x_pool=x_pool,
                k=k,
                strategy=AdaptiveStrategy(name=f"rep_{rep}", repeats=rep, design_size=design_size),
                noise_std=noise_std,
                seed=seed + 6_000_000,
                oracle_kind="raw",
            )
            rows.append(
                {
                    "sweep": "repeats",
                    "seed": seed,
                    "defense": f"noise_{noise_std:g}",
                    "level": rep,
                    "delta_relerr": result.delta_relerr,
                    "bias_abserr": result.bias_abserr,
                    "test_agreement": result.test_agreement,
                    "mean_feature_norm": result.extra["mean_feature_norm"],
                    "design_pinv_norm": result.design_pinv_norm,
                    "failure": result.failure,
                }
            )
    return rows


def _stat(rows: list[Row], key: str) -> tuple[float, float]:
    values = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
    if not values:
        return float("nan"), float("nan")
    return mean(values), (pstdev(values) if len(values) > 1 else 0.0)


def _bootstrap_ratio_ci(
    numerator: list[float], denominator: list[float], *, draws: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile CI for the per-seed gain of one lever over the row above it.

    The gain is a ratio of two errors measured on the *same* seed, so the
    resample has to be paired: resampling the two lists independently would
    destroy the pairing and inflate the interval.  This is what decides whether
    a measured gain that overshoots its predicted value -- amplification is
    predicted to buy ``A`` and sometimes measures more -- is a real effect or
    seed noise around the prediction.
    """

    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    paired = num / np.maximum(den, 1e-300)
    rng = np.random.default_rng(seed)
    n = paired.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    draws_arr = paired[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    return (
        float(paired.mean()),
        float(np.percentile(draws_arr, 2.5)),
        float(np.percentile(draws_arr, 97.5)),
    )


def _per_lever_table(rows: list[Row], amplify: float, repeats: int) -> list[str]:
    """Cumulative ablation reported as per-lever gains with paired CIs."""

    lines = [
        "",
        "## Per-lever gains, with paired bootstrap intervals",
        "",
        "Each row reports what *that* lever alone contributed, as the per-seed",
        "ratio of the previous row's error to its own. Brackets are 95% paired",
        "percentile bootstrap intervals over seeds. Corollary 1 predicts",
        f"amplification buys {amplify:g}x and repetition sqrt({repeats}) ="
        f" {repeats ** 0.5:.1f}x against stochastic perturbations and 1x against",
        "deterministic ones. An interval covering the prediction is the claim",
        "being confirmed; a point estimate above it with an interval that still",
        "covers it is seed noise, not an effect.",
        "",
        "| defense | lever | gain [95% CI] | predicted |",
        "|---|---|---|---|",
    ]
    predictions = {
        "+d_optimal": "conditioning",
        "+repeat": f"1x (deterministic) / {repeats ** 0.5:.1f}x (stochastic)",
        "+amplify": f"{amplify:g}x",
        "+bias_probe": "1x (head unchanged)",
    }
    defenses = sorted({str(r["defense"]) for r in rows}, key=lambda s: (s != "exact", s))
    order: list[str] = []
    for r in rows:
        if str(r["strategy"]) not in order:
            order.append(str(r["strategy"]))
    for defense in defenses:
        for previous, current in zip(order, order[1:]):
            prev_rows = sorted(
                [r for r in rows if r["defense"] == defense and r["strategy"] == previous],
                key=lambda r: r["seed"],
            )
            cur_rows = sorted(
                [r for r in rows if r["defense"] == defense and r["strategy"] == current],
                key=lambda r: r["seed"],
            )
            if not prev_rows or len(prev_rows) != len(cur_rows):
                continue
            mean, lo, hi = _bootstrap_ratio_ci(
                [float(r["delta_relerr"]) for r in prev_rows],
                [float(r["delta_relerr"]) for r in cur_rows],
            )
            lines.append(
                f"| {defense} | {current} | {mean:.2f}x [{lo:.2f}, {hi:.2f}] |"
                f" {predictions.get(current, '---')} |"
            )
    return lines


def _ablation_table(rows: list[Row], title: str, note: str) -> list[str]:
    defenses = sorted({str(r["defense"]) for r in rows}, key=lambda s: (s != "exact", s))
    strategies: list[str] = []
    for r in rows:
        if str(r["strategy"]) not in strategies:
            strategies.append(str(r["strategy"]))
    lines = [
        "",
        f"## {title}",
        "",
        note,
        "",
        "| defense | strategy | detail | oracle calls | delta relerr (mean +/- sd) | bias abserr | agreement | fail rate |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for defense in defenses:
        for strategy in strategies:
            group = [r for r in rows if r["defense"] == defense and r["strategy"] == strategy]
            if not group:
                continue
            relerr, relerr_sd = _stat(group, "delta_relerr")
            agreement, _ = _stat(group, "test_agreement")
            calls, _ = _stat(group, "oracle_calls")
            bias, _ = _stat(group, "bias_abserr")
            fail = sum(1 for r in group if r["failure"]) / len(group)
            lines.append(
                f"| {defense} | {strategy} | {group[0]['strategy_detail']} | {calls:.0f} | "
                f"{relerr:.3e} +/- {relerr_sd:.1e} | {bias:.3e} | {agreement:.4f} | {fail:.2f} |"
            )
    return lines


def _sweep_table(rows: list[Row], sweep: str, level_name: str) -> list[str]:
    group_all = [r for r in rows if r["sweep"] == sweep]
    if not group_all:
        return []
    levels = sorted({float(r["level"]) for r in group_all})
    defense = str(group_all[0]["defense"])
    lines = [
        "",
        f"### {sweep} sweep (raw-logit oracle, defense = {defense})",
        "",
        f"| {level_name} | mean feature norm | delta relerr | bias abserr | agreement |",
        "|---:|---:|---:|---:|---:|",
    ]
    for level in levels:
        group = [r for r in group_all if float(r["level"]) == level]
        relerr, _ = _stat(group, "delta_relerr")
        bias, _ = _stat(group, "bias_abserr")
        agreement, _ = _stat(group, "test_agreement")
        featnorm, _ = _stat(group, "mean_feature_norm")
        lines.append(f"| {level:g} | {featnorm:.3g} | {relerr:.4e} | {bias:.4e} | {agreement:.4f} |")

    first = [r for r in group_all if float(r["level"]) == levels[0]]
    last = [r for r in group_all if float(r["level"]) == levels[-1]]
    first_err, _ = _stat(first, "delta_relerr")
    last_err, _ = _stat(last, "delta_relerr")
    if last_err > 0:
        ratio = levels[-1] / levels[0]
        lines += [
            "",
            f"Level increased {ratio:.0f}x; adapter error fell {first_err / last_err:.1f}x "
            f"(1/level predicts {ratio:.0f}x, 1/sqrt(level) predicts {np.sqrt(ratio):.1f}x).",
        ]
    return lines


def summarize(raw_rows: list[Row], topk_rows: list[Row], sweep_rows: list[Row],
              *, amplify: float = 64.0, repeats: int = 64) -> str:
    lines = [
        "# Adaptive Attacker vs Score-Degradation Defenses",
        "",
        "Each row fixes a defense and varies the attacker. `delta relerr` is the",
        "relative Frobenius error of the recovered adapter (the linear head only);",
        "`bias abserr` is reported separately because amplification improves the",
        "linear part without improving the intercept.",
    ]
    lines += _ablation_table(
        raw_rows,
        "Panel A: raw-logit oracle",
        "Every class score is returned, so class coverage is never binding and the "
        "estimation levers are isolated.",
    )
    lines += _ablation_table(
        topk_rows,
        "Panel B: scored top-k oracle (k=2)",
        "Only k classes are returned per query. Amplification now competes with class "
        "coverage, and `fail rate` counts seeds where at least one class row was never "
        "solved.",
    )
    lines += _per_lever_table(raw_rows, amplify, repeats)
    lines += ["", "## Scaling laws"]
    lines += _sweep_table(sweep_rows, "amplification", "A")
    lines += _sweep_table(sweep_rows, "repeats", "R")
    lines += [
        "",
        "Interpretation: repeated averaging shrinks independent additive noise like",
        "1/sqrt(R) but leaves deterministic rounding untouched. Query amplification",
        "shrinks the effect of a fixed absolute grid on the linear head like 1/A, but",
        "does not improve the intercept, because the constant column of the augmented",
        "design does not scale with the queries. Under a top-k oracle, amplification is",
        "additionally limited by class coverage. Rounding and additive noise are",
        "therefore weak defenses against an adaptive attacker whose query domain is",
        "unconstrained, and their protection comes mainly from constraining the",
        "reachable feature geometry rather than from the score degradation itself.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _write_csv(path: Path, rows: list[Row]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    common = dict(
        seeds=args.seeds,
        datasets=args.datasets,
        feature_dim=args.feature_dim,
        rank=args.rank,
        max_rank=args.max_rank,
        pool_size=args.pool_size,
        design_size=args.design_size,
        round_decimals=args.round_decimals,
        noise_stds=args.noise_stds,
        amplify=args.amplify,
        repeats=args.repeats,
        bias_probe=args.bias_probe,
    )
    raw_rows = run_ablation(**common, oracle_kind="raw")
    topk_rows = run_ablation(**common, oracle_kind="topk")
    sweep_rows = run_scaling_sweeps(
        seeds=args.seeds,
        dataset=args.datasets[0],
        feature_dim=args.feature_dim,
        rank=args.rank,
        pool_size=args.pool_size,
        design_size=args.design_size,
        amplifications=args.amplification_sweep,
        repeat_counts=args.repeat_sweep,
        round_decimals=args.sweep_decimals,
        noise_std=args.sweep_noise,
    )

    all_rows = raw_rows + topk_rows
    (args.outdir / "adaptive_defense.json").write_text(
        json.dumps(all_rows, indent=2, allow_nan=False), encoding="utf-8"
    )
    (args.outdir / "scaling_sweeps.json").write_text(
        json.dumps(sweep_rows, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_csv(args.outdir / "adaptive_defense.csv", all_rows)
    _write_csv(args.outdir / "scaling_sweeps.csv", sweep_rows)

    summary = summarize(raw_rows, topk_rows, sweep_rows, amplify=args.amplify, repeats=args.repeats)
    (args.outdir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote outputs under {args.outdir}")


if __name__ == "__main__":
    main()
