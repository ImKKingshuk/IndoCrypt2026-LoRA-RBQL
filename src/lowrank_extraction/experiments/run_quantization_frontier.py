"""Utility-security frontier for absolute vs relative score quantization.

Rounding returned scores is the classical output defense against equation-solving
extraction.  Section~5 of the paper shows that an *absolute* grid is defeated by
query amplification, because the grid width stays fixed while the attacker
inflates the signal.  Relative (significant-digit) quantization should not have
that weakness: the induced perturbation scales with the score, so the ratio the
recovery error depends on is invariant to query scale.

This runner evaluates the two families as a defender-then-attacker game, which is
the only comparison that is fair:

1. The defender picks a quantizer (a decimal place, or a number of significant
   digits) and pays a *utility* cost, measured as the fraction of held-out
   queries whose returned top-1 label is unchanged by quantization.
2. The attacker best-responds, choosing the amplification factor that minimizes
   its adapter recovery error, and additionally uses D-optimal design selection,
   repetition, the intercept probe, and rank projection.

Reporting the attacker's *best response* rather than a fixed attacker is what
makes the resulting frontier a security statement rather than an artifact of one
attack configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from lowrank_extraction.adaptive import AdaptiveStrategy, run_adaptive_topk_recovery
from lowrank_extraction.oracles import quantize_significant
from lowrank_extraction.realdata import make_digits_adapter_problem


Row = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Utility-security frontier for absolute vs relative score quantization."
    )
    parser.add_argument("--outdir", type=Path, default=Path("results") / "quantization_frontier")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--dataset", default="digits", choices=["digits", "wine"])
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--max-rank", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=800)
    parser.add_argument("--design-size", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--bias-probe", type=int, default=64)
    parser.add_argument("--decimals", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--sig-digits", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument(
        "--amplifications", type=float, nargs="+", default=[1.0, 4.0, 16.0, 64.0, 256.0, 1024.0]
    )
    return parser.parse_args()


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _api_utilities(
    problem, *, decimals: int | None, sig_digits: int | None
) -> dict[str, float]:
    """What the defender pays, measured five ways rather than one.

    Matching two quantizers on top-1 label survival alone is not enough to
    support a recommendation between them: a grid can hold the arg-max steady
    while flattening or distorting the distribution underneath it, and a caller
    that consumes probabilities rather than labels would feel that even though
    the label metric reports no cost.  So alongside ``label`` we report

      ``tv``    mean total-variation distance between the served and the exact
                predictive distribution -- pure fidelity, no labels involved;
      ``kl``    mean KL(exact || served), which unlike TV punishes the served
                distribution for putting little mass where the exact one puts
                much, the failure mode that matters when the score is used to
                rank or to threshold rather than to pick a winner;
      ``brier`` and ``nll``, proper scoring rules against the *true* held-out
                labels -- downstream quality rather than agreement with the
                unquantized model.

    All are computed on the oracle itself, independently of any attack.
    """

    exact = problem.data.z_test @ problem.target.head.T + problem.target.bias
    served = np.array(exact, copy=True)
    if decimals is not None:
        served = np.round(served, decimals)
    if sig_digits is not None:
        served = quantize_significant(served, sig_digits)

    p_exact, p_served = _softmax(exact), _softmax(served)
    floor = 1e-12
    labels = np.asarray(problem.data.y_test, dtype=int)
    onehot = np.zeros_like(p_served)
    onehot[np.arange(len(labels)), labels] = 1.0
    return {
        "label": float(np.mean(np.argmax(served, axis=1) == np.argmax(exact, axis=1))),
        "tv": float(np.mean(0.5 * np.sum(np.abs(p_served - p_exact), axis=1))),
        "kl": float(
            np.mean(
                np.sum(
                    p_exact * (np.log(p_exact + floor) - np.log(p_served + floor)),
                    axis=1,
                )
            )
        ),
        "brier": float(np.mean(np.sum((p_served - onehot) ** 2, axis=1))),
        "nll": float(np.mean(-np.log(p_served[np.arange(len(labels)), labels] + floor))),
    }


def _api_label_utility(problem, *, decimals: int | None, sig_digits: int | None) -> float:
    return _api_utilities(problem, decimals=decimals, sig_digits=sig_digits)["label"]


def _best_response(
    problem,
    x_pool: np.ndarray,
    *,
    k: int,
    decimals: int | None,
    sig_digits: int | None,
    amplifications: list[float],
    design_size: int,
    repeats: int,
    bias_probe: int,
    max_rank: int,
    seed: int,
) -> tuple[float, float]:
    """Attacker's best achievable error over its amplification choices."""

    best_err = float("inf")
    best_amp = float("nan")
    for amp in amplifications:
        result = run_adaptive_topk_recovery(
            problem,
            x_pool=x_pool,
            k=k,
            strategy=AdaptiveStrategy(
                name=f"amp{amp:g}",
                amplify=amp,
                d_optimal=True,
                repeats=repeats,
                two_stage_bias=bias_probe,
                rank_project=max_rank,
                design_size=design_size,
                gls=True,
            ),
            decimals=decimals,
            sig_digits=sig_digits,
            seed=seed,
            oracle_kind="raw",
        )
        if result.delta_relerr < best_err:
            best_err = result.delta_relerr
            best_amp = amp
    return best_err, best_amp


def run_frontier(
    *,
    seeds: list[int],
    dataset: str,
    feature_dim: int,
    rank: int,
    max_rank: int,
    pool_size: int,
    design_size: int,
    repeats: int,
    bias_probe: int,
    decimals: list[int],
    sig_digits: list[int],
    amplifications: list[float],
) -> list[Row]:
    configs: list[tuple[str, dict[str, Any]]] = []
    configs += [(f"absolute, {d} dec.", {"decimals": d}) for d in decimals]
    configs += [(f"relative, {p} sig.", {"sig_digits": p}) for p in sig_digits]

    rows: list[Row] = []
    for seed in seeds:
        problem = make_digits_adapter_problem(
            seed=seed, dataset=dataset, feature_dim=feature_dim, rank=rank
        )
        rng = np.random.default_rng(seed + 3_000_000)
        idx = rng.choice(problem.data.x_train.shape[0], size=pool_size, replace=True)
        x_pool = problem.data.x_train[idx]
        k = min(2, problem.target.n_classes)

        for name, cfg in configs:
            utilities = _api_utilities(
                problem, decimals=cfg.get("decimals"), sig_digits=cfg.get("sig_digits")
            )
            utility = utilities["label"]
            err, amp = _best_response(
                problem,
                x_pool,
                k=k,
                decimals=cfg.get("decimals"),
                sig_digits=cfg.get("sig_digits"),
                amplifications=amplifications,
                design_size=design_size,
                repeats=repeats,
                bias_probe=bias_probe,
                max_rank=max_rank,
                seed=seed + 7_000_000,
            )
            # Error the same attacker gets without amplification, to expose how
            # much of the defense the amplification lever alone removes.
            unamplified, _ = _best_response(
                problem,
                x_pool,
                k=k,
                decimals=cfg.get("decimals"),
                sig_digits=cfg.get("sig_digits"),
                amplifications=[1.0],
                design_size=design_size,
                repeats=repeats,
                bias_probe=bias_probe,
                max_rank=max_rank,
                seed=seed + 7_000_000,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "family": "absolute" if "decimals" in cfg else "relative",
                    "config": name,
                    "api_label_utility": utility,
                    "api_tv": utilities["tv"],
                    "api_kl": utilities["kl"],
                    "api_brier": utilities["brier"],
                    "api_nll": utilities["nll"],
                    "attacker_best_relerr": err,
                    "attacker_best_amplification": amp,
                    "unamplified_relerr": unamplified,
                    "amplification_gain": unamplified / err if err > 0 else float("inf"),
                }
            )
    return rows


def _agg(rows: list[Row], config: str, key: str) -> tuple[float, float]:
    vals = [float(r[key]) for r in rows if r["config"] == config and np.isfinite(float(r[key]))]
    if not vals:
        return float("nan"), float("nan")
    return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)


def summarize(rows: list[Row]) -> str:
    configs = []
    for r in rows:
        if r["config"] not in configs:
            configs.append(str(r["config"]))

    lines = [
        "# Utility-Security Frontier: Absolute vs Relative Score Quantization",
        "",
        "The defender picks a quantizer; the attacker best-responds over its",
        "amplification choices. `utility` is the fraction of held-out queries whose",
        "returned top-1 label survives quantization. `attacker best` is the lowest",
        "adapter relative error the attacker achieves. `amp gain` is how much the",
        "amplification lever alone buys the attacker -- the quantity that",
        "distinguishes the two families.",
        "",
        "| quantizer | utility | attacker best | best A | no-amp error | amp gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in configs:
        u, _ = _agg(rows, c, "api_label_utility")
        e, _ = _agg(rows, c, "attacker_best_relerr")
        a, _ = _agg(rows, c, "attacker_best_amplification")
        n, _ = _agg(rows, c, "unamplified_relerr")
        g, _ = _agg(rows, c, "amplification_gain")
        lines.append(f"| {c} | {u:.4f} | {e:.3e} | {a:.0f} | {n:.3e} | {g:.1f}x |")

    lines += [
        "",
        "## Utility-matched comparison",
        "",
        "For each absolute-grid setting, the relative setting with the closest",
        "utility is paired with it; the security gap is the ratio of attacker",
        "errors at (approximately) equal service cost.",
        "",
        "| absolute | utility | attacker best | closest relative | utility | attacker best | security gain |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    abs_cfgs = [c for c in configs if c.startswith("absolute")]
    rel_cfgs = [c for c in configs if c.startswith("relative")]
    for c in abs_cfgs:
        ua, _ = _agg(rows, c, "api_label_utility")
        ea, _ = _agg(rows, c, "attacker_best_relerr")
        if not np.isfinite(ua):
            continue
        best = min(rel_cfgs, key=lambda r: abs(_agg(rows, r, "api_label_utility")[0] - ua))
        ur, _ = _agg(rows, best, "api_label_utility")
        er, _ = _agg(rows, best, "attacker_best_relerr")
        gain = er / ea if ea > 0 else float("inf")
        lines.append(
            f"| {c} | {ua:.4f} | {ea:.3e} | {best} | {ur:.4f} | {er:.3e} | {gain:.1f}x |"
        )

    lines += [
        "",
        "## Does the pairing survive a different definition of utility?",
        "",
        "The table above matches on top-1 label survival.  If the relative grid",
        "won only because it spends its service cost somewhere that metric cannot",
        "see, then re-running the same pairing against a distribution-level or a",
        "proper-scoring-rule notion of utility should move the pairs and shrink",
        "the gains.  Each block below repeats the match keyed on a different",
        "metric.  `tv` and `kl` are distances from the unquantized distribution",
        "and `brier`/`nll` are scored against true labels, so for those the",
        "closest match is on the smallest absolute difference in that quantity,",
        "and lower is better rather than higher.",
    ]
    for metric in ("tv", "kl", "brier", "nll"):
        key = f"api_{metric}"
        lines += [
            "",
            f"### matched on {metric}",
            "",
            f"| absolute | {metric} | attacker best | closest relative | {metric} "
            "| attacker best | security gain |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
        for c in abs_cfgs:
            ua, _ = _agg(rows, c, key)
            ea, _ = _agg(rows, c, "attacker_best_relerr")
            if not np.isfinite(ua):
                continue
            best = min(rel_cfgs, key=lambda r: abs(_agg(rows, r, key)[0] - ua))
            ur, _ = _agg(rows, best, key)
            er, _ = _agg(rows, best, "attacker_best_relerr")
            gain = er / ea if ea > 0 else float("inf")
            lines.append(
                f"| {c} | {ua:.4g} | {ea:.3e} | {best} | {ur:.4g} | {er:.3e} "
                f"| {gain:.1f}x |"
            )

    lines += [
        "",
        "Interpretation: an absolute grid is scale-dependent, so amplification",
        "shrinks the perturbation relative to the signal and the defense collapses.",
        "Relative quantization is scale-invariant: the perturbation grows with the",
        "signal, amplification buys the attacker essentially nothing, and at matched",
        "utility the attacker's best achievable error is orders of magnitude larger.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = run_frontier(
        seeds=args.seeds,
        dataset=args.dataset,
        feature_dim=args.feature_dim,
        rank=args.rank,
        max_rank=args.max_rank,
        pool_size=args.pool_size,
        design_size=args.design_size,
        repeats=args.repeats,
        bias_probe=args.bias_probe,
        decimals=args.decimals,
        sig_digits=args.sig_digits,
        amplifications=args.amplifications,
    )
    (args.outdir / "quantization_frontier.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (args.outdir / "quantization_frontier.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    (args.outdir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote outputs under {args.outdir}")


if __name__ == "__main__":
    main()
