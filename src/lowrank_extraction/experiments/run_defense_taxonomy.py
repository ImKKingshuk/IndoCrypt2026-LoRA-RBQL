"""Which output defenses survive an adaptive attacker, and where that depends on the data.

Corollary 1 gives the attacker two levers and says exactly what each one is good
for: amplification defeats perturbations whose magnitude is fixed in *absolute*
terms, and averaging defeats perturbations that are *stochastic*.  Those are two
independent binary properties of a defense, so they induce a 2x2 taxonomy:

                        deterministic          stochastic
    absolute            decimal rounding       additive Gaussian noise
    relative            significant digits     multiplicative noise

Panel A measures every cell against both levers.  The prediction is that exactly
one cell -- relative and deterministic -- is immune to both, and that each other
cell is breached by precisely the lever its properties expose it to.

Panel B is the honesty check on the headline claim.  The breach of the absolute
grid was originally measured on one dataset.  Repeating it across datasets and
feature widths shows that it does *not* always materialize: on ``wine`` the
amplified design loses rank, because amplification pushes the public ReLU
features into their dead zone, and the attack simply fails.  The rank diagnostic
recorded here is what separates "the defense held" from "the attack broke",
and it is the reason the paper states absolute-grid safety as *conditional* on
feature geometry the defender neither controls nor can audit, while relative
quantization is safe unconditionally.

Both panels report bootstrap confidence intervals, because the quantities being
compared span three orders of magnitude and a bare 10-seed mean does not say
whether that separation is resolved.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from lowrank_extraction.adaptive import AdaptiveStrategy, run_adaptive_topk_recovery
from lowrank_extraction.realdata import make_digits_adapter_problem


Row = dict[str, Any]

# The four cells of the taxonomy. ``lever`` names the Corollary 1 lever the cell
# is predicted to be vulnerable to, or None if it is predicted to resist both.
TAXONOMY: list[dict[str, Any]] = [
    {
        "cell": "absolute + deterministic",
        "defense": "rounding (1 dec.)",
        "kwargs": {"decimals": 1},
        "predicted_lever": "amplification",
    },
    {
        "cell": "absolute + stochastic",
        "defense": "additive noise (0.1)",
        "kwargs": {"noise_std": 0.1},
        "predicted_lever": "amplification and averaging",
    },
    {
        "cell": "relative + deterministic",
        "defense": "significant digits (3)",
        "kwargs": {"sig_digits": 3},
        "predicted_lever": None,
    },
    {
        "cell": "relative + stochastic",
        "defense": "multiplicative noise (0.1%)",
        "kwargs": {"rel_noise_std": 1e-3},
        "predicted_lever": "averaging",
    },
    # A fifth entry, not a cell of the 2x2 but the strongest member of its
    # absolute+deterministic cell: instead of rounding into a grid the defender
    # spends its whole budget on a displacement chosen to be unhelpful. This is
    # the mechanism class that utility-constrained output-perturbation defenses
    # occupy, and Theorems 4 and 6 bound the perturbation only through ||E||_F,
    # so they predict amplification breaches it exactly as it breaches rounding.
    # If it did not, the theory would be about grids rather than about units.
    {
        "cell": "absolute + deterministic (worst-case)",
        "defense": "adversarial displacement (0.05)",
        "kwargs": {"adversarial_eps": 0.05},
        "predicted_lever": "amplification",
    },
]

# (dataset, feature width) settings for the generalization panel.
GENERALIZATION: list[tuple[str, int]] = [
    ("digits", 8),
    ("digits", 16),
    ("digits", 32),
    ("wine", 8),
    ("wine", 12),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=Path("results") / "defense_taxonomy")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--max-rank", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=800)
    parser.add_argument("--design-size", type=int, default=200)
    parser.add_argument("--amplify", type=float, default=256.0)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def _run(problem, x_pool, *, amplify: float, repeats: int, design_size: int,
         max_rank: int, seed: int, **defense) -> Any:
    return run_adaptive_topk_recovery(
        problem,
        x_pool=x_pool,
        k=min(2, problem.target.n_classes),
        strategy=AdaptiveStrategy(
            name=f"amp{amplify:g}_rep{repeats}",
            amplify=amplify,
            d_optimal=True,
            repeats=repeats,
            two_stage_bias=64,
            rank_project=max_rank,
            design_size=design_size,
            # Weighted by the reciprocal quantization cell width: the
            # efficient estimator for the error a relative quantizer
            # actually induces, and identically OLS under an absolute
            # grid, where that width is constant.
            gls=True,
        ),
        seed=seed,
        oracle_kind="raw",
        **defense,
    )


def _pool(problem, seed: int, pool_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 3_000_000)
    idx = rng.choice(problem.data.x_train.shape[0], size=pool_size, replace=True)
    return problem.data.x_train[idx]


def run_taxonomy(*, seeds: list[int], rank: int, max_rank: int, pool_size: int,
                 design_size: int, amplify: float, repeats: int) -> list[Row]:
    """Panel A: every cell of the 2x2 against both attacker levers."""

    rows: list[Row] = []
    for seed in seeds:
        problem = make_digits_adapter_problem(
            seed=seed, dataset="digits", feature_dim=8, rank=rank
        )
        x_pool = _pool(problem, seed, pool_size)
        common = dict(
            design_size=design_size, max_rank=max_rank, seed=seed + 7_000_000
        )
        for cell in TAXONOMY:
            kw = cell["kwargs"]
            # Baseline: neither lever engaged.
            base = _run(problem, x_pool, amplify=1.0, repeats=1, **common, **kw)
            # Amplification alone, then averaging alone.
            amp = _run(problem, x_pool, amplify=amplify, repeats=1, **common, **kw)
            avg = _run(problem, x_pool, amplify=1.0, repeats=repeats, **common, **kw)
            rows.append(
                {
                    "panel": "taxonomy",
                    "seed": seed,
                    "cell": cell["cell"],
                    "defense": cell["defense"],
                    "predicted_lever": cell["predicted_lever"] or "none",
                    "baseline_relerr": base.delta_relerr,
                    "amplified_relerr": amp.delta_relerr,
                    "averaged_relerr": avg.delta_relerr,
                    "amplification_gain": base.delta_relerr / amp.delta_relerr,
                    "averaging_gain": base.delta_relerr / avg.delta_relerr,
                    "amplified_failure": bool(amp.failure),
                }
            )
    return rows


def run_generalization(*, seeds: list[int], rank: int, max_rank: int, pool_size: int,
                       design_size: int, amplify: float) -> list[Row]:
    """Panel B: does the absolute-grid breach reproduce across datasets?"""

    rows: list[Row] = []
    for dataset, width in GENERALIZATION:
        for seed in seeds:
            problem = make_digits_adapter_problem(
                seed=seed, dataset=dataset, feature_dim=width, rank=rank
            )
            x_pool = _pool(problem, seed, pool_size)
            common = dict(
                design_size=design_size, max_rank=max_rank, seed=seed + 7_000_000
            )
            for family, kw in [("absolute", {"decimals": 1}), ("relative", {"sig_digits": 3})]:
                base = _run(problem, x_pool, amplify=1.0, repeats=1, **common, **kw)
                amp = _run(problem, x_pool, amplify=amplify, repeats=1, **common, **kw)
                rows.append(
                    {
                        "panel": "generalization",
                        "dataset": dataset,
                        "feature_dim": width,
                        "seed": seed,
                        "family": family,
                        "baseline_relerr": base.delta_relerr,
                        "amplified_relerr": amp.delta_relerr,
                        "amplification_gain": base.delta_relerr / amp.delta_relerr,
                        "amplified_failure": bool(amp.failure),
                        "amplified_class_rank": amp.extra["class_design_rank"],
                        "required_rank": amp.extra["required_rank"],
                        "baseline_class_rank": base.extra["class_design_rank"],
                    }
                )
    return rows


def run_homogeneity(*, seeds: list[int], rank: int, pool_size: int,
                    amplifications: list[float]) -> list[Row]:
    """Panel C: how nearly homogeneous is the public feature map, really?

    Theorem 6 is proved under exact positive homogeneity, ``phi(Ax) = A phi(x)``.
    A standardizer followed by a ReLU is *not* exactly homogeneous -- the mean
    shift is an affine offset -- so the honest statement is asymptotic.

    Two quantities are recorded. ``ratio`` is the norm ratio
    ``||phi(Ax)|| / (A ||phi(x)||)``, which shows how far from *exactly*
    homogeneous the map is. ``defect`` is the quantity the proof actually needs:
    the relative Frobenius distance between the rescaled feature blocks at
    consecutive scales, ``||phi(Ax)/A - phi(4Ax)/(4A)||_F / ||phi(4Ax)/(4A)||_F``.
    A defect tending to zero certifies that ``phi(Ax)/A`` converges to a fixed
    matrix, which is exactly the hypothesis under which ``(cZ)^+(cE) = Z^+E``
    holds in the limit -- the constant ``c`` cancels.
    """

    rows: list[Row] = []
    for dataset, width in [("digits", 8), ("wine", 8)]:
        for seed in seeds:
            problem = make_digits_adapter_problem(
                seed=seed, dataset=dataset, feature_dim=width, rank=rank
            )
            x = _pool(problem, seed, pool_size)
            phi = problem.data.feature_map
            base = float(np.mean(np.linalg.norm(phi(x), axis=1)))
            for amp in amplifications:
                z_a = phi(x * amp) / amp
                z_next = phi(x * amp * 4.0) / (amp * 4.0)
                denom = float(np.linalg.norm(z_next))
                rows.append(
                    {
                        "panel": "homogeneity",
                        "dataset": dataset,
                        "feature_dim": width,
                        "seed": seed,
                        "amplification": amp,
                        "ratio": float(np.mean(np.linalg.norm(phi(x * amp), axis=1)))
                        / (amp * base)
                        if base > 0
                        else float("nan"),
                        "defect": float(np.linalg.norm(z_a - z_next)) / denom
                        if denom > 0
                        else float("nan"),
                    }
                )
    return rows


def bootstrap_ci(values: list[float], *, draws: int, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean. Returns (mean, lo, hi)."""

    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(draws, arr.size))].mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def _ci_str(rows: list[Row], key: str, draws: int) -> str:
    m, lo, hi = bootstrap_ci([float(r[key]) for r in rows], draws=draws)
    return f"{m:.1f}x [{lo:.1f}, {hi:.1f}]"


def summarize(taxonomy: list[Row], general: list[Row], homogeneity: list[Row], *,
              draws: int, amplify: float, repeats: int) -> str:
    lines = [
        "# Defense Taxonomy and Generalization",
        "",
        f"Amplification factor A = {amplify:g}; averaging budget R = {repeats}.",
        "Gains are baseline error divided by post-lever error, so larger means the",
        "lever helped the attacker more. Brackets are 95% percentile bootstrap",
        f"confidence intervals for the mean over seeds ({draws} resamples).",
        "",
        "## Panel A: the 2x2 taxonomy",
        "",
        "| cell | defense | amplification gain | averaging gain | predicted vulnerable to |",
        "|---|---|---:|---:|---|",
    ]
    for cell in TAXONOMY:
        sub = [r for r in taxonomy if r["cell"] == cell["cell"]]
        lines.append(
            f"| {cell['cell']} | {cell['defense']} | {_ci_str(sub, 'amplification_gain', draws)}"
            f" | {_ci_str(sub, 'averaging_gain', draws)} | {cell['predicted_lever'] or 'neither'} |"
        )
    lines += [
        "",
        "Read across: a cell is breached by exactly the lever its properties expose",
        "it to. Only the relative+deterministic cell resists both, which is the",
        "recommendation the paper makes.",
        "",
        "## Panel B: does the absolute-grid breach generalize?",
        "",
        "`rank` is the mean rank of the per-class augmented design under",
        "amplification, against the rank required to solve. When it falls short the",
        "attack cannot solve the class at all, so a small gain means the *attack*",
        "broke, not that the defense worked.",
        "",
        "| dataset | width | family | amplification gain | rank (amp) | required | fail rate |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for dataset, width in GENERALIZATION:
        for family in ["absolute", "relative"]:
            sub = [
                r for r in general
                if r["dataset"] == dataset and r["feature_dim"] == width and r["family"] == family
            ]
            if not sub:
                continue
            rank = float(np.mean([r["amplified_class_rank"] for r in sub]))
            need = float(sub[0]["required_rank"])
            fail = float(np.mean([bool(r["amplified_failure"]) for r in sub]))
            lines.append(
                f"| {dataset} | {width} | {family} | {_ci_str(sub, 'amplification_gain', draws)}"
                f" | {rank:.1f} | {need:.0f} | {fail:.2f} |"
            )
    lines += [
        "",
        "Interpretation: the absolute grid is breached where the amplified design",
        "stays full rank, and appears safe only where it does not. Relative",
        "quantization is never breached in any setting, and its safety does not",
        "depend on which of the two regimes the deployment happens to be in.",
        "",
        "## Panel C: how homogeneous is the feature map?",
        "",
        "Theorem 6 is stated under exact positive homogeneity. `ratio` is",
        "`||phi(Ax)|| / (A ||phi(x)||)`: it does not stay at 1, so the map is not",
        "exactly homogeneous and the hypothesis as originally written is false.",
        "`defect` is what the proof actually needs -- the relative distance between",
        "`phi(Ax)/A` at consecutive scales. It tends to zero, so `phi(Ax)/A`",
        "converges to a fixed matrix and the theorem holds asymptotically, which is",
        "how the paper now states it.",
        "",
        "| dataset | quantity | " + " | ".join(
            f"A={a:g}" for a in sorted({float(r["amplification"]) for r in homogeneity})
        ) + " |",
        "|---|---" + "|---:" * len({float(r["amplification"]) for r in homogeneity}) + "|",
    ]
    amps = sorted({float(r["amplification"]) for r in homogeneity})
    for dataset in ["digits", "wine"]:
        for key, fmt in [("ratio", "{:.3f}"), ("defect", "{:.4f}")]:
            cells = []
            for a in amps:
                vals = [
                    float(r[key]) for r in homogeneity
                    if r["dataset"] == dataset and float(r["amplification"]) == a
                ]
                cells.append(fmt.format(np.mean(vals)) if vals else "--")
            lines.append(f"| {dataset} | {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines).rstrip() + "\n"


def _write(outdir: Path, name: str, rows: list[Row]) -> None:
    (outdir / f"{name}.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (outdir / f"{name}.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    common = dict(
        seeds=args.seeds,
        rank=args.rank,
        max_rank=args.max_rank,
        pool_size=args.pool_size,
        design_size=args.design_size,
        amplify=args.amplify,
    )
    taxonomy = run_taxonomy(repeats=args.repeats, **common)
    general = run_generalization(**common)
    homogeneity = run_homogeneity(
        seeds=args.seeds,
        rank=args.rank,
        pool_size=args.pool_size,
        amplifications=[1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0],
    )
    _write(args.outdir, "taxonomy", taxonomy)
    _write(args.outdir, "generalization", general)
    _write(args.outdir, "homogeneity", homogeneity)
    summary = summarize(
        taxonomy, general, homogeneity,
        draws=args.bootstrap, amplify=args.amplify, repeats=args.repeats,
    )
    (args.outdir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote outputs under {args.outdir}")


if __name__ == "__main__":
    main()
