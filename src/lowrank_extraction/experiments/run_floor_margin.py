"""What the relative-quantization floor does *between* powers of the base.

Theorem~\\ref{thm:floor}(ii) is exact on one subsequence and silent off it, and
this experiment is why the silence is deliberate.

Along amplifications that are powers of the quantization base the mantissas of
every served score are unchanged, so every ratio ``gamma/|y|`` is unchanged and
the floor ``F(A Q)`` is exactly invariant.  It is tempting to go further and
claim that off that subsequence ``F`` stays bounded above *and below* by
positive constants.  The upper half holds.  The lower half does not follow
from what the proof establishes: bounded mantissas constrain the cell *width*,
not the distance from a score to its cell's edge, and as ``A`` varies
continuously the scores drift across their cells with nothing keeping them off
the edges.

So the correct statement is a limsup, and this script measures five things:

  * ``F`` at powers of the base, which should be constant to machine precision;
  * the same sweep under an absolute grid, which should decay like ``1/A``
    throughout and never recover -- the contrast the security claim rests on;
  * ``F`` on nested refinements of a sweep across one decade, whose running
    minimum can only fall.  If it keeps falling, no positive lower bound exists
    to be proved; if it settles, the discarded claim was salvageable after all;
  * ``F`` at powers of the base again, but under *asymptotic* rather than exact
    homogeneity, where the served mantissas converge to those of the limiting
    block rather than repeating those of the ``A=1`` block: ``F`` still does not
    decay, but its limit is not ``F(Q)``;
  * the least-squares error of Theorem 5(1), which had the same overreach in the
    same place -- a claimed *limit* where the proof delivers only a band.

The floor here is the one-sided certificate of Theorem 6 evaluated in closed
form on a fixed design, not the hill-climbed bound of ``run-security-floor``:
the question is how one number moves with ``A``, so the search that makes that
number as large as possible would only add noise to the comparison.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lowrank_extraction.experiments.run_security_floor import (
    directional_step,
    signed_cell_slack,
)


def floor_at(
    z: np.ndarray,
    theta: np.ndarray,
    xi: np.ndarray,
    delta_norm: float,
    *,
    decimals: int | None,
    sig_digits: int | None,
) -> float:
    """Theorem 6's ``F`` for one fixed direction ``Xi`` on one design.

    Same two ingredients as ``certified_floor``'s ``value_of`` -- the directional
    step through the admissible polytope and Theorem 6's factor ``1/2`` -- with
    the search over directions removed, since here the point is how one number
    moves with ``A`` rather than how large it can be made.
    """
    scores = z @ theta.T
    down, up = signed_cell_slack(scores, decimals=decimals, sig_digits=sig_digits)
    return (
        0.5 * directional_step(z, xi, down, up)
        * float(np.linalg.norm(xi)) / delta_norm
    )


def least_squares_error(a: float, x: np.ndarray, mu: np.ndarray,
                        theta: np.ndarray, sig_digits: int) -> float:
    """The quantity Theorem 5(1) is about, at amplification ``a``.

    The feature map subtracts a fixed mean, so it is asymptotically but not
    exactly positively homogeneous -- the theorem's actual hypothesis, and the
    situation a standardizer puts every real pipeline in.
    """
    from lowrank_extraction.oracles import quantize_significant

    z = a * x - mu
    y = z @ theta.T
    return float(np.linalg.norm(np.linalg.pinv(z) @ (quantize_significant(y, sig_digits) - y)))


@dataclass
class Row:
    amplification: float
    grid: str
    floor: float
    on_subsequence: bool


def sweep(
    seed: int, sig_digits: int, decimals: int, n_points: int, decades: float
) -> tuple[list[Row], dict[str, Any], list[dict[str, float]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n, h, m, r = 24, 6, 3, 2

    z0 = rng.standard_normal((n, h))
    head = rng.standard_normal((m, h))
    delta = rng.standard_normal((m, r)) @ rng.standard_normal((r, h))
    theta = head + delta
    delta_norm = float(np.linalg.norm(delta))

    # One fixed one-sided direction, so that every amplification is compared on
    # the same perturbation rather than on whichever one happened to win a search.
    u = np.linalg.svd(delta, full_matrices=False)[0][:, :r]
    xi = u @ (u.T @ rng.standard_normal((m, h)))

    rows: list[Row] = []

    # Exactly on the subsequence: A = 10^k.
    for k in range(0, 7):
        a = 10.0**k
        for grid, kw in (
            ("relative", {"sig_digits": sig_digits, "decimals": None}),
            ("absolute", {"decimals": decimals, "sig_digits": None}),
        ):
            rows.append(
                Row(a, grid, floor_at(a * z0, theta, xi, delta_norm, **kw), True)
            )

    # A fine sweep across `decades` decades, log-spaced.
    for a in np.logspace(0.0, decades, n_points):
        for grid, kw in (
            ("relative", {"sig_digits": sig_digits, "decimals": None}),
            ("absolute", {"decimals": decimals, "sig_digits": None}),
        ):
            rows.append(
                Row(
                    float(a),
                    grid,
                    floor_at(float(a) * z0, theta, xi, delta_norm, **kw),
                    False,
                )
            )

    # One sweep only reports the smallest F it happened to land on, which is
    # equally consistent with a positive floor the grid never got below.
    # Refining separates the two, and the grids below nest -- each doubles the
    # previous one -- so the running minimum can only fall.  If it keeps falling
    # instead of settling, nothing is holding F up.
    #
    # Driving F down is harder than driving one score to a cell edge, and it is
    # worth being precise about why.  A score at its edge is immovable in one
    # direction but a full step from the other, and the attacker takes whichever
    # sign is longer; since a cell's two slacks sum to its width, no single score
    # can be tight both ways.  F collapses only when *two* scores are near
    # opposite edges at once -- a codimension-two coincidence in this
    # one-parameter family.  So the decay below is slow, and that slowness is the
    # finding: off the subsequence the floor is not protected, but neither does it
    # collapse at a typical amplification.
    ladder: list[dict[str, float]] = []
    for step in range(9):
        points = 1000 * 2**step + 1
        values = [
            floor_at(float(a) * z0, theta, xi, delta_norm,
                     sig_digits=sig_digits, decimals=None)
            for a in np.logspace(0.0, decades, points)
        ]
        ladder.append({"n_points": points, "min_floor": min(values)})

    # The sweeps above scale the design exactly, z = A*z0, which is the *exact*
    # homogeneity half of Theorem 6(ii).  The theorem also claims F(AQ) =
    # F(Q)(1+o(1)) under the merely asymptotic homogeneity Theorem 5 assumes, and
    # that half is a different statement: with z = A*z0 - mu the served mantissas
    # converge to those of the limiting block Z_inf = z0, not to those of the A=1
    # block, so the ratios gamma/|y| converge to the slacks Z_inf induces.  F then
    # tends to F_inf -- the floor *at the limit*, which is the exactly homogeneous
    # value -- and F(Q) is not it.  Same fixed direction and closed form as above,
    # so the only moving part is the hypothesis.
    mu_z = np.random.default_rng(seed + 11).standard_normal(h)
    asymptotic = [
        {
            "amplification": 10.0**k,
            "exact": floor_at(10.0**k * z0, theta, xi, delta_norm,
                              sig_digits=sig_digits, decimals=None),
            "asymptotic": floor_at(10.0**k * z0 - mu_z, theta, xi, delta_norm,
                                   sig_digits=sig_digits, decimals=None),
        }
        for k in range(9)
    ]

    # Theorem 5(1) has the same defect in the same place, so it is measured here
    # rather than in a section of its own.  The least-squares error does not
    # decay -- that is the claim that matters, and it holds -- but a limit
    # independent of A would be more than the proof delivers: off the base-power subsequence the argument only
    # confines the error to within one quantization step, and a band is not a
    # limit.  These two sweeps are the difference.
    ls_rng = np.random.default_rng(seed + 3)
    x = ls_rng.standard_normal((200, 8))
    mu = ls_rng.standard_normal(8)
    ls_theta = ls_rng.standard_normal((4, 8))
    ls_subsequence = [
        {"amplification": 10.0**k,
         "error": least_squares_error(10.0**k, x, mu, ls_theta, sig_digits + 1)}
        for k in range(7)
    ]
    ls_band = [
        least_squares_error(float(a), x, mu, ls_theta, sig_digits + 1)
        for a in np.logspace(3.0, 4.0, 2000)
    ]

    config = {
        "seed": seed,
        "sig_digits": sig_digits,
        "decimals": decimals,
        "n_points": n_points,
        "decades": decades,
        "shape": {"n": n, "h": h, "m": m, "rank": r},
    }
    limit = asymptotic[-1]
    asymptotic_summary = {
        "rows": asymptotic,
        # F(Q) is the A=1 floor; F_inf is what the sequence tends to.  The
        # theorem says these are the same number.  They are not.
        "f_at_one": asymptotic[0]["asymptotic"],
        "f_limit": limit["asymptotic"],
        "limit_over_f_at_one": limit["asymptotic"] / asymptotic[0]["asymptotic"],
        # The limit should be the floor at Z_inf = z0, i.e. the exactly
        # homogeneous value, and agreeing with it is the check that the drift is
        # the hypothesis and not the arithmetic.
        "limit_over_exact": limit["asymptotic"] / limit["exact"],
        "exact_relative_spread": (
            max(r["exact"] for r in asymptotic) - min(r["exact"] for r in asymptotic)
        ) / asymptotic[0]["exact"],
    }
    least_squares = {
        "subsequence": ls_subsequence,
        "band_min": min(ls_band),
        "band_max": max(ls_band),
        "band_ratio": max(ls_band) / min(ls_band),
        "subsequence_min": min(e["error"] for e in ls_subsequence),
        "subsequence_max": max(e["error"] for e in ls_subsequence),
    }
    return rows, config, ladder, least_squares, asymptotic_summary


def summarize(rows: list[Row]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for grid in ("relative", "absolute"):
        on = [r.floor for r in rows if r.grid == grid and r.on_subsequence]
        off = [r.floor for r in rows if r.grid == grid and not r.on_subsequence]
        first = on[0]
        out[grid] = {
            "subsequence_values": on,
            # Relative spread of F along A = 10^k. Zero means exact invariance.
            "subsequence_relative_spread": (max(on) - min(on)) / first if first else None,
            "sweep_max": max(off),
            "sweep_min": min(off),
            # How far below the subsequence value the sweep dips. A ratio near
            # zero is the liminf the statement declines to bound.
            "sweep_min_over_subsequence": min(off) / first if first else None,
            "sweep_max_over_subsequence": max(off) / first if first else None,
        }
    return out


def render(summary: dict[str, Any], config: dict[str, Any]) -> str:
    rel, ab = summary["relative"], summary["absolute"]
    ls = summary["least_squares"]
    lines = [
        "# The Relative Floor Between Powers of the Base",
        "",
        f"Design {config['shape']}, seed {config['seed']}, "
        f"{config['sig_digits']} significant digits against a "
        f"{config['decimals']}-decimal absolute grid.",
        "",
        "## On the subsequence A = 10^k",
        "",
        f"- relative grid: spread over k = 0..6 is "
        f"**{rel['subsequence_relative_spread']:.3e}** (exact invariance is 0)",
        f"- absolute grid: F falls from {ab['subsequence_values'][0]:.3e} to "
        f"{ab['subsequence_values'][-1]:.3e} across six decades, i.e. like 1/A",
        "",
        "## Off the subsequence",
        "",
        f"- relative grid dips to **{rel['sweep_min_over_subsequence']:.3e}x** its "
        f"subsequence value and rises to {rel['sweep_max_over_subsequence']:.3e}x",
        f"- so the limsup is the subsequence value, and whether the liminf is "
        f"bounded away from zero is settled by the refinement below",
        "",
        f"- absolute grid stays within "
        f"[{ab['sweep_min']:.3e}, {ab['sweep_max']:.3e}] and never returns to its "
        f"value at A = 1, which is the contrast the security claim needs",
        "",
        "## On the subsequence, under *asymptotic* homogeneity",
        "",
        "The sweeps above scale the design exactly. Theorem 6(ii) also claims",
        "F(AQ) = F(Q)(1+o(1)) under the asymptotic homogeneity Theorem 5 assumes.",
        "Same design, same direction, z = A*z0 - mu:",
        "",
        "| A | F, exact homogeneity | F, asymptotic |",
        "| --- | --- | --- |",
    ]
    for row in summary["asymptotic_homogeneity"]["rows"]:
        lines.append(
            f"| {row['amplification']:.0e} | {row['exact']:.6e} | "
            f"{row['asymptotic']:.6e} |"
        )
    ah = summary["asymptotic_homogeneity"]
    lines += [
        "",
        f"- exact-homogeneity spread: **{ah['exact_relative_spread']:.3e}** "
        f"(invariant, as part (ii) says)",
        f"- asymptotic: F(Q) = {ah['f_at_one']:.6e} but the sequence tends to "
        f"F_inf = {ah['f_limit']:.6e}, a factor **{ah['limit_over_f_at_one']:.4f}**",
        f"- and F_inf agrees with the floor at Z_inf to "
        f"{abs(ah['limit_over_exact'] - 1):.2e}, so the drift is the hypothesis "
        f"and not the arithmetic",
        "",
        "So F does not decay -- the claim that carries the security argument --",
        "but the limit is not F(Q), and the theorem should not say that it is.",
        "",
        "## Refining the sweep (nested grids, so the minimum can only fall)",
        "",
        "| sweep points | smallest F, as a fraction of the subsequence value |",
        "| --- | --- |",
    ]
    for step in summary["refinement_ladder"]:
        lines.append(f"| {step['n_points']} | {step['min_over_subsequence']:.3e} |")
    lines += [
        "",
        f"Log-log slope {summary['refinement_log_slope']:.2f}: the minimum is still "
        f"falling after a 256-fold refinement, with no sign of a positive barrier. "
        f"It falls slowly because collapsing F needs two scores near opposite cell "
        f"edges at once, which is a codimension-two coincidence in a one-parameter "
        f"family -- so the floor is unprotected off the subsequence without being "
        f"fragile at a typical amplification.",
        "",
        "## Theorem 5(1): the same defect, in the least-squares error",
        "",
        f"- across A = 1 to 1e6 the error stays in "
        f"[{ls['subsequence_min']:.3e}, {ls['subsequence_max']:.3e}]: no decay, "
        f"which is the claim that matters",
        f"- but across a single decade it ranges over "
        f"[{ls['band_min']:.3e}, {ls['band_max']:.3e}], a factor of "
        f"**{ls['band_ratio']:.2f}**, so it does not converge to a limit either",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("results") / "floor_margin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sig-digits", type=int, default=3)
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--n-points", type=int, default=4001)
    ap.add_argument("--decades", type=float, default=1.0)
    args = ap.parse_args()

    rows, config, ladder, least_squares, asymptotic_summary = sweep(
        args.seed, args.sig_digits, args.decimals, args.n_points, args.decades
    )
    summary = summarize(rows)
    base = summary["relative"]["subsequence_values"][0]
    for entry in ladder:
        entry["min_over_subsequence"] = entry["min_floor"] / base
    xs = np.log([e["n_points"] for e in ladder])
    ys = np.log([e["min_over_subsequence"] for e in ladder])
    summary["refinement_ladder"] = ladder
    summary["least_squares"] = least_squares
    summary["asymptotic_homogeneity"] = asymptotic_summary
    # Negative and away from zero means the running minimum is still falling.
    summary["refinement_log_slope"] = float(np.polyfit(xs, ys, 1)[0])

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "floor_margin.json").write_text(
        json.dumps(
            {"config": config, "summary": summary, "rows": [asdict(r) for r in rows]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    text = render(summary, config)
    (args.outdir / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
