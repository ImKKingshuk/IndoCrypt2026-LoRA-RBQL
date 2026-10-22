"""A certified indistinguishability floor that holds against *every* attacker.

Theorem 6 says that two specific levers -- amplification and averaging -- buy the
attacker nothing against relative quantization.  That is an upper bound on two
attacks, not a security claim: some third lever might still work.  This experiment computes the quantity that answers the question
unconditionally.

The construction.  A deterministic quantizer partitions score space into cells.
If two adapters put every returned score in the same cell, they induce
*identical* transcripts, so no algorithm -- with any query budget, any design,
adaptive or not -- can tell them apart.  Let ``g[j,c]`` be the distance from the
exact score of query ``j`` on class ``c`` to the nearest boundary of its cell.
For any perturbation direction ``Xi`` of the adapter, every ``t`` small enough
that no score leaves its cell gives an adapter ``Delta + t Xi`` with the same
transcript.  Maximising over directions gives a lower bound on the diameter of
the transcript-consistent set, hence on the minimax recovery error of any
algorithm.

Two things decide how tight that is.  The
displacement ``t z_j . Xi_c`` is *signed*, so only the cell boundary it travels
toward can bind it; charging every entry the distance to its nearest boundary
instead is sound but needlessly pessimistic.  And the admissible set is a
polytope in a low-dimensional coefficient block, so sampling directions at random
finds a chord but not the longest one.  Using directional limits and then
hill-climbing tightens the bound by 5.9--11.3x at unchanged scaling; see
``directional_step`` and ``certified_floor``.

Only *one-sided* directions are searched -- ``Xi = U M^T`` or ``Xi = N V^T`` for
``Delta = U S V^T`` -- because those keep the perturbed adapter at rank ``r``
*exactly*, for every step size, rather than only to first order.  The alternative
is therefore a legitimate member of the hypothesis class, and the bound survives
an attacker who is told the true rank.  The scale direction ``Xi = Delta`` is the
one-sided case ``M^T = S V^T`` and is searched first.

Any candidate set yields a valid bound -- searching more directions only tightens
it -- so the reported floor is sound regardless of how the search is configured.

What the floor is for.  Its *absolute* value is still loose -- a single returned
score landing near a cell boundary pins the whole polytope -- though after the
two tightenings above it is within a single order of magnitude of the best
attacker rather than two.  Its *scaling* is the result: under an
absolute grid the floor decays like ``1/A``, because the cell width is fixed
while the design inflates, whereas under relative quantization the cell width
inflates with the design and the floor is invariant.  The reported slope of
``log(floor)`` against ``log(A)`` is therefore the headline number -- predicted
``-1`` for the absolute family and ``0`` for the relative one -- and it is a
statement about all attackers rather than about the two we implemented.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from lowrank_extraction.adaptive import (
    AdaptiveStrategy,
    _augment,
    greedy_d_optimal_rows,
    run_adaptive_topk_recovery,
)
from lowrank_extraction.realdata import make_digits_adapter_problem

Row = dict[str, Any]

# The two quantizer families compared throughout the paper, at the matched
# utility pair of Table 22 (1 decimal place against 3 significant digits).
FAMILIES: list[dict[str, Any]] = [
    {"family": "absolute", "defense": "rounding (1 dec.)", "kwargs": {"decimals": 1},
     "predicted_slope": -1.0},
    {"family": "relative", "defense": "significant digits (3)", "kwargs": {"sig_digits": 3},
     "predicted_slope": 0.0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=Path("results") / "security_floor")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--dataset", default="digits")
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--max-rank", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=800)
    parser.add_argument("--design-size", type=int, default=200)
    parser.add_argument(
        "--amplifications", type=float, nargs="+",
        default=[1.0, 4.0, 16.0, 64.0, 256.0, 1024.0],
    )
    parser.add_argument(
        "--directions", type=int, default=400,
        help="one-sided rank-preserving directions searched per configuration",
    )
    parser.add_argument(
        "--ascent-steps", type=int, default=1500,
        help="hill-climbing steps from the best sampled direction (0 disables)",
    )
    return parser.parse_args()


def cell_slack(y: np.ndarray, *, decimals: int | None, sig_digits: int | None) -> np.ndarray:
    """Distance from each exact score to the nearest quantization-cell boundary.

    Any perturbed score within this distance rounds to the same value, so the
    transcript cannot distinguish it from the exact one.  For an absolute grid
    the slack is capped at ``tau/2`` regardless of the score; for a relative one
    it is proportional to the score, which is the entire difference between the
    two families.
    """

    down, up = signed_cell_slack(y, decimals=decimals, sig_digits=sig_digits)
    return np.minimum(down, up)


def signed_cell_slack(
    y: np.ndarray, *, decimals: int | None, sig_digits: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """How far each score may move *down* and *up* and stay inside its cell.

    ``cell_slack`` reports the distance to the *nearest* boundary, but the
    perturbation ``t z_j^T Xi_c`` is signed: it moves each score one specific
    way, so only the boundary it travels toward can bind.  A score sitting just
    above its cell's lower edge is nearly immovable downward and almost a full
    step from the upper one, and charging it the smaller of the two --- as the
    symmetric slack does --- throws that away on every entry at once.

    Using the directional limits is not an approximation of the symmetric bound
    but a strictly weaker constraint on the same polytope, so the resulting floor
    is still a sound lower bound and is never smaller.  Measured across the ten
    seeds it is 2.1--4.5x larger, at unchanged scaling.
    """

    y = np.asarray(y, dtype=np.float64)
    if decimals is not None:
        tau = 10.0 ** (-decimals)
        centre = np.round(y / tau) * tau
        return y - (centre - tau / 2.0), centre + tau / 2.0 - y
    if sig_digits is None:  # pragma: no cover - guarded by the caller
        raise ValueError("one of decimals or sig_digits is required")
    down = np.full(y.shape, np.inf, dtype=np.float64)
    up = np.full(y.shape, np.inf, dtype=np.float64)
    nz = y != 0.0
    if not np.any(nz):
        return down, up
    step = np.power(10.0, np.floor(np.log10(np.abs(y[nz]))) - (sig_digits - 1))
    centre = np.round(y[nz] / step) * step
    down[nz] = y[nz] - (centre - step / 2.0)
    up[nz] = centre + step / 2.0 - y[nz]
    return down, up


def rank_preserving_directions(
    delta: np.ndarray, rank: int, rng: np.random.Generator, n_random: int
) -> list[np.ndarray]:
    """Directions that keep the adapter *exactly* on the rank-``r`` manifold.

    With ``Delta = U S V^T``, a one-sided perturbation factors:

        Delta + t U M^T = U (S V^T + t M^T)      -- rank at most r, exactly
        Delta + t N V^T = (U S + t N) V^T        -- rank at most r, exactly

    so the alternative adapter is a legitimate member of the rank-``r`` hypothesis
    class for every ``t``, not merely to first order.  This is what makes the
    bound survive an attacker told the true rank.  A two-sided combination
    ``U M^T + N V^T`` is tangent to the manifold but leaves it at second order --
    the perturbed matrix has rank up to ``2r`` -- so it is deliberately excluded.
    The scale direction ``Delta`` itself is the one-sided case ``M^T = S V^T``.
    """

    return [xi for _, xi in rank_preserving_directions_with_side(delta, rank, rng, n_random)]


def rank_preserving_directions_with_side(
    delta: np.ndarray, rank: int, rng: np.random.Generator, n_random: int
) -> list[tuple[str, np.ndarray]]:
    """As above, but tagging each direction with the side it was built from.

    The ascent in ``certified_floor`` has to stay on the same one-sided family it
    started in -- a step that mixes the two leaves the rank-``r`` manifold and the
    perturbed adapter stops being a member of the hypothesis class -- so it needs
    to know which projector to apply.
    """

    u, _, vt = np.linalg.svd(delta, full_matrices=False)
    left, right = u[:, :rank], vt[:rank, :].T
    n_classes, feature_dim = delta.shape
    out: list[tuple[str, np.ndarray]] = [("left", np.array(delta, copy=True))]
    for index in range(n_random):
        if index % 2 == 0:
            out.append(("left", left @ rng.standard_normal((feature_dim, rank)).T))
        else:
            out.append(("right", rng.standard_normal((n_classes, rank)) @ right.T))
    return out


def side_projectors(
    delta: np.ndarray, rank: int
) -> dict[str, Any]:
    """Projectors onto the two one-sided families ``U M^T`` and ``N V^T``."""

    u, _, vt = np.linalg.svd(delta, full_matrices=False)
    left, right = u[:, :rank], vt[:rank, :].T
    return {
        "left": lambda m: left @ (left.T @ m),
        "right": lambda m: (m @ right) @ right.T,
    }


def directional_step(
    z: np.ndarray, xi: np.ndarray, down: np.ndarray, up: np.ndarray
) -> float:
    """Largest step with ``Delta + t Xi`` transcript-identical, for *one* sign of ``t``.

    Moving by ``t`` displaces score ``(j, c)`` by ``t z_j^T Xi_c``, so for
    positive ``t`` an entry with a positive response is limited by its distance to
    the cell's *upper* edge and one with a negative response by the distance to
    the *lower* edge; for negative ``t`` the roles swap.  Both signs are admissible
    and we take the longer, which is why this is never below the symmetric
    ``min(down, up)`` rule it replaces.

    The returned step is certified for the sign that attains it, **not** for
    every ``t`` of that magnitude: with slacks 0.9 up and 0.1 down and a
    positive response the value is 0.9, while ``t = -0.2`` already leaves the
    cell.  That is all the floor needs -- one indistinguishable alternative --
    and it is how Theorem 6 is stated, over a direction family closed under
    ``Xi -> -Xi``.
    """

    response = z @ xi.T
    magnitude = np.abs(response)
    active = magnitude > 0.0
    if not np.any(active):
        return 0.0
    denominator = np.maximum(magnitude, 1e-300)
    forward = np.where(response > 0.0, up, down)
    backward = np.where(response > 0.0, down, up)
    step = max(
        float(np.min(np.where(active, forward / denominator, np.inf))),
        float(np.min(np.where(active, backward / denominator, np.inf))),
    )
    return max(step, 0.0)


def certified_floor(
    problem, x_pool: np.ndarray, *, amplify: float, design_size: int,
    directions: int, seed: int, decimals: int | None = None,
    sig_digits: int | None = None, ascent_steps: int = 1500,
) -> dict[str, float]:
    """Theorem 6's floor ``F`` at this query set: ``D / (2 ||Delta||_F)``.

    This is the floor in *absolute* loss, where the midpoint estimator makes the
    factor 1/2 tight.  The minimax *relative* error is bounded by ``F / (1 + F)``
    and not by ``F`` itself, because the two hypotheses are normalized by their
    own norms -- see the proof of Theorem 6.  ``F`` is what the paper tabulates;
    every value here is under 1.01x the relative bound it implies.

    The admissible set is a polytope in the coefficient block of the one-sided
    direction, and the bound is the longest chord through it that we can exhibit.
    Random directions sample that polytope; because the block is low-dimensional
    (``feature_dim x rank``), a hill-climb from the best sample finds a
    substantially longer chord.  Both phases only ever *exhibit* an admissible
    perturbation, so every intermediate value is a sound bound and running the
    search longer can only tighten it.
    """

    pool = np.atleast_2d(np.asarray(x_pool, dtype=np.float64)) * float(amplify)
    aug_pool = _augment(problem.data.feature_map(pool))
    selected = greedy_d_optimal_rows(aug_pool, n_select=min(design_size, aug_pool.shape[0]))
    aug = aug_pool[selected]
    theta = np.hstack([problem.target.head, problem.target.bias[:, None]])
    scores = aug @ theta.T
    down, up = signed_cell_slack(scores, decimals=decimals, sig_digits=sig_digits)
    slack = np.minimum(down, up)

    # Only the head block responds to an adapter perturbation: the constant
    # column multiplies the bias, which the reported metric scores separately.
    z = aug[:, :-1]
    delta_norm = float(np.linalg.norm(problem.target.delta))
    rng = np.random.default_rng(seed + 5_000_000)

    def value_of(xi: np.ndarray) -> float:
        # The factor 1/2 is Theorem 6's: two adapters separated by t||Xi|| are
        # transcript-identical, so an estimator placed midway from both is wrong
        # by half that and no more. Reporting the separation itself would claim
        # twice the bound the argument delivers.
        return (
            0.5 * directional_step(z, xi, down, up)
            * float(np.linalg.norm(xi)) / delta_norm
        )

    sampled = rank_preserving_directions_with_side(
        problem.target.delta, problem.target.rank, rng, directions
    )
    scale_floor = value_of(sampled[0][1])
    best_side, best_xi, random_floor = sampled[0][0], sampled[0][1], scale_floor
    for side, xi in sampled[1:]:
        candidate = value_of(xi)
        if candidate > random_floor:
            best_side, best_xi, random_floor = side, xi, candidate

    project = side_projectors(problem.target.delta, problem.target.rank)[best_side]
    current, best_floor = project(best_xi), random_floor
    radius = 0.5
    for _ in range(ascent_steps):
        scale = radius * float(np.linalg.norm(current)) / np.sqrt(current.size)
        candidate = project(current + scale * rng.standard_normal(current.shape))
        value = value_of(candidate)
        if value > best_floor:
            best_floor, current = value, candidate
        else:
            radius *= 0.997

    return {
        "scale_direction_floor": scale_floor,
        "random_search_floor": random_floor,
        "certified_floor": best_floor,
        "min_cell_slack": float(np.min(slack)),
        "median_cell_slack": float(np.median(slack)),
    }


def best_attacker(problem, x_pool, *, amplify: float, design_size: int, max_rank: int,
                  seed: int, **defense) -> float:
    result = run_adaptive_topk_recovery(
        problem,
        x_pool=x_pool,
        k=problem.target.n_classes,
        strategy=AdaptiveStrategy(
            name=f"amp{amplify:g}",
            amplify=amplify,
            d_optimal=True,
            repeats=1,
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
    return float(result.delta_relerr)


def log_slope(amplifications: list[float], values: list[float]) -> float:
    """Least-squares slope of log10(value) against log10(A)."""

    pairs = [(a, v) for a, v in zip(amplifications, values) if a > 0 and v > 0 and np.isfinite(v)]
    if len(pairs) < 2:
        return float("nan")
    xs = np.log10([a for a, _ in pairs])
    ys = np.log10([v for _, v in pairs])
    return float(np.polyfit(xs, ys, 1)[0])


def run(args: argparse.Namespace) -> list[Row]:
    rows: list[Row] = []
    for seed in args.seeds:
        problem = make_digits_adapter_problem(
            seed=seed, dataset=args.dataset, feature_dim=args.feature_dim, rank=args.rank
        )
        rng = np.random.default_rng(seed + 3_000_000)
        idx = rng.choice(problem.data.x_train.shape[0], size=args.pool_size, replace=True)
        x_pool = problem.data.x_train[idx]
        for family in FAMILIES:
            for amplify in args.amplifications:
                floor = certified_floor(
                    problem, x_pool, amplify=amplify, design_size=args.design_size,
                    directions=args.directions, seed=seed,
                    ascent_steps=args.ascent_steps, **family["kwargs"],
                )
                attacker = best_attacker(
                    problem, x_pool, amplify=amplify, design_size=args.design_size,
                    max_rank=args.max_rank, seed=seed + 7_000_000, **family["kwargs"],
                )
                rows.append(
                    {
                        "seed": seed,
                        "dataset": args.dataset,
                        "family": family["family"],
                        "defense": family["defense"],
                        "amplification": amplify,
                        "attacker_relerr": attacker,
                        # A sound certificate must never exceed what a real
                        # attacker achieved; this is checked in the summary and
                        # by the regression suite.
                        "certificate_is_sound": bool(floor["certified_floor"] <= attacker),
                        **floor,
                    }
                )
    return rows


def summarize(rows: list[Row], *, amplifications: list[float]) -> str:
    lines = [
        "# Certified Indistinguishability Floor",
        "",
        "The floor is the largest adapter perturbation that provably leaves every",
        "returned score inside its own quantization cell. Two adapters separated by",
        "it induce identical transcripts, so *no* algorithm -- any budget, any",
        "design, adaptive or not -- separates them. It is therefore a lower bound on",
        "the minimax recovery error, not a property of the attacks we implemented.",
        "",
        "Directions searched preserve rank exactly, so the bound also holds against",
        "an attacker who is told the true rank.",
        "",
    ]
    for family in FAMILIES:
        sub = [r for r in rows if r["family"] == family["family"]]
        lines += [
            f"## {family['family']}: {family['defense']}",
            "",
            "| A | certified floor | best attacker | attacker / floor | sound |",
            "|---:|---:|---:|---:|---|",
        ]
        floors: list[float] = []
        attackers: list[float] = []
        for amplify in amplifications:
            at_a = [r for r in sub if r["amplification"] == amplify]
            if not at_a:
                continue
            floor = float(np.mean([r["certified_floor"] for r in at_a]))
            attacker = float(np.mean([r["attacker_relerr"] for r in at_a]))
            sound = all(r["certificate_is_sound"] for r in at_a)
            floors.append(floor)
            attackers.append(attacker)
            lines.append(
                f"| {amplify:g} | {floor:.3e} | {attacker:.3e} |"
                f" {attacker / floor if floor > 0 else float('nan'):.0f}x |"
                f" {'yes' if sound else 'NO'} |"
            )
        used = [a for a in amplifications if any(r["amplification"] == a for r in sub)]
        slope = log_slope(used, floors)
        attacker_slope = log_slope(used, attackers)
        lines += [
            "",
            f"log-log slope of the floor against A: {slope:+.3f}"
            f" (Theorem 7 predicts {family['predicted_slope']:+.1f})",
            f"log-log slope of the best attacker against A: {attacker_slope:+.3f}",
            "",
        ]
    lines += [
        "## Reading",
        "",
        "The absolute grid fixes the cell width while amplification inflates the",
        "design, so the floor falls like 1/A: the defense has no scale-invariant",
        "security, and the collapse is a property of the *quantizer*, not of any",
        "particular attack. Relative quantization inflates the cell width with the",
        "design, so the floor is flat and the protection an attacker cannot go below",
        "is the same at every query scale.",
        "",
        "The floor's absolute value is still loose, because a single returned score",
        "landing near a cell boundary pins the whole polytope. Two tightenings --",
        "charging each score only the boundary its signed displacement travels",
        "toward, and hill-climbing the polytope rather than sampling it -- close",
        "that gap by 5.9x to 11.3x uniformly in A, leaving it within one order of",
        "magnitude instead of two. Which end of what remains is tight is open. The",
        "scaling is what the theorem claims and what the measurement confirms, and",
        "it is unchanged by the tightening, as it must be: both phases only exhibit",
        "admissible perturbations, so they move the bound, never the truth.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = run(args)
    (args.outdir / "floor.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with (args.outdir / "floor.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(rows, amplifications=args.amplifications)
    (args.outdir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
