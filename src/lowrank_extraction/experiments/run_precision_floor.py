"""The floor that survives every adaptive strategy, when requests are discrete.

Theorem 6 bounds recovery error at *the query set the attacker chose*, and
Proposition 4 shows an attacker that drives that bound to zero by placing cell
boundaries where it likes.  What neither settles is the quantity a defender
actually wants, the infimum over query sets.  On a request format carrying ``d`` significant
digits it is characterisable, and this experiment measures it.

The observation is that a transcript separates ``theta`` from ``theta'`` if and
only if some *achievable threshold* lies between them, where the achievable
thresholds are

    T = { (b - b_c) / s  :  b a cell boundary,  s an admissible query scale }.

Every adaptive strategy, at any budget, only ever learns which side of points of
``T`` the secret lies on, so the consistent set is the maximal ``T``-free
interval around ``theta`` and

    inf over all query sets of the achievable error  =  half the gap of T.

That is exact rather than a bound, and it is computable.  For a relative
quantizer of ``p`` significant digits and a ``d``-digit request format the
achievable thresholds are ``N/(2M) * 10^k`` with ``N`` odd of ``p`` digits and
``M`` of ``d`` digits, whose gaps an integer argument bounds below by
``(theta/4) * 10^-(p+d+1)`` -- so the served precision and the requested
precision compose *multiplicatively*, and neither alone determines the floor.

Three quantities are reported per cell, and the theorem is the claim that they
agree up to constants:

  certified   the exact gap of T, enumerated -- the infimum itself
  bound       the closed form (1/4) * 10^-(p+d+1) from the theorem
  attacker    what the joint-search bisection actually achieves

Enumeration is exact rational arithmetic: for each admissible ``M`` only the
few ``N`` nearest ``theta`` can matter, so the whole set need not be built.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path

import numpy as np


def achievable_gap(theta: float, p: int, d: int) -> tuple[float, float, float]:
    """Exact neighbours of ``theta`` in ``T``, by rational arithmetic.

    Returns the gap of ``T`` straddling ``theta``, its width relative to
    ``theta``, and the exact minimax *relative* error over that gap.  Only the
    ``N`` nearest ``theta`` for each ``M`` can bound the gap, so the enumeration
    is linear in the number of admissible ``M``.

    The third value is the one the paper reports.  Relative error is
    ``|theta_hat - theta| / |theta|`` (Definition 4), so over a consistent
    interval ``(a, b)`` the minimax estimator is the harmonic midpoint
    ``2ab/(a+b)`` and the minimax risk is ``(b - a)/(a + b)`` -- attained, and
    independent of where ``theta`` sits inside the gap.  The second value,
    ``gap/theta``, halves to the *absolute* radius divided by the sampled
    secret; it is a conservative proxy, never above the minimax value, and is
    retained only so the two can be compared.
    """

    # T holds positive numbers only, so a negative secret is handled by
    # reflection rather than by enumerating a signed set: Q_p is odd, so the
    # threshold set on the negative side is exactly -T and the gap straddling
    # -theta is the mirror of the gap straddling |theta|.  Without this the
    # enumeration below simply finds no admissible N and returns NaN.
    th = Fraction(abs(theta)).limit_denominator(10**15)
    below, above = None, None
    n_lo, n_hi = 2 * 10 ** (p - 1) + 1, 2 * 10**p - 1

    for M in range(10 ** (d - 1), 10**d):
        for k in range(-(p + d) - 2, (p + d) + 3):
            pow10 = Fraction(10) ** k
            # t = N/(2M) * 10^k ~= theta  ->  N ~= 2*M*theta / 10^k
            centre = 2 * M * th / pow10
            n0 = int(centre)
            for N in (n0 - 2, n0 - 1, n0, n0 + 1, n0 + 2):
                if N % 2 == 0 or not (n_lo <= N <= n_hi):
                    continue
                t = Fraction(N, 2 * M) * pow10
                if t == th:
                    # The secret sits *on* an achievable threshold.  Cells are
                    # half-open (ties away from zero), so such a theta is the
                    # left edge of its own cell and the consistent set is the
                    # gap to its right, not the union of the gaps either side.
                    # Continuous sampling never lands here, but the exact-floor
                    # enumeration must not silently widen the gap when it does.
                    below = th
                elif t < th:
                    if below is None or t > below:
                        below = t
                elif t > th:
                    if above is None or t < above:
                        above = t

    if below is None or above is None:
        return float("nan"), float("nan"), float("nan")
    gap = above - below
    return float(gap), float(gap / th), float(gap / (below + above))


def uniform_floor(p: int, d: int) -> tuple[float, float, int]:
    """The floor a defender may claim: the infimum of ``achievable_gap`` over theta.

    ``achievable_gap`` answers for one secret, so a table of it over sampled
    theta says what a typical secret gets, not what every secret gets.  A server
    advertising a floor needs the latter, which is

        inf over theta of  (minimax relative error on the gap straddling theta),

    and that is what this computes -- exactly, by enumerating T rather than by
    sampling.  ``T`` is invariant under multiplication by ten and so is the
    ratio, so one decade settles every theta.  Relative error is measured
    against the true secret (Definition 4), so on a gap ``(a, b)`` the minimax
    risk is ``(b - a)/(a + b)``: it does not depend on where theta sits inside
    the gap, so the infimum over one decade is a minimum over finitely many
    gaps, and is attained.

    Also returned is the older half-gap convention ``(b - a)/(2b)``, the
    absolute radius divided by the sampled secret at its least favorable
    point.  It never exceeds the minimax value, so it is conservative -- a
    server quoting it never claims more privacy than it has -- but it is not
    the minimax error, and it is approached rather than attained.  The two
    differ by at most 0.3% over the enumerated cells.

    Returns ``(inf_minimax_relative, inf_half_gap_over_theta, |T in one decade|)``.
    """

    ten = Fraction(10)
    pts = set()
    n_lo, n_hi = 2 * 10 ** (p - 1) + 1, 2 * 10**p - 1
    for M in range(10 ** (d - 1), 10**d):
        for N in range(n_lo, n_hi + 1, 2):
            t = Fraction(N, 2 * M)
            while t >= 10:
                t /= ten
            while t < 1:
                t *= ten
            pts.add(t)
    ordered = sorted(pts)
    # Wrap the decade so the gap spanning its right edge is the real one.
    ext = ordered + [ordered[0] * ten]

    worst_mmax = min((b - a) / (a + b) for a, b in zip(ext, ext[1:]) if b > a)
    worst_half = min((b - a) / (2 * b) for a, b in zip(ext, ext[1:]) if b > a)
    return float(worst_mmax), float(worst_half), len(ordered)


def exact_floor(p: int, d: int) -> float:
    """Alias for :func:`uniform_floor_closed`, the name used by consumers.

    Both the request-model and request-utility experiments read Theorem 7's
    floor off this rather than off part (ii)'s proven constant, which is twenty
    times smaller and would overstate the defense's cost by a digit.

    Every cell either experiment passes here is one where the closed form is
    known to be the infimum: all of them satisfy ``p + d <= 6`` and are
    enumerated outright, except ``(7, 7)``, which lies on the diagonal where
    :func:`uniform_floor_closed` documents an attaining pair.
    """

    return uniform_floor_closed(p, d)


def uniform_floor_closed(p: int, d: int) -> float:
    """Closed form ``1/(2*(10^d-1)*(2*10^p-1) - 1)``, compared against enumeration.

    The narrowest relative gap comes from two thresholds ``N*10^j/(2M)`` and
    ``N'*10^j'/(2M')`` whose cross-difference is ``+-1``: the gap is then
    ``1/(2MM')`` and, divided by the *sum* of the two, equals
    ``1/(2*M*N' - 1)``.  Making that smallest means taking ``M`` and ``N'`` as
    large as the formats allow, giving the value above -- about
    ``(1/4)*10^-(p+d)``, twenty times the theorem's proven
    ``(1/8)*10^-(p+d+1)``.

    The ``-1`` is exactly the difference between the minimax denominator
    ``a + b`` and the conservative ``2b`` of the older half-gap convention: the
    two closed forms differ only in that term, by at most 0.3% over the
    enumerated cells, and the same pair attains both.

    Whenever ``p >= d`` the witnessing pair is explicit: with
    ``N = 2*10^p - 2*10^(p-d) - 1`` -- an integer exactly because ``p >= d`` --
    the points ``a = N*10/(2*(10^d-1))`` and ``b = (2*10^p-1)/(2*10^(d-1))``
    both lie in ``T`` and differ by ``1/(2*10^(d-1)*(10^d-1))``, so
    ``(b-a)/(a+b)`` is the formula exactly.  Setting ``d = p`` recovers the
    diagonal pair.  That is what lets ``(7, 7)`` and ``(7, 1)`` be read off the
    formula rather than enumerated, in the upper-bound direction only.

    For ``p < d`` it is reported next to the enumeration rather than
    asserted, and that is exactly where the three exceptions below sit.  Over all fifteen cells with ``p + d <= 6`` it is the enumerated
    infimum in twelve and strictly below it in three -- ``(1, 2)``, ``(1, 5)``
    and ``(2, 4)``, every one of them with ``p < d`` -- and it is never above
    it.  We do not claim it as a
    universal lower bound: the elementary argument gives ``1/(2*M*N')`` only
    when the larger threshold carries the smaller decade shift, and loses a
    factor of ten otherwise, which is the regime those three cells sit in.
    """

    return 1.0 / (2.0 * (10.0**d - 1.0) * (2.0 * 10.0**p - 1.0) - 1.0)


def snap_relative(v: np.ndarray, p: int) -> np.ndarray:
    """Nearest cell boundary of a ``p``-significant-digit quantizer."""

    from lowrank_extraction.experiments.run_adaptive_boundary import snap_to_boundary

    return snap_to_boundary(v, decimals=None, sig_digits=p)


def attack_scalar(theta: float, p: int, d: int, *, steps: int) -> float:
    """Joint-search bisection on one scalar, homogeneous probing (no bias)."""

    mantissas = np.arange(10 ** (d - 1), 10**d, dtype=np.float64)
    scales = mantissas * 10.0 ** (-(d - 1))
    # Reflect a negative secret rather than bracketing it backwards: with
    # theta < 0, ``theta * 0.9`` exceeds ``theta * 1.1`` and the bisection below
    # would start with an inverted interval.  The transcript for -theta is the
    # negation of the transcript for theta, so the relative error is the same.
    theta = abs(theta)
    lo, hi = theta * 0.9, theta * 1.1
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        boundary = snap_relative(scales * mid, p)
        thresholds = boundary / scales
        inside = (thresholds > lo) & (thresholds < hi)
        if not inside.any():
            break
        j = int(np.argmin(np.abs(thresholds[inside] - mid)))
        t = float(thresholds[inside][j])
        if theta > t:
            lo = t
        else:
            hi = t
    return abs(0.5 * (lo + hi) - theta) / abs(theta)


def attack_free_threshold(theta: float, *, steps: int) -> float:
    """The adaptive one-bit compressed-sensing attacker, for contrast.

    That literature lets the *sensor's* threshold be chosen adaptively: the
    attacker observes ``sign(<a,x> - tau)`` for any real ``tau`` it names, and
    error then decays exponentially in the number of measurements with no floor
    at all.  Our attacker cannot do this.  It never names a threshold; it names
    a query scale, and the thresholds it thereby reaches are the fixed cell
    boundaries of the served quantizer pulled back through that scale -- the
    discrete set ``T``.  Running plain bisection on a free ``tau`` is therefore
    not a competing algorithm but a measurement of how much that modeling
    difference is worth.  It is worth everything: this attacker has no floor, so
    its error is simply whatever its budget buys, ``2^-steps`` of the starting
    bracket, and it keeps falling for as long as one keeps querying.  The
    comparison is therefore only meaningful at a *matched* budget, which is what
    the caller supplies -- quoting an unbudgeted number would report the
    underflow of the bracket rather than a property of the attacker.
    """

    theta = abs(theta)  # reflect, for the same reason as ``attack_scalar``
    lo, hi = theta * 0.9, theta * 1.1
    for _ in range(steps):
        tau = 0.5 * (lo + hi)  # a real threshold, nameable to full precision
        if theta > tau:
            lo = tau
        else:
            hi = tau
    return abs(0.5 * (lo + hi) - theta) / abs(theta)


@dataclass
class Row:
    seed: int
    served_digits: int
    request_digits: int
    theta: float
    certified_rel: float
    half_gap_rel: float
    bound_rel: float
    attacker_rel: float
    free_threshold_rel: float


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("results") / "precision_floor")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--served-digits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--request-digits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument(
        "--budget", type=int, default=40,
        help="matched query budget for both attackers; ours reaches its floor "
             "well inside it, the free-threshold one never does",
    )
    ap.add_argument(
        "--max-enumerable", type=int, default=6,
        help="largest p+d for which the uniform floor is enumerated exactly. "
             "The decade holds about 6.5*10^(p+d-1) thresholds, so beyond this "
             "the enumeration stops being worth its memory; those cells report "
             "the closed form only, and are labelled as such.",
    )
    ap.add_argument(
        "--negative-secrets", action="store_true",
        help="draw theta from [-10,-1] instead of [1,10]. Theorem 7's threshold "
             "set T contains positive numbers only, so the negative case is a "
             "separate claim that the default run never exercises; this flag is "
             "what checks it rather than asserting it.",
    )
    ap.add_argument(
        "--closed-form-scan", action="store_true",
        help="instead of the seeded sweep, enumerate every (p, d) with "
             "p + d <= max-sum and write closed_form.json: the exact minimax "
             "relative infimum over one decade against the closed form "
             "1/(2*(10^d-1)*(2*10^p-1) - 1). This is the evidence for "
             "Theorem 7(iii)'s twelve-of-fifteen claim, and it is enumerated "
             "rather than asserted.",
    )
    ap.add_argument(
        "--max-sum", type=int, default=6,
        help="largest p + d to enumerate under --closed-form-scan.",
    )
    args = ap.parse_args()

    if args.closed_form_scan:
        args.outdir.mkdir(parents=True, exist_ok=True)
        cells = []
        for s in range(2, args.max_sum + 1):
            for pp in range(1, s):
                dd = s - pp
                enumerated, half_gap, npts = uniform_floor(pp, dd)
                closed = uniform_floor_closed(pp, dd)
                cells.append({
                    "p": pp, "d": dd,
                    "enumerated": enumerated,
                    "closed_form": closed,
                    "half_gap": half_gap,
                    "ratio": enumerated / closed,
                    "exact": abs(enumerated - closed) <= 1e-15 * enumerated,
                    "decade_points": npts,
                })
        exact_cells = [c for c in cells if c["exact"]]
        witness = []
        for pp in range(1, 9):
            tp = 10**pp
            wa = Fraction((2 * tp - 3) * 10, 2 * (tp - 1))
            wb = Fraction(2 * tp - 1, 2 * 10 ** (pp - 1))
            wgap = wb - wa
            wmmax = wgap / (wa + wb)
            witness.append({
                "p": pp,
                "a": str(wa), "b": str(wb), "gap": str(wgap),
                "relative_minimax": str(wmmax),
                "matches_closed_form":
                    wmmax == Fraction(1, 2 * (tp - 1) * (2 * tp - 1) - 1),
            })
        payload = {
            "description":
                "Theorem 7(iii): the exact minimax relative error (b-a)/(a+b) "
                "over the achievable-threshold set T, enumerated over one "
                "decade, against the closed form "
                "1/(2*(10^d-1)*(2*10^p-1) - 1), for every (p, d) with "
                "p + d <= %d, and the explicit attaining pair on the diagonal "
                "p = d. `half_gap` is the older conservative convention "
                "(b-a)/(2b), never above the minimax value."
                % args.max_sum,
            "cells": cells,
            "summary": {
                "n_cells": len(cells),
                "n_exact": len(exact_cells),
                "exceptions": [[c["p"], c["d"]] for c in cells if not c["exact"]],
                "closed_form_ever_exceeds_enumeration": any(
                    c["closed_form"] > c["enumerated"] for c in cells),
                "half_gap_ever_exceeds_minimax": any(
                    c["half_gap"] > c["enumerated"] for c in cells),
                "max_half_gap_shortfall_pct": max(
                    100.0 * (1.0 - c["half_gap"] / c["enumerated"]) for c in cells),
            },
            "diagonal_witness": witness,
        }
        (args.outdir / "closed_form.json").write_text(json.dumps(payload, indent=2))
        print(f"{len(exact_cells)} of {len(cells)} cells attain the closed form")
        print("exceptions:", payload["summary"]["exceptions"])
        for c in cells:
            if not c["exact"]:
                print(f"  ({c['p']}, {c['d']}): enumerated exceeds closed form by "
                      f"{100.0 * (c['ratio'] - 1.0):.4f}%")
        print("diagonal witness matches the closed form for p = 1..8:",
              all(w["matches_closed_form"] for w in witness))
        print(f"half-gap convention below the minimax by at most "
              f"{payload['summary']['max_half_gap_shortfall_pct']:.4f}%")
        return

    rows: list[Row] = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(1.0, 10.0))
        if args.negative_secrets:
            theta = -theta
        for p in args.served_digits:
            for d in args.request_digits:
                _, half_rel, cert_rel = achievable_gap(theta, p, d)
                rows.append(
                    Row(
                        seed=seed, served_digits=p, request_digits=d, theta=theta,
                        certified_rel=cert_rel,
                        half_gap_rel=0.5 * half_rel,
                        # The theorem bounds the *gap* below by
                        # (1/4)*10^-(p+d+1); the minimax relative error is
                        # (b-a)/(a+b), which is what `certified` reports, and
                        # `half_gap` is the conservative (b-a)/(2b) convention.
                        bound_rel=0.125 * 10.0 ** (-(p + d + 1)),
                        attacker_rel=attack_scalar(theta, p, d, steps=args.budget),
                        free_threshold_rel=attack_free_threshold(
                            theta, steps=args.budget
                        ),
                    )
                )

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "precision_floor.json").write_text(
        json.dumps(
            {"config": {"seeds": args.seeds,
                        "served_digits": args.served_digits,
                        "request_digits": args.request_digits,
                        "budget": args.budget,
                        "negative_secrets": bool(args.negative_secrets)},
             "rows": [asdict(r) for r in rows]},
            indent=2,
        )
    )
    with (args.outdir / "precision_floor.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    lines = [
        f"seeds: {len(args.seeds)}",
        "",
        "`certified` is the exact minimax *relative* error over the achievable-",
        "threshold gap straddling the secret, (b-a)/(a+b): the minimax error",
        "over every adaptive query set at any budget, and exact rather than a",
        "bound.  Relative error is measured against the true secret, so this is",
        "attained and does not depend on where the secret sits in the gap.",
        "`half gap` is the older conservative convention (b-a)/(2b), never above",
        "it.  Being minimax it is a worst-case statement over",
        "the secrets a transcript cannot separate, so an attacker measured at a",
        "secret drawn uniformly inside that gap lands below it -- around half of",
        "it on average -- and a ratio under 1 is expected, not a violation.",
        "`bound` is the theorem's closed form (1/4)*10^-(p+d+1), which fixes the",
        "exponent with a deliberately conservative constant.",
        "",
        "`free tau` is the adaptive one-bit compressed-sensing model, where the",
        "threshold itself is chosen by the attacker rather than reached through a",
        "query scale.  It has no floor, and the gap between it and `certified` is",
        "exactly what the discreteness of the request format buys the defender.",
        "",
        "| served p | request d | certified (median) | half gap (median) | bound "
        "| attacker (median) | free tau (median) | attacker/certified "
        "| certified/bound |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    fit_x: list[float] = []
    fit_y: list[float] = []
    for p in args.served_digits:
        for d in args.request_digits:
            sel = [r for r in rows if r.served_digits == p and r.request_digits == d]
            cert = float(np.median([r.certified_rel for r in sel]))
            half = float(np.median([r.half_gap_rel for r in sel]))
            att = float(np.median([r.attacker_rel for r in sel]))
            free = float(np.median([r.free_threshold_rel for r in sel]))
            bound = sel[0].bound_rel
            fit_x.append(float(p + d))
            fit_y.append(float(np.log10(cert)))
            lines.append(
                f"| {p} | {d} | {cert:.3e} | {half:.3e} | {bound:.3e} | "
                f"{att:.3e} | {free:.3e} | {att / cert:.2f} | {cert / bound:.1f} |"
            )
    slope, intercept = np.polyfit(np.array(fit_x), np.array(fit_y), 1)

    # The table above is at sampled theta.  A server advertising a floor needs
    # the worst theta, which is a different number and the one Theorem 7(ii)
    # should be read against.
    lines += [
        "",
        "The medians above are what a *typical* secret gets.  A defender needs",
        "the worst one, so `uniform` is the exact infimum over theta of the same",
        "quantity, enumerated rather than sampled.  `closed` is the candidate",
        "closed form 1/(2*(10^d-1)*(2*10^p-1) - 1) ~ (1/4)*10^-(p+d), and",
        "`half gap` is the older conservative convention (b-a)/(2b) -- never",
        "above `uniform`, and below it by at most 0.3% over these cells.",
        "",
        "| served p | request d | uniform | closed | half gap | bound | uniform/bound |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in args.served_digits:
        for d in args.request_digits:
            if p + d > args.max_enumerable:
                lines.append(
                    f"| {p} | {d} | not enumerated (p+d > "
                    f"{args.max_enumerable}) | {uniform_floor_closed(p, d):.3e} "
                    f"| - | {0.125 * 10.0 ** (-(p + d + 1)):.3e} | - |"
                )
                continue
            unif, mmax, _ = uniform_floor(p, d)
            bound = 0.125 * 10.0 ** (-(p + d + 1))
            lines.append(
                f"| {p} | {d} | {unif:.4e} | {uniform_floor_closed(p, d):.4e} | "
                f"{mmax:.4e} | {bound:.3e} | {unif / bound:.1f} |"
            )

    lines += [
        "",
        f"log10(certified) against (p + d): slope {slope:.3f}, intercept "
        f"{intercept:.3f} (theorem predicts slope -1, i.e. the served and the "
        f"requested precision compose multiplicatively).",
        "",
        "Doubling the budget separates the two models: the free-threshold error",
        "falls by a further 2^-budget, ours does not move at all, because it is",
        "at a floor rather than at a rate.  A free-threshold entry of 0 is the",
        "bracket collapsing below double precision -- the model has no floor to",
        "report, which is exactly the point of the comparison.",
        "",
        f"| served p | request d | ours at {args.budget} | ours at "
        f"{2 * args.budget} | free tau at {args.budget} | free tau at "
        f"{2 * args.budget} |",
        "|---|---|---|---|---|---|",
    ]
    for p in args.served_digits:
        for d in args.request_digits:
            sel = [r for r in rows if r.served_digits == p and r.request_digits == d]
            ours_1 = float(np.median([r.attacker_rel for r in sel]))
            ours_2 = float(
                np.median(
                    [attack_scalar(r.theta, p, d, steps=2 * args.budget) for r in sel]
                )
            )
            free_1 = float(np.median([r.free_threshold_rel for r in sel]))
            free_2 = float(
                np.median(
                    [attack_free_threshold(r.theta, steps=2 * args.budget) for r in sel]
                )
            )
            lines.append(
                f"| {p} | {d} | {ours_1:.3e} | {ours_2:.3e} | {free_1:.3e} | "
                f"{free_2:.3e} |"
            )
    (args.outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
