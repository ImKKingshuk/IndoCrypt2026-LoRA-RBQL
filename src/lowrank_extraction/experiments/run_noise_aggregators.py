"""Does repetition against a stochastic server help an attacker that uses it well?

Section S12 measures one repeated-query attacker: it averages the ``R`` served
values and feeds the mean into the same cell-membership predicate the noiseless
search uses.  That attacker is sometimes *worse* at ``R=64`` than at ``R=1``,
which the section reads as repetition being harmful.

The reading does not follow from the measurement.  An attacker handed ``R``
draws may discard ``R-1`` of them, so no optimal attacker can be harmed by
receiving more; whatever is degrading must be the estimator, not the
information.  This script separates the two by holding the server fixed and
varying only what the attacker does with the draws:

``single``     one draw, ``R=1``.  The reference the others are judged against.
``first``      take ``R`` draws and discard all but one.  The fallback: costs
               ``R`` queries and must reproduce ``single`` exactly.
``mean``       average the draws, test cell membership.  Section S12's attacker.
``mode``       majority vote on the returned representative, test cell
               membership.  The vote returns an on-grid value, so the predicate
               is being asked the question it was designed for.
``mean-thr``   average the draws, but compare the average against the published
               boundary directly rather than against the representative of the
               cell above it.  The average is an estimate of the score, so the
               test that suits it is a threshold test.

Everything else -- server, noise, quantizer, query accounting, scoring against
the true secret -- is the tracked experiment's, imported rather than restated.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, asdict
from math import fsum
from pathlib import Path

import numpy as np

from lowrank_extraction.experiments.run_adaptive_boundary import (
    FloatArray,
    LinearLowRankTarget,
    cell_boundary_in_decade,
    make_target,
    quantize,
    relerr,
    represent,
)

POLICIES = ("single", "first", "mean", "mode", "mean-thr")


class PolicyServer:
    """The stochastic server of Section S12, with the aggregation exposed.

    The server's own behaviour is unchanged: it draws fresh noise per repeat and
    quantizes each draw, so one answer is ``Q(y + eps)``.  What varies is how
    the attacker collapses the ``R`` answers it is handed.
    """

    def __init__(self, target: LinearLowRankTarget, rng: np.random.Generator,
                 *, sigma: float, repeats: int, policy: str) -> None:
        self._target = target
        self._rng = rng
        self.sigma = float(sigma)
        self.repeats = int(repeats)
        self.policy = policy
        self.calls = 0

    def __getattr__(self, name: str):
        return getattr(self._target, name)

    def serve(self, x: FloatArray, *, decimals: int | None,
              sig_digits: int | None) -> FloatArray:
        clean = self._target.logits(x)
        self.calls += self.repeats
        if self.sigma <= 0.0:
            return quantize(clean, decimals=decimals, sig_digits=sig_digits)

        draws = np.empty((self.repeats, *np.shape(clean)), dtype=np.float64)
        for r in range(self.repeats):
            noisy = clean + self._rng.normal(0.0, self.sigma, size=clean.shape)
            draws[r] = quantize(noisy, decimals=decimals, sig_digits=sig_digits)

        flat = draws.reshape(self.repeats, -1)
        if self.policy in ("single", "first"):
            out = flat[0].copy()
        elif self.policy == "mode":
            out = np.array([_mode(flat[:, j]) for j in range(flat.shape[1])])
        else:  # mean and mean-thr aggregate identically; they differ at the test
            # fsum, as in the tracked experiment: the mean of equal draws must be
            # exact so that the sigma -> 0 limit reduces to a single query.
            out = np.array(
                [fsum(flat[:, j]) / self.repeats for j in range(flat.shape[1])],
                dtype=np.float64,
            )
        return out.reshape(np.shape(clean))


def _mode(values: FloatArray) -> float:
    """The most frequently returned representative; ties go to the one nearest
    the mean, so the rule is deterministic and does not depend on draw order."""

    counts = Counter(float(v) for v in values)
    best = max(counts.values())
    tied = [v for v, n in counts.items() if n == best]
    if len(tied) == 1:
        return tied[0]
    centre = fsum(float(v) for v in values) / len(values)
    return min(tied, key=lambda v: abs(v - centre))


def _above(served: float, beta: float, *, decimals: int | None,
           sig_digits: int | None, policy: str) -> bool:
    """Which side of the published boundary ``beta`` the attacker concludes on.

    For an on-grid answer -- a single draw, or a majority vote over draws -- the
    exact test is against the representative of the cell above ``beta``, since
    ``Q`` is monotone.  For an average, which is not a cell representative, the
    natural test is against ``beta`` itself.
    """

    if policy == "mean-thr":
        return bool(served >= beta)
    step = abs(beta) * 1e-9 + 1e-15
    upper = float(quantize(np.array([beta + step]),
                           decimals=decimals, sig_digits=sig_digits)[0])
    return bool(served >= upper)


def _locate_crossing(server, *, i: int, c: int, beta: float, s_guess: float,
                     bits: int, decimals: int | None, sig_digits: int | None,
                     input_digits: int | None) -> tuple[float, int]:
    """Bisect in scale space for the ``s`` at which the score crosses ``beta``.

    Identical to the tracked experiment's search apart from the decision rule,
    which the policy supplies.  Every probe is charged in full, and a repeated
    probe is charged ``R`` calls by the server itself.
    """

    h = server.n_features
    queries = 0

    def probe(s: float) -> bool:
        nonlocal queries
        x = np.zeros(h)
        x[i] = s
        y = server.serve(x, decimals=decimals, sig_digits=sig_digits)[0][c]
        queries += 1
        return _above(float(y), beta, decimals=decimals,
                      sig_digits=sig_digits, policy=server.policy)

    lo, hi = s_guess * 0.8, s_guess * 1.2
    a = probe(represent(lo, input_digits))
    b = probe(represent(hi, input_digits))
    widen = 0
    while a == b and widen < 40:
        mid = 0.5 * (lo + hi)
        half = hi - lo
        lo, hi = mid - half, mid + half
        a = probe(represent(lo, input_digits))
        b = probe(represent(hi, input_digits))
        widen += 1
    if a == b:
        return float("nan"), queries

    for _ in range(bits):
        mid_raw = 0.5 * (lo + hi)
        mid = represent(mid_raw, input_digits)
        if mid == represent(lo, input_digits) or mid == represent(hi, input_digits):
            break
        if probe(mid) == a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), queries


def attack(server, *, bits: int, decimals: int | None, sig_digits: int | None,
           input_digits: int | None) -> tuple[FloatArray, int]:
    """The unknown-bias two-boundary search of Section S12, under one policy."""

    m, h = server.n_classes, server.n_features
    beta1 = cell_boundary_in_decade(0, decimals=decimals, sig_digits=sig_digits)
    beta2 = cell_boundary_in_decade(1, decimals=decimals, sig_digits=sig_digits)
    weight_est = np.zeros((m, h))
    queries = 0

    served_zero = server.serve(
        np.zeros(h), decimals=decimals, sig_digits=sig_digits)[0]
    queries += 1

    for i in range(h):
        x = np.zeros(h)
        x[i] = 1.0
        served_one = server.serve(
            x, decimals=decimals, sig_digits=sig_digits)[0]
        queries += 1

        for c in range(m):
            b_hat = float(served_zero[c])
            t_hat = float(served_one[c]) - b_hat
            if t_hat == 0.0:
                continue
            crossings = []
            for beta in (beta1, beta2):
                s, q = _locate_crossing(
                    server, i=i, c=c, beta=beta, s_guess=(beta - b_hat) / t_hat,
                    bits=bits, decimals=decimals, sig_digits=sig_digits,
                    input_digits=input_digits)
                queries += q
                crossings.append(s)
            s1, s2 = crossings
            if not (np.isfinite(s1) and np.isfinite(s2)) or s1 == s2:
                weight_est[c, i] = t_hat
                continue
            weight_est[c, i] = (beta1 - beta2) / (s1 - s2)

    return weight_est, queries


@dataclass
class Row:
    seed: int
    grid: str
    input_digits: str
    sigma: float
    repeats: int
    policy: str
    queries: int
    delta_relerr: float


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--feature-dim", type=int, default=8)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--bits", type=int, default=60)
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--sig-digits", type=int, default=4)
    ap.add_argument("--noise-sigma", type=float, nargs="+",
                    default=[1e-6, 1e-3])
    ap.add_argument("--repeats", type=int, default=64)
    ap.add_argument("--input-digits", type=int, nargs="+", default=[12])
    ap.add_argument("--unbounded", action="store_true", default=True,
                    help="also run the unbounded-request-precision row")
    args = ap.parse_args()

    grids = (("absolute (decimals=%d)" % args.decimals,
              dict(decimals=args.decimals, sig_digits=None)),
             ("relative (sig=%d)" % args.sig_digits,
              dict(decimals=None, sig_digits=args.sig_digits)))
    digit_rows: list[int | None] = list(args.input_digits)
    if args.unbounded:
        digit_rows.append(None)

    rows: list[Row] = []
    for seed in args.seeds:
        target = make_target(np.random.default_rng(seed),
                             m=args.classes, h=args.feature_dim, r=args.rank)
        for grid_name, fmt in grids:
            for digits in digit_rows:
                for sigma in args.noise_sigma:
                    for policy in POLICIES:
                        reps = 1 if policy == "single" else args.repeats
                        server = PolicyServer(
                            target, np.random.default_rng(10_000 + seed),
                            sigma=sigma, repeats=reps, policy=policy)
                        w, q = attack(server, bits=args.bits,
                                      input_digits=digits, **fmt)
                        rows.append(Row(
                            seed=seed, grid=grid_name,
                            input_digits="inf" if digits is None else str(digits),
                            sigma=sigma, repeats=reps, policy=policy,
                            queries=server.calls,
                            delta_relerr=relerr(target, w)))
                        print(f"seed={seed} {grid_name} d={digits} "
                              f"sigma={sigma:g} {policy:9s} "
                              f"err={rows[-1].delta_relerr:.3e}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Repeated-query attackers against the stochastic server of "
            "Section S12, varying only how the attacker collapses the R draws."),
        "config": vars(args) | {"out": str(args.out)},
        "rows": [asdict(r) for r in rows],
    }
    (args.out / "noise_aggregators.json").write_text(
        json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {args.out / 'noise_aggregators.json'}")


if __name__ == "__main__":
    main()
