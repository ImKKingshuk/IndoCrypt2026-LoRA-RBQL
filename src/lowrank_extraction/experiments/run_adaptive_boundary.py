"""Adaptive boundary-seeking recovery against a quantized score oracle.

Theorem 6 lower-bounds the recovery error by ``F(Q)``, evaluated at *the query
set the attacker chose*.  The amplification attacker of Corollary 1 leaves the
cell slacks ``gamma`` alone and so runs into that bound; a relative quantizer
then holds it up.  This module implements the attacker that does not: it places
a quantizer cell boundary at the midpoint of its current uncertainty interval,
reads off which side the response falls on, and halves the interval.  Each
answer is one bit, the slack it induces is driven deliberately to zero, and
``F`` collapses at the query set it chose -- entirely consistently with
Theorem 6, and fatally for the reading that the relative floor is uniform.

The experiment measures the one thing that decides the defense recommendation:
what is actually left bounding the attacker once amplification is not the only
lever.  The answer is the resolution of the attacker's *own* queries, not the
precision of the returned score.  The required query magnitude converges to a
constant (``boundary/theta``), so bounding the input norm does not help; only
bounding the input's representable precision does.

Interface: freely chosen continuous query vectors with an exactly returned
quantized score -- the same interface amplification needs, so the two attackers
are compared on equal terms rather than the new one being handed a stronger
oracle.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from math import fsum

import numpy as np

from lowrank_extraction.models import FloatArray, LinearLowRankTarget
from lowrank_extraction.oracles import quantize_significant


# --- the defense -----------------------------------------------------------

def quantize(y: FloatArray, *, decimals: int | None, sig_digits: int | None) -> FloatArray:
    """The served score.  Exactly one of the two families is active."""

    if decimals is not None:
        return np.round(np.asarray(y, dtype=np.float64), int(decimals))
    if sig_digits is not None:
        return quantize_significant(np.asarray(y, dtype=np.float64), int(sig_digits))
    return np.asarray(y, dtype=np.float64)


def cell_boundary_near_one(*, decimals: int | None, sig_digits: int | None) -> float:
    """A value sitting exactly on a boundary between two quantizer cells.

    Any boundary works; the attack scales it to wherever it needs it.  Taking
    one near 1 keeps the scale factor well conditioned.
    """

    if decimals is not None:
        return 1.0 + 0.5 * 10.0 ** (-int(decimals))
    if sig_digits is not None:
        p = int(sig_digits)
        return (10.0 ** (p - 1) + 0.5) * 10.0 ** (-(p - 1))
    raise ValueError("a quantizer is required")


# --- the attacker's own resolution ----------------------------------------

def represent(s: float, input_digits: int | None) -> float:
    """What the attacker can actually ask for.

    An API accepts a number in some format.  ``input_digits`` is how many
    significant digits that format carries; ``None`` is the idealization of
    unbounded request precision.  This is the quantity the experiment shows to
    be load-bearing.
    """

    if input_digits is None:
        return float(s)
    return float(quantize_significant(np.array([float(s)]), int(input_digits))[0])


# --- attacker A: amplification + weighted least squares (the paper's) ------

def attack_amplify(
    target: LinearLowRankTarget,
    rng: np.random.Generator,
    *,
    design_size: int,
    amplify: float,
    decimals: int | None,
    sig_digits: int | None,
    input_digits: int | None,
) -> tuple[FloatArray, int]:
    """Corollary 1's lever: inflate the design, solve, project to rank.

    The bias is an unknown of the recovery problem, so it is carried as the
    augmented column of Theorem 1 and solved for alongside the head rather than
    subtracted from the observations.  The constant column is not amplified,
    which is exactly why Corollary 1(1) predicts amplification to help the head
    and not the intercept.
    """

    h = target.n_features
    design = rng.standard_normal((design_size, h))
    design = design / np.linalg.norm(design, axis=1, keepdims=True)
    design = amplify * design
    design = np.array([[represent(v, input_digits) for v in row] for row in design])

    served = serve(target, design, decimals=decimals, sig_digits=sig_digits)
    augmented = np.hstack([design, np.ones((design.shape[0], 1))])
    values = served

    if sig_digits is not None:
        # Heteroscedastic under a relative grid, so weight by the cell width --
        # the textbook response, and the stronger version of this attacker.
        scale = np.abs(served)
        width = np.where(scale > 0, scale, 1.0) * 10.0 ** (-(int(sig_digits) - 1))
        w = 1.0 / np.maximum(width, 1e-300)
        weight_est = np.empty((target.n_classes, h))
        for c in range(target.n_classes):
            sw = np.sqrt(w[:, c])[:, None]
            sol, *_ = np.linalg.lstsq(augmented * sw, values[:, c] * sw[:, 0], rcond=None)
            weight_est[c] = sol[:h]
    else:
        sol, *_ = np.linalg.lstsq(augmented, values, rcond=None)
        weight_est = sol[:h].T

    return weight_est, design_size


# --- attacker B: adaptive boundary seeking (the missing baseline) ----------

def attack_boundary(
    target: LinearLowRankTarget,
    *,
    bits: int,
    decimals: int | None,
    sig_digits: int | None,
    input_digits: int | None,
) -> tuple[FloatArray, int, float]:
    """Place a cell boundary at the midpoint of the interval; read the side.

    One scalar entry ``W[c, i]`` at a time, probing with ``x = s * e_i`` so the
    served score is ``s * W[c, i] + b_c``.  Queries are counted in full: no
    credit is taken for the other classes returned by the same call, which
    makes the reported cost an upper bound on what this attacker needs.

    Returns the mean bisection steps per coordinate--class pair alongside the
    estimate and the query count.  This attacker has its own early exits -- a
    zero midpoint, an unrepresentable scale, a collapsed bracket -- so that it
    spends its whole allowance is a measurement rather than a property of the
    loop, and Section~S10 quotes it as one.
    """

    m, h = target.n_classes, target.n_features
    boundary = cell_boundary_near_one(decimals=decimals, sig_digits=sig_digits)
    weight_est = np.zeros((m, h))
    queries = 0
    steps_taken = 0
    pairs = 0

    # One unamplified probe per coordinate brackets every class at once.
    for i in range(h):
        x = np.zeros(h)
        x[i] = 1.0
        served = serve(target, x, decimals=decimals, sig_digits=sig_digits)[0]
        queries += 1
        theta0 = served - target.bias

        for c in range(m):
            t0 = float(theta0[c])
            if t0 == 0.0:
                weight_est[c, i] = 0.0
                continue
            # A generous bracket around the single-query answer.  One probe
            # pins theta to the served precision, so this only has to cover
            # that; the bisection cost of a loose bracket is logarithmic.
            pad = max(abs(t0) * 0.05, 0.05)
            lo, hi = t0 - pad, t0 + pad
            bias_c = float(target.bias[c])
            # The served score is s*theta + b_c, so the scale that puts the
            # midpoint exactly on the boundary must account for the bias.
            thr = quantize(
                np.array([np.nextafter(boundary, -np.inf)]),
                decimals=decimals, sig_digits=sig_digits,
            )[0]

            pairs += 1
            for _ in range(bits):
                mid = 0.5 * (lo + hi)
                if mid == 0.0:
                    break
                s = represent((boundary - bias_c) / mid, input_digits)
                if not np.isfinite(s) or s == 0.0:
                    break
                x = np.zeros(h)
                x[i] = s
                y = serve(
                    target, x, decimals=decimals, sig_digits=sig_digits
                )[0][c]
                queries += 1
                # y > boundary  <=>  s*theta > s*mid, which is theta > mid only
                # when s > 0; a negative scale reverses the comparison.
                above_boundary = y > thr
                theta_above = above_boundary if s > 0 else not above_boundary
                if theta_above:
                    lo = mid
                else:
                    hi = mid
                steps_taken += 1
                if hi - lo <= 1e-18 * max(abs(mid), 1e-300):
                    break
            weight_est[c, i] = 0.5 * (lo + hi)

    return weight_est, queries, (steps_taken / pairs if pairs else 0.0)


# --- attacker B': boundary seeking without knowing the bias ---------------

def cell_boundary_in_decade(
    decade: int, *, decimals: int | None, sig_digits: int | None
) -> float:
    """A cell boundary of ``Q`` lying in ``[10**decade, 10**(decade+1))``.

    Attackers B and C place a *known* boundary at a *computed* score, which
    needs the bias.  Attacker B' instead uses two boundaries and never computes
    a score at all, so it needs two of them, far enough apart that the scales
    reaching them differ by much more than the attacker's own resolution.
    """

    if decimals is not None:
        return 10.0**decade + 0.5 * 10.0 ** (-int(decimals))
    if sig_digits is not None:
        p = int(sig_digits)
        return (10.0 ** (p - 1) + 0.5) * 10.0 ** (decade - (p - 1))
    raise ValueError("a quantizer is required")


def _above(served: float, beta: float, *, decimals: int | None, sig_digits: int | None) -> bool:
    """Did the true score lie above the cell boundary ``beta``?

    ``Q`` is monotone, so a served value at or above the representative of the
    cell immediately above ``beta`` reports exactly that.  This uses only the
    published serving format, never the secret.
    """

    step = abs(beta) * 1e-9 + 1e-15
    upper = float(quantize(np.array([beta + step]), decimals=decimals, sig_digits=sig_digits)[0])
    return bool(served >= upper)


def _locate_crossing(
    target: LinearLowRankTarget,
    *,
    i: int,
    c: int,
    beta: float,
    s_guess: float,
    bits: int,
    decimals: int | None,
    sig_digits: int | None,
    input_digits: int | None,
) -> tuple[float, int]:
    """Bisect in *scale* space for the ``s`` at which the score crosses ``beta``.

    The observable is one bit per query -- which side of a published cell
    boundary the served value fell on -- exactly as in attacker B.  What differs
    is the unknown being searched: B bisects the secret directly and needs the
    bias to aim, while this bisects the scale, which the attacker itself chose
    and therefore knows exactly.
    """

    h = target.n_features
    queries = 0

    def probe(s: float) -> bool:
        nonlocal queries
        x = np.zeros(h)
        x[i] = s
        y = serve(target, x, decimals=decimals, sig_digits=sig_digits)[0][c]
        queries += 1
        return _above(float(y), beta, decimals=decimals, sig_digits=sig_digits)

    # Bracket the crossing.  The one-query estimate is good to the served
    # precision, so a few doublings always suffice; each is charged in full.
    lo, hi = s_guess * 0.8, s_guess * 1.2
    a = probe(represent(lo, input_digits))
    b = probe(represent(hi, input_digits))
    widen = 0
    while a == b and widen < 40:
        mid = 0.5 * (lo + hi)
        half = (hi - lo)
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
            break  # the request format can no longer split the bracket
        if probe(mid) == a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), queries


def attack_boundary_unknown_bias(
    target: LinearLowRankTarget,
    *,
    bits: int,
    decimals: int | None,
    sig_digits: int | None,
    input_digits: int | None,
) -> tuple[FloatArray, FloatArray, int]:
    """Boundary seeking under the paper's actual threat model.

    Attackers B and C compute the scale ``(b* - b_c)/mid`` that lands a chosen
    boundary on a chosen score, which requires the deployed bias ``b_c`` -- an
    unknown of the recovery problem, not something the interface hands out.
    This attacker never needs it.  Along ``x = s e_i`` the served score is the
    affine map ``y(s) = s*theta + b_c``, so locating the scales ``s_1, s_2`` at
    which ``y`` crosses two *known* boundaries ``beta_1, beta_2`` gives

        theta = (beta_1 - beta_2) / (s_1 - s_2),    b_c = beta_1 - theta * s_1,

    recovering the intercept as a byproduct of the same queries.  The two boundaries
    are put a decade apart so that ``s_1 - s_2`` is large next to the attacker's
    own rounding of ``s``; the accuracy is therefore still set by the request
    resolution and not by the served precision, which is the paper's claim.
    """

    m, h = target.n_classes, target.n_features
    beta1 = cell_boundary_in_decade(0, decimals=decimals, sig_digits=sig_digits)
    beta2 = cell_boundary_in_decade(1, decimals=decimals, sig_digits=sig_digits)
    weight_est = np.zeros((m, h))
    bias_acc: list[list[float]] = [[] for _ in range(m)]
    queries = 0

    # Two unamplified probes bracket every class at once: x = 0 reads the bias
    # to the served precision, x = e_i reads theta + bias.  Neither is the exact
    # value; they only aim the search.
    served_zero = serve(
        target, np.zeros(h), decimals=decimals, sig_digits=sig_digits
    )[0]
    queries += 1

    for i in range(h):
        x = np.zeros(h)
        x[i] = 1.0
        served_one = serve(
            target, x, decimals=decimals, sig_digits=sig_digits
        )[0]
        queries += 1

        for c in range(m):
            b_hat = float(served_zero[c])
            t_hat = float(served_one[c]) - b_hat
            if t_hat == 0.0:
                continue
            crossings = []
            for beta in (beta1, beta2):
                s, q = _locate_crossing(
                    target, i=i, c=c, beta=beta, s_guess=(beta - b_hat) / t_hat,
                    bits=bits, decimals=decimals, sig_digits=sig_digits,
                    input_digits=input_digits,
                )
                queries += q
                crossings.append(s)
            s1, s2 = crossings
            if not (np.isfinite(s1) and np.isfinite(s2)) or s1 == s2:
                weight_est[c, i] = t_hat  # fall back on the single-query answer
                continue
            theta = (beta1 - beta2) / (s1 - s2)
            weight_est[c, i] = theta
            # Each coordinate yields the shared intercept independently, so the
            # median over coordinates is a free byproduct of the same queries.
            bias_acc[c].append(beta1 - theta * s1)

    bias_est = np.array([
        float(np.median(v)) if v else float(served_zero[c])
        for c, v in enumerate(bias_acc)
    ])
    return weight_est, bias_est, queries


# --- attacker C: joint (boundary, scale) search ----------------------------

def snap_to_boundary(
    v: FloatArray, *, decimals: int | None, sig_digits: int | None
) -> FloatArray:
    """The quantizer cell boundary nearest to each value of ``v``.

    Attacker B uses a single boundary and scales it into place, so its reach is
    the spacing of one scaled subgrid.  A quantizer offers far more than one:
    an absolute grid has a boundary every ``10**-decimals`` all the way out to
    whatever magnitude the attacker can provoke, and a relative grid has
    ``10**sig_digits`` of them in every decade.  This function is what lets an
    attacker use all of them.
    """

    v = np.asarray(v, dtype=np.float64)
    sign, mag = np.sign(v), np.abs(v)
    if decimals is not None:
        step = 10.0 ** (-int(decimals))
        return sign * ((np.floor(mag / step - 0.5) + 0.5) * step)
    if sig_digits is None:
        raise ValueError("a quantizer is required")

    # Relative cells are a decade at a time: inside the decade [10^e, 10^{e+1})
    # the boundaries are (m + 1/2)*ulp for mantissas m in [10^{p-1}, 10^p), and
    # the decades are joined by one further boundary just below 10^e.  Snapping
    # with the decade's own ulp can push the mantissa below 10^{p-1}, which
    # lands on a *representable value* of the decade underneath rather than on
    # a boundary, so the candidates are built explicitly and the nearest is
    # taken.
    p = int(sig_digits)
    with np.errstate(divide="ignore"):
        exponent = np.where(mag > 0, np.floor(np.log10(np.maximum(mag, 1e-300))), 0.0)
    ulp = 10.0 ** (exponent - (p - 1))
    m = np.floor(mag / ulp - 0.5)
    lo_m = float(10 ** (p - 1))
    hi_m = float(10**p - 1)
    candidates = [
        (np.clip(m, lo_m, hi_m) + 0.5) * ulp,
        (np.clip(m + 1.0, lo_m, hi_m) + 0.5) * ulp,
        10.0**exponent - 0.5 * 10.0 ** (exponent - p),          # decade join below
        10.0 ** (exponent + 1.0) - 0.5 * 10.0 ** (exponent - p + 1.0),  # and above
    ]
    stacked = np.stack(np.broadcast_arrays(*candidates))
    best = np.argmin(np.abs(stacked - mag[None, ...]), axis=0)
    return sign * np.take_along_axis(stacked, best[None, ...], axis=0)[0]


def scale_candidates(
    input_digits: int | None, *, scale_cap: float, max_candidates: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Every positive scale factor the attacker's request format can express.

    ``input_digits`` significant digits over the exponents that stay under
    ``scale_cap``.  For wide formats the full set is enormous, so we sample it:
    any scale drawn is one the attacker could really send, so a sampled search
    still yields an achievable probe and therefore a valid *upper* bound on the
    floor.  It can only understate the attacker.
    """

    if input_digits is None:
        # Unbounded request precision: the attacker can name any real, so the
        # nearest boundary is reachable exactly and the search is trivial.
        return np.array([1.0])
    d = int(input_digits)
    exponents = np.arange(0, int(np.floor(np.log10(scale_cap))) + 1, dtype=np.float64)
    lo, hi = 10 ** (d - 1), 10**d
    per_exponent = max(1, max_candidates // max(1, exponents.size))
    if hi - lo <= per_exponent:
        mantissas = np.arange(lo, hi, dtype=np.float64)
    else:
        # The full set is astronomically large for a wide request format, so it
        # is sampled rather than built.  Sampling can only lose the attacker
        # candidates, never invent unsendable ones.
        mantissas = np.unique(
            rng.integers(lo, hi, size=per_exponent, dtype=np.int64)
        ).astype(np.float64)
    scales = (mantissas[None, :] * 10.0 ** (exponents[:, None] - (d - 1))).ravel()
    scales = scales[(scales > 0.0) & (scales <= scale_cap)]
    return np.sort(scales)


def attack_boundary_joint(
    target: LinearLowRankTarget,
    rng: np.random.Generator,
    *,
    bits: int,
    decimals: int | None,
    sig_digits: int | None,
    input_digits: int | None,
    scale_cap: float,
    max_candidates: int,
    bias: FloatArray,
) -> tuple[FloatArray, int, float]:
    """Attacker B, but choosing the boundary and the scale together.

    To split its bracket at ``mid`` the attacker needs a served score to sit on
    a cell boundary when ``theta = mid``, i.e. it needs ``b = b_c + s*mid`` to
    be a boundary.  For each admissible scale ``s`` it snaps that ideal value to
    the nearest real boundary and gets the threshold ``t = (b - b_c)/s`` it can
    actually test.  Taking the best ``s`` makes the reachable thresholds the
    whole set ``{(b - b_c)/s}`` rather than one scaled copy of a single ``b``,
    and the gap of that set -- not the attacker's own rounding -- is what stops
    the bisection.

    Returns the estimate, the query count, and the mean number of bisection
    steps actually taken per coordinate--class pair.  That mean is the evidence
    that the search halts on the grid rather than on its allowance: a value
    below ``bits`` means every pair ran out of reachable thresholds first.
    """

    m, h = target.n_classes, target.n_features
    weight_est = np.zeros((m, h))
    queries = 0
    steps_taken = 0
    pairs = 0
    scales = scale_candidates(
        input_digits, scale_cap=scale_cap, max_candidates=max_candidates, rng=rng
    )

    for i in range(h):
        x = np.zeros(h)
        x[i] = 1.0
        served = serve(target, x, decimals=decimals, sig_digits=sig_digits)[0]
        queries += 1
        theta0 = served - bias

        for c in range(m):
            t0 = float(theta0[c])
            if t0 == 0.0:
                continue
            pad = max(abs(t0) * 0.05, 0.05)
            lo, hi = t0 - pad, t0 + pad
            bias_c = float(bias[c])

            pairs += 1
            for _ in range(bits):
                mid = 0.5 * (lo + hi)
                if input_digits is None:
                    # Unbounded request precision: the attacker names the scale
                    # that puts a boundary exactly at the midpoint, so the
                    # search has nothing to trade off and the bisection is
                    # exact.  The boundary is used as snapped -- re-snapping a
                    # value that already sits exactly on one can move it a whole
                    # cell, since floor() of a rounded integer can drop by one.
                    boundary = snap_to_boundary(
                        np.array([bias_c + mid]),
                        decimals=decimals, sig_digits=sig_digits,
                    )
                    scales = (boundary - bias_c) / mid
                else:
                    boundary = snap_to_boundary(
                        bias_c + scales * mid,
                        decimals=decimals, sig_digits=sig_digits,
                    )
                thresholds = (boundary - bias_c) / scales
                # Only act on a value that really does separate two cells.  The
                # attacker can check this offline, knowing the served format but
                # nothing about the secret, and a candidate that fails is one
                # whose answer would not report the sign it is meant to.
                step = np.abs(boundary) * 1e-9 + 1e-15
                separates = quantize(
                    boundary - step, decimals=decimals, sig_digits=sig_digits
                ) != quantize(
                    boundary + step, decimals=decimals, sig_digits=sig_digits
                )
                inside = (thresholds > lo) & (thresholds < hi) & separates
                if not inside.any():
                    break
                steps_taken += 1
                j = int(np.argmin(np.abs(thresholds[inside] - mid)))
                s = float(scales[inside][j])
                t = float(thresholds[inside][j])
                b = float(boundary[inside][j])

                x = np.zeros(h)
                x[i] = s
                y = serve(
                    target, x, decimals=decimals, sig_digits=sig_digits
                )[0][c]
                queries += 1
                # b is a cell boundary, so the served representative falls
                # strictly to one side of it.  Which side means theta > t
                # depends on the sign of the scale, exactly as for attacker B:
                # a negative scale reverses the comparison.
                above = y > b
                if s < 0:
                    above = not above
                if above:
                    lo = t
                else:
                    hi = t
            weight_est[c, i] = 0.5 * (lo + hi)

    return weight_est, queries, (steps_taken / pairs if pairs else 0.0)


# --- a server that is not noiseless ----------------------------------------


class NoisyServer:
    """A deployment that perturbs the score before it quantizes it.

    Every measurement elsewhere in this file assumes the quantizer is the only
    thing between the attacker and the true logit.  That is the strongest
    assumption the boundary results rest on, because boundary seeking works by
    bisecting on *which cell* a score landed in: if that answer is randomized
    near the cell wall, the bisection has nothing stable to converge to.

    Order of operations is the whole point here and is easy to get wrong.  The
    server adds its own noise and then rounds, so one answer is
    ``Q(y + eps)``; an attacker who repeats a query sees ``R`` such answers and
    can only average what it was *sent*, giving ``mean_r Q(y + eps_r)``.
    Averaging the noise before quantizing instead would be
    ``Q(y + mean_r eps_r)`` -- a different and strictly easier oracle, equal to
    a single query against a server of width ``sigma/sqrt(R)``, which assumes
    the very ``sqrt(R)`` gain the experiment is supposed to measure.  ``Q`` is
    nonlinear, so the two do not commute near the cell wall being attacked.

    Every method other than :meth:`serve` delegates to the real target, so
    error is still scored against the true secret.
    """

    def __init__(self, target: LinearLowRankTarget, rng: np.random.Generator,
                 *, sigma: float, repeats: int = 1) -> None:
        self._target = target
        self._rng = rng
        self.sigma = float(sigma)
        self.repeats = int(repeats)
        self.calls = 0

    def __getattr__(self, name: str):  # bias, weight, rank, delta, ...
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
        # Summed exactly, per component.  The attacker's comparison at the wall
        # is ``y > b`` against an exact boundary, and a cell representative can
        # be exactly ``b``.  Accumulating in floating point visits partial sums
        # ``k*v`` that are not representable (``3*v`` already wants 55 mantissa
        # bits), so the mean lands a few ulp *below* ``v`` and flips that
        # comparison -- an arithmetic artefact of how the mean is taken, not an
        # effect of the noise, and one that does not vanish as ``sigma -> 0``.
        # ``np.mean`` does not fix it: numpy only sums pairwise along a
        # contiguous axis, and reducing over axis 0 here walks strided memory
        # and so accumulates sequentially.  ``fsum`` is correctly rounded, which
        # makes the mean of equal draws exact and the ``sigma -> 0`` limit
        # reduce to a single query, as it must.
        flat = draws.reshape(self.repeats, -1)
        mean = np.array(
            [fsum(flat[:, j]) / self.repeats for j in range(flat.shape[1])],
            dtype=np.float64,
        )
        return mean.reshape(np.shape(clean))


def serve(target, x: FloatArray, *, decimals: int | None,
          sig_digits: int | None) -> FloatArray:
    """One query to the deployment.

    The server is what applies the quantizer, so this is the only place a
    served score is produced.  Against an ordinary target it is exactly
    ``quantize(target.logits(x))``, which keeps every noiseless measurement in
    this file bit for bit what it was.
    """

    if isinstance(target, NoisyServer):
        return target.serve(x, decimals=decimals, sig_digits=sig_digits)
    return quantize(target.logits(x), decimals=decimals, sig_digits=sig_digits)


# --- driver ----------------------------------------------------------------

@dataclass
class Row:
    seed: int
    defense: str
    attacker: str
    input_digits: str
    amplification: str
    queries: int
    delta_relerr: float
    scale_cap: str = "-"
    mean_bisection_steps: float | None = None
    noise_sigma: float = 0.0
    repeats: int = 1


def make_target(rng: np.random.Generator, *, m: int, h: int, r: int) -> LinearLowRankTarget:
    base = rng.standard_normal((m, h)) / np.sqrt(h)
    left = rng.standard_normal((m, r)) / np.sqrt(r)
    right = rng.standard_normal((h, r)) / np.sqrt(h)
    bias = rng.standard_normal(m) * 0.1
    return LinearLowRankTarget(base_weight=base, adapter_left=left, adapter_right=right, bias=bias)


def relerr(target: LinearLowRankTarget, weight_est: FloatArray) -> float:
    delta_hat = np.asarray(weight_est) - target.base_weight
    u, s, vt = np.linalg.svd(delta_hat, full_matrices=False)
    r = target.rank
    delta_hat = (u[:, :r] * s[:r]) @ vt[:r, :]
    d = target.delta
    return float(np.linalg.norm(delta_hat - d) / np.linalg.norm(d))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("results") / "adaptive_boundary")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--feature-dim", type=int, default=8)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--sig-digits", type=int, default=4)
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--bits", type=int, default=60)
    ap.add_argument("--design-size", type=int, default=200)
    ap.add_argument("--amplifications", type=float, nargs="+", default=[1.0, 64.0, 1024.0])
    ap.add_argument(
        "--input-digits", nargs="+", default=["4", "6", "8", "12", "16", "none"],
        help="significant digits the attacker's own query carries; 'none' is unbounded",
    )
    ap.add_argument(
        "--scale-caps", type=float, nargs="+", default=[1e2, 1e4, 1e6],
        help="largest query scales the joint-search attacker may send; this is the "
             "query-norm cap, and it bounds how many absolute-grid boundaries it "
             "can reach (a relative grid is unaffected by it)",
    )
    ap.add_argument(
        "--known-bias-control", action="store_true",
        help="additionally run the joint search with the deployed bias handed "
             "to it. The threat model forbids that, so this is never the "
             "reported attacker; it is the control Section S11 quotes to say "
             "what recovering b_c first costs in error and in bisection steps.",
    )
    ap.add_argument(
        "--noise-sigma", type=float, nargs="+", default=[],
        help="standard deviations of Gaussian noise the server adds to a score "
             "before quantizing it. Passing this adds a second, noise-only "
             "sweep of the boundary attackers; the noiseless rows above are "
             "unchanged, so a run without the flag reproduces bit for bit.",
    )
    ap.add_argument(
        "--noise-repeats", type=int, nargs="+", default=[1],
        help="how many independent answers the attacker averages per query "
             "under --noise-sigma. Averaging cuts the noise by sqrt of this and "
             "costs this factor in queries, so it is the attacker's reply to a "
             "stochastic defense and is charged for accordingly.",
    )
    ap.add_argument(
        "--max-scale-candidates", type=int, default=200_000,
        help="cap on the scale factors searched per step; sampling below the full "
             "set can only weaken the attacker, so reported floors stay valid",
    )
    args = ap.parse_args()

    defenses = [
        ("absolute (decimals=%d)" % args.decimals, {"decimals": args.decimals, "sig_digits": None}),
        ("relative (sig=%d)" % args.sig_digits, {"decimals": None, "sig_digits": args.sig_digits}),
    ]
    rows: list[Row] = []

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        target = make_target(rng, m=args.classes, h=args.feature_dim, r=args.rank)

        # The boundary attacker's cost is fixed by the problem size; give the
        # amplification baseline exactly that budget too, so the comparison is
        # query-controlled rather than flattering the new attacker.
        matched = args.feature_dim + args.feature_dim * args.classes * args.bits

        for dname, dkw in defenses:
            for amp in args.amplifications:
                for size, tag in ((args.design_size, "small"), (matched, "matched")):
                    w, q = attack_amplify(
                        target, np.random.default_rng(seed + 9973),
                        design_size=size, amplify=amp,
                        input_digits=None, **dkw,
                    )
                    rows.append(
                        Row(seed, dname, f"amplify+wls ({tag} budget)", "none",
                            f"{amp:g}", q, relerr(target, w))
                    )

            for idig in args.input_digits:
                d = None if idig == "none" else int(idig)
                w, q, b_steps = attack_boundary(
                    target, bits=args.bits, input_digits=d, **dkw
                )
                rows.append(
                    Row(seed, dname, "boundary", idig, "-", q, relerr(target, w),
                        mean_bisection_steps=b_steps)
                )

                # The same search without the deployed bias, which the threat
                # model does not grant.  This is the attacker the paper reports.
                w, bias_hat, q_ub = attack_boundary_unknown_bias(
                    target, bits=args.bits, input_digits=d, **dkw
                )
                rows.append(
                    Row(seed, dname, "boundary (unknown bias)", idig, "-", q_ub,
                        relerr(target, w))
                )

                # The joint attacker's reach into an absolute grid is set by how
                # large a score it may provoke, so the norm cap is swept: it is
                # the defender's other control, and the two families answer to
                # it very differently.  It runs as a second stage on the
                # intercept the unknown-bias search already recovered, and is
                # charged for those queries as well; an error e in that estimate
                # displaces the probed threshold by only e/s, which the large
                # scales this stage relies on drive below the floor it seeks.
                for cap in args.scale_caps:
                    w, q, steps = attack_boundary_joint(
                        target, np.random.default_rng(seed + 7717),
                        bits=args.bits, input_digits=d,
                        scale_cap=cap,
                        max_candidates=args.max_scale_candidates,
                        bias=bias_hat,
                        **dkw,
                    )
                    rows.append(
                        Row(seed, dname, "boundary (joint search)", idig, "-",
                            q + q_ub, relerr(target, w), f"{cap:g}", steps)
                    )
                    # The same search handed the deployed bias.  The threat
                    # model does not permit this, so it is never the reported
                    # attacker; it is the control that says how much of the
                    # error above is the price of recovering b_c first, and
                    # whether that price shows up in the step counts.
                    if args.known_bias_control:
                        w_kb, q_kb, steps_kb = attack_boundary_joint(
                            target, np.random.default_rng(seed + 7717),
                            bits=args.bits, input_digits=d,
                            scale_cap=cap,
                            max_candidates=args.max_scale_candidates,
                            bias=target.bias,
                            **dkw,
                        )
                        rows.append(
                            Row(seed, dname, "boundary (joint, bias given)", idig,
                                "-", q_kb, relerr(target, w_kb), f"{cap:g}", steps_kb)
                        )
                    if d is None:
                        break  # the cap is irrelevant when scales are exact

    # --- the noise sweep -----------------------------------------------
    # Boundary seeking is the one result that leans on the quantizer being the
    # only thing in the way, so it is the one swept here.  The attacker is
    # unchanged: it is not told sigma and does not model it.
    for sigma in args.noise_sigma:
        for reps in args.noise_repeats:
            for seed in args.seeds:
                target = make_target(
                    np.random.default_rng(seed),
                    m=args.classes, h=args.feature_dim, r=args.rank,
                )
                for dname, dkw in defenses:
                    for idig in args.input_digits:
                        d = None if idig == "none" else int(idig)
                        server = NoisyServer(
                            target, np.random.default_rng(seed + 4242),
                            sigma=sigma, repeats=reps,
                        )
                        w, _, b_steps = attack_boundary(
                            server, bits=args.bits, input_digits=d, **dkw
                        )
                        rows.append(
                            Row(seed, dname, "boundary (noisy server)", idig, "-",
                                server.calls, relerr(target, w),
                                mean_bisection_steps=b_steps,
                                noise_sigma=sigma, repeats=reps)
                        )

                        server = NoisyServer(
                            target, np.random.default_rng(seed + 4242),
                            sigma=sigma, repeats=reps,
                        )
                        w, _, _ = attack_boundary_unknown_bias(
                            server, bits=args.bits, input_digits=d, **dkw
                        )
                        rows.append(
                            Row(seed, dname, "boundary (noisy, unknown bias)",
                                idig, "-", server.calls, relerr(target, w),
                                noise_sigma=sigma, repeats=reps)
                        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "classes": args.classes, "feature_dim": args.feature_dim, "rank": args.rank,
            "sig_digits": args.sig_digits, "decimals": args.decimals, "bits": args.bits,
            "design_size": args.design_size, "seeds": args.seeds,
            "scale_caps": args.scale_caps,
            "max_scale_candidates": args.max_scale_candidates,
            "noise_sigma": args.noise_sigma,
            "noise_repeats": args.noise_repeats,
        },
        "rows": [asdict(r) for r in rows],
    }
    (args.outdir / "adaptive_boundary.json").write_text(json.dumps(payload, indent=2))

    # summary
    import collections
    agg: dict[tuple, list[float]] = collections.defaultdict(list)
    qs: dict[tuple, list[int]] = collections.defaultdict(list)
    for r in rows:
        key = (r.defense, r.attacker, r.input_digits, r.amplification, r.scale_cap)
        agg[key].append(r.delta_relerr)
        qs[key].append(r.queries)
    lines = [f"seeds: {len(args.seeds)}", "",
             "| defense | attacker | input digits | A | scale cap | queries | median rel. err |",
             "|---|---|---|---|---|---|---|"]
    for key in sorted(agg, key=lambda k: (k[0], k[1], str(k[2]), str(k[3]), str(k[4]))):
        d, a, idig, amp, cap = key
        v = float(np.median(agg[key]))
        lines.append(
            f"| {d} | {a} | {idig} | {amp} | {cap} | "
            f"{int(np.median(qs[key]))} | {v:.3e} |"
        )
    (args.outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
