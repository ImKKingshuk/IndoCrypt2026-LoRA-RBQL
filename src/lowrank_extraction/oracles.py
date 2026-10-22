from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import FloatArray, LinearLowRankTarget


def quantize_significant(y: FloatArray, digits: int) -> FloatArray:
    """Round to a fixed number of significant digits (relative quantization).

    Unlike rounding to an absolute grid, the induced perturbation scales with
    the magnitude of the score: ``|Q(y) - y| <= 5 * 10^{-digits} * |y|``.  This
    is what makes the defense scale-invariant, since amplifying a query
    inflates the signal and the perturbation by the same factor.
    """

    y = np.asarray(y, dtype=np.float64)
    out = np.array(y, copy=True)
    nz = y != 0.0
    if not np.any(nz):
        return out
    exponent = np.floor(np.log10(np.abs(y[nz])))
    factor = np.power(10.0, (digits - 1) - exponent)
    scaled = y[nz] * factor
    # Ties resolve away from zero, not to even.  This is not cosmetic.  The tie
    # rule decides whether a quantization cell is half-open or closed, and the
    # uniform floor of Theorem 7 needs it half-open.  Under numpy's default
    # ties-to-even a single achievable threshold can be resolved in opposite
    # directions by two admissible scales -- with p = d = 1 and theta = 1.5 the
    # scales 1 and 3 give Q(1.5) = 2 and Q(4.5) = 4, whose cells intersect in
    # the single point 1.5 -- which collapses the consistent set to a point and
    # makes the floor zero.  Away-from-zero keeps Q_p odd, so the reflection
    # argument for negative secrets is unaffected.
    out[nz] = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5) / factor
    return out


def adversarial_perturbation(y: FloatArray, eps: float) -> FloatArray:
    """A deterministic, absolute, worst-case-norm perturbation of the scores.

    This is the mechanism class that utility-constrained output-perturbation
    defenses occupy: the returned vector is displaced by a bounded amount chosen
    to be unhelpful rather than merely rounded off.  It is strictly stronger than
    rounding at the same budget -- rounding lands anywhere inside its cell, while
    this always spends the whole budget -- and it is deterministic in the query,
    so repetition cannot average it away either.

    It is a stand-in for the *class*, not a reimplementation of any particular
    published system.  The point of testing it is that Theorems 4 and 6 bound the
    perturbation only through ``||E||_F``, so they predict amplification breaches
    every absolute-magnitude defense, not just the grid we happened to analyse.
    """

    y = np.asarray(y, dtype=np.float64)
    # Deterministic in the returned scores, and deliberately not aligned with any
    # direction the attacker's estimator would find easy to cancel.
    direction = np.sin(1_000.0 * y) + np.cos(317.0 * y)
    norms = np.linalg.norm(direction, axis=-1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return eps * direction / norms


@dataclass
class RawLogitOracle:
    target: LinearLowRankTarget
    decimals: int | None = None
    sig_digits: int | None = None
    noise_std: float = 0.0
    rel_noise_std: float = 0.0
    adversarial_eps: float = 0.0
    seed: int = 0
    query_count: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def query(self, x: FloatArray) -> FloatArray:
        x2 = np.atleast_2d(np.asarray(x, dtype=np.float64))
        self.query_count += int(x2.shape[0])
        y = self.target.logits(x2)
        if self.adversarial_eps > 0:
            y = y + adversarial_perturbation(y, self.adversarial_eps)
        if self.noise_std > 0:
            y = y + self._rng.normal(0.0, self.noise_std, size=y.shape)
        if self.rel_noise_std > 0:
            # Multiplicative (relative) noise: scale-invariant like significant-digit
            # quantization, but stochastic, so repeated queries average it away.
            y = y * (1.0 + self._rng.normal(0.0, self.rel_noise_std, size=y.shape))
        if self.decimals is not None:
            y = np.round(y, self.decimals)
        if self.sig_digits is not None:
            y = quantize_significant(y, self.sig_digits)
        return y


@dataclass
class TopKOracle:
    target: LinearLowRankTarget
    k: int
    include_scores: bool = True
    decimals: int | None = None
    sig_digits: int | None = None
    noise_std: float = 0.0
    seed: int = 0
    query_count: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def query(self, x: FloatArray) -> tuple[NDArray[np.int64], FloatArray | None]:
        if not (1 <= self.k <= self.target.n_classes):
            raise ValueError("k must be between 1 and the number of classes")
        x2 = np.atleast_2d(np.asarray(x, dtype=np.float64))
        self.query_count += int(x2.shape[0])
        logits = self.target.logits(x2)
        if self.noise_std > 0:
            logits = logits + self._rng.normal(0.0, self.noise_std, size=logits.shape)
        indices = np.argsort(-logits, axis=1)[:, : self.k]
        if not self.include_scores:
            return indices.astype(np.int64), None
        scores = np.take_along_axis(logits, indices, axis=1)
        if self.decimals is not None:
            scores = np.round(scores, self.decimals)
        if self.sig_digits is not None:
            scores = quantize_significant(scores, self.sig_digits)
        return indices.astype(np.int64), scores


@dataclass
class HardLabelOracle:
    target: LinearLowRankTarget
    query_count: int = 0

    def query(self, x: FloatArray) -> NDArray[np.int64]:
        x2 = np.atleast_2d(np.asarray(x, dtype=np.float64))
        self.query_count += int(x2.shape[0])
        return self.target.labels(x2)
