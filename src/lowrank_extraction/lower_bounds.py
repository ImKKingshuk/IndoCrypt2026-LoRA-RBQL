from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import FloatArray, LinearLowRankTarget
from .oracles import TopKOracle


@dataclass(frozen=True)
class PartialTopKIndistinguishabilityResult:
    k: int
    n_features: int
    n_classes: int
    hidden_class: int
    query_count: int
    domain_radius: float
    transcript_equal: bool
    parameter_distance: float
    delta_rank_a: int
    delta_rank_b: int
    min_hidden_gap_on_queries: float
    certified_domain_gap: float


def _bounded_queries(rng: np.random.Generator, n_queries: int, n_features: int, radius: float) -> FloatArray:
    x = rng.normal(0.0, 1.0, size=(n_queries, n_features))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    scales = rng.random((n_queries, 1)) ** (1.0 / n_features)
    return radius * scales * x / norms


def make_partial_topk_indistinguishable_pair(
    *,
    n_features: int = 8,
    n_classes: int = 4,
    k: int = 2,
    n_queries: int = 512,
    domain_radius: float = 1.0,
    seed: int = 0,
) -> tuple[LinearLowRankTarget, LinearLowRankTarget, FloatArray, PartialTopKIndistinguishabilityResult]:
    """Construct two rank-one adapters with identical partial top-k transcripts.

    The construction is deliberately simple: the first k public classes have
    large fixed biases, while one hidden class has a low bias. Two different
    rank-one row adapters change only the hidden class. As long as the hidden
    score stays below the kth returned score on the bounded query domain, a
    top-k index-only oracle never reveals that hidden row, so the transcript is
    identical even though the adapter parameters differ.
    """

    if not (1 <= k < n_classes):
        raise ValueError("k must be a proper partial top-k value")
    if n_features <= 0 or n_queries <= 0 or domain_radius <= 0:
        raise ValueError("dimensions, query count, and radius must be positive")

    rng = np.random.default_rng(seed)
    hidden_class = n_classes - 1
    base_weight = np.zeros((n_classes, n_features), dtype=np.float64)
    bias = np.full(n_classes, -4.0, dtype=np.float64)
    for klass in range(k):
        bias[klass] = 4.0 - klass

    direction_a = rng.normal(0.0, 1.0, size=n_features)
    direction_a /= np.linalg.norm(direction_a)
    direction_b = rng.normal(0.0, 1.0, size=n_features)
    direction_b -= direction_a * float(direction_a @ direction_b)
    direction_b /= np.linalg.norm(direction_b)
    adapter_scale = 0.25

    left = np.zeros((n_classes, 1), dtype=np.float64)
    left[hidden_class, 0] = 1.0
    target_a = LinearLowRankTarget(
        base_weight=base_weight,
        adapter_left=left,
        adapter_right=(adapter_scale * direction_a).reshape(n_features, 1),
        bias=bias,
    )
    target_b = LinearLowRankTarget(
        base_weight=base_weight,
        adapter_left=left,
        adapter_right=(adapter_scale * direction_b).reshape(n_features, 1),
        bias=bias,
    )

    queries = _bounded_queries(rng, n_queries, n_features, domain_radius)
    oracle_a = TopKOracle(target_a, k=k, include_scores=False)
    oracle_b = TopKOracle(target_b, k=k, include_scores=False)
    transcript_a, _ = oracle_a.query(queries)
    transcript_b, _ = oracle_b.query(queries)

    logits_a = target_a.logits(queries)
    kth_scores = np.take_along_axis(logits_a, transcript_a[:, [-1]], axis=1).reshape(-1)
    hidden_scores = logits_a[:, hidden_class]
    min_hidden_gap = float(np.min(kth_scores - hidden_scores))

    kth_public_bias = bias[k - 1]
    hidden_bias = bias[hidden_class]
    certified_domain_gap = float(kth_public_bias - hidden_bias - adapter_scale * domain_radius)

    result = PartialTopKIndistinguishabilityResult(
        k=k,
        n_features=n_features,
        n_classes=n_classes,
        hidden_class=hidden_class,
        query_count=oracle_a.query_count + oracle_b.query_count,
        domain_radius=domain_radius,
        transcript_equal=bool(np.array_equal(transcript_a, transcript_b)),
        parameter_distance=float(np.linalg.norm(target_a.delta - target_b.delta)),
        delta_rank_a=int(np.linalg.matrix_rank(target_a.delta)),
        delta_rank_b=int(np.linalg.matrix_rank(target_b.delta)),
        min_hidden_gap_on_queries=min_hidden_gap,
        certified_domain_gap=certified_domain_gap,
    )
    return target_a, target_b, queries, result
