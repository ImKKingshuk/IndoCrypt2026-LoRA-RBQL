from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LinearLowRankTarget:
    """Known-base affine classifier with an unknown low-rank additive update."""

    base_weight: FloatArray
    adapter_left: FloatArray
    adapter_right: FloatArray
    bias: FloatArray

    @property
    def delta(self) -> FloatArray:
        return self.adapter_left @ self.adapter_right.T

    @property
    def weight(self) -> FloatArray:
        return self.base_weight + self.delta

    @property
    def n_classes(self) -> int:
        return int(self.base_weight.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.base_weight.shape[1])

    @property
    def rank(self) -> int:
        return int(self.adapter_left.shape[1])

    def logits(self, x: FloatArray) -> FloatArray:
        x2 = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return x2 @ self.weight.T + self.bias

    def labels(self, x: FloatArray) -> NDArray[np.int64]:
        return np.argmax(self.logits(x), axis=1)


@dataclass(frozen=True)
class ReLUFeatureLowRankTarget:
    """Public ReLU feature extractor with an unknown low-rank adapted head."""

    feature_weight: FloatArray
    feature_bias: FloatArray
    base_head: FloatArray
    adapter_left: FloatArray
    adapter_right: FloatArray
    bias: FloatArray

    @property
    def delta(self) -> FloatArray:
        return self.adapter_left @ self.adapter_right.T

    @property
    def head(self) -> FloatArray:
        return self.base_head + self.delta

    @property
    def n_classes(self) -> int:
        return int(self.base_head.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.feature_weight.shape[1])

    @property
    def hidden_features(self) -> int:
        return int(self.base_head.shape[1])

    @property
    def rank(self) -> int:
        return int(self.adapter_left.shape[1])

    def features(self, x: FloatArray) -> FloatArray:
        x2 = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return np.maximum(0.0, x2 @ self.feature_weight.T + self.feature_bias)

    def logits(self, x: FloatArray) -> FloatArray:
        return self.features(x) @ self.head.T + self.bias

    def labels(self, x: FloatArray) -> NDArray[np.int64]:
        return np.argmax(self.logits(x), axis=1)


@dataclass(frozen=True)
class HiddenLoRABlockTarget:
    """One-hidden-layer network with a low-rank adapter inside the hidden block."""

    base_hidden_weight: FloatArray
    adapter_left: FloatArray
    adapter_right: FloatArray
    hidden_bias: FloatArray
    output_head: FloatArray
    output_bias: FloatArray

    @property
    def delta(self) -> FloatArray:
        return self.adapter_left @ self.adapter_right.T

    @property
    def hidden_weight(self) -> FloatArray:
        return self.base_hidden_weight + self.delta

    @property
    def n_features(self) -> int:
        return int(self.base_hidden_weight.shape[1])

    @property
    def hidden_features(self) -> int:
        return int(self.base_hidden_weight.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.output_head.shape[0])

    @property
    def rank(self) -> int:
        return int(self.adapter_left.shape[1])

    def hidden_preactivations(self, x: FloatArray) -> FloatArray:
        x2 = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return x2 @ self.hidden_weight.T + self.hidden_bias

    def hidden_features_from_input(self, x: FloatArray) -> FloatArray:
        return np.maximum(0.0, self.hidden_preactivations(x))

    def logits(self, x: FloatArray) -> FloatArray:
        return self.hidden_features_from_input(x) @ self.output_head.T + self.output_bias

    def labels(self, x: FloatArray) -> NDArray[np.int64]:
        return np.argmax(self.logits(x), axis=1)


def make_linear_lowrank_target(
    *,
    n_features: int,
    n_classes: int,
    rank: int,
    seed: int,
    adapter_scale: float = 0.35,
    base_scale: float = 1.0,
) -> LinearLowRankTarget:
    if rank <= 0:
        raise ValueError("rank must be positive")
    if rank > min(n_features, n_classes):
        raise ValueError("rank must not exceed min(n_features, n_classes)")

    rng = np.random.default_rng(seed)
    base_weight = rng.normal(0.0, base_scale / np.sqrt(n_features), size=(n_classes, n_features))
    adapter_left = rng.normal(0.0, adapter_scale / np.sqrt(rank), size=(n_classes, rank))
    adapter_right = rng.normal(0.0, adapter_scale / np.sqrt(n_features), size=(n_features, rank))
    bias = rng.normal(0.0, 0.1, size=n_classes)
    return LinearLowRankTarget(base_weight, adapter_left, adapter_right, bias)


def make_relu_feature_lowrank_target(
    *,
    n_features: int,
    hidden_features: int,
    n_classes: int,
    rank: int,
    seed: int,
    adapter_scale: float = 0.35,
    base_scale: float = 1.0,
) -> ReLUFeatureLowRankTarget:
    if rank <= 0:
        raise ValueError("rank must be positive")
    if rank > min(hidden_features, n_classes):
        raise ValueError("rank must not exceed min(hidden_features, n_classes)")

    rng = np.random.default_rng(seed)
    feature_weight = rng.normal(0.0, 1.0 / np.sqrt(n_features), size=(hidden_features, n_features))
    feature_bias = rng.normal(0.0, 0.25, size=hidden_features)
    base_head = rng.normal(0.0, base_scale / np.sqrt(hidden_features), size=(n_classes, hidden_features))
    adapter_left = rng.normal(0.0, adapter_scale / np.sqrt(rank), size=(n_classes, rank))
    adapter_right = rng.normal(0.0, adapter_scale / np.sqrt(hidden_features), size=(hidden_features, rank))
    bias = rng.normal(0.0, 0.1, size=n_classes)
    return ReLUFeatureLowRankTarget(
        feature_weight=feature_weight,
        feature_bias=feature_bias,
        base_head=base_head,
        adapter_left=adapter_left,
        adapter_right=adapter_right,
        bias=bias,
    )


def make_hidden_lora_block_target(
    *,
    n_features: int,
    hidden_features: int,
    n_classes: int,
    rank: int,
    seed: int,
    adapter_scale: float = 0.25,
    base_scale: float = 0.5,
    hidden_bias: float = 3.0,
) -> HiddenLoRABlockTarget:
    if rank <= 0:
        raise ValueError("rank must be positive")
    if rank > min(hidden_features, n_features):
        raise ValueError("rank must not exceed min(hidden_features, n_features)")
    if n_classes < hidden_features:
        raise ValueError("n_classes must be at least hidden_features for the left-invertible output-head experiment")

    rng = np.random.default_rng(seed)
    base_hidden_weight = rng.normal(0.0, base_scale / np.sqrt(n_features), size=(hidden_features, n_features))
    adapter_left = rng.normal(0.0, adapter_scale / np.sqrt(rank), size=(hidden_features, rank))
    adapter_right = rng.normal(0.0, adapter_scale / np.sqrt(n_features), size=(n_features, rank))
    output_head = rng.normal(0.0, 1.0 / np.sqrt(hidden_features), size=(n_classes, hidden_features))
    while np.linalg.matrix_rank(output_head, tol=1e-10) < hidden_features:
        output_head = rng.normal(0.0, 1.0 / np.sqrt(hidden_features), size=(n_classes, hidden_features))
    return HiddenLoRABlockTarget(
        base_hidden_weight=base_hidden_weight,
        adapter_left=adapter_left,
        adapter_right=adapter_right,
        hidden_bias=np.full(hidden_features, hidden_bias, dtype=np.float64),
        output_head=output_head,
        output_bias=rng.normal(0.0, 0.1, size=n_classes),
    )


def random_probe_points(n_points: int, n_features: int, seed: int, scale: float = 1.0) -> FloatArray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=(n_points, n_features))
