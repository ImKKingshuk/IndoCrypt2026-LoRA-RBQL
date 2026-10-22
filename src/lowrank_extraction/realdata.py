from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .models import FloatArray


@dataclass(frozen=True)
class PublicFeatureMap:
    scaler: StandardScaler
    pca: PCA

    @property
    def input_dim(self) -> int:
        return int(self.scaler.n_features_in_)

    @property
    def feature_dim(self) -> int:
        return int(self.pca.n_components_)

    def __call__(self, x: FloatArray) -> FloatArray:
        x2 = np.atleast_2d(np.asarray(x, dtype=np.float64))
        transformed = self.pca.transform(self.scaler.transform(x2))
        return np.maximum(0.0, transformed)


@dataclass(frozen=True)
class DigitsAdapterData:
    dataset: str
    x_train: FloatArray
    x_test: FloatArray
    y_train: NDArray[np.int64]
    y_test: NDArray[np.int64]
    feature_map: PublicFeatureMap
    z_train: FloatArray
    z_test: FloatArray


@dataclass(frozen=True)
class MappedFeatureLowRankTarget:
    feature_map: PublicFeatureMap
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
    def rank(self) -> int:
        return int(self.adapter_left.shape[1])

    def features(self, x: FloatArray) -> FloatArray:
        return self.feature_map(x)

    def logits(self, x: FloatArray) -> FloatArray:
        return self.features(x) @ self.head.T + self.bias

    def labels(self, x: FloatArray) -> NDArray[np.int64]:
        return np.argmax(self.logits(x), axis=1)


@dataclass(frozen=True)
class RealAdapterProblem:
    data: DigitsAdapterData
    target: MappedFeatureLowRankTarget
    full_delta: FloatArray
    base_train_accuracy: float
    base_test_accuracy: float
    adapted_train_accuracy: float
    adapted_test_accuracy: float
    rank_energy: float


def load_public_features(
    *,
    dataset: str,
    seed: int,
    feature_dim: int = 8,
    classes: tuple[int, ...] | None = None,
    test_size: float = 0.35,
) -> DigitsAdapterData:
    if dataset == "digits":
        loaded = load_digits()
        x = loaded.data.astype(np.float64) / 16.0
        y = loaded.target.astype(np.int64)
        if classes is None:
            classes = (0, 1, 2)
    elif dataset == "wine":
        loaded = load_wine()
        x = loaded.data.astype(np.float64)
        y = loaded.target.astype(np.int64)
        if classes is None:
            classes = tuple(int(label) for label in sorted(np.unique(y)))
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    class_to_local = {klass: idx for idx, klass in enumerate(classes)}
    mask = np.isin(y, np.asarray(classes, dtype=np.int64))
    x = x[mask]
    y = np.asarray([class_to_local[int(label)] for label in y[mask]], dtype=np.int64)
    if feature_dim > min(x.shape[0], x.shape[1]):
        raise ValueError("feature_dim is too large for the selected dataset")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    scaler = StandardScaler().fit(x_train)
    pca = PCA(n_components=feature_dim, random_state=seed).fit(scaler.transform(x_train))
    feature_map = PublicFeatureMap(scaler=scaler, pca=pca)
    return DigitsAdapterData(
        dataset=dataset,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_map=feature_map,
        z_train=feature_map(x_train),
        z_test=feature_map(x_test),
    )


def load_digits_public_features(
    *,
    seed: int,
    feature_dim: int = 8,
    classes: tuple[int, ...] = (0, 1, 2),
    test_size: float = 0.35,
) -> DigitsAdapterData:
    return load_public_features(
        dataset="digits",
        seed=seed,
        feature_dim=feature_dim,
        classes=classes,
        test_size=test_size,
    )


def ridge_head(features: FloatArray, labels: NDArray[np.int64], *, n_classes: int, alpha: float = 1e-2) -> tuple[FloatArray, FloatArray]:
    z = np.asarray(features, dtype=np.float64)
    y = np.eye(n_classes, dtype=np.float64)[labels]
    design = np.hstack([z, np.ones((z.shape[0], 1), dtype=np.float64)])
    reg = alpha * np.eye(design.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    theta = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return theta[:-1].T, theta[-1]


def accuracy_from_head(features: FloatArray, labels: NDArray[np.int64], head: FloatArray, bias: FloatArray) -> float:
    pred = np.argmax(features @ head.T + bias, axis=1)
    return float(np.mean(pred == labels))


def shifted_digits_labels(labels: NDArray[np.int64], *, seed: int, n_classes: int, fraction: float = 0.35) -> NDArray[np.int64]:
    rng = np.random.default_rng(seed)
    shifted = np.array(labels, copy=True)
    mask = rng.random(labels.shape[0]) < fraction
    shifted[mask] = (shifted[mask] + 1) % n_classes
    return shifted


def make_digits_adapter_problem(
    *,
    seed: int,
    feature_dim: int = 8,
    rank: int = 2,
    classes: tuple[int, ...] = (0, 1, 2),
    dataset: str = "digits",
) -> RealAdapterProblem:
    data = load_public_features(seed=seed, feature_dim=feature_dim, classes=classes, dataset=dataset)
    n_classes = len(np.unique(data.y_train))
    base_head, base_bias = ridge_head(data.z_train, data.y_train, n_classes=n_classes)
    shifted_train = shifted_digits_labels(data.y_train, seed=seed + 10_000, n_classes=n_classes)
    adapted_full_head, adapted_bias = ridge_head(data.z_train, shifted_train, n_classes=n_classes)
    full_delta = adapted_full_head - base_head
    u, s, vt = np.linalg.svd(full_delta, full_matrices=False)
    rank = min(rank, len(s))
    low_rank_delta = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    adapted_head = base_head + low_rank_delta
    adapter_left = u[:, :rank] * s[:rank]
    adapter_right = vt[:rank, :].T

    target = MappedFeatureLowRankTarget(
        feature_map=data.feature_map,
        base_head=base_head,
        adapter_left=adapter_left,
        adapter_right=adapter_right,
        bias=adapted_bias,
    )

    total_energy = float(np.sum(s**2))
    rank_energy = 1.0 if total_energy == 0.0 else float(np.sum(s[:rank] ** 2) / total_energy)
    return RealAdapterProblem(
        data=data,
        target=target,
        full_delta=full_delta,
        base_train_accuracy=accuracy_from_head(data.z_train, data.y_train, base_head, base_bias),
        base_test_accuracy=accuracy_from_head(data.z_test, data.y_test, base_head, base_bias),
        adapted_train_accuracy=accuracy_from_head(data.z_train, data.y_train, adapted_head, adapted_bias),
        adapted_test_accuracy=accuracy_from_head(data.z_test, data.y_test, adapted_head, adapted_bias),
        rank_energy=rank_energy,
    )
