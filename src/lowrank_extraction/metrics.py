from __future__ import annotations

import numpy as np

from .models import FloatArray


def relative_frobenius_error(estimate: FloatArray, truth: FloatArray) -> float:
    denom = np.linalg.norm(truth, ord="fro")
    if denom == 0:
        return float(np.linalg.norm(estimate, ord="fro"))
    return float(np.linalg.norm(estimate - truth, ord="fro") / denom)


def max_absolute_error(estimate: FloatArray, truth: FloatArray) -> float:
    return float(np.max(np.abs(estimate - truth)))


def label_agreement(weight_a: FloatArray, bias_a: FloatArray, weight_b: FloatArray, bias_b: FloatArray, x: FloatArray) -> float:
    pred_a = np.argmax(x @ weight_a.T + bias_a, axis=1)
    pred_b = np.argmax(x @ weight_b.T + bias_b, axis=1)
    return float(np.mean(pred_a == pred_b))


def singular_value_energy(values: FloatArray, rank: int) -> float:
    total = float(np.sum(values**2))
    if total == 0:
        return 1.0
    return float(np.sum(values[:rank] ** 2) / total)


def binary_boundary_agreement(normal: FloatArray, offset: float, weight: FloatArray, bias: FloatArray, x: FloatArray) -> float:
    if weight.shape[0] != 2:
        raise ValueError("binary boundary agreement requires exactly two classes")
    truth_margin = x @ (weight[1] - weight[0]) + (bias[1] - bias[0])
    truth = (truth_margin > 0).astype(np.int64)
    pred = (x @ normal + offset > 0).astype(np.int64)
    return float(np.mean(pred == truth))


def prediction_agreement(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.asarray(prediction) == np.asarray(truth)))
