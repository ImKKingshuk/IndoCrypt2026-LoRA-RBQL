from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import FloatArray


@dataclass(frozen=True)
class ClassifierQuotientCertificate:
    hidden_dim: int
    rank: int
    nullity: int
    observable_fraction: float
    blind_fraction: float
    null_vector: FloatArray | None
    null_residual: float


@dataclass(frozen=True)
class LinearJacobianCertificate:
    rows: int
    parameters: int
    rank: int
    nullity_lower_bound: int
    observable_fraction: float


def classifier_quotient_certificate(classifier_weight: FloatArray, *, tol: float = 1e-10) -> ClassifierQuotientCertificate:
    weight = np.asarray(classifier_weight, dtype=np.float64)
    if weight.ndim != 2:
        raise ValueError("classifier_weight must be a matrix")
    _, singular_values, vt = np.linalg.svd(weight, full_matrices=True)
    rank = int(np.sum(singular_values > tol))
    hidden_dim = int(weight.shape[1])
    nullity = hidden_dim - rank
    null_vector = None
    null_residual = float("nan")
    if nullity > 0:
        null_vector = vt[rank].astype(np.float64)
        null_residual = float(np.linalg.norm(weight @ null_vector))
    return ClassifierQuotientCertificate(
        hidden_dim=hidden_dim,
        rank=rank,
        nullity=nullity,
        observable_fraction=rank / hidden_dim if hidden_dim else 0.0,
        blind_fraction=nullity / hidden_dim if hidden_dim else 0.0,
        null_vector=null_vector,
        null_residual=null_residual,
    )


def nullspace_drift_certificate(
    classifier_weight: FloatArray,
    classifier_bias: FloatArray,
    hidden: FloatArray,
    *,
    seed: int,
    tol: float = 1e-10,
) -> dict[str, float | int | bool]:
    quotient = classifier_quotient_certificate(classifier_weight, tol=tol)
    hidden_array = np.asarray(hidden, dtype=np.float64)
    bias = np.asarray(classifier_bias, dtype=np.float64)
    if quotient.null_vector is None:
        return {
            "endpoint_nonidentifiable": False,
            "endpoint_max_logit_drift": float("nan"),
            "endpoint_label_flip_rate": float("nan"),
            "endpoint_hidden_shift_norm": 0.0,
            "endpoint_hidden_norm": float(np.mean(np.linalg.norm(hidden_array, axis=1))) if hidden_array.size else 0.0,
            "endpoint_hidden_shift_ratio": 0.0,
        }
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(0.0, 1.0, size=(hidden_array.shape[0], 1))
    changed_hidden = hidden_array + coefficients @ quotient.null_vector.reshape(1, -1)
    logits_a = hidden_array @ np.asarray(classifier_weight, dtype=np.float64).T + bias
    logits_b = changed_hidden @ np.asarray(classifier_weight, dtype=np.float64).T + bias
    hidden_norm = float(np.mean(np.linalg.norm(hidden_array, axis=1)))
    shift_norm = float(np.mean(np.linalg.norm(changed_hidden - hidden_array, axis=1)))
    return {
        "endpoint_nonidentifiable": True,
        "endpoint_max_logit_drift": float(np.max(np.abs(logits_a - logits_b))),
        "endpoint_label_flip_rate": float(np.mean(np.argmax(logits_a, axis=1) != np.argmax(logits_b, axis=1))),
        "endpoint_hidden_shift_norm": shift_norm,
        "endpoint_hidden_norm": hidden_norm,
        "endpoint_hidden_shift_ratio": shift_norm / max(hidden_norm, 1e-12),
    }


def invert_endpoint_hidden_states(
    output_weight: FloatArray,
    output_bias: FloatArray,
    logits: FloatArray,
    *,
    tol: float = 1e-10,
) -> FloatArray:
    """Recover final hidden states from exact endpoint logits when the head is injective.

    This models full-logit language-model-style APIs with a public output head
    whose column rank equals the hidden dimension. It intentionally raises on
    low-rank classifier heads, where endpoint logits expose only a quotient.
    """

    weight = np.asarray(output_weight, dtype=np.float64)
    bias = np.asarray(output_bias, dtype=np.float64)
    logit_array = np.asarray(logits, dtype=np.float64)
    if weight.ndim != 2:
        raise ValueError("output_weight must be a matrix")
    if logit_array.ndim != 2:
        raise ValueError("logits must be a matrix")
    if bias.shape != (weight.shape[0],):
        raise ValueError("output_bias must match the output dimension")
    if logit_array.shape[1] != weight.shape[0]:
        raise ValueError("logits must have one column per output row")
    gram = weight.T @ weight
    eigvals = np.linalg.eigvalsh(gram)
    rank = int(np.sum(eigvals > tol))
    if rank < weight.shape[1]:
        raise ValueError("endpoint output head is not injective; hidden states are identifiable only modulo the head kernel")
    return np.linalg.solve(gram, ((logit_array - bias) @ weight).T).T


def unbiased_log_probs_from_biased_token_log_probs(biased_log_probs: FloatArray, *, bias: float) -> FloatArray:
    """Recover unbiased token log-probabilities from per-token biased logprobs.

    Each entry is the log-probability of token i after adding the same positive
    logit bias to token i alone. This is the standard logit-bias inversion used
    by top-k APIs that can force a token into the returned set.
    """

    if bias <= 0:
        raise ValueError("bias must be positive")
    observed = np.asarray(biased_log_probs, dtype=np.float64)
    q = np.clip(np.exp(observed), np.finfo(np.float64).tiny, 1.0 - np.finfo(np.float64).eps)
    log_odds = np.log(q) - np.log1p(-q) - bias
    return -np.logaddexp(0.0, -log_odds)


def invert_hidden_states_from_log_probs(
    output_weight: FloatArray,
    output_bias: FloatArray,
    log_probs: FloatArray,
    *,
    tol: float = 1e-10,
) -> FloatArray:
    """Recover hidden states from log-probability vectors up to softmax shift.

    Log-probabilities identify logits only up to an example-wise additive
    constant. A public output head still identifies hidden states if the
    augmented matrix [W, -1] has full column rank.
    """

    weight = np.asarray(output_weight, dtype=np.float64)
    bias = np.asarray(output_bias, dtype=np.float64)
    log_prob_array = np.asarray(log_probs, dtype=np.float64)
    if weight.ndim != 2:
        raise ValueError("output_weight must be a matrix")
    if log_prob_array.ndim != 2:
        raise ValueError("log_probs must be a matrix")
    if bias.shape != (weight.shape[0],):
        raise ValueError("output_bias must match the output dimension")
    if log_prob_array.shape[1] != weight.shape[0]:
        raise ValueError("log_probs must have one column per output row")
    augmented = np.column_stack([weight, -np.ones(weight.shape[0], dtype=np.float64)])
    gram = augmented.T @ augmented
    eigvals = np.linalg.eigvalsh(gram)
    rank = int(np.sum(eigvals > tol))
    if rank < augmented.shape[1]:
        raise ValueError("augmented softmax endpoint head is not injective; hidden states are identifiable only modulo a quotient")
    recovered = np.linalg.solve(gram, ((log_prob_array - bias) @ augmented).T).T
    return recovered[:, : weight.shape[1]]


def invert_hidden_states_from_shifted_logits(
    output_weight: FloatArray,
    output_bias: FloatArray,
    shifted_logits: FloatArray,
    *,
    tol: float = 1e-10,
) -> FloatArray:
    """Recover hidden states from logits known up to an example-wise shift.

    Temperature-zero top-1 oracles with controllable logit bias can recover
    pairwise logit differences but not the additive logit origin. The same
    augmented full-rank condition used for log-probability inversion removes
    this ambiguity.
    """

    return invert_hidden_states_from_log_probs(
        output_weight,
        output_bias,
        shifted_logits,
        tol=tol,
    )


def linear_jacobian_certificate(jacobian: FloatArray, *, tol: float = 1e-8) -> LinearJacobianCertificate:
    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("jacobian must be a matrix")
    rank = int(np.linalg.matrix_rank(matrix, tol=tol))
    parameters = int(matrix.shape[1])
    return LinearJacobianCertificate(
        rows=int(matrix.shape[0]),
        parameters=parameters,
        rank=rank,
        nullity_lower_bound=parameters - rank,
        observable_fraction=rank / parameters if parameters else 0.0,
    )
