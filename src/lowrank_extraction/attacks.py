from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import FloatArray
from .oracles import HardLabelOracle, RawLogitOracle


@dataclass(frozen=True)
class RawRecoveryResult:
    recovered_weight: FloatArray
    recovered_bias: FloatArray
    recovered_delta: FloatArray
    rank_projected_delta: FloatArray
    singular_values: FloatArray
    query_count: int


@dataclass(frozen=True)
class FeatureHeadRecoveryResult:
    recovered_head: FloatArray
    recovered_bias: FloatArray
    recovered_delta: FloatArray
    rank_projected_delta: FloatArray
    singular_values: FloatArray
    feature_rank: int
    query_count: int


@dataclass(frozen=True)
class TopKFeatureHeadRecoveryResult:
    recovered_head: FloatArray
    recovered_bias: FloatArray
    recovered_delta: FloatArray
    rank_projected_delta: FloatArray
    singular_values: FloatArray
    observations_per_class: np.ndarray
    solved_classes: np.ndarray
    query_count: int


@dataclass(frozen=True)
class LowRankAssistedFeatureHeadRecoveryResult:
    recovered_head: FloatArray
    recovered_bias: FloatArray
    recovered_delta: FloatArray
    rank_projected_delta: FloatArray
    singular_values: FloatArray
    observations_per_class: np.ndarray
    full_row_solved_classes: np.ndarray
    lowrank_solved_classes: np.ndarray
    solved_classes: np.ndarray
    row_space_rank: int
    observation_count: int
    query_count: int


@dataclass(frozen=True)
class HiddenLoRABlockRecoveryResult:
    recovered_hidden_weight: FloatArray
    recovered_hidden_bias: FloatArray
    recovered_delta: FloatArray
    rank_projected_delta: FloatArray
    singular_values: FloatArray
    output_head_rank: int
    activation_margin: float
    query_count: int


@dataclass(frozen=True)
class HardLabelSurrogateResult:
    weight: FloatArray
    bias: FloatArray
    train_agreement: float
    query_count: int


@dataclass(frozen=True)
class TopKScoreRecoveryResult:
    recovered_weight: FloatArray
    recovered_bias: FloatArray
    recovered_delta: FloatArray
    rank_projected_delta: FloatArray
    singular_values: FloatArray
    observations_per_class: np.ndarray
    solved_classes: np.ndarray
    query_count: int


@dataclass(frozen=True)
class BinaryBoundaryRecoveryResult:
    normal: FloatArray
    offset: float
    boundary_points: FloatArray
    solved: bool
    train_agreement: float
    query_count: int


@dataclass(frozen=True)
class MulticlassBoundaryRecoveryResult:
    pair_normals: FloatArray
    pair_offsets: FloatArray
    solved_pairs: np.ndarray
    boundary_counts: np.ndarray
    validation_agreement: float
    query_count: int


@dataclass(frozen=True)
class FeatureBoundaryRecoveryResult:
    pair_normals: FloatArray
    pair_offsets: FloatArray
    solved_pairs: np.ndarray
    boundary_counts: np.ndarray
    validation_agreement: float
    query_count: int


@dataclass(frozen=True)
class RankingBoundaryRecoveryResult:
    pair_normals: FloatArray
    pair_offsets: FloatArray
    solved_pairs: np.ndarray
    boundary_counts: np.ndarray
    validation_agreement: float
    query_count: int


def recover_raw_affine(
    oracle: RawLogitOracle,
    *,
    base_weight: FloatArray,
    rank: int | None = None,
) -> RawRecoveryResult:
    """Recover an affine model exactly from raw logits using zero and basis queries.

    For f(x) = W x + b in d dimensions, f(0) gives b and f(e_j)-f(0)
    gives column j of W. If the public base W0 is known, Delta = W-W0.
    """

    n_classes, n_features = base_weight.shape
    zero = np.zeros((1, n_features), dtype=np.float64)
    recovered_bias = oracle.query(zero)[0]

    basis = np.eye(n_features, dtype=np.float64)
    basis_logits = oracle.query(basis)
    recovered_weight = (basis_logits - recovered_bias).T

    recovered_delta = recovered_weight - base_weight
    u, s, vt = np.linalg.svd(recovered_delta, full_matrices=False)
    if rank is None:
        rank = min(n_classes, n_features)
    rank_projected_delta = (u[:, :rank] * s[:rank]) @ vt[:rank, :]

    return RawRecoveryResult(
        recovered_weight=recovered_weight,
        recovered_bias=recovered_bias,
        recovered_delta=recovered_delta,
        rank_projected_delta=rank_projected_delta,
        singular_values=s,
        query_count=oracle.query_count,
    )


def recover_raw_feature_head(
    oracle: RawLogitOracle,
    *,
    base_head: FloatArray,
    feature_map,
    input_dim: int,
    feature_dim: int,
    rank: int | None = None,
    seed: int = 0,
    max_attempts: int = 50_000,
    sample_scale: float = 2.0,
) -> FeatureHeadRecoveryResult:
    """Recover a linear adapted head over a public feature map from raw logits."""

    n_classes = base_head.shape[0]
    rng = np.random.default_rng(seed)
    chosen_x: list[FloatArray] = []
    design_rows: list[FloatArray] = []
    attempts = 0
    target_rank = feature_dim + 1

    while len(design_rows) < target_rank and attempts < max_attempts:
        attempts += 1
        x = rng.normal(0.0, sample_scale, size=input_dim)
        z = np.asarray(feature_map(x), dtype=np.float64).reshape(-1)
        if z.shape[0] != feature_dim:
            raise ValueError("feature_map returned the wrong feature dimension")
        row = np.concatenate([z, np.ones(1, dtype=np.float64)])
        candidate = design_rows + [row]
        if np.linalg.matrix_rank(np.vstack(candidate), tol=1e-10) > len(design_rows):
            chosen_x.append(x)
            design_rows.append(row)

    if len(design_rows) < target_rank:
        raise RuntimeError("could not find enough independent feature vectors")

    design = np.vstack(design_rows)
    values = oracle.query(np.vstack(chosen_x))
    theta, *_ = np.linalg.lstsq(design, values, rcond=None)
    recovered_head = theta[:feature_dim, :].T
    recovered_bias = theta[feature_dim, :]

    recovered_delta = recovered_head - base_head
    u, s, vt = np.linalg.svd(recovered_delta, full_matrices=False)
    if rank is None:
        rank = min(n_classes, feature_dim)
    rank_projected_delta = (u[:, :rank] * s[:rank]) @ vt[:rank, :]

    return FeatureHeadRecoveryResult(
        recovered_head=recovered_head,
        recovered_bias=recovered_bias,
        recovered_delta=recovered_delta,
        rank_projected_delta=rank_projected_delta,
        singular_values=s,
        feature_rank=int(np.linalg.matrix_rank(design)),
        query_count=oracle.query_count,
    )


def recover_hidden_lora_block_on_positive_cell(
    oracle: RawLogitOracle,
    *,
    base_hidden_weight: FloatArray,
    hidden_bias: FloatArray,
    output_head: FloatArray,
    output_bias: FloatArray,
    rank: int | None = None,
    step: float = 1e-2,
) -> HiddenLoRABlockRecoveryResult:
    """Recover a hidden low-rank adapter inside one all-active ReLU cell.

    If the public output head has full column rank, raw logits reveal hidden
    activations by left inversion. Around a point whose adapted hidden
    preactivations stay positive, ReLU is the identity and the hidden LoRA block
    reduces to affine recovery with d+1 local queries.
    """

    hidden_features, n_features = base_hidden_weight.shape
    if hidden_bias.shape != (hidden_features,):
        raise ValueError("hidden_bias has the wrong shape")
    if output_head.shape[1] != hidden_features:
        raise ValueError("output_head has the wrong hidden dimension")
    if output_bias.shape != (output_head.shape[0],):
        raise ValueError("output_bias has the wrong shape")
    if step <= 0:
        raise ValueError("step must be positive")

    output_head_rank = int(np.linalg.matrix_rank(output_head, tol=1e-10))
    if output_head_rank < hidden_features:
        raise ValueError("output_head must have full column rank")

    x = np.vstack([np.zeros(n_features, dtype=np.float64), step * np.eye(n_features, dtype=np.float64)])
    logits = oracle.query(x)
    hidden = (logits - output_bias) @ np.linalg.pinv(output_head.T)
    recovered_hidden_bias = hidden[0]
    recovered_hidden_weight = ((hidden[1:] - recovered_hidden_bias) / step).T
    recovered_delta = recovered_hidden_weight - base_hidden_weight

    if rank is None:
        rank = min(hidden_features, n_features)
    u, s, vt = np.linalg.svd(recovered_delta, full_matrices=False)
    rank_projected_delta = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    preactivations = hidden
    activation_margin = float(np.min(preactivations))

    return HiddenLoRABlockRecoveryResult(
        recovered_hidden_weight=recovered_hidden_weight,
        recovered_hidden_bias=recovered_hidden_bias,
        recovered_delta=recovered_delta,
        rank_projected_delta=rank_projected_delta,
        singular_values=s,
        output_head_rank=output_head_rank,
        activation_margin=activation_margin,
        query_count=oracle.query_count,
    )


def recover_topk_scored_affine(
    oracle,
    *,
    base_weight: FloatArray,
    rank: int | None = None,
    random_queries: int = 0,
    seed: int = 0,
) -> TopKScoreRecoveryResult:
    """Recover class rows from score-bearing top-k outputs by linear equations.

    Whenever class c is returned with its score for a query x, the attacker gets
    one equation <w_c, x> + b_c = score. A class row is identifiable once at
    least d+1 independent observations are available. With k equal to the number
    of classes this reduces to exact raw-score recovery; with small k it becomes
    a measurable restricted-oracle attack.
    """

    n_classes, n_features = base_weight.shape
    rng = np.random.default_rng(seed)
    chosen = [np.zeros(n_features, dtype=np.float64)]
    chosen.extend(np.eye(n_features, dtype=np.float64))
    if random_queries > 0:
        chosen.extend(rng.normal(0.0, 1.0, size=(random_queries, n_features)))
    x = np.asarray(chosen, dtype=np.float64)

    indices, scores = oracle.query(x)
    if scores is None:
        raise ValueError("recover_topk_scored_affine requires score-bearing top-k outputs")

    rows_by_class: list[list[np.ndarray]] = [[] for _ in range(n_classes)]
    y_by_class: list[list[float]] = [[] for _ in range(n_classes)]
    for i in range(x.shape[0]):
        x_aug = np.concatenate([x[i], np.ones(1, dtype=np.float64)])
        for pos, klass in enumerate(indices[i]):
            rows_by_class[int(klass)].append(x_aug)
            y_by_class[int(klass)].append(float(scores[i, pos]))

    recovered_weight = np.array(base_weight, copy=True)
    recovered_bias = np.zeros(n_classes, dtype=np.float64)
    observations = np.array([len(v) for v in rows_by_class], dtype=np.int64)
    solved = np.zeros(n_classes, dtype=bool)

    for klass in range(n_classes):
        if observations[klass] < n_features + 1:
            continue
        design = np.vstack(rows_by_class[klass])
        values = np.asarray(y_by_class[klass], dtype=np.float64)
        if np.linalg.matrix_rank(design) < n_features + 1:
            continue
        theta, *_ = np.linalg.lstsq(design, values, rcond=None)
        recovered_weight[klass] = theta[:n_features]
        recovered_bias[klass] = theta[n_features]
        solved[klass] = True

    recovered_delta = recovered_weight - base_weight
    u, s, vt = np.linalg.svd(recovered_delta, full_matrices=False)
    if rank is None:
        rank = min(n_classes, n_features)
    rank_projected_delta = (u[:, :rank] * s[:rank]) @ vt[:rank, :]

    return TopKScoreRecoveryResult(
        recovered_weight=recovered_weight,
        recovered_bias=recovered_bias,
        recovered_delta=recovered_delta,
        rank_projected_delta=rank_projected_delta,
        singular_values=s,
        observations_per_class=observations,
        solved_classes=solved,
        query_count=oracle.query_count,
    )


def recover_topk_scored_feature_head(
    oracle,
    *,
    base_head: FloatArray,
    feature_map,
    input_dim: int,
    feature_dim: int,
    rank: int | None = None,
    random_queries: int,
    seed: int = 0,
    sample_scale: float = 2.0,
) -> TopKFeatureHeadRecoveryResult:
    """Recover a public-feature adapted head from score-bearing top-k outputs.

    Each returned score for class c gives one linear equation in row c of the
    adapted head over public features phi(x). Rows are solved independently once
    enough independent scored feature observations are available.
    """

    n_classes = base_head.shape[0]
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, sample_scale, size=(random_queries, input_dim))
    z = np.asarray(feature_map(x), dtype=np.float64)
    if z.shape != (random_queries, feature_dim):
        raise ValueError("feature_map returned the wrong feature matrix shape")

    indices, scores = oracle.query(x)
    if scores is None:
        raise ValueError("recover_topk_scored_feature_head requires score-bearing top-k outputs")

    rows_by_class: list[list[np.ndarray]] = [[] for _ in range(n_classes)]
    y_by_class: list[list[float]] = [[] for _ in range(n_classes)]
    for i in range(random_queries):
        z_aug = np.concatenate([z[i], np.ones(1, dtype=np.float64)])
        for pos, klass in enumerate(indices[i]):
            rows_by_class[int(klass)].append(z_aug)
            y_by_class[int(klass)].append(float(scores[i, pos]))

    recovered_head = np.array(base_head, copy=True)
    recovered_bias = np.zeros(n_classes, dtype=np.float64)
    observations = np.array([len(v) for v in rows_by_class], dtype=np.int64)
    solved = np.zeros(n_classes, dtype=bool)

    for klass in range(n_classes):
        if observations[klass] < feature_dim + 1:
            continue
        design = np.vstack(rows_by_class[klass])
        values = np.asarray(y_by_class[klass], dtype=np.float64)
        if np.linalg.matrix_rank(design) < feature_dim + 1:
            continue
        theta, *_ = np.linalg.lstsq(design, values, rcond=None)
        recovered_head[klass] = theta[:feature_dim]
        recovered_bias[klass] = theta[feature_dim]
        solved[klass] = True

    recovered_delta = recovered_head - base_head
    u, s, vt = np.linalg.svd(recovered_delta, full_matrices=False)
    if rank is None:
        rank = min(n_classes, feature_dim)
    rank_projected_delta = (u[:, :rank] * s[:rank]) @ vt[:rank, :]

    return TopKFeatureHeadRecoveryResult(
        recovered_head=recovered_head,
        recovered_bias=recovered_bias,
        recovered_delta=recovered_delta,
        rank_projected_delta=rank_projected_delta,
        singular_values=s,
        observations_per_class=observations,
        solved_classes=solved,
        query_count=oracle.query_count,
    )


def recover_lowrank_assisted_feature_head_from_observations(
    observations: list[tuple[int, FloatArray, float]],
    *,
    base_head: FloatArray,
    rank: int,
    query_count: int = 0,
) -> LowRankAssistedFeatureHeadRecoveryResult:
    """Recover a low-rank adapted head from sparse scored class observations.

    A generic row-wise attack needs h+1 observations to recover a class row in
    h feature dimensions. If enough fully recovered rows identify the rank-r
    adapter row space, remaining rows need only r+1 observations: r coefficients
    in the known row-space basis plus one bias.
    """

    n_classes, feature_dim = base_head.shape
    rows_by_class: list[list[np.ndarray]] = [[] for _ in range(n_classes)]
    y_by_class: list[list[float]] = [[] for _ in range(n_classes)]
    for klass, feature, score in observations:
        c = int(klass)
        if not (0 <= c < n_classes):
            raise ValueError("observation class index out of range")
        z = np.asarray(feature, dtype=np.float64).reshape(-1)
        if z.shape[0] != feature_dim:
            raise ValueError("observation feature has wrong dimension")
        rows_by_class[c].append(np.concatenate([z, np.ones(1, dtype=np.float64)]))
        y_by_class[c].append(float(score))

    recovered_head = np.array(base_head, copy=True)
    recovered_bias = np.zeros(n_classes, dtype=np.float64)
    observations_per_class = np.array([len(v) for v in rows_by_class], dtype=np.int64)
    full_solved = np.zeros(n_classes, dtype=bool)
    lowrank_solved = np.zeros(n_classes, dtype=bool)

    for klass in range(n_classes):
        if observations_per_class[klass] < feature_dim + 1:
            continue
        design = np.vstack(rows_by_class[klass])
        if np.linalg.matrix_rank(design) < feature_dim + 1:
            continue
        theta, *_ = np.linalg.lstsq(design, np.asarray(y_by_class[klass]), rcond=None)
        recovered_head[klass] = theta[:feature_dim]
        recovered_bias[klass] = theta[feature_dim]
        full_solved[klass] = True

    full_deltas = recovered_head[full_solved] - base_head[full_solved]
    row_space_rank = 0
    row_basis = np.zeros((0, feature_dim), dtype=np.float64)
    if full_deltas.shape[0] > 0:
        _, singular_values, vt = np.linalg.svd(full_deltas, full_matrices=False)
        row_space_rank = int(np.sum(singular_values > 1e-10))
        if row_space_rank >= rank:
            row_basis = vt[:rank]
            row_space_rank = rank

    if row_basis.shape[0] == rank:
        for klass in range(n_classes):
            if full_solved[klass] or observations_per_class[klass] < rank + 1:
                continue
            design = np.vstack(rows_by_class[klass])
            z = design[:, :feature_dim]
            projected = z @ row_basis.T
            lowrank_design = np.hstack([projected, np.ones((projected.shape[0], 1), dtype=np.float64)])
            if np.linalg.matrix_rank(lowrank_design) < rank + 1:
                continue
            residual_scores = np.asarray(y_by_class[klass], dtype=np.float64) - z @ base_head[klass]
            theta, *_ = np.linalg.lstsq(lowrank_design, residual_scores, rcond=None)
            recovered_head[klass] = base_head[klass] + theta[:rank] @ row_basis
            recovered_bias[klass] = theta[rank]
            lowrank_solved[klass] = True

    solved = full_solved | lowrank_solved
    recovered_delta = recovered_head - base_head
    u, s, vt = np.linalg.svd(recovered_delta, full_matrices=False)
    rank_projected_delta = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    return LowRankAssistedFeatureHeadRecoveryResult(
        recovered_head=recovered_head,
        recovered_bias=recovered_bias,
        recovered_delta=recovered_delta,
        rank_projected_delta=rank_projected_delta,
        singular_values=s,
        observations_per_class=observations_per_class,
        full_row_solved_classes=full_solved,
        lowrank_solved_classes=lowrank_solved,
        solved_classes=solved,
        row_space_rank=row_space_rank,
        observation_count=len(observations),
        query_count=query_count,
    )


def fit_hard_label_surrogate(
    oracle: HardLabelOracle,
    *,
    n_features: int,
    n_classes: int,
    n_queries: int,
    seed: int,
    steps: int = 800,
    learning_rate: float = 0.2,
    l2: float = 1e-4,
) -> HardLabelSurrogateResult:
    """Train a simple multiclass linear surrogate from hard labels.

    This is a baseline, not a cryptanalytic recovery attack. It is included to
    measure the gap between functional stealing and parameter/subspace recovery.
    """

    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n_queries, n_features))
    y = oracle.query(x)

    weight = rng.normal(0.0, 0.01, size=(n_classes, n_features))
    bias = np.zeros(n_classes, dtype=np.float64)
    y_onehot = np.eye(n_classes, dtype=np.float64)[y]

    for _ in range(steps):
        logits = x @ weight.T + bias
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        diff = (probs - y_onehot) / n_queries
        grad_w = diff.T @ x + l2 * weight
        grad_b = diff.sum(axis=0)
        weight -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    pred = np.argmax(x @ weight.T + bias, axis=1)
    train_agreement = float(np.mean(pred == y))
    return HardLabelSurrogateResult(
        weight=weight,
        bias=bias,
        train_agreement=train_agreement,
        query_count=oracle.query_count,
    )


def recover_binary_hard_label_boundary(
    oracle: HardLabelOracle,
    *,
    n_features: int,
    seed: int,
    boundary_points_needed: int | None = None,
    max_attempts: int = 20_000,
    bisection_steps: int = 50,
    sample_scale: float = 3.0,
    validation_points: int = 4096,
) -> BinaryBoundaryRecoveryResult:
    """Recover a binary affine classifier boundary from hard labels.

    Hard labels determine only the sign of a decision function. For binary
    affine classifiers this is enough to recover the separating hyperplane up
    to scale, but it is not enough to recover logits or adapter magnitudes.
    """

    if boundary_points_needed is None:
        boundary_points_needed = n_features
    if boundary_points_needed < n_features:
        raise ValueError("need at least n_features boundary points")

    rng = np.random.default_rng(seed)
    boundary_points: list[FloatArray] = []
    augmented_rows: list[FloatArray] = []
    attempts = 0

    while len(boundary_points) < boundary_points_needed and attempts < max_attempts:
        attempts += 1
        a = rng.normal(0.0, sample_scale, size=n_features)
        b = rng.normal(0.0, sample_scale, size=n_features)
        label_a = int(oracle.query(a)[0])
        label_b = int(oracle.query(b)[0])
        if label_a == label_b:
            continue

        lo = a.copy()
        hi = b.copy()
        lo_label = label_a
        for _ in range(bisection_steps):
            mid = (lo + hi) / 2.0
            mid_label = int(oracle.query(mid)[0])
            if mid_label == lo_label:
                lo = mid
            else:
                hi = mid

        point = (lo + hi) / 2.0
        row = np.concatenate([point, np.ones(1, dtype=np.float64)])
        candidate_rows = augmented_rows + [row]
        if np.linalg.matrix_rank(np.vstack(candidate_rows), tol=1e-10) > len(augmented_rows):
            boundary_points.append(point)
            augmented_rows.append(row)

    solved = len(boundary_points) >= boundary_points_needed
    if not solved:
        return BinaryBoundaryRecoveryResult(
            normal=np.zeros(n_features, dtype=np.float64),
            offset=0.0,
            boundary_points=np.asarray(boundary_points, dtype=np.float64),
            solved=False,
            train_agreement=0.0,
            query_count=oracle.query_count,
        )

    design = np.vstack(augmented_rows)
    _, _, vt = np.linalg.svd(design, full_matrices=True)
    theta = vt[-1]
    normal = theta[:n_features]
    offset = float(theta[n_features])

    x_val = rng.normal(0.0, sample_scale, size=(validation_points, n_features))
    y_val = oracle.query(x_val)
    pred = (x_val @ normal + offset > 0).astype(np.int64)
    agreement = float(np.mean(pred == y_val))
    if agreement < 0.5:
        normal = -normal
        offset = -offset
        pred = (x_val @ normal + offset > 0).astype(np.int64)
        agreement = float(np.mean(pred == y_val))

    return BinaryBoundaryRecoveryResult(
        normal=normal,
        offset=offset,
        boundary_points=np.asarray(boundary_points, dtype=np.float64),
        solved=True,
        train_agreement=agreement,
        query_count=oracle.query_count,
    )


def _boundary_point_between_labels(
    oracle: HardLabelOracle,
    a: FloatArray,
    b: FloatArray,
    label_a: int,
    label_b: int,
    *,
    bisection_steps: int,
) -> FloatArray | None:
    lo = a.copy()
    hi = b.copy()
    lo_label = label_a
    hi_label = label_b
    for _ in range(bisection_steps):
        mid = (lo + hi) / 2.0
        mid_label = int(oracle.query(mid)[0])
        if mid_label == lo_label:
            lo = mid
        elif mid_label == hi_label:
            hi = mid
        else:
            return None
    return (lo + hi) / 2.0


def _predict_from_pairwise_boundaries(
    x: FloatArray,
    pair_normals: FloatArray,
    pair_offsets: FloatArray,
    solved_pairs: np.ndarray,
) -> np.ndarray:
    x2 = np.atleast_2d(np.asarray(x, dtype=np.float64))
    n_classes = pair_offsets.shape[0]
    votes = np.zeros((x2.shape[0], n_classes), dtype=np.int64)
    for c in range(n_classes):
        for d in range(c + 1, n_classes):
            if not solved_pairs[c, d]:
                continue
            margin = x2 @ pair_normals[c, d] + pair_offsets[c, d]
            votes[:, c] += margin <= 0
            votes[:, d] += margin > 0
    return np.argmax(votes, axis=1)


def _rank_order_sign(indices: np.ndarray, c: int, d: int) -> int:
    row = list(map(int, indices))
    if c not in row or d not in row:
        return 0
    return -1 if row.index(c) < row.index(d) else 1


def _boundary_point_between_rank_orders(
    oracle,
    a: FloatArray,
    b: FloatArray,
    c: int,
    d: int,
    sign_a: int,
    sign_b: int,
    *,
    bisection_steps: int,
) -> FloatArray | None:
    lo = a.copy()
    hi = b.copy()
    lo_sign = sign_a
    hi_sign = sign_b
    for _ in range(bisection_steps):
        mid = (lo + hi) / 2.0
        mid_indices, _ = oracle.query(mid)
        mid_sign = _rank_order_sign(mid_indices[0], c, d)
        if mid_sign == 0:
            return None
        if mid_sign == lo_sign:
            lo = mid
        elif mid_sign == hi_sign:
            hi = mid
        else:
            return None
    return (lo + hi) / 2.0


def recover_multiclass_hard_label_boundaries(
    oracle: HardLabelOracle,
    *,
    n_features: int,
    n_classes: int,
    seed: int,
    boundary_points_per_pair: int | None = None,
    max_attempts: int = 200_000,
    bisection_steps: int = 50,
    sample_scale: float = 3.0,
    validation_points: int = 4096,
) -> MulticlassBoundaryRecoveryResult:
    """Recover multiclass affine decision boundaries from hard labels.

    For every class pair (c, d), the attack searches for line segments whose
    labels change directly between c and d, bisects them to the pairwise
    boundary, and recovers the separating hyperplane up to positive scale.
    If all pairs are solved, pairwise voting is functionally equivalent except
    near numerical boundary error.
    """

    if boundary_points_per_pair is None:
        boundary_points_per_pair = n_features
    if boundary_points_per_pair < n_features:
        raise ValueError("need at least n_features boundary points per pair")

    rng = np.random.default_rng(seed)
    pair_rows: list[list[list[np.ndarray]]] = [[[] for _ in range(n_classes)] for _ in range(n_classes)]
    attempts = 0

    def pair_ready(c: int, d: int) -> bool:
        return len(pair_rows[c][d]) >= boundary_points_per_pair

    while attempts < max_attempts:
        if all(pair_ready(c, d) for c in range(n_classes) for d in range(c + 1, n_classes)):
            break
        attempts += 1
        a = rng.normal(0.0, sample_scale, size=n_features)
        b = rng.normal(0.0, sample_scale, size=n_features)
        label_a = int(oracle.query(a)[0])
        label_b = int(oracle.query(b)[0])
        if label_a == label_b:
            continue
        c, d = sorted((label_a, label_b))
        if pair_ready(c, d):
            continue
        point = _boundary_point_between_labels(
            oracle,
            a,
            b,
            label_a,
            label_b,
            bisection_steps=bisection_steps,
        )
        if point is None:
            continue
        row = np.concatenate([point, np.ones(1, dtype=np.float64)])
        candidate = pair_rows[c][d] + [row]
        if np.linalg.matrix_rank(np.vstack(candidate), tol=1e-10) > len(pair_rows[c][d]):
            pair_rows[c][d].append(row)

    pair_normals = np.zeros((n_classes, n_classes, n_features), dtype=np.float64)
    pair_offsets = np.zeros((n_classes, n_classes), dtype=np.float64)
    solved_pairs = np.zeros((n_classes, n_classes), dtype=bool)
    boundary_counts = np.zeros((n_classes, n_classes), dtype=np.int64)

    x_val = rng.normal(0.0, sample_scale, size=(validation_points, n_features))
    y_val = oracle.query(x_val)

    for c in range(n_classes):
        for d in range(c + 1, n_classes):
            rows = pair_rows[c][d]
            boundary_counts[c, d] = len(rows)
            if len(rows) < boundary_points_per_pair:
                continue
            design = np.vstack(rows)
            if np.linalg.matrix_rank(design, tol=1e-10) < n_features:
                continue
            _, _, vt = np.linalg.svd(design, full_matrices=True)
            theta = vt[-1]
            normal = theta[:n_features]
            offset = float(theta[n_features])

            mask = (y_val == c) | (y_val == d)
            if np.any(mask):
                pair_pred = np.where(x_val[mask] @ normal + offset > 0, d, c)
                agreement = float(np.mean(pair_pred == y_val[mask]))
                if agreement < 0.5:
                    normal = -normal
                    offset = -offset

            pair_normals[c, d] = normal
            pair_normals[d, c] = -normal
            pair_offsets[c, d] = offset
            pair_offsets[d, c] = -offset
            solved_pairs[c, d] = True
            solved_pairs[d, c] = True

    y_pred = _predict_from_pairwise_boundaries(x_val, pair_normals, pair_offsets, solved_pairs)
    validation_agreement = float(np.mean(y_pred == y_val))

    return MulticlassBoundaryRecoveryResult(
        pair_normals=pair_normals,
        pair_offsets=pair_offsets,
        solved_pairs=solved_pairs,
        boundary_counts=boundary_counts,
        validation_agreement=validation_agreement,
        query_count=oracle.query_count,
    )


def predict_multiclass_boundaries(result: MulticlassBoundaryRecoveryResult, x: FloatArray) -> np.ndarray:
    return _predict_from_pairwise_boundaries(x, result.pair_normals, result.pair_offsets, result.solved_pairs)


def recover_feature_hard_label_boundaries(
    oracle: HardLabelOracle,
    *,
    feature_map,
    input_dim: int,
    feature_dim: int,
    n_classes: int,
    seed: int,
    boundary_points_per_pair: int | None = None,
    max_attempts: int = 200_000,
    bisection_steps: int = 50,
    sample_scale: float = 3.0,
    validation_points: int = 4096,
) -> FeatureBoundaryRecoveryResult:
    """Recover pairwise hard-label boundaries in a public feature space."""

    if boundary_points_per_pair is None:
        boundary_points_per_pair = feature_dim
    if boundary_points_per_pair < feature_dim:
        raise ValueError("need at least feature_dim boundary points per pair")

    rng = np.random.default_rng(seed)
    pair_rows: list[list[list[np.ndarray]]] = [[[] for _ in range(n_classes)] for _ in range(n_classes)]
    attempts = 0

    def pair_ready(c: int, d: int) -> bool:
        return len(pair_rows[c][d]) >= boundary_points_per_pair

    while attempts < max_attempts:
        if all(pair_ready(c, d) for c in range(n_classes) for d in range(c + 1, n_classes)):
            break
        attempts += 1
        a = rng.normal(0.0, sample_scale, size=input_dim)
        b = rng.normal(0.0, sample_scale, size=input_dim)
        label_a = int(oracle.query(a)[0])
        label_b = int(oracle.query(b)[0])
        if label_a == label_b:
            continue
        c, d = sorted((label_a, label_b))
        if pair_ready(c, d):
            continue
        point = _boundary_point_between_labels(
            oracle,
            a,
            b,
            label_a,
            label_b,
            bisection_steps=bisection_steps,
        )
        if point is None:
            continue
        z = np.asarray(feature_map(point), dtype=np.float64).reshape(-1)
        if z.shape[0] != feature_dim:
            raise ValueError("feature_map returned the wrong feature dimension")
        row = np.concatenate([z, np.ones(1, dtype=np.float64)])
        candidate = pair_rows[c][d] + [row]
        if np.linalg.matrix_rank(np.vstack(candidate), tol=1e-10) > len(pair_rows[c][d]):
            pair_rows[c][d].append(row)

    pair_normals = np.zeros((n_classes, n_classes, feature_dim), dtype=np.float64)
    pair_offsets = np.zeros((n_classes, n_classes), dtype=np.float64)
    solved_pairs = np.zeros((n_classes, n_classes), dtype=bool)
    boundary_counts = np.zeros((n_classes, n_classes), dtype=np.int64)

    x_val = rng.normal(0.0, sample_scale, size=(validation_points, input_dim))
    z_val = np.asarray(feature_map(x_val), dtype=np.float64)
    y_val = oracle.query(x_val)

    for c in range(n_classes):
        for d in range(c + 1, n_classes):
            rows = pair_rows[c][d]
            boundary_counts[c, d] = len(rows)
            if len(rows) < boundary_points_per_pair:
                continue
            design = np.vstack(rows)
            if np.linalg.matrix_rank(design, tol=1e-10) < feature_dim:
                continue
            _, _, vt = np.linalg.svd(design, full_matrices=True)
            theta = vt[-1]
            normal = theta[:feature_dim]
            offset = float(theta[feature_dim])

            mask = (y_val == c) | (y_val == d)
            if np.any(mask):
                pair_pred = np.where(z_val[mask] @ normal + offset > 0, d, c)
                agreement = float(np.mean(pair_pred == y_val[mask]))
                if agreement < 0.5:
                    normal = -normal
                    offset = -offset

            pair_normals[c, d] = normal
            pair_normals[d, c] = -normal
            pair_offsets[c, d] = offset
            pair_offsets[d, c] = -offset
            solved_pairs[c, d] = True
            solved_pairs[d, c] = True

    y_pred = _predict_from_pairwise_boundaries(z_val, pair_normals, pair_offsets, solved_pairs)
    validation_agreement = float(np.mean(y_pred == y_val))

    return FeatureBoundaryRecoveryResult(
        pair_normals=pair_normals,
        pair_offsets=pair_offsets,
        solved_pairs=solved_pairs,
        boundary_counts=boundary_counts,
        validation_agreement=validation_agreement,
        query_count=oracle.query_count,
    )


def predict_feature_boundaries(result: FeatureBoundaryRecoveryResult, features: FloatArray) -> np.ndarray:
    return _predict_from_pairwise_boundaries(features, result.pair_normals, result.pair_offsets, result.solved_pairs)


def recover_feature_rank_index_boundaries(
    oracle,
    *,
    feature_map,
    input_dim: int,
    feature_dim: int,
    n_classes: int,
    seed: int,
    boundary_points_per_pair: int | None = None,
    max_attempts: int = 200_000,
    bisection_steps: int = 50,
    sample_scale: float = 3.0,
    validation_points: int = 4096,
) -> RankingBoundaryRecoveryResult:
    """Recover feature-space boundaries from index-only class rankings."""

    if boundary_points_per_pair is None:
        boundary_points_per_pair = feature_dim
    if boundary_points_per_pair < feature_dim:
        raise ValueError("need at least feature_dim boundary points per pair")

    rng = np.random.default_rng(seed)
    pair_rows: list[list[list[np.ndarray]]] = [[[] for _ in range(n_classes)] for _ in range(n_classes)]
    attempts = 0

    def pair_ready(c: int, d: int) -> bool:
        return len(pair_rows[c][d]) >= boundary_points_per_pair

    while attempts < max_attempts:
        if all(pair_ready(c, d) for c in range(n_classes) for d in range(c + 1, n_classes)):
            break
        attempts += 1
        a = rng.normal(0.0, sample_scale, size=input_dim)
        b = rng.normal(0.0, sample_scale, size=input_dim)
        indices, scores = oracle.query(np.vstack([a, b]))
        if scores is not None:
            raise ValueError("recover_feature_rank_index_boundaries requires index-only outputs")

        for c in range(n_classes):
            for d in range(c + 1, n_classes):
                if pair_ready(c, d):
                    continue
                sign_a = _rank_order_sign(indices[0], c, d)
                sign_b = _rank_order_sign(indices[1], c, d)
                if sign_a == 0 or sign_b == 0 or sign_a == sign_b:
                    continue
                point = _boundary_point_between_rank_orders(
                    oracle,
                    a,
                    b,
                    c,
                    d,
                    sign_a,
                    sign_b,
                    bisection_steps=bisection_steps,
                )
                if point is None:
                    continue
                z = np.asarray(feature_map(point), dtype=np.float64).reshape(-1)
                if z.shape[0] != feature_dim:
                    raise ValueError("feature_map returned the wrong feature dimension")
                row = np.concatenate([z, np.ones(1, dtype=np.float64)])
                candidate = pair_rows[c][d] + [row]
                if np.linalg.matrix_rank(np.vstack(candidate), tol=1e-10) > len(pair_rows[c][d]):
                    pair_rows[c][d].append(row)

    pair_normals = np.zeros((n_classes, n_classes, feature_dim), dtype=np.float64)
    pair_offsets = np.zeros((n_classes, n_classes), dtype=np.float64)
    solved_pairs = np.zeros((n_classes, n_classes), dtype=bool)
    boundary_counts = np.zeros((n_classes, n_classes), dtype=np.int64)

    x_val = rng.normal(0.0, sample_scale, size=(validation_points, input_dim))
    z_val = np.asarray(feature_map(x_val), dtype=np.float64)
    val_indices, val_scores = oracle.query(x_val)
    if val_scores is not None:
        raise ValueError("recover_feature_rank_index_boundaries requires index-only outputs")
    y_val = val_indices[:, 0]

    for c in range(n_classes):
        for d in range(c + 1, n_classes):
            rows = pair_rows[c][d]
            boundary_counts[c, d] = len(rows)
            if len(rows) < boundary_points_per_pair:
                continue
            design = np.vstack(rows)
            if np.linalg.matrix_rank(design, tol=1e-10) < feature_dim:
                continue
            _, _, vt = np.linalg.svd(design, full_matrices=True)
            theta = vt[-1]
            normal = theta[:feature_dim]
            offset = float(theta[feature_dim])

            signs = np.array([_rank_order_sign(row, c, d) for row in val_indices])
            mask = signs != 0
            if np.any(mask):
                pred_signs = np.where(z_val[mask] @ normal + offset > 0, 1, -1)
                agreement = float(np.mean(pred_signs == signs[mask]))
                if agreement < 0.5:
                    normal = -normal
                    offset = -offset

            pair_normals[c, d] = normal
            pair_normals[d, c] = -normal
            pair_offsets[c, d] = offset
            pair_offsets[d, c] = -offset
            solved_pairs[c, d] = True
            solved_pairs[d, c] = True

    y_pred = _predict_from_pairwise_boundaries(z_val, pair_normals, pair_offsets, solved_pairs)
    validation_agreement = float(np.mean(y_pred == y_val))

    return RankingBoundaryRecoveryResult(
        pair_normals=pair_normals,
        pair_offsets=pair_offsets,
        solved_pairs=solved_pairs,
        boundary_counts=boundary_counts,
        validation_agreement=validation_agreement,
        query_count=oracle.query_count,
    )


def predict_ranking_boundaries(result: RankingBoundaryRecoveryResult, features: FloatArray) -> np.ndarray:
    return _predict_from_pairwise_boundaries(features, result.pair_normals, result.pair_offsets, result.solved_pairs)
