"""Adaptive attacker strategies against score-degradation defenses.

The stable-recovery theorem bounds least-squares recovery error by
``\\|X^+\\|_2 \\|E\\|_F``.  A defender who rounds or perturbs returned scores
controls only ``E``.  An attacker who chooses the query design controls
``\\|X^+\\|_2``, and an attacker who repeats queries or exploits the declared
rank bound shrinks the effective ``E``.  This module implements the four
levers separately so that their contributions can be ablated:

``amplify``
    Scale query inputs so that public features grow while a fixed absolute
    score grid or a fixed noise standard deviation stays constant.  This
    shrinks ``\\|X^+\\|_2`` roughly linearly in the scale factor.

``d_optimal``
    Greedily select query points that maximize ``\\log\\det(X^\\top X)``,
    which improves the conditioning of the augmented design relative to an
    unselected random or dataset pool.

``repeats``
    Query each design point several times and average the returned scores.
    Independent additive noise shrinks by ``1/\\sqrt{R}``; deterministic
    rounding does not, which the experiments report as a negative result.

``rank_project``
    Project the recovered adapter onto the declared rank bound.  This is
    the low-rank structure of the secret working for the attacker as a
    denoiser rather than as a query-count reduction.

The four levers above were all designed against a perturbation whose size does
not depend on the score it lands on.  Relative quantization breaks that
assumption on purpose, so evaluating it against those levers alone would repeat
the mistake this paper accuses non-adaptive evaluations of making.  Two further
levers are therefore implemented specifically to attack the *relative* family:

``gls``
    Weighted least squares with weights equal to the reciprocal of the
    quantizer's cell width at each returned score.  This is the maximum
    likelihood estimator for the error model the defense actually induces,
    and it costs no queries.  The weights are observable: the attacker reads
    the returned precision off the response.  Note what the lever says about
    the paper's thesis --- under an *absolute* grid the cell width is constant,
    the weights are uniform, and GLS is identically OLS; it is only the
    relative family that offers a weighting to exploit at all.

``magnitude_design``
    Steer the design toward queries whose scores are small relative to their
    feature norm.  Under relative units the absolute error on a returned score
    is proportional to that score, so a small-score row carries a
    correspondingly precise equation.  The lever pays for this in conditioning,
    since rows with uniformly small scores cluster near a hyperplane; the
    implementation probes the pool, keeps the most precise candidates, and runs
    the D-optimal selection inside that subset so the trade is made explicitly.
    The probe cost is charged to the attacker.

Amplification alone is not sufficient.  It leaves the intercept limited by
the absolute grid, because the constant column of the augmented design does
not scale with the queries, and it can starve top-``k`` class coverage by
driving every amplified query into the same argmax cell.  The ``two_stage_bias``
and ``d_optimal`` levers repair those two failure modes respectively, and the
experiments report both effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import label_agreement, relative_frobenius_error
from .models import FloatArray
from .oracles import RawLogitOracle, TopKOracle


@dataclass(frozen=True)
class AdaptiveStrategy:
    """Configuration of the adaptive attacker's query and estimation levers."""

    name: str
    amplify: float = 1.0
    d_optimal: bool = False
    repeats: int = 1
    rank_project: int | None = None
    design_size: int | None = None
    two_stage_bias: int = 0
    coverage_aware: bool = False
    gls: bool = False
    magnitude_design: bool = False

    def describe(self) -> str:
        parts = []
        if self.amplify != 1.0:
            parts.append(f"amplify={self.amplify:g}")
        if self.coverage_aware:
            parts.append("class-coverage")
        if self.magnitude_design:
            parts.append("magnitude-steered")
        if self.d_optimal:
            parts.append("D-optimal")
        if self.gls:
            parts.append("GLS")
        if self.repeats > 1:
            parts.append(f"repeats={self.repeats}")
        if self.two_stage_bias > 0:
            parts.append(f"bias-probe={self.two_stage_bias}")
        if self.rank_project is not None:
            parts.append(f"rank<={self.rank_project}")
        if self.design_size is not None:
            parts.append(f"design={self.design_size}")
        return ", ".join(parts) if parts else "none"


@dataclass
class AdaptiveResult:
    strategy: str
    description: str
    oracle_calls: int
    distinct_design_points: int
    solved_classes: int
    delta_relerr: float
    bias_abserr: float
    test_agreement: float
    design_pinv_norm: float
    design_sigma_min: float
    failure: bool
    extra: dict[str, float] = field(default_factory=dict)


def greedy_d_optimal_rows(
    design: FloatArray,
    *,
    n_select: int,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Greedy D-optimal (Fedorov-style) row selection.

    Repeatedly adds the candidate row maximizing the leverage
    ``z^\\top M^{-1} z``, which is the standard greedy step for maximizing
    ``\\log\\det M`` under a rank-one update.  Selection touches only the
    public feature map, so it costs no oracle queries.
    """

    z = np.asarray(design, dtype=np.float64)
    n_candidates, dim = z.shape
    n_select = min(int(n_select), n_candidates)
    if n_select <= 0:
        return np.empty(0, dtype=np.int64)

    # D-optimality is invariant under a global rescaling of the design, so we
    # normalize before selecting. Without this, an amplified design (large A)
    # makes the information matrix numerically singular and the selection fails.
    scale = float(np.sqrt(np.mean(np.sum(z**2, axis=1))))
    if scale > 0:
        z = z / scale

    information = ridge * np.eye(dim, dtype=np.float64)
    chosen: list[int] = []
    available = np.ones(n_candidates, dtype=bool)

    for _ in range(n_select):
        # Solve rather than invert, and fall back to the pseudoinverse if the
        # accumulated information matrix is still badly conditioned.
        try:
            precision = np.linalg.solve(information, np.eye(dim))
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(information)
        leverage = np.einsum("ij,jk,ik->i", z, precision, z)
        leverage[~available] = -np.inf
        best = int(np.argmax(leverage))
        if not np.isfinite(leverage[best]):
            break
        chosen.append(best)
        available[best] = False
        row = z[best][:, None]
        information = information + row @ row.T

    return np.asarray(chosen, dtype=np.int64)


def coverage_aware_rows(
    augmented: FloatArray,
    base_head: FloatArray,
    base_bias: FloatArray,
    *,
    n_select: int,
    k: int,
    d_optimal: bool = True,
) -> np.ndarray:
    """Select a design that keeps every class inside the returned top-``k``.

    Amplifying queries improves conditioning but drives the argmax into a
    single cell, so a score-bearing top-``k`` oracle stops returning most
    classes and their rows are never solved.  Because the base model is
    public in this threat model, the attacker can predict which classes a
    query will surface *without spending a query*: it ranks pool points by
    the public base score of each class and allocates the design budget
    evenly across classes.  Within each class bucket the selection is
    D-optimal, so conditioning is preserved.
    """

    aug = np.asarray(augmented, dtype=np.float64)
    features = aug[:, :-1]
    base_scores = features @ np.asarray(base_head, dtype=np.float64).T + np.asarray(
        base_bias, dtype=np.float64
    )
    n_candidates, n_classes = base_scores.shape
    n_select = min(int(n_select), n_candidates)
    per_class = max(1, n_select // max(1, n_classes))

    chosen: list[int] = []
    taken = np.zeros(n_candidates, dtype=bool)
    for klass in range(n_classes):
        # Rank pool points by how strongly the public base model favours this
        # class relative to its best competitor; the top of that ranking is
        # where the class is most likely to stay inside the returned top-k.
        competitor = np.max(
            np.delete(base_scores, klass, axis=1), axis=1
        ) if n_classes > 1 else np.zeros(n_candidates)
        margin = base_scores[:, klass] - competitor
        order = np.argsort(-margin)
        bucket = [int(i) for i in order if not taken[i]][: max(per_class * 4, per_class)]
        if not bucket:
            continue
        bucket_arr = np.asarray(bucket, dtype=np.int64)
        if d_optimal and len(bucket) > per_class:
            local = greedy_d_optimal_rows(aug[bucket_arr], n_select=per_class)
            picked = bucket_arr[local]
        else:
            picked = bucket_arr[:per_class]
        for i in picked:
            if not taken[i]:
                taken[i] = True
                chosen.append(int(i))

    # Spend any leftover budget on the globally best-conditioned remaining rows.
    if len(chosen) < n_select:
        remaining = np.where(~taken)[0]
        if remaining.size:
            extra = greedy_d_optimal_rows(aug[remaining], n_select=n_select - len(chosen))
            chosen.extend(int(remaining[i]) for i in extra)

    return np.asarray(chosen[:n_select], dtype=np.int64)


def _augment(features: FloatArray) -> FloatArray:
    z = np.atleast_2d(np.asarray(features, dtype=np.float64))
    return np.hstack([z, np.ones((z.shape[0], 1), dtype=np.float64)])


def _project_to_rank(delta: FloatArray, rank: int) -> FloatArray:
    u, s, vt = np.linalg.svd(np.asarray(delta, dtype=np.float64), full_matrices=False)
    rank = max(0, min(int(rank), len(s)))
    if rank == 0:
        return np.zeros_like(delta)
    return (u[:, :rank] * s[:rank]) @ vt[:rank, :]


def quantizer_cell_width(
    y: FloatArray, *, decimals: int | None, sig_digits: int | None
) -> np.ndarray:
    """Width of the quantization cell each returned score landed in.

    This is the scale of the error the defense injected into that observation,
    and it is computable by the attacker from the response alone: the returned
    precision reveals the format.  For an absolute grid it is the constant
    ``10**-decimals`` regardless of the score, which is exactly why an absolute
    defense offers no weighting to exploit --- the error is homoscedastic, and
    the ordinary least-squares estimator this codebase uses everywhere else is
    already efficient.  For ``p`` significant digits the cell containing
    ``y in [10**k, 10**(k+1))`` has width ``10**(k-p+1)``, proportional to
    ``|y|`` up to one decade, so the error is heteroscedastic and weighting is
    the textbook response.
    """

    y = np.abs(np.asarray(y, dtype=np.float64))
    if decimals is not None:
        return np.full(y.shape, 10.0 ** (-int(decimals)), dtype=np.float64)
    if sig_digits is not None:
        width = np.full(y.shape, np.nan, dtype=np.float64)
        nz = y > 0.0
        # The decade index is a floor, so it flips on a one-ulp change to a score
        # sitting exactly on a decade boundary -- and a score that is exactly on
        # the grid, which is every score a deterministic quantizer returns, sits
        # there often.  Averaging repeated identical returns perturbs the last
        # bit and would otherwise flip the weight, making a deterministic defense
        # look as though repetition had bought the attacker something.  The
        # tolerance is far below any real precision and removes the artifact.
        width[nz] = np.power(
            10.0, np.floor(np.log10(y[nz]) + 1e-9) - (int(sig_digits) - 1)
        )
        # A returned zero carries no scale of its own; the smallest observed
        # cell is the most conservative stand-in, and it never up-weights.
        smallest = float(np.min(width[nz])) if np.any(nz) else 1.0
        width[~nz] = smallest
        return width
    return np.ones(y.shape, dtype=np.float64)


def _weighted_lstsq(
    design: FloatArray, values: FloatArray, weights: FloatArray | None
) -> np.ndarray:
    """Solve ``design @ theta = values``, optionally weighting the rows."""

    design = np.asarray(design, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        theta, *_ = np.linalg.lstsq(design, values, rcond=None)
        return theta
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    theta, *_ = np.linalg.lstsq(design * w, values * w.ravel(), rcond=None)
    return theta


def solve_scores(
    design: FloatArray,
    served: FloatArray,
    *,
    decimals: int | None,
    sig_digits: int | None,
    gls: bool,
) -> np.ndarray:
    """Solve ``design @ theta = served`` column by column, optionally weighted.

    ``served`` carries one column per class.  Ordinary least squares can solve
    all of them in one call because the design is shared, but the GLS weights
    are a function of the *returned score*, so they differ per class and the
    columns have to be solved separately.  Returns ``theta`` with one column per
    class, matching ``np.linalg.lstsq`` on the unweighted problem exactly when
    ``gls`` is false or the quantizer is absolute.
    """

    design = np.asarray(design, dtype=np.float64)
    served = np.atleast_2d(np.asarray(served, dtype=np.float64))
    if not gls or decimals is not None:
        # An absolute grid has a constant cell width, so the GLS weights are
        # uniform and the weighted normal equations are the unweighted ones.
        theta, *_ = np.linalg.lstsq(design, served, rcond=None)
        return theta
    columns = []
    for c in range(served.shape[1]):
        width = quantizer_cell_width(served[:, c], decimals=decimals, sig_digits=sig_digits)
        columns.append(
            _weighted_lstsq(design, served[:, c], 1.0 / np.maximum(width, 1e-300))
        )
    return np.column_stack(columns)


def _probe_precision(
    oracle, pool: FloatArray, aug_pool: FloatArray, n_classes: int
) -> np.ndarray:
    """Per-query score magnitude relative to feature norm, from one probe pass.

    Under relative units the absolute error injected into a returned score is
    proportional to that score, so this ratio is exactly the error a query
    contributes per unit of the equation it supplies: smaller is a more precise
    row.  Under an absolute grid the ratio is uninformative about the error,
    which is the asymmetry the lever is built to exploit.
    """

    pool = np.atleast_2d(np.asarray(pool, dtype=np.float64))
    if isinstance(oracle, RawLogitOracle):
        observed = np.abs(np.asarray(oracle.query(pool), dtype=np.float64))
        magnitude = np.linalg.norm(observed, axis=1)
    else:
        _, scores = oracle.query(pool)
        assert scores is not None
        magnitude = np.linalg.norm(np.abs(np.asarray(scores, dtype=np.float64)), axis=1)
    row_norm = np.maximum(np.linalg.norm(np.asarray(aug_pool, dtype=np.float64), axis=1), 1e-300)
    return magnitude / row_norm


def _design_conditioning(design: FloatArray) -> tuple[float, float]:
    singular = np.linalg.svd(np.asarray(design, dtype=np.float64), compute_uv=False)
    sigma_min = float(singular[-1]) if singular.size else 0.0
    pinv_norm = float("inf") if sigma_min <= 0.0 else 1.0 / sigma_min
    return pinv_norm, sigma_min


def run_adaptive_topk_recovery(
    problem,
    *,
    x_pool: FloatArray,
    k: int,
    strategy: AdaptiveStrategy,
    decimals: int | None = None,
    sig_digits: int | None = None,
    noise_std: float = 0.0,
    rel_noise_std: float = 0.0,
    adversarial_eps: float = 0.0,
    seed: int = 0,
    oracle_kind: str = "topk",
) -> AdaptiveResult:
    """Run one scored recovery under an adaptive query/estimation strategy.

    The oracle restriction (``decimals``, ``noise_std``) is the defense.  The
    strategy is the attacker's response.  Query accounting reports both the
    number of distinct design points and the total oracle calls, so that
    repeat averaging is never credited for free.

    ``oracle_kind`` selects between the raw-logit oracle, which returns every
    class score and therefore isolates the estimation levers, and the scored
    top-``k`` oracle, where class coverage becomes a second binding constraint
    on top of conditioning.
    """

    feature_map = problem.data.feature_map
    n_classes = problem.target.n_classes
    feature_dim = feature_map.feature_dim

    pool = np.atleast_2d(np.asarray(x_pool, dtype=np.float64))
    if strategy.amplify != 1.0:
        pool = pool * float(strategy.amplify)

    z_pool = feature_map(pool)
    aug_pool = _augment(z_pool)

    design_size = strategy.design_size if strategy.design_size is not None else pool.shape[0]
    design_size = min(design_size, pool.shape[0])

    oracle: RawLogitOracle | TopKOracle
    if oracle_kind == "raw":
        oracle = RawLogitOracle(
            problem.target,
            decimals=decimals,
            sig_digits=sig_digits,
            noise_std=noise_std,
            rel_noise_std=rel_noise_std,
            adversarial_eps=adversarial_eps,
            seed=seed,
        )
    elif oracle_kind == "topk":
        oracle = TopKOracle(
            problem.target,
            k=k,
            include_scores=True,
            decimals=decimals,
            sig_digits=sig_digits,
            noise_std=noise_std,
            seed=seed,
        )
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unsupported oracle kind: {oracle_kind}")

    # Magnitude steering, when requested, needs to see scores before it can
    # choose a design, so it probes the whole pool once.  Those calls are real
    # and are added to the total below.
    probe_calls = 0
    candidates = np.arange(aug_pool.shape[0], dtype=np.int64)
    if strategy.magnitude_design:
        probe_calls = int(pool.shape[0])
        precision = _probe_precision(oracle, pool, aug_pool, n_classes)
        # Keep the most precise half of what the design will use, leaving the
        # D-optimal pass below a real 2:1 choice: enough headroom to recover
        # conditioning, tight enough that the steering actually bites.
        keep = min(aug_pool.shape[0], max(design_size + 1, 2 * design_size))
        candidates = np.argsort(precision)[:keep].astype(np.int64)

    sub_aug = aug_pool[candidates]
    if strategy.coverage_aware:
        chosen = coverage_aware_rows(
            sub_aug,
            problem.target.base_head,
            np.zeros(n_classes, dtype=np.float64),
            n_select=design_size,
            k=k,
            d_optimal=strategy.d_optimal,
        )
    elif strategy.d_optimal:
        chosen = greedy_d_optimal_rows(sub_aug, n_select=design_size)
    else:
        chosen = np.arange(min(design_size, sub_aug.shape[0]), dtype=np.int64)
    selected = candidates[chosen]

    x_design = pool[selected]
    aug_design = aug_pool[selected]
    pinv_norm, sigma_min = _design_conditioning(aug_design)

    repeats = max(1, int(strategy.repeats))
    score_sums: dict[tuple[int, int], float] = {}
    score_counts: dict[tuple[int, int], int] = {}
    for _ in range(repeats):
        if isinstance(oracle, RawLogitOracle):
            values = oracle.query(x_design)
            for i in range(x_design.shape[0]):
                for klass in range(n_classes):
                    key = (i, klass)
                    score_sums[key] = score_sums.get(key, 0.0) + float(values[i, klass])
                    score_counts[key] = score_counts.get(key, 0) + 1
        else:
            indices, scores = oracle.query(x_design)
            assert scores is not None
            for i in range(x_design.shape[0]):
                for pos, klass in enumerate(indices[i]):
                    key = (i, int(klass))
                    score_sums[key] = score_sums.get(key, 0.0) + float(scores[i, pos])
                    score_counts[key] = score_counts.get(key, 0) + 1

    rows_by_class: list[list[np.ndarray]] = [[] for _ in range(n_classes)]
    y_by_class: list[list[float]] = [[] for _ in range(n_classes)]
    for (i, klass), total in score_sums.items():
        rows_by_class[klass].append(aug_design[i])
        y_by_class[klass].append(total / score_counts[(i, klass)])

    recovered_head = np.array(problem.target.base_head, copy=True)
    recovered_bias = np.zeros(n_classes, dtype=np.float64)
    solved = np.zeros(n_classes, dtype=bool)
    # Rank of the per-class augmented design, averaged over the classes the
    # oracle actually returned. Amplification can push ReLU features into their
    # dead zone, which collapses this rank and makes the class unsolvable; the
    # diagnostic distinguishes "the defense held" from "the attack broke".
    class_ranks: list[int] = []
    for klass in range(n_classes):
        if len(rows_by_class[klass]) < feature_dim + 1:
            continue
        class_design = np.vstack(rows_by_class[klass])
        class_ranks.append(int(np.linalg.matrix_rank(class_design)))
        if np.linalg.matrix_rank(class_design) < feature_dim + 1:
            continue
        class_values = np.asarray(y_by_class[klass], dtype=np.float64)
        class_weights = None
        if strategy.gls:
            width = quantizer_cell_width(
                class_values, decimals=decimals, sig_digits=sig_digits
            )
            class_weights = 1.0 / np.maximum(width, 1e-300)
        theta = _weighted_lstsq(class_design, class_values, class_weights)
        recovered_head[klass] = theta[:feature_dim]
        recovered_bias[klass] = theta[feature_dim]
        solved[klass] = True

    recovered_delta = recovered_head - problem.target.base_head
    if strategy.rank_project is not None and solved.all():
        recovered_delta = _project_to_rank(recovered_delta, strategy.rank_project)
        recovered_head = problem.target.base_head + recovered_delta

    # Second stage: amplification improves the linear head but leaves the
    # intercept limited by the absolute score grid, because the constant
    # column of the augmented design does not scale with the queries. With
    # the head already recovered, a small unamplified probe set determines
    # the intercept from a well-conditioned scalar problem.
    if strategy.two_stage_bias > 0 and solved.all():
        n_bias = min(int(strategy.two_stage_bias), int(np.asarray(x_pool).shape[0]))
        x_bias = np.atleast_2d(np.asarray(x_pool, dtype=np.float64))[:n_bias]
        z_bias = feature_map(x_bias)
        residual_sums = np.zeros(n_classes, dtype=np.float64)
        residual_counts = np.zeros(n_classes, dtype=np.float64)
        for _ in range(repeats):
            if isinstance(oracle, RawLogitOracle):
                values = oracle.query(x_bias)
                observed = [
                    (i, klass, float(values[i, klass]))
                    for i in range(x_bias.shape[0])
                    for klass in range(n_classes)
                ]
            else:
                indices, scores = oracle.query(x_bias)
                assert scores is not None
                observed = [
                    (i, int(indices[i][pos]), float(scores[i, pos]))
                    for i in range(x_bias.shape[0])
                    for pos in range(indices.shape[1])
                ]
            for i, klass, score in observed:
                predicted = float(z_bias[i] @ recovered_head[klass])
                residual_sums[klass] += score - predicted
                residual_counts[klass] += 1.0
        seen = residual_counts > 0
        recovered_bias = np.where(seen, residual_sums / np.maximum(residual_counts, 1.0), recovered_bias)

    bias_abserr = float(np.linalg.norm(recovered_bias - problem.target.bias))

    return AdaptiveResult(
        strategy=strategy.name,
        description=strategy.describe(),
        oracle_calls=int(oracle.query_count),
        distinct_design_points=int(x_design.shape[0]),
        solved_classes=int(solved.sum()),
        delta_relerr=relative_frobenius_error(recovered_delta, problem.target.delta),
        bias_abserr=bias_abserr,
        test_agreement=label_agreement(
            recovered_head,
            recovered_bias,
            problem.target.head,
            problem.target.bias,
            problem.data.z_test,
        ),
        design_pinv_norm=pinv_norm,
        design_sigma_min=sigma_min,
        failure=bool(not solved.all()),
        extra={
            "mean_feature_norm": float(np.mean(np.linalg.norm(z_pool[selected], axis=1))),
            "class_design_rank": float(np.mean(class_ranks)) if class_ranks else 0.0,
            "required_rank": float(feature_dim + 1),
            # Charged inside oracle.query_count already; reported separately so
            # the magnitude-steering lever is never credited with a free probe.
            "probe_calls": float(probe_calls),
        },
    )
