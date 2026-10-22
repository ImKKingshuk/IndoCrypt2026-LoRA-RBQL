from __future__ import annotations

import numpy as np

from lowrank_extraction.adaptive import (
    AdaptiveStrategy,
    coverage_aware_rows,
    greedy_d_optimal_rows,
    run_adaptive_topk_recovery,
)
from lowrank_extraction.realdata import make_digits_adapter_problem


def _problem_and_pool(seed: int = 0, pool: int = 200):
    problem = make_digits_adapter_problem(seed=seed, dataset="digits", feature_dim=8, rank=2)
    rng = np.random.default_rng(seed)
    idx = rng.choice(problem.data.x_train.shape[0], size=pool, replace=True)
    return problem, problem.data.x_train[idx]


def test_greedy_d_optimal_improves_conditioning() -> None:
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(300, 6))
    selected = greedy_d_optimal_rows(pool, n_select=20)
    assert selected.shape == (20,)
    assert len(set(selected.tolist())) == 20
    chosen = np.linalg.svd(pool[selected], compute_uv=False)[-1]
    baseline = np.linalg.svd(pool[:20], compute_uv=False)[-1]
    assert chosen > baseline


def test_coverage_aware_selection_spreads_over_classes() -> None:
    problem, x_pool = _problem_and_pool()
    z = problem.data.feature_map(x_pool)
    aug = np.hstack([z, np.ones((z.shape[0], 1))])
    n_classes = problem.target.n_classes
    selected = coverage_aware_rows(
        aug,
        problem.target.base_head,
        np.zeros(n_classes),
        n_select=60,
        k=2,
    )
    assert selected.shape[0] == 60
    assert len(set(selected.tolist())) == 60
    # Every class bucket contributes, so the base-model argmax over the chosen
    # design is not concentrated on a single class.
    base_scores = z[selected] @ problem.target.base_head.T
    assert len(np.unique(np.argmax(base_scores, axis=1))) > 1


def test_exact_oracle_recovers_adapter_under_every_strategy() -> None:
    problem, x_pool = _problem_and_pool()
    for strategy in [
        AdaptiveStrategy(name="naive", design_size=120),
        AdaptiveStrategy(name="d_opt", d_optimal=True, design_size=120),
        AdaptiveStrategy(name="rank", rank_project=4, design_size=120),
    ]:
        result = run_adaptive_topk_recovery(
            problem, x_pool=x_pool, k=2, strategy=strategy, oracle_kind="raw"
        )
        assert not result.failure
        assert result.delta_relerr < 1e-9
        assert result.test_agreement == 1.0


def test_repeat_averaging_reduces_noise_error() -> None:
    problem, x_pool = _problem_and_pool()
    errors = []
    for repeats in (1, 64):
        result = run_adaptive_topk_recovery(
            problem,
            x_pool=x_pool,
            k=2,
            strategy=AdaptiveStrategy(name=f"r{repeats}", repeats=repeats, design_size=200),
            noise_std=0.1,
            seed=11,
            oracle_kind="raw",
        )
        errors.append(result.delta_relerr)
    # Independent additive noise averages down; 64 repeats should give a clear
    # improvement over a single pass.
    assert errors[1] < errors[0] / 3.0


def test_repeat_averaging_does_not_help_deterministic_rounding() -> None:
    problem, x_pool = _problem_and_pool()
    errors = []
    for repeats in (1, 64):
        result = run_adaptive_topk_recovery(
            problem,
            x_pool=x_pool,
            k=2,
            strategy=AdaptiveStrategy(name=f"r{repeats}", repeats=repeats, design_size=200),
            decimals=1,
            seed=11,
            oracle_kind="raw",
        )
        errors.append(result.delta_relerr)
    assert np.isclose(errors[0], errors[1])


def test_amplification_reduces_rounding_error() -> None:
    problem, x_pool = _problem_and_pool()
    errors = []
    for amplify in (1.0, 64.0):
        result = run_adaptive_topk_recovery(
            problem,
            x_pool=x_pool,
            k=2,
            strategy=AdaptiveStrategy(name=f"a{amplify}", amplify=amplify, design_size=200),
            decimals=1,
            seed=11,
            oracle_kind="raw",
        )
        errors.append(result.delta_relerr)
    # The stable-recovery bound is linear in ||X^+||_2, which amplification
    # shrinks; a 64x amplification should buy well over an order of magnitude.
    assert errors[1] < errors[0] / 10.0


def test_oracle_call_accounting_charges_repeats() -> None:
    problem, x_pool = _problem_and_pool()
    result = run_adaptive_topk_recovery(
        problem,
        x_pool=x_pool,
        k=2,
        strategy=AdaptiveStrategy(name="r8", repeats=8, design_size=50),
        oracle_kind="raw",
    )
    assert result.distinct_design_points == 50
    assert result.oracle_calls == 400
