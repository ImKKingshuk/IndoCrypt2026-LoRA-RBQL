import numpy as np

from lowrank_extraction.attacks import (
    predict_feature_boundaries,
    predict_multiclass_boundaries,
    predict_ranking_boundaries,
    recover_binary_hard_label_boundary,
    recover_feature_hard_label_boundaries,
    recover_feature_rank_index_boundaries,
    recover_hidden_lora_block_on_positive_cell,
    recover_lowrank_assisted_feature_head_from_observations,
    recover_multiclass_hard_label_boundaries,
    recover_raw_affine,
    recover_raw_feature_head,
    recover_topk_scored_feature_head,
    recover_topk_scored_affine,
)
from lowrank_extraction.metrics import binary_boundary_agreement, label_agreement, relative_frobenius_error
from lowrank_extraction.models import LinearLowRankTarget, make_hidden_lora_block_target, make_linear_lowrank_target, make_relu_feature_lowrank_target, random_probe_points
from lowrank_extraction.oracles import HardLabelOracle, RawLogitOracle, TopKOracle
from lowrank_extraction.endpoint_observability import invert_endpoint_hidden_states


def test_raw_affine_recovery_is_exact_for_clean_logits() -> None:
    target = make_linear_lowrank_target(n_features=16, n_classes=5, rank=2, seed=123)
    oracle = RawLogitOracle(target)

    result = recover_raw_affine(oracle, base_weight=target.base_weight, rank=target.rank)

    assert result.query_count == target.n_features + 1
    assert np.allclose(result.recovered_weight, target.weight)
    assert np.allclose(result.recovered_bias, target.bias)
    assert relative_frobenius_error(result.recovered_delta, target.delta) < 1e-12


def test_rank_projection_recovers_exact_low_rank_delta() -> None:
    target = make_linear_lowrank_target(n_features=24, n_classes=9, rank=3, seed=456)
    oracle = RawLogitOracle(target)

    result = recover_raw_affine(oracle, base_weight=target.base_weight, rank=target.rank)

    assert relative_frobenius_error(result.rank_projected_delta, target.delta) < 1e-12


def test_recovered_model_has_identical_labels_on_random_points() -> None:
    target = make_linear_lowrank_target(n_features=12, n_classes=4, rank=2, seed=789)
    oracle = RawLogitOracle(target)
    result = recover_raw_affine(oracle, base_weight=target.base_weight, rank=target.rank)
    x = random_probe_points(2048, target.n_features, seed=111)

    agreement = label_agreement(result.recovered_weight, result.recovered_bias, target.weight, target.bias, x)

    assert agreement == 1.0


def test_scored_full_topk_recovery_is_exact() -> None:
    target = make_linear_lowrank_target(n_features=10, n_classes=6, rank=2, seed=321)
    oracle = TopKOracle(target, k=target.n_classes, include_scores=True)

    result = recover_topk_scored_affine(oracle, base_weight=target.base_weight, rank=target.rank)

    assert result.query_count == target.n_features + 1
    assert result.solved_classes.all()
    assert np.allclose(result.recovered_weight, target.weight)
    assert np.allclose(result.recovered_bias, target.bias)


def test_binary_hard_label_boundary_recovery_is_functionally_equivalent() -> None:
    target = make_linear_lowrank_target(n_features=8, n_classes=2, rank=1, seed=654)
    oracle = HardLabelOracle(target)

    result = recover_binary_hard_label_boundary(
        oracle,
        n_features=target.n_features,
        seed=987,
        validation_points=2048,
    )
    x = random_probe_points(4096, target.n_features, seed=222, scale=2.5)

    assert result.solved
    assert result.boundary_points.shape == (target.n_features, target.n_features)
    assert binary_boundary_agreement(result.normal, result.offset, target.weight, target.bias, x) > 0.995


def test_multiclass_hard_label_boundaries_are_functionally_equivalent() -> None:
    target = make_linear_lowrank_target(n_features=4, n_classes=3, rank=1, seed=765)
    oracle = HardLabelOracle(target)

    result = recover_multiclass_hard_label_boundaries(
        oracle,
        n_features=target.n_features,
        n_classes=target.n_classes,
        seed=876,
        max_attempts=80_000,
        validation_points=2048,
    )
    x = random_probe_points(4096, target.n_features, seed=333, scale=2.5)
    pred = predict_multiclass_boundaries(result, x)
    truth = target.labels(x)

    assert result.solved_pairs.sum() == target.n_classes * (target.n_classes - 1)
    assert np.mean(pred == truth) > 0.99


def test_raw_relu_feature_head_recovery_is_exact() -> None:
    target = make_relu_feature_lowrank_target(
        n_features=5,
        hidden_features=9,
        n_classes=4,
        rank=2,
        seed=4321,
    )
    oracle = RawLogitOracle(target)

    result = recover_raw_feature_head(
        oracle,
        base_head=target.base_head,
        feature_map=target.features,
        input_dim=target.n_features,
        feature_dim=target.hidden_features,
        rank=target.rank,
        seed=1234,
    )
    x = random_probe_points(2048, target.n_features, seed=444)

    assert result.feature_rank == target.hidden_features + 1
    assert np.allclose(result.recovered_head, target.head)
    assert np.allclose(result.recovered_bias, target.bias)
    assert relative_frobenius_error(result.recovered_delta, target.delta) < 1e-11
    assert label_agreement(result.recovered_head, result.recovered_bias, target.head, target.bias, target.features(x)) == 1.0


def test_scored_topk_relu_feature_head_recovery_is_exact() -> None:
    target = make_relu_feature_lowrank_target(
        n_features=5,
        hidden_features=7,
        n_classes=4,
        rank=2,
        seed=5432,
    )
    oracle = TopKOracle(target, k=2, include_scores=True)

    result = recover_topk_scored_feature_head(
        oracle,
        base_head=target.base_head,
        feature_map=target.features,
        input_dim=target.n_features,
        feature_dim=target.hidden_features,
        rank=target.rank,
        random_queries=2000,
        seed=2345,
    )
    x = random_probe_points(2048, target.n_features, seed=555)

    assert result.solved_classes.all()
    assert relative_frobenius_error(result.recovered_delta, target.delta) < 1e-10
    assert label_agreement(result.recovered_head, result.recovered_bias, target.head, target.bias, target.features(x)) == 1.0


def test_hidden_lora_block_recovery_on_positive_relu_cell_is_exact() -> None:
    target = make_hidden_lora_block_target(
        n_features=6,
        hidden_features=4,
        n_classes=5,
        rank=2,
        seed=2468,
        hidden_bias=4.0,
    )
    inversion_oracle = RawLogitOracle(target)
    query_x = np.vstack([np.zeros(target.n_features), 1e-2 * np.eye(target.n_features)])
    endpoint_logits = inversion_oracle.query(query_x)
    true_hidden = np.maximum(0.0, query_x @ target.hidden_weight.T + target.hidden_bias)
    inverted_hidden = invert_endpoint_hidden_states(target.output_head, target.output_bias, endpoint_logits)

    oracle = RawLogitOracle(target)
    result = recover_hidden_lora_block_on_positive_cell(
        oracle,
        base_hidden_weight=target.base_hidden_weight,
        hidden_bias=target.hidden_bias,
        output_head=target.output_head,
        output_bias=target.output_bias,
        rank=target.rank,
    )
    x = random_probe_points(1024, target.n_features, seed=1357, scale=0.05)
    recovered_hidden = np.maximum(0.0, x @ result.recovered_hidden_weight.T + result.recovered_hidden_bias)
    recovered_logits = recovered_hidden @ target.output_head.T + target.output_bias

    assert result.query_count == target.n_features + 1
    assert result.output_head_rank == target.hidden_features
    assert result.activation_margin > 0.0
    assert relative_frobenius_error(inverted_hidden, true_hidden) < 1e-12
    assert relative_frobenius_error(result.recovered_delta, target.delta) < 1e-10
    assert np.allclose(result.recovered_hidden_bias, target.hidden_bias)
    assert np.mean(np.argmax(recovered_logits, axis=1) == target.labels(x)) == 1.0


def test_hard_label_relu_feature_boundaries_are_functionally_equivalent() -> None:
    target = make_relu_feature_lowrank_target(
        n_features=5,
        hidden_features=4,
        n_classes=3,
        rank=1,
        seed=6543,
    )
    oracle = HardLabelOracle(target)

    result = recover_feature_hard_label_boundaries(
        oracle,
        feature_map=target.features,
        input_dim=target.n_features,
        feature_dim=target.hidden_features,
        n_classes=target.n_classes,
        seed=3456,
        max_attempts=120_000,
        validation_points=2048,
    )
    x = random_probe_points(4096, target.n_features, seed=666, scale=2.5)
    pred = predict_feature_boundaries(result, target.features(x))
    truth = target.labels(x)

    assert result.solved_pairs.sum() == target.n_classes * (target.n_classes - 1)
    assert np.mean(pred == truth) > 0.99


def test_index_only_ranking_relu_feature_boundaries_are_functionally_equivalent() -> None:
    target = make_relu_feature_lowrank_target(
        n_features=5,
        hidden_features=4,
        n_classes=3,
        rank=1,
        seed=7654,
    )
    oracle = TopKOracle(target, k=target.n_classes, include_scores=False)

    result = recover_feature_rank_index_boundaries(
        oracle,
        feature_map=target.features,
        input_dim=target.n_features,
        feature_dim=target.hidden_features,
        n_classes=target.n_classes,
        seed=4567,
        max_attempts=120_000,
        validation_points=2048,
    )
    x = random_probe_points(4096, target.n_features, seed=777, scale=2.5)
    pred = predict_ranking_boundaries(result, target.features(x))
    truth = target.labels(x)

    assert result.solved_pairs.sum() == target.n_classes * (target.n_classes - 1)
    assert np.mean(pred == truth) > 0.99


def test_observable_partial_topk_index_only_boundaries_are_functionally_equivalent() -> None:
    weights = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ],
        dtype=np.float64,
    )
    target = LinearLowRankTarget(
        base_weight=weights,
        adapter_left=np.zeros((3, 1), dtype=np.float64),
        adapter_right=np.zeros((2, 1), dtype=np.float64),
        bias=np.zeros(3, dtype=np.float64),
    )
    oracle = TopKOracle(target, k=2, include_scores=False)

    result = recover_feature_rank_index_boundaries(
        oracle,
        feature_map=lambda x: np.atleast_2d(np.asarray(x, dtype=np.float64)),
        input_dim=2,
        feature_dim=2,
        n_classes=3,
        seed=5678,
        max_attempts=20_000,
        validation_points=2048,
    )
    x = random_probe_points(4096, 2, seed=888, scale=2.5)
    pred = predict_ranking_boundaries(result, x)
    truth = target.labels(x)

    assert result.solved_pairs.sum() == target.n_classes * (target.n_classes - 1)
    assert np.mean(pred == truth) > 0.99


def test_lowrank_assisted_topk_recovers_rows_with_rank_plus_one_observations() -> None:
    target = make_relu_feature_lowrank_target(
        n_features=6,
        hidden_features=10,
        n_classes=6,
        rank=2,
        seed=200000,
    )
    rng = np.random.default_rng(210000)
    x = rng.normal(0.0, 2.0, size=(6000, target.n_features))
    z = target.features(x)
    indices, scores = TopKOracle(target, k=3, include_scores=True).query(x)
    assert scores is not None

    full_budget = target.hidden_features + 1
    lowrank_budget = target.rank + 1
    budgets = {klass: (full_budget if klass in {0, 1} else lowrank_budget) for klass in range(target.n_classes)}
    used = {klass: 0 for klass in range(target.n_classes)}
    observations = []
    for row_idx in range(x.shape[0]):
        for pos, klass_raw in enumerate(indices[row_idx]):
            klass = int(klass_raw)
            if used[klass] >= budgets[klass]:
                continue
            observations.append((klass, z[row_idx], float(scores[row_idx, pos])))
            used[klass] += 1
        if all(used[klass] >= budgets[klass] for klass in range(target.n_classes)):
            break

    result = recover_lowrank_assisted_feature_head_from_observations(
        observations,
        base_head=target.base_head,
        rank=target.rank,
    )
    eval_x = random_probe_points(2048, target.n_features, seed=9999)

    assert result.full_row_solved_classes.sum() >= target.rank
    assert result.solved_classes.all()
    assert np.sum(result.observations_per_class >= target.hidden_features + 1) < target.n_classes
    assert relative_frobenius_error(result.recovered_delta, target.delta) < 1e-10
    assert label_agreement(result.recovered_head, result.recovered_bias, target.head, target.bias, target.features(eval_x)) == 1.0
