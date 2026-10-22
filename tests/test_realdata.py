import numpy as np

from lowrank_extraction.experiments.run_realdata import hidden_block_markdown_summary, markdown_summary, run_case, run_hidden_block_realdata_case
from lowrank_extraction.experiments.run_defense_sweep import run_sweep, summarize_rows as summarize_defense_rows
from lowrank_extraction.experiments.run_endpoint_transformer import summarize_rows as summarize_endpoint_rows
from lowrank_extraction.experiments.run_lora_checkpoint import summarize_jacobian_sweep, summarize_rows as summarize_lora_rows
from lowrank_extraction.experiments.run_lm_endpoint_inversion import (
    _adaptive_coordinate_recovery,
    _coordinate_exposure_sweep,
    _forced_top1_biased_token_log_probs,
    _rank_diagnostics,
    _top1_index_bias_shifted_logits,
    summarize_rows as summarize_lm_endpoint_rows,
)
from lowrank_extraction.experiments.run_mlm_output_head import (
    SparseCoordinateOutputHeadOracle,
    _recover_output_head_lowrank_coordinate,
    _recover_output_head_unknown_rank_coordinate,
    summarize_rows as summarize_mlm_rows,
)
from lowrank_extraction.experiments.run_ablation import run_unknown_rank_case, summarize_unknown_rank
from lowrank_extraction.endpoint_observability import (
    classifier_quotient_certificate,
    invert_endpoint_hidden_states,
    invert_hidden_states_from_log_probs,
    invert_hidden_states_from_shifted_logits,
    linear_jacobian_certificate,
    nullspace_drift_certificate,
    unbiased_log_probs_from_biased_token_log_probs,
)
from lowrank_extraction.lower_bounds import make_partial_topk_indistinguishable_pair
from lowrank_extraction.realdata import load_digits_public_features, make_digits_adapter_problem


def test_digits_public_features_are_deterministic() -> None:
    first = load_digits_public_features(seed=11, feature_dim=6)
    second = load_digits_public_features(seed=11, feature_dim=6)

    assert np.allclose(first.x_train, second.x_train)
    assert np.array_equal(first.y_train, second.y_train)
    assert np.allclose(first.z_train, second.z_train)
    assert first.feature_map.input_dim == 64
    assert first.feature_map.feature_dim == 6


def test_digits_adapter_problem_has_low_rank_delta() -> None:
    problem = make_digits_adapter_problem(seed=12, feature_dim=6, rank=2)

    assert problem.target.n_classes == 3
    assert problem.target.rank == 2
    assert np.linalg.matrix_rank(problem.target.delta) <= 2
    assert 0.0 <= problem.rank_energy <= 1.0


def test_realdata_case_schema_and_partial_topk_failure() -> None:
    row = run_case(seed=13, feature_dim=6, rank=2, topk_queries=300, boundary_attempts=8_000)

    required = {
        "raw_test_agreement",
        "random_topk_test_agreement",
        "data_topk_test_agreement",
        "hard_label_test_agreement",
        "full_ranking_test_agreement",
        "partial_ranking_failure",
        "surrogate_test_agreement",
    }
    assert required.issubset(row)
    assert row["raw_test_agreement"] == 1.0
    assert isinstance(row["partial_ranking_failure"], bool)


def test_realdata_markdown_summary_contains_statistics() -> None:
    rows = [
        run_case(seed=14, feature_dim=6, rank=2, topk_queries=300, boundary_attempts=8_000),
        run_case(seed=15, feature_dim=6, rank=2, topk_queries=300, boundary_attempts=8_000),
    ]
    summary = markdown_summary(rows)

    assert "Real-Data Adapter Summary" in summary
    assert "digits public-feature adapted head" in summary
    assert "mean" in summary
    assert "std" in summary
    assert "Partial-ranking failure rate" in summary


def test_realdata_hidden_lora_block_case_schema() -> None:
    row = run_hidden_block_realdata_case(seed=18, rank=2)

    assert row["dataset"] == "digits"
    assert row["query_count"] == 65
    assert row["test_all_active_rate"] == 1.0
    assert row["test_agreement"] == 1.0
    assert float(row["hidden_delta_relerr"]) < 1e-9


def test_realdata_hidden_lora_markdown_summary_contains_statistics() -> None:
    rows = [
        run_hidden_block_realdata_case(seed=19, rank=2),
        run_hidden_block_realdata_case(seed=20, rank=2),
    ]
    summary = hidden_block_markdown_summary(rows)

    assert "Real-Data Hidden LoRA Block Summary" in summary
    assert "Test all-active rate" in summary
    assert "pretrained transformer LoRA checkpoint" in summary


def test_wine_realdata_case_smoke() -> None:
    row = run_case(seed=16, dataset="wine", feature_dim=6, rank=2, topk_queries=300, boundary_attempts=8_000)

    assert row["dataset"] == "wine"
    assert row["raw_test_agreement"] == 1.0
    assert row["classes"] == 3


def test_defense_sweep_reports_utility_and_leakage() -> None:
    rows = run_sweep(
        seeds=[21],
        datasets=["digits"],
        feature_dim=6,
        rank=2,
        topk_queries=250,
        round_decimals=[1],
        noise_stds=[0.05],
        endpoint_json=None,
        lora_json=None,
    )
    apis = {str(row["api"]) for row in rows}
    summary = summarize_defense_rows(rows)

    assert "raw_logits" in apis
    assert "topk_index_only" in apis
    assert "scored_topk_rounded" in apis
    assert "Utility-vs-Leakage Defense Sweep" in summary
    assert "Scoreless hard-label and top-k-index APIs preserve prediction service" in summary
    assert any(row["delta_relerr"] is None and row["failure_rate"] == 1.0 for row in rows)


def test_partial_topk_indistinguishability_certificate() -> None:
    target_a, target_b, _, result = make_partial_topk_indistinguishable_pair(seed=17)

    assert result.transcript_equal
    assert result.parameter_distance > 0.0
    assert result.delta_rank_a <= 1
    assert result.delta_rank_b <= 1
    assert result.certified_domain_gap > 0.0
    assert not np.allclose(target_a.delta, target_b.delta)


def test_endpoint_observability_certificates_quantify_blind_space() -> None:
    classifier = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hidden = np.array([[0.5, 0.25, 1.0], [0.1, 0.8, -0.3]])
    quotient = classifier_quotient_certificate(classifier)
    drift = nullspace_drift_certificate(classifier, np.zeros(2), hidden, seed=123)
    jacobian = linear_jacobian_certificate(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    assert quotient.rank == 2
    assert quotient.nullity == 1
    assert np.isclose(quotient.blind_fraction, 1 / 3)
    assert drift["endpoint_nonidentifiable"] is True
    assert float(drift["endpoint_max_logit_drift"]) < 1e-12
    assert jacobian.rank == 1
    assert jacobian.nullity_lower_bound == 2


def test_endpoint_hidden_state_inversion_requires_injective_output_head() -> None:
    output_weight = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.5],
            [-0.5, 0.25, 1.0],
        ]
    )
    output_bias = np.array([0.2, -0.1, 0.3, 0.0, 0.4])
    hidden = np.array([[0.5, -0.2, 1.0], [1.5, 0.25, -0.75]])
    logits = hidden @ output_weight.T + output_bias

    recovered = invert_endpoint_hidden_states(output_weight, output_bias, logits)

    assert np.allclose(recovered, hidden)


def test_endpoint_hidden_state_inversion_rejects_classifier_quotient() -> None:
    output_weight = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    output_bias = np.zeros(2)
    logits = np.zeros((4, 2))

    try:
        invert_endpoint_hidden_states(output_weight, output_bias, logits)
    except ValueError as exc:
        assert "not injective" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("rank-deficient endpoint head should not be invertible")


def test_logit_bias_top1_log_probs_recover_hidden_states() -> None:
    output_weight = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.5],
            [-0.5, 0.25, 1.0],
        ]
    )
    output_bias = np.array([0.2, -0.1, 0.3, 0.0, 0.4])
    hidden = np.array([[0.5, -0.2, 1.0], [1.5, 0.25, -0.75]])
    logits = hidden @ output_weight.T + output_bias
    bias_strength = 8.0
    biased_token_log_probs = np.empty_like(logits)
    for klass in range(output_weight.shape[0]):
        biased_logits = logits.copy()
        biased_logits[:, klass] += bias_strength
        shifted = biased_logits - np.max(biased_logits, axis=1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        biased_token_log_probs[:, klass] = log_probs[:, klass]

    recovered_log_probs = unbiased_log_probs_from_biased_token_log_probs(
        biased_token_log_probs,
        bias=bias_strength,
    )
    recovered_hidden = invert_hidden_states_from_log_probs(output_weight, output_bias, recovered_log_probs)

    assert np.allclose(recovered_hidden, hidden)


def test_forced_top1_oracle_loop_matches_biased_softmax_diagonal() -> None:
    logits = np.array([[0.1, -0.4, 1.2], [2.0, -1.0, 0.5]], dtype=np.float64)
    bias_strength = 7.0
    observed = _forced_top1_biased_token_log_probs(logits=logits, bias_strength=bias_strength)
    expected = np.empty_like(logits)
    for klass in range(logits.shape[1]):
        biased = logits.copy()
        biased[:, klass] += bias_strength
        shifted = biased - np.max(biased, axis=1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        expected[:, klass] = log_probs[:, klass]

    assert np.allclose(observed, expected)


def test_top1_index_bias_binary_search_recovers_shifted_logits() -> None:
    logits = np.array([[2.0, 0.5, -1.0], [0.1, 3.0, 1.5]], dtype=np.float64)
    shifted, calls, uncovered = _top1_index_bias_shifted_logits(
        logits=logits,
        bias_range=8.0,
        binary_steps=30,
    )
    base = np.max(logits, axis=1, keepdims=True)

    assert calls == logits.shape[0] * (logits.shape[1] - 1) * 30
    assert uncovered == 0.0
    assert np.allclose(shifted, logits - base, atol=1e-7)


def test_shifted_logit_inversion_recovers_hidden_states() -> None:
    output_weight = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.5],
            [-0.5, 0.25, 1.0],
        ]
    )
    output_bias = np.array([0.2, -0.1, 0.3, 0.0, 0.4])
    hidden = np.array([[0.5, -0.2, 1.0], [1.5, 0.25, -0.75]])
    logits = hidden @ output_weight.T + output_bias
    shifted = logits - np.max(logits, axis=1, keepdims=True)

    recovered = invert_hidden_states_from_shifted_logits(output_weight, output_bias, shifted)

    assert np.allclose(recovered, hidden)


def test_coordinate_exposure_sweep_reports_rank_thresholds() -> None:
    output_weight = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ],
        dtype=np.float64,
    )
    hidden = np.array([[0.5, -0.25], [1.0, 0.75]], dtype=np.float64)
    logits = hidden @ output_weight.T

    row = _coordinate_exposure_sweep(
        output_weight=output_weight,
        logits=logits,
        true_hidden=hidden,
        row_budgets=[1, 2, 3, 4],
        seed=0,
    )

    assert row["coordinate_exact_min_full_rank_budget"] >= 2
    assert row["coordinate_shifted_min_full_rank_budget"] >= 3
    assert float(row["coordinate_exact_hidden_relerr_at_min"]) < 1e-10
    assert float(row["coordinate_shifted_hidden_relerr_at_min"]) < 1e-10


def test_unknown_rank_summary_reports_failure_modes() -> None:
    rows = [run_unknown_rank_case(seed=0), run_unknown_rank_case(seed=1)]
    summary = summarize_unknown_rank(rows)

    assert "Unknown-Rank Recovery Diagnostic" in summary
    assert "exact logits reveal the numerical rank" in summary
    assert "Rounding and noise can inflate small singular values" in summary
    assert all(row["exact_rank_success"] for row in rows)


def test_jacobian_sweep_summary_reports_query_budget_scaling() -> None:
    summary = summarize_jacobian_sweep(
        [
            {
                "model": "bert-base-uncased",
                "dataset": "glue_sst2",
                "module_name": "bert.encoder.layer.0.attention.self.query",
                "endpoint_jacobian_examples": 1,
                "endpoint_jacobian_rows": 2,
                "endpoint_jacobian_quotient_dimension": 6128,
                "endpoint_jacobian_rank": 2,
                "endpoint_jacobian_nullity_lower_bound": 6142,
                "endpoint_jacobian_nongauge_nullity_lower_bound": 6126,
                "endpoint_jacobian_observable_fraction": 2 / 6144,
                "endpoint_jacobian_quotient_observable_fraction": 2 / 6128,
                "endpoint_null_direction_available": True,
                "endpoint_null_direction_update_shift_norm": 1e-5,
                "endpoint_null_direction_max_logit_drift": 1e-7,
                "endpoint_null_direction_label_flip_rate": 0.0,
            },
            {
                "model": "bert-base-uncased",
                "dataset": "glue_sst2",
                "module_name": "bert.encoder.layer.0.attention.self.query",
                "endpoint_jacobian_examples": 2,
                "endpoint_jacobian_rows": 4,
                "endpoint_jacobian_quotient_dimension": 6128,
                "endpoint_jacobian_rank": 4,
                "endpoint_jacobian_nullity_lower_bound": 6140,
                "endpoint_jacobian_nongauge_nullity_lower_bound": 6124,
                "endpoint_jacobian_observable_fraction": 4 / 6144,
                "endpoint_jacobian_quotient_observable_fraction": 4 / 6128,
                "endpoint_null_direction_available": True,
                "endpoint_null_direction_update_shift_norm": 2e-5,
                "endpoint_null_direction_max_logit_drift": 2e-7,
                "endpoint_null_direction_label_flip_rate": 0.0,
            },
        ]
    )

    assert "Endpoint LoRA Jacobian Sweep" in summary
    assert "finite-query local observability ceiling" in summary
    assert "non-gauge nullity" in summary
    assert "Local endpoint-null perturbation" in summary
    assert "| 2 |" in summary


def test_lora_checkpoint_summary_is_honest_about_scope() -> None:
    summary = summarize_lora_rows(
        [
            {
                "model": "google/bert_uncased_L-2_H-128_A-2",
                "dataset": "20newsgroups",
                "module_name": "bert.encoder.layer.0.attention.self.query",
                "train_accuracy": 0.75,
                "test_accuracy": 0.70,
                "module_delta_relerr": 1e-12,
                "module_rank_projected_relerr": 1e-12,
                "module_query_count": 129,
                "endpoint_nullity": 126,
                "endpoint_hidden_blind_fraction": 0.984375,
                "endpoint_max_logit_drift": 1e-14,
                "endpoint_real_max_logit_drift": 1e-14,
                "endpoint_real_label_flip_rate": 0.0,
                "endpoint_real_hidden_shift_norm": 0.8,
                "endpoint_real_hidden_shift_ratio": 0.2,
                "endpoint_null_direction_update_shift_norm": 1e-5,
                "endpoint_null_direction_max_logit_drift": 1e-7,
                "endpoint_null_direction_label_flip_rate": 0.0,
            }
        ]
    )

    assert "PEFT LoRA Checkpoint Summary" in summary
    assert "true PEFT LoRA adapter" in summary
    assert "not an end-to-end black-box recovery attack" in summary
    assert "nontrivial nullspace" in summary
    assert "held-out encoded test examples" in summary
    assert "Local endpoint-null perturbation" in summary


def test_endpoint_transformer_summary_states_positive_scope() -> None:
    summary = summarize_endpoint_rows(
        [
            {
                "model": "google/bert_uncased_L-2_H-128_A-2",
                "dataset": "20newsgroups",
                "train_accuracy": 0.82,
                "test_accuracy": 0.76,
                "endpoint_delta_relerr": 1e-12,
                "endpoint_rank_projected_relerr": 1e-12,
                "endpoint_bias_relerr": 1e-12,
                "endpoint_logit_max_abs_error": 1e-12,
                "endpoint_label_agreement": 1.0,
                "rounded_endpoint_delta_relerr": 1e-3,
                "rounded_endpoint_label_agreement": 1.0,
                "noisy_endpoint_delta_relerr": 1e-2,
                "noisy_endpoint_label_agreement": 0.99,
                "query_count": 129,
                "feature_rank": 129,
            }
        ]
    )

    assert "Endpoint Transformer LoRA-Head Summary" in summary
    assert "positive endpoint-only transformer experiment" in summary
    assert "final logits" in summary
    assert "rounded and noisy final-logit" in summary
    assert "does not contradict the hidden-adapter nullspace result" in summary


def test_lm_endpoint_summary_states_full_vocab_scope() -> None:
    summary = summarize_lm_endpoint_rows(
        [
            {
                "model": "distilgpt2",
                "seed": 0,
                "hidden_dim": 768,
                "vocab_size": 50257,
                "adapter_feature_dim": 16,
                "rank": 4,
                "unknown_rank_max_rank": 8,
                "unknown_rank_tol": 1e-3,
                "output_head_rank": 768,
                "output_head_min_singular": 0.17,
                "output_head_condition_number": 2700.0,
                "softmax_augmented_rank": 769,
                "softmax_augmented_condition_number": 3900.0,
                "public_feature_rank": 16,
                "full_logit_hidden_relerr": 1e-9,
                "top1_bias_hidden_relerr": 1e-9,
                "top1_index_bias_hidden_relerr": 1e-7,
                "full_logit_delta_relerr": 1e-5,
                "top1_bias_delta_relerr": 1e-5,
                "top1_index_bias_delta_relerr": 1e-3,
                "full_logit_unknown_rank_estimate": 4,
                "top1_bias_unknown_rank_estimate": 4,
                "top1_index_bias_unknown_rank_estimate": 4,
                "full_logit_unknown_rank_spectral_gap": 1e6,
                "top1_bias_unknown_rank_spectral_gap": 1e6,
                "top1_index_bias_unknown_rank_spectral_gap": 1e3,
                "full_logit_rank_projected_delta_relerr": 1e-5,
                "top1_bias_rank_projected_delta_relerr": 1e-5,
                "top1_index_bias_rank_projected_delta_relerr": 1e-3,
                "full_logit_eval_max_logit_error": 1e-8,
                "top1_bias_eval_max_logit_error": 1e-8,
                "top1_index_bias_eval_max_logit_error": 1e-6,
                "full_logit_eval_top1_agreement": 1.0,
                "top1_bias_eval_top1_agreement": 1.0,
                "top1_index_bias_eval_top1_agreement": 1.0,
                "full_logit_endpoint_queries": 28,
                "top1_bias_endpoint_calls": 1_407_196,
                "top1_index_bias_endpoint_calls": 33_770_000,
                "top1_index_bias_uncovered_fraction": 0.0,
                "coordinate_exact_min_full_rank_budget": 768,
                "coordinate_shifted_min_full_rank_budget": 769,
                "coordinate_exact_hidden_relerr_at_min": 1e-8,
                "coordinate_shifted_hidden_relerr_at_min": 1e-7,
                "coordinate_exact_condition_at_min": 1e4,
                "coordinate_shifted_condition_at_min": 1e4,
                "adaptive_exact_rows": 768,
                "adaptive_shifted_rows": 769,
                "adaptive_exact_condition": 1.4e4,
                "adaptive_shifted_condition": 2.6e4,
                "adaptive_exact_hidden_relerr": 1e-8,
                "adaptive_top1_bias_hidden_relerr": 1e-8,
                "adaptive_top1_index_bias_hidden_relerr": 1e-6,
                "adaptive_exact_delta_relerr": 1e-5,
                "adaptive_top1_bias_delta_relerr": 1e-5,
                "adaptive_top1_index_bias_delta_relerr": 1e-3,
                "adaptive_exact_endpoint_calls": 21_504,
                "adaptive_top1_bias_endpoint_calls": 21_532,
                "adaptive_top1_index_bias_endpoint_calls": 516_768,
                "adaptive_top1_index_bias_uncovered_fraction": 0.0,
            }
        ]
    )

    assert "real pretrained full-vocabulary endpoint-head experiment" in summary
    assert "rank upper bound" in summary
    assert "unknown-rank rows estimate the numerical rank" in summary
    assert "top-1/logit-bias log-prob interface" in summary
    assert "scoreless oracle reconstructs relative logits" in summary
    assert "coordinate-exposure rows report" in summary
    assert "adaptive-coordinate rows select vocabulary rows" in summary
    assert "not a claim that low-dimensional classifier endpoints or arbitrary hidden transformer LoRA modules" in summary


def test_adaptive_coordinate_recovery_uses_hidden_dim_rows() -> None:
    rng = np.random.default_rng(12)
    hidden_dim = 3
    vocab_size = 8
    feature_dim = 2
    output_weight = rng.normal(size=(vocab_size, hidden_dim))
    hidden = rng.normal(size=(5, hidden_dim))
    public_features = rng.normal(size=(5, feature_dim))
    delta = rng.normal(scale=0.01, size=(hidden_dim, feature_dim))
    adapted_hidden = hidden + public_features @ delta.T
    logits = adapted_hidden @ output_weight.T

    row = _adaptive_coordinate_recovery(
        output_weight=output_weight,
        logits=logits,
        true_hidden=adapted_hidden,
        base_hidden=hidden,
        public_features=public_features,
        delta=delta,
        bias_strength=8.0,
        bias_range=80.0,
        binary_steps=24,
    )

    assert row["adaptive_exact_rows"] == hidden_dim
    assert row["adaptive_shifted_rows"] == hidden_dim + 1
    assert float(row["adaptive_exact_hidden_relerr"]) < 1e-10
    assert float(row["adaptive_top1_bias_hidden_relerr"]) < 1e-10
    assert float(row["adaptive_top1_index_bias_hidden_relerr"]) < 1e-5


def test_lm_rank_diagnostics_estimates_low_rank_update() -> None:
    rng = np.random.default_rng(76)
    left = rng.normal(size=(7, 3))
    right = rng.normal(size=(5, 3))
    delta = left @ right.T
    recovered = delta + rng.normal(scale=1e-10, size=delta.shape)

    diag = _rank_diagnostics(recovered, delta, max_rank=6, rank_tol=1e-5)

    assert diag["rank_estimate"] == 3
    assert float(diag["rank_spectral_gap"]) > 1e5
    assert float(diag["rank_projected_delta_relerr"]) < 1e-8


def test_mlm_output_head_summary_states_observable_scope() -> None:
    summary = summarize_mlm_rows(
        [
            {
                "model": "bert-base-uncased",
                "seed": 0,
                "hidden_dim": 768,
                "vocab_size": 30522,
                "output_head_rank": 768,
                "output_head_min_singular": 1.7,
                "output_head_condition_number": 99.0,
                "output_head_injective": True,
                "rank": 4,
                "query_count": 900,
                "test_count": 128,
                "augmented_feature_rank": 769,
                "output_delta_relerr": 1e-7,
                "output_bias_relerr": 1e-10,
                "heldout_max_logit_error": 1e-9,
                "heldout_top1_agreement": 1.0,
                "coordinate_basis_rows": 4,
                "coordinate_basis_attempts": 4,
                "coordinate_row_space_rank": 4,
                "coordinate_basis_queries_per_row": 769,
                "coordinate_lowrank_queries_per_remaining_row": 5,
                "coordinate_oracle_calls": 30_522,
                "coordinate_total_scalar_observations": 155_666,
                "coordinate_full_logit_equivalent_scalars": 27_469_800,
                "coordinate_observation_reduction": 176.5,
                "coordinate_output_delta_relerr": 1e-7,
                "coordinate_heldout_max_logit_error": 1e-9,
                "coordinate_heldout_top1_agreement": 1.0,
                "unknown_rank_estimate": 4,
                "unknown_rank_basis_rows": 10,
                "unknown_rank_spectral_gap": 1e8,
                "unknown_rank_total_scalar_observations": 160_250,
                "unknown_rank_observation_reduction": 171.4,
                "unknown_rank_output_delta_relerr": 1e-7,
                "unknown_rank_heldout_top1_agreement": 1.0,
                "adaptive_exact_rows": 768,
                "adaptive_shifted_rows": 769,
                "adaptive_exact_condition": 1.7e3,
                "adaptive_shifted_condition": 2.6e3,
                "adaptive_exact_hidden_relerr": 1e-8,
                "adaptive_shifted_hidden_relerr": 1e-8,
                "adaptive_exact_delta_relerr": 1e-5,
                "adaptive_shifted_delta_relerr": 1e-5,
                "adaptive_exact_endpoint_calls": 691_200,
                "adaptive_shifted_endpoint_calls": 692_100,
            }
        ]
    )

    assert "real full-vocabulary masked-LM endpoint experiment" in summary
    assert "low-rank vocabulary-output-head update" in summary
    assert "explicit sparse coordinate-score oracle" in summary
    assert "not ordinary full-logit endpoint calls" in summary
    assert "Unknown-rank coordinate recovery" in summary
    assert "does not need the true rank" in summary
    assert "Low-rank coordinate output-head recovery" in summary
    assert "Adaptive-coordinate endpoint inversion" in summary
    assert "not arbitrary hidden-transformer LoRA recovery" in summary


def test_mlm_lowrank_coordinate_recovery_smoke() -> None:
    rng = np.random.default_rng(123)
    hidden_dim = 4
    vocab_size = 10
    rank = 2
    z_query = rng.normal(size=(8, hidden_dim))
    while np.linalg.matrix_rank(np.hstack([z_query[: hidden_dim + 1], np.ones((hidden_dim + 1, 1))])) < hidden_dim + 1:
        z_query = rng.normal(size=(8, hidden_dim))
    z_test = rng.normal(size=(5, hidden_dim))
    base_weight = rng.normal(size=(vocab_size, hidden_dim))
    base_bias = rng.normal(size=vocab_size)
    left = rng.normal(size=(vocab_size, rank))
    right = rng.normal(size=(rank, hidden_dim))
    delta = left @ right

    row = _recover_output_head_lowrank_coordinate(
        z_query=z_query,
        z_test=z_test,
        base_weight=base_weight,
        base_bias=base_bias,
        delta=delta,
        rank=rank,
        block_size=4,
    )

    assert row["coordinate_row_space_rank"] == rank
    assert row["coordinate_oracle_calls"] == vocab_size
    assert row["coordinate_total_scalar_observations"] == rank * (hidden_dim + 1) + (vocab_size - rank) * (rank + 1)
    assert float(row["coordinate_output_delta_relerr"]) < 1e-10
    assert float(row["coordinate_heldout_max_logit_error"]) < 1e-10
    assert row["coordinate_heldout_top1_agreement"] == 1.0


def test_mlm_unknown_rank_coordinate_recovery_smoke() -> None:
    rng = np.random.default_rng(456)
    hidden_dim = 4
    vocab_size = 12
    rank = 2
    z_query = rng.normal(size=(8, hidden_dim))
    while np.linalg.matrix_rank(np.hstack([z_query[: hidden_dim + 1], np.ones((hidden_dim + 1, 1))])) < hidden_dim + 1:
        z_query = rng.normal(size=(8, hidden_dim))
    z_test = rng.normal(size=(5, hidden_dim))
    base_weight = rng.normal(size=(vocab_size, hidden_dim))
    base_bias = rng.normal(size=vocab_size)
    left = rng.normal(size=(vocab_size, rank))
    right = rng.normal(size=(rank, hidden_dim))
    delta = left @ right

    row = _recover_output_head_unknown_rank_coordinate(
        z_query=z_query,
        z_test=z_test,
        base_weight=base_weight,
        base_bias=base_bias,
        delta=delta,
        max_rank=4,
        rank_tol=1e-6,
        block_size=4,
    )

    assert row["unknown_rank_estimate"] == rank
    assert row["unknown_rank_basis_rows"] == 6
    assert float(row["unknown_rank_output_delta_relerr"]) < 1e-10
    assert row["unknown_rank_heldout_top1_agreement"] == 1.0


def test_sparse_coordinate_output_head_oracle_counts_transcript() -> None:
    rng = np.random.default_rng(321)
    z_query = rng.normal(size=(6, 3))
    weight = rng.normal(size=(7, 3))
    bias = rng.normal(size=7)
    oracle = SparseCoordinateOutputHeadOracle(z_query, weight, bias)

    query_indices = np.asarray([0, 2, 5])
    row_indices = np.asarray([1, 4])
    observed = oracle.query(query_indices, row_indices)
    expected = z_query[query_indices] @ weight[row_indices].T + bias[row_indices]

    assert np.allclose(observed, expected)
    assert oracle.call_count == 1
    assert oracle.scalar_observations == 6
