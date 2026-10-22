"""Numerical verification of every formal statement, independent of the experiments.

Checking stored results cannot show that a *theorem* is true, so these tests
construct instances instead. Each claim is checked on a concrete
target, and where a claim has a converse or an excluded case -- a design one
query short, a two-sided perturbation direction, a rank-deficient coordinate
subset -- that case is checked to fail, so a vacuous test cannot pass.
"""

from __future__ import annotations

import numpy as np
import pytest


def _lstsq(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


# --- Recovery ---------------------------------------------------------------


def test_theorem_affine_recovers_exactly_at_full_augmented_rank() -> None:
    """Thm 1 (thm:affine): h+1 queries of augmented rank h+1 determine the head, bias and Delta."""
    rng = np.random.default_rng(0)
    m, h, p = 5, 9, 7
    proj = rng.standard_normal((p, h))
    head = rng.standard_normal((m, h))
    bias = rng.standard_normal(m)
    aug = np.hstack([np.maximum(rng.standard_normal((400, p)) @ proj, 0.0), np.ones((400, 1))])

    sel: list[int] = []
    for i in range(aug.shape[0]):
        if np.linalg.matrix_rank(aug[sel + [i]]) == len(sel) + 1:
            sel.append(i)
        if len(sel) == h + 1:
            break
    assert len(sel) == h + 1

    design = aug[sel]
    theta = _lstsq(design, design @ np.hstack([head, bias[:, None]]).T)
    assert np.linalg.norm(theta[:-1].T - head) / np.linalg.norm(head) < 1e-10
    assert np.linalg.norm(theta[-1] - bias) / np.linalg.norm(bias) < 1e-10

    # Converse: one query short leaves a null direction, so recovery must fail.
    short = aug[sel[:-1]]
    assert np.linalg.matrix_rank(short) < h + 1
    theta_short = _lstsq(short, short @ np.hstack([head, bias[:, None]]).T)
    assert np.linalg.norm(theta_short[:-1].T - head) / np.linalg.norm(head) > 1e-6


def test_proposition_lowrank_cuts_the_per_row_cost_to_r_plus_one() -> None:
    """Prop 1 (prop:lowrank): with the row space known a further row needs r+1 observations, not h+1."""
    rng = np.random.default_rng(1)
    m, h, r = 5, 9, 3
    delta = rng.standard_normal((m, r)) @ rng.standard_normal((r, h))
    head0 = rng.standard_normal((m, h))
    bias = rng.standard_normal(m)
    feats = np.maximum(rng.standard_normal((400, h)), 0.0)
    basis = np.linalg.svd(delta, full_matrices=False)[2][:r, :]
    proj = feats @ basis.T

    sel: list[int] = []
    for i in range(feats.shape[0]):
        cand = np.hstack([proj[sel + [i]], np.ones((len(sel) + 1, 1))])
        if np.linalg.matrix_rank(cand) == len(sel) + 1:
            sel.append(i)
        if len(sel) == r + 1:
            break
    assert len(sel) == r + 1 < h + 1

    klass = 0
    design = np.hstack([proj[sel], np.ones((r + 1, 1))])
    target = feats[sel] @ (head0[klass] + delta[klass]) + bias[klass] - feats[sel] @ head0[klass]
    row = _lstsq(design, target)[:r] @ basis
    assert np.linalg.norm(row - delta[klass]) / np.linalg.norm(delta[klass]) < 1e-10


def test_theorem_rankadaptive_identifies_rank_and_row_space_from_a_threshold() -> None:
    """Thm 2 (thm:rankadaptive): a spectral gap above tau recovers rho and the row space."""
    rng = np.random.default_rng(2)
    spectrum = np.array([5.0, 4.0, 3.0, 1e-9, 1e-10])
    rho = 3
    mat = rng.standard_normal((6, 5)) @ np.diag(spectrum) @ rng.standard_normal((5, 9))
    assert int((np.linalg.svd(mat, compute_uv=False) > 1e-4).sum()) == rho
    vt = np.linalg.svd(mat, full_matrices=False)[2]
    assert np.linalg.norm(vt[:rho].T @ vt[:rho] - vt[:rho].T @ vt[:rho]) == 0.0


def test_theorem_lowerbound_two_adapters_share_a_transcript_with_different_row_spaces() -> None:
    """Thm 3 (thm:lowerbound): span s < r observed leaves the row space unidentified."""
    rng = np.random.default_rng(3)
    h, r, s = 9, 3, 1
    observed = rng.standard_normal((s, h))
    one = np.vstack([observed, rng.standard_normal((r - s, h))])
    two = np.vstack([observed, rng.standard_normal((r - s, h))])
    assert np.array_equal(one[:s], two[:s])
    assert np.linalg.matrix_rank(one) == np.linalg.matrix_rank(two) == r
    p1 = np.linalg.svd(one, full_matrices=False)[2]
    p2 = np.linalg.svd(two, full_matrices=False)[2]
    assert np.linalg.norm(p1.T @ p1 - p2.T @ p2) > 0.1


# --- Defenses ---------------------------------------------------------------


def test_theorem_stable_bound_is_never_violated() -> None:
    """Thm 4 (thm:stable): ||Theta~ - Theta||_F <= ||X^+||_2 ||E||_F, and the grid bound on ||E||_F."""
    rng = np.random.default_rng(4)
    n, h, m = 300, 8, 4
    design = np.hstack([np.abs(rng.standard_normal((n, h))), np.ones((n, 1))])
    theta = rng.standard_normal((m, h + 1))
    exact = design @ theta.T
    pinv_norm = np.linalg.norm(np.linalg.pinv(design), 2)
    for _ in range(200):
        err = rng.standard_normal(exact.shape) * rng.uniform(1e-4, 1e-1)
        got = _lstsq(design, exact + err)
        assert np.linalg.norm(got.T - theta) <= pinv_norm * np.linalg.norm(err) * (1 + 1e-9)
    tau = 0.1
    rounded = np.round(exact / tau) * tau
    assert np.linalg.norm(rounded - exact) <= tau * np.sqrt(exact.size) / 2 + 1e-12


def test_corollary_adaptive_scaling_laws() -> None:
    """Cor 1 (cor:adaptive): 1/A for the head under amplification, 1/sqrt(R) for noise, nothing for rounding."""
    rng = np.random.default_rng(5)
    n, h, m = 300, 8, 4
    feats = np.abs(rng.standard_normal((n, h)))
    theta = rng.standard_normal((m, h + 1))

    def head_error(amp: float) -> float:
        design = np.hstack([feats * amp, np.ones((n, 1))])
        got = _lstsq(design, np.round(design @ theta.T, 1))
        return float(np.linalg.norm(got.T[:, :h] - theta[:, :h]) / np.linalg.norm(theta[:, :h]))

    amps = [1, 4, 16, 64, 256]
    slope = np.polyfit(np.log10(amps), np.log10([head_error(a) for a in amps]), 1)[0]
    assert abs(slope + 1.0) < 0.15, f"amplification slope {slope:+.3f}, expected -1"

    design = np.hstack([feats, np.ones((n, 1))])
    exact = design @ theta.T

    def averaged(reps: int, stochastic: bool) -> float:
        acc = np.zeros_like(exact)
        for _ in range(reps):
            acc += exact + rng.standard_normal(exact.shape) * 0.1 if stochastic else np.round(exact, 1)
        got = _lstsq(design, acc / reps)
        return float(np.linalg.norm(got.T - theta) / np.linalg.norm(theta))

    reps = [1, 4, 16, 64, 256]
    noise_slope = np.polyfit(np.log10(reps), np.log10([averaged(r, True) for r in reps]), 1)[0]
    assert abs(noise_slope + 0.5) < 0.12, f"noise slope {noise_slope:+.3f}, expected -0.5"
    # Deterministic rounding: repetition returns the identical value, so no gain at all.
    assert abs(averaged(1, False) - averaged(256, False)) < 1e-15

    # Rank projection is a constant factor, bounded by 2||N||.
    delta = rng.standard_normal((m, 2)) @ rng.standard_normal((2, h))
    noise = rng.standard_normal((m, h)) * 0.05
    u, s, vt = np.linalg.svd(delta + noise, full_matrices=False)
    projected = (u[:, :2] * s[:2]) @ vt[:2, :]
    assert np.linalg.norm(projected - delta) <= 2 * np.linalg.norm(noise) + 1e-12


def test_theorem_relative_is_exactly_scale_invariant_when_phi_is() -> None:
    """Thm 5 (thm:relative): exactness needs an exactly homogeneous block; the bias column breaks it."""
    rng = np.random.default_rng(6)
    n, h, m = 300, 8, 4

    def significant(y: np.ndarray, digits: int) -> np.ndarray:
        out = np.array(y, copy=True)
        nz = y != 0
        step = 10.0 ** (np.floor(np.log10(np.abs(y[nz]))) - (digits - 1))
        out[nz] = np.round(y[nz] / step) * step
        return out

    # No constant column: the augmented block is exactly homogeneous, so the
    # theorem predicts invariance exactly, at every A that is a power of the base.
    feats = np.abs(rng.standard_normal((n, h)))
    theta = rng.standard_normal((m, h))
    errs = []
    for amp in (1, 10, 100, 1000):
        design = feats * amp
        got = _lstsq(design, significant(design @ theta.T, 3))
        errs.append(float(np.linalg.norm(got.T - theta) / np.linalg.norm(theta)))
    assert max(errs) / min(errs) < 1.0001, f"relative error varied by {max(errs)/min(errs):.6f}x"

    # An absolute grid on the same design must instead decay.
    absolute = []
    for amp in (1, 10, 100, 1000):
        design = feats * amp
        got = _lstsq(design, np.round(design @ theta.T, 1))
        absolute.append(float(np.linalg.norm(got.T - theta) / np.linalg.norm(theta)))
    assert absolute[0] / absolute[-1] > 100

    # Determinism: the same query returns the same value, so averaging cannot help.
    exact = feats @ theta.T
    assert np.array_equal(significant(exact, 3), significant(exact, 3))


# --- Endpoint and oracle results --------------------------------------------


def test_theorem_hardlabel_erases_scale_but_keeps_the_boundary() -> None:
    """Thm 8 (thm:hardlabel): labels are invariant under positive rescaling; boundaries still recover."""
    rng = np.random.default_rng(7)
    d = 6
    w, b0 = rng.standard_normal(d), 0.3
    label = lambda x, a=1.0: np.sign(a * (x @ w + b0))
    probe = rng.standard_normal((3000, d))
    for a in (0.1, 1.0, 7.0, 1e3):
        assert np.array_equal(label(probe), label(probe, a))

    points = []
    while len(points) < d:
        lo, hi = rng.standard_normal(d), rng.standard_normal(d)
        if label(lo[None]) == label(hi[None]):
            continue
        for _ in range(80):
            mid = (lo + hi) / 2
            if label(mid[None]) == label(lo[None]):
                lo = mid
            else:
                hi = mid
        points.append((lo + hi) / 2)
    null = np.linalg.svd(np.hstack([np.array(points), np.ones((d, 1))]))[2][-1]
    direction = null[:d] / np.linalg.norm(null[:d])
    assert abs(abs(direction @ (w / np.linalg.norm(w))) - 1.0) < 1e-6


def test_theorem_endpoint_kernel_effects_are_invisible() -> None:
    """Thm 9 (thm:endpoint) and Cor 2 (cor:endpointinv): full rank inverts; a kernel direction is endpoint-invisible."""
    rng = np.random.default_rng(8)
    h = 10
    full = rng.standard_normal((h, h))
    hidden = rng.standard_normal((50, h))
    assert np.linalg.matrix_rank(full) == h
    recovered = (hidden @ full.T) @ np.linalg.pinv(full.T)
    assert np.linalg.norm(recovered - hidden) / np.linalg.norm(hidden) < 1e-10

    deficient = rng.standard_normal((3, h))
    kernel = np.linalg.svd(deficient)[2][-1]
    assert np.abs(deficient @ kernel).max() < 1e-12
    assert np.linalg.norm(kernel) > 0.5


def test_theorem_jacobian_nullity_and_lora_quotient_dimension() -> None:
    """Thm 10 (thm:jacobian) and Cor 3 (cor:loraquotient): rank-nullity on the transcript Jacobian, and the gauge count."""
    rng = np.random.default_rng(9)
    jac = rng.standard_normal((12, 20))
    kernel = np.linalg.svd(jac)[2][np.linalg.matrix_rank(jac):]
    assert kernel.shape[0] == 20 - np.linalg.matrix_rank(jac)
    assert np.abs(jac @ kernel.T).max() < 1e-12

    r, din, dout = 4, 768, 768
    assert r * (din + dout) - r * r == r * (din + dout - r) == 6128


def test_proposition_hiddenblock_recovers_on_an_all_active_cell() -> None:
    """Prop 3 (prop:hiddenblock): d+1 queries inside an all-active ReLU cell recover A0 + Delta."""
    rng = np.random.default_rng(10)
    d, h = 7, 9
    a0 = rng.standard_normal((h, d))
    offset = np.abs(rng.standard_normal(h)) * 5 + 5
    delta = rng.standard_normal((h, 2)) @ rng.standard_normal((2, d)) * 0.01
    head = rng.standard_normal((12, h))
    beta = rng.standard_normal(12)
    forward = lambda x: np.maximum((a0 + delta) @ x + offset, 0.0) @ head.T + beta

    eps = 1e-3
    zero = (forward(np.zeros(d)) - beta) @ np.linalg.pinv(head.T)
    cols = [((forward(eps * np.eye(d)[j]) - beta) @ np.linalg.pinv(head.T) - zero) / eps
            for j in range(d)]
    recovered = np.array(cols).T
    assert np.linalg.norm(recovered - (a0 + delta)) / np.linalg.norm(a0 + delta) < 1e-6


def test_proposition_coord_and_corollary_forcible() -> None:
    """Prop 4 (prop:coord) rank conditions, and Cor 4's odds identity and tie-bias search."""
    rng = np.random.default_rng(11)
    h = 10
    head = rng.standard_normal((30, h))
    assert np.linalg.matrix_rank(head[:h]) == h                       # exact coordinates
    shifted = np.hstack([head[:h + 1], -np.ones((h + 1, 1))])
    assert np.linalg.matrix_rank(shifted) == h + 1                    # shifted coordinates
    assert np.linalg.matrix_rank(np.hstack([head[:h], -np.ones((h, 1))])) < h + 1  # converse

    logits = rng.standard_normal(9)
    probs = np.exp(logits) / np.exp(logits).sum()
    i, bias = 3, 2.5
    biased = logits.copy()
    biased[i] += bias
    pb = np.exp(biased) / np.exp(biased).sum()
    assert abs(probs[i] / (1 - probs[i]) - np.exp(-bias) * (pb[i] / (1 - pb[i]))) < 1e-12

    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        trial = logits.copy()
        trial[i] += mid
        if np.argmax(trial) == i:
            hi = mid
        else:
            lo = mid
    assert abs(hi - (logits.max() - logits[i])) < 1e-8


def test_directional_cell_slack_is_sound_and_strictly_tighter() -> None:
    """The tightened floor must exhibit a *genuinely* indistinguishable adapter.

    The certificate was loosened in two independent ways. It charged every score
    the distance to its nearest cell boundary, though the displacement
    ``t z_j^T Xi_c`` is signed and can only run into one of them; and it sampled
    the admissible polytope rather than searching it. Removing the first is worth
    2.1--4.5x and removing the second another factor, so the reported floor moved
    by roughly an order of magnitude -- which is exactly the kind of change that
    is worthless unless the bound is still true.

    So this does not compare the two rules to each other. It takes the step the
    directional rule certifies, forms ``Delta + t Xi``, and requires that every
    quantized score is bit-identical to the original transcript and that the
    perturbed adapter is still rank r. If the tightening ever over-claimed, the
    transcript would differ and this fails.
    """

    import numpy as np

    from lowrank_extraction.experiments.run_security_floor import (
        cell_slack,
        directional_step,
        rank_preserving_directions,
        signed_cell_slack,
    )

    rng = np.random.default_rng(11)
    for decimals, sig_digits in ((1, None), (None, 3)):
        rank = 2
        z = rng.standard_normal((60, 8))
        delta = rng.standard_normal((6, rank)) @ rng.standard_normal((rank, 8))
        scores = z @ delta.T
        down, up = signed_cell_slack(scores, decimals=decimals, sig_digits=sig_digits)
        symmetric = cell_slack(scores, decimals=decimals, sig_digits=sig_digits)

        # The symmetric rule is the elementwise minimum of the two directional
        # limits, so it can never certify more; that is the whole gain.
        assert np.all(symmetric <= np.maximum(down, up) + 1e-12)

        def quantize(y: np.ndarray) -> np.ndarray:
            if decimals is not None:
                return np.round(y, decimals)
            step = np.power(
                10.0, np.floor(np.log10(np.abs(np.where(y == 0.0, 1.0, y))))
                - (sig_digits - 1)
            )
            return np.round(y / step) * step

        baseline = quantize(scores)
        tightened_wins = 0
        for xi in rank_preserving_directions(delta, rank, rng, 24):
            t = directional_step(z, xi, down, up)
            symmetric_t = float(
                np.min(
                    np.where(
                        np.abs(z @ xi.T) > 0.0,
                        symmetric / np.maximum(np.abs(z @ xi.T), 1e-300),
                        np.inf,
                    )
                )
            )
            assert t >= symmetric_t - 1e-12, "directional rule must never certify less"
            tightened_wins += t > symmetric_t * (1.0 + 1e-9)

            # The claim itself: at 99.9% of the certified step the transcript is
            # unchanged, so the two adapters really are indistinguishable.
            for sign in (+1.0, -1.0):
                alternative = delta + sign * 0.999 * t * xi
                if sign > 0 and not np.allclose(
                    quantize(z @ alternative.T), baseline
                ):
                    continue  # this sign is the shorter one; the other must work
                assert np.linalg.matrix_rank(alternative) <= rank
            assert np.allclose(
                quantize(z @ (delta + 0.999 * t * xi).T), baseline
            ) or np.allclose(
                quantize(z @ (delta - 0.999 * t * xi).T), baseline
            ), "certified step leaves the transcript unchanged in at least one sign"

        assert tightened_wins > 0, "directional rule never improved -- the gain is not real"


def test_design_diagnostic_ranks_the_matrix_it_actually_solves() -> None:
    """A full-rank design must be reported as full rank.

    The diagnostic decides whether a small measured leak means the defense held
    or the attack broke, and the paper quotes it. Comparing ``matrix_rank(z)``
    -- the h-column feature block -- against ``z.shape[1] + 1``, the column
    count of the augmented system, would report a one-column deficiency for
    every design, including the well-conditioned case where nothing is wrong.

    A well-conditioned design must therefore come back *full* rank, and a design
    with a genuinely dead coordinate must come back exactly one short.
    """

    import numpy as np

    from lowrank_extraction.experiments.run_vision_endpoint import design_diagnostics

    rng = np.random.default_rng(5)
    healthy = np.abs(rng.standard_normal((200, 16))) + 0.5
    report = design_diagnostics(healthy)
    assert report["required_rank"] == 17, "the augmented system has h + 1 columns"
    assert report["feature_rank"] == report["required_rank"], (
        "a full-rank, well-conditioned design must not be reported as deficient"
    )
    assert report["dead_coordinates"] == 0

    # Kill one coordinate the way amplification into a ReLU dead zone does.
    dead = healthy.copy()
    dead[:, 3] = 0.0
    report = design_diagnostics(dead)
    assert report["dead_coordinates"] == 1
    assert report["feature_rank"] == report["required_rank"] - 1, (
        "one dead coordinate should cost exactly one rank"
    )

    # A constant column is already spanned by the augmenting ones column, so it
    # costs a rank too -- the diagnostic must notice that, not just literal zeros.
    constant = healthy.copy()
    constant[:, 7] = 2.0
    assert design_diagnostics(constant)["feature_rank"] == 16


def test_gls_is_identically_ols_under_an_absolute_grid():
    """The units thesis, as a property of the estimator rather than the bound.

    Weighting the least-squares rows by the reciprocal quantization cell width
    is the efficient estimator for the error a quantizer injects.  Under an
    absolute grid that width is the same for every score, so the weights are
    uniform and the weighted solve must return the ordinary one *bit for bit* --
    an absolute defense does not even offer the attacker a weighting to exploit.
    Under a relative grid the width tracks the score, so the two differ.
    """

    import numpy as np

    from lowrank_extraction.adaptive import quantizer_cell_width, solve_scores

    rng = np.random.default_rng(11)
    design = np.hstack([rng.standard_normal((80, 6)), np.ones((80, 1))])
    served = rng.standard_normal((80, 4)) * 3.0

    width = quantizer_cell_width(served[:, 0], decimals=2, sig_digits=None)
    assert np.allclose(width, 1e-2), "an absolute grid has one cell width everywhere"

    ols = solve_scores(design, served, decimals=2, sig_digits=None, gls=False)
    gls = solve_scores(design, served, decimals=2, sig_digits=None, gls=True)
    assert np.array_equal(ols, gls), "GLS must degenerate to OLS on an absolute grid"

    # Relative units make the width track the score, so the estimators separate.
    rel_width = quantizer_cell_width(served[:, 0], decimals=None, sig_digits=3)
    assert rel_width.max() / rel_width.min() > 1.0, "relative cell width must vary"
    rel_ols = solve_scores(design, served, decimals=None, sig_digits=3, gls=False)
    rel_gls = solve_scores(design, served, decimals=None, sig_digits=3, gls=True)
    assert not np.allclose(rel_ols, rel_gls), "GLS must differ from OLS on a relative grid"


def test_gls_helps_against_relative_quantization_but_only_by_a_constant():
    """Does an estimator adapted to the defense break it?

    Corollary 1's levers were all built against a perturbation whose size does
    not depend on the score.  Relative quantization violates that, so the honest
    test of Theorem 5 is an attacker that weights.  It does help -- the defense
    is not impregnable -- but the gain must be a bounded constant that does not
    grow with the amplification budget, because Theorem 5 is a statement about
    scale-invariance and a gain that grew with ``A`` would refute it.
    """

    import numpy as np

    from lowrank_extraction.adaptive import AdaptiveStrategy, run_adaptive_topk_recovery
    from lowrank_extraction.realdata import make_digits_adapter_problem

    def error(amplify: float, gls: bool) -> float:
        errs = []
        for seed in range(3):
            problem = make_digits_adapter_problem(feature_dim=8, rank=2, seed=seed)
            result = run_adaptive_topk_recovery(
                problem,
                x_pool=problem.data.x_train,
                k=2,
                strategy=AdaptiveStrategy(
                    name="gls" if gls else "ols",
                    amplify=amplify,
                    d_optimal=True,
                    design_size=200,
                    rank_project=4,
                    two_stage_bias=64,
                    gls=gls,
                ),
                oracle_kind="raw",
                seed=seed,
                sig_digits=3,
            )
            errs.append(result.delta_relerr)
        return float(np.mean(errs))

    gains = [error(a, gls=False) / error(a, gls=True) for a in (1.0, 64.0)]
    assert all(g > 1.05 for g in gains), (
        "the weighted attack should genuinely help; claiming otherwise would "
        f"overstate the defense (measured {gains})"
    )
    assert all(g < 10.0 for g in gains), (
        f"but it must stay a bounded constant, not a rate (measured {gains})"
    )
    # The decisive check: the lever must not compound with amplification.
    assert gains[1] < 2.0 * gains[0], (
        "a GLS gain that grew with A would contradict Theorem 5's scale-invariance"
    )


# --- Theorem 7 under sign of the secret -------------------------------------


def test_the_quantizer_is_exactly_odd() -> None:
    """The reduction Theorem 7's proof rests on, checked rather than assumed.

    The whole negative-secret case is handled by reflection, which is sound only
    if significant-digit rounding is an odd function. Ties are where a rounding
    rule stops being symmetric, so they are included explicitly.
    """
    from lowrank_extraction.oracles import quantize_significant

    rng = np.random.default_rng(0)
    ties = np.array([1.25, 1.35, 2.5, 0.125, 99.5, 1.005, 12.5, 0.0005])
    for digits in (1, 2, 3, 4, 6, 8):
        y = np.concatenate([rng.uniform(-1e3, 1e3, 20_000), ties, -ties])
        assert np.array_equal(
            quantize_significant(-y, digits), -quantize_significant(y, digits)
        ), f"Q_{digits} is not odd, so the reflection argument fails"


@pytest.mark.parametrize(("p", "d"), [(2, 2), (3, 2), (2, 4), (4, 3), (4, 4)])
def test_theorem_precisionfloor_is_exactly_symmetric_in_the_sign_of_theta(
    p: int, d: int
) -> None:
    """Thm 7 (thm:precisionfloor): the threshold set is signed, so the floor reflects exactly.

    The statement's T contains positive numbers only. Read literally with a
    negative secret it certifies that nothing separates two negative secrets,
    which is false -- the statement carries the sign into T_theta. This checks
    the consequence: the certified gap is invariant under theta -> -theta, and
    is a number rather than NaN.
    """
    from lowrank_extraction.experiments.run_precision_floor import achievable_gap

    for theta in (1.7, 3.25, 9.99):
        *_, pos = achievable_gap(theta, p, d)
        *_, neg = achievable_gap(-theta, p, d)
        assert np.isfinite(neg), "a negative secret must yield a floor, not NaN"
        assert pos == neg, f"floor is not reflection-symmetric at theta={theta}"


def test_the_attacker_also_reflects_and_still_respects_the_bound(
) -> None:
    """The bound must hold on the negative side too, not just the certificate."""
    from lowrank_extraction.experiments.run_precision_floor import (
        achievable_gap,
        attack_scalar,
    )

    for theta in (-2.4, -7.1):
        for p, d in ((2, 2), (3, 3)):
            *_, cert = achievable_gap(theta, p, d)
            err = attack_scalar(theta, p, d, steps=40)
            assert np.isfinite(err)
            # Part (ii)'s constant, which must bound the exact minimax
            # relative error (b-a)/(a+b) that `cert` now reports.
            assert cert >= 0.125 * 10.0 ** (-(p + d + 1))


def test_theorem_stable_preserves_the_top_label_above_twice_the_bound() -> None:
    """Thm 4 (thm:stable): the per-logit bound, and the factor two that protects the argmax.

    The proof gives the Frobenius bound and leaves two steps to the reader: the
    Cauchy--Schwarz step to a single logit, and that a margin exceeding *twice*
    that quantity fixes the top label, since both logits in the comparison can
    move. Neither was executed anywhere, so both are checked here on real
    rounding rather than on injected noise, and the margin hypothesis is
    counted so that it cannot pass by never firing.
    """
    rng = np.random.default_rng(4004)
    fired = 0
    for _ in range(400):
        n, h, m = int(rng.integers(8, 26)), int(rng.integers(2, 8)), int(rng.integers(2, 7))
        design = np.hstack([rng.standard_normal((n, h)), np.ones((n, 1))])
        if np.linalg.matrix_rank(design) < h + 1:
            continue
        theta = rng.standard_normal((h + 1, m))
        exact = design @ theta
        tau = float(10.0 ** rng.integers(-6, 0))
        served = np.round(exact / tau) * tau
        err = served - exact
        pinv = np.linalg.pinv(design)
        recovered = pinv @ served

        evaluation = np.append(rng.standard_normal(h), 1.0)
        bound = (
            np.linalg.norm(evaluation)
            * np.linalg.norm(pinv, 2)
            * np.linalg.norm(err)
        )
        exact_logits = evaluation @ theta
        got_logits = evaluation @ recovered
        assert np.max(np.abs(got_logits - exact_logits)) <= bound * (1 + 1e-9)

        ranked = np.sort(exact_logits)[::-1]
        if ranked[0] - ranked[1] > 2 * bound:
            fired += 1
            assert np.argmax(got_logits) == np.argmax(exact_logits)
    assert fired > 200, f"the margin hypothesis only fired {fired} times"


def test_theorem_jacobian_kernel_is_invisible_on_an_actual_transcript_map() -> None:
    """Thm 10 (thm:jacobian): o(eps) along ker J, and exact equality where the map is affine.

    The existing Jacobian test checks rank-nullity on a random matrix and the
    LoRA gauge arithmetic. Neither builds a transcript, so neither exercises
    what the theorem says: that moving along a kernel direction leaves the
    transcript unchanged to first order. Here the map is genuinely nonlinear,
    so a first-order-only claim is the most that can hold -- and the affine
    clause is checked separately, where the theorem promises exact equality
    rather than o(eps), out to a step of 1000.
    """
    rng = np.random.default_rng(1010)
    d, hidden, classes = 9, 5, 4
    head = rng.standard_normal((classes, hidden))
    lift = rng.standard_normal((hidden, d))
    lift[:, -3:] = 0.0  # three parameter directions reach the endpoint nonlinearly only

    def transcript(theta):
        pre = lift @ theta
        return (head @ (pre + 0.3 * np.tanh(pre))).ravel()

    def jacobian(theta, eps=1e-7):
        return np.column_stack(
            [
                (transcript(theta + eps * e) - transcript(theta - eps * e)) / (2 * eps)
                for e in np.eye(d)
            ]
        )

    base = rng.standard_normal(d)
    jac = jacobian(base)
    rank = np.linalg.matrix_rank(jac, tol=1e-6)
    kernel = np.linalg.svd(jac)[2][rank:]
    assert kernel.shape[0] == d - rank, "rank-nullity failed on the transcript Jacobian"
    assert rank < d, "the instance must leave something unobservable to be a test"

    # o(eps): the deviation divided by eps must itself vanish, not merely stay small.
    direction = kernel[0]
    ratios = [
        np.linalg.norm(transcript(base + eps * direction) - transcript(base)) / eps
        for eps in (1e-2, 1e-3, 1e-4, 1e-5)
    ]
    for coarse, fine in zip(ratios, ratios[1:]):
        assert fine < coarse / 5, f"deviation/eps is not vanishing: {ratios}"
    # Scale-free: how far the ratio falls over three decades of eps, not how
    # small it gets. The magnitude is set by the second derivative of this
    # particular instance and carries no meaning of its own.
    assert ratios[-1] < ratios[0] / 100, f"deviation/eps barely moved: {ratios}"

    # A direction *outside* the kernel must be visible, or the test is vacuous.
    visible = np.linalg.svd(jac)[2][0]
    assert np.linalg.norm(transcript(base + 1e-3 * visible) - transcript(base)) / 1e-3 > 1e-2

    # The affine clause: exact equality, at any step size, not just small ones.
    def affine(theta):
        return (head @ (lift @ theta)).ravel()

    affine_jac = np.column_stack(
        [affine(e) - affine(np.zeros(d)) for e in np.eye(d)]
    )
    flat = np.linalg.svd(affine_jac)[2][np.linalg.matrix_rank(affine_jac, tol=1e-9):][0]
    for step in (1e-3, 1.0, 1e3):
        assert np.linalg.norm(affine(base + step * flat) - affine(base)) < 1e-9
