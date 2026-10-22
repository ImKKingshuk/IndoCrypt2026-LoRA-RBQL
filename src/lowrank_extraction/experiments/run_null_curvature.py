"""How far a real LoRA module can move while staying invisible at the endpoint.

Section 8.3 exhibits a non-gauge direction in the finite-transcript nullspace of
the endpoint Jacobian, perturbs the *trained module parameters* along it, and
reports that the merged update moves while the endpoint logits barely do.  That
establishes the effect is realizable by an upstream module rather than only by a
hand-built hidden-state construction, but it is a single point: one step size,
one number.  This experiment measures the law it obeys.

The law is curvature.  A null direction kills the first-order term by
construction, so the leading endpoint drift is quadratic:

    max_x |f(theta_0 + t v)(x) - f(theta_0)(x)|  <=  (L/2) t^2,

with ``L`` a bound on the endpoint map's second derivative along ``v``.  The
merged update, by contrast, moves *linearly* in ``t``.  Two consequences follow,
and both are measurable rather than asserted:

  * against an oracle quantizing the logits, the two adapters are
    indistinguishable on the transcript for every ``t <= sqrt(2 gamma / L)``,
    where ``gamma`` is the smallest distance from a served logit to its cell's
    boundary.  It is ``gamma`` and not the cell width ``eps`` that appears,
    which matters: a logit sitting near a boundary is separated by a drift far
    below the cell width, so a radius written in terms of ``eps`` alone would
    claim invisibility the quantizer does not provide.  This is the same
    quantity Theorem 6 is stated in, and the two sections now agree;
  * the update movement bought at that radius therefore scales like
    ``sqrt(eps)`` -- halving the served precision buys the defender only a
    ``sqrt(2)`` reduction in how far the secret can hide.

This experiment measures the exponent and ``L`` on the trained checkpoints
already saved by ``run-lora-checkpoint``, so it retrains nothing.  A fitted
slope near 2 is the claim that the direction really is first-order null in the
*parameters*, not merely in a linearized surrogate.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from lowrank_extraction.experiments.run_lora_checkpoint import (
    _require_lora_deps,
    encode_texts,
    first_lora_delta,
    load_text_task,
)


def _jacobian_rows(model: Any, module: Any, inputs: dict[str, Any], torch: Any) -> np.ndarray:
    params = [module.lora_A["default"].weight, module.lora_B["default"].weight]
    model.eval()
    model.zero_grad(set_to_none=True)
    logits = model(**inputs).logits
    flat = logits.reshape(-1)
    rows = []
    for idx, scalar in enumerate(flat):
        model.zero_grad(set_to_none=True)
        scalar.backward(retain_graph=idx + 1 < int(flat.numel()))
        grads = []
        for param in params:
            g = param.grad
            grads.append(
                (torch.zeros_like(param) if g is None else g).detach().reshape(-1).clone()
            )
        rows.append(torch.cat(grads).cpu().numpy().astype(np.float64))
    return np.vstack(rows)


def _best_null_direction(
    jacobian: np.ndarray, a_np: np.ndarray, b_np: np.ndarray, scale: float,
    *, seed: int, candidates: int = 24,
) -> tuple[np.ndarray, float]:
    """A unit parameter direction that the transcript cannot see to first order.

    Chosen, among random candidates projected off the Jacobian row space, to
    move the merged update as far as possible -- the same selection rule as the
    single-point certificate, so the two are directly comparable.
    """

    rank = int(np.linalg.matrix_rank(jacobian, tol=1e-8))
    _, _, vh = np.linalg.svd(jacobian, full_matrices=False)
    row_basis = vh[:rank]
    rng = np.random.default_rng(seed)
    split = a_np.size
    best, best_shift = None, 0.0
    for _ in range(candidates):
        direction = rng.normal(0.0, 1.0, size=jacobian.shape[1])
        direction = direction - row_basis.T @ (row_basis @ direction)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            continue
        direction /= norm
        d_a = direction[:split].reshape(a_np.shape)
        d_b = direction[split:].reshape(b_np.shape)
        shift = float(np.linalg.norm(scale * (d_b @ a_np + b_np @ d_a)))
        if shift > best_shift:
            best, best_shift = direction, shift
    if best is None:
        raise RuntimeError("no usable null direction")
    return best, best_shift


def min_cell_slack(logit_sets: list[list[float]], cell_width: float) -> float:
    """Smallest distance from any served logit to its quantization cell's edge.

    This is the ``gamma`` of Theorem 6, and it is what a drift has to exceed to
    change a served value.  The cell width bounds it -- ``gamma <= eps/2`` always
    -- but nothing bounds it from below, so it has to be measured on the logits
    the endpoint actually returns rather than inferred from the grid.

    We take the minimum over every logit of every seed because the transcript
    reveals a difference as soon as *one* value moves cells.
    """

    values = np.concatenate([np.asarray(s, dtype=np.float64) for s in logit_sets])
    centre = np.round(values / cell_width) * cell_width
    return float(np.min(np.minimum(
        values - (centre - cell_width / 2.0), centre + cell_width / 2.0 - values
    )))


@dataclass
class Row:
    seed: int
    t: float
    max_logit_drift: float
    update_shift: float
    update_shift_ratio: float
    noise_floor: float


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("results") / "null_curvature")
    ap.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("results") / "lora_checkpoint_sst2_bert_base_stronger_10seed",
        help="directory holding adapter_seed_* and tokenizer_seed_* from run-lora-checkpoint",
    )
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--text-dataset", choices=["20newsgroups", "sst2"], default="sst2")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--train-size", type=int, default=1800)
    ap.add_argument("--test-size", type=int, default=500)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--examples", type=int, default=16)
    ap.add_argument(
        "--steps", type=float, nargs="+",
        default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0],
    )
    ap.add_argument(
        "--served-precisions", type=float, nargs="+",
        default=[1e-2, 1e-3, 1e-4, 1e-6],
    )
    args = ap.parse_args()

    deps = _require_lora_deps()
    torch = deps["torch"]
    from peft import PeftModel

    rows: list[Row] = []
    base_logit_sets: list[list[float]] = []
    for seed in args.seeds:
        _, _, test_texts, _, _ = load_text_task(
            deps=deps, text_dataset=args.text_dataset, categories=("sci.crypt", "comp.graphics"),
            seed=seed, train_size=args.train_size, test_size=args.test_size,
            offline_mini_corpus=False,
        )
        adapter_dir = args.checkpoint_dir / f"adapter_seed_{seed}"
        tokenizer_dir = args.checkpoint_dir / f"tokenizer_seed_{seed}"
        tokenizer = deps["AutoTokenizer"].from_pretrained(str(tokenizer_dir), use_fast=False)
        base = deps["AutoModelForSequenceClassification"].from_pretrained(
            args.model, num_labels=2
        )
        # is_trainable=True is required, not cosmetic: the saved adapter config
        # carries inference_mode=true, so a default load freezes lora_A/lora_B
        # and the Jacobian backward pass fails outright with "does not require
        # grad".  The classifier is restored either way -- it is in the
        # checkpoint's modules_to_save -- so the endpoint map C here is the
        # trained head, not a freshly initialised one.
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
        model.eval()
        # The whole measurement is a second-order effect: at t=1e-4 the drift we
        # are trying to see is around (L/2)t^2 ~ 1e-9, while float32 resolves
        # logits of order 1 only to ~1e-7.  In single precision the small-t end
        # of the sweep would therefore measure rounding noise, report a slope
        # near zero rather than near two, and -- worst -- corrupt L, which is
        # read off the smallest step.  Double precision puts the numerical floor
        # around 1e-16 and the effect back above it by orders.
        model.double()

        enc = encode_texts(tokenizer, test_texts, max_length=args.max_length)
        count = min(args.examples, int(enc["input_ids"].shape[0]))
        inputs = {k: v[:count] for k, v in enc.items()}

        module_name = str(first_lora_delta(model)["module_name"])
        module = dict(model.named_modules())[module_name]
        a_w = module.lora_A["default"].weight
        b_w = module.lora_B["default"].weight
        if not (a_w.requires_grad and b_w.requires_grad):
            raise RuntimeError(
                "LoRA factors are frozen, so the endpoint Jacobian cannot be "
                "taken; load the adapter with is_trainable=True"
            )
        # The endpoint map C must be the *trained* classifier.  It reaches the
        # model through modules_to_save rather than through the base checkpoint,
        # so a silent failure would leave a randomly initialised head here and
        # every drift number below would describe that head instead.
        classifier = dict(model.named_modules())[
            [n for n, _ in model.named_modules() if n.endswith("classifier")][0]
        ]
        if type(classifier).__name__ != "ModulesToSaveWrapper":
            raise RuntimeError(
                f"classifier is a {type(classifier).__name__}, not a PEFT "
                "ModulesToSaveWrapper: the trained head was not restored"
            )
        a_np = a_w.detach().cpu().numpy().astype(np.float64)
        b_np = b_w.detach().cpu().numpy().astype(np.float64)
        scale = float(module.scaling["default"])
        base_update_norm = float(np.linalg.norm(scale * (b_np @ a_np)))

        jac = _jacobian_rows(model, module, inputs, torch)
        direction, _ = _best_null_direction(
            jac, a_np, b_np, scale, seed=seed + 4242
        )
        split = a_np.size
        d_a = direction[:split].reshape(a_np.shape)
        d_b = direction[split:].reshape(b_np.shape)

        a_base = a_w.detach().clone()
        b_base = b_w.detach().clone()
        with torch.no_grad():
            base_logits = model(**inputs).logits.detach().cpu().numpy().astype(np.float64)

        # Establish the arithmetic noise floor before trusting any drift.  At
        # t=1e-12 the quadratic term is ~1e-24, far below double precision, so
        # whatever moves here is rounding, not curvature.  Any step whose drift
        # is not well clear of this number is measuring the machine.
        with torch.no_grad():
            a_w.copy_(a_base + 1e-12 * torch.tensor(d_a, dtype=a_base.dtype))
            b_w.copy_(b_base + 1e-12 * torch.tensor(d_b, dtype=b_base.dtype))
            probe = model(**inputs).logits.detach().cpu().numpy().astype(np.float64)
        noise_floor = float(np.max(np.abs(probe - base_logits)))
        base_logit_sets.append(base_logits.reshape(-1).tolist())

        for t in args.steps:
            with torch.no_grad():
                a_w.copy_(a_base + t * torch.tensor(d_a, dtype=a_base.dtype))
                b_w.copy_(b_base + t * torch.tensor(d_b, dtype=b_base.dtype))
                logits = model(**inputs).logits.detach().cpu().numpy().astype(np.float64)
            a_t = a_np + t * d_a
            b_t = b_np + t * d_b
            update_shift = float(
                np.linalg.norm(scale * (b_t @ a_t) - scale * (b_np @ a_np))
            )
            rows.append(
                Row(
                    seed=seed, t=float(t),
                    max_logit_drift=float(np.max(np.abs(logits - base_logits))),
                    update_shift=update_shift,
                    update_shift_ratio=update_shift / max(base_update_norm, 1e-12),
                    noise_floor=noise_floor,
                )
            )
        with torch.no_grad():
            a_w.copy_(a_base)
            b_w.copy_(b_base)
        del model, base

    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "null_curvature.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    steps = sorted({r.t for r in rows})
    drift = [float(np.median([r.max_logit_drift for r in rows if r.t == t])) for t in steps]
    shift = [float(np.median([r.update_shift for r in rows if r.t == t])) for t in steps]
    floor = float(np.median([r.noise_floor for r in rows]))

    # Fit only where the effect is unambiguously above the arithmetic noise.
    # Including a noise-dominated point would flatten the slope towards zero and
    # would be indistinguishable, in the fitted number alone, from the direction
    # failing to be null -- the opposite of what we would be claiming.
    usable = [i for i, dr in enumerate(drift) if dr > 10.0 * max(floor, 1e-300)]
    if len(usable) < 3:
        raise RuntimeError(
            f"only {len(usable)} of {len(steps)} steps clear the {floor:.2e} noise "
            "floor; widen --steps upward rather than fitting through noise"
        )
    fit_steps = [steps[i] for i in usable]
    drift_slope = float(
        np.polyfit(np.log10(fit_steps), np.log10([drift[i] for i in usable]), 1)[0]
    )
    shift_slope = float(
        np.polyfit(np.log10(fit_steps), np.log10([shift[i] for i in usable]), 1)[0]
    )
    # L from the smallest *usable* step: the quadratic term is cleanest there,
    # but only once that step is known to be signal rather than rounding.
    curvature = 2.0 * drift[usable[0]] / steps[usable[0]] ** 2
    # A median is not a bound.  The radius below is a claim about *every*
    # adapter we measured, so it has to be written in the largest curvature any
    # of them exhibits, over every usable step -- not in the typical one.
    per_point = [
        (int(r.seed), float(r.t), 2.0 * float(r.max_logit_drift) / float(r.t) ** 2)
        for r in rows
        if any(abs(r.t - steps[i]) <= 0.0 for i in usable)
    ]
    curvature_envelope = max(L for _, _, L in per_point)
    envelope_seed, envelope_t, _ = max(per_point, key=lambda x: x[2])
    # The envelope is only a bound if it actually bounds: check the quadratic
    # law at every measured point rather than asserting it from the fit.
    envelope_holds = all(
        float(r.max_logit_drift) <= 0.5 * curvature_envelope * float(r.t) ** 2
        * (1.0 + 1e-12)
        for r in rows
        if any(abs(r.t - steps[i]) <= 0.0 for i in usable)
    )

    lines = [
        f"seeds: {len(args.seeds)}, examples: {args.examples}, float64",
        "",
        f"arithmetic noise floor (drift at t=1e-12): {floor:.3e}.  Fitted over "
        f"the {len(usable)} of {len(steps)} steps whose drift clears 10x that, "
        f"t={fit_steps[0]:.0e} to {fit_steps[-1]:.0e}.",
        "",
        "| step t | median max logit drift | median update shift | shift / base |",
        "|---|---|---|---|",
    ]
    for i, (t, dr, sh) in enumerate(zip(steps, drift, shift)):
        ratio = float(np.median([r.update_shift_ratio for r in rows if r.t == t]))
        mark = "" if i in usable else "  (noise)"
        lines.append(f"| {t:.0e} | {dr:.3e}{mark} | {sh:.3e} | {ratio:.3e} |")
    lines += [
        "",
        f"log-log slope of endpoint drift in t: {drift_slope:.3f} (quadratic => 2)",
        f"log-log slope of update shift in t:   {shift_slope:.3f} (linear => 1)",
        f"implied curvature L = 2*drift/t^2 at t={steps[usable[0]]:.0e}: "
        f"{curvature:.4g} at the median seed, and {curvature_envelope:.4g} at "
        f"the worst (seed {envelope_seed}, t={envelope_t:.0e}), a factor "
        f"{curvature_envelope / curvature:.2f}.  The radius is written in the "
        f"worst one: a median curvature is not an upper bound, and a radius "
        f"derived from it would not hold for every adapter measured.  The "
        f"quadratic law with the envelope constant holds at every measured "
        f"point: {envelope_holds}.",
        "",
        "The radius below follows from the quadratic law, so it is only as good "
        "as that law.  Rows whose radius falls outside the range of t actually "
        "measured are marked: there the number is an extrapolation past the "
        "data, and the quadratic bound will eventually break as the endpoint "
        "map saturates.",
        "",
        "What sets the radius is the cell *slack* gamma -- the smallest distance "
        "from a served logit to its cell's boundary -- and not the cell width "
        "eps.  The two differ by more than a constant: a logit near a boundary "
        "is separated by a drift far below eps, so the eps radius overstates "
        "what the quantizer actually hides.  The eps column below is kept so "
        "the size of that gap is visible.",
        "",
        "| grid | cell width eps | min slack gamma | radius sqrt(2 gamma / L) | "
        "eps radius (overstated) | update shift there | within measured t? |",
        "|---|---|---|---|---|---|---|",
    ]
    t_min, t_max = fit_steps[0], fit_steps[-1]
    per_t = shift[usable[0]] / steps[usable[0]]
    radius_rows: list[dict[str, Any]] = []
    for eps in args.served_precisions:
        gamma = min_cell_slack(base_logit_sets, eps)
        radius = float(np.sqrt(2.0 * gamma / curvature_envelope))
        eps_radius = float(np.sqrt(2.0 * eps / curvature_envelope))
        inside = "yes" if t_min <= radius <= t_max else "NO (extrapolated)"
        lines.append(
            f"| absolute | {eps:.0e} | {gamma:.3e} | {radius:.3e} | "
            f"{eps_radius:.3e} | {per_t * radius:.3e} | {inside} |"
        )
        radius_rows.append({
            "cell_width": eps, "min_slack": gamma, "radius": radius,
            "eps_radius": eps_radius, "overstatement": eps_radius / radius,
            "update_shift": per_t * radius, "within_swept_t": inside == "yes",
        })
    (args.outdir / "summary.md").write_text("\n".join(lines) + "\n")
    # Written after the fit so the fitted quantities are data rather than
    # something a reader (or a test) has to parse back out of the prose.
    (args.outdir / "null_curvature.json").write_text(
        json.dumps(
            {
                "config": {"seeds": args.seeds, "steps": args.steps,
                           "examples": args.examples, "model": args.model,
                           "checkpoint_dir": str(args.checkpoint_dir),
                           "dtype": "float64"},
                "curvature_median": curvature,
                "curvature_envelope": curvature_envelope,
                "curvature_envelope_seed": envelope_seed,
                "curvature_envelope_ratio": curvature_envelope / curvature,
                "envelope_holds_at_every_measured_point": envelope_holds,
                "fit": {
                    "noise_floor": floor,
                    "fitted_steps": fit_steps,
                    "drift_slope": drift_slope,
                    "shift_slope": shift_slope,
                    # The constant the radius is written in, so that sqrt(2*gamma/L)
                    # reproduces the radius rows below; the median is kept
                    # beside it rather than in its place.
                    "curvature_L": curvature_envelope,
                    "curvature_L_median": curvature,
                    "invisibility_radius": {
                        f"{row['cell_width']:g}": row["radius"]
                        for row in radius_rows
                    },
                    "radius_rows": radius_rows,
                },
                "rows": [asdict(r) for r in rows],
            },
            indent=2,
        )
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
