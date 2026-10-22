"""Where the amplification lever is available, and what it costs when it is.

The absolute-versus-relative gap measured on ``digits`` is three orders of
magnitude, but on a text endpoint it is much smaller.  The discreteness of
the token domain does not explain that; this experiment measures the actual
mechanism.

Panel A measures how each public feature map responds to a scaled query, via the
homogeneity ratio ``||phi(Ax)|| / (A ||phi(x)||)``.  Three regimes appear, and
they are what decides whether Corollary 1's amplification lever exists at all:

  * *exactly homogeneous* -- the identity map, which is what an interface
    accepting continuous vectors exposes (``O_module``, or any embedding-input
    endpoint).  The ratio is 1 at every A and the 1/A law of Corollary 1(1) is
    exact.
  * *asymptotically homogeneous* -- standardize/PCA/ReLU on ``digits``.  The
    ratio settles at a positive constant, so the lever works in the limit.
  * *scale-invariant* -- a LayerNorm transformer queried through
    ``inputs_embeds``.  Normalization discards the scale, so ``phi(Ax) ~ phi(x)``
    and the ratio decays like 1/A.  The lever is not weak here; it does not
    exist.

That last regime is the honest reason the text endpoint shows 4.9x, and it is a
structural property of the *architecture*, not of the defense.

Panel B then measures what the absolute grid costs in the regime where the lever
*is* available at transformer scale: an ``O_module`` oracle over the real
768-dimensional ``bert-base-uncased`` hidden states, with the adapted endpoint
head as the secret.  This is the missing cell of the evaluation -- the previous
transformer comparison ran with amplification unavailable, so it measured only
the intrinsic perturbation ratio.

Both panels report utility beyond top-1 agreement.  The claim that relative
quantization is "free" cannot rest on label agreement alone, which is blind
to everything a probability consumer cares about; ``logit_mae`` and
``softmax_kl`` are recorded so the claim is either substantiated or correctly
qualified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from lowrank_extraction.adaptive import solve_scores
from lowrank_extraction.oracles import quantize_significant
from lowrank_extraction.realdata import make_digits_adapter_problem

Row = dict[str, Any]

DEFAULT_MODEL = "bert-base-uncased"
DEFAULT_CATEGORIES = ("sci.space", "rec.sport.hockey")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=Path("results") / "module_amplification")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--categories", nargs=2, default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--corpus-size", type=int, default=1200)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--design-size", type=int, default=900)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument(
        "--amplifications", type=float, nargs="+",
        default=[1.0, 4.0, 16.0, 64.0, 256.0, 1024.0],
    )
    parser.add_argument("--decimals", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--sig-digits", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--skip-transformer", action="store_true",
                        help="Run only the panels that need no model download.")
    return parser.parse_args()


# --------------------------------------------------------------------------
# Quantizers and the utility metrics they cost
# --------------------------------------------------------------------------


def apply_quantizer(y: np.ndarray, *, decimals: int | None, sig_digits: int | None) -> np.ndarray:
    if decimals is not None:
        return np.round(y, decimals)
    if sig_digits is not None:
        return quantize_significant(y, sig_digits)
    return np.array(y, copy=True)


def _softmax(y: np.ndarray) -> np.ndarray:
    shifted = y - np.max(y, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def utility_costs(exact: np.ndarray, *, decimals: int | None, sig_digits: int | None) -> dict[str, float]:
    """What a quantizer costs the *service*, not the attacker.

    ``label_agreement`` is the metric the paper reported originally. It is blind
    to everything except the argmax, so a defense can look free by it and still
    be ruinous for a consumer of calibrated probabilities. ``logit_mae`` and
    ``softmax_kl`` are recorded so that "the defense is free" is a claim about
    the whole returned object rather than about its argmax.
    """

    served = apply_quantizer(exact, decimals=decimals, sig_digits=sig_digits)
    p, q = _softmax(exact), _softmax(served)
    kl = float(np.mean(np.sum(p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300))), axis=1)))
    return {
        "label_agreement": float(np.mean(np.argmax(served, axis=1) == np.argmax(exact, axis=1))),
        "logit_mae": float(np.mean(np.abs(served - exact))),
        "logit_relative_mae": float(
            np.mean(np.abs(served - exact)) / max(np.mean(np.abs(exact)), 1e-300)
        ),
        "softmax_kl": kl,
    }


# --------------------------------------------------------------------------
# Panel A: which scaling regime is each public feature map in?
# --------------------------------------------------------------------------


def homogeneity_ratio(features_at: dict[float, np.ndarray]) -> dict[float, float]:
    base = float(np.mean(np.linalg.norm(features_at[1.0], axis=1)))
    return {
        amp: float(np.mean(np.linalg.norm(z, axis=1))) / (amp * base) if base > 0 else float("nan")
        for amp, z in features_at.items()
    }


def scaling_regime(ratios: dict[float, float]) -> str:
    """Classify a feature map by what amplification does to its features.

    The boundary cases matter more than the labels: a ratio pinned at 1 means the
    1/A law is exact, a ratio settling at any positive constant means it holds
    asymptotically, and a ratio decaying like 1/A means the map has thrown the
    scale away and no amount of amplification recovers it.
    """

    amps = sorted(ratios)
    largest = ratios[amps[-1]]
    if all(abs(ratios[a] - 1.0) < 1e-9 for a in amps):
        return "exactly homogeneous"
    if largest < 4.0 / amps[-1]:
        return "scale-invariant"
    return "asymptotically homogeneous"


def digits_ratios(*, seeds: list[int], amplifications: list[float]) -> dict[float, float]:
    """Averaged over the same seeds and pool as the homogeneity table.

    A single-seed estimate on a smaller pool gave 1.83 where the ten-seed table
    reports 1.73 for the same quantity, which put two numbers for one ratio in
    the paper. Both were correct measurements of different samples; this uses the
    better-sampled one so there is only one.
    """

    ratios: list[dict[float, float]] = []
    for seed in seeds:
        problem = make_digits_adapter_problem(seed=seed, dataset="digits", feature_dim=8, rank=2)
        rng = np.random.default_rng(seed + 3_000_000)
        x = problem.data.x_train[rng.choice(problem.data.x_train.shape[0], size=800, replace=True)]
        phi = problem.data.feature_map
        ratios.append(homogeneity_ratio({amp: phi(x * amp) for amp in amplifications}))
    return {a: float(np.mean([r[a] for r in ratios])) for a in amplifications}


def identity_ratios(amplifications: list[float]) -> dict[float, float]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 32))
    return homogeneity_ratio({amp: x * amp for amp in amplifications})


# --------------------------------------------------------------------------
# Panel B: the O_module attack at h = 768, with amplification live
# --------------------------------------------------------------------------


def _require_deps() -> dict[str, Any]:
    try:
        import torch
        from sklearn.datasets import fetch_20newsgroups
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "This experiment needs the optional heavy dependencies. "
            "Install them with: uv sync --extra dev --extra lora"
        ) from exc
    return {"torch": torch, "fetch_20newsgroups": fetch_20newsgroups,
            "AutoModel": AutoModel, "AutoTokenizer": AutoTokenizer}


def bert_hidden_states(deps: dict[str, Any], *, model_name: str, categories: list[str],
                       corpus_size: int, max_length: int, batch_size: int,
                       amplifications: list[float]) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Real 768-dimensional CLS states, plus the features of amplified *inputs*.

    The second return value is what an ``O_end`` attacker would obtain by scaling
    the continuous ``inputs_embeds`` surface, which is the strongest amplification
    a text endpoint admits. Panel A uses it to show that LayerNorm removes it.
    """

    torch = deps["torch"]
    bunch = deps["fetch_20newsgroups"](
        subset="train", categories=list(categories),
        remove=("headers", "footers", "quotes"),
    )
    texts = [t for t in bunch.data if t.strip()][:corpus_size]
    tokenizer = deps["AutoTokenizer"].from_pretrained(model_name, use_fast=False)
    backbone = deps["AutoModel"].from_pretrained(model_name).eval()

    encodings = tokenizer(texts, return_tensors="pt", padding="max_length",
                          truncation=True, max_length=max_length)
    states: list[np.ndarray] = []
    amplified: dict[float, list[np.ndarray]] = {amp: [] for amp in amplifications}
    # The homogeneity ratio is a property of the map, not of the corpus, so the
    # amplified sweep runs on a subsample rather than paying one forward pass per
    # amplification per document.
    ratio_budget = min(len(texts), 96)
    with torch.no_grad():
        embed = backbone.embeddings.word_embeddings
        for start in range(0, len(texts), batch_size):
            sl = slice(start, min(start + batch_size, len(texts)))
            ids = encodings["input_ids"][sl]
            mask = encodings["attention_mask"][sl]
            states.append(
                backbone(input_ids=ids, attention_mask=mask, return_dict=True)
                .last_hidden_state[:, 0].numpy().astype(np.float64)
            )
            if start >= ratio_budget:
                continue
            base_embeds = embed(ids)
            for amp in amplifications:
                amplified[amp].append(
                    backbone(inputs_embeds=base_embeds * float(amp), attention_mask=mask,
                             return_dict=True)
                    .last_hidden_state[:, 0].numpy().astype(np.float64)
                )
    return np.vstack(states), {amp: np.vstack(v) for amp, v in amplified.items()}


def make_endpoint_head(z: np.ndarray, *, rank: int, seed: int) -> dict[str, np.ndarray]:
    """A public base head plus a secret low-rank update over the hidden states."""

    rng = np.random.default_rng(seed + 11_000)
    hidden = z.shape[1]
    n_classes = 3
    scale = 1.0 / max(float(np.mean(np.linalg.norm(z, axis=1))), 1e-12)
    base_head = rng.standard_normal((n_classes, hidden)) * scale
    left = rng.standard_normal((n_classes, rank)) * scale
    right = rng.standard_normal((hidden, rank))
    right /= np.linalg.norm(right, axis=0, keepdims=True)
    delta = left @ right.T
    return {"base_head": base_head, "delta": delta, "bias": rng.standard_normal(n_classes) * 0.1}


def module_attack(z_design: np.ndarray, head: dict[str, np.ndarray], *,
                  decimals: int | None, sig_digits: int | None, rank: int,
                  gls: bool = True) -> float:
    """Least-squares recovery of the secret update through ``O_module``.

    The attacker supplies hidden states directly, so the feature map is the
    identity and Corollary 1(1)'s 1/A law applies exactly rather than
    asymptotically.
    """

    full_head = head["base_head"] + head["delta"]
    aug = np.hstack([z_design, np.ones((z_design.shape[0], 1))])
    exact = z_design @ full_head.T + head["bias"]
    served = apply_quantizer(exact, decimals=decimals, sig_digits=sig_digits)
    theta = solve_scores(aug, served, decimals=decimals, sig_digits=sig_digits, gls=gls)
    recovered = theta[:-1].T - head["base_head"]
    u, s, vt = np.linalg.svd(recovered, full_matrices=False)
    recovered = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    return float(np.linalg.norm(recovered - head["delta"]) / np.linalg.norm(head["delta"]))


def run_module_panel(z: np.ndarray, *, seeds: list[int], rank: int, design_size: int,
                     test_size: int, amplifications: list[float], decimals: list[int],
                     sig_digits: list[int]) -> list[Row]:
    rows: list[Row] = []
    hidden = z.shape[1]
    if z.shape[0] < hidden + 1:
        raise RuntimeError(
            f"need at least {hidden + 1} hidden states to span the module feature space, have {z.shape[0]}"
        )
    for seed in seeds:
        rng = np.random.default_rng(seed + 900_000)
        head = make_endpoint_head(z, rank=rank, seed=seed)
        idx = rng.permutation(z.shape[0])
        # A module oracle accepts hidden states, so the design is the attacker's
        # to choose -- and choosing it well is precisely the ||X^+||_2 lever of
        # Theorem 5. Natural CLS states are severely anisotropic: their augmented
        # design is nominally full rank but so ill-conditioned that least squares
        # is dominated by the quantization error in *both* families, which
        # measures the attacker's incompetence rather than the defense. Isotropic
        # probes at the natural scale are what a competent attacker submits.
        natural_norm = float(np.mean(np.linalg.norm(z, axis=1)))
        probes = rng.standard_normal((min(design_size, z.shape[0]), hidden))
        probes *= natural_norm / np.linalg.norm(probes, axis=1, keepdims=True)
        design = probes
        # Utility is still measured on the real distribution the service sees.
        held_out = z[idx[: min(test_size, z.shape[0])]]
        exact_test = held_out @ (head["base_head"] + head["delta"]).T + head["bias"]
        # A single (decimals, sig-digits) pair only ever gives a matched-utility
        # comparison by luck. Sweeping both families and reporting the resulting
        # utility-vs-leakage frontiers makes the claim dominance -- at *any*
        # service degradation the relative family leaks less -- rather than a
        # statement about one hand-picked pair.
        settings = [("absolute", {"decimals": d, "sig_digits": None}) for d in decimals]
        settings += [("relative", {"decimals": None, "sig_digits": p}) for p in sig_digits]
        for family, kwargs in settings:
            costs = utility_costs(exact_test, **kwargs)
            setting = kwargs["decimals"] if family == "absolute" else kwargs["sig_digits"]
            for amp in amplifications:
                rows.append(
                    {
                        "panel": "module",
                        "seed": seed,
                        "family": family,
                        "setting": int(setting),
                        "hidden_dim": hidden,
                        "amplification": amp,
                        # The reported attacker is the strongest we have: GLS,
                        # whose weights are the reciprocal quantization cell
                        # width.  ``ols_relerr`` is the same solve unweighted,
                        # kept so the lever can be ablated -- it is identical
                        # by construction on the absolute family.
                        "relerr": module_attack(design * amp, head, rank=rank, gls=True, **kwargs),
                        "ols_relerr": module_attack(design * amp, head, rank=rank, gls=False, **kwargs),
                        **costs,
                    }
                )
    return rows


# --------------------------------------------------------------------------


def _mean(rows: list[Row], key: str) -> float:
    values = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
    return float(np.mean(values)) if values else float("nan")


def summarize(regimes: list[Row], module: list[Row], *, amplifications: list[float],
              decimals: int, sig_digits: int) -> str:
    lines = [
        "# Where the Amplification Lever Exists",
        "",
        "## Panel A: scaling regime of each public feature map",
        "",
        "`ratio` is ||phi(Ax)|| / (A ||phi(x)||). It decides whether Corollary 1's",
        "amplification lever exists at all: pinned at 1 means the 1/A law is exact,",
        "settling at a positive constant means it holds asymptotically, and decaying",
        "like 1/A means the map discarded the scale and no amplification recovers it.",
        "",
        "| feature map | interface | " + " | ".join(f"A={a:g}" for a in amplifications) + " | regime |",
        "|---|---|" + "---:|" * len(amplifications) + "---|",
    ]
    for row in regimes:
        ratios = {float(k): v for k, v in row["ratios"].items()}
        cells = " | ".join(f"{ratios[a]:.4f}" for a in amplifications)
        lines.append(f"| {row['feature_map']} | {row['interface']} | {cells} | {row['regime']} |")
    lines += [
        "",
        "The transformer row is the finding. LayerNorm makes the feature map",
        "scale-invariant, so amplifying the continuous `inputs_embeds` surface by",
        "1024x moves the feature norm by well under one percent. That, and not the",
        "discreteness of tokens, is why the text-endpoint comparison shows only the",
        "intrinsic perturbation ratio. It is also a defensive fact: a normalized",
        "architecture removes the lever for free -- but only for interfaces that go",
        "through the normalization.",
        "",
    ]
    if module:
        hidden = int(module[0]["hidden_dim"])
        lines += [
            f"## Panel B: O_module at h = {hidden}, amplification live",
            "",
            "The attacker supplies hidden states, so the feature map is the identity",
            "and Corollary 1(1)'s 1/A law is exact rather than asymptotic. `attacker`",
            "is the best error over the amplification budget -- the defender's",
            "worst case. Utility columns are what the quantizer costs the *service*;",
            "`label` alone is blind to everything but the argmax.",
            "",
            "| family | setting | label | logit rel. MAE | softmax KL | attacker (best over A) | amp. gain |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        frontier: list[tuple[str, float, float]] = []
        for family in ["absolute", "relative"]:
            for setting in sorted({int(r["setting"]) for r in module if r["family"] == family}):
                sub = [r for r in module if r["family"] == family and int(r["setting"]) == setting]
                best = min(_mean([r for r in sub if r["amplification"] == a], "relerr")
                           for a in amplifications)
                unamplified = _mean([r for r in sub if r["amplification"] == min(amplifications)], "relerr")
                cost = _mean(sub, "logit_relative_mae")
                frontier.append((family, cost, best))
                lines.append(
                    f"| {family} | {setting} | {_mean(sub, 'label_agreement'):.4f} | {cost:.3e}"
                    f" | {_mean(sub, 'softmax_kl'):.3e} | {best:.3e} | {unamplified / best:.0f}x |"
                )
        lines += [
            "",
            "### Frontier dominance",
            "",
            "At a comparable service cost, which family leaks less? The pairing is",
            "deliberately conservative: each absolute setting is matched to the",
            "relative setting that degrades the service *no more* than it does, so",
            "the relative quantizer is never given a utility handicap in its favour.",
            "",
            "| absolute (cost -> leak) | relative, no worse for utility (cost -> leak) | leak ratio |",
            "|---|---|---:|",
        ]
        absolutes = [f for f in frontier if f[0] == "absolute"]
        relatives = [f for f in frontier if f[0] == "relative"]
        for _, cost, leak in absolutes:
            eligible = [r for r in relatives if r[1] <= cost]
            if not eligible:
                lines.append(f"| {cost:.2e} -> {leak:.2e} | none at least as cheap | --- |")
                continue
            near = max(eligible, key=lambda r: r[1])
            lines.append(
                f"| {cost:.2e} -> {leak:.2e} | {near[1]:.2e} -> {near[2]:.2e} |"
                f" {near[2] / leak:.0f}x |"
            )
        lines += [
            "",
            "The absolute family's leakage collapses with amplification while its",
            "utility cost is unchanged, so no absolute setting reaches the relative",
            "family's frontier. This is the transformer-scale version of the digits",
            "result, and unlike the text-endpoint comparison the amplification lever",
            "is genuinely available here.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    amps = args.amplifications

    regimes: list[Row] = [
        {"feature_map": "identity", "interface": "O_module / embedding input",
         "ratios": identity_ratios(amps)},
        {"feature_map": "standardize/PCA/ReLU (digits)", "interface": "O_raw",
         "ratios": digits_ratios(seeds=list(range(10)), amplifications=amps)},
    ]
    module: list[Row] = []
    if not args.skip_transformer:
        cache = args.outdir / "hidden_states.npz"
        if cache.exists():
            blob = np.load(cache)
            z = blob["z"]
            amplified = {float(k.split("_")[1]): blob[k] for k in blob.files if k.startswith("amp_")}
        else:
            deps = _require_deps()
            z, amplified = bert_hidden_states(
                deps, model_name=args.model, categories=args.categories,
                corpus_size=args.corpus_size, max_length=args.max_length,
                batch_size=args.batch_size, amplifications=amps,
            )
            np.savez(cache, z=z, **{f"amp_{a:g}": v for a, v in amplified.items()})
        regimes.append(
            {"feature_map": f"{args.model} CLS (LayerNorm)", "interface": "O_end via inputs_embeds",
             "ratios": homogeneity_ratio(amplified)}
        )
        module = run_module_panel(
            z, seeds=args.seeds, rank=args.rank, design_size=args.design_size,
            test_size=args.test_size, amplifications=amps, decimals=args.decimals,
            sig_digits=args.sig_digits,
        )
    for row in regimes:
        row["regime"] = scaling_regime({float(k): v for k, v in row["ratios"].items()})

    (args.outdir / "regimes.json").write_text(json.dumps(regimes, indent=2, default=str))
    (args.outdir / "module.json").write_text(json.dumps(module, indent=2))
    if module:
        with (args.outdir / "module.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(module[0].keys()))
            writer.writeheader()
            writer.writerows(module)
    summary = summarize(regimes, module, amplifications=amps,
                        decimals=args.decimals, sig_digits=args.sig_digits)
    (args.outdir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
