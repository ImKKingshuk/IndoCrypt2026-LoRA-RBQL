"""The absolute-versus-relative comparison against a *trained* LoRA adapter.

Every other defense number in the paper is measured against an adapter nobody
trained: either a rank-2 low-rank approximation of a shifted-label ridge head on
8-dimensional ``digits`` features, or a random Gaussian head over real
transformer and vision features.  The one genuinely trained adapter in the paper
-- the rank-4 PEFT LoRA on ``bert-base-uncased``/SST-2 -- is spent on the
*impossibility* result.

This experiment tests the defensive claim on a trained adapter.  It trains the same PEFT LoRA, extracts the trained
module as a 768x768 rank-4 target, and runs the whole defender-then-attacker
game against it through ``O_module`` -- the interface where Section S19 shows all
three conditions for the amplification lever hold, so the comparison is made
where the lever is actually available rather than where it is structurally
absent.

Nothing about the setup is chosen to flatter the defense.  The attacker picks its
own design, which is the ``||X^+||_2`` lever of Theorem 4 and the strongest thing
a module oracle permits; utility is measured on the real hidden-state
distribution the deployed module actually sees; and each absolute setting is
paired with a relative setting that degrades the service *no more*, so relative
quantization never receives a utility handicap in its favour.

Two controls decide whether any given cell means anything.  ``exact_relerr``
repeats the solve with no quantizer: if it is large the attack has already failed
and no defense is being measured, which is the trap that made the first vision
run look like relative quantization losing.  ``design_rank`` records whether the
amplified design still spans, since amplification can collapse it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from lowrank_extraction.adaptive import solve_scores
from lowrank_extraction.experiments.run_lora_checkpoint import (
    _pooled_hidden_states,
    _require_lora_deps,
    encode_texts,
    first_lora_delta,
    load_text_task,
    train_lora_model,
)
from lowrank_extraction.experiments.run_module_amplification import (
    apply_quantizer,
    utility_costs,
)

Row = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=Path("results") / "trained_adapter_defense")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--text-dataset", choices=["20newsgroups", "sst2"], default="sst2")
    parser.add_argument("--categories", nargs=2, default=["sci.space", "rec.sport.hockey"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--target-modules", nargs="+", default=["query", "value"])
    parser.add_argument("--design-size", type=int, default=900)
    parser.add_argument(
        "--amplifications", type=float, nargs="+",
        default=[1.0, 4.0, 16.0, 64.0, 256.0, 1024.0],
    )
    parser.add_argument("--decimals", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--sig-digits", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--offline-mini-corpus", action="store_true")
    parser.add_argument(
        "--fresh", action="store_true",
        help="discard any checkpoint in --outdir and start over",
    )
    return parser.parse_args()


def module_attack(design: np.ndarray, target: dict[str, np.ndarray], *, rank: int,
                  decimals: int | None, sig_digits: int | None,
                  gls: bool = True) -> dict[str, float]:
    """Recover the trained update from quantized module outputs.

    The module is affine, ``x -> (W0 + BA) x + b`` with ``W0, b`` public, so the
    attacker subtracts the public part and solves for the update directly. The
    feature map is the identity, which is what makes Corollary 1(1)'s 1/A law
    exact here rather than asymptotic.
    """

    full = target["base_weight"] + target["delta"]
    exact = design @ full.T + target["base_bias"]
    served = apply_quantizer(exact, decimals=decimals, sig_digits=sig_digits)
    aug = np.hstack([design, np.ones((design.shape[0], 1))])
    theta = solve_scores(aug, served, decimals=decimals, sig_digits=sig_digits, gls=gls)
    recovered = theta[:-1].T - target["base_weight"]
    u, s, vt = np.linalg.svd(recovered, full_matrices=False)
    recovered = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    denom = float(np.linalg.norm(target["delta"]))
    return {
        "relerr": float(np.linalg.norm(recovered - target["delta"]) / denom),
        "design_rank": float(np.linalg.matrix_rank(aug)),
        "required_rank": float(aug.shape[1]),
    }


def run_seed(args: argparse.Namespace, deps: dict[str, Any], seed: int) -> list[Row]:
    train_texts, train_labels, test_texts, test_labels, task = load_text_task(
        deps=deps, text_dataset=args.text_dataset, categories=tuple(args.categories),
        seed=seed, train_size=args.train_size, test_size=args.test_size,
        offline_mini_corpus=args.offline_mini_corpus,
    )
    model, tokenizer, stats = train_lora_model(
        deps=deps, model_name=args.model, train_texts=train_texts,
        train_labels=train_labels, test_texts=test_texts, test_labels=test_labels,
        rank=args.rank, target_modules=list(args.target_modules), epochs=args.epochs,
        batch_size=args.batch_size, max_length=args.max_length,
        learning_rate=args.learning_rate, seed=seed,
    )
    info = first_lora_delta(model)
    target = {
        "base_weight": np.asarray(info["base_weight"], dtype=np.float64),
        "base_bias": np.asarray(info["base_bias"], dtype=np.float64),
        "delta": np.asarray(info["adapter_left"], dtype=np.float64)
        @ np.asarray(info["adapter_right"], dtype=np.float64).T,
    }
    hidden = target["base_weight"].shape[1]

    # The real distribution the module sees, used for utility only.
    encodings = encode_texts(tokenizer, test_texts, max_length=args.max_length)
    states = _pooled_hidden_states(
        model, encodings, batch_size=args.batch_size, torch=deps["torch"]
    )
    natural = float(np.mean(np.linalg.norm(states, axis=1)))

    # A module oracle accepts arbitrary vectors, so the design is the attacker's
    # to choose; isotropic probes at the natural scale are what a competent one
    # submits, and choosing them well is itself the Theorem 4 lever.
    rng = np.random.default_rng(seed + 77_000)
    probes = rng.standard_normal((max(args.design_size, hidden + 1), hidden))
    probes *= natural / np.linalg.norm(probes, axis=1, keepdims=True)

    exact_out = states @ (target["base_weight"] + target["delta"]).T + target["base_bias"]
    rows: list[Row] = []
    settings = [("absolute", {"decimals": d, "sig_digits": None}) for d in args.decimals]
    settings += [("relative", {"decimals": None, "sig_digits": p}) for p in args.sig_digits]
    for family, kwargs in settings:
        costs = utility_costs(exact_out, **kwargs)
        setting = kwargs["decimals"] if family == "absolute" else kwargs["sig_digits"]
        for amp in args.amplifications:
            attacked = module_attack(probes * amp, target, rank=args.rank, gls=True, **kwargs)
            # Under an absolute grid the quantizer's cell width does not depend on
            # the score, so the GLS weights are uniform and ``solve_scores``
            # short-circuits to the same unweighted solve. Re-running it with
            # gls=False would recompute a bit-identical result at full cost, so
            # the absolute half of the grid reuses it. Measured on the 3-seed
            # run this replaces: 54/54 absolute cells agreed to the last bit.
            if kwargs["decimals"] is not None:
                ols = attacked
            else:
                ols = module_attack(probes * amp, target, rank=args.rank, gls=False, **kwargs)
            control = module_attack(
                probes * amp, target, rank=args.rank, decimals=None, sig_digits=None
            )
            rows.append({
                "seed": seed,
                "task": task,
                "module": str(info["module_name"]),
                "hidden_dim": hidden,
                "adapter_rank": int(info["rank"]),
                "test_accuracy": float(stats.get("test_accuracy", float("nan"))),
                "family": family,
                "setting": int(setting),
                "amplification": amp,
                "exact_relerr": control["relerr"],
                "ols_relerr": ols["relerr"],
                **attacked,
                **costs,
            })
    return rows


def _mean(rows: list[Row], key: str) -> float:
    vals = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def _per_seed_best(rows: list[Row], family: str, setting: int,
                   amplifications: list[float]) -> dict[int, float]:
    """Each seed's own best error over the amplification budget.

    The frontier is a min over A of a mean over seeds, which is the right point
    estimate but cannot carry an interval: the pairing has to be per seed before
    it can be resampled. This returns that per-seed view.
    """

    out: dict[int, float] = {}
    sub = [r for r in rows if r["family"] == family and int(r["setting"]) == setting]
    for seed in sorted({int(r["seed"]) for r in sub}):
        errs = [
            float(r["relerr"])
            for r in sub
            if int(r["seed"]) == seed and r["amplification"] in amplifications
        ]
        if errs:
            out[seed] = min(errs)
    return out


def _paired_ratio_ci(numerator: dict[int, float], denominator: dict[int, float],
                     *, draws: int = 20000) -> tuple[float, float, float]:
    """Percentile CI for the leak ratio, resampling seeds as pairs.

    The reported ratio is the quotient of the two per-cell *means*, which is the
    aggregation convention used throughout the paper. The bootstrap therefore has
    to resample seeds as pairs and recompute that same quotient on each draw.
    Averaging the per-seed ratios instead estimates a different functional -- on
    this artifact it lands about 1.3 higher -- so an interval built that way does
    not bracket the number printed beside it, and the point estimate sits
    slightly off-centre in its own interval for no stated reason.

    At the seed counts a LoRA training run makes affordable a percentile
    bootstrap is coarse -- it can only land on a limited number of distinct
    resample values -- so the interval should be read as a spread indicator
    rather than a calibrated 95% statement. It is reported anyway because a
    headline ratio quoted with no spread at all invites more trust than a
    handful of seeds can support.
    """

    seeds = sorted(set(numerator) & set(denominator))
    if len(seeds) < 2:
        return float("nan"), float("nan"), float("nan")
    num = np.array([numerator[s] for s in seeds], dtype=np.float64)
    den = np.array([denominator[s] for s in seeds], dtype=np.float64)
    rng = np.random.default_rng(0)
    index = rng.integers(0, num.size, size=(draws, num.size))
    ratios = num[index].mean(axis=1) / np.maximum(den[index].mean(axis=1), 1e-300)
    point = float(num.mean() / max(den.mean(), 1e-300))
    return point, float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def summarize(rows: list[Row], *, amplifications: list[float]) -> str:
    hidden = int(rows[0]["hidden_dim"])
    lines = [
        "# Absolute vs Relative Quantization on a *Trained* LoRA Adapter",
        "",
        f"Target: the rank-{int(rows[0]['adapter_rank'])} `{rows[0]['module']}` module of a PEFT LoRA",
        f"trained on {rows[0]['task']} ({_mean(rows,'test_accuracy'):.4f} test accuracy),",
        f"attacked through O_module at h = {hidden}.",
        "",
        "This is the same defense comparison the paper makes on `digits` and on",
        "random Gaussian heads, now against an adapter that was actually trained by",
        "gradient descent.",
        "",
        "`exact` is the same solve with no quantizer: the control that says whether a",
        "defense is being measured at all rather than a broken attack.",
        "",
        "| family | setting | label | logit rel. MAE | attacker (best over A) | amp. gain | exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    frontier: list[tuple[str, float, float, int]] = []
    for family in ["absolute", "relative"]:
        for setting in sorted({int(r["setting"]) for r in rows if r["family"] == family}):
            sub = [r for r in rows if r["family"] == family and int(r["setting"]) == setting]
            at = lambda a: _mean([r for r in sub if r["amplification"] == a], "relerr")
            best = min(at(a) for a in amplifications)
            cost = _mean(sub, "logit_relative_mae")
            frontier.append((family, cost, best, setting))
            lines.append(
                f"| {family} | {setting} | {_mean(sub,'label_agreement'):.4f} | {cost:.3e} |"
                f" {best:.3e} | {at(min(amplifications))/best:.1f}x | {_mean(sub,'exact_relerr'):.2e} |"
            )
    lines += [
        "",
        "## Frontier, paired conservatively",
        "",
        "Each absolute setting is matched to a relative setting that degrades the",
        "service *no more*, so relative is never given a utility advantage.",
        "",
        "| absolute (cost -> leak) | relative, no worse (cost -> leak) | leak ratio [95% CI] |",
        "|---|---|---|",
    ]
    relatives = [f for f in frontier if f[0] == "relative"]
    ratios = []
    for _, cost, leak, abs_setting in [f for f in frontier if f[0] == "absolute"]:
        eligible = [r for r in relatives if r[1] <= cost]
        if not eligible:
            lines.append(f"| {cost:.2e} -> {leak:.2e} | none at least as cheap | --- |")
            continue
        near = max(eligible, key=lambda r: r[1])
        # A relative error above 1 means the recovered update is further from the
        # truth than zero is: recovery is destroyed, not merely degraded. Quoting
        # a ratio there would compare a working attack against a failed one and
        # overstate the gap, so the pair is reported and excluded from the range.
        if near[2] > 1.0 or leak > 1.0:
            lines.append(
                f"| {cost:.2e} -> {leak:.2e} | {near[1]:.2e} -> {near[2]:.2e} |"
                " recovery destroyed |"
            )
            continue
        # The point estimate comes back from the same helper that builds the
        # interval, so the two cannot drift apart.
        point, lo, hi = _paired_ratio_ci(
            _per_seed_best(rows, "relative", near[3], amplifications),
            _per_seed_best(rows, "absolute", abs_setting, amplifications),
        )
        ratios.append(point)
        lines.append(
            f"| {cost:.2e} -> {leak:.2e} | {near[1]:.2e} -> {near[2]:.2e} |"
            f" {point:.0f}x [{lo:.0f}, {hi:.0f}] |"
        )
    if ratios:
        lines += [
            "",
            f"Over the pairs where both families still permit recovery, relative",
            f"quantization leaves the attacker {min(ratios):.0f}x-{max(ratios):.0f}x worse off against a",
            "trained adapter, at no greater cost to the service. Pairs marked",
            "\"recovery destroyed\" are excluded: there the relative quantizer already",
            "puts the attacker beyond a relative error of 1, so the comparison would",
            "be a working attack against a failed one.",
        ]
    return "\n".join(lines) + "\n"


#: Every argument that changes what a seed *measures*. ``seeds`` and ``outdir``
#: are deliberately absent: resuming is precisely the act of adding seeds to an
#: existing set, and the output location does not affect a number.
_CONFIG_KEYS = (
    "model", "text_dataset", "categories", "train_size", "test_size", "epochs",
    "batch_size", "max_length", "rank", "learning_rate", "target_modules",
    "design_size", "amplifications", "decimals", "sig_digits",
    "offline_mini_corpus",
)


def _config_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _CONFIG_KEYS:
        value = getattr(args, key)
        out[key] = list(value) if isinstance(value, (list, tuple)) else value
    return out


def _load_checkpoint(path: Path, fingerprint: dict[str, Any]) -> list[Row]:
    """Rows from seeds already finished, or nothing if there is no usable resume.

    A seed costs a LoRA training run plus a 36-cell attack grid, and the whole
    experiment takes hours; a power cut with no checkpoint throws all of it
    away. So each seed is appended and fsync'd as it completes.

    Resuming across a *different* configuration would silently blend two
    experiments into one table, which is worse than losing the work, so a
    fingerprint mismatch refuses rather than guesses.
    """

    meta = path.with_suffix(".meta.json")
    if not path.exists() or not meta.exists():
        return []
    stored = json.loads(meta.read_text(encoding="utf-8"))
    if stored != fingerprint:
        differing = sorted(
            k for k in set(stored) | set(fingerprint)
            if stored.get(k) != fingerprint.get(k)
        )
        raise SystemExit(
            f"refusing to resume: {path} was written under a different configuration "
            f"(differs in: {', '.join(differing)}). Pass --fresh to discard it, or "
            f"point --outdir somewhere else."
        )
    rows: list[Row] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A row half-written when the power died. Everything before it is
            # intact by construction, so drop the tail and re-run that seed.
            break
    return rows


def _append_checkpoint(path: Path, rows: list[Row]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.outdir / "checkpoint.jsonl"
    fingerprint = _config_fingerprint(args)
    if args.fresh:
        checkpoint.unlink(missing_ok=True)
        checkpoint.with_suffix(".meta.json").unlink(missing_ok=True)
    rows: list[Row] = _load_checkpoint(checkpoint, fingerprint)
    checkpoint.with_suffix(".meta.json").write_text(
        json.dumps(fingerprint, indent=2), encoding="utf-8"
    )
    done = {int(r["seed"]) for r in rows}
    if done:
        print(f"resuming: {sorted(done)} already complete, {len(rows)} rows recovered")

    deps = _require_lora_deps()
    for seed in args.seeds:
        if seed in done:
            continue
        fresh_rows = run_seed(args, deps, seed)
        _append_checkpoint(checkpoint, fresh_rows)
        rows.extend(fresh_rows)
        print(f"seed {seed} complete and checkpointed ({len(rows)} rows total)", flush=True)

    rows = [r for r in rows if int(r["seed"]) in set(args.seeds)]
    rows.sort(key=lambda r: (int(r["seed"]), str(r["family"]), int(r["setting"]),
                             float(r["amplification"])))
    (args.outdir / "trained_adapter_defense.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with (args.outdir / "trained_adapter_defense.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(rows, amplifications=args.amplifications)
    (args.outdir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
