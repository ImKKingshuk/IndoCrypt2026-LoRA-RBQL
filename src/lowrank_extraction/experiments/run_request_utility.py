"""What request canonicalization and score coarsening cost a legitimate user.

Theorem 7 bounds an attacker's relative error below ``(1/8)*10^-(p+d+1)``, where
``p`` is the significant digits the server puts on the wire and ``d`` is the
significant digits it accepts in a request.  Section app:requestmodel measures
the defender's side of that bound and finds the constraint runs the unintuitive
way: the floor is *large* only when ``p + d`` is *small*, so a 1e-4 floor needs
``p + d <= 2`` -- two significant digits between the request and the response,
together.

The paper's conclusion then asserts that this is "a precision that costs real
utility".  That assertion was never measured.  This script measures it, on the
same three deployments the rest of the paper attacks, and it is deliberately
built so the answer can come back either way: if coarsening is cheap, the
defense is viable and we should say so.

The design point that makes the result non-obvious is that ``p`` and ``d`` fall
on different users:

  * ``d`` binds only a user whose *request* carries numbers.  A text
    classification endpoint takes a sentence, so canonicalizing its requests is
    free -- but that is also the interface where LayerNorm already denies the
    attacker a scale lever (Section sec:exp-regimes).  Where the attacker does
    have the lever, at an embedding or pixel interface, the user is sending
    floats and pays.
  * ``p`` binds everyone who reads a score.  Top-1 label survival hides this,
    which is why Section app:utility already argues against it as the sole
    metric; a user consuming *scores* loses what the distribution shift costs,
    so total variation and KL against the unquantized service are reported
    beside accuracy.

Three real deployments, each a public base model plus an adapted head, which is
the structure the recovery theory assumes:

  * **text**  -- BERT-base + a trained LoRA adapter on SST-2, at both the text
    interface (``d`` inapplicable) and the embedding interface (``d`` applies).
  * **vision** -- ResNet18 trunk + a linear head fitted on CIFAR-10, requests in
    the pixel domain.
  * **tabular** -- sklearn ``wine`` features + a ridge head, the case where a
    request is nothing but numbers.

For each cell of the ``(p, d)`` grid the served answer is
``quantize_p(f(quantize_d(x)))`` and the baseline is ``f(x)``.  The grid is swept
by featurizing once per ``d`` and applying every ``p`` afterwards, since ``p``
touches only the response.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from lowrank_extraction.experiments.run_precision_floor import exact_floor

from lowrank_extraction.oracles import quantize_significant

# ``None`` is the unrounded service: full float64 in, full float64 out. It is
# the baseline every cell is compared against, and it is also the deployment
# every real endpoint we surveyed actually runs.
EXACT = None


def _round_request(x: np.ndarray, d: int | None) -> np.ndarray:
    """Canonicalize a request to ``d`` significant digits, elementwise."""

    if d is EXACT:
        return x
    return quantize_significant(x, d)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def utility(served: np.ndarray, baseline: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """What the legitimate user gets, measured five ways.

    ``accuracy`` is the metric an operator quotes and the one most forgiving of
    rounding, because argmax survives perturbations that move every score.  The
    distributional metrics are what a user consuming scores actually loses, and
    ``nll`` is the one a downstream probabilistic consumer pays.
    """

    p_served, p_base = _softmax(served), _softmax(baseline)
    pred_served = served.argmax(axis=1)
    # Rounding creates exact ties that argmax breaks by index; a tie is a coin
    # flip for the user, so count it as such rather than letting index order
    # flatter the defense.
    tied = (served == served.max(axis=1, keepdims=True)).sum(axis=1) > 1
    eps = 1e-12
    return {
        "accuracy": float((pred_served == labels).mean()),
        "agreement": float((pred_served == baseline.argmax(axis=1)).mean()),
        "tie_rate": float(tied.mean()),
        "total_variation": float(0.5 * np.abs(p_served - p_base).sum(axis=1).mean()),
        "kl": float((p_base * (np.log(p_base + eps) - np.log(p_served + eps))).sum(axis=1).mean()),
        "nll": float(-np.log(p_served[np.arange(len(labels)), labels] + eps).mean()),
    }


def sweep(logits_by_d: dict[Any, np.ndarray], labels: np.ndarray,
          served_digits: list[int | None]) -> list[dict[str, Any]]:
    """Cross the request grid with the response grid.

    ``logits_by_d`` holds one forward pass per request precision; every served
    precision is then applied to those logits without re-running the model.
    """

    baseline = logits_by_d[EXACT]
    rows = []
    for d, logits in logits_by_d.items():
        for p in served_digits:
            served = logits if p is EXACT else quantize_significant(logits, p)
            row: dict[str, Any] = {
                "request_digits": d, "served_digits": p,
                **utility(served, baseline, labels),
            }
            # Theorem 7's guarantee at this cell, for the scalar case it
            # covers.  This is part (iii)'s exact infimum over the secret, not
            # part (ii)'s proven constant: the latter is twenty times smaller
            # and would overstate what the defense costs by a digit.
            row["theorem7_floor"] = (
                None if (p is EXACT or d is EXACT)
                else exact_floor(p, d)
            )
            rows.append(row)
    return rows


# --- the three deployments --------------------------------------------------


def panel_tabular(*, request_digits: list[int | None], served_digits: list[int | None],
                  seeds: list[int], domain: str = "raw") -> dict[str, Any]:
    """Requests that are nothing but numbers: the worst case for the defense.

    The wine test split is only ~70 examples, so a single seed cannot separate a
    real drop from a resampling artefact. This panel is cheap, so it runs over
    every seed and reports the spread alongside the mean.
    """

    from sklearn.datasets import load_wine
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_wine()
    per_seed = []
    for seed in seeds:
        x_train, x_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.4, random_state=seed,
            stratify=data.target)
        # The operator fits on clean data; canonicalization applies to what
        # arrives at inference, and it applies in the *raw* feature domain,
        # because that is what the request carries and all the defense sees.
        scaler = StandardScaler().fit(x_train)
        head = RidgeClassifier().fit(scaler.transform(x_train), y_train)
        x = np.asarray(x_test, dtype=np.float64)
        if domain == "raw":
            # What the request actually carries, and all a front-end sees.
            serve = lambda d: head.decision_function(
                scaler.transform(_round_request(x, d)))
        else:
            # A canonicalizer placed after the operator's own scaler. Still a
            # discrete grid from the attacker's side, and far kinder to columns
            # whose raw magnitudes differ by three orders.
            serve = lambda d: head.decision_function(
                _round_request(scaler.transform(x), d))
        logits_by_d = {d: serve(d) for d in request_digits}
        per_seed.append(sweep(logits_by_d, np.asarray(y_test), served_digits))

    keys = ("accuracy", "agreement", "tie_rate", "total_variation", "kl", "nll")
    rows = []
    for cells in zip(*per_seed):
        row = {k: v for k, v in cells[0].items() if k not in keys}
        for key in keys:
            values = [c[key] for c in cells]
            row[key] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))
        rows.append(row)

    return {
        "deployment": f"tabular ({domain} domain)", "interface": "feature vector",
        "model": "sklearn wine + ridge head", "request_carries_numbers": True,
        "canonicalization_domain": domain,
        "examples": int(len(y_test)), "seeds": seeds, "rows": rows,
    }


def panel_vision(*, request_digits: list[int | None], served_digits: list[int | None],
                 seed: int, data_root: Path, train_size: int, test_size: int,
                 image_size: int, batch_size: int, domain: str = "pixel") -> dict[str, Any]:
    """A pixel API: continuous requests, and the interface where the attacker's
    amplification lever survives (Section app:vision)."""

    import torch
    import torchvision
    from sklearn.linear_model import LogisticRegression
    from torchvision import transforms
    from torchvision.models import ResNet18_Weights, resnet18

    tf = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    data_root.mkdir(parents=True, exist_ok=True)
    train_set = torchvision.datasets.CIFAR10(root=str(data_root), train=True,
                                             download=True, transform=tf)
    test_set = torchvision.datasets.CIFAR10(root=str(data_root), train=False,
                                            download=True, transform=tf)
    rng = np.random.default_rng(seed)
    train_ids = rng.choice(len(train_set), size=train_size, replace=False)
    test_ids = rng.choice(len(test_set), size=test_size, replace=False)

    model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    trunk = torch.nn.Sequential(*list(model.children())[:-1])
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def featurize(pixels: torch.Tensor, d: int | None = EXACT) -> np.ndarray:
        """Featurize, canonicalizing the request in the chosen domain.

        The operator may snap the request to a grid in the pixel domain, which
        is what arrives on the wire, or after its own normalization. The two are
        related by an invertible affine map, so both are genuinely discrete from
        the attacker's side -- but they are not equally kind to the user, and
        which one an operator picks turns out to decide the whole cost.
        """
        out = []
        with torch.no_grad():
            for i in range(0, len(pixels), batch_size):
                chunk = pixels[i:i + batch_size]
                if domain == "pixel":
                    chunk = torch.from_numpy(_round_request(
                        chunk.numpy().astype(np.float64), d)).float()
                chunk = (chunk - mean) / std
                if domain == "normalized":
                    chunk = torch.from_numpy(_round_request(
                        chunk.numpy().astype(np.float64), d)).float()
                out.append(trunk(chunk).flatten(1).numpy())
        return np.concatenate(out).astype(np.float64)

    def pixels_of(dataset: Any, ids: np.ndarray) -> torch.Tensor:
        return torch.stack([dataset[int(i)][0] for i in ids])

    train_pixels = pixels_of(train_set, train_ids)
    y_train = np.array([train_set[int(i)][1] for i in train_ids])
    y_test = np.array([test_set[int(i)][1] for i in test_ids])
    head = LogisticRegression(max_iter=2000).fit(featurize(train_pixels), y_train)
    del train_pixels

    test_pixels = pixels_of(test_set, test_ids)
    logits_by_d = {d: head.decision_function(featurize(test_pixels, d))
                   for d in request_digits}

    return {
        "deployment": f"vision ({domain} domain)", "interface": "pixel array",
        "canonicalization_domain": domain,
        "model": "resnet18 trunk + linear head on CIFAR-10",
        "request_carries_numbers": True,
        "examples": int(len(y_test)), "train_examples": int(train_size),
        "rows": sweep(logits_by_d, y_test, served_digits),
    }


def panel_text(*, request_digits: list[int | None], served_digits: list[int | None],
               adapter_dir: Path, tokenizer_dir: Path, seed: int, test_size: int,
               train_size: int, max_length: int, batch_size: int) -> list[dict[str, Any]]:
    """The trained SST-2 adapter, at both interfaces it can be deployed behind.

    Returns two panels because the same model gives two different answers: at
    the text interface a request has no numeric field to canonicalize, and at
    the embedding interface it is all numbers.
    """

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from lowrank_extraction.experiments.run_lora_checkpoint import (
        _require_lora_deps, encode_texts, load_text_task,
    )

    deps = _require_lora_deps()
    _, _, x_test, y_test, _ = load_text_task(
        deps=deps, text_dataset="sst2", categories=("", ""), seed=seed,
        train_size=train_size, test_size=test_size, offline_mini_corpus=False)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=False)
    base = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
    enc = encode_texts(tokenizer, x_test, max_length=max_length)
    labels = np.asarray(y_test)

    def logits_from_embeddings(d: int | None) -> np.ndarray:
        """Run the encoder from input embeddings, canonicalized to d digits."""
        embed = model.get_input_embeddings()
        out = []
        with torch.no_grad():
            for i in range(0, len(labels), batch_size):
                ids = enc["input_ids"][i:i + batch_size]
                vectors = embed(ids)
                if d is not EXACT:
                    vectors = torch.from_numpy(_round_request(
                        vectors.numpy().astype(np.float64), d)).to(vectors.dtype)
                out.append(model(
                    inputs_embeds=vectors,
                    attention_mask=enc["attention_mask"][i:i + batch_size],
                ).logits.numpy().astype(np.float64))
        return np.concatenate(out)

    # Text interface: the request is a sentence. There is no numeric field, so
    # every request precision is the same deployment and costs the user nothing.
    text_logits = logits_from_embeddings(EXACT)
    text_panel = {
        "deployment": "text", "interface": "sentence",
        "model": "bert-base + trained LoRA adapter on SST-2",
        "request_carries_numbers": False,
        "examples": int(len(labels)),
        "rows": sweep({EXACT: text_logits}, labels, served_digits),
    }

    embedding_panel = {
        "deployment": "text-embedding", "interface": "input embeddings",
        "model": "bert-base + trained LoRA adapter on SST-2",
        "request_carries_numbers": True,
        "examples": int(len(labels)),
        "rows": sweep({d: logits_from_embeddings(d) for d in request_digits},
                      labels, served_digits),
    }
    return [text_panel, embedding_panel]


# --- what the defender can buy ----------------------------------------------


def frontier(panels: list[dict[str, Any]], targets: list[float]) -> list[dict[str, Any]]:
    """Best utility reachable at each security target.

    For a target floor ``F`` the defender may spend any ``(p, d)`` with
    ``exact_floor(p, d) >= F``; this reports the cell that keeps the most
    utility, so the cost quoted is the *cheapest* way to buy that floor rather
    than an unflattering corner of the grid.
    """

    import math

    out = []
    for target in targets:
        # Theorem 7(iii) gives F ~ (1/4)*10^-(p+d), so the target fixes a budget
        # on the *sum*. A server has to serve at least one digit and accept at
        # least one, so any budget below 2 is not a deployment at all -- which
        # is a different and stronger obstruction than an expensive one.
        budget = math.floor(-math.log10(4.0 * target))
        for panel in panels:
            eligible = [r for r in panel["rows"]
                        if r["theorem7_floor"] is not None and r["theorem7_floor"] >= target]
            if not eligible:
                out.append({
                    "target_floor": target, "deployment": panel["deployment"],
                    "reachable": False, "max_p_plus_d": budget,
                    "reason": (
                        "the request carries no numeric field, so there is no d "
                        "and Theorem 7 says nothing here"
                        if not panel["request_carries_numbers"] else
                        f"the budget on p + d is {budget}, and a server must "
                        "serve and accept at least one digit each"),
                    "obstruction": ("no request precision"
                                    if not panel["request_carries_numbers"]
                                    else "no deployable (p, d)"),
                })
                continue
            # Rank by accuracy first, then by how little the served scores moved.
            best = max(eligible, key=lambda r: (r["accuracy"], -r["total_variation"]))
            baseline = next(r for r in panel["rows"]
                            if r["request_digits"] is EXACT and r["served_digits"] is EXACT)
            out.append({
                "target_floor": target, "deployment": panel["deployment"],
                "reachable": True, "max_p_plus_d": budget,
                "p": best["served_digits"], "d": best["request_digits"],
                "accuracy": best["accuracy"], "baseline_accuracy": baseline["accuracy"],
                "accuracy_drop": baseline["accuracy"] - best["accuracy"],
                "agreement": best["agreement"],
                "total_variation": best["total_variation"], "kl": best["kl"],
            })
    return out


def _fmt(value: Any) -> str:
    if value is EXACT:
        return "exact"
    if isinstance(value, float):
        return f"{value:.4f}" if value >= 1e-3 or value == 0.0 else f"{value:.2e}"
    return str(value)


def write_summary(path: Path, panels: list[dict[str, Any]], front: list[dict[str, Any]],
                  targets: list[float]) -> str:
    lines = ["# What Rounding Costs the Legitimate User", ""]
    for panel in panels:
        lines += [
            f"## {panel['deployment']} -- {panel['model']}",
            "",
            f"Interface: {panel['interface']}; requests carry numbers: "
            f"{panel['request_carries_numbers']}; {panel['examples']} test examples.",
            "",
            "| request digits d | served digits p | Thm 7 floor | accuracy | agreement | TV | KL |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in panel["rows"]:
            lines.append(
                f"| {_fmt(row['request_digits'])} | {_fmt(row['served_digits'])} | "
                f"{_fmt(row['theorem7_floor'])} | {row['accuracy']:.4f} | "
                f"{row['agreement']:.4f} | {row['total_variation']:.4f} | {row['kl']:.4f} |"
            )
        lines.append("")

    lines += [
        "## The cheapest way to buy each floor",
        "",
        "| target floor | deployment | p | d | accuracy | baseline | drop | TV |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in front:
        if not row["reachable"]:
            detail = (row["obstruction"] if row["obstruction"] == "no request precision"
                      else f"no deployable (p, d): p + d <= {row['max_p_plus_d']}")
            lines.append(
                f"| {row['target_floor']:.0e} | {row['deployment']} | -- | -- | "
                f"{detail} | | | |")
            continue
        lines.append(
            f"| {row['target_floor']:.0e} | {row['deployment']} | {row['p']} | {row['d']} | "
            f"{row['accuracy']:.4f} | {row['baseline_accuracy']:.4f} | "
            f"{row['accuracy_drop']:+.4f} | {row['total_variation']:.4f} |"
        )
    lines.append("")
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("results") / "request_utility")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the vision and text panels, which run once")
    ap.add_argument("--tabular-seeds", type=int, nargs="+", default=list(range(10)),
                    help="the tabular panel is cheap enough to average over splits")
    ap.add_argument("--request-digits", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--served-digits", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--targets", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-6])
    ap.add_argument("--panels", nargs="+",
                    choices=["tabular", "vision", "text"],
                    default=["tabular", "vision", "text"])
    ap.add_argument("--data-root", type=Path,
                    default=Path.home() / ".cache" / "cifar10")
    ap.add_argument("--vision-train", type=int, default=5000)
    ap.add_argument("--vision-test", type=int, default=2000)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--checkpoint-dir", type=Path,
                    default=Path("results") / "lora_checkpoint_sst2_bert_base_stronger_10seed")
    ap.add_argument("--text-test", type=int, default=500)
    ap.add_argument("--text-train", type=int, default=1800)
    ap.add_argument("--max-length", type=int, default=96)
    args = ap.parse_args()

    request_digits: list[int | None] = [EXACT, *args.request_digits]
    served_digits: list[int | None] = [EXACT, *args.served_digits]

    started = time.time()
    panels: list[dict[str, Any]] = []
    if "tabular" in args.panels:
        for domain in ("raw", "standardized"):
            panels.append(panel_tabular(
                request_digits=request_digits, served_digits=served_digits,
                seeds=args.tabular_seeds, domain=domain))
    if "vision" in args.panels:
        for domain in ("pixel", "normalized"):
            panels.append(panel_vision(
                request_digits=request_digits, served_digits=served_digits,
                seed=args.seed, data_root=args.data_root,
                train_size=args.vision_train, test_size=args.vision_test,
                image_size=args.image_size, batch_size=args.batch_size,
                domain=domain))
    if "text" in args.panels:
        panels.extend(panel_text(
            request_digits=request_digits, served_digits=served_digits,
            adapter_dir=args.checkpoint_dir / f"adapter_seed_{args.seed}",
            tokenizer_dir=args.checkpoint_dir / f"tokenizer_seed_{args.seed}",
            seed=args.seed, test_size=args.text_test, train_size=args.text_train,
            max_length=args.max_length, batch_size=args.batch_size))

    front = frontier(panels, args.targets)
    # The strongest guarantee a server that serves and accepts at least one
    # digit can claim at all. Everything the defense could ever offer sits at
    # or below this number, whatever it costs.
    best_deployable = {
        "p": 1, "d": 1, "floor": exact_floor(1, 1),
        "note": "the largest Theorem 7(iii) floor over deployable (p, d), both >= 1",
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "request_utility.json").write_text(
        json.dumps({"panels": panels, "frontier": front,
                    "best_deployable_floor": best_deployable,
                    "seed": args.seed, "targets": args.targets,
                    "runtime_seconds": round(time.time() - started, 1)}, indent=2) + "\n",
        encoding="utf-8")
    print(write_summary(args.outdir / "summary.md", panels, front, args.targets))
    print(f"runtime {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
