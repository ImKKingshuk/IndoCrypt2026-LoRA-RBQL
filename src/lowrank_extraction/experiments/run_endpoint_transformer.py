from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from lowrank_extraction.metrics import relative_frobenius_error
from lowrank_extraction.oracles import quantize_significant

from .run_lora_checkpoint import DEFAULT_CATEGORIES, DEFAULT_MODEL, encode_texts, load_text_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover an observable low-rank transformer endpoint head from final-logit oracle queries."
    )
    parser.add_argument("--outdir", type=Path, default=Path("results") / "endpoint_transformer")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text-dataset", choices=["20newsgroups", "sst2"], default="20newsgroups")
    parser.add_argument("--categories", nargs=2, default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--train-size", type=int, default=192)
    parser.add_argument("--test-size", type=int, default=96)
    parser.add_argument("--query-size", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rounded-decimals", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument(
        "--sig-digits",
        type=int,
        default=3,
        help="Relative (significant-digit) quantization arm, for the utility-matched "
        "absolute-vs-relative comparison on a real endpoint head.",
    )
    parser.add_argument("--offline-mini-corpus", action="store_true", help="Use a tiny embedded corpus for smoke tests only.")
    return parser.parse_args()


def _require_deps() -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.model_selection import train_test_split
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "The endpoint transformer experiment needs optional heavy dependencies. "
            "Install them with: uv sync --extra dev --extra lora"
        ) from exc
    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "fetch_20newsgroups": fetch_20newsgroups,
        "train_test_split": train_test_split,
        "AutoModel": AutoModel,
        "AutoTokenizer": AutoTokenizer,
    }


def transformer_cls_features(
    *,
    deps: dict[str, Any],
    model_name: str,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> tuple[Any, Any, np.ndarray]:
    torch = deps["torch"]
    tokenizer = deps["AutoTokenizer"].from_pretrained(model_name, use_fast=False)
    backbone = deps["AutoModel"].from_pretrained(model_name)
    backbone.eval()
    encodings = encode_texts(tokenizer, texts, max_length=max_length)
    features = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = slice(start, min(start + batch_size, len(texts)))
            inputs = {key: value[batch] for key, value in encodings.items()}
            outputs = backbone(**inputs, return_dict=True)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            features.append(pooled.detach().cpu().numpy().astype(np.float64))
    return backbone, tokenizer, np.vstack(features)


def train_lowrank_endpoint_head(
    *,
    deps: dict[str, Any],
    z_train: np.ndarray,
    y_train: list[int],
    z_test: np.ndarray,
    y_test: list[int],
    rank: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    torch = deps["torch"]
    torch.manual_seed(seed)
    n_classes = int(max(y_train) + 1)
    hidden_dim = int(z_train.shape[1])
    if rank > min(n_classes, hidden_dim):
        raise ValueError("rank must not exceed min(n_classes, hidden_dim)")

    z_train_t = torch.tensor(z_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    z_test_t = torch.tensor(z_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    base_head = torch.zeros((n_classes, hidden_dim), dtype=torch.float32)
    adapter_left = torch.nn.Parameter(0.02 * torch.randn(n_classes, rank, dtype=torch.float32))
    adapter_right = torch.nn.Parameter(0.02 * torch.randn(hidden_dim, rank, dtype=torch.float32))
    bias = torch.nn.Parameter(torch.zeros(n_classes, dtype=torch.float32))
    optimizer = torch.optim.AdamW([adapter_left, adapter_right, bias], lr=learning_rate, weight_decay=weight_decay)

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        delta = adapter_left @ adapter_right.T
        logits = z_train_t @ (base_head + delta).T + bias
        loss = torch.nn.functional.cross_entropy(logits, y_train_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        delta = adapter_left @ adapter_right.T
        train_logits = z_train_t @ (base_head + delta).T + bias
        test_logits = z_test_t @ (base_head + delta).T + bias
        train_acc = float((torch.argmax(train_logits, dim=1) == y_train_t).float().mean().item())
        test_acc = float((torch.argmax(test_logits, dim=1) == y_test_t).float().mean().item())

    return {
        "base_head": base_head.detach().cpu().numpy().astype(np.float64),
        "adapter_left": adapter_left.detach().cpu().numpy().astype(np.float64),
        "adapter_right": adapter_right.detach().cpu().numpy().astype(np.float64),
        "bias": bias.detach().cpu().numpy().astype(np.float64),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }


def recover_endpoint_head_from_final_logits(
    *,
    z_query: np.ndarray,
    logits_query: np.ndarray,
    base_head: np.ndarray,
    rank: int,
) -> dict[str, Any]:
    design = np.hstack([z_query, np.ones((z_query.shape[0], 1), dtype=np.float64)])
    feature_rank = int(np.linalg.matrix_rank(design, tol=1e-10))
    if feature_rank < design.shape[1]:
        raise RuntimeError("endpoint query features do not span the transformer feature space")
    theta, *_ = np.linalg.lstsq(design, logits_query, rcond=None)
    recovered_head = theta[:-1].T
    recovered_bias = theta[-1]
    recovered_delta = recovered_head - base_head
    u, singular_values, vt = np.linalg.svd(recovered_delta, full_matrices=False)
    rank_projected_delta = (u[:, :rank] * singular_values[:rank]) @ vt[:rank]
    return {
        "recovered_head": recovered_head,
        "recovered_bias": recovered_bias,
        "recovered_delta": recovered_delta,
        "rank_projected_delta": rank_projected_delta,
        "singular_values": singular_values,
        "feature_rank": feature_rank,
        "query_count": int(z_query.shape[0]),
    }


def _api_label_utility(
    exact_logits: np.ndarray, *, decimals: int | None = None, sig_digits: int | None = None
) -> float:
    """Fraction of held-out queries whose returned top-1 label survives quantization."""

    served = np.array(exact_logits, copy=True)
    if decimals is not None:
        served = np.round(served, decimals)
    if sig_digits is not None:
        served = quantize_significant(served, sig_digits)
    return float(np.mean(np.argmax(served, axis=1) == np.argmax(exact_logits, axis=1)))


def _api_distribution_cost(
    exact_logits: np.ndarray, *, decimals: int | None = None, sig_digits: int | None = None
) -> dict[str, float]:
    """What the quantizer costs a legitimate consumer of the probabilities.

    Top-1 survival is the cheapest possible reading of utility: it is blind to a
    caller that thresholds on confidence, ranks, or feeds the posterior
    downstream.  These are the same divergences used for recovery fidelity, but
    between the served and the exact answer rather than between the recovered and
    the true model, so the defense's cost and the attacker's gain are measured on
    one scale.
    """

    served = np.array(exact_logits, copy=True)
    if decimals is not None:
        served = np.round(served, decimals)
    if sig_digits is not None:
        served = quantize_significant(served, sig_digits)
    p = _softmax(np.asarray(exact_logits, dtype=np.float64))
    q = _softmax(served)
    eps = 1e-300
    return {
        "kl": max(0.0, float(np.mean(np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=1)))),
        "tv": float(np.max(0.5 * np.sum(np.abs(p - q), axis=1))),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=1, keepdims=True)


def _ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the top-1 prediction, equal-width bins."""

    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == np.asarray(labels)).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not np.any(m):
            continue
        total += float(np.mean(m)) * abs(float(np.mean(correct[m])) - float(np.mean(conf[m])))
    return total


def fidelity(
    recovered_logits: np.ndarray, true_logits: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Fidelity of a recovered model beyond top-1 agreement.

    Top-1 agreement saturates at 1.0 long before the recovered model matches the
    target, so it cannot separate a good recovery from a merely adequate one and
    says nothing about the predictive distribution.  These three do: ``kl`` is the
    mean KL divergence from the true to the recovered posterior in nats, ``tv``
    the worst total-variation distance between the two posteriors over the
    held-out set, and ``ece_gap`` the absolute difference in expected calibration
    error, which is the quantity a downstream consumer of the probabilities feels.
    """

    p = _softmax(np.asarray(true_logits, dtype=np.float64))
    q = _softmax(np.asarray(recovered_logits, dtype=np.float64))
    eps = 1e-300
    # Clamped at zero: an exact recovery lands at -1e-18 by roundoff, and a
    # negative divergence in a results table is a distraction, not a finding.
    kl = max(0.0, float(np.mean(np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=1))))
    tv = float(np.max(0.5 * np.sum(np.abs(p - q), axis=1)))
    return {
        "kl": kl,
        "tv": tv,
        "ece_gap": abs(_ece(q, labels) - _ece(p, labels)),
    }


def endpoint_logits(z: np.ndarray, base_head: np.ndarray, adapter_left: np.ndarray, adapter_right: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return z @ (base_head + adapter_left @ adapter_right.T).T + bias


def run_seed(args: argparse.Namespace, deps: dict[str, Any], seed: int) -> dict[str, float | int | str]:
    train_texts, train_labels, test_texts, test_labels, dataset_name = load_text_task(
        deps=deps,
        text_dataset=args.text_dataset,
        categories=tuple(args.categories),
        seed=seed,
        train_size=args.train_size,
        test_size=args.test_size,
        offline_mini_corpus=args.offline_mini_corpus,
    )
    all_texts = train_texts + test_texts
    _, _, z_all = transformer_cls_features(
        deps=deps,
        model_name=args.model,
        texts=all_texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    z_train = z_all[: len(train_texts)]
    z_test = z_all[len(train_texts) :]
    head = train_lowrank_endpoint_head(
        deps=deps,
        z_train=z_train,
        y_train=train_labels,
        z_test=z_test,
        y_test=test_labels,
        rank=args.rank,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=seed,
    )
    query_count = min(args.query_size, z_all.shape[0])
    z_query = z_all[:query_count]
    logits_query = endpoint_logits(
        z_query,
        head["base_head"],
        head["adapter_left"],
        head["adapter_right"],
        head["bias"],
    )
    recovery = recover_endpoint_head_from_final_logits(
        z_query=z_query,
        logits_query=logits_query,
        base_head=head["base_head"],
        rank=args.rank,
    )
    rounded_recovery = recover_endpoint_head_from_final_logits(
        z_query=z_query,
        logits_query=np.round(logits_query, args.rounded_decimals),
        base_head=head["base_head"],
        rank=args.rank,
    )
    # Relative quantization arm. On this endpoint the logit range is wide, so
    # one significant digit is emphatically not one decimal place -- which is the
    # setting the public-feature experiments could not exhibit, and the reason
    # this comparison is worth running at transformer scale.
    relative_recovery = recover_endpoint_head_from_final_logits(
        z_query=z_query,
        logits_query=quantize_significant(logits_query, args.sig_digits),
        base_head=head["base_head"],
        rank=args.rank,
    )
    rng = np.random.default_rng(seed + 100_000)
    noisy_recovery = recover_endpoint_head_from_final_logits(
        z_query=z_query,
        logits_query=logits_query + rng.normal(0.0, args.noise_std, size=logits_query.shape),
        base_head=head["base_head"],
        rank=args.rank,
    )

    true_delta = head["adapter_left"] @ head["adapter_right"].T
    recovered_logits_test = z_test @ (head["base_head"] + recovery["recovered_delta"]).T + recovery["recovered_bias"]
    rounded_logits_test = z_test @ (head["base_head"] + rounded_recovery["recovered_delta"]).T + rounded_recovery["recovered_bias"]
    noisy_logits_test = z_test @ (head["base_head"] + noisy_recovery["recovered_delta"]).T + noisy_recovery["recovered_bias"]
    relative_logits_test = z_test @ (head["base_head"] + relative_recovery["recovered_delta"]).T + relative_recovery["recovered_bias"]
    true_logits_test = endpoint_logits(
        z_test,
        head["base_head"],
        head["adapter_left"],
        head["adapter_right"],
        head["bias"],
    )
    # Distributional fidelity for each arm, so the paper is not resting a
    # transformer-scale claim on a metric that is already saturated at 1.0.
    fid = {
        name: fidelity(logits, true_logits_test, test_labels)
        for name, logits in (
            ("endpoint", recovered_logits_test),
            ("rounded_endpoint", rounded_logits_test),
            ("relative_endpoint", relative_logits_test),
            ("noisy_endpoint", noisy_logits_test),
        )
    }
    row = {
        "model": args.model,
        "dataset": dataset_name,
        "categories": ",".join(args.categories),
        "seed": seed,
        "rank": args.rank,
        "hidden_dim": int(z_train.shape[1]),
        "classes": int(head["base_head"].shape[0]),
        "train_examples": len(train_labels),
        "test_examples": len(test_labels),
        "query_count": recovery["query_count"],
        "feature_rank": recovery["feature_rank"],
        "train_accuracy": head["train_accuracy"],
        "test_accuracy": head["test_accuracy"],
        "endpoint_delta_relerr": relative_frobenius_error(recovery["recovered_delta"], true_delta),
        "endpoint_rank_projected_relerr": relative_frobenius_error(recovery["rank_projected_delta"], true_delta),
        "endpoint_bias_relerr": float(np.linalg.norm(recovery["recovered_bias"] - head["bias"]) / max(np.linalg.norm(head["bias"]), 1e-12)),
        "endpoint_logit_max_abs_error": float(np.max(np.abs(recovered_logits_test - true_logits_test))),
        "endpoint_label_agreement": float(
            np.mean(np.argmax(recovered_logits_test, axis=1) == np.argmax(true_logits_test, axis=1))
        ),
        "rounded_decimals": args.rounded_decimals,
        "rounded_endpoint_delta_relerr": relative_frobenius_error(rounded_recovery["recovered_delta"], true_delta),
        "rounded_endpoint_logit_max_abs_error": float(np.max(np.abs(rounded_logits_test - true_logits_test))),
        "rounded_endpoint_label_agreement": float(
            np.mean(np.argmax(rounded_logits_test, axis=1) == np.argmax(true_logits_test, axis=1))
        ),
        # Service cost of each quantizer, measured on the oracle alone: the
        # fraction of held-out queries whose returned top-1 label survives it.
        # Comparing recovery error at matched utility is what makes the
        # absolute-vs-relative gap a security statement rather than an artifact
        # of one quantizer being harsher than the other.
        "sig_digits": args.sig_digits,
        "absolute_api_label_utility": _api_label_utility(true_logits_test, decimals=args.rounded_decimals),
        "relative_api_label_utility": _api_label_utility(true_logits_test, sig_digits=args.sig_digits),
        "relative_endpoint_delta_relerr": relative_frobenius_error(relative_recovery["recovered_delta"], true_delta),
        "relative_endpoint_logit_max_abs_error": float(np.max(np.abs(relative_logits_test - true_logits_test))),
        "relative_endpoint_label_agreement": float(
            np.mean(np.argmax(relative_logits_test, axis=1) == np.argmax(true_logits_test, axis=1))
        ),
        "noise_std": args.noise_std,
        "noisy_endpoint_delta_relerr": relative_frobenius_error(noisy_recovery["recovered_delta"], true_delta),
        "noisy_endpoint_logit_max_abs_error": float(np.max(np.abs(noisy_logits_test - true_logits_test))),
        "noisy_endpoint_label_agreement": float(
            np.mean(np.argmax(noisy_logits_test, axis=1) == np.argmax(true_logits_test, axis=1))
        ),
    }
    for name, metrics in fid.items():
        for key, value in metrics.items():
            row[f"{name}_{key}"] = value
    # The service cost of each quantizer on the same divergences, so "cheap
    # defense" is a measured claim rather than an assertion about top-1 labels.
    for name, kw in (
        ("absolute_api", {"decimals": args.rounded_decimals}),
        ("relative_api", {"sig_digits": args.sig_digits}),
    ):
        for key, value in _api_distribution_cost(true_logits_test, **kw).items():
            row[f"{name}_{key}"] = value
    return row


def summarize_rows(rows: list[dict[str, float | int | str]]) -> str:
    if not rows:
        return "# Endpoint Transformer LoRA-Head Summary\n\nNo rows.\n"
    numeric_keys = [
        ("train_accuracy", "Train accuracy"),
        ("test_accuracy", "Test accuracy"),
        ("endpoint_delta_relerr", "Endpoint head delta recovery relerr"),
        ("endpoint_rank_projected_relerr", "Rank-projected relerr"),
        ("endpoint_bias_relerr", "Bias relerr"),
        ("endpoint_logit_max_abs_error", "Held-out max logit error"),
        ("endpoint_label_agreement", "Held-out endpoint label agreement"),
        ("absolute_api_label_utility", "Absolute-grid API label utility"),
        ("relative_api_label_utility", "Relative-quant API label utility"),
        ("rounded_endpoint_delta_relerr", "Rounded-logit delta relerr"),
        ("relative_endpoint_delta_relerr", "Relative-quant delta relerr"),
        ("rounded_endpoint_label_agreement", "Rounded-logit label agreement"),
        ("noisy_endpoint_delta_relerr", "Noisy-logit delta relerr"),
        ("noisy_endpoint_label_agreement", "Noisy-logit label agreement"),
        ("query_count", "Endpoint final-logit queries"),
        ("feature_rank", "Augmented feature rank"),
        # Fidelity beyond top-1, which saturates on this endpoint.
        ("endpoint_kl", "Exact-oracle mean KL (nats)"),
        ("endpoint_tv", "Exact-oracle max total variation"),
        ("endpoint_ece_gap", "Exact-oracle calibration-error gap"),
        ("rounded_endpoint_kl", "Rounded-logit mean KL (nats)"),
        ("rounded_endpoint_tv", "Rounded-logit max total variation"),
        ("rounded_endpoint_ece_gap", "Rounded-logit calibration-error gap"),
        ("relative_endpoint_kl", "Relative-quant mean KL (nats)"),
        ("relative_endpoint_tv", "Relative-quant max total variation"),
        ("relative_endpoint_ece_gap", "Relative-quant calibration-error gap"),
        ("noisy_endpoint_kl", "Noisy-logit mean KL (nats)"),
        ("noisy_endpoint_tv", "Noisy-logit max total variation"),
        ("noisy_endpoint_ece_gap", "Noisy-logit calibration-error gap"),
        ("absolute_api_kl", "Absolute-grid API served KL (nats)"),
        ("absolute_api_tv", "Absolute-grid API served max total variation"),
        ("relative_api_kl", "Relative-quant API served KL (nats)"),
        ("relative_api_tv", "Relative-quant API served max total variation"),
    ]
    lines = [
        "# Endpoint Transformer LoRA-Head Summary",
        "",
        f"Base transformer: `{rows[0]['model']}`",
        f"Dataset: `{rows[0]['dataset']}`",
        "",
        "| metric | mean | std |",
        "|---|---:|---:|",
    ]
    for key, label in numeric_keys:
        # Rows produced by older runs (and by the smoke fixtures) predate the
        # relative-quantization arm, so a metric absent from every row is simply
        # skipped rather than failing the summary.
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        lines.append(f"| {label} | {mean(values):.4g} | {(pstdev(values) if len(values) > 1 else 0.0):.4g} |")
    lines.extend(
        [
            "",
            "Interpretation: this is a positive endpoint-only transformer experiment for an observable LoRA-style classifier head. The attacker queries text inputs and receives only final logits. Because the frozen transformer feature map is public and the adapted component is the endpoint head, final logits provide linear equations in the low-rank head update.",
            "",
            "Robustness: rounded and noisy final-logit rows report how much recovery degrades when the endpoint API does not return exact scores.",
            "",
            "Scope: this does not contradict the hidden-adapter nullspace result. Hidden transformer LoRA modules remain unidentifiable from endpoint logits unless their effects are observable through the final classifier head.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    deps = _require_deps()
    args.outdir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    rows = [run_seed(args, deps, seed) for seed in seeds]

    json_path = args.outdir / "endpoint_transformer.json"
    csv_path = args.outdir / "endpoint_transformer.csv"
    md_path = args.outdir / "summary.md"
    payload: dict[str, float | int | str] | list[dict[str, float | int | str]] = rows[0] if len(rows) == 1 else rows
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_rows(rows)
    md_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
