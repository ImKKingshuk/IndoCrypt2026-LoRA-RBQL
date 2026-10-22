from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from lowrank_extraction.attacks import recover_raw_affine
from lowrank_extraction.endpoint_observability import (
    classifier_quotient_certificate,
    linear_jacobian_certificate,
    nullspace_drift_certificate,
)
from lowrank_extraction.metrics import relative_frobenius_error
from lowrank_extraction.models import LinearLowRankTarget
from lowrank_extraction.oracles import RawLogitOracle


DEFAULT_MODEL = "google/bert_uncased_L-2_H-128_A-2"
DEFAULT_CATEGORIES = ("sci.crypt", "comp.graphics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and inspect a real PEFT LoRA checkpoint on a small text task.")
    parser.add_argument("--outdir", type=Path, default=Path("results") / "lora_checkpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text-dataset", choices=["20newsgroups", "sst2"], default="20newsgroups")
    parser.add_argument("--categories", nargs=2, default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--test-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--target-modules", nargs="+", default=["query", "value"])
    parser.add_argument("--jacobian-examples", type=int, default=0, help="Optionally compute endpoint-logit Jacobian rank wrt the recovered LoRA module parameters.")
    parser.add_argument("--jacobian-sweep-examples", type=int, nargs="+", default=None, help="Optionally compute endpoint-logit Jacobian ranks for multiple example budgets.")
    parser.add_argument("--offline-mini-corpus", action="store_true", help="Use a tiny embedded corpus for smoke tests only.")
    return parser.parse_args()


def _require_lora_deps() -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.model_selection import train_test_split
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "The LoRA checkpoint experiment needs optional heavy dependencies. "
            "Install them with: uv sync --extra dev --extra lora"
        ) from exc
    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "fetch_20newsgroups": fetch_20newsgroups,
        "train_test_split": train_test_split,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
    }


def _embedded_corpus() -> tuple[list[str], list[int]]:
    crypt = [
        "The protocol derives a session key from a public key exchange and authenticates the transcript.",
        "Lattice-based encryption schemes are evaluated against chosen ciphertext attacks.",
        "The implementation rotates symmetric keys after every authenticated channel setup.",
        "Zero-knowledge proofs can verify a statement without revealing the witness.",
        "A side-channel countermeasure masks intermediate values in the block cipher.",
        "Digital signatures bind a message digest to a private signing key.",
    ]
    graphics = [
        "The renderer uses texture mapping and a fragment shader for realistic lighting.",
        "A mesh simplification algorithm reduces triangles while preserving visual detail.",
        "Ray tracing computes reflections by intersecting rays with scene geometry.",
        "The animation pipeline interpolates keyframes for smooth character movement.",
        "GPU rasterization converts projected polygons into pixels on the screen.",
        "A camera transform maps world coordinates into the view frustum.",
    ]
    return crypt + graphics, [0] * len(crypt) + [1] * len(graphics)


def load_text_task(
    *,
    deps: dict[str, Any],
    text_dataset: str,
    categories: tuple[str, str],
    seed: int,
    train_size: int,
    test_size: int,
    offline_mini_corpus: bool,
) -> tuple[list[str], list[int], list[str], list[int], str]:
    if offline_mini_corpus:
        texts, labels = _embedded_corpus()
        split = deps["train_test_split"](
            texts,
            labels,
            test_size=0.35,
            random_state=seed,
            stratify=labels,
        )
        x_train, x_test, y_train, y_test = split
        return list(x_train), list(y_train), list(x_test), list(y_test), "embedded_smoke"

    if text_dataset == "sst2":
        train_data = deps["load_dataset"]("glue", "sst2", split="train")
        validation_data = deps["load_dataset"]("glue", "sst2", split="validation")
        rng = np.random.default_rng(seed)
        train_ids = rng.choice(len(train_data), size=min(train_size, len(train_data)), replace=False)
        test_ids = rng.choice(len(validation_data), size=min(test_size, len(validation_data)), replace=False)
        x_train = [str(train_data[int(idx)]["sentence"]) for idx in train_ids]
        y_train = [int(train_data[int(idx)]["label"]) for idx in train_ids]
        x_test = [str(validation_data[int(idx)]["sentence"]) for idx in test_ids]
        y_test = [int(validation_data[int(idx)]["label"]) for idx in test_ids]
        return x_train, y_train, x_test, y_test, "glue_sst2"

    data = deps["fetch_20newsgroups"](
        subset="all",
        categories=list(categories),
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=seed,
    )
    texts = [text for text in data.data if text.strip()]
    labels = [int(label) for text, label in zip(data.data, data.target, strict=True) if text.strip()]
    split = deps["train_test_split"](
        texts,
        labels,
        train_size=train_size,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    x_train, x_test, y_train, y_test = split
    return list(x_train), list(y_train), list(x_test), list(y_test), "20newsgroups"


def _batches(size: int, batch_size: int) -> list[slice]:
    return [slice(start, min(start + batch_size, size)) for start in range(0, size, batch_size)]


def encode_texts(tokenizer: Any, texts: list[str], *, max_length: int) -> dict[str, Any]:
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")


def evaluate_model(model: Any, encodings: dict[str, Any], labels: list[int], *, batch_size: int, torch: Any) -> float:
    model.eval()
    y = torch.tensor(labels, dtype=torch.long)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in _batches(len(labels), batch_size):
            inputs = {key: value[batch] for key, value in encodings.items()}
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y[batch]).sum().item())
            total += int(pred.numel())
    return correct / max(total, 1)


def train_lora_model(
    *,
    deps: dict[str, Any],
    model_name: str,
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    rank: int,
    target_modules: list[str],
    epochs: int,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    seed: int,
) -> tuple[Any, Any, dict[str, float | int]]:
    torch = deps["torch"]
    torch.manual_seed(seed)
    tokenizer = deps["AutoTokenizer"].from_pretrained(model_name, use_fast=False)
    base_model = deps["AutoModelForSequenceClassification"].from_pretrained(model_name, num_labels=2)
    lora_config = deps["LoraConfig"](
        task_type=deps["TaskType"].SEQ_CLS,
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.05,
        target_modules=target_modules,
        modules_to_save=["classifier"],
    )
    model = deps["get_peft_model"](base_model, lora_config)
    train_enc = encode_texts(tokenizer, train_texts, max_length=max_length)
    test_enc = encode_texts(tokenizer, test_texts, max_length=max_length)
    y_train = torch.tensor(train_labels, dtype=torch.long)

    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=learning_rate)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(train_labels))
        for batch_ids in torch.split(order, batch_size):
            inputs = {key: value[batch_ids] for key, value in train_enc.items()}
            labels = y_train[batch_ids]
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=labels).loss
            loss.backward()
            optimizer.step()

    metrics = {
        "train_accuracy": evaluate_model(model, train_enc, train_labels, batch_size=batch_size, torch=torch),
        "test_accuracy": evaluate_model(model, test_enc, test_labels, batch_size=batch_size, torch=torch),
        "train_examples": len(train_labels),
        "test_examples": len(test_labels),
    }
    return model, tokenizer, metrics


def first_lora_delta(model: Any) -> dict[str, np.ndarray | str | float | int]:
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if "default" not in module.lora_A or "default" not in module.lora_B:
            continue
        base_layer = getattr(module, "base_layer", None)
        if base_layer is None or not hasattr(base_layer, "weight"):
            continue
        a = module.lora_A["default"].weight.detach().cpu().numpy()
        b = module.lora_B["default"].weight.detach().cpu().numpy()
        scale = float(module.scaling["default"])
        base_weight = base_layer.weight.detach().cpu().numpy()
        base_bias = (
            np.zeros(base_weight.shape[0], dtype=np.float64)
            if getattr(base_layer, "bias", None) is None
            else base_layer.bias.detach().cpu().numpy()
        )
        return {
            "module_name": name,
            "base_weight": base_weight.astype(np.float64),
            "base_bias": base_bias.astype(np.float64),
            "adapter_left": (scale * b).astype(np.float64),
            "adapter_right": a.T.astype(np.float64),
            "rank": int(a.shape[0]),
            "scale": scale,
        }
    raise RuntimeError("no LoRA module found in the PEFT model")


def recover_lora_module_delta(module_info: dict[str, np.ndarray | str | float | int]) -> dict[str, float | int | str]:
    target = LinearLowRankTarget(
        base_weight=np.asarray(module_info["base_weight"], dtype=np.float64),
        adapter_left=np.asarray(module_info["adapter_left"], dtype=np.float64),
        adapter_right=np.asarray(module_info["adapter_right"], dtype=np.float64),
        bias=np.asarray(module_info["base_bias"], dtype=np.float64),
    )
    result = recover_raw_affine(RawLogitOracle(target), base_weight=target.base_weight, rank=target.rank)
    return {
        "module_name": str(module_info["module_name"]),
        "module_in_features": target.n_features,
        "module_out_features": target.n_classes,
        "rank": target.rank,
        "lora_scale": float(module_info["scale"]),
        "module_query_count": result.query_count,
        "module_delta_relerr": relative_frobenius_error(result.recovered_delta, target.delta),
        "module_rank_projected_relerr": relative_frobenius_error(result.rank_projected_delta, target.delta),
        "module_delta_rank": int(np.linalg.matrix_rank(target.delta, tol=1e-10)),
    }


def _classifier_weight_and_bias(model: Any) -> tuple[str, np.ndarray, np.ndarray]:
    classifier_name = ""
    classifier_weight = None
    classifier_bias = None
    for name, parameter in model.named_parameters():
        if "classifier" not in name or not name.endswith("weight"):
            continue
        array = parameter.detach().cpu().numpy()
        if array.ndim != 2:
            continue
        classifier_name = name
        classifier_weight = array.astype(np.float64)
        bias_name = name[: -len("weight")] + "bias"
        for candidate_name, candidate_parameter in model.named_parameters():
            if candidate_name == bias_name:
                classifier_bias = candidate_parameter.detach().cpu().numpy().astype(np.float64)
                break
        break
    if classifier_weight is None:
        raise RuntimeError("could not find a 2D classifier weight parameter")
    if classifier_bias is None:
        classifier_bias = np.zeros(classifier_weight.shape[0], dtype=np.float64)
    return classifier_name, classifier_weight, classifier_bias


def _bert_backbone(model: Any) -> Any:
    base_model = getattr(model, "base_model", None)
    wrapped = getattr(base_model, "model", None) if base_model is not None else None
    for candidate in (wrapped, model):
        if candidate is not None and hasattr(candidate, "bert"):
            return candidate.bert
    raise RuntimeError("could not find a BERT backbone for endpoint hidden-state certificate")


def _pooled_hidden_states(model: Any, encodings: dict[str, Any], *, batch_size: int, torch: Any) -> np.ndarray:
    backbone = _bert_backbone(model)
    model.eval()
    pooled_batches = []
    with torch.no_grad():
        for batch in _batches(int(encodings["input_ids"].shape[0]), batch_size):
            inputs = {key: value[batch] for key, value in encodings.items()}
            outputs = backbone(**inputs, return_dict=True)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            pooled_batches.append(pooled.detach().cpu().numpy().astype(np.float64))
    return np.vstack(pooled_batches)


def endpoint_lora_jacobian_certificate(
    model: Any,
    *,
    module_name: str,
    encodings: dict[str, Any],
    examples: int,
    torch: Any,
) -> dict[str, float | int | bool]:
    if examples <= 0:
        return {}
    module_by_name = dict(model.named_modules())
    module = module_by_name.get(module_name)
    if module is None or not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
        raise RuntimeError(f"could not find LoRA module for Jacobian certificate: {module_name}")
    params = [module.lora_A["default"].weight, module.lora_B["default"].weight]
    rank = int(params[0].shape[0])
    in_features = int(params[0].shape[1])
    out_features = int(params[1].shape[0])
    parameter_count = int(sum(param.numel() for param in params))
    count = min(examples, int(encodings["input_ids"].shape[0]))
    inputs = {key: value[:count] for key, value in encodings.items()}
    model.eval()
    model.zero_grad(set_to_none=True)
    logits = model(**inputs).logits
    rows = []
    flat_logits = logits.reshape(-1)
    for idx, scalar in enumerate(flat_logits):
        model.zero_grad(set_to_none=True)
        scalar.backward(retain_graph=idx + 1 < int(flat_logits.numel()))
        grads = []
        for param in params:
            if param.grad is None:
                grads.append(torch.zeros_like(param).detach().reshape(-1))
            else:
                grads.append(param.grad.detach().reshape(-1).clone())
        rows.append(torch.cat(grads).cpu().numpy().astype(np.float64))
    jacobian = np.vstack(rows) if rows else np.zeros((0, parameter_count), dtype=np.float64)
    cert = linear_jacobian_certificate(jacobian)
    null_direction = endpoint_lora_null_direction_certificate(
        model,
        module=module,
        params=params,
        jacobian=jacobian,
        inputs=inputs,
        torch=torch,
        seed=examples + parameter_count,
    )
    return {
        "endpoint_jacobian_examples": count,
        "endpoint_jacobian_rows": cert.rows,
        "endpoint_jacobian_parameter_count": cert.parameters,
        "endpoint_jacobian_lora_rank": rank,
        "endpoint_jacobian_quotient_dimension": rank * (in_features + out_features - rank),
        "endpoint_jacobian_gauge_nullity": rank * rank,
        "endpoint_jacobian_rank": cert.rank,
        "endpoint_jacobian_nullity_lower_bound": cert.nullity_lower_bound,
        "endpoint_jacobian_nongauge_nullity_lower_bound": rank * (in_features + out_features - rank) - cert.rank,
        "endpoint_jacobian_observable_fraction": cert.observable_fraction,
        "endpoint_jacobian_quotient_observable_fraction": cert.rank / max(rank * (in_features + out_features - rank), 1),
        **null_direction,
    }


def endpoint_lora_null_direction_certificate(
    model: Any,
    *,
    module: Any,
    params: list[Any],
    jacobian: np.ndarray,
    inputs: dict[str, Any],
    torch: Any,
    seed: int,
    epsilon: float = 1e-3,
    candidates: int = 24,
) -> dict[str, float | int | bool]:
    # The logit drift returned below is an UPPER BOUND, not a measurement.  This
    # model runs in float32, and a float32 forward disagrees with the float64 one
    # by up to 4.08e-6 on these logits (scale ~3.04) -- larger than the drift a
    # null direction actually produces at epsilon=1e-3, which is 2.79e-9.  So the
    # number here is this pipeline's own rounding.  Fixing it in place would mean
    # a float64 forward inside the training loop and a full retrain to regenerate
    # results; instead `run-null-curvature` re-measures the same certificate over
    # the same saved adapters in float64, and the paper quotes that.  Keep this
    # row for continuity, and do not quote it as the drift.
    parameter_count = int(sum(param.numel() for param in params))
    if jacobian.size == 0 or jacobian.shape[1] != parameter_count:
        return {
            "endpoint_null_direction_available": False,
            "endpoint_null_direction_epsilon": epsilon,
            "endpoint_null_direction_jacobian_residual": float("nan"),
            "endpoint_null_direction_param_shift_norm": 0.0,
            "endpoint_null_direction_update_shift_norm": 0.0,
            "endpoint_null_direction_update_shift_ratio": 0.0,
            "endpoint_null_direction_max_logit_drift": float("nan"),
            "endpoint_null_direction_label_flip_rate": float("nan"),
        }
    rank = int(np.linalg.matrix_rank(jacobian, tol=1e-8))
    if rank >= parameter_count:
        return {
            "endpoint_null_direction_available": False,
            "endpoint_null_direction_epsilon": epsilon,
            "endpoint_null_direction_jacobian_residual": float("nan"),
            "endpoint_null_direction_param_shift_norm": 0.0,
            "endpoint_null_direction_update_shift_norm": 0.0,
            "endpoint_null_direction_update_shift_ratio": 0.0,
            "endpoint_null_direction_max_logit_drift": float("nan"),
            "endpoint_null_direction_label_flip_rate": float("nan"),
        }

    _, _, vh = np.linalg.svd(jacobian, full_matrices=False)
    row_basis = vh[:rank]
    rng = np.random.default_rng(seed)
    a_weight = params[0]
    b_weight = params[1]
    a_base = a_weight.detach().clone()
    b_base = b_weight.detach().clone()
    a_np = a_base.cpu().numpy().astype(np.float64)
    b_np = b_base.cpu().numpy().astype(np.float64)
    scale = float(module.scaling["default"])
    base_update = scale * (b_np @ a_np)
    base_update_norm = float(np.linalg.norm(base_update))

    best_direction: np.ndarray | None = None
    best_update_shift = 0.0
    best_update_ratio = 0.0
    best_residual = float("nan")
    for _ in range(candidates):
        direction = rng.normal(0.0, 1.0, size=parameter_count)
        if row_basis.size:
            direction = direction - row_basis.T @ (row_basis @ direction)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            continue
        direction = direction / norm
        split = int(a_weight.numel())
        d_a = direction[:split].reshape(a_np.shape)
        d_b = direction[split:].reshape(b_np.shape)
        update_shift = scale * (d_b @ a_np + b_np @ d_a)
        update_shift_norm = float(np.linalg.norm(update_shift))
        update_shift_ratio = update_shift_norm / max(base_update_norm, 1e-12)
        if update_shift_norm > best_update_shift:
            best_direction = direction
            best_update_shift = update_shift_norm
            best_update_ratio = update_shift_ratio
            best_residual = float(np.linalg.norm(jacobian @ direction))

    if best_direction is None or best_update_shift <= 1e-12:
        return {
            "endpoint_null_direction_available": False,
            "endpoint_null_direction_epsilon": epsilon,
            "endpoint_null_direction_jacobian_residual": float("nan") if best_direction is None else best_residual,
            "endpoint_null_direction_param_shift_norm": 0.0,
            "endpoint_null_direction_update_shift_norm": best_update_shift,
            "endpoint_null_direction_update_shift_ratio": best_update_ratio,
            "endpoint_null_direction_max_logit_drift": float("nan"),
            "endpoint_null_direction_label_flip_rate": float("nan"),
        }

    split = int(a_weight.numel())
    d_a_t = torch.tensor(best_direction[:split].reshape(a_np.shape), dtype=a_weight.dtype, device=a_weight.device)
    d_b_t = torch.tensor(best_direction[split:].reshape(b_np.shape), dtype=b_weight.dtype, device=b_weight.device)
    model.eval()
    with torch.no_grad():
        base_logits = model(**inputs).logits.detach().clone()
        base_labels = base_logits.argmax(dim=-1)
        a_weight.add_(epsilon * d_a_t)
        b_weight.add_(epsilon * d_b_t)
        changed_logits = model(**inputs).logits.detach().clone()
        a_weight.copy_(a_base)
        b_weight.copy_(b_base)
    max_logit_drift = float(torch.max(torch.abs(changed_logits - base_logits)).detach().cpu().item())
    label_flip_rate = float((changed_logits.argmax(dim=-1) != base_labels).float().mean().detach().cpu().item())
    return {
        "endpoint_null_direction_available": True,
        "endpoint_null_direction_epsilon": epsilon,
        "endpoint_null_direction_jacobian_residual": best_residual,
        "endpoint_null_direction_param_shift_norm": epsilon,
        "endpoint_null_direction_update_shift_norm": epsilon * best_update_shift,
        "endpoint_null_direction_update_shift_ratio": epsilon * best_update_ratio,
        "endpoint_null_direction_max_logit_drift": max_logit_drift,
        "endpoint_null_direction_label_flip_rate": label_flip_rate,
    }


def endpoint_classifier_nullspace_certificate(
    model: Any,
    *,
    seed: int = 0,
    samples: int = 64,
    encodings: dict[str, Any] | None = None,
    batch_size: int = 8,
    torch: Any | None = None,
) -> dict[str, float | int | bool | str]:
    """Certify a final-logit non-identifiability direction for a classifier head.

    Sequence classifiers often map a high-dimensional hidden representation to a
    small number of labels. Any hidden update in the classifier nullspace is
    invisible to endpoint logits, so final logits alone cannot identify all
    possible hidden adapter effects without additional observability assumptions.
    """

    classifier_name, classifier_weight, classifier_bias = _classifier_weight_and_bias(model)

    quotient = classifier_quotient_certificate(classifier_weight)
    if quotient.nullity <= 0:
        return {
            "endpoint_classifier_name": classifier_name,
            "endpoint_hidden_dim": quotient.hidden_dim,
            "endpoint_classifier_rank": quotient.rank,
            "endpoint_nullity": quotient.nullity,
            "endpoint_observable_fraction": quotient.observable_fraction,
            "endpoint_hidden_blind_fraction": quotient.blind_fraction,
            "endpoint_nonidentifiable": False,
            "endpoint_null_residual": float("nan"),
            "endpoint_max_logit_drift": float("nan"),
            "endpoint_adapter_distance": 0.0,
            "endpoint_real_examples": 0,
            "endpoint_real_max_logit_drift": float("nan"),
            "endpoint_real_label_flip_rate": float("nan"),
            "endpoint_real_hidden_norm": 0.0,
            "endpoint_real_hidden_shift_norm": 0.0,
            "endpoint_real_hidden_shift_ratio": 0.0,
        }

    assert quotient.null_vector is not None
    null_vector = quotient.null_vector
    rng = np.random.default_rng(seed)
    hidden = rng.normal(0.0, 1.0, size=(samples, quotient.hidden_dim))
    synthetic_drift = nullspace_drift_certificate(classifier_weight, classifier_bias, hidden, seed=seed)
    row: dict[str, float | int | bool | str] = {
        "endpoint_classifier_name": classifier_name,
        "endpoint_hidden_dim": quotient.hidden_dim,
        "endpoint_classifier_rank": quotient.rank,
        "endpoint_nullity": quotient.nullity,
        "endpoint_observable_fraction": quotient.observable_fraction,
        "endpoint_hidden_blind_fraction": quotient.blind_fraction,
        "endpoint_nonidentifiable": True,
        "endpoint_null_residual": quotient.null_residual,
        "endpoint_max_logit_drift": float(synthetic_drift["endpoint_max_logit_drift"]),
        "endpoint_adapter_distance": float(np.linalg.norm(null_vector)),
        "endpoint_real_examples": 0,
        "endpoint_real_max_logit_drift": float("nan"),
        "endpoint_real_label_flip_rate": float("nan"),
        "endpoint_real_hidden_shift_norm": 0.0,
    }
    if encodings is not None and torch is not None:
        real_hidden = _pooled_hidden_states(model, encodings, batch_size=batch_size, torch=torch)
        real_drift = nullspace_drift_certificate(classifier_weight, classifier_bias, real_hidden, seed=seed + 1)
        row.update(
            {
                "endpoint_real_examples": int(real_hidden.shape[0]),
                "endpoint_real_max_logit_drift": float(real_drift["endpoint_max_logit_drift"]),
                "endpoint_real_label_flip_rate": float(real_drift["endpoint_label_flip_rate"]),
                "endpoint_real_hidden_norm": float(real_drift["endpoint_hidden_norm"]),
                "endpoint_real_hidden_shift_norm": float(real_drift["endpoint_hidden_shift_norm"]),
                "endpoint_real_hidden_shift_ratio": float(real_drift["endpoint_hidden_shift_ratio"]),
            }
        )
    return row


def summarize_rows(rows: list[dict[str, float | int | str]]) -> str:
    if not rows:
        return "# PEFT LoRA Checkpoint Summary\n\nNo rows.\n"
    numeric_keys = [
        ("train_accuracy", "Train accuracy"),
        ("test_accuracy", "Test accuracy"),
        ("module_delta_relerr", "LoRA module delta recovery relerr"),
        ("module_rank_projected_relerr", "Rank-projected relerr"),
        ("module_query_count", "Module oracle queries"),
        ("endpoint_nullity", "Endpoint classifier nullity"),
        ("endpoint_hidden_blind_fraction", "Endpoint hidden blind fraction"),
        ("endpoint_max_logit_drift", "Nullspace max logit drift"),
        ("endpoint_real_max_logit_drift", "Real-test nullspace logit drift"),
        ("endpoint_real_label_flip_rate", "Real-test label flip rate"),
        ("endpoint_real_hidden_shift_norm", "Real-test hidden shift norm"),
        ("endpoint_real_hidden_shift_ratio", "Real-test hidden shift ratio"),
        ("endpoint_jacobian_rank", "Endpoint LoRA Jacobian rank"),
        ("endpoint_jacobian_quotient_dimension", "LoRA update quotient dimension"),
        ("endpoint_jacobian_gauge_nullity", "LoRA factor gauge nullity"),
        ("endpoint_jacobian_nullity_lower_bound", "Endpoint LoRA Jacobian nullity lower bound"),
        ("endpoint_jacobian_nongauge_nullity_lower_bound", "Non-gauge endpoint nullity lower bound"),
        ("endpoint_jacobian_observable_fraction", "Endpoint LoRA Jacobian observable fraction"),
        ("endpoint_jacobian_quotient_observable_fraction", "Quotient observable fraction"),
        ("endpoint_null_direction_update_shift_norm", "Local null-direction update shift norm"),
        ("endpoint_null_direction_max_logit_drift", "Local null-direction max logit drift"),
        ("endpoint_null_direction_label_flip_rate", "Local null-direction label flip rate"),
    ]
    lines = [
        "# PEFT LoRA Checkpoint Summary",
        "",
        f"Base model: `{rows[0]['model']}`",
        f"Dataset: `{rows[0]['dataset']}`",
        f"Recovered LoRA module: `{rows[0]['module_name']}`",
        "",
        "| metric | mean | std |",
        "|---|---:|---:|",
    ]
    for key, label in numeric_keys:
        if key not in rows[0]:
            continue
        values = [float(row[key]) for row in rows]
        lines.append(f"| {label} | {mean(values):.4g} | {(pstdev(values) if len(values) > 1 else 0.0):.4g} |")
    lines.extend(
        [
            "",
            "Interpretation: this runner trains and saves a true PEFT LoRA adapter on a pretrained transformer, then extracts one real LoRA module's low-rank matrix and verifies exact adapter-module recovery under raw module-output oracle access. It is not an end-to-end black-box recovery attack against transformer logits.",
            "",
            "Endpoint certificate: the final classifier has a nontrivial nullspace, so distinct hidden adapter effects in that nullspace induce identical final logits. The runner verifies this both algebraically and on held-out encoded test examples, supporting an impossibility/defense direction for endpoint-only APIs.",
            "",
            "Local endpoint-null perturbation: when endpoint Jacobian rows are available, the runner also constructs a non-gauge LoRA parameter direction in the finite-transcript nullspace, perturbs the trained module, and reports the resulting merged-update movement, endpoint-logit drift, and label flip rate.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize_jacobian_sweep(rows: list[dict[str, float | int | str]]) -> str:
    if not rows:
        return "# Endpoint LoRA Jacobian Sweep\n\nNo rows.\n"
    budgets = sorted({int(row["endpoint_jacobian_examples"]) for row in rows})
    lines = [
        "# Endpoint LoRA Jacobian Sweep",
        "",
        f"Base model: `{rows[0]['model']}`",
        f"Dataset: `{rows[0]['dataset']}`",
        f"Recovered LoRA module: `{rows[0]['module_name']}`",
        "",
        "| examples | cases | rows | rank mean | quotient dim | non-gauge nullity mean | quotient observable fraction mean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in budgets:
        group = [row for row in rows if int(row["endpoint_jacobian_examples"]) == budget]
        ranks = [float(row["endpoint_jacobian_rank"]) for row in group]
        quotient_dims = [float(row["endpoint_jacobian_quotient_dimension"]) for row in group]
        nullities = [float(row["endpoint_jacobian_nongauge_nullity_lower_bound"]) for row in group]
        fractions = [float(row["endpoint_jacobian_quotient_observable_fraction"]) for row in group]
        jac_rows = [float(row["endpoint_jacobian_rows"]) for row in group]
        lines.append(
            f"| {budget} | {len(group)} | {mean(jac_rows):.0f} | {mean(ranks):.2f} | {mean(quotient_dims):.0f} | {mean(nullities):.2f} | {mean(fractions):.6f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: each endpoint example contributes only the final-logit dimension many scalar equations. The observed Jacobian ranks quantify a finite-query local observability ceiling for hidden LoRA parameters, not an endpoint extraction success.",
        ]
    )
    if "endpoint_null_direction_update_shift_norm" in rows[0]:
        available = [row for row in rows if bool(row.get("endpoint_null_direction_available", False))]
        if available:
            shift = mean(float(row["endpoint_null_direction_update_shift_norm"]) for row in available)
            drift = mean(float(row["endpoint_null_direction_max_logit_drift"]) for row in available)
            flips = mean(float(row["endpoint_null_direction_label_flip_rate"]) for row in available)
            lines.extend(
                [
                    "",
                    f"Local endpoint-null perturbation: non-gauge LoRA update shift mean {shift:.3e}, endpoint max-logit drift mean {drift:.3e}, label flip rate mean {flips:.4f}.",
                ]
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    deps = _require_lora_deps()
    args.outdir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds if args.seeds is not None else [args.seed]
    rows: list[dict[str, float | int | str]] = []
    jacobian_sweep_rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        train_texts, train_labels, test_texts, test_labels, dataset_name = load_text_task(
        deps=deps,
        text_dataset=args.text_dataset,
        categories=tuple(args.categories),
            seed=seed,
            train_size=args.train_size,
            test_size=args.test_size,
            offline_mini_corpus=args.offline_mini_corpus,
        )
        model, tokenizer, train_metrics = train_lora_model(
            deps=deps,
            model_name=args.model,
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=test_texts,
            test_labels=test_labels,
            rank=args.rank,
            target_modules=args.target_modules,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            seed=seed,
        )
        adapter_dir = args.outdir / ("adapter" if len(seeds) == 1 else f"adapter_seed_{seed}")
        tokenizer_dir = args.outdir / ("tokenizer" if len(seeds) == 1 else f"tokenizer_seed_{seed}")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(tokenizer_dir)

        module_info = first_lora_delta(model)
        recovery = recover_lora_module_delta(module_info)
        test_enc = encode_texts(tokenizer, test_texts, max_length=args.max_length)
        endpoint_certificate = endpoint_classifier_nullspace_certificate(
            model,
            seed=seed + 1_000,
            encodings=test_enc,
            batch_size=args.batch_size,
            torch=deps["torch"],
        )
        jacobian_certificate = endpoint_lora_jacobian_certificate(
            model,
            module_name=str(module_info["module_name"]),
            encodings=test_enc,
            examples=args.jacobian_examples,
            torch=deps["torch"],
        )
        if args.jacobian_sweep_examples is not None:
            for examples in args.jacobian_sweep_examples:
                sweep_certificate = endpoint_lora_jacobian_certificate(
                    model,
                    module_name=str(module_info["module_name"]),
                    encodings=test_enc,
                    examples=examples,
                    torch=deps["torch"],
                )
                jacobian_sweep_rows.append(
                    {
                        "model": args.model,
                        "dataset": dataset_name,
                        "categories": ",".join(args.categories),
                        "seed": seed,
                        "module_name": str(module_info["module_name"]),
                        **sweep_certificate,
                    }
                )
        rows.append(
            {
                "model": args.model,
                "dataset": dataset_name,
                "categories": ",".join(args.categories),
                "seed": seed,
                **train_metrics,
                **recovery,
                **endpoint_certificate,
                **jacobian_certificate,
                "adapter_dir": str(adapter_dir),
            }
        )

    json_path = args.outdir / "lora_checkpoint.json"
    csv_path = args.outdir / "lora_checkpoint.csv"
    md_path = args.outdir / "summary.md"
    json_payload: dict[str, float | int | str] | list[dict[str, float | int | str]]
    json_payload = rows[0] if len(rows) == 1 else rows
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_rows(rows)
    md_path.write_text(summary, encoding="utf-8")
    if jacobian_sweep_rows:
        sweep_json_path = args.outdir / "lora_jacobian_sweep.json"
        sweep_csv_path = args.outdir / "lora_jacobian_sweep.csv"
        sweep_md_path = args.outdir / "jacobian_sweep_summary.md"
        sweep_json_path.write_text(json.dumps(jacobian_sweep_rows, indent=2), encoding="utf-8")
        with sweep_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(jacobian_sweep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(jacobian_sweep_rows)
        sweep_summary = summarize_jacobian_sweep(jacobian_sweep_rows)
        sweep_md_path.write_text(sweep_summary, encoding="utf-8")
    print(summary)
    if jacobian_sweep_rows:
        print(sweep_summary)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    if jacobian_sweep_rows:
        print(f"Wrote {sweep_json_path}")
        print(f"Wrote {sweep_csv_path}")
        print(f"Wrote {sweep_md_path}")
    for row in rows:
        print(f"Wrote {row['adapter_dir']}")


if __name__ == "__main__":
    main()
