"""The amplification lever on an ordinary vision prediction API.

The transformer-scale amplification result is measured through ``O_module``, an
interface that bypasses the network's normalization.  That is real but it is not
the most common deployment.  This experiment asks whether the lever
survives on a plain ``O_end`` prediction API.  A vision endpoint is the honest
test: inputs are continuous, so unlike a text endpoint there is nothing stopping
the attacker from scaling them.

Panel A asks which architectures are exposed, and finds that the answer is
decided by the normalization layer:

  * **BatchNorm** in eval mode is an affine map with *frozen* statistics, so a
    ResNet is a composition of linear maps, fixed offsets and ReLU -- exactly the
    structure that makes a feature map asymptotically positively homogeneous.
    The ratio ||phi(Ax)|| / (A ||phi(x)||) settles at a positive constant, and
    Corollary 1's 1/A law applies.
  * **LayerNorm** rescales each token by its own norm, so a ViT (like BERT)
    discards the query scale entirely.  The ratio decays like 1/A and the lever
    does not exist.

So the same family of models splits on a single architectural choice, and a
defender can read off which of its endpoints are exposed.

Panel B is the reachability check, and it is where the honest limit lives.  The
argument above scales the *normalized tensor*, which a deployment accepting raw
images does not permit: pixels are bounded, so an attacker can only amplify
contrast about the mid-grey point and then clip.  This panel measures the
effective amplification actually achievable in the pixel domain, which is the
number a defender should care about.

Panel C runs the absolute-versus-relative comparison on the ResNet endpoint head
under both query domains, so the reported gap is separated into what the
mathematics permits and what a pixel API permits.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from lowrank_extraction.experiments.run_module_amplification import (
    apply_quantizer,
    homogeneity_ratio,
    scaling_regime,
    utility_costs,
)

Row = dict[str, Any]

# Above this exact-arithmetic relative error the solve has already failed with no
# quantizer in play, so any "defense" measured at that amplification is really a
# broken attack. Recovery on a healthy design lands near 1e-12, and the failures
# are order 1, so the threshold sits far from both.
EXACT_CONTROL_TOLERANCE = 1e-3

# ImageNet preprocessing constants, applied after the pixel-domain clamp so that
# the "what a pixel API permits" panel is measured on genuine images.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=Path("results") / "vision_endpoint")
    parser.add_argument("--data-root", type=Path, default=Path.home() / ".cache" / "cifar10")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--corpus-size", type=int, default=700)
    parser.add_argument("--regime-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument(
        "--amplifications", type=float, nargs="+",
        default=[1.0, 4.0, 16.0, 64.0, 256.0, 1024.0],
    )
    parser.add_argument("--decimals", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--sig-digits", type=int, nargs="+", default=[3, 4, 5])
    return parser.parse_args()


def _require_deps() -> dict[str, Any]:
    try:
        import torch
        import torchvision
        from torchvision import transforms
        from torchvision.models import (
            ResNet18_Weights, ViT_B_16_Weights, resnet18, vit_b_16,
        )
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "The vision endpoint experiment needs torchvision. Install it with: "
            "uv sync --extra vision"
        ) from exc
    return {
        "torch": torch, "torchvision": torchvision, "transforms": transforms,
        "resnet18": resnet18, "ResNet18_Weights": ResNet18_Weights,
        "vit_b_16": vit_b_16, "ViT_B_16_Weights": ViT_B_16_Weights,
    }


def load_images(deps: dict[str, Any], *, root: Path, count: int, size: int):
    """Real images in the *pixel* domain, in [0, 1], before normalization.

    Keeping the pixel domain explicit is what lets Panel B distinguish scaling a
    normalized tensor (unbounded, mathematically clean) from scaling an actual
    image (bounded, which is what a deployment accepts).
    """

    torch, transforms = deps["torch"], deps["transforms"]
    root.mkdir(parents=True, exist_ok=True)
    dataset = deps["torchvision"].datasets.CIFAR10(
        root=str(root), train=False, download=True,
        transform=transforms.Compose([transforms.Resize(size), transforms.ToTensor()]),
    )
    images = torch.stack([dataset[i][0] for i in range(min(count, len(dataset)))])
    return images


def normalize(deps: dict[str, Any], pixels):
    torch = deps["torch"]
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (pixels - mean) / std


def resnet_features(deps: dict[str, Any], model, tensor, *, batch_size: int) -> np.ndarray:
    torch = deps["torch"]
    trunk = torch.nn.Sequential(*list(model.children())[:-1])
    out = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            chunk = tensor[start : start + batch_size]
            out.append(trunk(chunk).flatten(1).numpy().astype(np.float64))
    return np.vstack(out)


def vit_features(deps: dict[str, Any], model, tensor, *, batch_size: int) -> np.ndarray:
    torch = deps["torch"]
    out = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            chunk = tensor[start : start + batch_size]
            z = model._process_input(chunk)
            cls = model.class_token.expand(chunk.shape[0], -1, -1)
            z = model.encoder(torch.cat([cls, z], dim=1))
            out.append(z[:, 0].numpy().astype(np.float64))
    return np.vstack(out)


def amplify_pixels(deps: dict[str, Any], pixels, amp: float):
    """Contrast amplification about mid-grey, clipped to the valid pixel range.

    This is the strongest scaling a deployment accepting ordinary images allows.
    It saturates: past a certain point every pixel is already at 0 or 1 and
    further amplification changes nothing, which is precisely the reachability
    constraint of Section 5 rather than a property of any defense.
    """

    torch = deps["torch"]
    return torch.clamp(0.5 + (pixels - 0.5) * float(amp), 0.0, 1.0)


# --------------------------------------------------------------------------


def endpoint_head(z: np.ndarray, *, rank: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 21_000)
    hidden = z.shape[1]
    n_classes = 3
    scale = 1.0 / max(float(np.mean(np.linalg.norm(z, axis=1))), 1e-12)
    left = rng.standard_normal((n_classes, rank)) * scale
    right = rng.standard_normal((hidden, rank))
    right /= np.linalg.norm(right, axis=0, keepdims=True)
    return {
        "base_head": rng.standard_normal((n_classes, hidden)) * scale,
        "delta": left @ right.T,
        "bias": rng.standard_normal(n_classes) * 0.1,
    }


def endpoint_attack(z_design: np.ndarray, head: dict[str, np.ndarray], *,
                    decimals: int | None, sig_digits: int | None, rank: int) -> float:
    aug = np.hstack([z_design, np.ones((z_design.shape[0], 1))])
    exact = z_design @ (head["base_head"] + head["delta"]).T + head["bias"]
    served = apply_quantizer(exact, decimals=decimals, sig_digits=sig_digits)
    theta, *_ = np.linalg.lstsq(aug, served, rcond=None)
    recovered = theta[:-1].T - head["base_head"]
    u, s, vt = np.linalg.svd(recovered, full_matrices=False)
    recovered = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    return float(np.linalg.norm(recovered - head["delta"]) / np.linalg.norm(head["delta"]))


def design_diagnostics(z: np.ndarray) -> dict[str, float]:
    """Is the amplified design still solvable at all?

    Amplification pushes public ReLU features toward their dead zone. When a
    coordinate stops activating for *every* query the augmented design drops
    below full rank and the class becomes unsolvable -- so a small measured gain
    means the attack broke, not that the defense worked. This is the same
    diagnostic the paper uses on ``wine``, recorded here at h = 512.

    The rank has to be measured on the *augmented* matrix, the one actually
    solved. Comparing ``matrix_rank(z)`` against ``z.shape[1] + 1`` compares the
    rank of the ``h``-column feature block against the ``h + 1`` columns of the
    augmented system, which no full-rank design can ever reach: it reported a
    phantom one-column deficiency on every row, including the well-conditioned
    ``A = 1`` case where the augmented design is exactly full rank.
    """

    aug = np.hstack([z, np.ones((z.shape[0], 1))])
    return {
        "feature_rank": float(np.linalg.matrix_rank(aug)),
        "required_rank": float(aug.shape[1]),
        "dead_coordinates": float(np.sum(np.abs(z).max(axis=0) == 0.0)),
    }


def run_attack_panel(features_by_amp: dict[str, dict[float, np.ndarray]], *, seeds: list[int],
                     rank: int, test_size: int, decimals: list[int],
                     sig_digits: list[int]) -> list[Row]:
    rows: list[Row] = []
    for domain, by_amp in features_by_amp.items():
        base = by_amp[1.0]
        hidden = base.shape[1]
        if base.shape[0] < hidden + 1:
            raise RuntimeError(
                f"{domain}: need {hidden + 1} queries to span the endpoint feature space, "
                f"have {base.shape[0]}"
            )
        for seed in seeds:
            head = endpoint_head(base, rank=rank, seed=seed)
            held_out = base[: min(test_size, base.shape[0])]
            exact_test = held_out @ (head["base_head"] + head["delta"]).T + head["bias"]
            settings = [("absolute", {"decimals": d, "sig_digits": None}) for d in decimals]
            settings += [("relative", {"decimals": None, "sig_digits": p}) for p in sig_digits]
            for family, kwargs in settings:
                costs = utility_costs(exact_test, **kwargs)
                setting = kwargs["decimals"] if family == "absolute" else kwargs["sig_digits"]
                for amp, z in by_amp.items():
                    rows.append({
                        "panel": "attack",
                        "domain": domain,
                        "seed": seed,
                        "family": family,
                        "setting": int(setting),
                        "hidden_dim": hidden,
                        "amplification": amp,
                        "relerr": endpoint_attack(z, head, rank=rank, **kwargs),
                        # Exact arithmetic, no quantizer: the control that says
                        # whether a comparison at this amplification means
                        # anything. If this is large the attack has already
                        # failed and no defense is being measured.
                        "exact_relerr": endpoint_attack(
                            z, head, decimals=None, sig_digits=None, rank=rank
                        ),
                        **design_diagnostics(z),
                        **costs,
                    })
    return rows


def _mean(rows: list[Row], key: str) -> float:
    values = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
    return float(np.mean(values)) if values else float("nan")


def summarize(regimes: list[Row], attack: list[Row], *, amplifications: list[float]) -> str:
    lines = [
        "# The Amplification Lever on a Vision Prediction API",
        "",
        "## Panel A: the normalization layer decides exposure",
        "",
        "`ratio` is ||phi(Ax)|| / (A ||phi(x)||), measured on real CIFAR-10 images in",
        "the normalized tensor domain. A positive constant means Corollary 1's 1/A",
        "law applies; decay like 1/A means the map discarded the scale.",
        "",
        "| model | normalization | " + " | ".join(f"A={a:g}" for a in amplifications) + " | regime |",
        "|---|---|" + "---:|" * len(amplifications) + "---|",
    ]
    for row in [r for r in regimes if r.get("domain") == "tensor"]:
        ratios = {float(k): v for k, v in row["ratios"].items()}
        cells = " | ".join(f"{ratios[a]:.4f}" for a in amplifications)
        lines.append(f"| {row['model']} | {row['normalization']} | {cells} | {row['regime']} |")
    lines += [
        "",
        "Same task, same input domain, opposite exposure. BatchNorm in eval mode is",
        "an affine map with frozen statistics, so the network stays a composition of",
        "linear maps, fixed offsets and ReLU and the lever applies. LayerNorm",
        "rescales by the query's own norm and removes it. A defender can read off",
        "which of its endpoints are exposed from the architecture alone.",
        "",
        "## Panel B: what a pixel-domain API actually permits",
        "",
        "The rows above scale the normalized tensor. A deployment accepting ordinary",
        "images does not allow that: pixels are bounded, so the attacker can only",
        "amplify contrast about mid-grey and clip. This is the reachability limit of",
        "Section 5, and it is the honest cap on the lever.",
        "",
    ]
    pixel = [r for r in regimes if r.get("domain") == "pixel"]
    if pixel:
        lines += [
            "| model | " + " | ".join(f"A={a:g}" for a in amplifications) + " |",
            "|---|" + "---:|" * len(amplifications),
        ]
        for row in pixel:
            ratios = {float(k): v for k, v in row["ratios"].items()}
            lines.append(
                f"| {row['model']} | " + " | ".join(f"{ratios[a]:.4f}" for a in amplifications) + " |"
            )
        lines.append("")
    if attack:
        for domain in sorted({r["domain"] for r in attack}):
            sub_all = [r for r in attack if r["domain"] == domain]
            hidden = int(sub_all[0]["hidden_dim"])
            lines += [
                f"## Panel C ({domain} domain): absolute versus relative at h = {hidden}",
                "",
                "| family | setting | label | logit rel. MAE | attacker (best over A) | amp. gain |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            # An amplification whose exact-arithmetic solve has already failed is
            # measuring a broken attack, not a defense, and averaging its utility
            # cost into the frontier is what made these numbers swing by 3x
            # between environments. The control is recorded per row precisely so
            # that it can be applied here; drop those amplifications and say so
            # rather than truncating silently.
            healthy = {a for a in amplifications
                       if _mean([r for r in sub_all if r["amplification"] == a],
                                "exact_relerr") <= EXACT_CONTROL_TOLERANCE}
            dropped = [a for a in amplifications if a not in healthy]
            if not healthy:
                lines += ["", "No amplification passes the exact-arithmetic control in this "
                          "domain; nothing here measures a defense.", ""]
                continue
            frontier: list[tuple[str, float, float]] = []
            for family in ["absolute", "relative"]:
                for setting in sorted({int(r["setting"]) for r in sub_all if r["family"] == family}):
                    sub = [r for r in sub_all
                           if r["family"] == family and int(r["setting"]) == setting
                           and r["amplification"] in healthy]
                    at = lambda a: _mean([r for r in sub if r["amplification"] == a], "relerr")
                    best = min(at(a) for a in sorted(healthy))
                    cost = _mean(sub, "logit_relative_mae")
                    frontier.append((family, cost, best))
                    lines.append(
                        f"| {family} | {setting} | {_mean(sub, 'label_agreement'):.4f} |"
                        f" {cost:.3e} | {best:.3e} | {at(min(healthy)) / best:.2f}x |"
                    )
            if dropped:
                lines += [
                    "",
                    "Excluded as failed attacks rather than successful defenses (exact-arithmetic "
                    f"relative error above {EXACT_CONTROL_TOLERANCE:g} with no quantizer present): "
                    "A = " + ", ".join(f"{a:g}" for a in dropped) + ".",
                ]
            # Integer digit settings cannot be cost-matched exactly, and a
            # nearest-neighbour pairing silently hands one family a utility
            # advantage. Within each family leakage is very nearly proportional
            # to the utility cost, so the scale-free comparison is the ratio of
            # the two slopes: leakage per unit of service degradation.
            lines += ["", "Leakage per unit of utility cost (leak / cost, averaged over settings).",
                      "Both families are near-proportional, so this is the cost-matched",
                      "comparison that integer digit settings cannot express directly.", "",
                      "| family | leak / cost per setting | mean slope |", "|---|---|---:|"]
            slopes: dict[str, float] = {}
            for family in ["absolute", "relative"]:
                vals = [leak / cost for f, cost, leak in frontier if f == family and cost > 0]
                slopes[family] = float(np.mean(vals)) if vals else float("nan")
                lines.append(
                    f"| {family} | {', '.join(f'{v:.1f}' for v in vals)} | {slopes[family]:.1f} |"
                )
            advantage = slopes["relative"] / slopes["absolute"] if slopes["absolute"] else float("nan")
            lines += [
                "",
                f"At equal service cost the attacker's error is {advantage:.2f}x larger under",
                "relative quantization.",
                "",
            ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    deps = _require_deps()
    amps = args.amplifications

    pixels = load_images(deps, root=args.data_root, count=args.corpus_size, size=args.image_size)
    small = pixels[: args.regime_size]

    resnet = deps["resnet18"](weights=deps["ResNet18_Weights"].DEFAULT).eval()
    vit = deps["vit_b_16"](weights=deps["ViT_B_16_Weights"].DEFAULT).eval()

    regimes: list[Row] = []
    # Panel A: scale the normalized tensor, which is the mathematically clean lever.
    for name, model, extract, norm in [
        ("resnet18", resnet, resnet_features, "BatchNorm (eval)"),
        ("vit_b_16", vit, vit_features, "LayerNorm"),
    ]:
        by_amp = {
            a: extract(deps, model, normalize(deps, small) * a, batch_size=args.batch_size)
            for a in amps
        }
        ratios = homogeneity_ratio(by_amp)
        regimes.append({"model": name, "normalization": norm, "domain": "tensor",
                        "ratios": ratios, "regime": scaling_regime(ratios)})

    # Panel B: scale actual images, which is what a pixel API permits.
    for name, model, extract, norm in [
        ("resnet18", resnet, resnet_features, "BatchNorm (eval)"),
        ("vit_b_16", vit, vit_features, "LayerNorm"),
    ]:
        by_amp = {
            a: extract(deps, model, normalize(deps, amplify_pixels(deps, small, a)),
                       batch_size=args.batch_size)
            for a in amps
        }
        ratios = homogeneity_ratio(by_amp)
        regimes.append({"model": name, "normalization": norm, "domain": "pixel",
                        "ratios": ratios, "regime": scaling_regime(ratios)})

    # Panel C: the attack, in both domains, on the ResNet endpoint.
    features_by_amp = {
        "tensor": {
            a: resnet_features(deps, resnet, normalize(deps, pixels) * a, batch_size=args.batch_size)
            for a in amps
        },
        "pixel": {
            a: resnet_features(deps, resnet, normalize(deps, amplify_pixels(deps, pixels, a)),
                               batch_size=args.batch_size)
            for a in amps
        },
    }
    attack = run_attack_panel(
        features_by_amp, seeds=args.seeds, rank=args.rank, test_size=args.test_size,
        decimals=args.decimals, sig_digits=args.sig_digits,
    )

    (args.outdir / "regimes.json").write_text(json.dumps(regimes, indent=2, default=str))
    (args.outdir / "attack.json").write_text(json.dumps(attack, indent=2))
    if attack:
        with (args.outdir / "attack.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(attack[0].keys()))
            writer.writeheader()
            writer.writerows(attack)
    summary = summarize(regimes, attack, amplifications=amps)
    (args.outdir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
