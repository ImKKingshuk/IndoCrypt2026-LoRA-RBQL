# Artifact Guide

This guide explains how to check and rebuild the results in

> Kingshuk Mondal. *Recovery Boundaries and Quantization Limits for Low-Rank
> Neural Adaptations under Restricted Oracles.* Indocrypt 2026.

Table and section numbers refer to the full version in `paper/main.pdf`.

## Requirements

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).
- CPU only. No GPU is used, even if one is present.
- About 4.3 GB of disk for the public model weights and datasets that the
  transformer and vision experiments download on first use.

`uv.lock` pins the exact library versions used for the paper (listed in §S24).

## Running the tests

```bash
uv sync --extra dev
uv run pytest
```

This reports **93 passed** in under a minute and needs no downloads.

- `tests/test_theorem_verification.py` checks the formal statements numerically
  on constructed instances, including the cases each statement excludes.
- The other test files cover the attacks, oracles, defenses and experiment
  summaries.

The stored outputs of every reported experiment are under `results/`, so the
figures in the paper can be compared with them without rerunning anything.

## Running the experiments

Each experiment is a console script. By default it writes into `results/` and
overwrites the stored output. To compare a rerun with the stored one, pass a
different `--outdir`.

The paper's machine was an Apple M1 (8 CPU cores, 8 GB, macOS 26.6.1), with the
model cache already downloaded. On it, the 32 reproduce commands take about
6 hours in total. The slowest, `run-trained-adapter-defense`, takes 3 h 13 m,
almost all of it training the victim adapter; the other 31 together take under
three hours. Table 31 lists every command's time, and
`results/compute/timings.json` holds the measurements.

### Core experiments (NumPy and scikit-learn only)

```bash
uv run run-synthetic --features 32 --classes 8 --rank 3 --seed 7
uv run run-ablation
uv run run-realdata
uv run run-defense-sweep
uv run run-adaptive-defense --seeds 0 1 2 3 4 5 6 7 8 9
uv run run-quantization-frontier --seeds 0 1 2 3 4 5 6 7 8 9
uv run run-defense-taxonomy --seeds 0 1 2 3 4 5 6 7 8 9
uv run run-relative-attacker
uv run run-security-floor
uv run run-lower-bound
uv run run-floor-margin --outdir results/floor_margin_asymptotic
uv run run-request-model --outdir results/request_model_exactfloor
```

To include the endpoint and PEFT rows of Table 30, run `run-defense-sweep`
after the transformer commands below.

Boundary seeking and the request-precision floor (§5.2, §S11–S13):

```bash
uv run run-adaptive-boundary
uv run run-adaptive-boundary --outdir results/adaptive_boundary_unknownbias
uv run run-adaptive-boundary --input-digits 4 --known-bias-control --outdir results/adaptive_boundary_unknownbias_steps
uv run run-adaptive-boundary --input-digits 4 6 8 12 --known-bias-control --outdir results/adaptive_boundary_knownbias_digits
uv run run-adaptive-boundary --seeds 0 1 2 3 4 5 6 7 8 9 --input-digits 4 6 8 12 16 none --amplifications 1 --scale-caps 1e2 --noise-sigma 1e-9 1e-6 1e-4 1e-3 --noise-repeats 1 64 --outdir results/adaptive_boundary_noise_v3
uv run run-noise-aggregators --out results/noise_aggregators --seeds 0 1 2 3 4 5 6 7 8 9 --noise-sigma 1e-6 1e-3 --input-digits 12 --repeats 64
uv run run-precision-floor --outdir results/precision_floor_rerun
uv run run-precision-floor --negative-secrets --outdir results/precision_floor_negative_rerun
uv run run-precision-floor --budget 80 --outdir results/precision_floor_budget80_rerun
uv run run-precision-floor --closed-form-scan --outdir results/precision_floor_closedform_rerun
```

The second command runs the joint (boundary, scale) search of Table 7. It
takes about 1 h 25 m, because it enumerates every scale the attacker's request
format can express. The fourth takes about 2 h 10 m and the noisy-server sweep
about 56 minutes. The rest finish in seconds to minutes.

### Experiments that need the optional extras

```bash
uv sync --extra dev --extra lora --extra vision
```

Transformer endpoints (§8.2, §S15):

```bash
uv run run-endpoint-transformer --model bert-base-uncased --outdir results/endpoint_transformer_bert_base --seeds 0 1 2 3 4 5 6 7 8 9 --train-size 900 --test-size 300 --query-size 900 --epochs 60 --batch-size 8 --max-length 128
uv run run-endpoint-transformer --model bert-base-uncased --outdir results/endpoint_transformer_defense_bert_d8_rerun --seeds 0 1 2 3 4 5 6 7 8 9 --train-size 900 --test-size 300 --query-size 900 --epochs 60 --batch-size 8 --max-length 128 --rounded-decimals 8 --sig-digits 8
uv run run-endpoint-transformer --model bert-base-uncased --outdir results/endpoint_transformer_defense_bert_d6_rerun --seeds 0 1 2 3 4 5 6 7 8 9 --train-size 900 --test-size 300 --query-size 900 --epochs 60 --batch-size 8 --max-length 128 --rounded-decimals 6 --sig-digits 6
uv run run-lm-endpoint-inversion --model distilgpt2 --query-prompts 28 --eval-prompts 12 --adapter-feature-dim 16 --rank 4 --max-rank 8 --rank-tol 1e-2 --seeds 0 1 2 3 4 5 6 7 8 9 --row-budgets 128 256 384 512 640 767 768 769 1024 --outdir results/lm_endpoint_inversion_distilgpt2_10seed_ranktol_1e2
uv run run-mlm-output-head --model bert-base-uncased --query-size 900 --test-size 128 --batch-size 8 --rank 4 --max-rank 8 --rank-tol 1e-3 --seeds 0 1 2 3 4 5 6 7 8 9 --outdir results/mlm_output_head_bert_base_10seed_unknown_rank_tol1e3
```

PEFT LoRA checkpoints and the curvature of the endpoint map (§8.3, §S3, §S6).
`run-null-curvature` reads the adapters saved by the checkpoint command, so run
that one first:

```bash
uv run run-lora-checkpoint --model bert-base-uncased --text-dataset sst2 --outdir results/lora_checkpoint_sst2_bert_base_stronger_10seed --seeds 0 1 2 3 4 5 6 7 8 9 --train-size 1800 --test-size 500 --epochs 2 --batch-size 8 --max-length 128 --rank 4 --jacobian-examples 16 --jacobian-sweep-examples 1 2 4 8 16
uv run run-null-curvature --seeds 0 1 2 3 4 5 6 7 8 9 --checkpoint-dir results/lora_checkpoint_sst2_bert_base_stronger_10seed --outdir results/null_curvature_rerun
```

Amplification regimes and the trained-adapter defense (§8.5, §S19–S20):

```bash
uv run run-module-amplification --seeds 0 1 2 3 4 5 6 7 8 9 --outdir results/module_amplification_10seed
uv run run-vision-endpoint --corpus-size 2000 --amplifications 1 4 16 64 256 1024 4096 --seeds 0 1 2 3 4 5 6 7 8 9 --outdir results/vision_endpoint_10seed
uv run run-trained-adapter-defense --train-size 1800 --test-size 500 --epochs 2 --max-length 128 --seeds 0 1 2 3 4 5 6 7 8 9 --outdir results/trained_adapter_defense_10seed
```

The last command trains ten adapters. Table 31 times the default three seeds,
so the ten-seed run takes about ten times as long.

Request formats and utility (§S14). The second step reuses the measured
accuracies and recomputes each cell's floor at the exact value given by
Theorem 7(iii):

```bash
uv run run-request-utility
uv run python -m lowrank_extraction.experiments.rederive_request_utility --indir results/request_utility --outdir results/request_utility_exactfloor_rerun
```

### Served precision of a real inference server (§S13)

This probe reads the output precision of a self-hosted vLLM server. Run it only
against a server you operate. vLLM is not a dependency of this package, so
install it in a separate environment:

```bash
python -m venv /tmp/vllmenv && /tmp/vllmenv/bin/pip install 'vllm==0.11.0' 'transformers<5'
```

Start the server in one shell:

```bash
VLLM_TARGET_DEVICE=cpu VLLM_HOST_IP=127.0.0.1 /tmp/vllmenv/bin/python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m --dtype float32 --max-model-len 512 --host 127.0.0.1 --port 8000 --enforce-eager
```

and probe it from another:

```bash
uv run run-served-precision --base-url http://127.0.0.1:8000 --model facebook/opt-125m --server-note "vLLM 0.11.0 OpenAI api_server, CPU backend, --dtype float32 --enforce-eager, macOS arm64" --outdir results/served_precision
```

For the second configuration, repeat both steps with `--dtype float16` and
`--outdir results/served_precision_float16`. vLLM's CPU backend does not
support bfloat16.

## Where each result is used

| Result | Command | Paper |
| --- | --- | --- |
| `synthetic_f32_c8_r3_s7.json`, `ablation/`, `lower_bound/` | `run-synthetic`, `run-ablation`, `run-lower-bound` | Table 26 |
| `realdata/` | `run-realdata` | Table 17, §S16 |
| `adaptive_defense/` | `run-adaptive-defense` | Tables 18, 29, Figure 1 |
| `quantization_frontier/` | `run-quantization-frontier` | Tables 6, 24 |
| `defense_taxonomy/` | `run-defense-taxonomy` | Tables 22, 25, §S21 |
| `relative_attacker/` | `run-relative-attacker` | Table 23 |
| `security_floor/` | `run-security-floor` | Table 21 |
| `defense_sweep/` | `run-defense-sweep` | Table 30 |
| `floor_margin/`, `floor_margin_asymptotic/` | `run-floor-margin` | §S9 |
| `adaptive_boundary/` | `run-adaptive-boundary` | Table 2 |
| `adaptive_boundary_unknownbias/`, `adaptive_boundary_unknownbias_steps/`, `adaptive_boundary_knownbias_digits/` | `run-adaptive-boundary` | Table 7, §S11 |
| `adaptive_boundary_noise_v3/` | `run-adaptive-boundary` | Table 8 |
| `noise_aggregators/` | `run-noise-aggregators` | Table 9 |
| `precision_floor_rerun/`, `precision_floor_negative/`, `precision_floor_negative_rerun/`, `precision_floor_budget80_rerun/` | `run-precision-floor` | Table 10, §S13 |
| `precision_floor_closedform_rerun/` | `run-precision-floor --closed-form-scan` | Table 11 |
| `request_model_exactfloor/` | `run-request-model` | §S13 |
| `request_utility/`, `request_utility_exactfloor_rerun/` | `run-request-utility`, then the re-derivation | Table 12, §S14 |
| `served_precision/`, `served_precision_float16/` | `run-served-precision` | §S13 |
| `endpoint_transformer_bert_base/` | `run-endpoint-transformer` | Table 19 |
| `endpoint_transformer_defense_bert_d6*/`, `endpoint_transformer_defense_bert_d8*/` | `run-endpoint-transformer` | Table 14 |
| `lm_endpoint_inversion_distilgpt2*/` | `run-lm-endpoint-inversion` | Table 27 |
| `mlm_output_head_bert_base*/` | `run-mlm-output-head` | Table 28 |
| `lora_checkpoint_sst2_bert_base_*/` | `run-lora-checkpoint` | Table 3, Figure 2, §S3 |
| `null_curvature/`, `null_curvature_rerun/` | `run-null-curvature` | Tables 4, 5, §S6 |
| `module_amplification*/`, `vision_endpoint*/` | `run-module-amplification`, `run-vision-endpoint` | Table 20, §S19–S20 |
| `trained_adapter_defense_10seed/`, `trained_adapter_defense_heavy/` | `run-trained-adapter-defense` | Table 15, §8.5 |
| `compute/timings.json` | measured wall-clock | Table 31 |

Some folders hold a smaller or earlier run of the same command, which the
paper also reports:

- `*_d6/`, `*_d8/`, `trained_adapter_defense_heavy/`, `module_amplification/`
  and `vision_endpoint/` are the three-seed runs. The commands above produce
  the ten-seed runs.
- `lm_endpoint_inversion_distilgpt2/` and `mlm_output_head_bert_base/` are
  three-seed runs with a known rank.
- The `_unknown_rank`, `_ranktol_3e3` and `_tol1e3` folders repeat the ten-seed
  run with `--max-rank 8` and a different `--rank-tol`.
- `lm_endpoint_inversion_distilgpt2_10seed_spectrum_tol_sweep/` holds the
  recovered singular spectra of the ten-seed run, and the rank estimates
  recomputed from them at thresholds from 1e-4 to 1e-1.
- `lora_checkpoint_sst2_bert_base_stronger/` is the three-seed run.
  `lora_checkpoint_sst2_bert_base_jacobian_sweep/` trains on 900 examples and
  tests on 300.
- `null_curvature/` is the first run of the curvature sweep. The rerun adds the
  measured cell slack used in Table 5.
- `precision_floor_negative/` is the first run with negative secrets.

The paper also times a few commands whose output it does not report, such as
the wine-dataset runs and a small DistilBERT smoke test. Their outputs are not
included.
