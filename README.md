# Recovery Boundaries and Quantization Limits for Low-Rank Neural Adaptations

Code, experiment results and tests for the paper

> Kingshuk Mondal. *Recovery Boundaries and Quantization Limits for Low-Rank
> Neural Adaptations under Restricted Oracles.* Indocrypt 2026, Lecture Notes
> in Computer Science, Springer (to appear).

The paper studies which part of a known-base low-rank adapter (LoRA) a
prediction API reveals. Score-bearing oracles recover observable heads exactly,
while hard labels, index-only outputs and rank-deficient endpoints leave only a
quotient. It also shows how far score quantization works as a defense against
an adaptive attacker.

## Contents

- `src/lowrank_extraction/`: attacks, oracles, defenses and the experiment
  runners (`experiments/run_*.py`).
- `results/`: the stored output of the experiments the paper reports, mostly
  as JSON.
- `tests/`: unit tests for the attacks, oracles and defenses, and numerical
  checks of the paper's formal statements.
- `paper/main.pdf`: the full version of the paper.
- `ARTIFACT.md`: requirements, the command behind each result, where each
  result appears in the paper, and run times.

## Quick start

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest
```

The transformer and vision experiments need the optional extras:

```bash
uv sync --extra dev --extra lora --extra vision
```

Every experiment is a console script, for example `uv run run-synthetic` or
`uv run run-trained-adapter-defense`. `ARTIFACT.md` lists each one with the
arguments used in the paper. All runs are CPU-only and seed-deterministic.

## Citation

If you use this code, please cite the paper. `CITATION.cff` has the details.

## License

The code, tests and experiment results are released under the MIT License
(see `LICENSE`). The paper in `paper/` is not covered by that license: it is
the author's own full version, © 2026 Kingshuk Mondal, all rights reserved.
The final publication will be available from Springer.
