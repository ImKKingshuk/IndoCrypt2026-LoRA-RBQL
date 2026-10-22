"""Re-derive the request-utility frontier at Theorem 7(iii)'s exact floor.

The utility measurements in ``run_request_utility`` are a property of the
deployment and the (p, d) cell; the *floor* attached to each cell is a property
of the theorem alone.  Replacing the floor from part (ii)'s proven constant with
part (iii)'s exact infimum therefore changes which cells clear a target and what
the best deployable guarantee is, but not a single accuracy number -- so the
expensive panels do not have to be recomputed to get the exact-floor frontier.

This reads the panels from an existing run, reattaches ``theorem7_floor`` from
``exact_floor``, and rebuilds the frontier and ``best_deployable_floor`` with the
exact budget.  It writes a fresh results directory rather than editing the
original, so the two remain separately auditable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lowrank_extraction.experiments.run_precision_floor import exact_floor
from lowrank_extraction.experiments.run_request_utility import (
    EXACT,
    frontier,
    write_summary,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", type=Path, default=Path("results") / "request_utility")
    ap.add_argument(
        "--outdir", type=Path, default=Path("results") / "request_utility_exactfloor"
    )
    args = ap.parse_args()

    started = time.time()
    source = json.loads((args.indir / "request_utility.json").read_text(encoding="utf-8"))
    panels = source["panels"]
    targets = source["targets"]

    changed = 0
    for panel in panels:
        for row in panel["rows"]:
            p, d = row["served_digits"], row["request_digits"]
            # EXACT survives the JSON round trip as null, so an unquantized cell
            # is recognised by the absent digit count rather than by the sentinel.
            was = row["theorem7_floor"]
            row["theorem7_floor"] = (
                None
                if (p is None or d is None or p == EXACT or d == EXACT)
                else exact_floor(int(p), int(d))
            )
            if was != row["theorem7_floor"]:
                changed += 1

    front = frontier(panels, targets)
    best_deployable = {
        "p": 1,
        "d": 1,
        "floor": exact_floor(1, 1),
        "note": "the largest Theorem 7(iii) floor over deployable (p, d), both >= 1",
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "request_utility.json").write_text(
        json.dumps(
            {
                "panels": panels,
                "frontier": front,
                "best_deployable_floor": best_deployable,
                "seed": source["seed"],
                "targets": targets,
                "rederived_from": str(args.indir),
                "cells_refloored": changed,
                "runtime_seconds": round(time.time() - started, 1),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(write_summary(args.outdir / "summary.md", panels, front, targets))
    print(f"refloored {changed} cells from {args.indir}")


if __name__ == "__main__":
    main()
