"""What precision does a real inference server actually put on the wire?

Theorem~\\ref{thm:precisionfloor} is stated over two numbers: the precision `p`
the oracle serves and the precision `d` the attacker's request carries.  Both
are properties of a deployment, not of our analysis, and everywhere else in this
paper `p` is something *we* chose when we built a quantizer.  That leaves the
theorem's practical content resting on an assumption about real servers.

This script removes the assumption.  It talks to an OpenAI-compatible
completions endpoint -- vLLM, TGI, llama.cpp's server, or anything else that
speaks the same protocol -- and reads the *raw response bytes* rather than a
parsed float, because the parsed float cannot tell us what the server chose to
serialize.  ``-2.3456789169311523`` and ``-2.35`` both round-trip to a Python
float; only the bytes distinguish a server that serves seventeen significant
digits from one that serves three.

What it reports, per served logprob field:

  * significant digits actually present in the serialized decimal
  * whether the value round-trips to the nearest float32 (i.e. the server
    serialized enough digits to preserve its own arithmetic) or was truncated
  * the implied `p` for Theorem 7, taken as the *minimum* significant-digit
    count observed, since an attacker is bounded by the worst case it can rely
    on

Run against a server you host yourself.  Pointing it at a third-party
production API would be probing someone else's service, which
Section~\\ref{app:ethics} rules out and which this script does not do for you:
it takes a base URL and sends ordinary completion requests, nothing adaptive.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# A float32 mantissa is 24 bits, or 24*log10(2) = 7.22 decimal digits, of which
# 7 are always recoverable.  This is the precision an attacker actually gets
# from a server that serializes its arithmetic without rounding it first.
FLOAT32_DECIMAL_DIGITS = 7

# Decimal digits carried by each candidate arithmetic width, as
# floor(mantissa_bits * log10(2)).  bfloat16 keeps 8 explicit mantissa bits,
# float16 keeps 11, float32 keeps 24 (all counting the implicit leading one).
_WIDTH_DECIMAL_DIGITS = {"bfloat16": 2, "float16": 3, "float32": FLOAT32_DECIMAL_DIGITS}

# A serialized JSON number: optional sign, digits, optional fraction, optional
# exponent.  We keep the raw text so the significant-digit count is a fact about
# the response and not about our parsing.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

# The same number, but only where it stands as a JSON *value* -- after a colon
# in an object, or as an element of an array.  Keys are excluded on purpose.
_VALUE = re.compile(r"[:,\[]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?=[,}\]]|$)")


def significant_digits(literal: str) -> int:
    """Significant digits in a serialized decimal, as written.

    Leading zeros are not significant; trailing zeros after a decimal point are,
    because a server that writes them chose to.  ``0.00250`` is three.
    """
    mantissa = literal.split("e")[0].split("E")[0].lstrip("+-")
    if "." in mantissa:
        whole, frac = mantissa.split(".")
        digits = (whole + frac).lstrip("0")
        # An all-zero mantissa carries no significant digit at all.
        return len(digits) if digits else 0
    return len(mantissa.strip("0")) or (1 if mantissa else 0)


def round_trips_float32(literal: str) -> bool:
    """Does the serialized text preserve the server's own float32 value?

    If it does not, the server has already applied a coarser grid than its
    arithmetic -- which is exactly the defense of Section 5, applied by
    accident.
    """
    import numpy as np

    value = float(literal)
    as32 = np.float32(value)
    # Re-serialize the float32 at full precision and compare to what we got.
    return float(repr(float(as32))) == value


def narrowest_width(literal: str) -> str:
    """The narrowest float width that represents this value exactly.

    ``round_trips_float32`` alone cannot separate a server computing in float32
    from one computing in bfloat16 and widening before it serializes: both
    round-trip.  A bfloat16 value, though, is a float32 whose low sixteen
    mantissa bits are zero, and a float16 value is one that survives a cast.
    Reporting the narrowest width that fits keeps us from crediting an attacker
    with precision the server's arithmetic never held.

    On a single value this is weak evidence -- ``-0.5`` fits every width.  It
    becomes a measurement only in aggregate, over values that had no reason to
    land on a coarse grid.
    """
    import numpy as np

    value = float(literal)
    as32 = np.float32(value)
    if float(as32) != value:
        return "wider_than_float32"
    if int(as32.view(np.uint32)) & 0xFFFF == 0:
        return "bfloat16"
    if float(np.float16(as32)) == float(as32):
        return "float16"
    return "float32"


def _post(url: str, payload: dict[str, Any], timeout: float) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        # Raw bytes: the whole point is to see what was serialized.
        return response.read().decode("utf-8")


def harvest_logprobs(raw: str) -> list[str]:
    """Every logprob literal in the response, as written on the wire.

    We walk the raw text rather than the parsed object so the literals survive.
    """
    literals: list[str] = []
    for key in ("logprob", "\"logprobs\""):
        for match in re.finditer(re.escape(key) + r"\"?\s*:\s*", raw):
            tail = raw[match.end() : match.end() + 64]
            number = _NUMBER.match(tail)
            if number:
                literals.append(number.group(0))
    # token_logprobs arrays in the legacy completions schema.
    for match in re.finditer(r"\"token_logprobs\"\s*:\s*\[([^\]]*)\]", raw):
        literals.extend(_NUMBER.findall(match.group(1)))
    for match in re.finditer(r"\"top_logprobs\"\s*:\s*(\{[^}]*\}|\[[^\]]*\])", raw):
        # Only values, never keys: a top-token can itself be a numeral (ask a
        # server to continue "2 + 2 =" and the candidate tokens are "2", "3",
        # "4"), and scraping those as logprobs would report a served precision
        # of one digit that no server ever served.
        literals.extend(_VALUE.findall(match.group(1)))
    return [lit for lit in literals if lit not in {"0", "-0"}]


def probe(
    base_url: str, model: str, prompts: list[str], top_logprobs: int, timeout: float
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for prompt in prompts:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 4,
            "temperature": 0.0,
            "logprobs": top_logprobs,
            "echo": False,
        }
        try:
            raw = _post(f"{base_url.rstrip('/')}/v1/completions", payload, timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{prompt!r}: {exc}")
            continue
        literals = harvest_logprobs(raw)
        for literal in literals:
            rows.append(
                {
                    "prompt": prompt,
                    "literal": literal,
                    "significant_digits": significant_digits(literal),
                    "round_trips_float32": round_trips_float32(literal),
                    "narrowest_width": narrowest_width(literal),
                }
            )

    if not rows:
        return {"rows": [], "errors": errors, "summary": None}

    digits = [r["significant_digits"] for r in rows]
    summary = {
        "n_values": len(rows),
        "min_significant_digits": min(digits),
        "max_significant_digits": max(digits),
        "median_significant_digits": statistics.median(digits),
        # The worst case over the *printed* literals.  This is not yet Theorem
        # 7's p: printing a float32 at 14 digits does not carry 14 digits of
        # information.  See implied_p_for_theorem7 below.
        "implied_p_from_printed_digits": min(digits),
        "fraction_round_tripping_float32": sum(
            1 for r in rows if r["round_trips_float32"]
        )
        / len(rows),
    }
    # If every literal recovers the server's own float32, the server applied no
    # decimal rounding at all, and the digit count is an artifact of widening a
    # float32 to float64 before repr.  What bounds the attacker there is the
    # arithmetic, whose 24-bit mantissa carries about 7 decimal digits -- still
    # far more than any deliberate output quantizer would leave.
    summary["applied_no_output_rounding"] = (
        summary["fraction_round_tripping_float32"] == 1.0
    )
    # The widest width any single value needed is the width the arithmetic must
    # have had: one value that is not a bfloat16 rules bfloat16 out for all.
    order = ["bfloat16", "float16", "float32", "wider_than_float32"]
    widths = [r["narrowest_width"] for r in rows]
    implied_width = max(widths, key=order.index)
    summary["width_histogram"] = {w: widths.count(w) for w in order if w in widths}
    summary["implied_arithmetic_width"] = implied_width
    summary["effective_decimal_digits"] = (
        _WIDTH_DECIMAL_DIGITS.get(implied_width, FLOAT32_DECIMAL_DIGITS)
        if summary["applied_no_output_rounding"]
        else summary["implied_p_from_printed_digits"]
    )
    # Theorem 7's p is what the attacker can actually rely on.  Where the server
    # applied no output rounding that is its arithmetic width, not the printed
    # count -- which runs wide on a float32 widened before repr, and short on a
    # value like -0.125 that repr prints exactly.  Where it did round, the
    # printed count is the measurement.
    summary["implied_p_for_theorem7"] = summary["effective_decimal_digits"]
    return {"rows": rows, "errors": errors, "summary": summary}


def render(payload: dict[str, Any], model: str, base_url: str) -> str:
    s = payload["summary"]
    if s is None:
        return "# Served precision\n\nNo values harvested. Errors:\n\n" + "\n".join(
            f"- {e}" for e in payload["errors"]
        )
    lines = [
        "# Served Wire Precision of a Self-Hosted Inference Server",
        "",
        f"Server: `{base_url}`  Model: `{model}`",
        "",
        "Every figure below is read from the raw HTTP response bytes, not from a",
        "parsed float, because a parsed float cannot report how many digits the",
        "server chose to serialize.",
        "",
        f"- logprob values observed: **{s['n_values']}**",
        f"- significant digits on the wire: **{s['min_significant_digits']}** to "
        f"**{s['max_significant_digits']}** (median {s['median_significant_digits']})",
        f"- fraction preserving the server's own float32 value: "
        f"**{s['fraction_round_tripping_float32']:.3f}**",
        "",
        f"Printed significant digits, worst case: "
        f"**{s['implied_p_from_printed_digits']}**.",
        "",
        f"Implied served precision for Theorem 7: **p = {s['implied_p_for_theorem7']}** "
        "-- the arithmetic's width where the server applied no output rounding, "
        "and the printed digit count where it did.  A printed count taken on an "
        "unrounded server measures the serializer, not a grid.",
        "",
    ]
    if s["applied_no_output_rounding"]:
        lines += [
            "Every literal recovers the server's own float32 exactly, so this "
            "deployment applied **no output rounding**: the digit counts above are "
            "an artifact of widening a float32 to float64 before printing it, not "
            "a grid the server chose. What bounds the attacker here is the "
            f"arithmetic: the narrowest width every served value fits is "
            f"**{s['implied_arithmetic_width']}**, worth "
            f"**{s['effective_decimal_digits']} decimal digits** -- which is the "
            "quantity to substitute for `p`.",
            "",
        ]
    lines += [
        "| significant digits | count |",
        "|---:|---:|",
    ]
    counts: dict[int, int] = {}
    for row in payload["rows"]:
        counts[row["significant_digits"]] = counts.get(row["significant_digits"], 0) + 1
    for digits in sorted(counts):
        lines.append(f"| {digits} | {counts[digits]} |")
    if payload["errors"]:
        lines += ["", "Errors:", ""] + [f"- {e}" for e in payload["errors"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--top-logprobs", type=int, default=5)
    ap.add_argument(
        "--server-note",
        default="",
        help=(
            "Free text recorded verbatim in the output, e.g. the stack and dtype "
            "the operator launched. The probe cannot see these from outside, so "
            "they are the operator's claim and are labelled as such."
        ),
    )
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument(
        "--outdir", type=Path, default=Path("results") / "served_precision"
    )
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "The capital of France is",
            "2 + 2 =",
            "Once upon a time",
            "def add(a, b):",
        ],
    )
    args = ap.parse_args()

    payload = probe(
        args.base_url, args.model, args.prompts, args.top_logprobs, args.timeout
    )
    payload["config"] = {
        "base_url": args.base_url,
        "model": args.model,
        "top_logprobs": args.top_logprobs,
        "prompts": args.prompts,
        "server_note_from_operator": args.server_note,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "served_precision.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    summary = render(payload, args.model, args.base_url)
    (args.outdir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
