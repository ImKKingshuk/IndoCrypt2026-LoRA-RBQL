"""The wire-precision probe, checked on values whose answer is known.

Theorem 7's served precision `p` is the one parameter in the paper that is a
property of somebody else's deployment rather than of our own quantizer, so the
tool that measures it has to be right about what a decimal literal says. These
tests pin the two things it can get wrong: counting significant digits in text,
and deciding whether a literal preserves the float32 the server computed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lowrank_extraction.experiments.run_served_precision import (
    FLOAT32_DECIMAL_DIGITS,
    harvest_logprobs,
    narrowest_width,
    probe,
    render,
    round_trips_float32,
    significant_digits,
)


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ("-2.3456788063049316", 17),  # a float32 widened and repr'd in full
        ("-2.35", 3),
        ("0.00250", 3),  # trailing zero after the point is a choice, so it counts
        ("-0.5", 1),
        ("1e-05", 1),  # the exponent is not a significant digit
        ("100", 1),  # trailing zeros before the point are not
        ("-12.0", 3),
        ("0.0", 0),  # no significant digit at all
    ],
)
def test_significant_digits_counts_what_the_server_wrote(
    literal: str, expected: int
) -> None:
    assert significant_digits(literal) == expected


def test_round_trip_separates_a_full_repr_from_a_rounded_one() -> None:
    """The distinction the measurement rests on.

    A server that computes in float32 and serializes with an ordinary JSON
    encoder emits the full float64 repr of the widened value, which recovers its
    float32 exactly. A server that rounds first does not -- and a server that
    rounds has applied the defense of Section 5 whether it meant to or not.
    """
    exact = repr(float(np.float32(-2.3456789)))
    assert round_trips_float32(exact)
    assert significant_digits(exact) >= 9, "a full float32 repr is not a short one"

    assert not round_trips_float32("-2.35")
    assert not round_trips_float32("-2.3")


def test_harvest_reads_both_completion_schemas() -> None:
    """Literals must survive as text; parsing them to float would lose the point."""
    raw = json.dumps(
        {
            "choices": [
                {
                    "logprobs": {
                        "token_logprobs": [-2.3456788063049316, -0.125],
                        "top_logprobs": [{"a": -1.5, "b": -2.25}],
                    }
                }
            ]
        }
    )
    got = harvest_logprobs(raw)
    assert "-2.3456788063049316" in got, "the full-precision literal was lost"
    assert {"-0.125", "-1.5", "-2.25"} <= set(got)


def test_harvest_ignores_numeric_token_names() -> None:
    """A top-candidate token can itself be a numeral.

    Ask a server to continue ``2 + 2 =`` and the candidates it ranks are the
    tokens ``2``, ``3``, ``4``.  Those numerals are token *names*; reading them
    as logprobs reports a served precision of one significant digit that the
    server never served.
    """
    raw = (
        '{"choices":[{"logprobs":{"top_logprobs":'
        '[{"2": -0.6931471824645996, "4": -1.3862943649291992}]}}]}'
    )
    got = harvest_logprobs(raw)
    assert got == ["-0.6931471824645996", "-1.3862943649291992"]
    assert min(significant_digits(lit) for lit in got) >= 16, (
        "a token name leaked in as a one-digit logprob"
    )


def test_harvest_keeps_the_literal_and_not_its_float() -> None:
    """A rounded literal must not be silently reinflated by json parsing."""
    raw = '{"choices":[{"logprobs":{"token_logprobs":[-2.35]}}]}'
    assert harvest_logprobs(raw) == ["-2.35"]


def test_probe_reports_the_worst_case_as_the_printed_digit_count(monkeypatch) -> None:
    """The printed count is the minimum; Theorem 7's `p` is not that count.

    A server that serves seventeen digits on one token and three on another has
    not printed seventeen everywhere, so the printed figure has to be the worst
    case.  But here both literals recover the server's own float32, so neither
    count is a grid: the three-digit one is `repr` printing an exactly
    representable value, not the server rounding.  What bounds the attacker is
    the arithmetic, and `p` is its seven digits rather than the printed three.
    """
    responses = [
        '{"choices":[{"logprobs":{"token_logprobs":[-2.3456788063049316]}}]}',
        '{"choices":[{"logprobs":{"token_logprobs":[-0.125]}}]}',
    ]

    seen: list[str] = []

    def post(url: str, payload: dict, timeout: float) -> str:
        out = responses[len(seen) % len(responses)]
        seen.append(url)
        return out

    monkeypatch.setattr(
        "lowrank_extraction.experiments.run_served_precision._post", post
    )
    out = probe("http://localhost:8000", "m", ["a", "b"], 5, 5.0)
    summary = out["summary"]
    assert summary["n_values"] == 2
    assert summary["max_significant_digits"] == 17
    assert summary["min_significant_digits"] == 3
    assert summary["implied_p_from_printed_digits"] == 3
    assert summary["applied_no_output_rounding"], "both literals recover the float32"
    assert summary["implied_p_for_theorem7"] == 7
    text = render(out, "m", "http://localhost:8000")
    assert "Printed significant digits, worst case: **3**" in text
    assert "p = 7" in text


def _probe_over(monkeypatch, responses: list[str]) -> dict:
    seen: list[str] = []

    def post(url: str, payload: dict, timeout: float) -> str:
        out = responses[len(seen) % len(responses)]
        seen.append(url)
        return out

    monkeypatch.setattr(
        "lowrank_extraction.experiments.run_served_precision._post", post
    )
    return probe("http://localhost:8000", "m", ["a"] * len(responses), 5, 5.0)


def test_full_repr_is_reported_as_no_rounding_and_falls_back_to_float32(
    monkeypatch,
) -> None:
    """Seventeen digits is not seventeen digits of precision.

    A server that serializes its float32 without rounding emits the full
    float64 repr of the widened value. The digit count is then an artifact of
    the widening, and the quantity to put in Theorem 7 is the mantissa's own
    seven decimal digits -- which is what a stock deployment actually leaks.
    """
    out = _probe_over(
        monkeypatch,
        ['{"choices":[{"logprobs":{"token_logprobs":[-2.3456788063049316]}}]}'],
    )
    summary = out["summary"]
    assert summary["applied_no_output_rounding"]
    assert summary["implied_arithmetic_width"] == "float32"
    assert summary["effective_decimal_digits"] == FLOAT32_DECIMAL_DIGITS == 7


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.float32(-2.3456789), "float32"),
        (np.float16(-2.3456789), "float16"),
        # A bfloat16 is a float32 with its low sixteen mantissa bits cleared.
        (np.uint32(np.float32(-2.3456789).view(np.uint32) & 0xFFFF0000).view(
            np.float32
        ), "bfloat16"),
    ],
)
def test_narrowest_width_recovers_the_grid_a_value_came_from(value, expected) -> None:
    assert narrowest_width(repr(float(value))) == expected


def test_a_bfloat16_server_is_not_credited_with_float32_precision(monkeypatch) -> None:
    """The check that keeps the headline honest.

    A bfloat16 widened to float64 still round-trips, so the rounding test alone
    calls this an unrounded server and hands the attacker the digit count. But
    the value sits on a coarse grid the *arithmetic* imposed, which the digit
    count overstates by four -- and only the bit pattern reveals that.
    """
    bf16 = np.uint32(np.float32(-2.3456789).view(np.uint32) & 0xFFFF0000).view(
        np.float32
    )
    raw = '{"choices":[{"logprobs":{"token_logprobs":[%s]}}]}' % repr(float(bf16))
    summary = _probe_over(monkeypatch, [raw])["summary"]
    assert summary["applied_no_output_rounding"], "round-tripping alone sees no grid"
    assert summary["implied_p_from_printed_digits"] == 6, "what repr printed"
    assert summary["implied_arithmetic_width"] == "bfloat16"
    assert summary["effective_decimal_digits"] == 2
    assert summary["implied_p_for_theorem7"] == 2, "the digit count would overstate"


def test_a_rounding_server_is_taken_at_its_word(monkeypatch) -> None:
    """When the server does round, the digit count is the measurement."""
    out = _probe_over(
        monkeypatch, ['{"choices":[{"logprobs":{"token_logprobs":[-2.35,-1.07]}}]}']
    )
    summary = out["summary"]
    assert not summary["applied_no_output_rounding"]
    assert summary["effective_decimal_digits"] == summary["implied_p_for_theorem7"] == 3


def test_probe_reports_errors_rather_than_inventing_a_measurement(monkeypatch) -> None:
    """A server that does not answer must not produce a number."""

    def post(url: str, payload: dict, timeout: float) -> str:
        raise OSError("connection refused")

    monkeypatch.setattr(
        "lowrank_extraction.experiments.run_served_precision._post", post
    )
    out = probe("http://localhost:9", "m", ["a"], 5, 1.0)
    assert out["summary"] is None
    assert out["rows"] == []
    assert out["errors"] and "connection refused" in out["errors"][0]
    assert "No values harvested" in render(out, "m", "http://localhost:9")
