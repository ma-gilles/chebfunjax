"""Actual MATLAB linspace bit controls within the locator's tiny range.

Provenance
----------
MATLAB source : @fun/detectEdge.m; installed MATLAB R2025b linspace.m
Chebfun commit: 7574c77
Literal source inputs only enter the library; captured outputs assert results.
"""
import json
import struct
from pathlib import Path

import jax
import pytest

FIXTURE = json.loads(Path(__file__).with_name(
    "tiny_source_grid_matlab_expanded_fixture.json").read_text())
BOUND = float.fromhex("0x1p-1020")


def decode(word):
    return struct.unpack(">d", bytes.fromhex(word))[0]


CASES = [c for c in FIXTURE["inputs"]
         if -BOUND <= decode(c["a_hex"]) < decode(c["b_hex"]) <= BOUND]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_fresh_matlab_bounded_grid_bits(case):
    from chebfunjax.utils._binary64_grid import tiny_source_grid
    expected = next(c for c in FIXTURE["expected_cases"] if c["id"] == case["id"])
    assert (expected["a_hex"], expected["b_hex"], expected["n"]) == (
        case["a_hex"], case["b_hex"], case["n"])
    # Input coordinates come from the pre-run literal operand fixture.
    a, b, n = decode(case["a_hex"]), decode(case["b_hex"]), case["n"]
    for fn in (tiny_source_grid, jax.jit(tiny_source_grid, static_argnums=2)):
        actual = [struct.pack(">d", v).hex() for v in fn(a, b, n).tolist()]
        assert actual == expected["x_hex"]
