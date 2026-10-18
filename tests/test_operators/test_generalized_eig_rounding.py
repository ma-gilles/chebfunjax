"""Directed exact-rational controls for QZ exponent transport, CPU binary64.

Fraction constructs exact input ratios/products; Python's Fraction-to-float
conversion rounds the exact rational once. No NumPy arithmetic or eigensolver
supplies expected results. These controls do not qualify all IEEE operations,
NaN payloads, exception flags, differentiation or other execution backends.
"""

import math
import random
import struct
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators import _qz_ieee as binary

SIGN = 1 << 63


def from_word(word):
    return struct.unpack(">d", struct.pack(">Q", word))[0]


def word(value):
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def exact_result(a, b, operation):
    x, y = Fraction.from_float(from_word(a)), Fraction.from_float(from_word(b))
    value = x * y if operation == "multiply" else x / y
    try:
        rounded = float(value)
    except OverflowError:
        rounded = math.copysign(math.inf, -1 if value < 0 else 1)
    bits = word(rounded)
    # Exact rational zero has no sign; multiplication/division use sign XOR.
    if value == 0:
        bits = (a ^ b) & SIGN
    return bits


def directed_pairs(operation):
    pairs = []
    # Normal/subnormal boundary, tie-to-even and the directed double-rounding
    # counterexample tiny*(1+u) * .5*(1+4u), u=2^-52.
    for a in [
        1,
        2,
        3,
        4,
        0xFFFFFFFFFFFFF,
        0x10000000000000,
        0x10000000000001,
        0x10000000000002,
        0x1FFFFFFFFFFFFF,
    ]:
        for b in [
            0x3FDFFFFFFFFFFFFF,
            0x3FE0000000000000,
            0x3FE0000000000001,
            0x3FE0000000000004,
            0x3FF0000000000000,
            0x4000000000000000,
        ]:
            pairs.append((a, b))
    # Both operands subnormal, tiny denominators and overflow/underflow edges.
    for a in [0, 1, 3, 0xFFFFFFFFFFFFF, 0x10000000000000, 0x3FF0000000000000, 0x7FEFFFFFFFFFFFFF]:
        for b in [1, 3, 0xFFFFFFFFFFFFF, 0x10000000000000, 0x3FF0000000000000, 0x7FEFFFFFFFFFFFFF]:
            pairs.append((a, b))
    rng = random.Random(48177)
    fields = [0, 1, 2, 500, 1000, 1022, 1023, 1024, 1500, 2046]
    for _ in range(256):
        a = (rng.choice(fields) << 52) | rng.getrandbits(52)
        b = (rng.choice(fields) << 52) | rng.getrandbits(52)
        pairs.append((a, b or 1))
    if operation == "divide":
        # Exponents separated by1022/1023 target subnormal quotient midpoints.
        for i in range(32):
            pairs.append((0x10000000000000 + i, 0x4000000000000000 + i))
    return [
        (a | sa, b | sb)
        for a, b in pairs
        for sa, sb in [(0, 0), (SIGN, 0), (0, SIGN), (SIGN, SIGN)]
    ]


@pytest.mark.parametrize("operation", ["multiply", "divide"])
def test_directed_exact_rational_rounding(operation):
    pairs = directed_pairs(operation)
    a = np.array([p[0] for p in pairs], np.uint64).view(np.float64)
    b = np.array([p[1] for p in pairs], np.uint64).view(np.float64)
    expected = np.array([exact_result(x, y, operation) for x, y in pairs], np.uint64)
    function = getattr(binary, operation)
    for disable_jit in [True, False]:
        with jax.disable_jit(disable_jit):
            actual = np.asarray(function(jnp.asarray(a), jnp.asarray(b))).view(np.uint64)
        np.testing.assert_array_equal(actual, expected)
    if operation == "multiply":
        index = pairs.index((0x0010000000000001, 0x3FE0000000000004))
        assert expected[index] == 0x0008000000000003


def test_signed_zero_addition_and_special_operand_policy():
    pairs = [
        (0, 0),
        (SIGN, SIGN),
        (0, SIGN),
        (SIGN, 0),
        (0x3FF0000000000000, 0xBFF0000000000000),
        (0x0010000000000001, 0x8010000000000000),
    ]
    a = np.array([x for x, _ in pairs], np.uint64).view(np.float64)
    b = np.array([y for _, y in pairs], np.uint64).view(np.float64)
    expected = np.array([0, SIGN, 0, 0, 0, 1], np.uint64)
    np.testing.assert_array_equal(
        np.asarray(binary.add(jnp.asarray(a), jnp.asarray(b))).view(np.uint64), expected
    )
    # Only classification/sign are promised for nonfinite results, not payloads.
    multiply = np.asarray(
        binary.multiply(jnp.array([np.inf, -np.inf, np.nan]), jnp.array([0.0, 2.0, 1.0]))
    )
    assert np.isnan(multiply[0]) and np.isneginf(multiply[1]) and np.isnan(multiply[2])
    divide = np.asarray(
        binary.divide(
            jnp.array([1.0, -1.0, 0.0, np.inf, 1.0]), jnp.array([0.0, 0.0, 0.0, np.inf, -np.inf])
        )
    )
    assert np.isposinf(divide[0]) and np.isneginf(divide[1])
    assert np.isnan(divide[2]) and np.isnan(divide[3])
    assert word(divide[4]) == SIGN


def test_nonfinite_and_subnormal_operand_classes():
    tiny = np.array([1, SIGN | 1], np.uint64).view(np.float64)
    infinity = jnp.array([np.inf, np.inf])
    expected_infinity = np.array([0x7FF0000000000000, 0xFFF0000000000000], np.uint64)
    expected_zero = np.array([0, SIGN], np.uint64)
    for a, b in [(jnp.asarray(tiny), infinity), (infinity, jnp.asarray(tiny))]:
        np.testing.assert_array_equal(
            np.asarray(binary.multiply(a, b)).view(np.uint64), expected_infinity
        )
    np.testing.assert_array_equal(
        np.asarray(binary.divide(infinity, jnp.asarray(tiny))).view(np.uint64), expected_infinity
    )
    np.testing.assert_array_equal(
        np.asarray(binary.divide(jnp.asarray(tiny), infinity)).view(np.uint64), expected_zero
    )
