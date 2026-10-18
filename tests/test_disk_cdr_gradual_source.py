"""Independent reciprocal rounding oracles for local Diskfun CDR weights.

Host Fraction/struct are test-only exact binary64 oracles. Production must
compute the source reciprocal from its operand; there is no fixture lookup.
"""

import json
import math
import random
import struct
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.diskfun.diskfun import _disk_cdr_weights


def _word(x):
    return struct.unpack('>Q', struct.pack('>d', x))[0]


def _float(bits):
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


def _oracle(bits):
    x = _float(bits)
    if math.isnan(x):
        return bits
    if x == 0:
        return 0  # source infinite-reciprocal assignment
    if math.isinf(x):
        return bits & (1 << 63)
    try:
        value = float(1 / Fraction(x))
    except OverflowError:
        return 0
    return _word(value)


def _words():
    values = {0, 1, (1 << 52) - 1, 1 << 52, 0x7FEFFFFFFFFFFFFF}
    # Every positive power of two, plus representable neighbors.
    for exponent in range(-1074, 1024):
        w = _word(math.ldexp(1.0, exponent))
        values.update(q for q in [w - 1, w, w + 1] if 0 <= q < 0x7FF0000000000000)
    # Reciprocal overflow boundary and mantissa transitions, independent of
    # the captured native operands.
    for power in range(1, 53):
        for offset in range(-8, 9):
            q = (1 << power) + offset
            if 0 < q < (1 << 52):
                values.add(q)
    rng = random.Random(6178)
    for _ in range(4096):
        values.add(rng.randrange(0x7FF0000000000000))
    positive = sorted(values)
    return positive + [q | (1 << 63) for q in positive]


def _array(words):
    return jax.lax.bitcast_convert_type(jnp.asarray(words, dtype=jnp.uint64), jnp.float64)


@pytest.mark.parametrize('mode', ['eager', 'jit', 'vmap', 'jit_vmap'])
def test_exact_fraction_reciprocal_words(mode):
    words = _words()
    x = _array(words)
    op = _disk_cdr_weights
    if 'vmap' in mode:
        op = jax.vmap(op)
    if 'jit' in mode:
        op = jax.jit(op)
    actual = jax.lax.bitcast_convert_type(op(x), jnp.uint64)
    expected = jnp.asarray([_oracle(w) for w in words], dtype=jnp.uint64)
    mismatch = actual != expected
    assert not jnp.any(mismatch), (mode, int(jnp.sum(mismatch)))


@pytest.mark.parametrize('compiled', [False, True])
def test_actual_native16_words(compiled):
    p = Path(__file__).parent / 'fixtures/disk_cdr_ieee_native_20261018.json'
    rows = json.loads(p.read_text())['rows']
    x = _array([int(r['input'], 16) for r in rows])
    op = jax.jit(_disk_cdr_weights) if compiled else _disk_cdr_weights
    actual = jax.lax.bitcast_convert_type(op(x), jnp.uint64)
    expected = jnp.asarray([int(r['cdr'], 16) for r in rows], dtype=jnp.uint64)
    assert jnp.array_equal(actual, expected)


@pytest.mark.parametrize('compiled', [False, True])
def test_finite_signed_first_second_derivatives(compiled):
    x = jnp.asarray([-8.0, -2.0, -.75, .625, 1.25, 3.0, 16.0])
    one = jnp.ones_like(x)

    def first(t):
        return jax.jvp(_disk_cdr_weights, (t,), (one,))[1]

    def both(t):
        return first(t), jax.jvp(first, (t,), (one,))[1]

    op = jax.jit(both) if compiled else both
    d1, d2 = op(x)
    eps = jnp.finfo(jnp.float64).eps
    assert jnp.max(jnp.abs((d1 + 1 / x**2) / (1 / x**2))) < 8 * eps
    assert jnp.max(jnp.abs((d2 - 2 / x**3) / (2 / x**3))) < 8 * eps


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex128])
def test_other_dtype_retains_source_division_fallback(dtype):
    from chebfunjax.utils._ordered_binary import binary

    x = jnp.asarray([0., -0., 1., -2., jnp.inf, -jnp.inf, jnp.nan], dtype=dtype)
    reciprocal = binary(jnp.ones_like(x), x, 3)
    expected = jnp.where(jnp.isinf(jnp.abs(reciprocal)), jnp.zeros_like(reciprocal), reciprocal)
    actual = _disk_cdr_weights(x)
    if dtype == jnp.float32:
        actual_words = jax.lax.bitcast_convert_type(actual, jnp.uint32)
        expected_words = jax.lax.bitcast_convert_type(expected, jnp.uint32)
    else:
        actual_words = jax.lax.bitcast_convert_type(jnp.stack([actual.real, actual.imag]), jnp.uint64)
        expected_words = jax.lax.bitcast_convert_type(jnp.stack([expected.real, expected.imag]), jnp.uint64)
    assert jnp.array_equal(actual_words, expected_words)
