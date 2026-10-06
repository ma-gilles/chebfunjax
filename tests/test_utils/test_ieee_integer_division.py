"""Independent binary64 quotient tests; no FFT/weight acceptance substituted.

Expected results use native CPU NumPy division and exact rational division
in tests only. Provenance
----------
MATLAB source : @chebtech1/quadwts.m (inverse-DFT scaling consumer)
Chebfun commit: 7574c77
Independent IEEE binary64 reference checks; configure CPU and JAX x64.
"""
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._binary64 import (
    _divide_binary64_by_positive_integer,
)

DIVISORS = [1, 2, 3, 7, 9, 10, 11, 51, 2**20+1, 2**52-1, 2**53-1, 2**53]


def bitwise_equal(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64),
                                   np.asarray(expected).view(np.uint64))


@pytest.mark.parametrize("divisor", DIVISORS)
def test_native_cpu_boundary_and_random_quotients(divisor):
    rng = np.random.default_rng(6178)
    random_bits = rng.integers(0, 2**64, size=4096, dtype=np.uint64)
    # Every exponent, with low and high significand: subnormal/normal/finite
    # overflow boundary and both signs included. Exclude nonfinite here.
    exponents = np.arange(2047, dtype=np.uint64) << np.uint64(52)
    boundary = np.concatenate((exponents, exponents | np.uint64((1 << 52)-1),
                               np.arange(65, dtype=np.uint64)))
    bits = np.concatenate((random_bits, boundary, boundary | np.uint64(1 << 63)))
    finite = ((bits >> np.uint64(52)) & np.uint64(0x7ff)) != 0x7ff
    values = bits[finite].view(np.float64)
    with np.errstate(all="ignore"):
        expected = np.divide(values, np.float64(divisor))
    eager = _divide_binary64_by_positive_integer(jnp.asarray(values), divisor)
    compiled = jax.jit(lambda x: _divide_binary64_by_positive_integer(x, divisor))
    bitwise_equal(eager, expected)
    bitwise_equal(compiled(jnp.asarray(values)), expected)


@pytest.mark.parametrize("divisor", [2, 3, 10, 51, 2**53])
def test_exact_fraction_rounding_reference(divisor):
    rng = np.random.default_rng(13)
    bits = rng.integers(0, 2**64, size=128, dtype=np.uint64)
    bits &= np.uint64(0xffefffffffffffff)  # Keep all inputs finite.
    # Explicit min-normal seam and half-minimum-subnormal ties.
    bits = np.concatenate((bits, np.array([0, 1, 2, 3, 5, 7,
        (1 << 52)-1, 1 << 52, (1 << 52)+1], dtype=np.uint64)))
    values = bits.view(np.float64)
    exact = np.asarray([float(Fraction.from_float(float(x))/divisor)
                        for x in values], dtype=np.float64)
    # Fraction has no signed zero; preserve the mathematical input sign.
    exact = np.copysign(exact, values)
    bitwise_equal(_divide_binary64_by_positive_integer(values, divisor), exact)


def test_signed_zero_infinity_and_nan_payload_contract():
    bits = np.array([0, 1 << 63, 0x7ff0000000000000, 0xfff0000000000000,
                     0x7ff0000000000001, 0xfff0000000000007,
                     0x7ff8123456789abc, 0xfff8123456789abc], dtype=np.uint64)
    expected = bits.copy()
    expected[4:] |= np.uint64(1 << 51)
    for fn in (lambda x: _divide_binary64_by_positive_integer(x, 10),
               jax.jit(lambda x: _divide_binary64_by_positive_integer(x, 10))):
        np.testing.assert_array_equal(np.asarray(fn(bits.view(np.float64))).view(np.uint64),
                                       expected)


@pytest.mark.parametrize("divisor", [0, -1, 2**53+1])
def test_out_of_contract_divisor_rejected(divisor):
    with pytest.raises(ValueError):
        _divide_binary64_by_positive_integer(1.0, divisor)


def test_shapes_preserved():
    for shape in [(), (0,), (2, 3)]:
        values = jnp.ones(shape, dtype=jnp.float64)
        assert _divide_binary64_by_positive_integer(values, 3).shape == shape
