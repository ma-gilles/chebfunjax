"""Source controls for the distinct Chebfun3 zero/rank/length predicates.

Provenance
----------
MATLAB source : @chebfun3/{iszero,rank,length}.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.tech.chebtech import Chebtech2

DOMAIN = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


def _factor(coeffs):
    return Chebtech2.from_coeffs(jnp.asarray(coeffs, dtype=jnp.float64))


@pytest.mark.parametrize("zero_mode", ["cols", "rows", "tubes"])
def test_public_iszero_checks_each_zero_factor_mode(zero_mode):
    """Native iszero sees a nonzero core, zero 10^3 sample, then zero factors."""
    zero = _factor([0.0, 0.0, 0.0])
    one = _factor([1.0])
    factors = {"cols": [one], "rows": [one], "tubes": [one]}
    factors[zero_mode] = [zero]
    f = Chebfun3(factors["cols"], factors["rows"], factors["tubes"],
                 jnp.ones((1, 1, 1)), DOMAIN)

    points = jnp.linspace(-1.0, 1.0, 10)
    grid = jnp.meshgrid(points, points, points, indexing="ij")
    assert jnp.array_equal(f(*grid), jnp.zeros((10, 10, 10)))
    assert f.iszero()
    # Native rank is based on core zero only; length uses full iszero.
    assert f.rank == (1, 1, 1)
    assert f.length() == (1, 1, 1)


def test_core_zero_keeps_rank_and_length_source_semantics():
    one = _factor([1.0])
    f = Chebfun3([one], [one], [one], jnp.zeros((1, 1, 1)), DOMAIN)
    assert f.iszero()
    assert f.rank == (0, 0, 0)
    assert f.length() == (1, 1, 1)


def test_nonzero_core_and_factors_are_not_iszero():
    one = _factor([1.0])
    f = Chebfun3([one], [one], [one], jnp.ones((1, 1, 1)), DOMAIN)
    assert not f.iszero()
    assert f.rank == (1, 1, 1)


def test_zero_sample_alone_does_not_replace_native_factor_check():
    one = _factor([1.0])
    f = Chebfun3([one, one], [one], [one],
                 jnp.asarray([1.0, -1.0]).reshape((2, 1, 1)), DOMAIN)
    assert float(f(.2, .3, .4)) == 0.0
    assert not f.iszero()
    assert f.rank == (2, 1, 1)
