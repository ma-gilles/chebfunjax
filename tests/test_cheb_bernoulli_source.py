"""Callable Bernoulli columns from Chebfun7574c77 +cheb/bernoulli.m.

The native example evaluates B(0,:). Native rank-one singleton axes are
adapted to the existing Python Chebfun convention; polynomial columns remain
callable together and through column selection. Numerical comparisons retain
the existing Bernoulli port's supplemental1e-13 bound.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax import cheb
from chebfunjax.chebfun1d.chebfun import Chebfun


@pytest.mark.parametrize("n", [0, 1, 4, 8])
def test_native_callable_columns(n):
    functions = cheb.bernoulli(n)
    assert isinstance(functions, Chebfun)
    assert functions.n_columns == n + 1
    assert tuple(functions.domain.breakpoints) == (0.0, 1.0)
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/bernoulli_callable_native.json").read_text()
    )
    case = next(row for row in fixture["cases"] if row["n"] == n)
    points = jnp.asarray(fixture["x"])
    actual = jnp.asarray(functions(points)).reshape((len(points), n + 1))
    expected = jnp.asarray(case["values"]).reshape(actual.shape)
    assert float(jnp.max(jnp.abs(actual - expected))) < 1e-13
    # MATLAB's documented B(0,:) call and Python's column adapter agree.
    endpoint = jnp.asarray(functions(0.0)).reshape((-1,))
    assert endpoint.shape == (n + 1,)
    assert float(jnp.max(jnp.abs(endpoint - expected[0]))) < 1e-13
    assert float(jnp.max(jnp.abs(functions[n](points) - expected[:, n]))) < 1e-13


def test_documented_bernoulli_numbers():
    functions = cheb.bernoulli(4)
    expected = jnp.array([1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0])
    assert float(jnp.max(jnp.abs(functions(0.0) - expected))) < 1e-13
