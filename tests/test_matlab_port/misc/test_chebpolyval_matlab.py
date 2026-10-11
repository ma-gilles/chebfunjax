"""Literal predicates from tests/misc/test_chebpolyval.m.

Provenance
----------
MATLAB source: tests/misc/test_chebpolyval.m.
Chebfun commit: 7574c77.
RandomState 42 supplies Python test inputs; native RNG bit parity is not asserted.
"""

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun, chebpoly
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.utils.polynomials import chebpolyval


def test_native_numeric_matrix_inf_norm():
    rng = np.random.RandomState(42)
    c = rng.rand(10, 2)
    x = rng.rand(3, 3)
    tc = chebpoly(np.arange(10)) @ jnp.asarray(c[::-1])
    # Native FEVAL concatenates coefficient columns into a 2D matrix.
    expected = np.asarray(tc(jnp.asarray(x))).reshape((3, 6), order="F")
    error = np.linalg.norm(np.asarray(chebpolyval(c, x)) - expected, np.inf)
    assert error < 10 * 10 * ChebfunPref().chebfuneps


def test_native_chebfun_array_inf_norm():
    c = np.random.RandomState(42).rand(10, 2)
    tc = chebpoly(np.arange(10)) @ jnp.asarray(c[::-1])
    result = chebpolyval(c, chebfun("x"))
    assert (result - tc).norm(jnp.inf) < 10 * 10 * ChebfunPref().chebfuneps
