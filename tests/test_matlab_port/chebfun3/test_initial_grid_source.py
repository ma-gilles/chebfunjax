"""Focused default-grid and constructor source controls.

Provenance
----------
MATLAB source: @chebfun3/chebfun3f.m; @chebtech/techPref.m;
              tests/chebfun3/test_constructor.m, clauses 8--11.
Chebfun commit: 7574c77

The callback shape is a supplemental algorithm control, not an original
MATLAB test predicate. Numerical controls retain the original bounds.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.chebpref import ChebfunPref


class _SampledGrid(Exception):
    pass


@pytest.mark.parametrize("factory,kwargs,n", [
    (chebfun3, {}, 17),
    (Chebfun3.from_function, {}, 17),
    (chebfun3, {"min_samples": 9}, 9),
    (chebfun3, {"min_samples": 25}, 25),
])
def test_first_tensor_grid(factory, kwargs, n):
    shapes = []

    def record(x, y, z):
        if getattr(x, "ndim", 0) == 3:
            shapes.append(x.shape)
            raise _SampledGrid
        return x + y + z

    with pytest.raises(_SampledGrid):
        factory(record, **kwargs)
    # Native starts n=[grid,grid,grid], r=[6,6,6], then samples mode 1.
    assert shapes == [(n, 6, 6)]


def test_constructor_clause_8():
    tol = 1e2 * ChebfunPref().cheb3Prefs.chebfun3eps
    def g(x, y, z):
        return jnp.cos(x) * y * z + x * z * jnp.sin(y)
    f = chebfun3(g)
    assert abs(float(f(.1, jnp.pi / 6, -.3)) - float(g(.1, jnp.pi / 6, -.3))) < tol


def test_constructor_clauses_9_to_11():
    f = chebfun3(lambda x, y, z: x * y * z + y ** 2 * z)
    assert f.rank == (2, 2, 1)
