"""All24 evaluation predicates from MATLAB Chebfun7574c77.

Provenance
----------
MATLAB source: tests/chebfun2/test_feval.m.
Retains source vector/matrix2-norms, sizes, grids and bounds. NumPy RNG0
provides deterministic probe points, not MATLAB random-stream identity.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref

TOL = 100 * ChebfunPref().cheb2Prefs.chebfun2eps


@pytest.fixture(scope="module")
def errors():
    result = {}

    def norm(x):
        a = jnp.asarray(x)
        return float(jnp.abs(a) if a.ndim == 0 else jnp.linalg.norm(a, ord=2))

    f = chebfun2(lambda x, y: x)
    result[1] = norm(f(jnp.pi / 6, jnp.pi / 12) - jnp.pi / 6)
    f = chebfun2(lambda x, y: x, domain=(-1, 2, -jnp.pi / 2, jnp.pi))
    result[2] = norm(f(0.0, 0.0))
    result[3] = norm(f(jnp.pi / 6, jnp.pi / 12) - jnp.pi / 6)
    f = chebfun2(lambda x, y: y, domain=(-1, 2, -jnp.pi / 2, jnp.pi))
    result[4] = norm(f(0.0, 0.0))
    result[5] = norm(f(jnp.pi / 6, jnp.pi / 12) - jnp.pi / 12)
    op = lambda x, y: jnp.cos(x) + jnp.sin(x * y)
    g = chebfun2(op)
    r, s = 0.126986816293506, 0.632359246225410
    result[6] = norm(op(r, s) - g(r, s))
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.random(10))
    s = jnp.asarray(rng.random(10))
    rr, ss = jnp.meshgrid(r, s)
    result[7] = norm(op(r, s) - g(r, s))
    result[8] = norm(op(rr, ss) - g(rr, ss))
    g = chebfun2(op, domain=(-jnp.pi / 6, jnp.pi / 2, -jnp.pi / 12, jnp.sqrt(3)))
    result[9] = norm(
        op(0.126986816293506, 0.632359246225410) - g(0.126986816293506, 0.632359246225410)
    )
    r = jnp.asarray(rng.random(10))
    s = jnp.asarray(rng.random(10))
    rr, ss = jnp.meshgrid(r, s)
    result[10] = norm(op(r, s) - g(r, s))
    result[11] = norm(op(rr, ss) - g(rr, ss))
    f = chebfun2(lambda x, y: x + 1j * y)
    result[12] = norm(f(-1.0, -1.0) - (-1 - 1j))
    result[13] = norm(f(r, s) - (r + 1j * s))
    result[14] = norm(f(rr, ss) - (rr + 1j * ss))
    f = chebfun2(lambda z: z)
    result[15] = norm(f(-1 - 1j) - (-1 - 1j))
    result[16] = norm(f(r + 1j * s) - (r + 1j * s))
    result[17] = norm(f(rr, ss) - (rr + 1j * ss))
    op = lambda x, y: jnp.cos(jnp.pi * (x + y)) + y * jnp.sin(jnp.pi * x)
    grid = jnp.linspace(-1, 1, 1001)
    yy, xx = jnp.meshgrid(grid, grid)
    f = chebfun2(op)
    result[18] = norm(f(xx, yy) - op(xx, yy))
    return result


@pytest.mark.parametrize("clause", range(1, 19))
def test_source_numeric_clause(errors, clause):
    bound = (
        1e-14
        if clause in (2, 4)
        else (2 * TOL if clause == 11 else 100 * TOL if clause == 18 else TOL)
    )
    assert errors[clause] < bound, (clause, errors[clause], bound)


@pytest.mark.parametrize(
    "clause,shape",
    [(19, ()), (20, (1, 2)), (21, (2, 1)), (22, (2, 2)), (23, (1, 10)), (24, (10, 10))],
)
def test_source_output_shape(clause, shape):
    f = chebfun2(lambda x, y: jnp.cos(x * y))
    x = jnp.ones(shape)
    assert f(x, x).shape == shape  # Python scalar shape() corresponds to MATLAB1x1.
