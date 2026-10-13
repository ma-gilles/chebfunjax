"""Native complex abs preserves existing domain and point values.

Provenance
----------
MATLAB source: @chebfun/abs.m, @chebfun/addBreaksAtRoots.m.
Chebfun commit: 7574c77.
"""

import jax.numpy as jnp
import pytest

import chebfunjax as cj


@pytest.mark.parametrize("tech", ["chebtech1", "chebtech2"])
@pytest.mark.parametrize("transpose", [False, True])
def test_complex_abs_preserves_existing_breaks(tech, transpose):
    breaks = (-2.0, -1.0, 0.0, 1.0, 2.0)
    f = cj.chebfun(
        lambda x: jnp.sin(jnp.pi * x)
        + 1j * jnp.cos(jnp.pi * (x - 0.5)),
        domain=breaks, tech=tech)
    if transpose:
        f = f.T
    g = f.abs()
    assert g.domain.breakpoints == breaks
    assert g.is_transposed == f.is_transposed
    assert all(type(p.tech) is type(q.tech)
               and p.interval == q.interval for p, q in zip(f.funs, g.funs))
    assert bool(jnp.array_equal(g.point_values, jnp.abs(f.point_values)))
    x = jnp.asarray([-1.83, -0.39, 0.42, 1.67])
    assert float(jnp.max(jnp.abs(g(x) - jnp.abs(f(x))))) < 10 * float(
        jnp.finfo(jnp.float64).eps) * f.vscale
