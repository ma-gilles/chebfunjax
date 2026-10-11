"""real/imag/isreal source contracts, Chebfun commit: 7574c77.

Sources: @separableApprox/real.m, imag.m, isreal.m; @chebfun2/compose.m.
Analytic evaluation bounds are supplemental; native regressions stay unchanged.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize("part", ["real", "imag"])
def test_empty_part_returns_input(part):
    f = Chebfun2.empty()
    assert getattr(f, part)() is f
    assert f.isreal()


@pytest.mark.parametrize("trigx,trigy", [(True, False), (False, True), (True, True)])
def test_independent_periodic_axes_are_inherited(trigx, trigy):
    def xpart(x):
        return jnp.cos(jnp.pi * x) if trigx else x

    def ypart(y):
        return jnp.sin(jnp.pi * y) if trigy else y

    f = Chebfun2.from_function(lambda x, y: xpart(x) + 1j * ypart(y), trigx=trigx, trigy=trigy)
    t = jnp.linspace(-0.8, 0.8, 13)
    xx, yy = jnp.meshgrid(t, t)
    for g, expected in [(f.real(), xpart(xx)), (f.imag(), ypart(yy))]:
        assert all(isinstance(c, Trigtech) == trigy for c in g.approx.cols)
        assert all(isinstance(r, Trigtech) == trigx for r in g.approx.rows)
        assert jnp.max(jnp.abs(g(xx, yy) - expected)) < 1000 * jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize("value,expected", [(1.0, True), (1 + 1j, False)])
def test_source_storage_real_predicate(value, expected):
    f = Chebfun2.from_function(lambda x, y: jnp.asarray(value))
    assert f.isreal() is expected
