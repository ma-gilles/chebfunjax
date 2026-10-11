"""All ten native tests/chebfun2v/test_compose.m predicates, pin7574c77."""

import gc

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2, chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
from chebfunjax.chebpref import ChebfunPref

TOL = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()
    gc.collect()


@pytest.mark.parametrize("clause", range(1, 11))
def test_native_compose_clause(clause):
    x = lambda x, y: x  # noqa: E731 - literal native coordinate handle
    y = lambda x, y: y  # noqa: E731 - literal native coordinate handle
    if clause == 1:
        f = Chebfun2v.from_functions(x, y, domain=(0.0, 1.0, 0.0, 1.0))
        g = chebfun2(lambda x, y: x + y)
        truth = chebfun2(lambda x, y: x + y, domain=(0.0, 1.0, 0.0, 1.0))
        assert (f.compose(g) - truth).norm() < TOL
    elif clause in (2, 3):
        f = Chebfun2v.from_functions(x, y)
        ops = [lambda x, y: x + y, lambda x, y: x - y]
        if clause == 3:
            ops.append(lambda x, y: x * y)
        g = Chebfun2v.from_functions(*ops)
        assert (f.compose(g) - g).norm() < TOL
    elif clause in (4, 5, 6):
        f = Chebfun2v.from_functions(x, y, lambda x, y: x + y)
        dom = (-1.0, 1.0, -1.0, 1.0, -2.0, 2.0)
        if clause == 4:
            g = Chebfun3.from_function(lambda x, y, z: x + y + z, domain=dom)
            truth = chebfun2(lambda x, y: 2 * x + 2 * y)
        elif clause == 5:
            g = Chebfun3v.from_functions(lambda x, y, z: x + y, lambda x, y, z: z - x, domain=dom)
            truth = Chebfun2v.from_functions(lambda x, y: x + y, y)
        else:
            g = Chebfun3v.from_functions(
                lambda x, y, z: x + y, lambda x, y, z: x - y, lambda x, y, z: z, domain=dom
            )
            truth = Chebfun2v.from_functions(
                lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x + y
            )
        assert (f.compose(g) - truth).norm() < TOL
    else:
        f1 = chebfun2(lambda x, y: jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y), trig=True)
        f2 = chebfun2(lambda x, y: jnp.cos(jnp.pi * x) * jnp.cos(jnp.pi * y), trig=True)
        f = Chebfun2v([f1.approx, f2.approx] + ([f1.approx] if clause >= 9 else []))
        if clause == 7:
            g = chebfun2(lambda x, y: x * x + y * y)
        elif clause == 8:
            g = Chebfun2v.from_functions(lambda x, y: x * x + y * y, x)
        elif clause == 9:
            g = Chebfun3.from_function(lambda x, y, z: jnp.exp(x) * jnp.cos(y) + 1)
        else:
            g = Chebfun3v.from_functions(
                lambda x, y, z: jnp.exp(x), lambda x, y, z: x * x + y * y, lambda x, y, z: z - 2
            )
        h = g(f)
        assert h.isPeriodicTech() if isinstance(h, Chebfun2) else h.is_periodic_tech()
