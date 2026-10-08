"""All 35 original composition predicates, in source order and at source bounds.

Provenance
----------
MATLAB source : tests/chebfun3/test_compose.m
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp
import pytest

from chebfunjax import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

TOL = 1000 * float(jnp.finfo(jnp.float64).eps)


def _base(x, y, z):
    return jnp.cos(x * y * z) + jnp.sin(x * y * z) + y - 0.1


def test_pass01_multiplication():
    f = chebfun3(_base)
    g = chebfun3(lambda x, y, z: _base(x, y, z)
                * jnp.sin((x - .1) * (y + .4) * (z + .8)))
    x, y, z = (chebfun3(lambda x, y, z, k=k: (x, y, z)[k]) for k in range(3))
    assert (g - f * ((x - .1) * (y + .4) * (z + .8)).sin()).norm() < TOL


@pytest.fixture(scope="module", params=["sin", "cos", "sinh", "cosh", "tanh", "exp"])
def unary(request):
    name = request.param
    op = getattr(jnp, name)
    f = chebfun3(_base)
    argument = -f if name == "tanh" else f
    result = getattr(argument, name)()
    g = chebfun3(lambda x, y, z: op(-_base(x, y, z) if name == "tanh"
                                     else _base(x, y, z)))
    yield result, g
    # Each source group creates different Tucker shapes; bound the CI cache.
    jax.clear_caches()


@pytest.mark.parametrize("fiber_dim", [None, 1, 2, 3])
def test_pass02_through25_unary(unary, fiber_dim):
    result, g = unary
    if fiber_dim is not None:
        g_source = g
        g = chebfun3(lambda x, y, z: g_source(x, y, z), fiberDim=fiber_dim)
    assert (g - result).norm() < TOL


@pytest.fixture(scope="module")
def oscillatory():
    return chebfun3(lambda x, y, z: jnp.sin(10 * x * y * z),
                    (-1, 2, -1, 1, -3, -1))


def test_pass26_addition(oscillatory):
    f = oscillatory
    assert (f + f + f - 3 * f).norm() < 100 * TOL


def test_pass27_square(oscillatory):
    f = oscillatory
    assert (f * f - f ** 2).norm() < TOL


def test_pass28_scalar_chebfun():
    f = chebfun3(lambda x, y, z: x)
    g = chebfun(lambda t: t ** 2)
    assert (f.compose(g) - f ** 2).norm() < TOL


def test_pass29_array_chebfun():
    f = chebfun3(lambda x, y, z: x)
    g = chebfun(lambda t: jnp.stack((t, t ** 2), axis=-1))
    assert (f.compose(g) - Chebfun3v([f, f ** 2])).norm() < TOL


@pytest.fixture(scope="module")
def complex_composition():
    f = chebfun3(lambda x, y, z: x + y + 1j * z)
    g = Chebfun2.from_function(lambda x, y: x ** 2 + y ** 2, domain=(-2, 2, -1, 1))
    return f.compose(g)


def test_pass30_complex_chebfun2(complex_composition):
    h_true = chebfun3(lambda x, y, z: (x + y) ** 2 + z ** 2)
    assert (complex_composition - h_true).norm() < TOL


def test_pass31_nonperiodic(complex_composition):
    assert not complex_composition.isPeriodicTech()


@pytest.mark.parametrize("columns", [1, 2], ids=["pass32", "pass33"])
def test_periodic_chebfun(columns):
    f = chebfun3(lambda x, y, z: jnp.cos(jnp.pi * x), trig=True)
    g = chebfun(lambda t: t ** 2) if columns == 1 else chebfun(
        lambda t: jnp.stack((t ** 2, jnp.cos(t)), axis=-1))
    assert f.compose(g).isPeriodicTech()


@pytest.mark.parametrize("vector", [False, True], ids=["pass34", "pass35"])
def test_periodic_complex_chebfun2(vector):
    f = chebfun3(lambda x, y, z: jnp.exp(1j * jnp.pi * x), trig=True)
    g = (Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y,
                                domain=(-2, 2, -2, 2)) if vector else
         Chebfun2.from_function(lambda x, y: x + y, domain=(-2, 2, -2, 2)))
    assert f.compose(g).isPeriodicTech()
