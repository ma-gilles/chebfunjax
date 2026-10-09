"""API branches of @chebfun3v/compose.m at Chebfun commit 7574c77."""

import jax.numpy as jnp
import pytest

from chebfunjax import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v


@pytest.fixture(scope="module")
def coordinates():
    return [chebfun3(lambda x, y, z, k=k: (x, y, z)[k]) for k in range(3)]


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("end", [-1, 1])
def test_three_component_range_rejection(coordinates, axis, end):
    fields = list(coordinates)
    fields[axis] = fields[axis] + end * .1
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3V:COMPOSE:DomainMismatch3"):
        Chebfun3v(fields).compose(coordinates[0])


def test_source_scaled_range_tolerance():
    # Tolerance includes the first component domain norm and the largest
    # component/output vscale; all six range endpoints obey source isSubset.
    dom = (-10, 10, -10, 10, -10, 10)
    f = chebfun3(lambda x, y, z: 1 + 1e-13 + 0*x, dom)
    g = chebfun3(lambda x, y, z: x + y + z)
    h = Chebfun3v(f, f, f).compose(g)
    assert abs(float(h(0, 0, 0)) - 3*(1+1e-13)) < 1e-13


@pytest.mark.parametrize("n", [1, 2, 3])
def test_real_restriction_precedes_dispatch(coordinates, n):
    f = Chebfun3v([coordinates[0] + 1j] + coordinates[1:n])
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3V:COMPOSE:Complex"):
        f.compose(lambda x: x)


@pytest.mark.parametrize("n", [2, 3])
def test_operand_diagnostics(coordinates, n):
    with pytest.raises(ValueError, match=f"CHEBFUN:CHEBFUN3V:COMPOSE:OP{n}"):
        Chebfun3v(coordinates[:n]).compose(lambda x: x)


def test_single_component_scalar_dispatch(coordinates):
    h = Chebfun3v(coordinates[0]).compose(chebfun(lambda t: t*t))
    assert isinstance(h, Chebfun3)
    assert abs(float(h(.2, .3, .4)) - .04) < 1e-14


@pytest.mark.parametrize("vector", [False, True])
def test_single_component_complex_split(coordinates, vector):
    f = Chebfun3v(coordinates[0] + 1j*coordinates[1])
    op = (Chebfun2v.from_functions(lambda x, y: x+y, lambda x, y: x-y) if vector
          else Chebfun2.from_function(lambda x, y: x+y))
    h = f.compose(op)
    scalar = h[0] if vector else h
    assert abs(float(scalar(.2, .3, .4)) - .5) < 1e-14
    if vector:
        assert abs(float(h[1](.2, .3, .4)) + .1) < 1e-14


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_empty_scalar_result_source_error(coordinates, n):
    f = Chebfun3v([chebfun3() for _ in range(n)])
    assert f.isempty()
    assert f.domain is None
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:get:propName"):
        f.compose(coordinates[0])


def test_empty_scalar_operand_source_error(coordinates):
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:get:propName"):
        Chebfun3v(coordinates).compose(chebfun3())


@pytest.mark.parametrize("dimension", [2, 3])
def test_empty_vector_result(dimension, coordinates):
    op = Chebfun2v.empty() if dimension == 2 else Chebfun3v()
    assert Chebfun3v(coordinates).compose(op).isempty()
    assert Chebfun3v().compose(op).isempty()


def test_periodic_values_and_vector_orientation():
    f = chebfun3(lambda x, y, z: jnp.cos(jnp.pi*x)*jnp.sin(jnp.pi*y), trig=True)
    g = chebfun3(lambda x, y, z: jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y), trig=True)
    F = Chebfun3v(f, g, f, is_transposed=True)
    op = Chebfun3v.from_functions(lambda x, y, z: x+z, lambda x, y, z: y)
    h = F.compose(op)
    assert h.isPeriodicTech()
    assert not h.is_transposed
    assert abs(float(h[0](.2, .3, .4) - 2*f(.2, .3, .4))) < 1e-13
    assert abs(float(h[1](.2, .3, .4) - g(.2, .3, .4))) < 1e-13


def test_single_component_vector_operand_returns_scalar(coordinates):
    h = Chebfun3v(coordinates).compose(Chebfun3v(coordinates[0]))
    assert isinstance(h, Chebfun3)
    assert abs(float(h(.2, .3, .4)) - .2) < 1e-14


@pytest.mark.parametrize("empty_first", [False, True])
def test_mixed_empty_component_domains_rejected(coordinates, empty_first):
    components = [chebfun3(), coordinates[0]]
    if not empty_first:
        components.reverse()
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3V:domainCheck"):
        Chebfun3v(components)
