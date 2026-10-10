"""Native empty and identity routes, @singfun/{flipud,fliplr,diff}.m7574c77."""
import jax.numpy as jnp
import pytest

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2


@pytest.mark.parametrize("exponents", [(), (0., 0.), (-0.25, 0.5)])
def test_empty_flip_and_diff_exponents(exponents):
    f = Singfun(Chebtech2.empty(), exponents)
    reflected = f.flipud()
    assert reflected.isempty()
    assert reflected.exponents == tuple(reversed(exponents))
    assert f.exponents == exponents
    assert f.fliplr() is f
    assert f.diff() is f
    assert f.diff(0) is f


def test_nonempty_identity_and_reflection():
    f = Singfun(Chebtech2.from_coeffs(jnp.array([2., 0.5])), (-0.25, 0.5))
    assert f.fliplr() is f
    assert f.diff(0) is f
    g = f.flipud()
    assert g.exponents == (0.5, -0.25)
    assert bool(jnp.all(g.smoothPart.coeffs == jnp.array([2., -0.5])))
