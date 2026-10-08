"""Independent integral and selector controls for Chebfun3 source norms."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


@pytest.fixture(scope="module")
def constant():
    return Chebfun3.from_function(lambda x, y, z: 2+3j,
                                 domain=(0., 2., -1., 2., 3., 7.))


def test_even_norm_on_nonunit_domain(constant):
    expected = jnp.sqrt(13.)*24**(1/6)
    assert jnp.abs(constant.norm(jnp.asarray(6))-expected) < 2e-13


@pytest.mark.parametrize("p", [1, 2, 3, "min", "op", "operator", -jnp.inf])
def test_source_unsupported_norms(constant, p):
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:norm:norm:"):
        constant.norm(p)


def test_unknown_norm(constant):
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:norm:unknown:"):
        constant.norm("spectral")


def test_infinity_location():
    f = Chebfun3.from_function(lambda x, y, z: x+2*y+3*z)
    value, loc = f.norm("max", return_location=True)
    assert jnp.abs(value-6) < 2e-13
    assert jnp.abs(jnp.abs(f(*loc))-value) < 2e-13
