"""Source API controls for @chebfun3/jacobian.m and vector determinant dispatch.

Chebfun pin7574c77. These supplement the original vector definition predicate.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
from chebfunjax.chebpref import ChebfunPref


def test_scalar_jacobian_nondiagonal_and_source_empty_returns():
    domain = (0., 2., -3., 1., 2., 5.)
    f = Chebfun3.from_function(lambda x, y, z: x + 2*y + 3*z, domain=domain)
    g = Chebfun3.from_function(lambda x, y, z: 4*x + 5*y + 6*z, domain=domain)
    h = Chebfun3.from_function(lambda x, y, z: 7*x + 8*y + 10*z, domain=domain)
    expected = Chebfun3.from_function(lambda x, y, z: -3 + 0*x, domain=domain)
    actual = f.jacobian(g, h)
    assert tuple(actual.domain) == domain
    assert (actual - expected).norm() < 100*ChebfunPref().cheb3Prefs.chebfun3eps

    empty = Chebfun3.empty()
    for operands in ((empty, g, h), (f, empty, h), (f, g, empty)):
        assert jnp.asarray(Chebfun3.jacobian(*operands)).shape == (0, 0)
    assert jnp.asarray(Chebfun3v().jacobian()).shape == (0, 0)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3V:jacobian:notSquare"):
        Chebfun3v(f, g).jacobian()
