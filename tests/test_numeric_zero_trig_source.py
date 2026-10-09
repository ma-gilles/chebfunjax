"""Native numeric zero -> selected Trigtech operator -> Unbndfun mapping."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('shape', [(), (1, 2), (3, 2)])
@pytest.mark.parametrize('domain,points', [
    ((0., float('inf')), (0., 1., float('inf'))),
    ((float('-inf'), 0.), (float('-inf'), -1., 0.)),
    ((float('-inf'), float('inf')), (float('-inf'), 0., float('inf'))),
    ((float('-inf'), 0., 1., float('inf')), (float('-inf'), 0., .5, 1., float('inf'))),
])
def test_zero_trig_wrapper_values(shape, domain, points):
    saved = ChebfunPref._defaults
    try:
        ChebfunPref.setDefaults('factory')
        ChebfunPref.setDefaults('tech', 'trigtech')
        f = chebfun(jnp.zeros(shape), domain=domain)
        for piece in f.funs:
            assert isinstance(piece.tech, Trigtech)
            if bool(jnp.any(jnp.isinf(jnp.asarray(piece.interval)))):
                assert isinstance(piece, Unbndfun)
            assert bool(jnp.all(piece.tech.coeffs == 0))
        assert bool(jnp.all(f(jnp.asarray(points)) == 0))
        assert bool(jnp.all(f.point_values == 0))
        assert f.n_columns == (1 if not shape else shape[1])
    finally:
        ChebfunPref._defaults = saved
