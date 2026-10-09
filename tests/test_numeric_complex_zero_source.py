"""Unbndfun native zeros() is real double, independent of numeric input dtype."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2, Trigtech])
@pytest.mark.parametrize('shape', [(), (3,), (1, 2), (3, 2)])
def test_complex_zero_native_storage_and_columns(cls, shape):
    saved = ChebfunPref._defaults
    try:
        ChebfunPref.setDefaults('factory')
        ChebfunPref.setDefaults('tech', cls)
        values = jnp.zeros(shape, dtype=jnp.complex128)
        for domain in [(0., float('inf')), (-1., 0., float('inf'))]:
            f = chebfun(values, domain=domain)
            assert f.n_columns == (shape[1] if len(shape) == 2 else 1)
            for p in f.funs:
                assert isinstance(p.tech, cls)
                assert bool(jnp.all(p.tech.coeffs == 0))
                if isinstance(p, Unbndfun):
                    if cls is Trigtech:
                        assert p.tech.is_real
                        assert p.tech.values.dtype == jnp.float64
                    else:
                        assert p.tech.coeffs.dtype == jnp.float64
                elif cls is not Trigtech:
                    assert p.tech.coeffs.dtype == jnp.complex128
            assert bool(jnp.all(f(jnp.array([0., 1., jnp.inf])) == 0))
            assert bool(jnp.all(f.point_values == 0))
    finally:
        ChebfunPref._defaults = saved
