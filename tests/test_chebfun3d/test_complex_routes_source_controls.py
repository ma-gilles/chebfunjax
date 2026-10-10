"""Additional source contracts from @chebfun3/{conj,imag,permute}.m7574c77."""
import itertools

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('method', ['conj', 'imag', 'permute'])
def test_empty(method):
    f = Chebfun3.empty()
    result = f.permute([3, 1, 2]) if method == 'permute' else getattr(f, method)()
    assert result.isempty()


def test_periodic_factor_preservation():
    f = chebfun3(lambda x, y, z: jnp.cos(jnp.pi*x)+2*jnp.sin(jnp.pi*y)+3j*jnp.cos(jnp.pi*z), trig=True)
    factors = (f.cols, f.rows, f.tubes)
    for order in itertools.permutations(range(3)):
        g = f.permute(tuple(i+1 for i in order))
        assert jnp.array_equal(g.core, jnp.transpose(f.core, order))
        for actual, i in zip((g.cols, g.rows, g.tubes), order):
            assert all(a is b for a, b in zip(actual, factors[i], strict=True))
            assert all(isinstance(a, Trigtech) for a in actual)
    h = f.conj()
    assert all(isinstance(a, Trigtech) for a in h.cols+h.rows+h.tubes)
    x = jnp.array([-.7, .1, .8])
    assert jnp.max(jnp.abs(h(x, x/2, -x)-jnp.conj(f(x, x/2, -x)))) < 100*float(jnp.finfo(jnp.float64).eps)
