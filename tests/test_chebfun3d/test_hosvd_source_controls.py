"""Independent HOSVD checks for complex, periodic and dependent factors.

Source: @chebfun3/{hosvd,discreteHOSVD}.m, Chebfun7574c77.
The explicit Tucker inputs isolate this algorithm from adaptive construction.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('case', ['complex', 'periodic', 'dependent'])
def test_explicit_factor_hosvd(case):
    domain = (-1., 1., -1., 1., -1., 1.)
    constant = Chebtech2.from_coeffs(jnp.asarray([1.]))
    if case == 'complex':
        factor = Chebtech2.from_coeffs(jnp.asarray([1.+1.j]))
        f = Chebfun3([factor], [constant], [constant], jnp.asarray([[[2.-1.j]]]), domain)
        expected_norm = jnp.sqrt(80.)
    elif case == 'periodic':
        factor = Trigtech.from_function(lambda x: jnp.cos(jnp.pi*x))
        f = Chebfun3([factor], [factor], [factor], jnp.asarray([[[2.]]]), domain)
        expected_norm = 2.
    else:
        f = Chebfun3([constant, constant], [constant], [constant],
                     jnp.asarray([[[1.]], [[2.]]]), domain)
        expected_norm = 3*jnp.sqrt(8.)
    values, g = f.hosvd()
    eps = jnp.finfo(jnp.float64).eps
    for mode in values:
        assert mode.shape == (1,)
        assert jnp.abs(mode[0]-expected_norm) < 200*eps*expected_norm
    x = jnp.asarray([-.91, -.27, .13, .79])
    xx, yy, zz = jnp.meshgrid(x, x, x, indexing='ij')
    assert jnp.max(jnp.abs(g(xx, yy, zz)-f(xx, yy, zz))) < 200*eps*expected_norm
    if case == 'periodic':
        assert all(isinstance(t, Trigtech) for group in (g.cols, g.rows, g.tubes) for t in group)
