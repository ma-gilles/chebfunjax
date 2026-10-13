"""Unbounded FUN ownership and root impulses in native abs/simplify.

Provenance
----------
MATLAB source: @chebfun/abs.m, addBreaksAtRoots.m, hscale.m, simplify.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.tech.chebtech import Chebtech2


def test_simplify_preserves_right_unbounded_map():
    domain = Domain((0., jnp.inf))
    piece = Unbndfun.from_chebtech(Chebtech2.from_coeffs(jnp.asarray([1., .5])), domain)
    f = Chebfun(funs=[piece], domain=domain)
    g = f.simplify()
    assert isinstance(g.funs[0], Unbndfun)
    assert g.funs[0].mapping_type == piece.mapping_type
    # Native right inverse map is (x-a-15)/(x-a+15).
    expected = jnp.asarray([.5, 1., 1.25, 1.4])
    np.testing.assert_allclose(g(jnp.asarray([0., 15., 45., 135.])), expected,
                               rtol=0, atol=4*jnp.finfo(jnp.float64).eps)


def test_entire_abs_preserves_maps_and_new_root_impulse():
    domain = Domain((-jnp.inf, jnp.inf))
    piece = Unbndfun.from_chebtech(Chebtech2.from_coeffs(jnp.asarray([0., 1.])), domain)
    f = Chebfun(funs=[piece], domain=domain).set_point_values(jnp.asarray([3., -4.]))
    g = f.abs()
    assert len(g.funs) == 2
    assert all(isinstance(part, Unbndfun) for part in g.funs)
    assert tuple(g.domain.breakpoints) == (-jnp.inf, 0., jnp.inf)
    np.testing.assert_array_equal(g.point_values, [3., 0., 4.])
    assert float(g(0.)) == 0.
