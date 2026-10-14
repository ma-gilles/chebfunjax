"""Default order, physical domains and source empty directional derivatives."""
import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


def test_directional_defaults_orders_domain_and_empty():
    domain = (-2., 1., 0., 2., -1., 3.)
    f = Chebfun3.from_function(lambda x, y, z: x*x+2*y**3+3*z**4, domain=domain)
    point = (.2, .3, .4)
    first = (.4, 6*.3**2, 12*.4**3)
    second = (2., 12*.3, 36*.4**2)
    empty = Chebfun3.empty()
    for axis, expected1, expected2 in zip('xyz', first, second):
        method = getattr(f, 'diff'+axis)
        d1, d2 = method(), method(2)
        tol = (1e4 if axis == 'x' else 1e6)*jnp.finfo(float).eps
        assert tuple(d1.domain) == domain and tuple(d2.domain) == domain
        assert abs(float(d1(*point))-expected1) < tol
        assert abs(float(d2(*point))-expected2) < tol
        assert getattr(empty, 'diff'+axis)() is empty
        assert getattr(empty, 'diff'+axis)(2) is empty
