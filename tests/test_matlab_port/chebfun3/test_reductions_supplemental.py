"""Previously useful analytic smoke controls, retained as supplements."""
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


def test_supplemental_sum_over_z_of_xyz():
    f = Chebfun3.from_function(lambda x, y, z: x*y*z**2)
    s = f.sum(dim=3)
    for xv in np.linspace(-0.9, 0.9, 8):
        for yv in np.linspace(-0.9, 0.9, 8)[::3]:
            assert abs(float(s(jnp.asarray(xv), jnp.asarray(yv)))
                       - (2/3)*xv*yv) < 1e5 * np.finfo(np.float64).eps


def test_supplemental_sum_over_x_of_odd_is_zero():
    f = Chebfun3.from_function(lambda x, y, z: x*jnp.cos(y*z))
    s = f.sum(dim=1)
    assert abs(float(s(jnp.asarray(0.4), jnp.asarray(0.3)))) < 1e5 * np.finfo(np.float64).eps


def test_supplemental_sum2_over_xy():
    f = Chebfun3.from_function(lambda x, y, z: x**2*y**2*jnp.cos(z))
    s = f.sum2((1, 2))
    for z in np.linspace(-0.9, 0.9, 7):
        assert abs(float(s(jnp.asarray(z))) - (4/9)*np.cos(z)) < 1e5 * np.finfo(np.float64).eps
