"""Native solvepde output construction and periodic Nyquist controls.

Provenance: @spinoperator/solvepde.m output fun(vOut,dom,'trig'),
Chebfun commit7574c77. Analytic supplemental bounds use100eps from
original tests/chebfun3/test_constructor.m.
"""
import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.operators.spinop3 import _make_trig_interp


def test_output_is_native_numeric_chebfun3():
    points = jnp.arange(8)*2*jnp.pi/8
    x, y, z = jnp.meshgrid(points, points, points, indexing='ij')
    f = _make_trig_interp(jnp.cos(x)+2*jnp.sin(y)+3*jnp.cos(z),
                          [(0., 2*jnp.pi)]*3)
    assert isinstance(f, Chebfun3)
    assert f.isPeriodicTech()
    assert f.length() == (8, 8, 8)


def test_real_nyquist_output_retains_cosine_off_grid():
    values = jnp.broadcast_to((-1.)**jnp.arange(8)[:, None, None], (8, 8, 8))
    f = _make_trig_interp(values, [(0., 2*jnp.pi)]*3)
    x = jnp.asarray([.13, .37, 1.21])
    actual = jnp.asarray(f(x, x*.3, x*.7))
    expected = jnp.cos(4*x)
    assert float(jnp.max(jnp.abs(actual-expected))) < 100*jnp.finfo(jnp.float64).eps
