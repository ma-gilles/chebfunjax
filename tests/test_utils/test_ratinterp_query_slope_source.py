"""Query AD at source-continuous and source-discontinuous nodes.

Provenance
----------
MATLAB source : ratinterp.m/ratbary (primal isolated-node behavior)
Chebfun commit: 7574c77
Query derivative policy is a JAX extension; undefined source-node slopes are NaN.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._ratbary import _source_type2_rational_handle


@pytest.mark.parametrize("compiled", [False, True])
def test_mixed_regular_invalid_forward_reverse_slopes(compiled):
    r = _source_type2_rational_handle(jnp.array([1.5, 0., .5]), jnp.array([2., 1.]), (-1., 1.))
    # Midpoint is the source's isolated numerator-only discontinuity.
    x = jnp.array([.25, 0., -.25])
    direction = jnp.array([2., 3., -4.])
    def forward(z, dz):
        return jax.jvp(r, (z,), (dz,))[1]
    reverse = jax.grad(lambda z: jnp.sum(r(z)))
    if compiled:
        forward, reverse = jax.jit(forward), jax.jit(reverse)
    tangent, gradient = forward(x, direction), reverse(x)
    slope = (2*x*(2+x)-(1+x*x))/(2+x)**2
    idx = jnp.array([0, 2])
    np.testing.assert_allclose(tangent[idx], (slope*direction)[idx], rtol=0, atol=200*np.finfo(float).eps)
    np.testing.assert_allclose(gradient[idx], slope[idx], rtol=0, atol=200*np.finfo(float).eps)
    assert jnp.isnan(tangent[1])
    assert jnp.isnan(gradient[1])


def test_complex_direction_uses_holomorphic_elementwise_slope():
    a0, a1, b0, b1 = 1+.2j, .3-.1j, 2+.1j, -.2+.05j
    r = _source_type2_rational_handle(jnp.array([a0,a1]), jnp.array([b0,b1]), (-1.,1.))
    x = jnp.array([.2+.3j, -.4+.1j])
    direction = jnp.array([-.7+.2j, .1-.9j])
    expected = (a1*b0-a0*b1)/(b0+b1*x)**2
    _, tangent = jax.jvp(r, (x,), (direction,))
    gradient = jax.vmap(jax.grad(r, holomorphic=True))(x)
    np.testing.assert_allclose(tangent, expected*direction, rtol=0, atol=200*np.finfo(float).eps)
    np.testing.assert_allclose(gradient, expected, rtol=0, atol=200*np.finfo(float).eps)
