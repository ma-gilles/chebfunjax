"""Source IFFT normalization and unchanged polynomial/AD controls."""

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils.transforms import coeffs2vals, vals2coeffs


def test_native_constant_reconstruction_dc_word():
    # Native paired capture: fixed n=4000 reconstruction of constant3 has
    # c0 exactly3, while multiplication by rounded 1/7998 yields3+ulp.
    c = vals2coeffs(jnp.full((4000,), 3., dtype=jnp.float64))
    assert c[0].view(jnp.uint64).item() == 0x4008000000000000


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("columns", [False, True])
def test_independent_polynomial_and_tangent(complex_input, columns):
    c = jnp.zeros(18, dtype=jnp.float64).at[:4].set(jnp.array([.7, -.2, .3, .125]))
    if complex_input:
        c = c + 1j*jnp.roll(c, 2)
    if columns:
        c = jnp.stack((c, -2*c, 3*c), axis=1)
    v = coeffs2vals(c)
    actual, tangent = jax.jvp(vals2coeffs, (v,), (v*.125,))
    assert bool(jnp.all(jnp.abs(actual-c) < 32*jnp.finfo(jnp.float64).eps))
    assert bool(jnp.all(jnp.abs(tangent-c*.125) < 32*jnp.finfo(jnp.float64).eps))
