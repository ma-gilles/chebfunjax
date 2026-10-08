"""JAX shape-preserving Hermite interpolation for Chebfun.pchip.

Provenance
----------
MATLAB source: @chebfun/pchip.m, Chebfun commit 7574c77.
The standard weighted harmonic slope construction follows Fritsch and Carlson
(1980). Complex interpolation applies the real construction to each component,
as verified in the installed MATLAB R2025b pchip adapter. Coefficients are
obtained directly from the endpoint values and derivatives of each cubic.
"""
import jax
import jax.numpy as jnp


def _real_slopes(h, delta):
    if delta.shape[0] == 1:
        return jnp.concatenate((delta, delta), axis=0)
    before, after = delta[:-1], delta[1:]
    same = (jnp.sign(before) * jnp.sign(after)) > 0
    a = jnp.where(same, before, 1.0)
    b = jnp.where(same, after, 1.0)
    w1, w2 = 2 * h[1:] + h[:-1], h[1:] + 2 * h[:-1]
    interior = jnp.where(same, (w1 + w2) / (w1 / a + w2 / b), 0.0)

    def endpoint(h0, h1, d0, d1):
        derivative = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        derivative = jnp.where(jnp.sign(derivative) != jnp.sign(d0), 0.0, derivative)
        limited = (jnp.sign(d0) != jnp.sign(d1)) & (jnp.abs(derivative) > jnp.abs(3 * d0))
        return jnp.where(limited, 3 * d0, derivative)

    left = endpoint(h[0], h[1], delta[0], delta[1])
    right = endpoint(h[-1], h[-2], delta[-1], delta[-2])
    return jnp.concatenate((left[None], interior, right[None]), axis=0)


@jax.jit
def pchip_coefficients(x, y):
    """Ascending local-power coefficients for sites-first real/complex data."""
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.result_type(y, jnp.float64))
    h = jnp.diff(x).reshape((-1,) + (1,) * (y.ndim - 1))
    delta = jnp.diff(y, axis=0) / h
    if jnp.iscomplexobj(y):
        slopes = _real_slopes(h, jnp.real(delta)) + 1j * _real_slopes(h, jnp.imag(delta))
    else:
        slopes = _real_slopes(h, delta)
    quadratic = (3 * delta - 2 * slopes[:-1] - slopes[1:]) / h
    cubic = (slopes[:-1] + slopes[1:] - 2 * delta) / h**2
    return jnp.stack((y[:-1], slopes[:-1], quadratic, cubic), axis=1)
