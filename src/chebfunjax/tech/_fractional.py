"""Fractional-integral smooth coefficients, @chebtech/fracInt.m7574c77."""
import jax
import jax.numpy as jnp
from jax.scipy.special import beta, gamma


def fractional_smooth_coefficients(coefficients, mu, b=0.0):
    """Return native smooth-factor coefficients using only JAX arithmetic."""
    from chebfunjax.utils.transforms import cheb2jac, cheb2leg, jac2cheb

    n = coefficients.shape[0]
    if coefficients.ndim == 2 and coefficients.shape[1] > 1:
        raise ValueError("CHEFBFUN:CHEBTECH:fracInt:arrayvalued: "
                         "Array-valued CHEBTECH objects are not supported.")
    matrix = coefficients.ndim == 2
    c = coefficients[:, 0] if matrix and coefficients.shape[1] else coefficients
    if n == 0:
        return coefficients
    k = jnp.arange(n, dtype=jnp.float64)
    if b == 0:
        c_leg = cheb2leg(c)
        if mu == 0.5:
            c_scl = c_leg / ((k + 0.5) * gamma(0.5))
            z = jnp.zeros((1,), dtype=c_scl.dtype)
            rhs = (jnp.concatenate((c_scl, z)) +
                   jnp.concatenate((z, c_scl)))[1:]
            # Native sparse upper-triangular D has diagonals(.5,1,.5)
            # and D[0,0]=1. Back substitution needs only O(n) storage.
            def step(j, work):
                i = n - 1 - j
                value = rhs[i] - work[i + 1] - 0.5 * work[i + 2]
                return work.at[i].set(value / jnp.where(i == 0, 1., 0.5))
            work = jax.lax.fori_loop(0, n, step, jnp.zeros((n + 2,), dtype=rhs.dtype))
            out = jnp.concatenate((work[:n], z))
        else:
            out = jac2cheb(c_leg * beta(k + 1., mu) / gamma(mu), -mu, mu)
    else:
        c_jac = cheb2jac(c, 0., b)
        out = jac2cheb(c_jac * beta(k + b + 1., mu) / gamma(mu), -mu, b + mu)
    return out[:, None] if matrix else out
