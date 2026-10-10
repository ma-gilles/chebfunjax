"""Fractional-integral smooth coefficients, @chebtech/fracInt.m7574c77."""
import jax
import jax.numpy as jnp
from jax.scipy.special import gamma, gammaln, gammasgn


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
            out = jac2cheb(c_leg * _fractional_beta(k + 1., mu) / gamma(mu), -mu, mu)
    else:
        c_jac = cheb2jac(c, 0., b)
        out = jac2cheb(c_jac * _fractional_beta(k + b + 1., mu) / gamma(mu), -mu, b + mu)
    return out[:, None] if matrix else out


def _fractional_beta(a, b):
    """Signed beta with the source algdiv geometric ratio.

    The installed JAX 0.11 algdiv uses a/(a+b) where its upstream
    SciPy cdflib algdiv.f (c89dfc2, lines 38--40) uses b/(a+b),
    for sorted a <= b. This changes the Stirling remainder difference.
    Keep native beta/gamma scaling; correct that ratio locally in JAX.
    Parameters of the fractional operator are real.
    """
    a, b = jnp.minimum(a, b), jnp.maximum(a, b)
    h = a / b
    c = h / (1 + h)
    x = 1 / (1 + h)
    d = b + (a - 0.5)
    x2 = x * x
    s3 = 1 + (x + x2)
    s5 = 1 + (x + x2 * s3)
    s7 = 1 + (x + x2 * s5)
    s9 = 1 + (x + x2 * s7)
    s11 = 1 + (x + x2 * s9)
    t = (1 / b) ** 2
    w = (
        (
            (
                (-0.165322962780713e-2 * s11 * t + 0.837308034031215e-3 * s9) * t
                - 0.595202931351870e-3 * s7
            )
            * t
            + 0.793650666825390e-3 * s5
        )
        * t
        - 0.277777777760991e-2 * s3
    ) * t + 0.833333333333333e-1
    w = w * (c / b)
    u = d * jnp.log1p(a / b)
    v = a * (jnp.log(b) - 1)
    alg = jnp.where(u <= v, (w - v) - u, (w - u) - v)
    ln = jnp.where(b < 8, gammaln(a) + (gammaln(b) - gammaln(a + b)), gammaln(a) + alg)
    return gammasgn(a) * gammasgn(b) * gammasgn(a + b) * jnp.exp(ln)
