"""JAX principal-branch Hankel functions for static real order.

Provenance
----------
Chebfun @chebfun/besselh.m, pin 7574c77 (builtin adapter).
Temme, J. Comput. Phys. 21, 343 (1976), small-argument K series;
Thompson and Barnett, Comput. Phys. Commun. 47, 245 (1987), CF2.
The recurrences follow Boost.Math bessel_ik.hpp, Copyright (c) 2006
Xiaogang Zhang, under the Boost Software License 1.0; see
LICENSES/BSL-1.0.txt.
DLMF 10.27.8 and 10.11.3--4 give rotation and branch continuation.
Numerical qualification is bounded; this is not a uniform asymptotic
implementation for arbitrarily large order and argument.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from chebfunjax.utils.besselj import besselj


def sinhc(z):
    return jnp.where(jnp.abs(z) < 1e-4, 1 + z * z / 6 + z**4 / 120 + z**6 / 5040, jnp.sinh(z) / z)


def gammas(mu):
    # log Gamma(1 +/- mu), |mu| <= 1/2. The 60-term remainder is
    # below binary64 roundoff. Odd differences are divided analytically
    # before evaluation, so integer and near-integer orders stay regular.
    k = jnp.arange(2, 62, dtype=jnp.float64)
    coeff = jsp.zeta(k, 1.0) / k
    average = jnp.sum(jnp.where(k % 2 == 0, coeff * mu**k, 0.0))
    divided = -0.5772156649015328606 - jnp.sum(jnp.where(k % 2 == 1, coeff * mu ** (k - 1), 0.0))
    half = mu * divided
    g1 = jnp.exp(-average) * divided * sinhc(half)
    g2 = jnp.exp(-average) * jnp.cosh(half)
    return g1, g2, average + half, average - half


def small(mu, w):
    g1, g2, lp, lm = gammas(mu)
    d = -jnp.log(w / 2)
    sigma = mu * d
    fact = 1 / jnp.sinc(mu)
    f = fact * (g1 * jnp.cosh(sigma) + g2 * d * sinhc(sigma))
    p = 0.5 * fact * jnp.exp(sigma - lm)
    q = 0.5 * fact * jnp.exp(-sigma - lp)

    def step(k, state):
        f, p, q, c, total, total1 = state
        f = (k * f + p + q) / (k * k - mu * mu)
        p = p / (k - mu)
        q = q / (k + mu)
        c = c * w * w / (4 * k)
        return f, p, q, c, total + c * f, total1 + c * (p - k * f)

    state = jax.lax.fori_loop(1, 80, step, (f, p, q, jnp.ones_like(w), f, p))
    return state[-2], 2 * state[-1] / w


def large(mu, w):
    a = mu * mu - 0.25
    b = 2 * (w + 1)
    D = 1 / b
    delta = D
    Q = -a
    S = 1 + Q * D

    def step(k, state):
        a, b, D, delta, f, prev, cur, C, Q, S, done = state
        a = a - 2 * (k - 1)
        b = b + 2
        D = 1 / (b + a * D)
        delta = delta * (b * D - 1)
        f = f + delta
        q = (prev - (b - 2) * cur) / a
        C = C * (-a / k)
        Q = Q + C * q
        Snew = S + Q * delta
        scale = jnp.where((jnp.abs(q) < 1e-50) & (jnp.abs(q) > 0), q, 1)
        new = (
            a,
            b,
            D,
            delta,
            f,
            cur / scale,
            q / scale,
            C * scale,
            Q,
            Snew,
            jnp.abs(Q * delta) < 2.220446049250313e-16 * jnp.abs(Snew),
        )
        return tuple(jnp.where(done, old, value) for old, value in zip(state, new))

    zero = jnp.zeros_like(w)
    one = jnp.ones_like(w)
    state = jax.lax.fori_loop(
        2, 401, step, (zero + a, b, D, delta, D, zero, one, zero - a, zero + Q, S, jnp.array(False))
    )
    f, S = state[4], state[9]
    K = jnp.sqrt(jnp.pi / (2 * w)) * jnp.exp(-w) / S
    return K, K * (0.5 + mu + w + (mu * mu - 0.25) * f) / w


def kval(nu, w):
    n = int(math.floor(nu + 0.5))
    mu = nu - n
    k, k1 = jax.lax.cond(jnp.abs(w) <= 2.5, lambda x: small(mu, x), lambda x: large(mu, x), w)

    def step(i, state):
        a, b = state
        return b, a + 2 * (mu + i) / w * b

    return k if n == 0 else jax.lax.fori_loop(1, n, step, (k, k1))[1]


def upper(nu, z):
    return 2 / (1j * jnp.pi) * jnp.exp(-0.5j * jnp.pi * nu) * kval(nu, -1j * z)


def right(nu, z):
    h = jax.lax.cond(
        jnp.imag(z) >= 0, lambda z: upper(nu, z), lambda z: jnp.conj(upper(nu, jnp.conj(z))), z
    )
    opposite = jnp.where(jnp.imag(z) == 0, jnp.conj(h), 2 * besselj(nu, z) - h)
    return jnp.where(jnp.imag(z) >= 0, h, opposite), jnp.where(jnp.imag(z) >= 0, opposite, h)


def scalar(nu, z, k):
    reflected = jnp.real(z) < 0
    r = jnp.where(reflected, -z, z)
    h1, h2 = right(abs(nu), r)
    phase = jnp.exp(1j * jnp.pi * abs(nu))
    cos = jnp.cos(jnp.pi * abs(nu))
    a = jnp.where(jnp.imag(z) >= 0, -h2 / phase, 2 * cos * h1 + h2 / phase)
    b = jnp.where(jnp.imag(z) >= 0, phase * h1 + 2 * cos * h2, -phase * h1)
    result = jnp.where(reflected, a if k == 1 else b, h1 if k == 1 else h2)
    if nu < 0:
        result = result * jnp.exp((1j if k == 1 else -1j) * jnp.pi * abs(nu))
    return result


@partial(jax.custom_jvp, nondiff_argnums=(0, 2))
def besselh(nu, z, k=1):
    """Evaluate H_nu^(k)(z), preserving complex values and principal cuts."""
    if isinstance(nu, complex):
        if nu.imag != 0:
            raise ValueError("besselh: order must be finite and real")
        nu = nu.real
    nu = float(nu)
    if not math.isfinite(nu):
        raise ValueError("besselh: order must be finite and real")
    if k not in (1, 2):
        raise ValueError("besselh: k must be 1 or 2")
    z = jnp.asarray(z, dtype=jnp.complex128)
    return jax.vmap(lambda x: scalar(nu, x, k))(z.reshape(-1)).reshape(z.shape)


@besselh.defjvp
def _besselh_jvp(nu, k, primals, tangents):
    (z,) = primals
    (dz,) = tangents
    value = besselh(nu, z, k)
    derivative = (besselh(nu - 1, z, k) - besselh(nu + 1, z, k)) / 2
    return value, derivative * dz


besselh = jax.jit(besselh, static_argnums=(0, 2))
