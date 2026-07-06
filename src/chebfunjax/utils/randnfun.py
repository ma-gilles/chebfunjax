"""Smooth random functions (randnfun / randnfuntrig).

A finite Fourier--Wiener series: a band-limited random function with a
prescribed wavelength ``lam``, normalized to roughly unit amplitude.
Deterministic given a JAX PRNG ``key``.

Translated (spirit of) MATLAB Chebfun ``randnfun.m`` (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun
Developers.  Added by Claude Opus 4.8.

Provenance
----------
MATLAB source : randnfun.m (periodic / trig branch)
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["randnfun"]


def randnfun(lam: float = 0.2, domain=(-1.0, 1.0), *, key=None):
    """Smooth random (band-limited) periodic function.

    Parameters
    ----------
    lam : float, default 0.2
        Approximate wavelength of the random oscillations.
    domain : (float, float), default (-1, 1)
        Interval.
    key : jax PRNGKey, optional
        Randomness source.  If None, a fixed key is used (deterministic).

    Returns
    -------
    Chebfun (periodic / trig).
    """
    from chebfunjax.chebfun1d.chebfun import chebfun

    a, b = float(domain[0]), float(domain[1])
    length = b - a
    m = max(1, int(jnp.floor(length / lam)))
    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    # real Fourier coefficients for cos/sin modes 1..m plus a mean term,
    # normalized so E[f^2] ~ 1.
    acoef = jax.random.normal(k1, (m + 1,), dtype=jnp.float64)
    bcoef = jax.random.normal(k2, (m,), dtype=jnp.float64)
    scale = 1.0 / jnp.sqrt(m + 0.5)
    acoef = acoef * scale
    bcoef = bcoef * scale
    ks = jnp.arange(1, m + 1, dtype=jnp.float64)

    def f(x):
        theta = 2.0 * jnp.pi * (jnp.asarray(x) - a) / length
        out = acoef[0] / jnp.sqrt(2.0)
        out = out + jnp.sum(
            acoef[1:] * jnp.cos(ks * theta[..., None])
            + bcoef * jnp.sin(ks * theta[..., None]), axis=-1)
        return out

    return chebfun(f, domain=(a, b), trig=True)
