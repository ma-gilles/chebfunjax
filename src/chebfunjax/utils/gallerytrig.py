"""Gallery of interesting periodic (trigonometric) functions.

The periodic analogue of :mod:`chebfunjax.utils.gallery`: a curated set
of ``Chebfun`` objects built in the Fourier (``trig=True``) basis.

Translated from MATLAB Chebfun ``+cheb/gallerytrig.m`` (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun
Developers.  See https://www.chebfun.org/.

Added by Claude Opus 4.8 (task #20 — gallerytrig was missing).

Provenance
----------
MATLAB source : +cheb/gallerytrig.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import random as _random
from typing import Callable

import jax.numpy as jnp

__all__ = ["gallerytrig", "list_gallerytrig"]

_REGISTRY: dict[str, tuple[str, Callable]] = {}


def _register(name: str, description: str):
    def decorator(fn: Callable):
        _REGISTRY[name.lower()] = (description, fn)
        return fn
    return decorator


@_register("amsignal", "Amplitude-modulated signal cos(50x)(1+0.2cos(5x))")
def _amsignal():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: jnp.cos(50 * x) * (1 + 0.2 * jnp.cos(5 * x)),
                   domain=(-jnp.pi, jnp.pi), trig=True)


@_register("fmsignal", "Frequency-modulated signal cos(50x + 4 sin(5x))")
def _fmsignal():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: jnp.cos(50 * x + 4 * jnp.sin(5 * x)),
                   domain=(-jnp.pi, jnp.pi), trig=True)


@_register("wavepacket", "Gaussian wave packet exp(-5x^2) cos(50x)")
def _wavepacket():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: jnp.exp(-5 * x ** 2) * jnp.cos(50 * x),
                   domain=(-jnp.pi, jnp.pi), trig=True)


@_register("sinefun1", "1.75 + sin(16*pi*x) on [-1, 1] (periodic)")
def _sinefun1():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: 1.75 + jnp.sin(16 * jnp.pi * x), trig=True)


@_register("sinefun2", "(1.75 + sin(16*pi*x))^1.0001 — nearly non-smooth")
def _sinefun2():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: (1.75 + jnp.sin(16 * jnp.pi * x)) ** 1.0001,
                   trig=True)


@_register("starburst", "(3 + sin(10t) + sin(61 e^{.8 sin t + .7})) e^{it}")
def _starburst():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(
        lambda t: (3 + jnp.sin(10 * t)
                   + jnp.sin(61 * jnp.exp(0.8 * jnp.sin(t) + 0.7)))
        * jnp.exp(1j * t),
        domain=(-jnp.pi, jnp.pi), trig=True)


@_register("gibbs", "Fourier partial sum of a square wave (Gibbs phenomenon)")
def _gibbs():
    from chebfunjax.chebfun1d.chebfun import chebfun
    modes = jnp.arange(1, 20, 2, dtype=jnp.float64)  # 1,3,...,19

    def f(x):
        x = jnp.asarray(x)[..., None]
        return jnp.sum(4.0 / jnp.pi * jnp.sin(modes * x) / modes, axis=-1)
    return chebfun(f, domain=(-jnp.pi, jnp.pi), trig=True)


@_register("weierstrass", "Weierstrass-type sum 2^-k cos(4^k x), k=1..8")
def _weierstrass():
    from chebfunjax.chebfun1d.chebfun import chebfun
    K = jnp.arange(1, 9, dtype=jnp.float64)

    def f(x):
        x = jnp.asarray(x)[..., None]
        return jnp.sum(2.0 ** (-K) * jnp.cos(4.0 ** K * x), axis=-1)
    return chebfun(f, domain=(-jnp.pi / 4, jnp.pi / 4), trig=True)


@_register("tsunami", "Periodic BVP u''+u'+600(1+sin x)u = 1 on [-pi,pi]")
def _tsunami():
    # Added by Claude Opus 4.8 (needed chebop periodic BC, now available).
    from chebfunjax.operators.chebop import Chebop
    N = Chebop(lambda x, u: u.diff(2) + u.diff()
               + 600 * (1 + jnp.sin(x)) * u, domain=(-jnp.pi, jnp.pi))
    N.bc = "periodic"
    return N.solve(1.0)


@_register("random", "Smooth band-limited random periodic function")
def _random_entry():
    # Added by Claude Opus 4.8. MATLAB: 2*randnfun(0.1, 'trig').
    import random as _r

    import jax
    from chebfunjax.utils.randnfun import randnfun
    key = jax.random.PRNGKey(_r.randint(0, 2 ** 31 - 1))
    f = randnfun(0.1, domain=(-jnp.pi, jnp.pi), key=key)
    return 2.0 * f


@_register("noisyfun", "Smooth curve plus band-limited random noise")
def _noisyfun():
    # Added by Claude Opus 4.8.
    import random as _r

    import jax
    from chebfunjax.chebfun1d.chebfun import chebfun
    from chebfunjax.utils.randnfun import randnfun
    base = chebfun(lambda x: jnp.sin(2 * x), domain=(-jnp.pi, jnp.pi),
                   trig=True)
    key = jax.random.PRNGKey(_r.randint(0, 2 ** 31 - 1))
    noise = randnfun(0.15, domain=(-jnp.pi, jnp.pi), key=key)
    return base + 0.2 * noise


def list_gallerytrig() -> dict[str, str]:
    """Return a mapping from gallerytrig name to description."""
    return {name: desc for name, (desc, _) in sorted(_REGISTRY.items())}


def gallerytrig(name: str | None = None):
    """Return a periodic gallery Chebfun by name (case-insensitive).

    If ``name`` is None, a random entry is returned.  Use
    :func:`list_gallerytrig` for the available names.
    """
    if name is None:
        name = _random.choice(list(_REGISTRY.keys()))
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"gallerytrig function {name!r} not found. "
            f"Available entries: {available}."
        )
    _, factory = _REGISTRY[key]
    return factory()
