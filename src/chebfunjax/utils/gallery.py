# uses-numpy: bessel gallery entry uses scipy.special (not JIT-safe by design)
"""Gallery of interesting functions as Chebfun examples.

A curated collection of ``Chebfun`` objects illustrating a range of
interesting mathematical functions — smooth, oscillatory, nearly-non-smooth,
and functions with endpoint singularities.  Intended for demonstrations,
benchmarks, and testing.

Translated from MATLAB Chebfun ``+cheb/gallery.m`` (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

Provenance
----------
MATLAB source : +cheb/gallery.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import random as _random
from typing import Callable

import jax.numpy as jnp

__all__ = ["gallery", "list_gallery"]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[str, Callable]] = {}


def _register(name: str, description: str):
    """Decorator that registers a zero-argument factory in ``_REGISTRY``."""
    def decorator(fn: Callable):
        _REGISTRY[name.lower()] = (description, fn)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Gallery entries  (each returns a Chebfun)
# ---------------------------------------------------------------------------


@_register("runge", "Runge function 1/(1 + 25x^2) on [-1, 1]")
def _runge():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: 1.0 / (1.0 + 25.0 * x ** 2))


@_register("bump", "C-infinity bump exp(-1/(1-x^2)) on [-2, 2]")
def _bump():
    from chebfunjax.chebfun1d.chebfun import chebfun
    def f(x):
        # Use jnp.where to handle the case |x| >= 1
        inner = jnp.where(jnp.abs(x) < 1.0, -1.0 / (1.0 - x ** 2), 0.0)
        return jnp.where(jnp.abs(x) < 1.0, jnp.exp(inner), 0.0)
    return chebfun(f, domain=(-2.0, 2.0))


@_register("chirp", "Chirp sin(x * exp(x)) on [0, 5]")
def _chirp():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: jnp.sin(x * jnp.exp(x)), domain=(0.0, 5.0))


@_register("erf", "Error function erf(x) on [-10, 10]")
def _erf():
    import jax.scipy.special as jsp

    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(jsp.erf, domain=(-10.0, 10.0))


@_register("fishfillet", "Wild oscillations cos(x)*sin(exp(x)) on [0, 6]")
def _fishfillet():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: jnp.cos(x) * jnp.sin(jnp.exp(x)), domain=(0.0, 6.0))


@_register("sinefun1", "1.75 + sin(50x) on [-1, 1] — smooth as it looks")
def _sinefun1():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: 1.75 + jnp.sin(50.0 * x))


@_register("sinefun2", "(1.75 + sin(50x))^1.0001 — not as smooth as it looks")
def _sinefun2():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: (1.75 + jnp.sin(50.0 * x)) ** 1.0001)


@_register("kahaner", "Four-spike integrand on [0, 1] (Kahaner benchmark)")
def _kahaner():
    from chebfunjax.chebfun1d.chebfun import chebfun
    def f(x):
        return (
            1.0 / jnp.cosh(10.0 * (x - 0.2)) ** 2
            + 1.0 / jnp.cosh(100.0 * (x - 0.4)) ** 4
            + 1.0 / jnp.cosh(1000.0 * (x - 0.6)) ** 6
            + 1.0 / jnp.cosh(1000.0 * (x - 0.8)) ** 8
        )
    return chebfun(f, domain=(0.0, 1.0))


@_register("seismograph", "tanh(20*sin(12x)) + 0.02*exp(3x)*sin(300x) on [-1, 1]")
def _seismograph():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(
        lambda x: jnp.tanh(20.0 * jnp.sin(12.0 * x))
        + 0.02 * jnp.exp(3.0 * x) * jnp.sin(300.0 * x)
    )


@_register("gaussian", "Standard Gaussian exp(-x^2/2)/sqrt(2*pi) on [-6, 6]")
def _gaussian():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(
        lambda x: jnp.exp(-0.5 * x ** 2) / jnp.sqrt(2.0 * jnp.pi),
        domain=(-6.0, 6.0),
    )


@_register("bessel", "Bessel J_0 on [-50, 50]")
def _bessel():
    import scipy.special as ssp

    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: jnp.array(ssp.j0(x), dtype=jnp.float64),
                   domain=(-50.0, 50.0))


@_register("wiggly", "exp(x)*sin(10*pi*x) on [-1, 1]")
def _wiggly():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: jnp.exp(x) * jnp.sin(10.0 * jnp.pi * x))


@_register("spikycomb", "exp(x)*sech(4*sin(40x))^exp(x) on [-1, 1] — 25 peaks")
def _spikycomb():
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(
        lambda x: jnp.exp(x) * (1.0 / jnp.cosh(4.0 * jnp.sin(40.0 * x))) ** jnp.exp(x)
    )


@_register("wild", "cos(x)^2 * sin(x^3) on [-1, 1]")
def _wild():
    """A rapidly oscillating function near x=1."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: jnp.cos(x) ** 2 * jnp.sin(x ** 3))


@_register("zigzag", "Degree-high polynomial that looks piecewise linear on [-1, 1]")
def _zigzag():
    """Chebyshev polynomial T_10000 approximated at lower resolution."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    # T_n has n+1 extreme values in [-1,1]; for large n it looks piecewise linear
    # We use n=5000 for a tractable demo
    n = 5000
    return chebfun(lambda x: jnp.cos(n * jnp.arccos(jnp.clip(x, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_gallery() -> dict[str, str]:
    """Return a mapping from gallery name to description.

    Returns
    -------
    mapping : dict[str, str]
        Dictionary with gallery function names as keys and one-line
        descriptions as values.

    Examples
    --------
    >>> from chebfunjax.utils.gallery import list_gallery
    >>> 'runge' in list_gallery()
    True
    """
    return {name: desc for name, (desc, _) in sorted(_REGISTRY.items())}


def gallery(name: str | None = None):
    """Return a gallery Chebfun by name.

    Parameters
    ----------
    name : str or None
        Name of the gallery function (case-insensitive).  If ``None`` or not
        provided, a random entry is returned.  Use :func:`list_gallery` to
        see all available names.

    Returns
    -------
    f : Chebfun
        The requested gallery Chebfun.

    Raises
    ------
    KeyError
        If *name* is not found in the gallery.

    Examples
    --------
    >>> from chebfunjax.utils.gallery import gallery
    >>> f = gallery('runge')
    >>> abs(float(f(0.0)) - 1.0) < 1e-12
    True
    >>> f = gallery('chirp')
    >>> f.domain.a, f.domain.b
    (0.0, 5.0)

    Notes
    -----
    Gallery functions are constructed lazily; each call to :func:`gallery`
    builds the Chebfun from scratch.

    Provenance
    ----------
    MATLAB source : +cheb/gallery.m
    Chebfun commit: 7574c77

    See Also
    --------
    list_gallery
    """
    if name is None:
        name = _random.choice(list(_REGISTRY.keys()))

    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Gallery function {name!r} not found. "
            f"Available entries: {available}."
        )

    _, factory = _REGISTRY[key]
    return factory()
