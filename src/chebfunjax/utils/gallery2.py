# uses-numpy: the airy gallery entries use scipy.special.airy on the
# complex plane (not JIT-safe by design).
"""Gallery of interesting bivariate functions as Chebfun2 examples.

A curated collection of ``Chebfun2`` objects illustrating a range of
interesting bivariate functions -- oscillatory, near-characteristic,
and classic optimisation test surfaces.  Intended for demonstrations,
benchmarks, and testing.

Translated from MATLAB Chebfun ``+cheb/gallery2.m`` (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

Provenance
----------
MATLAB source : +cheb/gallery2.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import random as _random
from typing import Callable

import jax.numpy as jnp

__all__ = ["gallery2", "list_gallery2"]

# ---------------------------------------------------------------------------
# Registry: name -> (description, domain, anonymous function fa)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[str, tuple, Callable]] = {}


def _register(name: str, description: str, domain: tuple):
    """Register a gallery2 entry: its description, domain, and handle fa."""
    def decorator(fa: Callable):
        _REGISTRY[name.lower()] = (description, domain, fa)
        return fa
    return decorator


# ---------------------------------------------------------------------------
# Gallery entries (each is the anonymous function fa(x, y); the domain is
# stored alongside).  gallery2(name) builds chebfun2(fa, domain).
# ---------------------------------------------------------------------------


@_register("airyreal", "Real part of Airy Ai on the complex plane",
           (-10.0, 10.0, 0.0, 1.0))
def _airyreal(x, y):
    import numpy as _np
    from scipy.special import airy as _airy
    z = _np.asarray(x) + 1j * _np.asarray(y)
    return jnp.asarray(_np.real(_airy(z)[0]))


@_register("airycomplex", "Airy Ai on the complex plane (complex-valued)",
           (-10.0, 10.0, -5.0, 5.0))
def _airycomplex(x, y):
    import numpy as _np
    from scipy.special import airy as _airy
    z = _np.asarray(x) + 1j * _np.asarray(y)
    return jnp.asarray(_airy(z)[0])


@_register("bump", "2D C-infinity function with compact support",
           (-2.0, 2.0, -2.0, 2.0))
def _bump(x, y):
    r2 = x ** 2 + y ** 2
    inside = r2 < 1.0
    return jnp.where(inside,
                     jnp.exp(-1.0 / jnp.where(inside, 1.0 - r2, 1.0)),
                     0.0)


@_register("challenge", "Function from the SIAM 100-digit challenge",
           (-1.0, 1.0, -1.0, 1.0))
def _challenge(x, y):
    return (jnp.exp(jnp.sin(50 * x)) + jnp.sin(60 * jnp.exp(y))
            + jnp.sin(70 * jnp.sin(x)) + jnp.sin(jnp.sin(80 * y))
            - jnp.sin(10 * (x + y)) + (x ** 2 + y ** 2) / 4)


@_register("peaks", "Classic MATLAB peaks function",
           (-3.0, 3.0, -3.0, 3.0))
def _peaks(x, y):
    return (3 * (1 - x) ** 2 * jnp.exp(-x ** 2 - (y + 1) ** 2)
            - 10 * (x / 5 - x ** 3 - y ** 5) * jnp.exp(-x ** 2 - y ** 2)
            - 1 / 3 * jnp.exp(-(x + 1) ** 2 - y ** 2))


@_register("rosenbrock", "Rosenbrock optimisation test function",
           (-2.0, 2.0, -1.0, 3.0))
def _rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


@_register("roundpeg", "Approx. characteristic function of a disk (rank 45)",
           (-1.0, 1.0, -1.0, 1.0))
def _roundpeg(x, y):
    return 1.0 / (1.0 + ((2 * x) ** 2 + (2 * y) ** 2) ** 10)


@_register("smokering", "A halo / hoop / hole",
           (-1.0, 1.0, -1.0, 1.0))
def _smokering(x, y):
    return jnp.exp(-100 * (x ** 2 - x * y + 2 * y ** 2 - 0.5) ** 2)


@_register("squarepeg", "Approx. characteristic function of a square (rank 1)",
           (-1.0, 1.0, -1.0, 1.0))
def _squarepeg(x, y):
    return 1.0 / ((1 + (2 * x) ** 20) * (1 + (2 * y) ** 20))


@_register("tiltedpeg", "A tilted version of squarepeg",
           (-1.0, 1.0, -1.0, 1.0))
def _tiltedpeg(x, y):
    return 1.0 / ((1 + (2 * x + 0.4 * y) ** 20) * (1 + (2 * y - 0.4 * x) ** 20))


@_register("waffle", "A function with horizontal and vertical ridges",
           (-1.0, 1.0, -1.0, 1.0))
def _waffle(x, y):
    return 1.0 / (1.0 + 1e3 * ((x ** 2 - 0.25) ** 2 * (y ** 2 - 0.25) ** 2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_gallery2() -> dict[str, str]:
    """Return a mapping from gallery2 name to one-line description.

    Examples
    --------
    >>> from chebfunjax.utils.gallery2 import list_gallery2
    >>> 'challenge' in list_gallery2()
    True
    """
    return {name: desc for name, (desc, _dom, _fa) in sorted(_REGISTRY.items())}


def gallery2(name: "str | None" = None, *, return_handle: bool = False):
    """Return a gallery Chebfun2 by name (MATLAB ``cheb.gallery2``).

    Parameters
    ----------
    name : str or None
        Name of the gallery function (case-insensitive).  If ``None`` a
        random entry is returned.  Use :func:`list_gallery2` for the names.
    return_handle : bool, optional
        If True, also return the anonymous function ``fa`` used to build the
        Chebfun2 (MATLAB ``[F, FA] = cheb.gallery2(NAME)``).

    Returns
    -------
    f : Chebfun2
        The requested gallery Chebfun2.
    fa : callable, optional
        The anonymous function (only if ``return_handle`` is True).

    Raises
    ------
    KeyError
        If *name* is not found.

    Notes
    -----
    Gallery functions are constructed lazily; each call rebuilds the
    Chebfun2 from scratch.  Some near-characteristic entries (``bump``,
    ``roundpeg``, ``tiltedpeg``, ``smokering``) are high rank and take
    several seconds to resolve.

    Provenance
    ----------
    MATLAB source : +cheb/gallery2.m
    Chebfun commit: 7574c77

    See Also
    --------
    list_gallery2, chebfunjax.utils.gallery.gallery
    """
    from chebfunjax.chebfun2d.chebfun2 import chebfun2

    if name is None:
        name = _random.choice(list(_REGISTRY.keys()))
    key = str(name).lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"gallery2 function {name!r} not found. "
            f"Available entries: {available}.")
    _desc, domain, fa = _REGISTRY[key]
    f = chebfun2(fa, domain=domain)
    if return_handle:
        return f, fa
    return f
