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


@_register("airy", "Airy function Ai(x) on [-40, 40]")
def _airy():
    # Added by Claude Opus 4.8 (missing entry flagged by Claude Fable 5
    # while translating the approx/Galleries example page).
    # MATLAB: chebfun(@airy, [-40 40]).
    import scipy.special as ssp

    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: jnp.array(ssp.airy(x)[0], dtype=jnp.float64),
                   domain=(-40.0, 40.0))


@_register("rose", "Rose curve cos(5t/4) e^{it} on [0, 8*pi] (trig)")
def _rose():
    # Added by Claude Opus 4.8 (missing entry flagged by Claude Fable 5).
    # MATLAB: fa = @(t) cos(m/n*t).*cos(t) + 1i*cos(m/n*t).*sin(t) with
    # m = 5, n = 4; chebfun(fa, [0, 8*pi], 'trig').
    from chebfunjax.chebfun1d.chebfun import chebfun
    m, n = 5.0, 4.0
    return chebfun(
        lambda t: jnp.cos(m / n * t) * jnp.exp(1j * t),
        domain=(0.0, 8.0 * jnp.pi),
        trig=True,
    )


@_register("motto", "exp(3i*scribble('there is no fun like chebfun'))")
def _motto():
    # Added by Claude Opus 4.8 (missing entry flagged by Claude Fable 5).
    # MATLAB: exp(3i*scribble('there is no fun like chebfun')).
    from chebfunjax.utils.scribble import scribble
    s = scribble("there is no fun like chebfun")
    return (3j * s).exp()


@_register("daubechies", "Daubechies db10 scaling function on [0, 19]")
def _daubechies():
    # Added by Claude Opus 4.8.  MATLAB: daubechies(10).
    import numpy as np

    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.domain import Domain
    from chebfunjax.utils.daubechies import scaling_function
    from chebfunjax.utils.quadrature import chebpts
    xs, phi = scaling_function(10, levels=7)
    a, b = float(xs[0]), float(xs[-1])
    # sample the (continuous) scaling function at Chebyshev-2 points
    cx = np.asarray(chebpts(4097, kind=2))
    phys = 0.5 * (b - a) * cx + 0.5 * (a + b)
    vals = np.interp(phys, xs, phi)
    return Chebfun.from_values(jnp.asarray(vals, dtype=jnp.float64),
                               Domain((a, b)))


@_register("gamma", "Gamma function on [-4, 4] with poles at 0, -1, ..., -4")
def _gamma():
    # Added by Claude Opus 4.8. MATLAB: chebfun(@gamma, [-4 4], 'blowup',
    # 'on', 'splitting', 'on'). Built as a piecewise Chebfun whose pieces
    # between the integer poles are Singfuns with (-1) endpoint exponents
    # (simple poles).
    import warnings as _warnings

    import numpy as np
    import scipy.special as _ssp

    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain
    from chebfunjax.fun.singfun import Singfun

    funs = []
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        for k in (-4, -3, -2, -1):        # poles at both integer ends
            def full(t, _k=k):
                x = _k + (jnp.asarray(t) + 1.0) / 2.0
                return jnp.asarray(_ssp.gamma(np.asarray(x)))
            s = Singfun.from_function(full, exponents=(-1.0, -1.0))
            funs.append(_Piece(tech=s, interval=(float(k), float(k + 1))))

        def full04(t):                    # pole at 0 only
            x = (jnp.asarray(t) + 1.0) * 2.0
            return jnp.asarray(_ssp.gamma(np.asarray(x)))
        s04 = Singfun.from_function(full04, exponents=(-1.0, 0.0))
        funs.append(_Piece(tech=s04, interval=(0.0, 4.0)))
    return Chebfun(funs=funs,
                   domain=Domain((-4.0, -3.0, -2.0, -1.0, 0.0, 4.0)))


@_register("blasius", "Blasius boundary-layer profile 2u'''+u u''=0 on [0,10]")
def _blasius():
    # Added by Claude Opus 4.8 (needed a chebop initial guess, N.init).
    # MATLAB: N = chebop(@(u) 2*diff(u,3)+u.*diff(u,2), [0 10]) with
    # u(0)=0, u'(0)=0, u'(10)=1 and a Chebyshev initial guess.
    from chebfunjax.chebfun1d.chebfun import chebfun
    from chebfunjax.operators.chebop import Chebop
    N = Chebop(lambda x, u: 2 * u.diff(3) + u * u.diff(2),
               domain=(0.0, 10.0))
    N.lbc = lambda u: [u, u.diff()]
    N.rbc = lambda u: u.diff() - 1.0
    N.init = chebfun(lambda x: x - 1.7 * (1 - jnp.exp(-x)),
                     domain=(0.0, 10.0))
    # n_max=256 gives the same profile as 512 (u'(10)=0.99992,
    # u''(0)=1.700) at ~3.5x less cost -- keeps CI within per-test
    # timeout budgets (Fable 5).
    return N.solve(0.0, n_max=256, max_iter=40)


@_register("vandermonde", "Quasimatrix of monomials 1, x, ..., x^5")
def _vandermonde():
    # Added by Claude Opus 4.8.  MATLAB: chebfun(@(x) x.^(0:5)).
    from chebfunjax.chebfun1d.chebfun import chebfun
    from chebfunjax.chebfun1d.linalg import Quasimatrix
    from chebfunjax.domain import Domain
    cols = [chebfun(lambda x, _k=k: x ** _k) for k in range(6)]
    return Quasimatrix(cols, Domain((-1.0, 1.0)))


@_register("vandercheb", "Quasimatrix of Chebyshev polys T_0, ..., T_5")
def _vandercheb():
    # Added by Claude Opus 4.8.  MATLAB: chebpoly(0:5).
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.chebfun1d.linalg import Quasimatrix
    from chebfunjax.domain import Domain
    cols = []
    for k in range(6):
        c = jnp.zeros(k + 1, dtype=jnp.float64).at[k].set(1.0)
        cols.append(Chebfun.from_coeffs(c, Domain((-1.0, 1.0))))
    return Quasimatrix(cols, Domain((-1.0, 1.0)))


@_register("random", "Interpolant through 100 random values at Cheb points")
def _random_interpolant():
    # Added by Claude Opus 4.8.  MATLAB: chebfun(rand(100,1)).
    import random as _r

    import jax

    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.domain import Domain
    key = jax.random.PRNGKey(_r.randint(0, 2 ** 31 - 1))
    vals = jax.random.uniform(key, (100,), dtype=jnp.float64)
    return Chebfun.from_values(vals, Domain((-1.0, 1.0)))


@_register("stegosaurus", "max(sin(x)+sin(x^2), x/10) on [0, 10]")
def _stegosaurus():
    # Added by Claude Opus 4.8 (needed two-arg max, now available).
    # MATLAB: chebfun(@(x) max(sin(x)+sin(x.^2), x/10), [0 10], 'splitting','on').
    from chebfunjax.chebfun1d.chebfun import chebfun
    f = chebfun(lambda x: jnp.sin(x) + jnp.sin(x ** 2), domain=(0.0, 10.0))
    g = chebfun(lambda x: x / 10.0, domain=(0.0, 10.0))
    return f.maximum(g)


@_register("jitter", "round(2*exp(x)*sin(8x)) on [-1, 1] — a step staircase")
def _jitter():
    # Added by Claude Opus 4.8 (needed round(), now available).
    # MATLAB: chebfun(@(x) round(exp(x)*2.*sin(8*x)), 'splitting','on').
    from chebfunjax.chebfun1d.chebfun import chebfun
    g = chebfun(lambda x: jnp.exp(x) * 2.0 * jnp.sin(8.0 * x))
    return g.round()


@_register("si", "Sine integral Si(x) = cumsum(sin(x)/x) on [-50, 50]")
def _si():
    # Added by Claude Opus 4.8 (missing gallery entry).
    # MATLAB: cumsum(chebfun(@(x) sin(x)./x, [-50, 50])).
    from chebfunjax.chebfun1d.chebfun import chebfun

    def sinc_like(x):
        # sin(x)/x with the removable singularity filled (limit 1 at 0)
        return jnp.where(jnp.abs(x) < 1e-300, 1.0,
                         jnp.sin(x) / jnp.where(x == 0.0, 1.0, x))
    return chebfun(sinc_like, domain=(-50.0, 50.0)).cumsum()


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


@_register("wild", "Iterated-map sum s = sum f_j, f_{j+1}=(3/4)(1-2 f_j^4)")
def _wild():
    """MATLAB +cheb/gallery.m 'wild': a wildly complicated function.

    Corrected by Claude Opus 4.8 — the previous entry
    (cos(x)^2 sin(x^3)) did not match MATLAB's ``wild`` subfunction,
    which sums 16 iterates of the map f -> (3/4)(1 - 2 f^4) starting
    from sin(pi x).  (The real one lives in [5.5, 9.5], not near 0.)
    """
    from chebfunjax.chebfun1d.chebfun import chebfun

    def wild(x):
        f = jnp.sin(jnp.pi * x)
        s = f
        for _ in range(15):
            f = 0.75 * (1.0 - 2.0 * f ** 4)
            s = s + f
        return s
    return chebfun(wild)


@_register("zigzag", "Degree 10000 polynomial that looks piecewise linear on [-1, 1]")
def _zigzag():
    """MATLAB: cumsum(chebfun(@(t) sign(sin(100*t./(2-t))), 10000)).

    From the ATAP appendix — integrating a square wave of increasing
    frequency gives a degree-10000 polynomial that looks piecewise linear.
    """
    from chebfunjax.chebfun1d.chebfun import chebfun
    square = chebfun(lambda t: jnp.sign(jnp.sin(100.0 * t / (2.0 - t))), n=10000)
    return square.cumsum()


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
