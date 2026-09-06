"""The ``cheb`` namespace: gallery functions and small utilities.

MATLAB Chebfun ships a ``+cheb`` package (``cheb.gallery``, ``cheb.x``,
``cheb.bernoulli``, ...).  This module collects the chebfunjax ports so
that ``from chebfunjax import cheb; cheb.bernoulli(4)`` reads like the
original.

Provenance
----------
MATLAB source : +cheb/bernoulli.m, +cheb/bspline.m, +cheb/galleryball.m,
    +cheb/normal2.m, +cheb/revolution.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from chebfunjax.utils.gallery import gallery
from chebfunjax.utils.gallery2 import gallery2
from chebfunjax.utils.gallerytrig import gallerytrig

__all__ = [
    "bernoulli",
    "bspline",
    "gallery",
    "gallery2",
    "gallerytrig",
    "galleryball",
    "normal2",
    "revolution",
]


def bernoulli(N: int) -> list:
    """Bernoulli polynomials ``B_0, ..., B_N`` on ``[0, 1]``.

    Built by the recurrence ``B_{j} = j * cumsum(B_{j-1})`` with each
    polynomial shifted to have zero mean, exactly as MATLAB does; the
    columns of MATLAB's quasimatrix are returned as a list of Chebfuns.

    Provenance
    ----------
    MATLAB source : +cheb/bernoulli.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.chebfun import chebfun

    x = chebfun(lambda t: t, domain=(0.0, 1.0))
    B = [0.0 * x + 1.0]
    for j in range(1, int(N) + 1):
        Bj = float(j) * B[j - 1].cumsum()
        Bj = Bj - float(Bj.sum())
        B.append(Bj)
    return B


def bspline(m: int) -> list:
    """The first ``m`` cardinal B-splines ``B_1, ..., B_m``.

    ``B_1`` is the indicator of ``[-1/2, 1/2]`` and ``B_{k+1} = B_k * B_1``
    (convolution), so ``B_k`` is a piecewise polynomial of degree ``k-1``
    supported on ``[-k/2, k/2]``.  Returned as a list of Chebfuns (MATLAB
    returns a chebmatrix).

    Provenance
    ----------
    MATLAB source : +cheb/bspline.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.chebfun import chebfun

    s = chebfun(lambda t: 0.0 * t + 1.0, domain=(-0.5, 0.5))
    B = [s]
    for _k in range(1, int(m)):
        B.append(B[-1].conv(s))
    return B


def revolution(f):
    """Geometric properties of the surface of revolution of ``f``.

    Revolving the graph of ``f >= 0`` on ``[a, b]`` about the z-axis
    gives a solid whose surface area, volume, centroid (z coordinate) and
    moment of inertia about the axis are::

        SA = 2*pi*sum(f*sqrt(1 + f'^2)),   V  = pi*sum(f^2),
        zC = pi/V*sum(z*f^2),               I  = pi/2*sum(f^4).

    Returns a dict with keys ``surfaceArea``, ``volume``, ``centroidZ``
    and ``momentOfInertia`` (MATLAB's single-output struct).

    Provenance
    ----------
    MATLAB source : +cheb/revolution.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.chebfun import chebfun

    _xmin, fmin = f.min()
    if float(fmin) < -1e-15:
        raise ValueError(
            "Radius of the revolved surface must be nonnegative.")
    bp = f.domain.breakpoints
    a = float(bp[0])
    b = float(bp[-1])
    z = chebfun(lambda t: t, domain=(a, b))
    SA = 2.0 * math.pi * float((f * (1.0 + abs(f.diff()) ** 2).sqrt()).sum())
    V = math.pi * float((f ** 2).sum())
    zC = math.pi / V * float((z * f ** 2).sum())
    Iz = math.pi / 2.0 * float((f ** 4).sum())
    return {
        "surfaceArea": SA,
        "volume": V,
        "centroidZ": zC,
        "momentOfInertia": Iz,
    }


def normal2(mu, Sigma, dom=None):
    """Bivariate normal density as a Chebfun2.

    ``p(x, y)`` with mean ``mu`` and covariance ``Sigma`` (which must be
    symmetric positive definite).  The default domain is
    ``mu +/- 4*sigma_max`` in each variable, ``sigma_max`` the largest
    singular value of ``Sigma``.

    Provenance
    ----------
    MATLAB source : +cheb/normal2.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun2d.chebfun2 import chebfun2

    S = jnp.asarray(Sigma, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64).reshape(-1)
    if (bool(jnp.any(jnp.linalg.eigvals(S).real < 0))
            or float(jnp.linalg.norm(S - S.T)) > 0):
        raise ValueError(
            "CHEB:NORMAL2:covariance:nonSymPosDef: "
            "Covariance matrix must be symmetric positive definite.")
    if dom is None:
        sig = float(jnp.linalg.svd(S, compute_uv=False)[0])
        dom = (float(mu[0]) - 4 * sig, float(mu[0]) + 4 * sig,
               float(mu[1]) - 4 * sig, float(mu[1]) + 4 * sig)
    invSig = jnp.linalg.inv(S)
    a11 = float(invSig[0, 0])
    a12 = float(invSig[0, 1] + invSig[1, 0])
    a22 = float(invSig[1, 1])
    const = 2.0 * math.pi * math.sqrt(float(jnp.linalg.det(S)))
    m0, m1 = float(mu[0]), float(mu[1])

    def _p(x, y):
        xx = x - m0
        yy = y - m1
        z = a11 * xx ** 2 + a12 * xx * yy + a22 * yy ** 2
        return jnp.exp(-z / 2.0) / const

    return chebfun2(_p, domain=tuple(float(d) for d in dom))


_BALL_NAMES = ("deathstar", "gaussian", "helmholtz", "moire", "peaks",
               "roundpeg", "solharm", "stripes", "wave")


def galleryball(name: str | None = None):
    """Gallery of Ballfun examples: ``(f, fa)`` for the named function.

    ``f`` is the Ballfun and ``fa`` the anonymous function it was built
    from (a Ballfun itself for ``'helmholtz'`` and ``'solharm'``).  With
    no name a random example is chosen, as in MATLAB.

    Provenance
    ----------
    MATLAB source : +cheb/galleryball.m
    Chebfun commit: 7574c77
    """
    import random as _random

    from jax.scipy.special import bessel_jn

    from chebfunjax.ballfun import Ballfun
    from chebfunjax.utils.specfun import besselroots

    if name is None:
        name = _random.choice(_BALL_NAMES)
    key = str(name).lower()
    if key == "deathstar":
        def fa(x, y, z):
            return -(jnp.exp(-30 * ((y + math.sqrt(3) / 2) ** 2 + x ** 2
                                    + (z - 0.5) ** 2))
                     + jnp.exp(-25 * z ** 2))
        f = Ballfun.from_function(fa)
    elif key == "gaussian":
        def fa(x, y, z):
            return jnp.exp(-20 * ((x + 0.5) ** 2 + y ** 2 + z ** 2))
        f = Ballfun.from_function(fa)
    elif key == "helmholtz":
        u = Ballfun.from_function(lambda x, y, z: -80 * jnp.sin(10 * x))

        def bc(lam, th):
            return (10 * jnp.cos(lam) * jnp.sin(th)
                    * jnp.cos(10 * jnp.cos(lam) * jnp.sin(th)))
        f = Ballfun.helmholtz(u, math.sqrt(20), bc, 50, 50, 50,
                              bc_type="neumann")
        fa = f
    elif key == "moire":
        boise = (-116.237651 * math.pi / 180, 43.613739 * math.pi / 180)
        oxford = (-1.257778 * math.pi / 180, 51.751944 * math.pi / 180)

        def _sph2cart(az, el):
            return (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
                    math.sin(el))
        xb, yb, zb = _sph2cart(*boise)
        xo, yo, zo = _sph2cart(*oxford)
        omega = float(besselroots(0, 30)[-1]) / 2.0

        def _j0(r):
            return bessel_jn(r, v=0)[0]

        def fa(x, y, z, omega=omega):
            rb = jnp.sqrt((x - xb) ** 2 + (y - yb) ** 2 + (z - zb) ** 2)
            ro = jnp.sqrt((x - xo) ** 2 + (y - yo) ** 2 + (z - zo) ** 2)
            return 2 + _j0(omega * rb) + 2 + _j0(omega * ro)
        f = Ballfun.from_function(lambda x, y, z: fa(x, y, z, omega))
    elif key == "peaks":
        def fa(x, y, z):
            return (8 * (1 - x) ** 2 * jnp.exp(-4 * (x - 0.059) ** 2
                                               - 2 * (y + 0.337) ** 2
                                               - 2 * (z + 0.940) ** 2)
                    - 30 * (z / 10 - x ** 3 - y ** 5)
                    * jnp.exp(-3 * (x - 0.250) ** 2 - 2 * (y - 0.433) ** 2
                              - 3 * (z - 0.866) ** 2)
                    + (20 * y - 8 * z ** 3)
                    * jnp.exp(-2 * (x + 0.696) ** 2 - 3 * (y + 0.123) ** 2
                              - 2 * (z - 0.707) ** 2)
                    + (7 * y - 10 * x + 10 * z ** 3)
                    * jnp.exp(-3 * (x - 0.296) ** 2 - 3 * (y + 0.814) ** 2
                              - 3 * (z + 0.5) ** 2))
        f = Ballfun.from_function(fa)
    elif key == "roundpeg":
        def fa(r, l, t):
            return 1.0 / (1.0 + (2 * r) ** 100)
        f = Ballfun.from_function(fa, spherical=True)
    elif key == "solharm":
        fa = Ballfun.solharm(5, 3)
        f = fa
    elif key == "stripes":
        def fa(x, y, z):
            return jnp.sin(50 * z) - x ** 2
        f = Ballfun.from_function(fa)
    elif key == "wave":
        def fa(x, y, z):
            return (jnp.cos(10 * x + 5 * y) ** 2
                    * (1 - (x ** 2 + y ** 2 + z ** 2)))
        f = Ballfun.from_function(fa)
    else:
        raise ValueError(
            "CHEB:GALLERYBALL:unknown:unknownFunction: Unknown function.")
    return f, fa
