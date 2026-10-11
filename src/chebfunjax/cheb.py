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
    "x", "xy", "xyz", "xydisk", "xyzsphere",
    "bernoulli",
    "bspline",
    "gallery",
    "gallery2",
    "gallery3",
    "gallerydisk",
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


def gallery3(name: str | None = None, *, return_handle: bool = False):
    """Return a named Chebfun3 gallery function.

    ``return_handle=True`` also returns its defining callable, matching the
    adjacent gallery2 adapter for MATLAB's ``[F, FA]`` output. With no name,
    choose a random entry using Python's random stream. Constructors retain
    their defaults, explicit source domains, and the lattice trig flag.
    Python's returned-object API does not reproduce MATLAB's no-output plots.

    Provenance
    ----------
    MATLAB source : +cheb/gallery3.m, all 22 cases
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    import random

    from chebfunjax.chebfun3d.chebfun3 import chebfun3

    r = .6
    p = (jnp.sqrt(jnp.asarray(5.)) + 1) / 2

    def barth10(x, y, z):
        a = 1
        return (8*(x**2 - p**4*y**2)*(y**2 - p**4*z**2)
                * (z**2 - p**4*x**2)
                * (x**4 + y**4 + z**4 - 2*x**2*y**2
                   - 2*x**2*z**2 - 2*y**2*z**2)
                + a*(3 + 5*p)*(x**2 + y**2 + z**2 - a)**2
                * (x**2 + y**2 + z**2 - (2-p)*a)**2)

    def bessel(x, y, z):
        from chebfunjax.utils.besselj import besselj
        return besselj(0, 10*jnp.sqrt(x**2 + y**2 + z**2))

    def clebsch(x, y, z):
        return (81*(x**3 + y**3 + z**3)
                - 189*(x**2*y + x**2*z + y**2*x + y**2*z
                       + z**2*x + z**2*y)
                + 54*x*y*z + 126*(x*y + x*z + y*z)
                - 9*(x**2 + y**2 + z**2) - 9*(x + y + z) + 1)

    def levy(x, y, z):
        return (jnp.sin(jnp.pi*(1 + (x-1)/4))**2
                + ((z-1)/4)**2*(1 + jnp.sin(2*jnp.pi*(z-1)/4)**2)
                + ((x-1)/4)**2
                * (1 + 10*jnp.sin(jnp.pi*(1 + (x-1)/4) + 1)**2)
                + ((y-1)/4)**2
                * (1 + 10*jnp.sin(jnp.pi*(1 + (y-1)/4) + 1)**2))

    def shubert_factor(x):
        # The final source term is 5*cos(4*x+5), not the usual Shubert term.
        return (jnp.cos(2*x+1) + 2*jnp.cos(3*x+2) + 3*jnp.cos(4*x+3)
                + 4*jnp.cos(5*x+4) + 5*jnp.cos(4*x+5))

    def wagon(x, y, z):
        return (jnp.exp(jnp.sin(50*x))
                + jnp.sin(60*jnp.exp(y))*jnp.sin(60*z)
                + jnp.sin(70*jnp.sin(x))*jnp.cos(10*z)
                + jnp.sin(jnp.sin(80*y)) - jnp.sin(10*(x+z))
                + (x**2 + y**2 + z**2)/4)

    entries = {
        'aurentz': lambda x, y, t: jnp.exp(
            -((x-r*jnp.cos(jnp.pi*t))**2
              + (y-r*jnp.sin(jnp.pi*t))**2)/.1),
        'barth10': barth10,
        'bessel': bessel,
        'cassini': lambda x, y, z: ((x-.4)**2+y**2)*((x+.4)**2+y**2)-z**4,
        'clebsch': clebsch,
        'doublehelix': lambda x, y, z: jnp.abs(
            ((x+1j*y)*jnp.exp(1j*z))**2-1)**2,
        'genz1': lambda x, y, z: jnp.cos(jnp.pi+5*(x+y+z)),
        'genz2': lambda x, y, z: 1/(16+5*(x+y+z)),
        'genz3': lambda x, y, z: jnp.exp(-25*(x**2+y**2+z**2)),
        'genz4': lambda x, y, z: 1/((.04+x**2)*(.04+y**2)*(.04+z**2)),
        'griewank': lambda x, y, z: (
            x**2/4000+y**2/4000+z**2/4000
            - jnp.cos(x)*jnp.cos(y/jnp.sqrt(2))*jnp.cos(z/jnp.sqrt(3))+1),
        'hoop': lambda x, y, z: z**2+.5*(x**2+y**2-.5)**2,
        'kummer': lambda x, y, z: (
            x**4+y**4+z**4-.1*(x**2+y**2+z**2)
            - .5*(x**2*y**2+x**2*z**2+y**2*z**2)+.1*x*y*z-1),
        'lattice': lambda x, y, z: (
            jnp.cos(2*jnp.pi*x)**2+jnp.cos(2*jnp.pi*y)**2
            + jnp.cos(2*jnp.pi*z)**2),
        'levy': levy,
        'octant': lambda x, y, z: jnp.sqrt(x**2+y**2+z**2),
        'rastrigin': lambda x, y, z: (
            30+(x**2-10*jnp.cos(2*jnp.pi*x))
            + (y**2-10*jnp.cos(2*jnp.pi*y))
            + (z**2-10*jnp.cos(2*jnp.pi*z))),
        'rosenbrock': lambda x, y, z: (
            100*(y-x**2)**2+(x-1)**2+100*(z-y**2)**2+(y-1)**2),
        'runge': lambda x, y, z: 1/(1+x**2+y**2+z**2),
        'shubert': lambda x, y, z: (
            shubert_factor(x)*shubert_factor(y)*shubert_factor(z)),
        'toyfunction': lambda x, y, z: (
            3*x**7*z+y*z+y*z**2+jnp.log(2+y)*z**3-2*z**5),
        'wagon': wagon,
    }
    if name is None:
        name = random.choice(tuple(entries))
    key = str(name).lower()
    if key not in entries:
        raise ValueError('CHEB:GALLERY3:unknown:unknownFunction: Unknown function.')
    domains = {
        'doublehelix': (-1.5, 1.5, -1.5, 1.5, -6, 6),
        'griewank': (-50, 50, -50, 50, -50, 50),
        'levy': (-10, 10, -10, 10, -10, 10),
        'octant': (0, 1, 0, 1, 0, 1),
        'rosenbrock': (-2, 2, -2, 2, -2, 2),
    }
    kwargs = {}
    if key in domains:
        kwargs['domain'] = domains[key]
    if key == 'lattice':
        kwargs['trig'] = True
    fa = entries[key]
    f = chebfun3(fa, **kwargs)
    return (f, fa) if return_handle else f


def gallerydisk(name: str | None = None, *, return_handle: bool = False):
    """Return a native disk gallery entry, optionally with its defining handle.

    ``return_handle=True`` adapts MATLAB's two outputs to a Python tuple.
    Handles take Cartesian ``(x, y)`` or polar ``(theta, r)`` coordinates
    according to the native constructor route. For ``poisson``, the second
    output is the constructed Diskfun itself. No-argument selection uses
    Python's random stream over the native nine-entry selection list.

    Provenance
    ----------
    MATLAB source : +cheb/gallerydisk.m
    Chebfun commit: 7574c77
    """
    import random

    from chebfunjax.diskfun.diskfun import Diskfun, diskfun

    names = ("challenge", "flower", "poisson", "ellipsepeg", "roundpeg",
             "squarepeg", "tiltedpeg", "wave", "yinyang")
    if name is None:
        name = random.choice(names)
    key = name.lower()
    polar = False
    if key == "challenge":
        fa = lambda x, y: (jnp.exp(jnp.sin(50*x/3)) + jnp.sin(60*jnp.exp(y/3))  # noqa: E731
                          + jnp.sin(70*jnp.sin(x/3)) + jnp.sin(jnp.sin(80*y/3))
                          - jnp.sin(10*((x+y)/3)) + (x**2+y**2)/(4*3**2))
    elif key == "ellipsepeg":
        fa = lambda t, r: 1/(1+(4*((r*jnp.cos(t))**2+.5*(r*jnp.sin(t))**2))**20)  # noqa: E731
        polar = True
    elif key == "flower":
        fa = lambda t, r: jnp.sin(21*jnp.pi*(1+jnp.cos(jnp.pi*r))  # noqa: E731
                                *(r**2-2*r**5*jnp.cos(5*(t-.11))))
        polar = True
    elif key == "poisson":
        rhs = lambda t, r: (-jnp.exp(-40*(r**2-1)**4)  # noqa: E731
                           *jnp.sinh(5*(1-r**11*jnp.cos(11*(t-1/jnp.sqrt(2))))))
        bc = lambda t: 0*t+1  # noqa: E731
        f = Diskfun.poisson(rhs, bc, 512)
        return (f, f) if return_handle else f
    elif key == "roundpeg":
        fa = lambda t, r: 1/(1+(2*r)**100)  # noqa: E731
        polar = True
    elif key == "translatepeg":
        fa = lambda x, y: 1/(1+(2*((x-.2)**2+x**2)**20))  # noqa: E731
    elif key == "squarepeg":
        fa = lambda x, y: 1/((1+(2*x)**20)*(1+(2*y)**20))  # noqa: E731
    elif key == "tiltedpeg":
        fa = lambda x, y: 1/((1+(2*x+.4*y)**20)*(1+(2*y-.4*x)**20))  # noqa: E731
    elif key == "wave":
        fa = lambda x, y: jnp.cos(20*x+5*y)**2*(1-(x**2+y**2))  # noqa: E731
    elif key == "yinyang":
        fa = lambda t, r: -jnp.cos((jnp.sin(jnp.pi*r)*jnp.cos(t)  # noqa: E731
                                  +jnp.sin(2*jnp.pi*r)*jnp.sin(t))/4)
        polar = True
    else:
        raise ValueError("CHEB:GALLERYDISK:unknown:unknownFunction: Unknown function.")
    f = diskfun(fa, "polar") if polar else diskfun(fa)
    return (f, fa) if return_handle else f


def x():
    """Return the identity Chebfun on the default interval [-1, 1].

    Source: +cheb/x.m, Chebfun 7574c77. Python returns the object rather
    than assigning MATLAB's base-workspace variable.
    """
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: x)



def xy():
    """Return Cartesian coordinate Chebfun2s on the default square.

    Source: +cheb/xy.m, Chebfun 7574c77. Always returns an (x, y) tuple;
    MATLAB workspace assignment and requested-output truncation are not used.
    """
    from chebfunjax.chebfun2d.chebfun2 import chebfun2
    return chebfun2(lambda x, y: x), chebfun2(lambda x, y: y)



def xyz():
    """Return Cartesian coordinate Chebfun3s on the default cube.

    Source: +cheb/xyz.m, Chebfun 7574c77. Always returns an (x, y, z)
    tuple, replacing MATLAB workspace assignment (including its one-output
    source branch) with explicit Python return values.
    """
    from chebfunjax.chebfun3d.chebfun3 import chebfun3
    return (chebfun3(lambda x, y, z: x), chebfun3(lambda x, y, z: y),
            chebfun3(lambda x, y, z: z))



def xydisk():
    """Return Cartesian coordinate Diskfuns on the unit disk.

    Source: +cheb/xydisk.m, Chebfun 7574c77 (its declaration mistakenly
    names xyzsphere). The filename and two-coordinate body define this
    adapter. Always returns (x, y), without workspace assignment.
    Diskfun evaluation accepts (angle, radius); use feval_cart for (x, y).
    """
    from chebfunjax.diskfun.diskfun import diskfun
    return diskfun(lambda x, y: x), diskfun(lambda x, y: y)



def xyzsphere():
    """Return Cartesian coordinate Spherefuns on the unit sphere.

    Source: +cheb/xyzsphere.m, Chebfun 7574c77. Always returns an (x, y, z)
    tuple, without workspace assignment. Spherefun evaluation accepts
    (longitude, colatitude).
    """
    from chebfunjax.spherefun import spherefun
    return (spherefun(lambda x, y, z: x), spherefun(lambda x, y, z: y),
            spherefun(lambda x, y, z: z))
