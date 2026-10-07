"""Random smooth functions as chebfuns (MATLAB ``randnfun``,
``smoothie``, ``randnfundisk``, ``randnfunsphere``).

These follow the MATLAB argument conventions: positional numbers are
``lambda`` (wavelength) then ``n`` (number of columns), a vector is the
domain, and the strings ``'big'``/``'norm'``, ``'trig'`` and
``'complex'`` are flags.  Random numbers come from NumPy's global
generator for the legacy non-randnfun helpers (``numpy.random.seed(k)``
plays the role of MATLAB's
``rng(k)``; the streams differ, so only statistics are reproducible).
The randnfun entry point now uses the shared JAX engine and key/seed API.

Provenance
----------
MATLAB source : randnfun.m, smoothie.m, randnfundisk.m, randnfunsphere.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np  # uses-numpy: host-side random coefficient generation

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.quadrature import chebpts_ab

__all__ = ["randnfun", "smoothie", "randnfundisk", "randnfunsphere"]


def _randn_matlab(rows: int, cols: int) -> np.ndarray:
    """``randn(rows, cols)`` drawn in MATLAB's column-major order."""
    return np.random.randn(cols, rows).T




def _to_chebfun_matrix(c, dom, trig):
    c = jnp.asarray(c)
    return chebfun(c[:, 0] if c.ndim == 2 and c.shape[1] == 1 else c,
                   domain=(dom[0], dom[-1]), trig=trig, coeffs=True)


def randnfun(*args, **kwargs):
    """Construct the source-shaped random Chebfun through the shared JAX engine.

    NumPy global seeding no longer controls this route. Use key= or seed= for
    explicit JAX reproducibility; default calls advance a private JAX stream.

    Provenance
    ----------
    MATLAB source : randnfun.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    from chebfunjax.utils._randnfun import randnfun as source_randnfun
    return source_randnfun(*args, **kwargs)


def _parse_smoothie(args):
    n, dom = None, None
    trig = cmplx = False
    for v in args:
        if isinstance(v, str):
            key = v.lower()
            if key[:1] == "t":
                trig = True
            elif key[:1] == "c":
                cmplx = True
            else:
                raise ValueError("CHEBFUN:smoothie: Unrecognized string "
                                 "input")
        elif np.ndim(v) > 0 and np.size(v) > 1:
            dom = [float(t) for t in np.ravel(np.asarray(v, dtype=float))]
        else:
            n = int(v)
    if n is None:
        n = 1
    if dom is None:
        dom = [-1.0, 1.0]
    return n, dom, trig, cmplx


def smoothie(*args):
    """Random smooth (C-infinity but not analytic) function(s) (MATLAB
    ``smoothie``): a Fourier series with root-exponentially decaying
    random coefficients, restricted from a periodic function on a 20%
    larger interval in the non-periodic case.  Flags ``'trig'`` and
    ``'complex'``; a number is the column count and a vector the domain.

    Provenance
    ----------
    MATLAB source : smoothie.m
    Chebfun commit: 7574c77
    """
    n, dom, trig, cmplx = _parse_smoothie(args)
    L = dom[-1] - dom[0]
    m = int(round(math.ceil(2000 * L))) + 1
    if cmplx:
        flags = ["trig"] if trig else []
        return (smoothie(n, dom, *flags)
                + 1j * smoothie(n, dom, *flags)) / math.sqrt(2)
    if not trig:
        dom2 = [dom[0], dom[0] + 1.2 * L]
        f = smoothie(n, dom2, "trig")
        x = jnp.asarray(chebpts_ab(int(round(2.5 * m)) + 20, dom[0],
                                   dom[-1]))
        g = chebfun(jnp.asarray(f(x)), domain=(dom[0], dom[-1]))
        return g.simplify()
    c = _randn_matlab(m, n) + 1j * _randn_matlab(m, n)
    c[0, :] = math.sqrt(2) * np.real(c[0, :])
    decay = np.exp(-np.sqrt(np.arange(1, m + 1) / L))[:, None]
    c = (decay * c) / math.sqrt(L)
    c = np.concatenate([np.conj(c[:0:-1, :]), c], axis=0)
    return _to_chebfun_matrix(c, dom, True)


def randnfundisk(lam: float = 1.0):
    """Random smooth function on the unit disk (MATLAB ``randnfundisk``):
    a random band-limited function on the enclosing square, restricted
    to the disk as a Diskfun.

    Provenance
    ----------
    MATLAB source : randnfundisk.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.diskfun.diskfun import Diskfun
    from chebfunjax.utils.random import randnfun2
    box = (-1.25, 1.25, -1.25, 1.25)
    fsquare = randnfun2(float(lam), box, seed=int(np.random.randint(2**31)),
                        trig=lam <= 1)

    def _polar(t, r):
        return fsquare(r * jnp.cos(t), r * jnp.sin(t))
    return Diskfun.from_function(_polar)


def randnfunsphere(lam: float = 1.0, type_: str | None = None):
    """Random smooth function on the unit sphere (MATLAB
    ``randnfunsphere``): a random combination of spherical harmonics up
    to degree ``floor(2*pi/lambda)``, or of that single degree with
    ``'monochromatic'``.

    Provenance
    ----------
    MATLAB source : randnfunsphere.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.spherefun.spherefun import Spherefun
    from chebfunjax.utils.random import _sph_harm_sum, _sph_harm_sum_fixed_deg
    if isinstance(lam, str):
        type_, lam = lam, 1.0
    mono = type_ is not None and str(type_).lower().startswith("m")
    deg = int(math.floor(2 * math.pi / float(lam)))
    if mono:
        c = np.random.randn(2 * deg + 1)
        c = math.sqrt(4 * math.pi / np.count_nonzero(c)) * c
        _sum = _sph_harm_sum_fixed_deg
    else:
        c = np.random.randn((deg + 1) ** 2)
        c = math.sqrt(4 * math.pi / np.count_nonzero(c)) * c
        _sum = _sph_harm_sum

    def _op(ll, tt):
        # The constructor samples on a (theta, lambda) meshgrid; the
        # harmonic sum takes the 1-D axes and returns (n_theta, n_lam).
        ll = np.asarray(ll)
        tt = np.asarray(tt)
        if ll.ndim == 2:
            lam1 = ll[0, :]
            th1 = tt[:, 0]
            return jnp.asarray(_sum(lam1, th1, deg, np.asarray(c)))
        return jnp.asarray(np.diag(_sum(ll.ravel(), tt.ravel(), deg,
                                        np.asarray(c))))
    f = Spherefun.from_function(_op)
    return f.simplify() if hasattr(f, "simplify") else f
