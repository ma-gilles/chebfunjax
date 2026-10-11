# uses-numpy: polynomial coefficient manipulation is one-shot numpy/scipy
"""Polynomial residue and integral-operator utilities (MATLAB residue,
fred, volt).

Added by Claude Fable 5 (MISSING_FEATURES named-utilities sweep).

Provenance
----------
MATLAB source : @chebfun/residue.m, @chebfun/fred.m, @chebfun/volt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.signal import invres
from scipy.signal import residue as _sresidue

__all__ = ["poly", "residue", "fred", "volt"]


def poly(f, n=None):
    """Descending polynomial coefficients of the selected FUN (one-based).

    Provenance
    ----------
    MATLAB source: @chebfun/poly.m, @bndfun/poly.m, @chebtech/poly.m.
    Chebfun commit: 7574c77.
    Scalar output uses a Python vector; array-valued output retains rows.
    """
    import warnings

    from jax.scipy.special import gammaln

    from chebfunjax.fun.singfun import Singfun
    from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

    if n is None:
        if len(f.funs) > 1:
            warnings.warn('F has more than one FUN. Only the polynomial '
                          'coefficients of the first FUN are returned.',
                          UserWarning, stacklevel=2)
        n = 1
    index = jnp.asarray(n)
    if index.ndim != 0 or not bool(jnp.isfinite(index)) or float(index) != int(index) or int(index) < 1:
        raise ValueError('N should be a positive scalar integer.')
    if int(index) > len(f.funs):
        raise ValueError(f'Chebfun only has {len(f.funs)} FUNS')
    piece = f.funs[int(index)-1]
    a, b = piece.interval
    if not bool(jnp.isfinite(a) & jnp.isfinite(b)):
        raise ValueError('POLY does not support UNBNDFUN objects.')
    tech = piece.tech
    if isinstance(tech, Singfun):
        raise ValueError('POLY does not support SINGFUN objects.')
    out = tech.poly()
    # Native constant array-valued CHEBTECH returns its coefficient row.
    if isinstance(tech, (Chebtech1, Chebtech2)) and tech.coeffs.ndim == 2 and tech.n == 1:
        out = tech.coeffs
    scalar = out.ndim == 1
    out = jnp.atleast_2d(out)
    if a != -1.0 or b != 1.0:
        alpha, beta = 2/(b-a), -(b+a)/(b-a)
        out = out[:, ::-1]
        for j in range(out.shape[1]):
            k = jnp.arange(j, out.shape[1])
            binom = jnp.round(jnp.exp(gammaln(k+1.)-gammaln(k-j+1.)-gammaln(j+1.)))
            bba = binom*beta**(k-j)*alpha**j
            out = out.at[:, j].set(jnp.sum(out[:, j:]*bba, axis=1))
        out = out[:, ::-1]
    return out[0] if scalar else out


def _poly_chebfun(coeffs_desc, domain=(-1.0, 1.0)):
    from chebfunjax.chebfun1d.chebfun import Chebfun, Domain
    cd = np.atleast_1d(np.asarray(coeffs_desc))
    if cd.size == 0:
        cd = np.zeros(1)
    if np.iscomplexobj(cd):
        return Chebfun.from_function(
            lambda x: jnp.polyval(jnp.asarray(cd), x),
            Domain((float(domain[0]), float(domain[1]))))
    return Chebfun.from_function(
        lambda x: jnp.polyval(jnp.asarray(cd, dtype=jnp.float64), x),
        Domain((float(domain[0]), float(domain[1]))))


def residue(u, v, k=None):
    """Partial-fraction expansion (MATLAB residue, both directions).

    ``residue(g, f)`` with chebfun inputs returns ``(r, p, k)``:
    residues, poles, and the polynomial quotient as a chebfun, such
    that g/f = sum r_j/(x - p_j) + k(x).

    ``residue(r, p, k)`` with vector inputs (k a chebfun or None)
    returns ``(B, A)``: numerator/denominator chebfuns.

    Provenance
    ----------
    MATLAB source : @chebfun/residue.m
    Chebfun commit: 7574c77
    """
    if hasattr(u, "funs"):  # forward: chebfun numerator/denominator
        g, f = u, v
        bb = np.asarray(poly(g))
        aa = np.asarray(poly(f))
        r, p, kk = _sresidue(bb, aa)
        dom = (float(f.domain.a), float(f.domain.b))
        return jnp.asarray(r), jnp.asarray(p), _poly_chebfun(kk, dom)
    # inverse: vectors (+ optional chebfun quotient)
    r = np.atleast_1d(np.asarray(u))
    p = np.atleast_1d(np.asarray(v))
    if k is None or (hasattr(k, "funs") and len(k.funs) == 0):
        kk = np.array([])
        dom = (-1.0, 1.0)
    elif hasattr(k, "funs"):
        kk = np.asarray(poly(k))
        dom = (float(k.domain.a), float(k.domain.b))
    else:
        kk = np.atleast_1d(np.asarray(k))
        dom = (-1.0, 1.0)
    bb, aa = invres(r, p, kk)
    B = _poly_chebfun(np.real_if_close(bb), dom)
    A = _poly_chebfun(np.real_if_close(aa), dom)
    return B, A


def fred(kernel, f, onevar=None):
    """Apply the shared JAX Fredholm integral action or its AD counterpart.

    Provenance
    ----------
    MATLAB source: @chebfun/fred.m, @adchebfun/adchebfun.m (fred).
    Chebfun commit: 7574c77
    """
    from chebfunjax.operators.integral import fred as _fred
    return _fred(kernel, f, onevar)


def volt(kernel, f, onevar=None):
    """Apply the shared JAX Volterra integral action or its AD counterpart.

    Provenance
    ----------
    MATLAB source: @chebfun/volt.m, @adchebfun/adchebfun.m (volt).
    Chebfun commit: 7574c77
    """
    from chebfunjax.operators.integral import volt as _volt
    return _volt(kernel, f, onevar)
