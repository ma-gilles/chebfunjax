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


def poly(f):
    """Monomial coefficients of a (polynomial) chebfun, in DESCENDING
    powers — MATLAB poly(f).

    Provenance
    ----------
    MATLAB source : @chebfun/poly.m
    Chebfun commit: 7574c77
    """
    a = float(f.domain.a)
    b = float(f.domain.b)
    c = np.asarray(f.funs[0].tech.coeffs)
    # trim trailing (numerically zero) Chebyshev coefficients
    mag = np.abs(c)
    tol = 1e-13 * max(mag.max(), 1.0)
    nz = np.nonzero(mag > tol)[0]
    c = c[: nz[-1] + 1] if len(nz) else c[:1]
    T = np.polynomial.Chebyshev(c, domain=[a, b])
    p = T.convert(kind=np.polynomial.Polynomial)
    return p.coef[::-1]  # descending


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


def _gauss_nodes(n, a, b):
    from chebfunjax.utils.quadrature import legpts
    x, w = (np.asarray(t) for t in legpts(n))
    return (a + (b - a) * (x + 1) / 2.0,
            w * (b - a) / 2.0)


def fred(kernel, f, onevar: int | None = None):
    """Fredholm integral operator applied to a chebfun (MATLAB fred):
    F(x) = int_a^b K(x, y) f(y) dy.

    Provenance
    ----------
    MATLAB source : @chebfun/fred.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun, Domain
    a, b = float(f.domain.a), float(f.domain.b)
    t, w = _gauss_nodes(120, a, b)
    tj = jnp.asarray(t)
    wf = jnp.asarray(w) * f(tj)

    def F(x):
        return jnp.asarray(kernel(x[:, None], tj[None, :])) @ wf

    return Chebfun.from_function(F, Domain((a, b)))


def volt(kernel, f, onevar: int | None = None):
    """Volterra integral operator applied to a chebfun (MATLAB volt):
    F(x) = int_a^x K(x, y) f(y) dy.

    Provenance
    ----------
    MATLAB source : @chebfun/volt.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun, Domain
    from chebfunjax.utils.quadrature import legpts
    a, b = float(f.domain.a), float(f.domain.b)
    xi, w = (jnp.asarray(np.asarray(t)) for t in legpts(120))

    def F(x):
        # y_j(x) = a + (x - a)(xi_j + 1)/2, weight (x - a)/2 w_j
        half = (x[:, None] - a) / 2.0
        y = a + half * (xi[None, :] + 1.0)
        Kv = jnp.asarray(kernel(x[:, None], y)) * f(y)
        return jnp.sum(half * w[None, :] * Kv, axis=1)

    return Chebfun.from_function(F, Domain((a, b)))
