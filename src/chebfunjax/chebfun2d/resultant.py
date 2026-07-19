# uses-numpy: the hidden-variable Bezout resultant needs dense numpy/scipy
# linear algebra (generalized eigenproblems, colleague matrices) that JAX does
# not provide on CPU; this is a one-shot rootfinding backend, not a hot path.
"""Bezout / hidden-variable resultant rootfinder for Chebfun2 common zeros.

This is the alternative backend to the marching-squares
:func:`chebfunjax.chebfun2d.zerocurves.common_zeros` engine.  It ports the
resultant path of MATLAB ``@chebfun2v/roots.m`` (the
Nakatsukasa--Noferini--Townsend Bezout-resultant method):

1. Both Chebfun2 are subdivided over the domain until, on each subrectangle,
   the tensor-product Chebyshev degrees of ``f`` and ``g`` drop below a small
   threshold (``max_degree`` = 16).
2. On a small subrectangle the hidden-variable ``y`` is eliminated: the
   Bezoutian resultant matrix ``B(y)`` of the two univariate ``x``-polynomials
   ``f(.,y)``, ``g(.,y)`` is sampled at Chebyshev ``y``-nodes, assembled into a
   matrix polynomial in the Chebyshev basis, and its polynomial eigenvalue
   problem is solved by the first-kind colleague matrix *pencil* (a regularized
   QZ).  The real eigenvalues in ``[-1, 1]`` are the candidate ``y``-values.
3. For each candidate ``y`` the two univariate ``x``-polynomials are root-found
   by the colleague matrix, and only the ``x`` shared by both survives.
4. The ballpark ``(x, y)`` from all subrectangles are polished by a batched 2D
   Newton with the exact Jacobian, filtered to the domain, and de-duplicated.

The end result is the isolated common zeros as an ``(m, 2)`` array of ``[x, y]``
points, matching the marching-squares engine to Newton accuracy.

Provenance
----------
MATLAB source : @chebfun2v/roots.m (roots_resultant and its private
    functions subrootsreptwo, runbezval, formBez, DLPforbez, matrixChebfft,
    balancecong, chebT1rtsmatgep, xunivariate, onevar, vandercheb)
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
Algorithm:
    [1] Y. Nakatsukasa, V. Noferini, and A. Townsend, "Computing the common
        zeros of two bivariate functions via Bezout resultants",
        Numer. Math. 129 (2015).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla

__all__ = ["resultant_common_zeros"]

_EPS = float(np.finfo(np.float64).eps)
_MAGIC = 0.004849834917525       # 'magic number' (off-centre split point)
_HONOUR = -0.0005194318842611    # 'honourary number' (second split point)


# ----------------------------------------------------------------------
# Chebyshev sampling / transforms (self-consistent, native chebtech2 order)
# ----------------------------------------------------------------------
def _chebpts_desc(n: int) -> np.ndarray:
    """``n`` second-kind Chebyshev points, descending ``+1 -> -1``.

    Native chebtech2 ordering: ``x_k = cos(pi k / (n-1))`` for ``k = 0..n-1``.
    """
    if n == 1:
        return np.array([0.0])
    return np.cos(np.pi * np.arange(n) / (n - 1))


def _vals2coeffs(v: np.ndarray) -> np.ndarray:
    """Values at ``_chebpts_desc(n)`` to Chebyshev-``T`` coefficients (ascending).

    Operates column-wise on 2D input.  Mirrors ``@chebtech2/vals2coeffs.m``.
    """
    v = np.asarray(v)
    n = v.shape[0]
    if n <= 1:
        return v
    # Samples come from _chebpts_desc (descending +1 -> -1); the chebtech2
    # transform below assumes ascending order, so reverse first.
    v = v[::-1]
    tmp = np.concatenate([v[n - 1:0:-1], v[:n - 1]], axis=0)
    if np.isrealobj(v):
        c = np.real(np.fft.ifft(tmp, axis=0))
    else:
        c = np.fft.ifft(tmp, axis=0)
    c = c[:n]
    c[1:n - 1] = 2.0 * c[1:n - 1]
    return c


def _cheb2(f, xmin, xmax, ymin, ymax, tol, npts):
    """Nonadaptive tensor-product Chebyshev sampling of ``f`` on a subrectangle.

    Returns the truncated coefficient matrix ``F`` with rows indexed by
    ``y``-degree *descending* and columns by ``x``-degree *descending* (the
    orientation MATLAB ``cheb2`` produces with ``rot90(C, -2)``): ``F[-1, -1]``
    multiplies ``T_0(y) T_0(x)``.
    """
    xs = 0.5 * (xmin + xmax) + 0.5 * (xmax - xmin) * _chebpts_desc(npts)
    ys = 0.5 * (ymin + ymax) + 0.5 * (ymax - ymin) * _chebpts_desc(npts)
    xx, yy = np.meshgrid(xs, ys)   # shape (npts_y, npts_x)
    vals = np.asarray(
        f(jnp.asarray(xx), jnp.asarray(yy)), dtype=np.float64)
    vscl = max(1.0, float(np.max(np.abs(vals))))
    # 2D coefficients, ascending in both x (axis=1) and y (axis=0).
    c = _vals2coeffs(_vals2coeffs(vals.T).T)   # x then y
    # Trim negligible high-degree rows/cols.
    keep_row = np.where(np.max(np.abs(c), axis=1) > tol * vscl)[0]
    keep_col = np.where(np.max(np.abs(c), axis=0) > tol * vscl)[0]
    if keep_row.size == 0 or keep_col.size == 0:
        return np.zeros((1, 1)), vscl
    c = c[:keep_row[-1] + 1, :keep_col[-1] + 1]
    # Flip to MATLAB descending orientation (highest degree first).
    return c[::-1, ::-1].copy(), vscl


# ----------------------------------------------------------------------
# Bezoutian construction (Chebyshev basis)
# ----------------------------------------------------------------------
def _vandercheb(x: float, n: int) -> np.ndarray:
    """Chebyshev-vandermonde vector ``[T_{n-1}(x), ..., T_1(x), T_0(x)]``."""
    v = np.zeros(n)
    t = np.arccos(complex(x)) if abs(x) > 1 else np.arccos(x)
    for i in range(1, n + 1):
        v[n - i] = np.real(np.cos((i - 1) * t))
    return v


def _dlp_for_bez(aa: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Chebyshev-basis Bezoutian via the diagonal-Lyapunov recurrence.

    ``aa`` is a length-``m`` row of Chebyshev-``x`` coefficients (descending)
    and ``v`` the length-``m-1`` companion vector; returns the ``k x k``
    Bezout matrix (``k = m - 1``).  Ports MATLAB ``DLPforbez``.
    """
    aa = np.atleast_2d(np.asarray(aa, dtype=np.float64))   # (1, m)
    m = aa.shape[1]
    k = m - 1
    v = np.asarray(v, dtype=np.float64).reshape(-1, 1)     # (k, 1)
    s_mat = np.vstack([np.zeros((1, k + 1)), (2.0 * v) @ aa])   # (k+1, k+1)
    r = s_mat.T - s_mat
    if k == 1:
        return np.array([[r[0, 1]]])
    y = np.zeros((k, k))
    y[0, :] = r[0, 1:]
    y[1, :] = (r[1, 1:]
               + np.concatenate([y[0, 1:k - 1], [2.0 * y[0, k - 1]], [0.0]])
               + np.concatenate([[0.0], y[0, :k - 1]]))
    for i in range(2, k):
        y[i, :] = (r[i, 1:] - y[i - 2, :]
                   + np.concatenate([y[i - 1, 1:k - 1],
                                     [2.0 * y[i - 1, k - 1]], [0.0]])
                   + np.concatenate([[0.0], y[i - 1, :k - 1]]))
    y[k - 1, :] = y[k - 1, :] / 2.0
    return y


def _form_bez(F: np.ndarray, G: np.ndarray, y: float) -> np.ndarray:
    """Bezoutian resultant matrix of ``f(.,y)``, ``g(.,y)`` (ports ``formBez``)."""
    nf, ng = F.shape[0], G.shape[0]
    yv = _vandercheb(y, max(nf, ng))
    ff = yv[len(yv) - nf:] @ F           # (mf,)
    gg = yv[len(yv) - ng:] @ G           # (mg,)
    ff = np.atleast_1d(ff)
    gg = np.atleast_1d(gg)
    if len(ff) < len(gg):
        ff, gg = gg, ff
    if len(ff) == len(gg):
        ff = np.concatenate([[0.0], ff])
        shrink = True
    else:
        shrink = False
    v = np.concatenate([np.zeros(len(ff) - len(gg) - 1), gg])
    b = _dlp_for_bez(ff, v)
    if shrink:
        b = b[1:, 1:]
    return b


def _balance_cong(B: np.ndarray) -> np.ndarray:
    """Diagonal congruence balancing (ports ``balancecong``)."""
    n = B.shape[0]
    d = np.ones(n)
    last = np.linalg.norm(B[-1, :])
    for i in range(n - 2, -1, -1):
        ni = np.linalg.norm(B[i, :])
        if ni > 0:
            d[i] = max(1.0, np.sqrt(last / ni))
        else:
            d[i] = d[i + 1]
    return np.diag(d)


def _matrix_cheb_coeffs(bvals: np.ndarray) -> np.ndarray:
    """Convert matrix *values* sampled at ``_chebpts_desc(mc)`` (3rd axis) to
    matrix Chebyshev *coefficients* (ports ``matrixChebfft``).

    ``bvals`` has shape ``(n, n, mc)``; returns the same shape holding the
    Chebyshev-``y`` coefficients (ascending) of each entry.
    """
    n, _, mc = bvals.shape
    flat = bvals.reshape(n * n, mc).T          # (mc, n*n)
    coeffs = _vals2coeffs(flat)                # (mc, n*n)
    return coeffs.T.reshape(n, n, mc)


# ----------------------------------------------------------------------
# Polynomial rootfinders (colleague matrix / pencil)
# ----------------------------------------------------------------------
def _chebT1rts(c_desc: np.ndarray) -> np.ndarray:
    """Real roots in ``[-1, 1]`` of a Chebyshev-``T`` poly with coefficients
    ``c_desc`` ordered highest degree first.  Uses the shared colleague engine.
    """
    from chebfunjax.tech.chebtech import _roots_colleague
    c_desc = np.asarray(c_desc, dtype=np.float64).ravel()
    nz = np.where(np.abs(c_desc) > 0)[0]
    if nz.size == 0:
        return np.array([])
    c_desc = c_desc[nz[0]:]
    if c_desc.size <= 1:
        return np.array([])
    r = np.asarray(_roots_colleague(jnp.asarray(c_desc[::-1])))
    return np.sort(r)


def _chebT1rts_matgep(c: np.ndarray) -> np.ndarray:
    """Roots of a matrix polynomial in the first-kind Chebyshev-``T`` basis via
    the colleague matrix *pencil* (ports ``chebT1rtsmatgep``).

    ``c`` has shape ``(n, n, k)``: ``k-1`` is the degree (index 0 = highest
    degree), ``n`` the matrix size.  Returns the (complex) eigenvalues.
    """
    c = np.array(c, dtype=np.float64)
    n = c.shape[0]
    k = c.shape[2]
    if k == 2:                       # linear pencil
        return sla.eig(c[:, :, 1], -c[:, :, 0], right=False)
    for ii in range(1, k):
        c[:, :, ii] = c[:, :, ii] * (-0.5)
    c[:, :, 2] = c[:, :, 2] + 0.5 * c[:, :, 0]
    oh = 0.5 * np.ones(n * (k - 2))
    A = np.diag(oh, n) + np.diag(oh, -n)
    A[-n:, -2 * n:-n] = np.eye(n)
    for ii in range(k - 1):
        A[:n, ii * n:(ii + 1) * n] = c[:, :, ii + 1]
    B = np.eye(A.shape[0])
    B[:n, :n] = c[:, :, 0]
    return sla.eig(A, B, right=False)


# ----------------------------------------------------------------------
# Univariate x-recovery from candidate y-values
# ----------------------------------------------------------------------
def _xunivariate(F, G, xmin, xmax, ymin, ymax, y, doswap):
    """Recover ``x`` for each candidate ``y`` (ports ``xunivariate``)."""
    ep = 10 * _EPS
    y = np.sort(np.asarray(y, dtype=np.float64))
    if y.size == 0:
        return np.array([]), np.array([])
    dist = np.abs(y - np.concatenate([y[1:], [2.0]]))
    y = y[dist > ep]

    xtmp, ytmp = [], []
    for yj in y:
        vy = _vandercheb(yj, G.shape[0])
        coef1 = vy @ G                       # g(.,yj) x-coeffs (descending)
        vy = _vandercheb(yj, F.shape[0])
        coef2 = vy @ F                       # f(.,yj) x-coeffs (descending)
        coef1 = np.atleast_1d(coef1)
        coef2 = np.atleast_1d(coef2)

        rts1 = _chebT1rts(coef1) if (coef1.size > 1
                                     and np.linalg.norm(coef1) != 0) \
            else np.array([])
        rts2 = _chebT1rts(coef2) if (coef2.size > 1
                                     and np.linalg.norm(coef2) != 0) \
            else np.array([])
        rts = np.sort(np.concatenate([rts1, rts2]))
        if rts.size == 0:
            continue
        # residual of each candidate x in BOTH polynomials.
        res = np.array([
            max(abs(coef1 @ _vandercheb(rx, G.shape[1])),
                abs(coef2 @ _vandercheb(rx, F.shape[1])))
            for rx in rts])
        if xmax - xmin > 2e-3:
            rtscan = rts[res < 100 * np.sqrt(_EPS)]
        elif xmax - xmin > 1e-7:
            rtscan = rts[res < 1e-10]
        else:
            rtscan = rts[res < 1e-12]
        # Collapse near-duplicate candidates (a shared x found in both polys).
        i = 0
        while i < len(rtscan):
            if i + 1 < len(rtscan) and abs(rtscan[i] - rtscan[i + 1]) < 10 * ep:
                vi = max(abs(coef1 @ _vandercheb(rtscan[i], G.shape[1])),
                         abs(coef2 @ _vandercheb(rtscan[i], F.shape[1])))
                vip = max(abs(coef1 @ _vandercheb(rtscan[i + 1], G.shape[1])),
                          abs(coef2 @ _vandercheb(rtscan[i + 1], F.shape[1])))
                xtmp.append(rtscan[i] if vi < vip else rtscan[i + 1])
                ytmp.append(yj)
                i += 2
            else:
                xtmp.append(rtscan[i])
                ytmp.append(yj)
                i += 1

    xtmp = np.asarray(xtmp)
    ytmp = np.asarray(ytmp)
    if doswap == 0:
        xval = 0.5 * (xmin + xmax) + 0.5 * (xmax - xmin) * xtmp
        yval = 0.5 * (ymin + ymax) + 0.5 * (ymax - ymin) * ytmp
    else:
        xval = 0.5 * (xmin + xmax) + 0.5 * (xmax - xmin) * ytmp
        yval = 0.5 * (ymin + ymax) + 0.5 * (ymax - ymin) * xtmp
    return xval, yval


def _onevar(F, G, xmin, xmax, ymin, ymax):
    """Common zeros when one of ``F``, ``G`` is univariate (ports ``onevar``)."""
    doswap = 0
    if min(F.shape) == 1:
        if F.shape[1] == 1:
            FF, GG, doswap = F.T, G.T, 1
        else:
            FF, GG = F, G
    else:
        if G.shape[1] == 1:
            FF, GG, doswap = G.T, F.T, 1
        else:
            FF, GG = G, F

    xroots, yroots = [], []
    if FF.size == 1:
        if abs(FF.ravel()[0]) < 1e-15:
            if min(GG.shape) == 1:
                gg = GG.ravel()
                nz = np.where(np.abs(gg) > 0)[0]
                rr = _chebT1rts(gg[nz[0]:]) if nz.size else np.array([])
            else:
                v = _vandercheb(0.0, GG.shape[1])
                gg = GG @ v
                nz = np.where(np.abs(gg) > 0)[0]
                rr = _chebT1rts(gg[nz[0]:]) if nz.size else np.array([])
            if GG.shape[0] > GG.shape[1]:
                yroots, xroots = list(rr), list(np.zeros_like(rr))
            else:
                xroots, yroots = list(rr), list(np.zeros_like(rr))
    else:
        ff = FF.ravel()
        nz = np.where(np.abs(ff) > 0)[0]
        rx = _chebT1rts(ff[nz[0]:]) if nz.size else np.array([])
        for rxj in rx:
            vy = _vandercheb(rxj, GG.shape[1])
            coef = GG @ vy
            coef = np.atleast_1d(coef)
            if coef.size <= 1:
                ry = np.array([0.0]) if np.linalg.norm(coef) < 1e-15 \
                    else np.array([])
            else:
                nz2 = np.where(np.abs(coef) > 0)[0]
                ry = _chebT1rts(coef[nz2[0]:]) if nz2.size else np.array([])
            for ryj in ry:
                yroots.append(ryj)
                xroots.append(rxj)

    xroots = np.asarray(xroots, dtype=np.float64)
    yroots = np.asarray(yroots, dtype=np.float64)
    if doswap:
        xroots, yroots = yroots, xroots
    xroots = 0.5 * (xmin + xmax) + 0.5 * (xmax - xmin) * xroots
    yroots = 0.5 * (ymin + ymax) + 0.5 * (ymax - ymin) * yroots
    return xroots, yroots


# ----------------------------------------------------------------------
# Core: form B(y), solve the polynomial eigenproblem, recover x
# ----------------------------------------------------------------------
def _runbezval(F, G, xmin, xmax, ymin, ymax, tol):
    """Bezout resultant core on a resolved subrectangle (ports ``runbezval``)."""
    doswap = 0
    # Swap x <-> y for a better-conditioned / smaller eigenproblem.
    if (max(F.shape[0], G.shape[0]) * (F.shape[1] + G.shape[1])
            > max(F.shape[1], G.shape[1]) * (F.shape[0] + G.shape[0])):
        F, G, doswap = F.T, G.T, 1

    mc = max(F.shape[0] - 1 + G.shape[0] - 1 + 1, 2)

    B0 = _form_bez(F, G, _MAGIC)
    nB = B0.shape[0]
    xnodes = _chebpts_desc(mc)
    B = np.zeros((nB, nB, mc))
    Bsum = np.zeros((nB, nB))
    for i in range(mc):
        B[:, :, i] = _form_bez(F, G, xnodes[i])
        Bsum += np.abs(B[:, :, i])

    # Regularization: strip the numerically singular leading block.
    if tol == 0:
        k = 0
    else:
        d = np.array([np.max(np.abs(Bsum[:i + 1, :i + 1]))
                      for i in range(nB)])
        m = np.where(np.abs(d) / np.max(np.abs(d)) > tol)[0]
        if m.size == 0:
            k = 0
        else:
            m0 = m[0]                       # 0-based first significant index
            offd = np.array([np.max(np.abs(Bsum[i + 1:, :i + 1]))
                             for i in range(m0)]) if m0 > 0 else np.array([])
            mm = np.where(offd / np.max(np.abs(d)) < np.sqrt(tol))[0] \
                if offd.size else np.array([])
            m0 = 0 if mm.size == 0 else int(mm[-1] + 1) - 1
            k = max(m0, 0)

    if nB - k < 1:
        ei = np.array([])
    else:
        B = B[k:, k:, :]
        nsB = B.shape[0]
        # values -> Chebyshev coefficients along the y-axis.
        B = _matrix_cheb_coeffs(B)
        # Drop negligible leading (high-degree) coefficient slices.
        nrmB = np.linalg.norm(B[:, :, -1], "fro")
        ii = 0
        if nrmB > 0:
            for ii in range(B.shape[2]):
                if np.linalg.norm(B[:, :, ii], "fro") / nrmB > 10 * _EPS:
                    break
        B = B[:, :, ii:]
        if B.shape[2] < 2:
            ei = np.array([])
        else:
            Dori = _balance_cong(B0[k:, k:])
            for i in range(B.shape[2]):
                B[:, :, i] = Dori @ B[:, :, i] @ Dori
                B[:, :, i] = np.rot90(B[:, :, i], 2)
            # coefficients are ascending in degree along axis 2; the pencil
            # wants highest-degree first.
            Bp = B[:, :, ::-1]
            nrm = np.linalg.norm(
                B0[k:, k:].reshape(nsB, -1), "fro") / nsB
            if nrm == 0:
                nrm = 1.0
            ei = _chebT1rts_matgep(Bp / nrm)

    ei = np.asarray(ei)
    if ei.size == 0:
        yreal = np.array([])
    else:
        mask = ((np.abs(np.real(ei)) <= 1 + 10 * _EPS)
                & (np.abs(np.imag(ei)) < np.sqrt(_EPS) * 10))
        yreal = np.sort(np.real(ei[mask]))

    return _xunivariate(F, G, xmin, xmax, ymin, ymax, yreal, doswap)


# ----------------------------------------------------------------------
# Subdivision
# ----------------------------------------------------------------------
def _subroots(f, g, xmin, xmax, ymin, ymax, xwid, ywid, tolreg, maxd,
              subdividestop):
    """Recursive subdivision then Bezout on small cells (ports ``subrootsreptwo``)."""
    if xmin == xmax or ymin == ymax:
        return np.array([]), np.array([])
    approx_tol = 1e-13

    F, _ = _cheb2(f, xmin, xmax, ymin, ymax, approx_tol, maxd + 2)
    G, _ = _cheb2(g, xmin, xmax, ymin, ymax, approx_tol, maxd + 2)

    if (xmax - xmin) > xwid * subdividestop \
            or (ymax - ymin) > ywid * subdividestop:
        both = ((F.shape[0] > maxd and F.shape[1] > maxd)
                or (G.shape[0] > maxd and G.shape[1] > maxd))
        in_y = (F.shape[0] > maxd) or (G.shape[0] > maxd)
        in_x = (F.shape[1] > maxd) or (G.shape[1] > maxd)
        if both:
            xmed = 0.5 * (xmax + xmin) - _MAGIC * 0.5 * (xmax - xmin)
            ymed = 0.5 * (ymin + ymax) - _HONOUR * 0.5 * (ymax - ymin)
            out = [
                _subroots(f, g, xmin, xmed, ymin, ymed, xwid, ywid, tolreg,
                          maxd, subdividestop),
                _subroots(f, g, xmed, xmax, ymin, ymed, xwid, ywid, tolreg,
                          maxd, subdividestop),
                _subroots(f, g, xmin, xmed, ymed, ymax, xwid, ywid, tolreg,
                          maxd, subdividestop),
                _subroots(f, g, xmed, xmax, ymed, ymax, xwid, ywid, tolreg,
                          maxd, subdividestop),
            ]
            return (np.concatenate([o[0] for o in out]),
                    np.concatenate([o[1] for o in out]))
        if in_y:
            ymed = 0.5 * (ymin + ymax) - _MAGIC * 0.5 * (ymax - ymin)
            a = _subroots(f, g, xmin, xmax, ymin, ymed, xwid, ywid, tolreg,
                          maxd, subdividestop)
            b = _subroots(f, g, xmin, xmax, ymed, ymax, xwid, ywid, tolreg,
                          maxd, subdividestop)
            return (np.concatenate([a[0], b[0]]),
                    np.concatenate([a[1], b[1]]))
        if in_x:
            xmed = 0.5 * (xmax + xmin) - _HONOUR * 0.5 * (xmax - xmin)
            a = _subroots(f, g, xmin, xmed, ymin, ymax, xwid, ywid, tolreg,
                          maxd, subdividestop)
            b = _subroots(f, g, xmed, xmax, ymin, ymax, xwid, ywid, tolreg,
                          maxd, subdividestop)
            return (np.concatenate([a[0], b[0]]),
                    np.concatenate([a[1], b[1]]))
    else:
        # Domain small enough: resample a little denser and solve here.
        F, _ = _cheb2(f, xmin, xmax, ymin, ymax, approx_tol, int(round(1.5 * maxd)))
        G, _ = _cheb2(g, xmin, xmax, ymin, ymax, approx_tol, int(round(1.5 * maxd)))

    if F.size == 0:
        F = np.zeros((1, 1))
    if G.size == 0:
        G = np.zeros((1, 1))
    # "No roots here" test: dominated by the T_0 T_0 term alone.
    if ((2 - _EPS * 10) * abs(F[-1, -1]) > np.sum(np.abs(F))
            or (2 - _EPS * 10) * abs(G[-1, -1]) > np.sum(np.abs(G))):
        return np.array([]), np.array([])

    if min(min(F.shape), min(G.shape)) <= 1:
        if F.size <= 1 and G.size <= 1:
            if abs(F.ravel()[0]) <= 1e-15 and abs(G.ravel()[0]) <= 1e-15:
                return (np.array([0.5 * (xmax + xmin)]),
                        np.array([0.5 * (ymax + ymin)]))
            return np.array([]), np.array([])
        xr, yr = _onevar(F, G, xmin, xmax, ymin, ymax)
        return np.atleast_1d(xr), np.atleast_1d(yr)

    xr, yr = _runbezval(F, G, xmin, xmax, ymin, ymax, tolreg)
    return np.atleast_1d(xr), np.atleast_1d(yr)


# ----------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------
def resultant_common_zeros(f, g):
    """Isolated common zeros of two Chebfun2 by the Bezout resultant method.

    Parameters
    ----------
    f, g : Chebfun2
        The two bivariate functions, sharing a rectangular domain.

    Returns
    -------
    numpy.ndarray, shape (m, 2)
        Rows are the ``[x, y]`` points where both ``f`` and ``g`` vanish.

    Provenance
    ----------
    MATLAB source : @chebfun2v/roots.m (roots_resultant)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    xa, xb, ya, yb = (float(v) for v in f.domain)
    xwid = 0.5 * (xb - xa)
    ywid = 0.5 * (yb - ya)

    # Max tensor degree drives the subdivision threshold.
    dd = 0
    for h in (f, g):
        dd = max(dd,
                 max(len(c.coeffs) for c in h.approx.cols),
                 max(len(r.coeffs) for r in h.approx.rows))
    max_degree = min(16, max(dd, 2))
    reg_tol = 1e-15
    overlook = 1e-10

    xmax = xb + xwid * overlook
    xmin = xa - xwid * overlook
    ymax = yb + ywid * overlook
    ymin = ya - ywid * overlook

    subdividestop = 2 * (0.5) ** ((np.log(16) - np.log(max(dd, 2)))
                                  / np.log(0.79))
    subdividestop = min(subdividestop, 0.25)

    # Scale to O(1) for conditioning.
    def _scale(h):
        gx = np.linspace(xa, xb, 17)
        gy = np.linspace(ya, yb, 17)
        xx, yy = np.meshgrid(gx, gy)
        s = float(np.max(np.abs(
            np.asarray(h(jnp.asarray(xx), jnp.asarray(yy))))))
        return s if s > 0 else 1.0

    sf, sg = _scale(f), _scale(g)

    def fs(x, y):
        return f(x, y) / sf

    def gs(x, y):
        return g(x, y) / sg

    xr, yr = _subroots(fs, gs, xmin, xmax, ymin, ymax, xwid, ywid,
                       reg_tol, max_degree, subdividestop)
    xr = np.asarray(xr, dtype=np.float64).ravel()
    yr = np.asarray(yr, dtype=np.float64).ravel()
    if xr.size == 0:
        return np.zeros((0, 2))

    # 2D Newton polish with the exact Jacobian (batched).
    xr, yr = _newton_polish(f, g, xr, yr, xa, xb, ya, yb)

    # Keep genuine, in-domain common zeros; de-duplicate.
    scl = max(xb - xa, yb - ya)
    x = jnp.asarray(xr)
    y = jnp.asarray(yr)
    rf = np.abs(np.asarray(f(x, y), dtype=np.float64))
    rg = np.abs(np.asarray(g(x, y), dtype=np.float64))
    good = ((np.maximum(rf, rg) < 1e-8 * scl ** 2)
            & (xr >= xa - 1e-10) & (xr <= xb + 1e-10)
            & (yr >= ya - 1e-10) & (yr <= yb + 1e-10))
    pts = np.column_stack([xr[good], yr[good]])
    if pts.shape[0] == 0:
        return np.zeros((0, 2))

    keep = []
    for p in pts:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 1e-6 * scl for q in keep):
            keep.append(p)
    return np.array(keep)


def _newton_polish(f, g, xr, yr, xa, xb, ya, yb):
    """Batched 2D Newton refinement of ballpark roots to Newton accuracy."""
    fx, fy = f.diff(dim=2), f.diff(dim=1)
    gx, gy = g.diff(dim=2), g.diff(dim=1)
    m = 1e-6 * max(xb - xa, yb - ya)
    P = np.column_stack([xr, yr])
    for _ in range(40):
        x = jnp.asarray(P[:, 0])
        y = jnp.asarray(P[:, 1])
        F1 = np.asarray(f(x, y), dtype=np.float64)
        F2 = np.asarray(g(x, y), dtype=np.float64)
        J11 = np.asarray(fx(x, y), dtype=np.float64)
        J12 = np.asarray(fy(x, y), dtype=np.float64)
        J21 = np.asarray(gx(x, y), dtype=np.float64)
        J22 = np.asarray(gy(x, y), dtype=np.float64)
        det = J11 * J22 - J12 * J21
        with np.errstate(divide="ignore", invalid="ignore"):
            sx = (J22 * F1 - J12 * F2) / det
            sy = (-J21 * F1 + J11 * F2) / det
        sx = np.where(np.abs(det) > 0, sx, 0.0)
        sy = np.where(np.abs(det) > 0, sy, 0.0)
        P = P - np.column_stack([sx, sy])
        P[:, 0] = np.clip(P[:, 0], xa - m, xb + m)
        P[:, 1] = np.clip(P[:, 1], ya - m, yb + m)
    return P[:, 0], P[:, 1]
