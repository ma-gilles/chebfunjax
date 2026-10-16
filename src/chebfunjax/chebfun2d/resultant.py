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
4. Ballpark roots undergo the source danger-region search, local Bezout
   refinement and residual-based collapse, followed by domain filtering.

The end result is the isolated common zeros as an ``(m, 2)`` array of ``[x, y]``
points. Local Bezout refinement follows the native resultant algorithm.

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

from chebfunjax.utils._generalized_schur import qz

__all__ = ["resultant_common_zeros"]

_EPS = float(jnp.finfo(jnp.float64).eps)
_MAGIC = 0.004849834917525       # 'magic number' (off-centre split point)
_HONOUR = -0.0005194318842611    # 'honourary number' (second split point)


# ----------------------------------------------------------------------
# Chebyshev sampling / transforms (self-consistent, native chebtech2 order)
# ----------------------------------------------------------------------
def _chebpts_desc(n: int) -> jnp.ndarray:
    """``n`` second-kind Chebyshev points, descending ``+1 -> -1``.

    Uses the native sine formula in reverse order to preserve symmetry.
    """
    if n == 1:
        return jnp.array([0.0])
    m = n - 1
    # MATLAB colon -m:2:m includes floor(n) nodes for half-integer n.
    return jnp.sin(jnp.pi * (-m + 2*jnp.arange(int(n))) / (2*m))[::-1]


def _vals2coeffs(v: jnp.ndarray) -> jnp.ndarray:
    """Values at ``_chebpts_desc(n)`` to Chebyshev-``T`` coefficients (ascending).

    Operates column-wise on 2D input.  Mirrors ``@chebtech2/vals2coeffs.m``.
    """
    v = jnp.asarray(v)
    n = v.shape[0]
    if n <= 1:
        return v
    # Samples come from _chebpts_desc (descending +1 -> -1); the chebtech2
    # transform below assumes ascending order, so reverse first.
    v = v[::-1]
    tmp = _concat([v[n - 1:0:-1], v[:n - 1]], axis=0)
    if jnp.isrealobj(v):
        c = jnp.real(jnp.fft.ifft(tmp, axis=0))
    else:
        c = jnp.fft.ifft(tmp, axis=0)
    c = c[:n]
    c = c.at[1:n - 1].set(2.0 * c[1:n - 1])
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
    xx, yy = jnp.meshgrid(xs, ys)   # shape (npts_y, npts_x)
    vals = jnp.asarray(
        f(jnp.asarray(xx), jnp.asarray(yy)), dtype=jnp.float64)
    vscl = max(1.0, float(jnp.max(jnp.abs(vals))))
    # 2D coefficients, ascending in both x (axis=1) and y (axis=0).
    c = _vals2coeffs(_vals2coeffs(vals.T).T)   # x then y
    # Trim negligible high-degree rows/cols.
    keep_row = jnp.where(jnp.max(jnp.abs(c), axis=1) > tol * vscl)[0]
    keep_col = jnp.where(jnp.max(jnp.abs(c), axis=0) > tol * vscl)[0]
    if keep_row.size == 0 or keep_col.size == 0:
        return jnp.zeros((1, 1)), vscl
    c = c[:keep_row[-1] + 1, :keep_col[-1] + 1]
    # Flip to MATLAB descending orientation (highest degree first).
    return c[::-1, ::-1].copy(), vscl


# ----------------------------------------------------------------------
# Bezoutian construction (Chebyshev basis)
# ----------------------------------------------------------------------
def _vandercheb(x: float, n: int) -> jnp.ndarray:
    """Chebyshev-vandermonde vector ``[T_{n-1}(x), ..., T_1(x), T_0(x)]``."""
    v = jnp.zeros(n)
    t = jnp.arccos(complex(x)) if abs(x) > 1 else jnp.arccos(x)
    for i in range(1, n + 1):
        v = v.at[n - i].set(jnp.real(jnp.cos((i - 1) * t)))
    return v


def _dlp_for_bez(aa: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Chebyshev-basis Bezoutian via the diagonal-Lyapunov recurrence.

    ``aa`` is a length-``m`` row of Chebyshev-``x`` coefficients (descending)
    and ``v`` the length-``m-1`` companion vector; returns the ``k x k``
    Bezout matrix (``k = m - 1``).  Ports MATLAB ``DLPforbez``.
    """
    aa = jnp.atleast_2d(jnp.asarray(aa, dtype=jnp.float64))   # (1, m)
    m = aa.shape[1]
    k = m - 1
    v = jnp.asarray(v, dtype=jnp.float64).reshape(-1, 1)     # (k, 1)
    s_mat = jnp.vstack([jnp.zeros((1, k + 1)), (2.0 * v) @ aa])   # (k+1, k+1)
    r = s_mat.T - s_mat
    if k == 1:
        return jnp.array([[r[0, 1]]])
    y = jnp.zeros((k, k))
    y = y.at[0, :].set(r[0, 1:])
    y = y.at[1, :].set(r[1, 1:] + _concat([y[0, 1:k - 1], [2.0 * y[0, k - 1]], [0.0]]) + _concat([[0.0], y[0, :k - 1]]))
    for i in range(2, k):
        y = y.at[i, :].set(r[i, 1:] - y[i - 2, :] + _concat([y[i - 1, 1:k - 1], [2.0 * y[i - 1, k - 1]], [0.0]]) + _concat([[0.0], y[i - 1, :k - 1]]))
    y = y.at[k - 1, :].set(y[k - 1, :] / 2.0)
    return y


def _form_bez(F: jnp.ndarray, G: jnp.ndarray, y: float) -> jnp.ndarray:
    """Bezoutian resultant matrix of ``f(.,y)``, ``g(.,y)`` (ports ``formBez``)."""
    nf, ng = F.shape[0], G.shape[0]
    yv = _vandercheb(y, max(nf, ng))
    ff = yv[len(yv) - nf:] @ F           # (mf,)
    gg = yv[len(yv) - ng:] @ G           # (mg,)
    ff = jnp.atleast_1d(ff)
    gg = jnp.atleast_1d(gg)
    if len(ff) < len(gg):
        ff, gg = gg, ff
    if len(ff) == len(gg):
        ff = _concat([[0.0], ff])
        shrink = True
    else:
        shrink = False
    v = _concat([jnp.zeros(len(ff) - len(gg) - 1), gg])
    b = _dlp_for_bez(ff, v)
    if shrink:
        b = b[1:, 1:]
    return b


def _balance_cong(B: jnp.ndarray) -> jnp.ndarray:
    """Diagonal congruence balancing (ports ``balancecong``)."""
    n = B.shape[0]
    d = jnp.ones(n)
    last = jnp.linalg.norm(B[-1, :])
    for i in range(n - 2, -1, -1):
        ni = jnp.linalg.norm(B[i, :])
        if ni > 0:
            d = d.at[i].set(max(1.0, jnp.sqrt(last / ni)))
        else:
            d = d.at[i].set(d[i + 1])
    return jnp.diag(d)


def _matrix_cheb_coeffs(bvals: jnp.ndarray) -> jnp.ndarray:
    """Convert matrix *values* sampled at ``_chebpts_desc(mc)`` (3rd axis) to
    matrix Chebyshev *coefficients* (ports ``matrixChebfft``).

    ``bvals`` has shape ``(n, n, mc)``; returns the same shape holding the
    Chebyshev-``y`` coefficients (descending) of each entry.
    """
    n, _, mc = bvals.shape
    flat = bvals.reshape(n * n, mc).T          # (mc, n*n)
    coeffs = _vals2coeffs(flat)                # (mc, n*n)
    return coeffs.T.reshape(n, n, mc)[:, :, ::-1]


# ----------------------------------------------------------------------
# Polynomial rootfinders (colleague matrix / pencil)
# ----------------------------------------------------------------------
def _chebT1rts(c_desc, imagtol=1e-8):
    """Native colleague eigenvalues with caller-specific real-root filtering."""
    c = jnp.asarray(c_desc, dtype=jnp.float64).ravel()
    nz = jnp.where(jnp.abs(c) > 0)[0]
    if nz.size == 0:
        return jnp.empty(0)
    c = c[int(nz[0]):]
    if c.size <= 1:
        return jnp.empty(0)
    if c.size == 2:
        roots = jnp.array([-c[1] / c[0]], dtype=jnp.complex128)
    else:
        tail = -0.5 * c[:0:-1] / c[0]
        tail = tail.at[-2].add(0.5)
        off = 0.5 * jnp.ones(tail.size - 1)
        matrix = jnp.diag(off, 1) + jnp.diag(off, -1)
        matrix = matrix.at[-1, -2].set(1.0).at[0, :].set(tail[::-1])
        roots = jnp.linalg.eigvals(matrix)
    keep = (jnp.abs(roots.imag) <= imagtol) & (jnp.abs(roots) <= 1 + 10*_EPS)
    return jnp.sort(roots[keep].real)


def _chebT1rts_matgep(c: jnp.ndarray) -> jnp.ndarray:
    """Roots of a matrix polynomial in the first-kind Chebyshev-``T`` basis via
    the colleague matrix *pencil* (ports ``chebT1rtsmatgep``).

    ``c`` has shape ``(n, n, k)``: ``k-1`` is the degree (index 0 = highest
    degree), ``n`` the matrix size.  Returns the (complex) eigenvalues.
    """
    c = jnp.array(c, dtype=jnp.float64)
    n = c.shape[0]
    k = c.shape[2]
    if k == 2:                       # linear pencil
        return _generalized_eigvals(c[:, :, 1], -c[:, :, 0])
    for ii in range(1, k):
        c = c.at[:, :, ii].set(c[:, :, ii] * -0.5)
    c = c.at[:, :, 2].set(c[:, :, 2] + 0.5 * c[:, :, 0])
    oh = 0.5 * jnp.ones(n * (k - 2))
    A = jnp.diag(oh, n) + jnp.diag(oh, -n)
    A = A.at[-n:, -2 * n:-n].set(jnp.eye(n))
    for ii in range(k - 1):
        A = A.at[:n, ii * n:(ii + 1) * n].set(c[:, :, ii + 1])
    B = jnp.eye(A.shape[0])
    B = B.at[:n, :n].set(c[:, :, 0])
    return _generalized_eigvals(A, B)


# ----------------------------------------------------------------------
# Univariate x-recovery from candidate y-values
# ----------------------------------------------------------------------
def _xunivariate(F, G, xmin, xmax, ymin, ymax, y, doswap):
    """Recover ``x`` for each candidate ``y`` (ports ``xunivariate``)."""
    ep = 10 * _EPS
    y = jnp.sort(jnp.asarray(y, dtype=jnp.float64))
    if y.size == 0:
        return jnp.array([]), jnp.array([])
    dist = jnp.abs(y - _concat([y[1:], [2.0]]))
    y = y[dist > ep]

    xtmp, ytmp = [], []
    for yj in y:
        vy = _vandercheb(yj, G.shape[0])
        coef1 = vy @ G                       # g(.,yj) x-coeffs (descending)
        vy = _vandercheb(yj, F.shape[0])
        coef2 = vy @ F                       # f(.,yj) x-coeffs (descending)
        coef1 = jnp.atleast_1d(coef1)
        coef2 = jnp.atleast_1d(coef2)

        rts1 = _chebT1rts(coef1, 1e-7) if (coef1.size > 1
                                     and jnp.linalg.norm(coef1) != 0) \
            else jnp.array([])
        rts2 = _chebT1rts(coef2, 1e-7) if (coef2.size > 1
                                     and jnp.linalg.norm(coef2) != 0) \
            else jnp.array([])
        # Source collapses each polynomial's near-multiple roots first.
        rts1 = rts1[jnp.abs(rts1 - _concat((rts1[1:], [2.]))) > 10*ep]
        rts2 = rts2[jnp.abs(rts2 - _concat((rts2[1:], [2.]))) > 10*ep]
        rts = jnp.sort(_concat([rts1, rts2]))
        if rts.size == 0:
            continue
        # residual of each candidate x in BOTH polynomials.
        res = jnp.array([
            max(abs(coef1 @ _vandercheb(rx, G.shape[1])),
                abs(coef2 @ _vandercheb(rx, F.shape[1])))
            for rx in rts])
        if xmax - xmin > 2e-3:
            rtscan = rts[res < 100 * jnp.sqrt(_EPS)]
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

    xtmp = jnp.asarray(xtmp)
    ytmp = jnp.asarray(ytmp)
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
                nz = jnp.where(jnp.abs(gg) > 0)[0]
                rr = _chebT1rts(gg[nz[0]:]) if nz.size else jnp.array([])
            else:
                v = _vandercheb(0.0, GG.shape[1])
                GG = (GG @ v).reshape(-1, 1)
                gg = GG.ravel()
                nz = jnp.where(jnp.abs(gg) > 0)[0]
                rr = _chebT1rts(gg[nz[0]:]) if nz.size else jnp.array([])
            if GG.shape[0] > GG.shape[1]:
                yroots, xroots = list(rr), list(jnp.zeros_like(rr))
            else:
                xroots, yroots = list(rr), list(jnp.zeros_like(rr))
    else:
        ff = FF.ravel()
        nz = jnp.where(jnp.abs(ff) > 0)[0]
        rx = _chebT1rts(ff[nz[0]:]) if nz.size else jnp.array([])
        for rxj in rx:
            vy = _vandercheb(rxj, GG.shape[1])
            coef = GG @ vy
            coef = jnp.atleast_1d(coef)
            if coef.size <= 1:
                ry = jnp.array([0.0]) if jnp.linalg.norm(coef) < 1e-15 \
                    else jnp.array([])
            else:
                nz2 = jnp.where(jnp.abs(coef) > 0)[0]
                ry = _chebT1rts(coef[nz2[0]:]) if nz2.size else jnp.array([])
            for ryj in ry:
                yroots.append(ryj)
                xroots.append(rxj)

    xroots = jnp.asarray(xroots, dtype=jnp.float64)
    yroots = jnp.asarray(yroots, dtype=jnp.float64)
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
    B = jnp.zeros((nB, nB, mc))
    Bsum = jnp.zeros((nB, nB))
    for i in range(mc):
        B = B.at[:, :, i].set(_form_bez(F, G, xnodes[i]))
        Bsum += jnp.abs(B[:, :, i])

    # Regularization: strip the numerically singular leading block.
    if tol == 0:
        k = 0
    else:
        d = jnp.array([jnp.max(jnp.abs(Bsum[:i + 1, :i + 1]))
                      for i in range(nB)])
        m = jnp.where(jnp.abs(d) / jnp.max(jnp.abs(d)) > tol)[0]
        if m.size == 0:
            k = 0
        else:
            m0 = int(m[0])                       # 0-based first significant index
            offd = jnp.array([jnp.max(jnp.abs(Bsum[i + 1:, :i + 1]))
                             for i in range(m0)]) if m0 > 0 else jnp.array([])
            mm = jnp.where(offd / jnp.max(jnp.abs(d)) < jnp.sqrt(tol))[0] \
                if offd.size else jnp.array([])
            m0 = 0 if mm.size == 0 else int(mm[-1] + 1) - 1
            k = max(m0, 0)

    if nB - k < 1:
        ei = jnp.array([])
    else:
        B = B[k:, k:, :]
        nsB = B.shape[0]
        sampled_norm = jnp.linalg.norm(B.reshape(nsB, -1), "fro") / nsB
        # values -> Chebyshev coefficients along the y-axis.
        B = _matrix_cheb_coeffs(B)
        # Drop negligible leading (high-degree) coefficient slices.
        nrmB = jnp.linalg.norm(B[:, :, -1], "fro")
        ii = 0
        if nrmB > 0:
            for ii in range(B.shape[2]):
                if jnp.linalg.norm(B[:, :, ii], "fro") / nrmB > 10 * _EPS:
                    break
        B = B[:, :, ii:]
        if B.shape[2] < 2:
            ei = jnp.array([])
        else:
            Dori = _balance_cong(B0[k:, k:])
            for i in range(B.shape[2]):
                B = B.at[:, :, i].set(Dori @ B[:, :, i] @ Dori)
                B = B.at[:, :, i].set(jnp.rot90(B[:, :, i], 2))
            # Matrix transform already returns the source descending order.
            Bp = B
            nrm = sampled_norm
            if nrm == 0:
                nrm = 1.0
            ei = _chebT1rts_matgep(Bp / nrm)

    ei = jnp.asarray(ei)
    if ei.size == 0:
        yreal = jnp.array([])
    else:
        mask = ((jnp.abs(jnp.real(ei)) <= 1 + 10 * _EPS)
                & (jnp.abs(jnp.imag(ei)) < jnp.sqrt(_EPS) * 10))
        yreal = jnp.sort(jnp.real(ei[mask]))

    return _xunivariate(F, G, xmin, xmax, ymin, ymax, yreal, doswap)


# ----------------------------------------------------------------------
# Subdivision
# ----------------------------------------------------------------------
def _subroots(f, g, xmin, xmax, ymin, ymax, xwid, ywid, tolreg, maxd,
              subdividestop):
    """Recursive subdivision then Bezout on small cells (ports ``subrootsreptwo``)."""
    if xmin == xmax or ymin == ymax:
        return jnp.array([]), jnp.array([])
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
            return (_concat([out[i][0] for i in (0, 2, 1, 3)]),
                    _concat([out[i][1] for i in (0, 2, 1, 3)]))
        if in_y:
            ymed = 0.5 * (ymin + ymax) - _MAGIC * 0.5 * (ymax - ymin)
            a = _subroots(f, g, xmin, xmax, ymin, ymed, xwid, ywid, tolreg,
                          maxd, subdividestop)
            b = _subroots(f, g, xmin, xmax, ymed, ymax, xwid, ywid, tolreg,
                          maxd, subdividestop)
            return (_concat([a[0], b[0]]),
                    _concat([a[1], b[1]]))
        if in_x:
            xmed = 0.5 * (xmax + xmin) - _HONOUR * 0.5 * (xmax - xmin)
            a = _subroots(f, g, xmin, xmed, ymin, ymax, xwid, ywid, tolreg,
                          maxd, subdividestop)
            b = _subroots(f, g, xmed, xmax, ymin, ymax, xwid, ywid, tolreg,
                          maxd, subdividestop)
            return (_concat([a[0], b[0]]),
                    _concat([a[1], b[1]]))
    else:
        # Domain small enough: resample a little denser and solve here.
        F, _ = _cheb2(f, xmin, xmax, ymin, ymax, approx_tol, int(1.5 * maxd + 0.5))
        G, _ = _cheb2(g, xmin, xmax, ymin, ymax, approx_tol, int(1.5 * maxd + 0.5))

    if F.size == 0:
        F = jnp.zeros((1, 1))
    if G.size == 0:
        G = jnp.zeros((1, 1))
    # "No roots here" test: dominated by the T_0 T_0 term alone.
    if ((2 - _EPS * 10) * abs(F[-1, -1]) > jnp.sum(jnp.abs(F))
            or (2 - _EPS * 10) * abs(G[-1, -1]) > jnp.sum(jnp.abs(G))):
        return jnp.array([]), jnp.array([])

    if min(min(F.shape), min(G.shape)) <= 1:
        if F.size <= 1 and G.size <= 1:
            if abs(F.ravel()[0]) <= 1e-15 and abs(G.ravel()[0]) <= 1e-15:
                return (jnp.array([0.5 * (xmax + xmin)]),
                        jnp.array([0.5 * (ymax + ymin)]))
            return jnp.array([]), jnp.array([])
        xr, yr = _onevar(F, G, xmin, xmax, ymin, ymax)
        return jnp.atleast_1d(xr), jnp.atleast_1d(yr)

    xr, yr = _runbezval(F, G, xmin, xmax, ymin, ymax, tolreg)
    return jnp.atleast_1d(xr), jnp.atleast_1d(yr)


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
    jax.Array, shape (m, 2)
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
    max_degree = min(16, dd)
    reg_tol = 1e-15
    overlook = 1e-10

    xmax = xb + xwid * overlook
    xmin = xa - xwid * overlook
    ymax = yb + ywid * overlook
    ymin = ya - ywid * overlook

    subdividestop = 2 * (0.5) ** ((jnp.log(16) - jnp.log(dd))
                                  / jnp.log(0.79))
    subdividestop = min(subdividestop, 0.25)

    # Scale to O(1) for conditioning.
    def _scale(h):
        # MATLAB: f = f/abs(f.pivotValues(1)) -- the first (largest) GE
        # pivot; chebfunjax stores pivots = 1/pivotValues.
        piv = jnp.asarray(h.approx.pivots, dtype=float).ravel()
        s = 1.0 / abs(float(piv[0])) if piv.size and piv[0] != 0 else 1.0
        return s if jnp.isfinite(s) and s > 0 else 1.0

    sf, sg = _scale(f), _scale(g)

    def fs(x, y):
        return f(x, y) / sf

    def gs(x, y):
        return g(x, y) / sg

    xr, yr = _subroots(fs, gs, xmin, xmax, ymin, ymax, xwid, ywid,
                       reg_tol, max_degree, subdividestop)
    return _refine_resultant(
        f, g, fs, gs, sf, sg, xr, yr, xmin, xmax, ymin, ymax,
        xwid, ywid, dd, max_degree,
    )


def _blockingxy(x, y, delta):
    """Source coordinate-gap grouping, preserving the native block order.

    Provenance
    ----------
    MATLAB source: @chebfun2v/roots.m (blockingxy)
    Chebfun commit: 7574c77
    """
    x, y = jnp.asarray(x), jnp.asarray(y)
    if x.size == 0:
        return []
    ix = jnp.argsort(x, stable=True)
    starts = [0]
    for k in range(x.size - 1):
        if bool(jnp.abs(x[ix[k]] - x[ix[k + 1]]) > delta):
            starts.append(k + 1)
    starts.append(x.size)
    groups = []
    for lo, hi in zip(starts[:-1], starts[1:]):
        ids = ix[lo:hi]
        iy = jnp.argsort(y[ids], stable=True)
        begin = 0
        for k in range(ids.size - 1):
            if bool(jnp.abs(y[ids[iy[k]]] - y[ids[iy[k + 1]]]) > delta):
                # MATLAB labels in sorted order but retains original point order.
                chosen = ids[iy[begin:k + 1]]
                groups.append(jnp.sort(chosen))
                begin = k + 1
        groups.append(jnp.sort(ids[iy[begin:]]))
    return groups


def _refine_resultant(f, g, fs, gs, sf, sg, xr, yr, xmin, xmax,
                      ymin, ymax, xwid, ywid, dd, max_degree):
    """Source danger-region search, local Bezout refinement and root collapse.

    Provenance
    ----------
    MATLAB source: @chebfun2v/roots.m (roots_resultant, lines 133--240)
    Chebfun commit: 7574c77
    """
    eps = jnp.finfo(jnp.float64).eps
    fx0, fy0 = f.diff(dim=2), f.diff(dim=1)
    gx0, gy0 = g.diff(dim=2), g.diff(dim=1)

    def fx(x, y):
        return fx0(x, y) / sf

    def fy(x, y):
        return fy0(x, y) / sf

    def gx(x, y):
        return gx0(x, y) / sg

    def gy(x, y):
        return gy0(x, y) / sg

    def jac(x, y):
        return jnp.array([[fx(x, y), fy(x, y)], [gx(x, y), gy(x, y)]]).reshape(2, 2)

    def local(xlo, xhi, ylo, yhi, reg=0.0, degree=max_degree):
        x, y = _subroots(fs, gs, float(xlo), float(xhi), float(ylo),
                         float(yhi), xwid, ywid, reg, degree, 0.008)
        return jnp.asarray(x), jnp.asarray(y)

    xr, yr = jnp.asarray(xr), jnp.asarray(yr)
    nx = max(max(len(r.coeffs) for r in h.approx.rows) for h in (f, g))
    ny = max(max(len(c.coeffs) for c in h.approx.cols) for h in (f, g))
    xp = jnp.linspace(xmin, xmax, 2 * nx)
    yp = jnp.linspace(ymin, ymax, 2 * ny)
    xx, yy = jnp.meshgrid(xp, yp)
    fxv, fyv, gxv, gyv = fx(xx, yy), fy(xx, yy), gx(xx, yy), gy(xx, yy)
    fg = jnp.abs(fxv * gyv - fyv * gxv)
    fpg = jnp.abs(fs(xx, yy)) + jnp.abs(gs(xx, yy))
    danger = (jnp.log10(fpg + eps) < -6) & (jnp.log10(fg / jnp.max(fg) + eps) < -12)
    candidates = []
    for i, j in zip(*jnp.where(danger)):
        singular = jnp.linalg.svd(jnp.array([[fxv[i, j], fyv[i, j]],
                                            [gxv[i, j], gyv[i, j]]]), compute_uv=False)
        if bool(1.0 / singular[-1] < 1e15):
            candidates.append(jnp.array([xp[j], yp[i]]))
    if candidates:
        points = jnp.stack(candidates)
        xd, yd = xp[1] - xp[0], yp[1] - yp[0]
        for ids in _blockingxy(points[:, 0], points[:, 1], jnp.maximum(xd, yd) * 1.1):
            x, y = points[ids, 0], points[ids, 1]
            xt, yt = local(jnp.min(x)-xd, jnp.max(x)+xd, jnp.min(y)-yd, jnp.max(y)+yd, 1e-15)
            xr, yr = _concat((xr, xt)), _concat((yr, yt))

    tolblock = xwid * min(3e-3, 1.0 / (10 * dd))
    default = min(5 * xwid * float(jnp.sqrt(jnp.sqrt(eps))), tolblock)
    outx, outy = [], []
    for ids in _blockingxy(xr, yr, tolblock):
        x, y = xr[ids], yr[ids]
        matrix = jac(jnp.mean(x), jnp.mean(y))
        cond = 1.0 / jnp.linalg.svd(matrix, compute_uv=False)[-1]
        bezcond = 1.0 / jnp.abs(jnp.linalg.det(matrix))
        width = jnp.minimum(jnp.maximum(default, eps * 100 * bezcond), xwid * 1e-2)
        xt, yt = local(jnp.min(x)-width, jnp.max(x)+width, jnp.min(y)-width, jnp.max(y)+width)
        if bool(cond < 1e10):
            width = jnp.maximum(xwid * jnp.sqrt(eps), cond * 1000 * eps)
            for x, y in zip(xt, yt):
                xs, ys = local(x-width, x+width, y-width, y+width, degree=max_degree / 2)
                outx.append(xs)
                outy.append(ys)
        else:
            outx.append(xt)
            outy.append(yt)
    xr = _concat(outx) if outx else jnp.empty(0)
    yr = _concat(outy) if outy else jnp.empty(0)
    points = []
    for ids in _blockingxy(xr, yr, xwid * jnp.sqrt(eps) * 10):
        x, y = xr[ids], yr[ids]
        residual = jnp.sqrt(jnp.abs(fs(x, y))**2 + jnp.abs(gs(x, y))**2)
        best = jnp.argmin(residual)
        if bool(residual[best] < 1e-10):
            points.append(jnp.array([x[best], y[best]]))
    if not points:
        return jnp.empty((0, 2))
    points = jnp.stack(points)
    xa, xb, ya, yb = f.domain
    x, y = points[:, 0], points[:, 1]
    inside = ((x < xb + xwid*1e-15) & (x > xa - xwid*1e-15)
              & (y < yb + ywid*1e-15) & (y > ya - ywid*1e-15))
    points = points[inside]
    return jnp.stack((jnp.clip(points[:, 0], xa, xb),
                      jnp.clip(points[:, 1], ya, yb)), axis=1)


def _concat(arrays, axis=0):
    return jnp.concatenate(tuple(jnp.asarray(a) for a in arrays), axis=axis)


def _generalized_eigvals(a, b):
    """Generalized eigenvalues from the JAX complex Schur pencil.

    Provenance
    ----------
    MATLAB source: @chebfun2v/roots.m (chebT1rtsmatgep eig(A,B))
    Chebfun commit: 7574c77
    """
    h, t, _, _, status, _ = qz(a, b)
    if int(status) != 0:
        raise RuntimeError("Resultant generalized Schur iteration did not converge")
    alpha, beta = jnp.diag(h), jnp.diag(t)
    return jnp.where(beta == 0, jnp.where(alpha == 0, jnp.nan, jnp.inf), alpha / beta)
