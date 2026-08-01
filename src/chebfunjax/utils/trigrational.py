# uses-numpy: Toeplitz/null-space manipulation is one-shot numpy
"""Trigonometric (Fourier) rational approximation: trigpade.

Added by Claude Fable 5 (Big-Three directive, trig rational
approximation).

Provenance
----------
MATLAB source : @chebfun/trigpade.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.linalg import null_space, toeplitz

__all__ = ["trigpade", "trigremez"]


def _trig_coeffs_ascending(f) -> np.ndarray:
    """MATLAB trigcoeffs layout [c_{-N} ... c_0 ... c_N] (odd length)
    from a single-piece trig chebfun."""
    c = np.asarray(f.funs[0].tech.coeffs)  # descending wavenumber
    c = c[::-1].astype(complex)            # ascending
    if len(c) % 2 == 0:
        # split the top mode symmetrically to make the length odd
        c = np.concatenate([[c[0] / 2.0], c[1:], [c[0] / 2.0]])
    return c


def _trig_chebfun_from_ascending(c, domain):
    """Build a trig chebfun from ascending Laurent coefficients."""
    from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, _Piece
    from chebfunjax.tech.trigtech import Trigtech
    c = np.atleast_1d(np.asarray(c, dtype=complex))
    tech = Trigtech.from_coeffs(jnp.asarray(c[::-1]))
    piece = _Piece(tech=tech,
                   interval=(float(domain[0]), float(domain[1])))
    return Chebfun(funs=[piece],
                   domain=Domain((float(domain[0]),
                                  float(domain[1]))))


def _laurent_approx(c, m, n, N, tol):
    """One-sided Laurent-Pade coefficients (MATLAB laurent_approx)."""
    col = c[(m + 1) + N: (m + n) + N + 1]
    row = c[np.arange((m + 1) + N, (m - n) + N, -1)]
    C = toeplitz(col, row)
    b = null_space(C)
    if b.size == 0:
        # fall back to the smallest right singular vector
        _, _, Vh = np.linalg.svd(C)
        b = Vh[-1, :].conj()[:, None]
    if b.shape[1] > 1:
        b = b[:, :1]
    b = b[:, 0]
    if abs(b[0]) < tol:
        raise ValueError(
            "trigpade: denominator zero at the origin detected")
    b = b / b[0]
    M = max(m, n)
    col2 = c[N: M + N + 1].copy()
    col2[0] = col2[0] / 2.0
    C2 = np.tril(toeplitz(col2, col2))
    bb = np.concatenate([b, np.zeros(M + 1 - len(b))])
    a = C2 @ bb
    return a, b


def _laurent_pade(c, m, n, tol):
    c = np.asarray(c, dtype=complex).ravel()
    N = (len(c) - 1) // 2
    if n == 0:
        ap = c[N: N + m + 1].copy()
        am = c[N:: -1][: m + 1].copy()
        ap[0] /= 2.0
        am[0] /= 2.0
        ap = np.concatenate([np.zeros(len(ap) - 1), ap])
        am = np.concatenate([np.zeros(len(am) - 1), am])
        return ap, np.array([1.0]), am[::-1], np.array([1.0])
    ap, bp = _laurent_approx(c, m, n, N, tol)
    c_rev = c[::-1]
    if np.max(np.abs(c - np.conj(c_rev))) < 10 * tol:
        am, bm = np.conj(ap), np.conj(bp)
    else:
        am, bm = _laurent_approx(c_rev, m, n, N, tol)
    ap = np.concatenate([np.zeros(len(ap) - 1), ap])
    bp = np.concatenate([np.zeros(len(bp) - 1), bp])
    am = np.concatenate([np.zeros(len(am) - 1), am])[::-1]
    bm = np.concatenate([np.zeros(len(bm) - 1), bm])[::-1]
    return ap, bp, am, bm


def _chop(c, tol):
    mid = (len(c) - 1) // 2
    nz = np.where(np.abs(c) > tol)[0]
    if len(nz) == 0:
        return np.zeros(1, dtype=c.dtype)
    nn = max(mid - nz[0], nz[-1] - mid)
    return c[mid - nn: mid + nn + 1]


def _center_pad(v, L):
    k = (len(v) - 1) // 2
    pad = L - k
    return np.concatenate([np.zeros(pad), v, np.zeros(pad)])


def trigpade(f, m: int, n: int):
    """Trigonometric (Fourier) Pade approximation of a periodic
    chebfun (MATLAB trigpade): returns
    ``(p, q, r, tn_p, td_p, tn_m, td_m)`` with
    ``p/q = tn_p/td_p + tn_m/td_m``.

    Provenance
    ----------
    MATLAB source : @chebfun/trigpade.m
    Chebfun commit: 7574c77
    """
    dom = (float(f.domain.a), float(f.domain.b))
    if getattr(f, "isempty", lambda: False)():
        return f
    c = _trig_coeffs_ascending(f)
    N = (len(c) - 1) // 2
    tol = 100 * np.finfo(float).eps * np.max(np.abs(c))

    d = 2 * max(m, n) - N
    if d > 0:
        c = np.concatenate([np.zeros(d), c, np.zeros(d)])
        N += d

    ap, bp, am, bm = _laurent_pade(c, m, n, tol)

    tn_p = _trig_chebfun_from_ascending(ap, dom)
    td_p = _trig_chebfun_from_ascending(bp, dom)
    tn_m = _trig_chebfun_from_ascending(am, dom)
    td_m = _trig_chebfun_from_ascending(bm, dom)

    L = (max(len(ap), len(am), len(bp), len(bm)) - 1) // 2
    ap_, bp_, am_, bm_ = (
        _center_pad(v, L) for v in (ap, bp, am, bm))
    pk = np.convolve(ap_, bm_) + np.convolve(am_, bp_)
    qk = np.convolve(bm_, bp_)
    pk = _chop(pk, tol)
    qk = _chop(qk, tol)

    p = _trig_chebfun_from_ascending(pk, dom)
    q = _trig_chebfun_from_ascending(qk, dom)

    # discard imaginary rounding errors for real input
    if np.max(np.abs(c - np.conj(c[::-1]))) < tol:
        p = p.real()
        q = q.real()

    def r(t):
        return p(t) / q(t)

    return p, q, r, tn_p, td_p, tn_m, td_m


def _trig_trial_rational(fk, xk, m, n):
    """Rational trigonometric trial function via the generalized
    eigenvalue problem (computeTrialFunctionRational).

    Returns (p_handle, q_handle, ac, bc, h) with handles evaluating the
    numerator/denominator trig polynomials on [-pi, pi]; raises
    RuntimeError when no pole-free approximation exists.

    Provenance
    ----------
    MATLAB source : @chebfun/trigremez.m (computeTrialFunctionRational)
    Chebfun commit: 7574c77
    """
    import scipy.linalg as sla
    th = np.asarray(xk)
    L = len(th)
    P = np.ones((L, 2 * m + 1))
    for j in range(1, m + 1):
        P[:, 2 * j - 1] = np.cos(j * th)
        P[:, 2 * j] = np.sin(j * th)
    Q = np.ones((L, 2 * n + 1))
    for j in range(1, n + 1):
        Q[:, 2 * j - 1] = np.cos(j * th)
        Q[:, 2 * j] = np.sin(j * th)
    Ntot = m + n
    F = np.diag(fk)
    A = np.hstack([P, -F @ Q])
    Imat = np.diag((-1.0) ** np.arange(2 * Ntot + 2))
    B = -Imat @ np.hstack([np.zeros_like(P), Q])
    h_eigs, V = sla.eig(A, B)

    def _to_complex(v):
        # real cos/sin coefficients -> ascending complex exponentials
        tmp = (v[1::2] - 1j * v[2::2]) / 2.0
        return np.concatenate([np.conj(tmp[::-1]), [v[0] + 0j], tmp])

    imag_tol = 1e-13
    for j in range(V.shape[1]):
        hj = h_eigs[j]
        if not np.isfinite(hj) or abs(np.imag(hj)) > imag_tol:
            continue
        # MATLAB uses the (possibly complex-scaled) eigenvector columns
        # directly; normalise the arbitrary complex phase so the trig
        # coefficients come out conjugate-symmetric (real function).
        av = V[: 2 * m + 1, j]
        bv = V[2 * m + 1:, j]
        pivot = bv[np.argmax(np.abs(bv))]
        if abs(pivot) > 0:
            phase = pivot / abs(pivot)
            av = av / phase
            bv = bv / phase
        if (np.max(np.abs(np.imag(av))) > 1e-8 * max(np.max(np.abs(av)), 1e-300)
                or np.max(np.abs(np.imag(bv)))
                > 1e-8 * max(np.max(np.abs(bv)), 1e-300)):
            continue
        ac = _to_complex(np.real(av))
        bc = _to_complex(np.real(bv))

        def q_h(t, _bc=bc):
            ks = np.arange(-n, n + 1)
            return np.real(np.exp(1j * np.outer(np.asarray(t), ks))
                           @ _bc)
        # pole-free check: q has no real roots (dense sample sign test
        # + magnitude floor; MATLAB uses roots of the trig chebfun)
        ts = np.linspace(-np.pi, np.pi, 2000, endpoint=False)
        qv = q_h(ts)
        if np.min(np.abs(qv)) < 1e-12 * np.max(np.abs(qv)) or \
                np.any(qv[:-1] * qv[1:] < 0):
            continue

        def p_h(t, _ac=ac):
            ks = np.arange(-m, m + 1)
            return np.real(np.exp(1j * np.outer(np.asarray(t), ks))
                           @ _ac)
        return p_h, q_h, ac, bc, float(np.real(hj))
    raise RuntimeError("trigremez: no pole-free approximation found")


def trigremez(f, m: int, n: int | None = None, max_iter: int = 40,
              tol: float = 1e-14):
    """Best trigonometric polynomial approximation of degree m to a
    periodic chebfun by the Remez algorithm (MATLAB trigremez,
    polynomial case).  Returns ``(p, err_max, status)`` where
    ``status["xk"]`` is the final reference (equioscillation) set.

    Provenance
    ----------
    MATLAB source : @chebfun/trigremez.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and
        The Chebfun Developers (algorithm of Javed & Trefethen).
    """
    from chebfunjax.chebfun1d.chebfun import chebfun as _cf
    from chebfunjax.utils.trigutils import trigBary

    a, b = float(f.domain.a), float(f.domain.b)
    if getattr(f, "isempty", lambda: False)():
        return f

    def to_ref(x):       # [a, b] -> [-pi, pi]
        return -np.pi + 2 * np.pi * (np.asarray(x) - a) / (b - a)

    def from_ref(y):     # [-pi, pi] -> [a, b]
        y = np.asarray(y)
        return b * (y + np.pi) / (2 * np.pi) \
            + a * (np.pi - y) / (2 * np.pi)

    def f_ref(y):
        return np.asarray(f(jnp.asarray(from_ref(y))))

    normf = float(f.norm(np.inf)) or 1.0
    if n is not None and n > 0:
        return _trigremez_rational(f, m, n, max_iter, tol,
                                   a, b, to_ref, from_ref, f_ref, normf)
    N = 2 * m + 2
    xk = -np.pi + 2 * np.pi * np.arange(N) / N
    xo = xk.copy()
    sigma = np.ones(N)
    sigma[1::2] = -1.0

    from chebfunjax.utils.trigutils import trigBaryWeights

    best = None
    deltamin = np.inf
    delta, diffx = normf, 1.0
    it = 0
    while delta / normf > tol and it < max_iter and diffx > 0:
        fk = f_ref(xk)
        w = trigBaryWeights(xk)
        h = float((w @ fk) / (w @ sigma))
        if h == 0:
            h = 1e-19
        pk = fk - h * sigma

        def p_ref(y, pk=pk, xk=xk):
            return trigBary(np.asarray(y), pk, xk,
                            (-np.pi, np.pi))

        # error extrema: dense sampling + local refinement (the
        # MATLAB code uses roots(diff(f - p)); a fine grid with
        # parabolic refinement reaches the same reference set)
        yy = np.linspace(-np.pi, np.pi, max(4000, 40 * N),
                         endpoint=False)
        ee = f_ref(yy) - p_ref(yy)

        # candidate extrema: sign changes of the discrete derivative
        de = np.diff(ee)
        idx = np.where(np.sign(de[1:]) != np.sign(de[:-1]))[0] + 1
        rr = yy[idx]
        er = ee[idx]

        # keep alternating signs, largest magnitude per run
        s_pts, s_val = [rr[0]], [er[0]]
        for r_i, e_i in zip(rr[1:], er[1:]):
            if np.sign(e_i) == np.sign(s_val[-1]):
                if abs(e_i) > abs(s_val[-1]):
                    s_pts[-1], s_val[-1] = r_i, e_i
            else:
                s_pts.append(r_i)
                s_val.append(e_i)
        s_pts = np.array(s_pts)
        s_val = np.array(s_val)

        err = float(np.max(np.abs(s_val)))
        imax = int(np.argmax(np.abs(s_val)))
        d0 = max(imax - N + 1, 0)
        if len(s_pts) >= N:
            xk = np.sort(s_pts[d0: d0 + N])
        else:
            break

        diffx = float(np.max(np.abs(np.sort(xo) - np.sort(xk)))) \
            if len(xo) == len(xk) else 1.0
        delta = err - abs(h)
        if delta < deltamin:
            deltamin = delta
            best = (pk.copy(), xo.copy(), abs(h), err)
        xo = xk.copy()
        it += 1

    if best is None:
        fk = f_ref(xk)
        w = trigBaryWeights(xk)
        h = float((w @ fk) / (w @ sigma))
        best = (fk - h * sigma, xk.copy(), abs(h),
                float(np.max(np.abs(fk))))
    pk_b, xk_b, h_b, err_b = best

    def p_phys(x):
        return jnp.asarray(trigBary(
            to_ref(np.asarray(x)), pk_b, xk_b, (-np.pi, np.pi)))

    p = _cf(p_phys, domain=(a, b), trig=True, n=2 * m + 1)
    status = {"xk": jnp.asarray(from_ref(np.sort(xk_b)))}
    return p, err_b, status


def _trigremez_rational(f, m, n, max_iter, tol, a, b,
                        to_ref, from_ref, f_ref, normf):
    """Rational (m, n) trig Remez main loop (MATLAB trigremez).

    Returns ``(p, q, r_handle, err, status)`` mirroring MATLAB's
    ``[P, Q, R_HANDLE, ERR]``.

    Provenance
    ----------
    MATLAB source : @chebfun/trigremez.m (rational mode)
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.chebfun import chebfun as _cf

    N = 2 * (m + n) + 2
    xk = -np.pi + 2 * np.pi * np.arange(N) / N
    xo = xk.copy()
    best = None
    deltamin = np.inf
    delta, diffx = normf, 1.0
    it = 0
    while delta / normf > tol and it < max_iter and diffx > 0:
        fk = f_ref(xk)
        p_h, q_h, ac, bc, h = _trig_trial_rational(fk, xk, m, n)
        if h == 0:
            h = 1e-19

        def r_ref(y, _p=p_h, _q=q_h):
            return _p(y) / _q(y)

        yy = np.linspace(-np.pi, np.pi, max(8000, 80 * N),
                         endpoint=False)
        ee = f_ref(yy) - r_ref(yy)
        de = np.diff(ee)
        idx = np.where(np.sign(de[1:]) != np.sign(de[:-1]))[0] + 1
        if idx.size == 0:
            break
        # Parabolic refinement of each extremum (3-point fit), then a
        # re-evaluation -- sharpens the reference beyond grid spacing.
        hgrid = yy[1] - yy[0]
        rr = []
        for i in idx:
            y0, ym, yp = ee[i], ee[i - 1], ee[(i + 1) % len(ee)]
            denom = ym - 2 * y0 + yp
            shift = 0.5 * (ym - yp) / denom if denom != 0 else 0.0
            shift = float(np.clip(shift, -1.0, 1.0))
            rr.append(yy[i] + shift * hgrid)
        rr = np.asarray(rr)
        er = f_ref(rr) - r_ref(rr)
        s_pts, s_val = [rr[0]], [er[0]]
        for r_i, e_i in zip(rr[1:], er[1:]):
            if np.sign(e_i) == np.sign(s_val[-1]):
                if abs(e_i) > abs(s_val[-1]):
                    s_pts[-1], s_val[-1] = r_i, e_i
            else:
                s_pts.append(r_i)
                s_val.append(e_i)
        s_pts = np.array(s_pts)
        s_val = np.array(s_val)
        err = float(np.max(np.abs(s_val)))
        delta = err - abs(h)
        if delta < deltamin:
            deltamin = delta
            best = (p_h, q_h, ac, bc, abs(h), err, xk.copy())
        imax = int(np.argmax(np.abs(s_val)))
        d0 = max(imax - N + 1, 0)
        if len(s_pts) >= N:
            xk = np.sort(s_pts[d0: d0 + N])
        else:
            break
        diffx = float(np.max(np.abs(np.sort(xo) - np.sort(xk)))) \
            if len(xo) == len(xk) else 1.0
        xo = xk.copy()
        it += 1

    if best is None:
        raise RuntimeError("trigremez: rational iteration failed")
    p_h, q_h, ac, bc, h_b, err_b, xk_b = best

    p = _cf(jnp.asarray(ac), domain=(a, b), trig=True, coeffs=True)
    q = _cf(jnp.asarray(bc), domain=(a, b), trig=True, coeffs=True)

    def r_phys(x):
        y = to_ref(np.asarray(x))
        return jnp.asarray(p_h(y) / q_h(y))

    status = {"xk": jnp.asarray(from_ref(np.sort(xk_b)))}
    return p, q, r_phys, err_b, status
