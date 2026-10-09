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

__all__ = ["trigpade", "trigremez"]


class TrigpadeError(ValueError):
    """Source diagnostic with a MATLAB-compatible identifier."""

    def __init__(self, identifier, message):
        self.identifier = identifier
        super().__init__(f"{identifier}: {message}")


def _trig_coeffs_ascending(f):
    """Source @chebfun/trigpade.m: obtain its public Fourier coefficients."""
    return jnp.asarray(f.trigcoeffs(), dtype=jnp.complex128)


def _trig_chebfun_from_ascending(c, domain):
    """Source coefficient constructor; storage is c[-N],...,c[N]."""
    from chebfunjax.chebfun1d.chebfun import chebfun

    return chebfun(jnp.atleast_1d(c), domain=domain, coeffs=True, trig=True)


def _laurent_approx(c, m, n, N, tol):
    """@chebfun/trigpade.m laurent_approx, including first null vector.

    MATLAB null uses max(size(A))*eps(norm(A)); singular vectors themselves
    can differ between LAPACK versions, especially in multidimensional null
    spaces. The source selects the first returned null vector.
    """
    rows = jnp.arange(n)[:, None]
    columns = jnp.arange(n + 1)[None, :]
    matrix = c[N + m + 1 + rows - columns]
    _, singular, vh = jnp.linalg.svd(matrix, full_matrices=True)
    threshold = max(matrix.shape) * jnp.spacing(singular[0])
    rank = int(jnp.sum(singular > threshold))
    b = jnp.conj(vh[rank])
    if float(jnp.abs(b[0])) < tol:
        raise TrigpadeError('CHEBFUN:TRIGPADE:laurent_approx',
                           'denominator zero at the origin detected')
    b = b / b[0]
    degree = max(m, n)
    col = c[N:N + degree + 1].at[0].multiply(.5)
    delta = jnp.arange(degree + 1)[:, None] - jnp.arange(degree + 1)[None, :]
    lower = jnp.where(delta >= 0, col[jnp.maximum(delta, 0)], 0)
    bb = jnp.pad(b, (0, degree + 1 - b.size))
    return lower @ bb, b


def _laurent_pade(c, m, n, tol):
    """Source Laurent positive/negative solves and centered zero padding."""
    c = jnp.ravel(jnp.asarray(c, dtype=jnp.complex128))
    N = (c.size - 1) // 2
    ap, bp = _laurent_approx(c, m, n, N, tol)
    reversed_c = c[::-1]
    if float(jnp.max(jnp.abs(c - jnp.conj(reversed_c)))) < 10 * tol:
        am, bm = jnp.conj(ap), jnp.conj(bp)
    else:
        am, bm = _laurent_approx(reversed_c, m, n, N, tol)
    return (jnp.pad(ap, (ap.size - 1, 0)),
            jnp.pad(bp, (bp.size - 1, 0)),
            jnp.pad(am, (am.size - 1, 0))[::-1],
            jnp.pad(bm, (bm.size - 1, 0))[::-1])


def _chop(c, tol):
    """Source chop_coeffs preserves the largest nonnegligible Fourier mode."""
    mid = (c.size - 1) // 2
    active = jnp.abs(c) > tol
    indices = jnp.arange(c.size)
    if not bool(jnp.any(active)):
        # Source find returns empty at both ends, hence empty support.
        return c[:0]
    first = int(jnp.min(jnp.where(active, indices, c.size)))
    last = int(jnp.max(jnp.where(active, indices, -1)))
    degree = max(mid - first, last - mid)
    return c[mid - degree:mid + degree + 1]


def _center_pad(v, L):
    """Source symmetric Laurent coefficient padding."""
    pad = L - (v.size - 1) // 2
    return jnp.pad(v, (pad, pad))


def trigpade(f, m=None, n=None):
    """Source Fourier-Pade approximation of a periodic Chebfun.

    Return ``(p, q, r, tn_p, td_p, tn_m, td_m)`` with
    ``p/q = tn_p/td_p + tn_m/td_m``. Empty input returns the empty
    Chebfun before inspecting degrees, matching the source early return.

    Provenance
    ----------
    MATLAB source : @chebfun/trigpade.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    from chebfunjax.chebfun1d.chebfun import chebfun
    from chebfunjax.tech.trigtech import Trigtech

    if f.isempty():
        return f
    if not all(isinstance(piece.tech, Trigtech) for piece in f.funs):
        raise TrigpadeError('CHEBFUN:CHEBFUN:trigpade:trig',
                           'Input chebfun F must have a periodic representation')
    if m is None or n is None:
        raise TypeError('trigpade requires numerator and denominator degrees')
    dom = (float(f.domain.a), float(f.domain.b))
    c = _trig_coeffs_ascending(f)
    if c.size % 2 != 1:
        raise ValueError('c must have odd length')
    N = (c.size - 1) // 2
    tol = float(100 * jnp.finfo(jnp.float64).eps * jnp.max(jnp.abs(c)))
    if n == 0:
        # The m<N source call deliberately supplies these entries as VALUES,
        # without its 'coeffs' flag; retain that public constructor behavior.
        p = f if m >= N else chebfun(c[N-m:N+m+1], domain=dom, trig=True)
        q = chebfun(1, domain=dom, trig=True)
        return p, q, lambda x: p(x), p / 2, q, p / 2, q

    padding = 2 * max(m, n) - N
    if padding > 0:
        c = jnp.pad(c, (padding, padding))
    ap, bp, am, bm = _laurent_pade(c, m, n, tol)
    tn_p, td_p, tn_m, td_m = (
        _trig_chebfun_from_ascending(v, dom) for v in (ap, bp, am, bm))
    L = (max(ap.size, am.size, bp.size, bm.size) - 1) // 2
    ap, bp, am, bm = (_center_pad(v, L) for v in (ap, bp, am, bm))
    pk = _chop(jnp.convolve(ap, bm) + jnp.convolve(am, bp), tol)
    qk = _chop(jnp.convolve(bm, bp), tol)
    p = _trig_chebfun_from_ascending(pk, dom).simplify()
    q = _trig_chebfun_from_ascending(qk, dom).simplify()
    if float(jnp.max(jnp.abs(c - jnp.conj(c[::-1])))) < tol:
        if float((p / q).imag().norm()) > tol:
            import warnings

            warnings.warn('CHEBFUN:CHEBFUN:trigpade:imag: imaginary part not negligible.',
                          RuntimeWarning, stacklevel=2)
        else:
            p, q = p.real(), q.real()

    def r(x):
        return p(x) / q(x)

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
        if float(np.max(np.abs(ee))) <= 1e-13 * normf:
            # f is itself a trigonometric polynomial of degree <= m:
            # the interpolant reproduces it and the error curve has no
            # sign structure to alternate on.
            best = (pk.copy(), xk.copy(), abs(h),
                    float(np.max(np.abs(ee))))
            break

        # candidate extrema: sign changes of the discrete derivative
        de = np.diff(ee)
        idx = np.where(np.sign(de[1:]) != np.sign(de[:-1]))[0] + 1
        rr = yy[idx]
        er = ee[idx]

        # keep alternating signs, largest magnitude per run
        # Alternation set.  An extremum where the error is at rounding
        # level (|e| <= 1e-13 ||f||) counts as a zero-sign point: at the
        # equispaced starting reference of a function whose maxima sit
        # exactly on it, the error there is +-1 ulp and its sign is not
        # information (MATLAB's chebfun-based error curve gives exact 0).
        _lvl = 1e-13 * normf

        def _sgn(v):
            return 0.0 if abs(v) <= _lvl else float(np.sign(v))
        s_pts, s_val = [rr[0]], [er[0]]
        for r_i, e_i in zip(rr[1:], er[1:]):
            if _sgn(e_i) == _sgn(s_val[-1]):
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
        # Alternation set.  An extremum where the error is at rounding
        # level (|e| <= 1e-13 ||f||) counts as a zero-sign point: at the
        # equispaced starting reference of a function whose maxima sit
        # exactly on it, the error there is +-1 ulp and its sign is not
        # information (MATLAB's chebfun-based error curve gives exact 0).
        _lvl = 1e-13 * normf

        def _sgn(v):
            return 0.0 if abs(v) <= _lvl else float(np.sign(v))
        s_pts, s_val = [rr[0]], [er[0]]
        for r_i, e_i in zip(rr[1:], er[1:]):
            if _sgn(e_i) == _sgn(s_val[-1]):
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
