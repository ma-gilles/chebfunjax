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

from chebfunjax.utils._trigremez_iteration_options import OPTION_OMITTED

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


class TrigRemezError(ValueError):
    """Public source diagnostic with a MATLAB-compatible identifier.

    Provenance
    ----------
    MATLAB source: @chebfun/trigremez.m, validation and parseInputs.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """

    def __init__(self, identifier, message):
        self.identifier = identifier
        super().__init__(f"{identifier}: {message}")


def trigremez(f, m: int | None = None, n: int | None = None, *options,
              max_iter: int = 100, tol: float | None = None,
              display=OPTION_OMITTED, plotfcns=OPTION_OMITTED, ax=None, hold=False):
    """Best real trigonometric polynomial or rational approximation.

    Polynomial outputs are ``(p, error, status)``; positive denominator
    degree returns ``(p, q, evaluator, error, status)``. Native guards run
    after the default L2 norm and before degree parsing, with source error
    identifiers on ``TrigRemezError``. An empty input returns itself.

    ``max_iter`` and ``tol`` adapt native ``'maxiter'`` and ``'tol'`` options.
    Native case-insensitive name/value pairs are also accepted. Display
    and plotfcns enable iteration presentation whenever supplied, matching
    native option presence semantics. ``ax`` and ``hold`` expose native
    current axes and hold state; plots use the source 5000-point error grid.

    Provenance
    ----------
    MATLAB source: @chebfun/trigremez.m.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """

    from chebfunjax.chebfun1d.linalg import Quasimatrix
    from chebfunjax.fun.deltafun import Deltafun
    from chebfunjax.fun.singfun import Singfun

    if not isinstance(f, Quasimatrix) and f.isempty():
        return f
    # Native computes the default L2 norm before validating the input.
    normf = float(f.norm()) or 1.0
    columns = f.cols if isinstance(f, Quasimatrix) else [f]
    if not all(column.isreal() for column in columns):
        raise TrigRemezError("CHEBFUN:CHEBFUN:trigremez:real",
                             "TRIGREMEZ only supports real-valued functions.")
    count = f.n_cols if isinstance(f, Quasimatrix) else f.n_columns
    if count > 1:
        raise TrigRemezError("CHEBFUN:CHEBFUN:trigremez:quasi",
                             "TRIGREMEZ does not currently support quasimatrices.")
    if any(isinstance(piece.tech, Singfun) for column in columns for piece in column.funs):
        raise TrigRemezError("CHEBFUN:CHEBFUN:trigremez:singularFunction",
                             "TRIGREMEZ does not currently support functions with singularities.")
    if bool(jnp.any(jnp.isinf(jnp.abs(jnp.asarray(f.domain.breakpoints))))):
        raise TrigRemezError("CHEBFUN:CHEBFUN:trigremez:unboundedDomain",
                             "Function must be defined on a bounded domain.")
    if any(column.deltas or any(isinstance(piece, Deltafun) for piece in column.funs)
           for column in columns):
        raise TrigRemezError("CHEBFUN:CHEBFUN:trigremez:deltaFunctions",
                             "Function must not have any delta functions.")
    # A one-column Python Quasimatrix is the native scalar-column adapter.
    if isinstance(f, Quasimatrix):
        f = columns[0]
    a, b = float(f.domain.a), float(f.domain.b)

    from chebfunjax.utils._trigremez_iteration_options import iteration_hook, parse_options

    if m is None:
        raise TypeError("trigremez requires a degree for nonempty inputs")
    n, max_iter, tol, show, plot = parse_options(m, n, options, max_iter, tol, display, plotfcns)
    m = int(m)
    n = None if n is None else int(n)

    def to_ref(x):       # [a, b] -> [-pi, pi]
        return -np.pi + 2 * np.pi * (np.asarray(x) - a) / (b - a)

    def from_ref(y):     # [-pi, pi] -> [a, b]
        y = np.asarray(y)
        return b * (y + np.pi) / (2 * np.pi) \
            + a * (np.pi - y) / (2 * np.pi)

    def f_ref(y):
        return np.asarray(f(jnp.asarray(from_ref(y))))

    # Native parseInputs: N = 2*(m+n)+2 and tol = 1e-15*(N/2+10).
    if tol is None:
        tol = 1e-15 * (m + (0 if n is None else n) + 11)
    callback = iteration_hook(show, plot, ax=ax, hold=hold)
    if n is not None and n > 0:
        arguments = (f, m, n, max_iter, tol, a, b, to_ref, from_ref, f_ref, normf)
        if callback is None:
            return _trigremez_rational(*arguments)
        return _trigremez_rational(*arguments, iteration_callback=callback)
    from chebfunjax.utils._trigremez_polynomial_source import polynomial_trigremez

    if callback is None:
        return polynomial_trigremez(f, m, max_iter, tol, normf)
    return polynomial_trigremez(f, m, max_iter, tol, normf, iteration_callback=callback)


def _trigremez_rational(f, m, n, max_iter, tol, a, b,
                        to_ref, from_ref, f_ref, normf, iteration_callback=None):
    """Rational (m, n) trig Remez main loop (MATLAB trigremez).

    Returns ``(p, q, r_handle, err, status)`` mirroring MATLAB's
    ``[P, Q, R_HANDLE, ERR]``.

    Provenance
    ----------
    MATLAB source : @chebfun/trigremez.m (rational mode)
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    from chebfunjax.utils._trigremez_rational_source import rational_trigremez

    if iteration_callback is None:
        return rational_trigremez(f, m, n, max_iter, tol, normf)
    return rational_trigremez(f, m, n, max_iter, tol, normf, iteration_callback=iteration_callback)
