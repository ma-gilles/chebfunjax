# uses-numpy: eigenvalue/FFT/Toeplitz manipulation is one-shot numpy
"""Caratheodory-Fejer and Chebyshev-Pade rational approximation
(MATLAB cf.m / chebpade.m).

Added by Claude Fable 5 (MISSING_FEATURES named-utilities sweep).
Single-column chebfuns only (no quasimatrix branch).

Provenance
----------
MATLAB source : @chebfun/cf.m, @chebfun/chebpade.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers (cf based on the v4 code by Joris Van Deun
    and L.N. Trefethen; chebpade based on the v4 code by Ricardo
    Pachon).
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
from scipy.linalg import hankel, toeplitz

__all__ = ["cf", "chebpade"]


def _coeffs(f) -> np.ndarray:
    """Ascending Chebyshev coefficients of a single-piece chebfun."""
    if len(f.funs) != 1:
        raise ValueError("requires a single-piece chebfun "
                         "(pass M to resample piecewise inputs)")
    return np.asarray(f.funs[0].tech.coeffs, dtype=float)


def _from_coeffs(ck, domain):
    """Build a chebfun from ascending Chebyshev coefficients."""
    from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, _Piece
    from chebfunjax.tech.chebtech import Chebtech2
    a, b = float(domain[0]), float(domain[1])
    tech = Chebtech2.from_coeffs(
        jnp.asarray(np.atleast_1d(np.asarray(ck, dtype=float))))
    piece = _Piece(tech=tech, interval=(a, b))
    return Chebfun(funs=[piece], domain=Domain((a, b)))


def _resample(f, npts: int):
    from chebfunjax.chebfun1d.chebfun import Domain
    a, b = float(f.domain.a), float(f.domain.b)
    k = np.arange(npts)
    x = np.cos(np.pi * k / (npts - 1))            # descending
    xp = a + (b - a) * (x + 1.0) / 2.0
    vals = np.asarray(f(jnp.asarray(xp)))
    # Chebyshev interpolation coefficients via DCT-I style FFT
    V = np.concatenate([vals, vals[-2:0:-1]])
    c = np.real(np.fft.fft(V)) / (npts - 1)
    c = c[:npts]
    c[0] /= 2.0
    c[-1] /= 2.0
    _ = Domain
    return c


def cf(f, m: int, n: int = 0, M: int | None = None):
    """Caratheodory-Fejer approximation of degree (m, n) (MATLAB cf):
    returns ``(p, q, r, s)`` -- numerator/denominator chebfuns, a
    handle ``r(x) = p(x)/q(x)``, and the approximate approximation
    error ``s``.

    Provenance
    ----------
    MATLAB source : @chebfun/cf.m
    Chebfun commit: 7574c77
    """
    dom = (float(f.domain.a), float(f.domain.b))
    if np.isinf(dom).any():
        raise ValueError("cf does not support unbounded domains")

    if len(f.funs) > 1:
        if M is None:
            raise ValueError("piecewise chebfuns require the M argument")
        a = _resample(f, M + 1)
    else:
        a = _coeffs(f)
    if M is None:
        M = len(a) - 1
    if M >= len(a):
        a = np.concatenate([a, np.zeros(M + 1 - len(a))])
    a = a[: M + 1]

    if np.iscomplexobj(a):
        a = np.real(a)

    # Trivial case
    if m >= M:
        q = _from_coeffs([1.0], dom)
        return f, q, (lambda x: f(x)), 0.0

    if n == 0:
        return _polynomial_cf(f, a, m, M, dom)
    return _rational_cf(f, a[::-1].copy(), m, n, M, dom)


def _polynomial_cf(f, a, m, M, dom):
    if m == M - 1:
        p = _from_coeffs(a[:M], dom)
        q = _from_coeffs([1.0], dom)
        return p, q, (lambda x: p(x)), abs(a[M])

    c = a[m + 1: M + 1]
    D, V = np.linalg.eigh(hankel(c))
    i = int(np.argmax(np.abs(D)))
    s = float(np.abs(D[i]))
    u = V[:, i]
    u1 = u[0]
    uu = u[1: M - m]

    b = c.copy()
    for _k in range(m, -m - 1, -1):
        b = np.concatenate(
            [[-(b[: M - m - 1] @ uu) / u1], b])
    bb = b[m: 2 * m + 1].copy()
    bb[1:] += b[m - 1:: -1][: m] if m > 0 else 0.0
    pk = a[: m + 1] - bb
    p = _from_coeffs(pk, dom)
    q = _from_coeffs([1.0], dom)
    return p, q, (lambda x: p(x)), s


def _get_block(a_rev, m, n, M):
    """Eigenvalue/block-structure helper (MATLAB getBlock).

    ``a_rev`` is the reversed coefficient vector (a_rev[0] = a_M,
    a_rev[-1] = 2*a_0)."""
    tol = 1e-14
    if n > M + m + 1:
        c = np.zeros(n - m - M - 1)
        nn = M + m + 1
    else:
        c = np.zeros(0)
        nn = n
    idx = np.abs(np.arange(m - nn + 1, M + 1))
    # MATLAB: a(M + 1 - abs(m-nn+1:M)) with 1-based a == a_rev
    c = np.concatenate([c, a_rev[M - idx]])
    D, V = np.linalg.eigh(hankel(c))
    order = np.argsort(-np.abs(D))
    S = np.abs(D[order])
    s = float(D[order[n]])
    u = V[:, order[n]]
    tmp = np.abs(S - abs(s)) < tol
    k = 0
    while k < n and tmp[n - k - 1]:
        k += 1
    ell = 0
    while (n + ell + 1) < len(tmp) and tmp[n + ell + 1]:
        ell += 1
    r_flag = (n + ell + 1) == len(tmp)
    return s, u, k, ell, r_flag


def _rational_cf(f, a_rev, m, n, M, dom):
    tolfft = 1e-14
    maxnfft = 2 ** 17
    a_rev = a_rev.copy()
    a_rev[-1] = 2.0 * a_rev[-1]
    vsc = max(float(np.max(np.abs(a_rev))), 1e-300)

    # Even/odd symmetry adjustments (MATLAB indices translated)
    if np.max(np.abs(a_rev[-2::-2])) / vsc < 1e-15:      # even
        if not (m % 2 or n % 2):
            m += 1
        elif (m % 2) and (n % 2):
            n -= 1
            if n == 0:
                return cf(f, m, n, M)
    elif np.max(np.abs(a_rev[::-2])) / vsc < 1e-15:      # odd
        if (m % 2) and not (n % 2):
            m += 1
        elif not (m % 2) and (n % 2):
            n -= 1
            if n == 0:
                return cf(f, m, n, M)

    s, u, k, ell, r_flag = _get_block(a_rev, m, n, M)
    if k > 0 or ell > 0:
        if r_flag:
            p, q, r = chebpade(f, m - k, n - k)
            return p, q, r, float(np.finfo(float).eps)
        nnew = n - k
        s, u, knew, lnew, _ = _get_block(a_rev, m + ell, nnew, M)
        if knew > 0 or lnew > 0:
            n = n + ell
            s, u, k, ell, _ = _get_block(a_rev, m - k, n, M)
        else:
            n = nnew

    # q from Laurent coefficients via FFT
    N = max(2 ** int(np.ceil(np.log2(len(u)))), 256)
    # derivative of the polynomial with ascending coefficients u
    ud = np.arange(1, len(u)) * u[1:]

    def _ac(NN):
        return np.fft.fft(
            np.conj(np.fft.fft(ud, NN) / np.fft.fft(u, NN))) / NN

    ac = _ac(N)
    act = np.zeros(N)
    while (np.max(np.abs(1 - act[-n - 1:-1] / ac[-n - 1:-1])) > tolfft
           and N < maxnfft):
        act = ac
        N *= 2
        ac = _ac(N)
    ac = np.real(ac)

    b = np.ones(n + 1)
    for j in range(1, n + 1):
        # MATLAB: b(j+1) = -(b(1:j) * ac(end-j:end-1)) / j
        b[j] = -(b[:j] @ ac[-j - 1:-1]) / j
    z = np.roots(b)
    if np.any(np.abs(z) > 1):
        warnings.warn("cf: ill-conditioning detected; results may be "
                      "inaccurate", RuntimeWarning, stacklevel=2)
    z = z[np.abs(z) < 1]
    rho = 1.0 / float(np.max(np.abs(z)))
    zj = 0.5 * (z + 1.0 / z)

    aa, bb_dom = dom

    def qfun(x):
        t = (2.0 * np.asarray(x, dtype=float) - (aa + bb_dom)) \
            / (bb_dom - aa)
        tt = t[..., None] - zj[None, :]
        return jnp.asarray(
            np.real(np.prod(tt, axis=-1) / np.prod(-zj)),
            dtype=jnp.float64)

    # q is a degree-n polynomial: build from exact interpolation
    npts = n + 1
    kk = np.arange(npts)
    xref = np.cos(np.pi * kk / max(npts - 1, 1))
    xq = aa + (bb_dom - aa) * (xref + 1) / 2.0
    qv = np.asarray(qfun(xq))
    if npts == 1:
        qc = qv
    else:
        Vd = np.concatenate([qv, qv[-2:0:-1]])
        qc = np.real(np.fft.fft(Vd)) / (npts - 1)
        qc = qc[:npts]
        qc[0] /= 2.0
        qc[-1] /= 2.0
    q = _from_coeffs(qc, dom)

    # Chebyshev coefficients of the approximation from the Blaschke
    # product, again via FFT
    v = u[::-1]
    N = max(2 ** int(np.ceil(np.log2(len(u)))), 256)

    def _ac2(NN):
        ph = np.exp(2j * np.pi * M * np.arange(NN) / NN)
        return np.fft.fft(
            ph * np.conj(np.fft.fft(u, NN) / np.fft.fft(v, NN))) / NN

    ac = _ac2(N)
    act = np.zeros(N)
    while (np.max(np.abs(1 - act[: m + 1] / ac[: m + 1])) > tolfft
           and np.max(np.abs(1 - act[-m:] / ac[-m:])) > tolfft
           and N < maxnfft) if m > 0 else False:
        act = ac
        N *= 2
        ac = _ac2(N)
    ac = s * np.real(ac)
    ct = a_rev[-1: -m - 2: -1] - ac[: m + 1]
    ct[0] -= ac[0]
    if m > 0:
        ct[1:] -= ac[-1: -m - 1: -1]
    s = abs(s)

    # Chebyshev coefficients of 1/q on its exact ellipse of analyticity
    nrecip = int(np.ceil(np.log(4 / np.finfo(float).eps / (rho - 1))
                         / np.log(rho)))
    nrecip = max(nrecip, 2 * m + 2)
    kk = np.arange(nrecip)
    xref = np.cos(np.pi * kk / (nrecip - 1))
    xq = aa + (bb_dom - aa) * (xref + 1) / 2.0
    rv = 1.0 / np.asarray(q(jnp.asarray(xq)))
    Vd = np.concatenate([rv, rv[-2:0:-1]])
    gam_full = np.real(np.fft.fft(Vd)) / (nrecip - 1)
    gam_full = gam_full[:nrecip]
    gam_full[0] /= 2.0
    gam_full[-1] /= 2.0

    gam = gam_full[::-1]                        # descending
    if len(gam) < 2 * m + 1:
        gam = np.concatenate(
            [np.zeros(2 * m + 1 - len(gam)), gam])
    gam = gam[-1: -2 * m - 2: -1]               # last 2m+1, reversed
    gam = gam.copy()
    gam[0] *= 2.0
    G_full = toeplitz(gam)

    if m == 0:
        bc = np.array([ct[0] / gam[0]])
    else:
        A = G_full[:m, :m]
        B = G_full[:m, m: m + 1]
        # MATLAB: C = gam(1:m, end:-1:m+2)
        C = G_full[:m, 2 * m: m: -1]
        G = A + C - 2.0 * (B @ B.T) / gam[0]
        rhs = -2.0 * (B[:, 0] * ct[0] / gam[0] - ct[m:0:-1])
        bcv = np.linalg.solve(G, rhs)
        bc0 = (ct[0] - B[:, 0] @ bcv) / gam[0]
        bc = np.concatenate([[bc0], bcv[::-1]])
    p = _from_coeffs(bc, dom)

    def r(x):
        return p(x) / q(x)

    return p, q, r, s


def chebpade(f, m: int, n: int, type: str = "clenshawlord",
             M: int = -1):
    """Chebyshev-Pade approximation of degree (m, n)
    (MATLAB chebpade): returns ``(p, q, r)``.

    Provenance
    ----------
    MATLAB source : @chebfun/chebpade.m
    Chebfun commit: 7574c77
    """
    if isinstance(type, (int, np.integer)):
        type, M = "clenshawlord", int(type)
    if type == "clenshawlord":
        return _chebpade_clenshaw_lord(f, m, n, M)
    if type == "maehly":
        return _chebpade_maehly(f, m, n)
    raise ValueError("type must be 'clenshawlord' or 'maehly'")


def _chebpade_clenshaw_lord(f, m, n, M):
    dom = (float(f.domain.a), float(f.domain.b))
    length = max(len(_coeffs(f)) if len(f.funs) == 1 else 0, 0)
    if M >= 0:
        c = _resample(f, M + 1)
    elif len(f.funs) > 1:
        raise ValueError("piecewise chebfuns require the M argument")
    else:
        c = _coeffs(f)
    c = np.asarray(c, dtype=complex) \
        if np.iscomplexobj(np.asarray(f.funs[0].tech.coeffs)) \
        else np.asarray(c, dtype=float)
    if np.iscomplexobj(np.asarray(f.funs[0].tech.coeffs)):
        c = np.asarray(f.funs[0].tech.coeffs)
    ell = max(m, n)
    if len(c) < m + 2 * n + 1:
        rng = np.random.default_rng(0)
        pad = np.finfo(float).eps * rng.standard_normal(
            m + 2 * n + 1 - len(c))
        c = np.concatenate([c, pad.astype(c.dtype)])
    c = c.copy()
    c[0] = 2 * c[0]

    if n > 0:
        top = c[np.abs(np.arange(m - n + 1, m + 1))]
        bot = c[np.arange(m, m + n)]
        rhs = c[np.arange(m + 1, m + n + 1)]
        beta = np.concatenate(
            [-np.linalg.solve(hankel(top, bot), rhs), [1.0]])[::-1]
    else:
        beta = np.array([1.0], dtype=c.dtype)

    c[0] = c[0] / 2
    alpha = np.convolve(c[: ell + 1], beta)[: ell + 1]

    D = np.zeros((ell + 1, ell + 1), dtype=c.dtype)
    D[:, : n + 1] = alpha[:, None] * beta[None, :]
    pk = np.zeros(m + 1, dtype=c.dtype)
    pk[0] = np.trace(D)
    for k in range(1, m + 1):
        pk[k] = np.trace(D, k) + np.trace(D, -k)

    qk = np.zeros(n + 1, dtype=c.dtype)
    for k in range(1, n + 2):
        u = beta[: n + 2 - k]
        v = beta[k - 1:]
        qk[k - 1] = u @ v
    pk = pk / qk[0]
    qk = 2 * qk / qk[0]
    qk[0] = 1.0
    _ = length
    p = _from_coeffs_any(pk, dom)
    q = _from_coeffs_any(qk, dom)
    return p, q, (lambda x: p(x) / q(x))


def _from_coeffs_any(ck, dom):
    from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, _Piece
    from chebfunjax.tech.chebtech import Chebtech2
    ck = np.atleast_1d(np.asarray(ck))
    tech = Chebtech2.from_coeffs(jnp.asarray(ck))
    piece = _Piece(tech=tech, interval=(float(dom[0]), float(dom[1])))
    return Chebfun(funs=[piece], domain=Domain((float(dom[0]),
                                                float(dom[1]))))


def _chebpade_maehly(f, m, n):
    dom = (float(f.domain.a), float(f.domain.b))
    tol = 1e-10
    a = _coeffs(f)
    if len(a) < m + 2 * n + 1:
        warnings.warn(
            f"chebpade: not enough coefficients for [{m}/{n}]; "
            "assuming the remainder are noise",
            RuntimeWarning, stacklevel=2)
        rng = np.random.default_rng(0)
        a = np.concatenate(
            [a, np.finfo(float).eps
             * rng.standard_normal(m + 2 * n + 1 - len(a))])

    row = np.arange(1, n + 1)
    col = np.arange(m + 1, m + n + 1)[:, None]
    D = a[col + row[None, :]] + a[np.abs(col - row[None, :])]
    if n > m:
        D = D + a[0] * np.diag(np.ones(n - m), m)[:n, :n]

    if np.linalg.matrix_rank(D, tol) < min(D.shape):
        if m > 1:
            warnings.warn(
                f"chebpade: singular matrix; computing "
                f"[{m - 1}/{n}]", RuntimeWarning, stacklevel=2)
            return chebpade(f, m - 1, n, "maehly")
        if n > 1:
            warnings.warn(
                f"chebpade: singular matrix; computing "
                f"[{m}/{n - 1}]", RuntimeWarning, stacklevel=2)
            return chebpade(f, m, n - 1, "maehly")
        raise ValueError("chebpade: singular matrix; cannot compute "
                         "[1/1] approximation")
    qk = np.concatenate(
        [[1.0], -np.linalg.solve(D, 2 * a[m + 1: m + n + 1])])

    colB = np.arange(1, m + 1)[:, None]
    B = a[colB + row[None, :]] + a[np.abs(colB - row[None, :])]
    dmin = min(m, n)
    for i in range(dmin):
        B[i, i] += a[0]
    B = np.vstack([a[1: n + 1][None, :], B]) if m >= 1 \
        else a[1: n + 1][None, :]
    if B.size == 0:
        pk = qk[0] * a[: m + 1]
    else:
        pk = 0.5 * (B @ qk[1: n + 1]) + qk[0] * a[: m + 1]

    p = _from_coeffs_any(pk, dom)
    q = _from_coeffs_any(qk, dom)
    return p, q, (lambda x: p(x) / q(x))
