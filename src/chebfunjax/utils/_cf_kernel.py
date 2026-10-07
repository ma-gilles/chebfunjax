"""Bounded JAX port of nondegenerate rational MATLAB ``cf`` branch.

Provenance
----------
MATLAB source : ``@chebfun/cf.m`` (``rationalCF`` / ``getBlock``)
Chebfun commit: ``7574c77``
Source file SHA256: ``4ea509555941fe0e7d03f85bd6a3fa7069beddedd96a592b60a891d17429d0e9``

Input ``coeffs`` are
the real Chebyshev coefficients ``a_0 .. a_M`` on the canonical interval
[-1, 1]. All numerical operations in this module use JAX arrays. The supported
branch is deliberately limited; see ``UnsupportedCFBranch``.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp


class UnsupportedCFBranch(NotImplementedError):
    """The requested source branch is not implemented by this JAX kernel."""


def _scan_block_source(repeated, n: int) -> tuple[int, int, bool]:
    """Translate the source getBlock k/l scans and rational flag exactly."""
    k = 0
    while k < n and repeated[n - k - 1]:
        k += 1
    ell = 0
    # MATLAB uses tmp(n+l+2), and tests that one-based position against length.
    while n + ell + 2 < len(repeated) and repeated[n + ell + 1]:
        ell += 1
    r_flag = (n + ell + 2) == len(repeated)
    return k, ell, r_flag


def _adjust_degrees_source(is_even: bool, is_odd: bool, m: int, n: int):
    """Translate rationalCF's even/odd degree adjustment predicates."""
    if is_even:
        if not (m % 2 or n % 2):
            return m + 1, n
        if (m % 2) and (n % 2):
            return m, n - 1
    elif is_odd:
        if (m % 2) and not (n % 2):
            return m + 1, n
        if not (m % 2) and (n % 2):
            return m, n - 1
    return m, n


def _cheb_values_to_coeffs(values):
    """JAX DCT-I for values at Chebyshev second-kind nodes."""
    count = values.shape[0]
    if count == 1:
        return values
    mirrored = jnp.concatenate((values[count - 1:0:-1], values[:count - 1]))
    coeffs = jnp.real(jnp.fft.ifft(mirrored))[:count]
    return coeffs.at[1:count - 1].multiply(2.0)


def chebfun_source_coefficients_jax(f, M: int | None = None):
    """Prepare CF coefficient input like ``cfOneColumn`` for scalar Chebfuns.

    A one-piece input reuses its stored Chebtech2 series (truncated/padded to
    M when supplied). Piecewise input requires M and performs the source global
    ``M+1``-point Chebfun resampling on the whole domain. Returns coefficients,
    original interval, source vertical scale, piece count and full original
    coefficients (needed because MATLAB's Cheb-Pade fallback omits M).
    """
    if getattr(f, "isTransposed", False):
        raise UnsupportedCFBranch("array-valued/transposed Chebfuns are unsupported")
    source_piece_count = len(f.funs)
    full_coefficients = (jnp.asarray(f.funs[0].tech.coeffs)
                         if source_piece_count == 1 else None)
    if source_piece_count != 1:
        if M is None:
            raise UnsupportedCFBranch(
                "piecewise CF input requires explicit source M resampling")
        if M < 1:
            raise ValueError("source M must be positive")
        a, b = float(f.domain.a), float(f.domain.b)
        theta = jnp.pi * jnp.arange(M + 1, dtype=jnp.float64) / M
        # Chebtech2.chebpts returns points in ascending order (-1 to 1).
        nodes = -jnp.cos(theta)
        x = a + (b - a) * (nodes + 1.0) / 2.0
        coefficients = _cheb_values_to_coeffs(jnp.asarray(f(x)))
    else:
        coefficients = jnp.asarray(f.funs[0].tech.coeffs)
        if M is not None and M < coefficients.shape[0] - 1:
            coefficients = coefficients[:M + 1]
        elif M is not None and M > coefficients.shape[0] - 1:
            coefficients = jnp.pad(coefficients,
                                   (0, M + 1 - coefficients.shape[0]))
        a, b = float(f.domain.a), float(f.domain.b)
    source_vscale = (jnp.asarray(f.vscale) if source_piece_count == 1 else None)
    return (coefficients, (a, b), source_vscale,
            source_piece_count, full_coefficients)


def _cheb_eval(coeffs, x):
    """Evaluate a Chebyshev series using the source T_k convention."""
    x = jnp.asarray(x)
    t0 = jnp.ones_like(x)
    if coeffs.shape[0] == 1:
        return coeffs[0] * t0
    t1 = x
    result = coeffs[0] * t0 + coeffs[1] * t1
    for k in range(2, coeffs.shape[0]):
        t0, t1 = t1, 2 * x * t1 - t0
        result = result + coeffs[k] * t1
    return result


def _denominator_coeffs_from_roots(roots):
    """Source root-product q, normalized by q(0)=1, as Cheb coefficients."""
    roots = jnp.asarray(roots)

    def q_eval(x):
        return jnp.real(jnp.prod(x[..., None] - roots, axis=-1) /
                        jnp.prod(-roots))

    degree = roots.shape[0]
    nodes = -jnp.cos(jnp.pi * jnp.arange(degree + 1) / degree)
    return _cheb_values_to_coeffs(q_eval(nodes))


def _hankel_from_vector(c):
    n = c.shape[0]
    padded = jnp.concatenate((c, jnp.zeros((n - 1,), dtype=c.dtype)))
    return padded[jnp.arange(n)[:, None] + jnp.arange(n)[None, :]]


def _top_abs_eigs_lobpcg_candidate(H, k: int, v0, *, max_iterations: int = 100):
    """Candidate JAX path for MATLAB's largest-magnitude symmetric eigpairs.

    Uses a symmetric lift to turn |lambda(H)| into positive algebraic
    eigenvalues for JAX 0.11 LOBPCG, then a projected Rayleigh-Ritz solve maps
    lifted vectors back to signed H eigenpairs. Existing dispatch selects this
    route for Hankel dimensions greater than 1024. This large-degree route
    remains unqualified against the source oracle; convergence and cluster
    behavior can differ from MATLAB ARPACK.
    """
    n = H.shape[0]
    if k <= 0 or 5 * k >= 2 * n:
        raise UnsupportedCFBranch("LOBPCG lifted problem violates 0 < 5*k < 2*N")

    def lifted_action(x):
        left, right = x[:n], x[n:]
        return jnp.concatenate((H @ right, H @ left), axis=0)

    # Start from the source deterministic v0, then form a Krylov block.
    seed = jnp.concatenate((jnp.asarray(v0), jnp.zeros_like(v0)))
    columns = [seed]
    for _ in range(1, k):
        columns.append(lifted_action(columns[-1]))
    start = jnp.stack(columns, axis=1)
    singular_values = jnp.linalg.svd(start, compute_uv=False)
    if bool(jax.device_get(singular_values[-1] <=
                           jnp.finfo(H.dtype).eps * singular_values[0])):
        raise UnsupportedCFBranch("source-v0 Krylov start lost rank")
    X = jnp.linalg.qr(start, mode="reduced")[0]

    # JAX 0.11 LOBPCG returns largest algebraic Ritz values. The symmetric
    # lift encodes the largest |lambda(H)| as its largest positive values.
    from jax.experimental.sparse.linalg import lobpcg_standard

    theta, vectors, iterations = lobpcg_standard(
        lifted_action, X, m=max_iterations, tol=jnp.finfo(H.dtype).eps)
    del iterations
    if bool(jax.device_get(jnp.any(theta <= 0))):
        raise UnsupportedCFBranch("LOBPCG did not resolve all positive lifted Ritz values")

    candidate_vectors = []
    for col in range(k):
        left = vectors[:n, col]
        right = vectors[n:, col]
        for candidate in (left + right, left - right):
            length = jnp.linalg.norm(candidate)
            if bool(jax.device_get(length >
                                   100 * jnp.finfo(H.dtype).eps)):
                candidate_vectors.append(candidate / length)
    if len(candidate_vectors) < k:
        raise UnsupportedCFBranch("lifted eigenspace did not recover k H vectors")
    span = jnp.stack(candidate_vectors, axis=1)
    Q, R = jnp.linalg.qr(span, mode="reduced")
    rank = jnp.linalg.svd(R, compute_uv=False)
    if bool(jax.device_get(rank[-1] <=
                           100 * jnp.finfo(H.dtype).eps * rank[0])):
        raise UnsupportedCFBranch("lifted-to-H Ritz vectors lost rank")
    projected = Q.T @ H @ Q
    signed, rotation = jnp.linalg.eigh(projected)
    order = jnp.argsort(-jnp.abs(signed))[:k]
    return signed[order], (Q @ rotation[:, order])


def _chebpade_clenshaw_lord_jax(coeffs, m: int, n: int):
    """Source Clenshaw-Lord branch, including epsilon-normal padding."""
    if m < 0 or n < 0:
        raise UnsupportedCFBranch("source Cheb-Pade received a negative reduced degree")
    c = jnp.asarray(coeffs)
    required = m + 2 * n + 1
    if c.shape[0] < required:
        from chebfunjax.utils._cf_padding import _epsilon_pad

        c = _epsilon_pad(c, required)
    c = c.at[0].multiply(2.0)
    if n > 0:
        top_idx = jnp.abs(jnp.arange(m - n + 1, m + 1))
        bot_idx = jnp.arange(m, m + n)
        rhs_idx = jnp.arange(m + 1, m + n + 1)
        top = c[top_idx]
        bot = c[bot_idx]
        rhs = c[rhs_idx]
        # MATLAB hankel(top, bot): first column is top, last row is bot.
        sums = jnp.arange(n)[:, None] + jnp.arange(n)[None, :]
        hankel = jnp.where(sums < n, top[jnp.minimum(sums, n - 1)],
                           bot[jnp.clip(sums - n + 1, 0, n - 1)])
        beta = jnp.concatenate((-jnp.linalg.solve(hankel, rhs),
                                jnp.ones((1,), c.dtype)))[::-1]
    else:
        beta = jnp.ones((1,), c.dtype)
    degree = max(m, n)
    c = c.at[0].multiply(0.5)
    alpha_full = jnp.convolve(c[:degree + 1], beta)
    alpha = alpha_full[:degree + 1]
    D = jnp.zeros((degree + 1, degree + 1), dtype=c.dtype)
    D = D.at[:, :n + 1].set(alpha[:, None] * beta[None, :])
    p_coeffs = []
    for k in range(m + 1):
        p_coeffs.append(jnp.trace(D) if k == 0 else
                        jnp.trace(D, offset=k) + jnp.trace(D, offset=-k))
    p_coeffs = jnp.stack(p_coeffs)
    q_coeffs = []
    for k in range(n + 1):
        left = beta[:n + 1 - k]
        right = beta[k:]
        q_coeffs.append(left @ right)
    q_coeffs = jnp.stack(q_coeffs)
    p_coeffs = p_coeffs / q_coeffs[0]
    q_coeffs = (2.0 * q_coeffs / q_coeffs[0]).at[0].set(1.0)
    return p_coeffs, q_coeffs


def _cf_polynomial_jax(coeffs, m: int):
    """Source rationalCF-to-polynomial reduction when symmetry sets n=0."""
    a = jnp.asarray(coeffs, dtype=jnp.float64)
    M = a.shape[0] - 1
    if m == M - 1:
        return a[:M], jnp.ones((1,), dtype=a.dtype), jnp.abs(a[M])
    c = a[m + 1:M + 1]
    H = _hankel_from_vector(c)
    if H.shape[0] > 1024:
        eigvals, eigvecs = _top_abs_eigs_lobpcg_candidate(
            H, 1, jnp.ones((H.shape[0],), dtype=H.dtype) / H.shape[0])
        s, u = jnp.abs(eigvals[0]), eigvecs[:, 0]
    else:
        eigvals, eigvecs = jnp.linalg.eigh(H)
        pick = jnp.argmax(jnp.abs(eigvals))
        s = jnp.abs(eigvals[pick])
        u = eigvecs[:, pick]
    u1 = u[0]
    # MATLAB uu = u(2:(M-m)); map the inclusive 1-based end to Python stop M-m.
    uu = u[1:M - m]
    b = c
    for _ in range(m, -m - 1, -1):
        new_head = -(b[:M - m - 1] @ uu) / u1
        b = jnp.concatenate((new_head[None], b))
    bb = b[m:2 * m + 1]
    if m:
        bb = bb.at[1:].add(b[m - 1::-1][:m])
    p_coeffs = a[:m + 1] - bb
    return p_coeffs, jnp.ones((1,), dtype=a.dtype), s


def _get_block_source(a_rev, m: int, n: int, M: int):
    """JAX translation of MATLAB ``getBlock`` for dense-size matrices.

    The ``l`` scan and ``rflag`` compare the original one-based position
    ``n+l+2`` against ``length(tmp)``. This is why the translated boundary
    condition includes ``+2`` rather than reusing the legacy P helper's
    ``+1`` comparison.
    """
    if n > M + m + 1:
        c = jnp.zeros((n - m - M - 1,), dtype=a_rev.dtype)
        nn = M + m + 1
    else:
        c = jnp.zeros((0,), dtype=a_rev.dtype)
        nn = n
    indices = jnp.abs(jnp.arange(m - nn + 1, M + 1))
    c = jnp.concatenate((c, a_rev[M - indices]))
    hankel = _hankel_from_vector(c)
    if hankel.shape[0] > 1024:
        eigvals, eigvecs = _top_abs_eigs_lobpcg_candidate(
            hankel, min(n + 10, c.shape[0]),
            jnp.ones((c.shape[0],), dtype=c.dtype) / c.shape[0])
        order = jnp.argsort(-jnp.abs(eigvals))
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
    else:
        eigvals, eigvecs = jnp.linalg.eigh(hankel)
        order = jnp.argsort(-jnp.abs(eigvals))
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
    if n >= eigvals.shape[0]:
        raise UnsupportedCFBranch("source getBlock selected eigenvalue index is unavailable")
    singular_order = jnp.abs(eigvals)
    s = eigvals[n]
    u = eigvecs[:, n]
    repeated = jnp.abs(singular_order - jnp.abs(s)) < 1e-14
    # Boolean conversion is control flow on small static metadata, not numeric
    # approximation arithmetic. The helper is not JIT-polymorphic in m/n.
    repeated_host = jax.device_get(repeated).tolist()
    k, ell, r_flag = _scan_block_source(repeated_host, n)
    return s, u, k, ell, r_flag


def cf_rational_small_jax(
    coeffs, m: int, n: int, *, source_vscale=None,
    source_piece_count: int = 1, source_full_coeffs=None,
):
    """Return numerator/denominator coefficients and the CF error estimate.

    The kernel handles real single-column input, source symmetry adjustment,
    polynomial degree reduction, selected repeated-block transitions, and
    Clenshaw-Lord fallback with epsilon-normal coefficient padding. Dense eigensolve
    is used for Hankel dimension <=1024; larger matrices use the unqualified
    lifted LOBPCG route. Missing branches raise ``UnsupportedCFBranch``,
    including degenerate reciprocal construction. Padding shares the advancing
    JAX normal stream with randnfun; MATLAB RNG bits are not reproduced.

    Provenance
    ----------
    MATLAB source : @chebfun/cf.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    a = jnp.asarray(coeffs)
    if bool(jnp.any(jnp.imag(a) != 0)):
        warnings.warn("CHEBFUN:CHEBFUN:cf:complex: Taking real part.",
                      RuntimeWarning, stacklevel=2)
    a = jnp.asarray(jnp.real(a), dtype=jnp.float64)
    if a.ndim != 1 or m < 0 or n < 0 or a.shape[0] < 2:
        raise ValueError("expected real 1-D Chebyshev coefficients and m>=0,n>=0")
    M = a.shape[0] - 1
    if m >= M:
        if source_piece_count != 1 or source_full_coeffs is None:
            if m == M and source_piece_count == 1:
                return a, jnp.ones((1,), dtype=a.dtype), jnp.asarray(0.0, dtype=a.dtype)
            raise UnsupportedCFBranch(
                "source trivial m>=M returns original Chebfun, not a coefficient subset")
        original = jnp.asarray(source_full_coeffs)
        return original, jnp.ones((1,), dtype=a.dtype), jnp.asarray(0.0, dtype=a.dtype)

    # MATLAB rationalCF reverses coefficients and doubles the T0 coefficient.
    a_rev = a[::-1].at[-1].multiply(2.0)
    vscale = (jnp.max(jnp.abs(a_rev))
              if source_vscale is None else jnp.asarray(source_vscale, dtype=a.dtype))
    symmetry_tol = jnp.finfo(a.dtype).eps
    is_even = jnp.max(jnp.abs(a_rev[-2::-2])) / vscale < symmetry_tol
    is_odd = False if is_even else (
        jnp.max(jnp.abs(a_rev[::-2])) / vscale < symmetry_tol)
    adjusted_m, adjusted_n = _adjust_degrees_source(
        bool(jax.device_get(is_even)), bool(jax.device_get(is_odd)), m, n)
    if (adjusted_m, adjusted_n) != (m, n):
        return cf_rational_small_jax(
            a, adjusted_m, adjusted_n, source_vscale=source_vscale,
            source_piece_count=source_piece_count,
            source_full_coeffs=source_full_coeffs)
    if n == 0:
        return _cf_polynomial_jax(a, m)

    s, u, k, ell, r_flag = _get_block_source(a_rev, m, n, M)
    if k > 0 or ell > 0:
        if r_flag:
            if source_piece_count != 1:
                raise UnsupportedCFBranch(
                    "source cf calls Cheb-Pade without M on piecewise input")
            pade_coeffs = (a if source_full_coeffs is None
                           else jnp.asarray(source_full_coeffs))
            return (*_chebpade_clenshaw_lord_jax(
                        pade_coeffs, m - k, n - k),
                    jnp.asarray(jnp.finfo(a.dtype).eps, dtype=a.dtype))
        n_new = n - k
        s, u, knew, lnew, _ = _get_block_source(a_rev, m + ell, n_new, M)
        if knew > 0 or lnew > 0:
            n = n + ell
            s, u, k, ell, _ = _get_block_source(a_rev, m - k, n, M)
        else:
            n = n_new
    if n == 0:
        raise UnsupportedCFBranch(
            "source rationalCF degree collapsed to zero after block transition")

    # Laurent coefficients for the denominator polynomial.
    nfft = max(2 ** int(jnp.ceil(jnp.log2(u.shape[0]))), 256)
    ud = jnp.arange(1, u.shape[0], dtype=u.dtype) * u[1:]

    def ac_den(length):
        return jnp.fft.fft(jnp.conj(jnp.fft.fft(ud, length) /
                                      jnp.fft.fft(u, length))) / length

    ac = ac_den(nfft)
    previous = jnp.zeros_like(ac)
    while nfft < 2**17 and jnp.max(jnp.abs(
            1.0 - previous[-n - 1:-1] / ac[-n - 1:-1])) > 1e-14:
        previous = ac
        nfft *= 2
        ac = ac_den(nfft)
    ac = jnp.real(ac)
    b = jnp.ones((n + 1,), dtype=ac.dtype)
    for j in range(1, n + 1):
        b = b.at[j].set(-(b[:j] @ ac[-j - 1:-1]) / j)

    # Roots of descending monic polynomial b, via its companion matrix.
    companion = jnp.zeros((n, n), dtype=b.dtype)
    companion = companion.at[0, :].set(-b[1:])
    if n > 1:
        companion = companion.at[1:, :-1].set(jnp.eye(n - 1, dtype=b.dtype))
    roots = jnp.linalg.eigvals(companion)
    if bool(jax.device_get(jnp.any(jnp.abs(roots) > 1))):
        warnings.warn("Ill-conditioning detected. Results may be inaccurate.",
                      RuntimeWarning, stacklevel=2)
    roots = roots[jnp.abs(roots) < 1]
    if roots.shape[0] == 0:
        raise UnsupportedCFBranch("source root filter left no denominator roots")
    rho = 1.0 / jnp.max(jnp.abs(roots))
    zj = 0.5 * (roots + 1.0 / roots)

    # MATLAB constructs q from a root-product Chebfun; DCT-I recovers its
    # degree-n Chebyshev series from values at second-kind points.
    q_coeffs = _denominator_coeffs_from_roots(zj)

    # Blaschke-product Laurent coefficients and source ct formula.
    v = u[::-1]
    nfft = max(2 ** int(jnp.ceil(jnp.log2(u.shape[0]))), 256)

    def ac_num(length):
        phase = jnp.exp(2j * jnp.pi * M * jnp.arange(length) / length)
        return jnp.fft.fft(phase * jnp.conj(
            jnp.fft.fft(u, length) / jnp.fft.fft(v, length))) / length

    ac = ac_num(nfft)
    previous = jnp.zeros_like(ac)
    while (m > 0 and nfft < 2**17 and
           jnp.max(jnp.abs(1.0 - previous[:m + 1] / ac[:m + 1])) > 1e-14 and
           jnp.max(jnp.abs(1.0 - previous[-m:] / ac[-m:])) > 1e-14):
        previous = ac
        nfft *= 2
        ac = ac_num(nfft)
    ac = s * jnp.real(ac)
    ct = a_rev[-1:-m - 2:-1] - ac[:m + 1]
    ct = ct.at[0].add(-ac[0])
    if m > 0:
        ct = ct.at[1:].add(-ac[-1:-m - 1:-1])
    s = jnp.abs(s)

    nrecip = int(jax.device_get(jnp.ceil(
        jnp.log(4.0 / jnp.finfo(a.dtype).eps / (rho - 1.0)) / jnp.log(rho))))
    if nrecip < 2:
        raise UnsupportedCFBranch("source reciprocal Chebfun resolution is degenerate")
    qrecip_nodes = -jnp.cos(jnp.pi * jnp.arange(nrecip) / (nrecip - 1))
    gamma = _cheb_values_to_coeffs(1.0 / _cheb_eval(q_coeffs, qrecip_nodes))
    # MATLAB flips coefficients, pads/truncates exactly 2*m+1 entries, and
    # doubles the first entry before constructing the symmetric Toeplitz matrix.
    gamma = gamma[::-1]
    if gamma.shape[0] < 2 * m + 1:
        gamma = jnp.concatenate((jnp.zeros((2 * m + 1 - gamma.shape[0],),
                                            dtype=gamma.dtype), gamma))
    gamma = gamma[-1:-2 * m - 2:-1].at[0].multiply(2.0)
    row = gamma
    toeplitz = jnp.stack([row[jnp.abs(jnp.arange(2 * m + 1) - i)]
                          for i in range(2 * m + 1)])
    A = toeplitz[:m, :m]
    B = toeplitz[:m, m:m + 1]
    C = toeplitz[:m, 2 * m:m:-1]
    G = A + C - 2.0 * (B @ B.T) / gamma[0]
    rhs = -2.0 * (B[:, 0] * ct[0] / gamma[0] - ct[m:0:-1])
    bcv = jnp.linalg.solve(G, rhs)
    bc0 = (ct[0] - B[:, 0] @ bcv) / gamma[0]
    p_coeffs = jnp.concatenate((bc0[None], bcv[::-1]))
    return p_coeffs, q_coeffs, s
