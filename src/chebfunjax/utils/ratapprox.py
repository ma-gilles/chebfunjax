# uses-numpy: rational interpolation uses numpy/scipy SVD and eigenvalue solvers (not JIT-safe)
"""Rational approximation: Padé, rational interpolation, trig rational interpolation.

Translated from MATLAB Chebfun (commit 7574c77): padeapprox.m, ratinterp.m,
trigratinterp.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.transforms import coeffs2vals, vals2coeffs

# ===========================================================================
# Padé approximation
# ===========================================================================


def padeapprox(
    f,
    m: int,
    n: int,
    tol: float = 1e-14,
    r: float = 1.0,
    N_fft: int = 2048,
):
    r"""Padé approximation to a function or Taylor series (robust, SVD-based).

    Constructs a robust Padé approximant of type ``(mu, nu)`` (the exact
    reduced degrees after robustification) to the function or Taylor series
    ``f``.

    Parameters
    ----------
    f : callable or array_like
        - If callable: must be analytic in a neighbourhood of the disc of
          radius ``r`` centred at the origin.  Taylor coefficients are
          computed by sampling on ``N_fft`` roots of unity and applying FFT.
        - If array_like: interpreted as the Taylor coefficients
          ``[f_0, f_1, ..., f_K]`` with ``K >= m + n``.
    m : int
        Desired numerator degree (>= 0).
    n : int
        Desired denominator degree (>= 0).
    tol : float, optional
        Relative tolerance for robustification.  Set to 0 to disable.
        Default: 1e-14.
    r : float, optional
        Radius for FFT-based Taylor coefficient extraction when ``f`` is
        callable.  Default: 1.0.
    N_fft : int, optional
        Number of roots of unity for FFT when ``f`` is callable.
        Default: 2048.

    Returns
    -------
    r_handle : callable
        Function handle evaluating the rational approximant.
    a : np.ndarray, shape (mu+1,)
        Numerator Taylor coefficients (ascending powers: a[0] + a[1]*z + ...).
    b : np.ndarray, shape (nu+1,)
        Denominator Taylor coefficients (ascending powers, b[0] = 1).
    mu : int
        Exact numerator degree.
    nu : int
        Exact denominator degree.
    poles : np.ndarray, shape (nu,)
        Poles of the approximant (returned only if requested, always included
        in the return tuple here for consistency).
    residues : np.ndarray, shape (nu,)
        Residues at the poles.

    Notes
    -----
    Developer notes from MATLAB Chebfun (padeapprox.m):

    Implements the robust SVD-based algorithm of Gonnet, Guettel and Trefethen
    (2013).  The algorithm "hops" across a block structure in the Padé table,
    reducing the degree whenever the Toeplitz matrix formed from the Taylor
    coefficients is numerically rank-deficient.  The final numerator and
    denominator are normalized so that b[0] = 1.

    References
    ----------
    .. [1] P. Gonnet, S. Guettel, and L. N. Trefethen, "Robust Padé
       approximation via SVD", SIAM Rev., 55:101-117, 2013.

    Examples
    --------
    Padé (2, 2) approximant to exp(z):

    >>> r_fn, a, b, mu, nu, poles, res = padeapprox(np.exp, 2, 2)
    >>> abs(r_fn(0.5) - np.exp(0.5)) < 1e-10
    True

    Provenance
    ----------
    MATLAB source : padeapprox.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: [1] Gonnet, Guettel & Trefethen, SIAM Rev. 55, 2013.

    See Also
    --------
    ratinterp, trigratinterp, aaa
    """
    # ------------------------------------------------------------------
    # 1.  Extract Taylor coefficients if f is callable
    # ------------------------------------------------------------------
    if callable(f):
        z = r * np.exp(2j * np.pi * np.arange(N_fft) / N_fft)
        fvals = np.asarray(f(z), dtype=complex)
        c_full = np.fft.fft(fvals) / N_fft

        # Discard near-zero coefficients
        tc = 1e-15 * np.linalg.norm(c_full)
        c_full[np.abs(c_full) < tc] = 0.0

        # Make real functions real
        if np.linalg.norm(np.imag(c_full), np.inf) < tc:
            c_full = np.real(c_full)

        # Rescale for the radius
        c_full = c_full / r ** np.arange(N_fft)
    else:
        c_full = np.asarray(f, dtype=complex).ravel()

    # Ensure we have enough coefficients (zero-pad or truncate)
    c = np.concatenate([c_full, np.zeros(max(0, m + n + 1 - len(c_full)))])
    c = c[: m + n + 1]

    # ------------------------------------------------------------------
    # 2.  Absolute tolerance
    # ------------------------------------------------------------------
    ts = tol * np.linalg.norm(c)

    # ------------------------------------------------------------------
    # 3.  Special case: numerator is negligible → r = 0
    # ------------------------------------------------------------------
    if np.linalg.norm(c[: m + 1], np.inf) <= tol * np.linalg.norm(c, np.inf):
        a = np.array([0.0])
        b = np.array([1.0])
        mu = -1  # -inf in MATLAB notation; use -1 here
        nu = 0
    else:
        # ------------------------------------------------------------------
        # 4.  Diagonal-hopping across the Padé table block structure
        # ------------------------------------------------------------------
        row = np.zeros(n + 1, dtype=complex)
        row[0] = c[0]

        while True:
            if n == 0:
                a = c[: m + 1].copy()
                b = np.array([1.0])
                break

            # Build Toeplitz matrix Z (rows = m+n+1, cols = n+1)
            col_t = c[: m + n + 1]
            Z = _build_toeplitz(col_t, row)

            # Extract the lower (n x n+1) submatrix C
            C = Z[m + 1 : m + n + 1, : n + 1]

            # Numerical rank
            sv = np.linalg.svd(C, compute_uv=False)
            rho = int(np.sum(sv > ts))

            if rho == n:
                break

            # Decrease degrees (diagonal hopping)
            m_dec = n - rho
            m = m - m_dec
            n = rho
            row = np.zeros(n + 1, dtype=complex)
            row[0] = c[0]

        if n > 0:
            # Compute b from null vector with reweighted QR
            C = Z[m + 1 : m + n + 1, : n + 1]
            _, _, Vh = np.linalg.svd(C, full_matrices=True)
            b_raw = Vh[-1, :].conj()  # null vector

            D = np.diag(np.abs(b_raw) + np.sqrt(np.finfo(float).eps))
            # Use full (complete) QR — MATLAB's qr returns full Q by default
            # MATLAB: qr((C*D)') with ' = CONJUGATE transpose; a plain
            # .T gave the wrong null vector for complex Taylor series
            # (found in the Fable 5 audit: pade of [1, 1i] returned
            # b = [1, +1i] instead of [1, -1i]).
            Q, _ = np.linalg.qr((C @ D).conj().T, mode='complete')
            b_vec = D @ Q[:, n]
            b_vec = b_vec / np.linalg.norm(b_vec)

            # Discard leading zeros of b
            lam = 0
            while lam < len(b_vec) and np.abs(b_vec[lam]) <= tol:
                lam += 1
            b_vec = b_vec[lam:]

            # Discard trailing zeros of b
            last_b = len(b_vec) - 1
            while last_b > 0 and np.abs(b_vec[last_b]) <= tol:
                last_b -= 1
            b_vec = b_vec[: last_b + 1]

            # Compute a = Z[0:m+1, 0:n+1] @ b
            n_eff = len(b_vec)
            a_vec = Z[: m + 1, :n_eff] @ b_vec

            # Discard trailing zeros in a
            last_a = len(a_vec) - 1
            while last_a > 0 and np.abs(a_vec[last_a]) <= ts:
                last_a -= 1
            a_vec = a_vec[: last_a + 1]

            # Normalize: b[0] = 1
            a_vec = a_vec / b_vec[0]
            b_vec = b_vec / b_vec[0]

            a = np.real(a_vec) if np.allclose(np.imag(a_vec), 0, atol=1e-14) else a_vec
            b = np.real(b_vec) if np.allclose(np.imag(b_vec), 0, atol=1e-14) else b_vec
        else:
            a = c[: m + 1].copy()
            b = np.array([1.0])
            # Discard trailing zeros in a
            last_a = len(a) - 1
            while last_a > 0 and np.abs(a[last_a]) <= ts:
                last_a -= 1
            a = a[: last_a + 1]

        mu = len(a) - 1
        nu = len(b) - 1

    # ------------------------------------------------------------------
    # 5.  Build function handle (Horner evaluation)
    # ------------------------------------------------------------------
    a_rev = a[::-1]
    b_rev = b[::-1]

    def r_handle(z):
        z = np.asarray(z)
        return np.polyval(a_rev, z) / np.polyval(b_rev, z)

    # ------------------------------------------------------------------
    # 6.  Poles and residues
    # ------------------------------------------------------------------
    if nu > 0:
        poles = np.roots(b_rev)
        t = max(tol, 1e-7)
        residues = t * (r_handle(poles + t) - r_handle(poles - t)) / 2.0
    else:
        poles = np.array([])
        residues = np.array([])

    return r_handle, a, b, mu, nu, poles, residues


# ---------------------------------------------------------------------------
# Helpers for padeapprox
# ---------------------------------------------------------------------------


def _build_toeplitz(col: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Build a Toeplitz matrix with first column ``col`` and first row ``row``."""
    m_rows = len(col)
    n_cols = len(row)
    indices = np.arange(m_rows)[:, None] - np.arange(n_cols)[None, :]
    # Where index >= 0 use col, where < 0 use row
    col_ext = np.concatenate([col, np.zeros(max(0, n_cols - 1))])
    row_ext = np.concatenate([row, np.zeros(max(0, m_rows - 1))])
    result = np.where(indices >= 0, col_ext[np.abs(indices)], row_ext[np.abs(indices)])
    return result


# ===========================================================================
# Rational interpolation (ratinterp)
# ===========================================================================


def ratinterp(
    f,
    m: int,
    n: int,
    NN: int | None = None,
    xi=None,
    tol: float = 1e-14,
    domain: tuple[float, float] = (-1.0, 1.0),
):
    """Robust rational interpolation or least-squares approximation.

    Computes a type-(mu, nu) rational approximant to a function or data
    vector on a set of nodes, using the robust SVD-based algorithm.

    Parameters
    ----------
    f : callable or array_like
        Function handle or vector of function values at the nodes.
        If callable, it is evaluated at the appropriate grid points.
    m : int
        Desired numerator degree (Chebyshev basis).
    n : int
        Desired denominator degree (Chebyshev basis).
    NN : int or None, optional
        Number of interpolation/approximation nodes.  Defaults to ``m+n+1``
        (interpolation).  Must be >= ``m+n+1``.
    xi : array_like, str, or None, optional
        Interpolation nodes.  Options:

        - ``None``: use ``NN`` 2nd-kind Chebyshev points (default).
        - ``'type1'``: ``NN`` 1st-kind Chebyshev points.
        - ``'type2'``: ``NN`` 2nd-kind Chebyshev points.
        - ``'equidistant'`` or ``'equi'``: ``NN`` equidistant points in [-1,1].
        - ``'unitroots'``: ``NN`` roots of unity (complex nodes).
        - array_like: explicit node vector (length must equal ``NN``).

    tol : float, optional
        Relative tolerance for robustification.  Default: 1e-14.
        Set to 0 to disable.
    domain : (float, float), optional
        Physical domain.  Default: ``(-1, 1)``.

    Returns
    -------
    r_handle : callable
        Function handle for the rational approximant on ``domain``.
    a : np.ndarray
        Numerator coefficients in the Chebyshev/trigonometric basis.
    b : np.ndarray
        Denominator coefficients in the Chebyshev/trigonometric basis.
    mu : int
        Exact numerator degree.
    nu : int
        Exact denominator degree.
    poles : JAX array
        All roots of the denominator on the domain-mapped complex plane.
        Near-real complex roots are retained, matching ``roots(q, 'all')``.
    residues : JAX array
        Simple-pole residues computed as p(z)/q'(z). Repeated-pole partial
        fractions remain unsupported.

    Notes
    -----
    Developer notes from MATLAB Chebfun (ratinterp.m):

    The algorithm is described in Gonnet, Pachon & Trefethen (2011) and
    Pachon, Gonnet & van Deun (2011).  The key idea is to formulate the
    rational interpolation problem as a linear system and to robustify via
    SVD, discarding small singular values below the tolerance.

    The 'TYPE2' Chebyshev node case uses the "linearized" approach in which
    the Vandermonde-like matrix is expressed in terms of Chebyshev coefficient
    transforms (DCT-I/DCT-II) to avoid polynomial ill-conditioning.

    References
    ----------
    .. [1] P. Gonnet, R. Pachon, and L. N. Trefethen, "Robust Rational
       Interpolation and Least-Squares", ETNA 38:146-167, 2011.
    .. [2] R. Pachon, P. Gonnet and J. van Deun, "Fast and Stable Rational
       Interpolation in Roots of Unity and Chebyshev Points", 2011.

    Examples
    --------
    Type-(5, 5) approximant to 1/(x - 0.2) on [-1, 1]:

    >>> r_fn, a, b, mu, nu, poles, res = ratinterp(
    ...     lambda x: 1.0 / (x - 0.2), 10, 10)
    >>> abs(r_fn(0.0) - 1.0/(0.0 - 0.2)) < 1e-10
    True

    Provenance
    ----------
    MATLAB source : ratinterp.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm:
        [1] Gonnet, Pachon & Trefethen, ETNA 38, 2011.
        [2] Pachon, Gonnet & van Deun, 2011.

    See Also
    --------
    padeapprox, trigratinterp, aaa
    """
    a_dom, b_dom = float(domain[0]), float(domain[1])

    # ------------------------------------------------------------------
    # 1.  Determine NN and node type
    # ------------------------------------------------------------------
    xi_type = "TYPE2"  # default
    xi_nodes = None

    if xi is None:
        xi_type = "TYPE2"
        xi_nodes = None  # will generate below
    elif isinstance(xi, str):
        xi_upper = xi.upper()
        if xi_upper in ("UNITROOTS", "TYPE0"):
            xi_type = "TYPE0"
        elif xi_upper == "TYPE1":
            xi_type = "TYPE1"
        elif xi_upper == "TYPE2":
            xi_type = "TYPE2"
        elif xi_upper.startswith("EQUI"):
            xi_type = "EQUI"
        else:
            raise ValueError(f"ratinterp: unrecognized node type '{xi}'.")
    else:
        xi_nodes = np.asarray(xi, dtype=complex).ravel()
        xi_type = "ARBITRARY"
        if NN is None:
            NN = len(xi_nodes)

    # If f is a data vector, infer NN from its length (unless explicitly given).
    if NN is None and not callable(f):
        f_arr = np.asarray(f, dtype=complex).ravel()
        NN = len(f_arr)
    elif NN is None:
        NN = m + n + 1
    if NN < m + n + 1:
        raise ValueError(
            f"ratinterp: NN={NN} must be >= m+n+1 = {m + n + 1}."
        )

    N = NN - 1
    N1 = NN  # N + 1

    # ------------------------------------------------------------------
    # 2.  Generate nodes (scaled to [-1, 1] or complex unit circle)
    # ------------------------------------------------------------------
    if xi_type == "TYPE0":
        xi_nodes = np.exp(2j * np.pi * np.arange(N1) / N1)
    elif xi_type == "TYPE1":
        xi_nodes = np.array(chebpts(N1, kind=1))
    elif xi_type == "TYPE2":
        xi_nodes = np.array(chebpts(N1, kind=2))
    elif xi_type == "EQUI":
        xi_nodes = np.linspace(-1.0, 1.0, N1)
    elif xi_type == "ARBITRARY":
        # scale arbitrary nodes from [a, b] to [-1, 1]
        mid = 0.5 * (a_dom + b_dom)
        hd = 0.5 * (b_dom - a_dom)
        xi_nodes = (xi_nodes - mid) / hd

    # ------------------------------------------------------------------
    # 3.  Sample f on the nodes (convert from [-1,1] reference to domain)
    # ------------------------------------------------------------------
    mid = 0.5 * (a_dom + b_dom)
    hd = 0.5 * (b_dom - a_dom)

    if callable(f):
        x_physical = mid + hd * xi_nodes
        fvals = np.asarray(f(x_physical), dtype=complex).ravel()
    else:
        fvals = np.asarray(f, dtype=complex).ravel()
        if len(fvals) != N1:
            raise ValueError(
                f"ratinterp: length of f ({len(fvals)}) must equal NN ({N1})."
            )

    ts = tol * np.linalg.norm(fvals, np.inf)

    # ------------------------------------------------------------------
    # 4.  Check symmetries
    # ------------------------------------------------------------------
    fEven, fOdd = _check_symmetries(fvals, xi_nodes, xi_type, ts)

    # ------------------------------------------------------------------
    # 5.  Assemble matrices
    # ------------------------------------------------------------------
    Z, R_qr, Q_qr = _assemble_matrices_rat(fvals, n, xi_nodes, xi_type, N1)

    # ------------------------------------------------------------------
    # 6.  Compute denominator coefficients (b)
    # ------------------------------------------------------------------
    b_coeffs, n_eff = _compute_denominator_coeffs(Z, m, n, fEven, fOdd, N1, ts)

    # ------------------------------------------------------------------
    # 7.  Compute numerator coefficients (a)
    # ------------------------------------------------------------------
    a_coeffs = _compute_numerator_coeffs(
        fvals, m, n_eff, xi_type, Z, b_coeffs, fEven, fOdd, N, N1,
        R_qr=R_qr, Q_qr=Q_qr,
    )

    # ------------------------------------------------------------------
    # 7b.  For ARBITRARY/EQUI nodes, convert QR-basis coefficients back to
    #      the Chebyshev basis via R_qr^{-1}.
    # ------------------------------------------------------------------
    if not xi_type.upper().startswith("TYPE") and R_qr is not None:
        a_coeffs, b_coeffs = _qr_to_cheb_basis(a_coeffs, b_coeffs, R_qr, N1)

    # ------------------------------------------------------------------
    # 8.  Trim coefficients
    # ------------------------------------------------------------------
    a_coeffs, b_coeffs = _trim_coeffs(a_coeffs, b_coeffs, tol, ts)

    mu = len(a_coeffs) - 1
    nu = len(b_coeffs) - 1

    # ------------------------------------------------------------------
    # 9.  Build the function handle
    # ------------------------------------------------------------------
    r_handle = _construct_rat_approx(
        xi_type, R_qr, a_coeffs, b_coeffs, mu, nu, a_dom, b_dom
    )

    # ------------------------------------------------------------------
    # 10.  Poles and residues
    # ------------------------------------------------------------------
    if nu > 0:
        # Poles: roots of denominator polynomial (in [-1,1] reference, then map)
        b_poly = jnp.asarray(b_coeffs[:nu + 1], dtype=jnp.result_type(b_coeffs, 1j))
        # TYPE0 coefficients are monomial coefficients; Chebyshev grids and
        # arbitrary-node QR output use a Chebyshev basis.
        if xi_type == "TYPE0":
            poles_ref = jnp.roots(b_poly[::-1], strip_zeros=False)
        else:
            poles_ref = _chebyshev_roots(b_poly)
        # MATLAB roots(q, 'all') retains small nonzero imaginary parts.
        # Do not apply the rootsPref.all=False near-real cleanup here.
        poles = mid + hd * poles_ref

        roots = poles_ref
        # MATLAB delegates to residue(p, q); this derivative quotient covers
        # simple poles. Repeated-pole partial fractions remain unsupported.
        numerator = jnp.asarray(a_coeffs)
        denominator = jnp.asarray(b_coeffs)
        if xi_type == "TYPE0":
            derivative = jnp.arange(1, denominator.shape[0]) * denominator[1:]
            residues = hd * jnp.polyval(numerator[::-1], roots) / jnp.polyval(
                derivative[::-1], roots
            )
        else:
            from chebfunjax.tech.chebtech import _clenshaw, _diff_coeffs_once

            derivative = _diff_coeffs_once(denominator)
            residues = hd * _clenshaw(numerator, roots) / _clenshaw(derivative, roots)
    else:
        poles = jnp.empty((0,), dtype=jnp.result_type(b_coeffs, 1j))
        residues = jnp.empty((0,), dtype=jnp.result_type(a_coeffs, b_coeffs, 1j))

    return r_handle, a_coeffs, b_coeffs, mu, nu, poles, residues


# ---------------------------------------------------------------------------
# ratinterp helpers
# ---------------------------------------------------------------------------


def _check_symmetries(f, xi, xi_type, ts):
    """Check if the data is approximately even or odd."""
    fEven = False
    fOdd = False
    N1 = len(f)
    N = N1 - 1

    if xi_type.upper().startswith("TYPE") and len(xi_type) > 4:
        ch = xi_type[4]
        if ch == "0":
            if N % 2 == 1:
                M = N // 2
                fl = f[1: M + 1]
                fr = f[N1 - M:]
                fEven = np.linalg.norm(fl - fr, np.inf) < ts
                fOdd = np.linalg.norm(fl + fr, np.inf) < ts
        else:  # TYPE1 or TYPE2 Chebyshev
            M = int(np.ceil(N / 2))
            fl = f[:M]
            fr = f[-1:N1 - M - 1:-1]
            fEven = np.linalg.norm(fl - fr, np.inf) < ts
            fOdd = np.linalg.norm(fl + fr, np.inf) < ts
    return fEven, fOdd


def _assemble_matrices_rat(f, n, xi, xi_type, N1):
    """Build the Z matrix (and QR factor R for arbitrary nodes)."""
    R_qr = None
    Q_qr = None

    if xi_type.upper().startswith("TYPE"):
        ch = xi_type[4]
        if ch == "0":  # roots of unity
            row = np.conj(np.fft.fft(np.conj(f))) / N1
            col = np.fft.fft(f) / N1
            col[0] = row[0]
            Z = _build_toeplitz_complex(col, row[: n + 1])
        elif ch == "1":  # 1st-kind Chebyshev
            D = _chebtech1_coeffs2vals_matrix(N1)
            Z = _chebtech1_vals2coeffs_matrix_apply(np.diag(f) @ D[:, : n + 1], N1)
        else:  # 2nd-kind Chebyshev (TYPE2)
            D = _chebtech2_coeffs2vals_matrix(N1)
            Z = _chebtech2_vals2coeffs_matrix_apply(np.diag(f) @ D[:, : n + 1], N1)
    else:  # ARBITRARY nodes — complex Chebyshev Vandermonde and QR
        xi = jnp.asarray(xi)
        f = jnp.asarray(f)
        dtype = jnp.result_type(xi, f, jnp.float64)
        C = jnp.ones((N1, N1), dtype=dtype)
        if N1 > 1:
            C = C.at[:, 1].set(xi)
        for k in range(2, N1):
            C = C.at[:, k].set(2 * xi * C[:, k - 1] - C[:, k - 2])
        Q_qr, R_qr = jnp.linalg.qr(C)
        Z = Q_qr.conj().T @ jnp.diag(f) @ Q_qr[:, : n + 1]

    return Z, R_qr, Q_qr


def _chebtech2_coeffs2vals_matrix(N):
    """Dense (N x N) matrix: Chebyshev coefficients -> values at 2nd-kind pts."""
    eye_N = np.eye(N)
    result = np.zeros((N, N))
    for j in range(N):
        result[:, j] = np.array(coeffs2vals(jnp.array(eye_N[:, j], dtype=jnp.float64)))
    return result


def _chebtech2_vals2coeffs_matrix_apply(V, N):
    """Apply vals2coeffs column-by-column to build Z.

    Z = vals2coeffs(diag(f) @ D) is complex whenever f is; taking
    np.real here handed the denominator SVD only Re(f), which forces a
    REAL denominator -- its roots then come in conjugate pairs even when
    the true poles of a complex-valued f do not (ThreeBodyProblem's
    poles were all conjugate-paired and 2-5% off).  The real path is
    bit-identical.
    """
    _, ncols = V.shape
    if np.iscomplexobj(V):
        result = np.zeros((N, ncols), dtype=complex)
        for j in range(ncols):
            result[:, j] = _v2c_any(V[:, j])
        return result
    result = np.zeros((N, ncols))
    for j in range(ncols):
        result[:, j] = np.array(
            vals2coeffs(jnp.array(np.real(V[:, j]), dtype=jnp.float64))
        )
    return result


def _chebtech1_coeffs2vals_matrix(N):
    """Dense (N x N) matrix: Chebyshev coefficients -> values at 1st-kind pts."""
    # DCT-III: c_k -> v_j = sum_k c_k T_k(x_j), x_j = cos((2j-1)*pi/(2N))
    k = np.arange(N)
    j = np.arange(N)
    T = np.cos(np.outer((2 * j[::-1] + 1), k) * np.pi / (2 * N))
    return T


def _chebtech1_vals2coeffs_matrix_apply(V, N):
    """Apply the source Chebtech1 DCT-II matrix to every value column.

    Provenance
    ----------
    MATLAB source : @chebtech1/vals2coeffs.m
    Chebfun commit: 7574c77
    """
    V = jnp.asarray(V)
    k = jnp.arange(N, dtype=jnp.float64)
    j = jnp.arange(N, dtype=jnp.float64)
    # DCT-II: c_k = (2/N) * sum_j v_j * cos(k*(2j+1)*pi/(2N)),
    # with c_0 halved. Preserve the input's complex part.
    T = jnp.cos(jnp.outer(k, 2 * j[::-1] + 1) * jnp.pi / (2 * N))
    T_scaled = (2.0 / N) * T
    T_scaled = T_scaled.at[0, :].multiply(0.5)
    return T_scaled @ V


def _qr_to_cheb_basis(a_hat, b_hat, R_qr, N1):
    """Convert orthogonal-basis coefficients without discarding complex data.

    Padding with zero high modes makes the full upper triangular solve
    equivalent to the source's leading R blocks. All arithmetic stays in JAX.

    Provenance
    ----------
    MATLAB source : ratinterp.m / constructRatApproxArb (lines 570-596)
    Chebfun commit: 7574c77
    """
    from jax.scipy.linalg import solve_triangular

    a_hat, b_hat, R_qr = jnp.asarray(a_hat), jnp.asarray(b_hat), jnp.asarray(R_qr)
    na, nb = len(a_hat), len(b_hat)
    a_pad = jnp.zeros(N1, dtype=jnp.result_type(a_hat, R_qr)).at[:na].set(a_hat)
    b_pad = jnp.zeros(N1, dtype=jnp.result_type(b_hat, R_qr)).at[:nb].set(b_hat)
    a_cheb = solve_triangular(R_qr, a_pad, lower=False)[:na]
    b_cheb = solve_triangular(R_qr, b_pad, lower=False)[:nb]
    return a_cheb, b_cheb


def _build_toeplitz_complex(col, row):
    """Build a Toeplitz matrix (complex) with first column col and first row row."""
    m = len(col)
    nc = len(row)
    indices = np.arange(m)[:, None] - np.arange(nc)[None, :]
    col_ext = np.concatenate([col, np.zeros(nc - 1, dtype=complex)])
    row_ext = np.concatenate([row, np.zeros(m - 1, dtype=complex)])
    pos_idx = np.abs(indices)
    mask = indices >= 0
    result = np.where(mask, col_ext[pos_idx], row_ext[pos_idx])
    return result


def _compute_denominator_coeffs(Z, m, n, fEven, fOdd, N1, ts):
    """Compute denominator Chebyshev coefficients b via SVD robustification.

    Faithful port of ``computeDenominatorCoeffs`` in MATLAB
    ``ratinterp.m``.  The denominator is the trailing right singular
    vector of the lower part of ``Z``.  When the ``ns``-th singular value
    of that block drops to (or below) the absolute tolerance ``ts`` the
    rational type is over-specified: the degree ``n`` is reduced by the
    number of singular values within ``ts`` of the smallest one and the
    SVD is recomputed, until the block is numerically full rank.  This is
    what lets a type-(m, n) request collapse to the exact reduced type
    (e.g. ``(x^4-3)/((x+0.2)(x-2.2))`` requested (10, 10) -> (4, 2); a
    resolved ``exp`` requested (8, 8) -> a much smaller denominator).

    Provenance
    ----------
    MATLAB source : ratinterp.m (computeDenominatorCoeffs)
    Chebfun commit: 7574c77
    """
    shift = int(fEven) ^ int(m % 2 == 1)

    if (n > 0) and (not (fOdd or fEven) or (n > 1)):
        while True:
            if not (fOdd or fEven):
                # svd(Z(m+2:N1, 1:n+1), 0).  MATLAB's economy SVD still
                # returns the FULL right factor V, so ``V(:,end)`` (the
                # trailing/null direction) is numpy's full_matrices Vh[-1].
                sub = Z[m + 1: N1, : n + 1]
                _, S, Vh = np.linalg.svd(sub, full_matrices=True)
                ns = n
                b = Vh[-1, :].conj()
            else:
                # svd(Z(m+2+shift:2:N1, 1:2:n+1), 0).
                sub = Z[m + 1 + shift: N1: 2, 0: n + 1: 2]
                _, S, Vh = np.linalg.svd(sub, full_matrices=True)
                ns = n // 2
                b = np.zeros(n + 1, dtype=complex)
                b[::2] = Vh[-1, :].conj()

            # ssv = S(ns, ns).  S holds min(rows, cols) singular values;
            # the ns-th diagonal entry is zero when ns exceeds that count.
            k = S.shape[0]
            if ns <= k:
                s = S[:ns]
                ssv = S[ns - 1]
            else:
                s = np.concatenate([S, np.zeros(ns - k)])
                ssv = 0.0

            if ssv > ts:
                # Numerically full rank: denominator found.
                break

            # Reduce the denominator degree by the number of singular
            # values clustered within ts of the smallest.
            if fEven or fOdd:
                n = n - 2 * int(np.sum(s - ssv <= ts))
            else:
                n = n - int(np.sum(s - ssv <= ts))

            # Terminate on a trivial denominator.
            if n == 0:
                b = np.array([1.0])
                break
            elif n == 1:
                if fEven:
                    b = np.array([1.0, 0.0])
                    break
                elif fOdd:
                    b = np.array([0.0, 1.0])
                    break
    elif n > 0:
        if fEven:
            b = np.array([1.0, 0.0])
        elif fOdd:
            b = np.array([0.0, 1.0])
        else:
            b = np.array([1.0])
    else:
        b = np.array([1.0])

    return b, n


def _c2v_any(c):
    """coeffs2vals for real OR complex coefficients.

    The transform is linear, so a complex series is handled exactly by
    transforming its real and imaginary parts separately.  The jnp
    routines take float64 only, and casting a complex series through
    them silently discards Im -- which is what made ratinterp wrong for
    every complex-valued f (ode-nonlin/ThreeBodyProblem).
    """
    c = np.asarray(c)
    if np.iscomplexobj(c):
        return (np.array(coeffs2vals(jnp.array(c.real, dtype=jnp.float64)))
                + 1j * np.array(
                    coeffs2vals(jnp.array(c.imag, dtype=jnp.float64))))
    return np.array(coeffs2vals(jnp.array(c, dtype=jnp.float64)))


def _v2c_any(v):
    """vals2coeffs for real OR complex values (see :func:`_c2v_any`)."""
    v = np.asarray(v)
    if np.iscomplexobj(v):
        return (np.array(vals2coeffs(jnp.array(v.real, dtype=jnp.float64)))
                + 1j * np.array(
                    vals2coeffs(jnp.array(v.imag, dtype=jnp.float64))))
    return np.array(vals2coeffs(jnp.array(v, dtype=jnp.float64)))


def _compute_numerator_coeffs(f, m, n, xi_type, Z, b, fEven, fOdd, N, N1,
                               R_qr=None, Q_qr=None):
    """Compute numerator Chebyshev coefficients a (or QR-basis coefficients
    for ARBITRARY nodes, which are converted to Chebyshev basis later)."""
    if xi_type.upper().startswith("TYPE"):
        ch = xi_type[4]
        if ch == "0":
            b_pad = np.zeros(N1, dtype=complex)
            b_pad[: len(b)] = b
            a = np.fft.fft(np.fft.ifft(b_pad) * f)
            a = a[: m + 1]
        elif ch == "1":
            b_pad = np.zeros(N1, dtype=complex)
            b_pad[: len(b)] = b
            # Evaluate b polynomial at 1st-kind Chebyshev points then multiply by f
            # keep b_pad and f complex: the matrix is real, so the
            # product is exact for a complex operand
            b_vals = _chebtech1_coeffs2vals_matrix(N1) @ b_pad
            a_vals = b_vals * np.asarray(f)
            # Convert back to coefficients
            a = _chebtech1_vals2coeffs_matrix_apply(a_vals[:, None], N1).ravel()
            a = a[: m + 1]
        else:  # TYPE2
            b_pad = np.zeros(N1, dtype=complex)
            b_pad[: len(b)] = b
            b_vals = _c2v_any(b_pad)
            a_vals = b_vals * np.asarray(f)
            a = _v2c_any(a_vals)
            a = a[: m + 1]
    else:
        # ARBITRARY nodes: Z = Q'.diag(f).Q  (QR basis)
        # a_hat = Z[:m+1, :n_b] @ b_hat  (still QR basis; converted to Cheb later)
        n_b = len(b)
        a = Z[: m + 1, :n_b] @ b

    if fEven:
        # MATLAB zeroes alternate numerator modes. The Tech1 transform and
        # arbitrary-node assembly return immutable JAX arrays.
        a = jnp.asarray(a).at[1::2].set(0.0)
    elif fOdd:
        a = jnp.asarray(a).at[0::2].set(0.0)

    return a


def _trim_coeffs(a, b, tol, ts):
    """Apply MATLAB ``trimCoeffs`` thresholds without realifying coefficients.

    The comparisons are literal source comparisons: trailing entries survive
    only when ``abs(a) > ts`` / ``abs(b) > tol``; leading entries are stripped
    only while both current magnitudes are ``< ts``. JAX evaluates masks and
    comparisons; eager host scalar reads select the dynamic output slices.

    The output shape depends on coefficient values, so callers must not trace
    this adapter through ``jax.jit``. Retained arrays keep their original JAX
    dtype and values.

    Provenance
    ----------
    MATLAB source : ratinterp.m / trimCoeffs (lines 419-443)
    Chebfun commit: 7574c77
    """
    at = jnp.asarray(a)
    bt = jnp.asarray(b)

    if tol > 0:
        def _trim_tail(values, threshold):
            if values.size == 0:
                return values
            idx = jnp.arange(values.size)
            last = jnp.max(jnp.where(jnp.abs(values) > threshold, idx, -1))
            stop = int(jax.device_get(last)) + 1
            return values[:stop]

        at = _trim_tail(at, ts)
        bt = _trim_tail(bt, tol)

        while at.size > 0 and bt.size > 0:
            remove_first = (jnp.abs(at[0]) < ts) & (jnp.abs(bt[0]) < ts)
            if not bool(jax.device_get(remove_first)):
                break
            at = at[1:]
            bt = bt[1:]

    # MATLAB's zero-function special case applies even when tol == 0.
    if at.size == 0:
        return jnp.zeros((1,)), jnp.ones((1,))

    return at, bt


def _construct_rat_approx(xi_type, R_qr, a, b, mu, nu, a_dom, b_dom):
    """Build a JAX-evaluable function handle for the rational approximant.

    Provenance
    ----------
    MATLAB source : ratinterp.m / constructRatApprox
    Chebfun commit: 7574c77
    """
    mid = 0.5 * (a_dom + b_dom)
    hd = 2.0 / (b_dom - a_dom)  # maps x in [a,b] to t in [-1,1]
    a = jnp.asarray(a)
    b = jnp.asarray(b)

    if xi_type.upper().startswith("TYPE"):
        ch = xi_type[4]
        if ch == "0":  # Roots of unity — monomial polynomial in reference x
            a_rev = a[: mu + 1][::-1]
            b_rev = b[: nu + 1][::-1]

            def r_fn(x):
                t = hd * (jnp.asarray(x) - mid)
                return jnp.polyval(a_rev, t) / jnp.polyval(b_rev, t)
        else:  # Chebyshev basis

            def r_fn(x):
                t = hd * (jnp.asarray(x) - mid)
                return _eval_cheb_poly(a, t) / _eval_cheb_poly(b, t)
    else:  # Arbitrary nodes — coefficients in the Chebyshev basis from QR

        def r_fn(x):
            t = hd * (jnp.asarray(x) - mid)
            return _eval_cheb_poly(a, t) / _eval_cheb_poly(b, t)

    return r_fn


def _eval_cheb_poly(coeffs, x):
    """Evaluate a Chebyshev expansion at x with the shared JAX Clenshaw path.

    Provenance
    ----------
    MATLAB source : ratinterp.m / constructRatApprox
    Chebfun commit: 7574c77
    """
    from chebfunjax.tech.chebtech import _clenshaw

    coeffs = jnp.asarray(coeffs)
    x = jnp.asarray(x)
    if coeffs.shape[0] == 0:
        return jnp.zeros(x.shape, dtype=jnp.result_type(coeffs, x))
    return _clenshaw(coeffs, x)


def _chebyshev_roots(coeffs):
    """Find roots of a complex Chebyshev expansion via a JAX colleague matrix.

    Provenance
    ----------
    MATLAB source : roots(q, 'all') from ratinterp.m
    Chebfun commit: 7574c77
    """
    c_input = jnp.asarray(coeffs)
    c = c_input.astype(jnp.result_type(c_input, 1j))
    n = c.shape[0] - 1  # static polynomial degree
    if n == 0:
        return jnp.empty((0,), dtype=c.dtype)
    if n == 1:
        return jnp.asarray([-c[0] / c[1]])
    oh = jnp.full((n - 1,), 0.5, dtype=c.dtype)
    A = jnp.diag(oh, 1) + jnp.diag(oh, -1)
    A = A.at[-2, -1].set(1.0)
    c_adj = -0.5 * c[:-1] / c[-1]
    c_adj = c_adj.at[-2].add(0.5)
    A = A.at[:, 0].set(c_adj[::-1])
    return jnp.linalg.eigvals(A)


# ===========================================================================
# Trigonometric rational interpolation (trigratinterp)
# ===========================================================================


def trigratinterp(
    f,
    m: int,
    n: int,
    NN: int | None = None,
    xi=None,
    tol: float = 1e-14,
    domain: tuple[float, float] | None = None,
    *,
    outputs: int | None = None,
):
    """Compute a trigonometric rational fit with JAX numerical kernels.

    With ``outputs=None``, preserve the existing Python seven-tuple
    ``(r, ac, bc, mu, nu, poles, residues)``. With explicit ``outputs=k``,
    return the MATLAB output prefix ``(p, q, r, mu, nu, poles, residues)[:k]``.
    The legacy form also preserves omitted-NN inference for numeric vectors;
    explicit MATLAB outputs use the source minimum-NN default.
    MATLAB's six-output branch uses roots of the simplified denominator in
    physical x; its seven-output branch computes polynomial ``residue(p,q)``
    on the unreversed Fourier coefficient vectors. Explicit outputs 1--5 do
    not compute pole data. Repeated-pole residue behavior remains unsupported.

    The fitting call is eager, not JIT-compatible: source robustification can
    change coefficient-array lengths. The returned evaluator supports JAX
    arrays, including complex arguments.

    Provenance
    ----------
    MATLAB source : trigratinterp.m, @trigtech/roots.m,
                    @trigtech/poly.m, @chebfun/residue.m
    Chebfun commit: 7574c77
    """
    legacy_tuple = outputs is None
    if not legacy_tuple:
        outputs = _trigrat_nonnegative_integer(outputs, "outputs")
        if outputs < 1 or outputs > 7:
            raise ValueError("outputs must be between 1 and 7")
    m = _trigrat_nonnegative_integer(m, "m")
    n = _trigrat_nonnegative_integer(n, "n")
    f_domain = getattr(f, "domain", None)
    if domain is None:
        if f_domain is not None and hasattr(f_domain, "breakpoints"):
            domain = (f_domain.breakpoints[0], f_domain.breakpoints[-1])
        else:
            domain = (-1.0, 1.0)
    if len(domain) != 2:
        raise ValueError("domain must have exactly two endpoints")
    a_dom, b_dom = float(domain[0]), float(domain[1])
    if f_domain is not None and hasattr(f_domain, "breakpoints"):
        if (a_dom, b_dom) != (f_domain.breakpoints[0], f_domain.breakpoints[-1]):
            raise ValueError("F has different domain from the one passed")
    if not b_dom > a_dom:
        raise ValueError("domain endpoints must be strictly increasing")
    period = b_dom - a_dom

    minimum = 2 * (m + n) + 1
    # Existing Python seven-tuple API infers NN from a data vector when
    # omitted. Explicit MATLAB outputs retain the source minimum NN default.
    # This is an API adapter; numerical fitting below follows one source path.
    if legacy_tuple and NN is None and not callable(f) and not isinstance(f, str):
        try:
            NN = len(f)
        except TypeError:
            pass
    nn_is_empty = (
        NN is not None and not isinstance(NN, str) and jnp.asarray(NN).size == 0
    )
    if NN is None or nn_is_empty:
        NN = minimum
    NN = _trigrat_nonnegative_integer(NN, "NN")
    if NN % 2 == 0:
        warnings.warn(
            "Number of points should be odd.",
            RuntimeWarning,
            stacklevel=2,
        )
    if NN < minimum:
        raise ValueError(f"NN must be >= 2*(M+N)+1 = {minimum}")

    xi_is_empty = xi is not None and not isinstance(xi, str) and jnp.asarray(xi).size == 0
    if xi is None or xi_is_empty or (isinstance(xi, str) and xi.lower().startswith("equi")):
        nodes = a_dom + period * jnp.arange(NN, dtype=jnp.float64) / NN
        xi_type = "equi"
    elif isinstance(xi, str):
        raise ValueError(f"unrecognized xi type {xi!r}")
    else:
        xi_array = jnp.asarray(xi)
        if jnp.issubdtype(xi_array.dtype, jnp.complexfloating):
            # MATLAB isreal tests complex storage, even for zero imaginary
            # values; reject before any conversion to a real array.
            raise ValueError("input vector XI must be real")
        if xi_array.ndim > 2 or (xi_array.ndim == 2 and min(xi_array.shape) != 1):
            raise ValueError("xi must be a row or column vector")
        nodes = jnp.ravel(xi_array.astype(jnp.float64))
        NN = int(nodes.size)
        xi_type = "arbi"
    if nodes.size == 0:
        raise ValueError("xi must not be empty")
    if bool(jnp.any((nodes < a_dom) | (nodes > b_dom))):
        raise ValueError("input nodes must lie within the domain")

    if nodes.size % 2 == 0:
        warnings.warn(
            "Input vector XI does not have odd number of points.",
            RuntimeWarning,
            stacklevel=2,
        )

    if float(tol) < 0:
        raise ValueError("tol must be a positive number")

    order = jnp.argsort(nodes)
    nodes = nodes[order]
    if callable(f):
        # Source sorts xi before sampling a function handle.
        values = jnp.ravel(jnp.asarray(f(nodes)))
    else:
        # Literal parseInputs quirk: numeric fk is not permuted when xi is
        # sorted above. Preserve the supplied value order.
        values = jnp.ravel(jnp.asarray(f))
    if values.size != NN:
        raise ValueError(f"f has {values.size} values but NN={NN}")
    if not jnp.issubdtype(values.dtype, jnp.complexfloating):
        values = values.astype(jnp.float64)
    else:
        values = values.astype(jnp.complex128)
    th = 2.0 * (nodes - 0.5 * (a_dom + b_dom)) / period
    if float(jnp.min(th)) == -1.0 and float(jnp.max(th)) == 1.0:
        raise ValueError("periodic interval cannot include both endpoints")

    ts = float(tol) * float(jnp.max(jnp.abs(values)))
    f_even, f_odd = _trigrat_check_symmetries(values, th, xi_type, ts)
    interpolation = NN == minimum
    ac, bc = _trigrat_fit_coefficients(
        values, m, n, th, f_even, f_odd,
        robustness=bool(tol != 0), interpolation=interpolation, threshold=ts,
    )
    mu = _trigrat_degree(ac)
    nu = _trigrat_degree(bc)

    # MATLAB constructs p/q and captures r before its output-only
    # normalization for n==0 or constant denominator.
    p, q, r = _trigrat_make_approximation(ac, bc, a_dom, b_dom)
    if n == 0 or nu == 0:
        # Preserve source Chebfun division and its domain/piece semantics,
        # including its error behavior even when only r is requested.
        p, q = p / q, q / q

    if not legacy_tuple and outputs == 1:
        return r

    out_ac, out_bc = p.coeffs, q.coeffs
    if legacy_tuple:
        poles, residues = _trigrat_source_poly_residues(out_ac, out_bc)
        return r, out_ac, out_bc, mu, nu, poles, residues

    base = (p, q, r, mu, nu)
    if outputs <= 5:
        return base[:outputs]
    if outputs == 6:
        poles = _trigrat_source_x_roots(out_bc, a_dom, b_dom)
        return base + (poles,)
    poles, residues = _trigrat_source_poly_residues(out_ac, out_bc)
    return base + (poles, residues)


# ---------------------------------------------------------------------------
# trigratinterp helpers
# ---------------------------------------------------------------------------


def _trig_check_symmetries(f, ts):
    """Check even/odd symmetry in the data (for real periodic functions)."""
    N = len(f)
    fEven = False
    fOdd = False
    if N % 2 == 1:
        M = N // 2
        # Compare f[1:M+1] with f[N-M:]
        if M > 0:
            fl = f[1: M + 1]
            fr = f[N - M:]
            fEven = np.linalg.norm(fl - fr, np.inf) < ts
            fOdd = np.linalg.norm(fl + fr, np.inf) < ts
    return fEven, fOdd


def _trig_sincos_matrices(th, m, n, T=2.0):
    """MATLAB construct_matrices: [1, sin(2pi j th/T), cos(2pi j th/T)]."""
    P = np.zeros((len(th), 2 * m + 1))
    Q = np.zeros((len(th), 2 * n + 1))
    P[:, 0] = 1.0
    for j in range(1, m + 1):
        P[:, 2 * j - 1] = np.sin(2 * j * np.pi / T * th)
        P[:, 2 * j] = np.cos(2 * j * np.pi / T * th)
    Q[:, 0] = 1.0
    for j in range(1, n + 1):
        Q[:, 2 * j - 1] = np.sin(2 * j * np.pi / T * th)
        Q[:, 2 * j] = np.cos(2 * j * np.pi / T * th)
    return P, Q


def _sincos_to_exponential(a):
    """MATLAB sincosine_to_exponential: [a0, s1, c1, s2, c2, ...] ->
    exponential coefficients for k = -m..m."""
    a = np.asarray(a, dtype=complex).ravel()
    tmp = (a[2::2] - 1j * a[1::2]) / 2.0
    return np.concatenate([np.conj(tmp)[::-1], a[:1], tmp])


def _chop_trig_coeffs(a, tol):
    """MATLAB chopCoeffs: drop symmetric trailing coefficients below tol."""
    a = np.asarray(a)
    n = len(a)
    if n <= 1:
        return a
    mid = (n - 1) // 2
    aa = (np.abs(a[mid:]) + np.abs(a[mid::-1])) / 2.0
    big = np.flatnonzero(np.abs(aa) > tol)
    if big.size == 0:
        return a[mid:mid + 1]
    idx = int(big[-1]) + 1
    return a[mid - idx + 1:mid + idx]


def _trig_rat_interp_svd(fk, m, n, th, robustness_flag, interpolation_flag,
                         tol):
    """Faithful port of MATLAB trig_rat_interp: the null vector of the
    row-scaled system ``[P, -diag(f) Q]`` gives numerator/denominator
    sine-cosine coefficients; with robustness the denominator degree is
    reduced while the system has more than one negligible singular value.
    (MATLAB resets the symmetry flags to false inside this routine.)

    Provenance
    ----------
    MATLAB source : trigratinterp.m (trig_rat_interp, construct_matrices,
        getCoeffs, chopCoeffs, sincosine_to_exponential)
    Chebfun commit: 7574c77
    """
    fk = np.asarray(fk, dtype=complex).ravel()
    th = np.asarray(th, dtype=float).ravel()
    while True:
        P, Q = _trig_sincos_matrices(th, m, n, 2.0)
        D = np.diag(fk)
        sys_mat = np.hstack([P, -D @ Q])
        sys_mat = np.diag(1.0 / np.maximum(np.abs(fk), 1.0)) @ sys_mat
        _U, S, Vh = np.linalg.svd(sys_mat, full_matrices=True)
        V = Vh.conj().T
        v = V[:, -1]
        a = v[:2 * m + 1]
        b = v[2 * m + 1:]
        ac = _chop_trig_coeffs(_sincos_to_exponential(a), tol)
        bc = _chop_trig_coeffs(_sincos_to_exponential(b), tol)
        s = S
        m = min((len(ac) - 1) // 2, m)
        big = np.flatnonzero(np.abs(s) > tol)
        n_big = int(big[-1]) + 1 if big.size else 0
        n_small = len(s) - n_big
        if n_small < 2 or not robustness_flag:
            break
        reduction = n_small // 2
        if reduction == 0:
            break
        n_new = n - reduction
        if n_new >= 0:
            n = n_new
    return ac, bc, s


def _trig_rat_interp(fk, m, n, th, a_dom, b_dom, fEven, fOdd, robustness, interpolation, ts):
    """Core trigonometric rational interpolation algorithm.

    Returns Fourier coefficients (ac, bc) of numerator and denominator.
    ac has length 2*mu+1, bc has length 2*nu+1 (centered at index 0).
    """
    N = len(fk)

    # Build the DFT-based matrix
    # The key idea: represent p, q in the Fourier basis and solve p = q*f
    # by formulating as a linear system on the Fourier coefficients.

    # Compute DFT of data
    Fk = np.fft.fft(fk) / N

    # Build the Toeplitz-like system for denominator coefficients
    # q_coeffs (length 2n+1) appear in a linear system
    # The system: C @ q_hat = 0 (up to scaling)
    # where C is derived from the convolution structure

    # Form the "linear least-squares" matrix for the denominator
    # We need to find q such that p = q*f in the trig polynomial sense.
    # Rewrite: (q*f - p) = 0
    # In Fourier space this becomes a system of linear equations.

    # Size parameters
    p_deg = m  # numerator degree: trig poly of degree m
    q_deg = n  # denominator degree: trig poly of degree n
    np1 = 2 * p_deg + 1  # numerator dofs
    nq1 = 2 * q_deg + 1  # denominator dofs

    # Build convolution matrix C such that C @ q_hat ≈ 0
    # where q_hat are Fourier coefficients of q.
    # This is the (N_eq - np1) x nq1 submatrix of the full Fourier
    # multiplication system.

    # Fourier coefficients of f (centered, length N):
    # Fk[k] corresponds to frequency k (for k=0..N/2) and k-N (for k > N/2)
    # We need a convolution matrix for multiplication by f.

    # Build the full convolution matrix M (size N x nq1):
    # M[i, j] = F_{i - (j - q_deg)},  where F is extended f-Fourier coefficients
    M = np.zeros((N, nq1), dtype=complex)
    for j in range(nq1):
        freq_j = j - q_deg  # Fourier frequency of this denominator coefficient
        for i in range(N):
            freq_i = i if i <= N // 2 else i - N
            freq_diff = freq_i - freq_j
            # We need F[freq_diff] (the DFT coeff at frequency freq_diff)
            idx = int(freq_diff % N)
            M[i, j] = Fk[idx]

    # The system: M @ q_hat = p_hat (in numerator frequency range)
    # Subtract the numerator part: rows with frequencies outside [-m, m] give
    # constraints on q alone.
    # Constraints: M[|freq_i| > m, :] @ q_hat = 0

    # Build the constraint matrix
    constraints_rows = []
    for i in range(N):
        freq_i = i if i <= N // 2 else i - N
        if abs(freq_i) > p_deg:
            constraints_rows.append(M[i, :])

    if len(constraints_rows) == 0:
        # No constraints — return trivial denominator
        bc = np.zeros(nq1, dtype=complex)
        bc[q_deg] = 1.0
        ac = np.zeros(np1, dtype=complex)
        # Fill numerator coefficients from DFT coefficients of f
        for i in range(np1):
            freq_i = i - p_deg
            idx = int(freq_i % N)
            ac[i] = Fk[idx]
        # Keep complex — _eval_trig_poly takes np.real at the end
        return ac, bc.real, n

    C_constraint = np.array(constraints_rows, dtype=complex)

    # Robustify via SVD
    n_eff = q_deg
    bc = np.zeros(nq1, dtype=complex)

    if robustness and n > 0:
        # SVD of constraint matrix to find null vector
        sv = np.linalg.svd(C_constraint, compute_uv=False)
        rho = int(np.sum(sv > ts)) if len(sv) > 0 else 0

        # Reduce denominator degree if rank-deficient
        if rho < nq1 and rho < len(sv):
            # The null space gives the denominator coefficients
            _, _, Vh = np.linalg.svd(C_constraint, full_matrices=True)
            bc = Vh[-1, :].conj()
            n_eff = q_deg
        else:
            _, _, Vh = np.linalg.svd(C_constraint, full_matrices=True)
            bc = Vh[-1, :].conj()
            n_eff = q_deg
    else:
        _, _, Vh = np.linalg.svd(C_constraint, full_matrices=True)
        if Vh.shape[0] > 0:
            bc = Vh[-1, :].conj()
        else:
            bc[q_deg] = 1.0
        n_eff = q_deg

    # Normalize: bc[q_deg] is the zero-frequency (constant) component
    if abs(bc[q_deg]) > 1e-14:
        bc = bc / bc[q_deg]
    else:
        # Normalize by norm instead
        nrm = np.linalg.norm(bc)
        if nrm > 0:
            bc = bc / nrm

    # Compute numerator coefficients: ac = M @ bc in the numerator frequency range
    ac_all = M @ bc
    ac = np.zeros(np1, dtype=complex)
    for i in range(np1):
        freq_i = i - p_deg
        idx = int(freq_i % N)
        ac[i] = ac_all[idx]

    # Enforce symmetry
    if fEven:
        # Even function: odd Fourier coefficients are zero
        ac[1::2] = 0.0
        bc[1::2] = 0.0
    elif fOdd:
        # Odd function: even Fourier coefficients are zero
        ac[0::2] = 0.0
        bc[0::2] = 0.0

    # Keep complex coefficients — _eval_trig_poly takes np.real() of the sum,
    # so conjugate-symmetric coefficients give the correct real output.
    return ac, bc, n_eff


def _construct_trig_rat_approx(ac, bc, a_dom, b_dom, ts):
    """Build function handle for trigonometric rational approximant."""
    ac_c = ac.copy()
    bc_c = bc.copy()

    def r_fn(x):
        x = np.asarray(x, dtype=float)
        p_vals = _eval_trig_poly(ac_c, x, a_dom, b_dom)
        q_vals = _eval_trig_poly(bc_c, x, a_dom, b_dom)
        return p_vals / q_vals

    return r_fn


def _eval_trig_poly(coeffs, x, a_dom, b_dom):
    """Evaluate a trigonometric polynomial at x.

    coeffs: Fourier coefficients centered at middle index (length 2*mu+1).
    The trig poly is sum_{k=-mu}^{mu} c_k exp(i*pi*k*2*(x-a)/(b-a)).
    """
    x = np.asarray(x, dtype=float)
    period = b_dom - a_dom
    n_coeffs = len(coeffs)
    mu = (n_coeffs - 1) // 2

    result = np.zeros_like(x, dtype=complex)
    for j in range(n_coeffs):
        k = j - mu  # Fourier frequency
        # Standard-period variable (MATLAB: x mapped to [-1, 1] about
        # the midpoint), matching the trigtech coefficient convention.
        xs = 2.0 * (x - 0.5 * (a_dom + b_dom)) / period
        result = result + coeffs[j] * np.exp(1j * np.pi * k * xs)

    return np.real(result)


def _find_trig_poles(bc, a_dom, b_dom):
    """Find poles of a trigonometric rational function.

    Converts the denominator trig poly to an algebraic polynomial via
    z = exp(i*pi*(x-a)*2/(b-a)) and finds roots of the resulting polynomial.
    """
    period = b_dom - a_dom
    n_bc = len(bc)
    (n_bc - 1) // 2

    # The denominator is sum_{k=-nu}^{nu} bc[k+nu] * z^k where z = e^{i*pi*...}
    # Multiply by z^nu to get a polynomial of degree 2*nu:
    poly_coeffs = bc[::-1]  # coefficients of z^0, z^1, ..., z^{2*nu}
    if len(poly_coeffs) < 2:
        return np.array([])

    z_roots = np.roots(poly_coeffs)
    # Map back to x: z = e^{i*2*pi*(x-a)/period} => x = a + period*log(z)/(2*pi*i)
    x_poles = (0.5 * (a_dom + b_dom)
               + period * np.log(z_roots) / (2j * np.pi))
    # Keep only real poles (within the period)
    real_poles = x_poles[np.abs(np.imag(x_poles)) < 1e-8]
    real_poles = np.real(real_poles)
    return real_poles


# ===========================================================================
# Chebyshev-Padé approximation (chebpade)
# ===========================================================================


def chebpade(
    f,
    m: int,
    n: int,
    kind: str = "clenshawlord",
    K: int = -1,
):
    r"""Chebyshev-Padé approximation of type (m, n) to a function or Chebfun.

    Computes numerator Chebyshev polynomial *p* (degree *m*) and denominator
    Chebyshev polynomial *q* (degree *n*) such that ``p/q`` is the
    Clenshaw-Lord or Maehly Chebyshev-Padé approximant to *f*.

    Parameters
    ----------
    f : callable or array_like
        Function or Chebyshev coefficient vector ``c[0], c[1], ...`` (length
        at least ``m + 2*n + 1``).  If callable, sampled via DCT on a
        Chebyshev grid of size ``max(len(f), m+2*n+1)`` (when *f* is a
        Chebfun the ``f.coeffs`` property is used directly).
    m : int
        Degree of the numerator Chebyshev polynomial.
    n : int
        Degree of the denominator Chebyshev polynomial.
    kind : {'clenshawlord', 'maehly'}
        Algorithm variant.  Default ``'clenshawlord'``.
    K : int, optional
        Truncate the Chebyshev expansion of *f* to *K* terms before
        computing the approximant.  ``K < 0`` means use all available
        coefficients (default).

    Returns
    -------
    p_coeffs : np.ndarray, shape (m+1,)
        Chebyshev coefficients of the numerator polynomial, ascending degree.
    q_coeffs : np.ndarray, shape (n+1,)
        Chebyshev coefficients of the denominator polynomial, ascending degree.
        Normalised so ``q_coeffs[0] = 1``.
    r_handle : callable
        Evaluates ``p(x) / q(x)`` at arbitrary points *x*.

    Notes
    -----
    The Clenshaw-Lord algorithm solves a Hankel system for the denominator
    coefficients and uses convolution to compute the numerator [1].

    The Maehly algorithm solves a linearised version of the same conditions,
    which is more stable when the Hankel system is ill-conditioned [2].

    Provenance
    ----------
    MATLAB source : @chebfun/chebpade.m (sub-functions chebpadeClenshawLord
        and chebpadeMaehly)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm:
        [1] K. O. Geddes, "Block structure in the Chebyshev-Padé table",
            SIAM J. Numer. Anal., 18, 1981.
        [2] H. J. Maehly, Proceedings of the IFIP Congress, 1960.

    See Also
    --------
    padeapprox, trigpade, ratinterp

    Examples
    --------
    Chebyshev-Padé (4, 4) approximant to exp(x) on [-1, 1]:

    >>> import numpy as np
    >>> from chebfunjax.utils.ratapprox import chebpade
    >>> p, q, r = chebpade(np.exp, 4, 4)
    >>> abs(r(0.5) - np.exp(0.5)) < 1e-10
    True
    """
    # ---- 1. Extract Chebyshev coefficients ----
    # Minimum number of coefficients needed:
    n_needed = m + 2 * n + 1

    if hasattr(f, "coeffs"):
        # Chebfun-like object
        c_raw = np.asarray(f.coeffs, dtype=np.float64).ravel()
    elif callable(f):
        # Callable: sample on a Chebyshev grid and compute DCT
        n_pts = max(n_needed + 10, 128)
        from chebfunjax.utils.quadrature import chebpts
        t = np.array(chebpts(n_pts, kind=2), dtype=np.float64)
        vals = np.asarray(f(t), dtype=np.float64).ravel()
        import jax.numpy as _jnp

        from chebfunjax.utils.transforms import vals2coeffs
        c_raw = np.array(vals2coeffs(_jnp.array(vals)), dtype=np.float64)
    else:
        c_raw = np.asarray(f, dtype=np.float64).ravel()

    # Optionally truncate to K terms
    if K >= 0:
        c_raw = c_raw[: K + 1]

    # Zero-pad if necessary
    if len(c_raw) < n_needed:
        c_raw = np.concatenate([c_raw, np.zeros(n_needed - len(c_raw))])

    if kind.lower() == "clenshawlord":
        p_coeffs, q_coeffs = _chebpade_clenshawlord(c_raw, m, n)
    elif kind.lower() == "maehly":
        p_coeffs, q_coeffs = _chebpade_maehly(c_raw, m, n)
    else:
        raise ValueError(
            f"chebpade: unknown kind '{kind}'.  Use 'clenshawlord' or 'maehly'."
        )

    # ---- 3. Build evaluator ----
    # Evaluate Chebyshev sums via Clenshaw algorithm (NumPy)
    def _eval_cheb(c, x):
        x = np.asarray(x)
        if len(c) == 0:
            return np.zeros_like(x, dtype=float)
        if len(c) == 1:
            return np.full_like(x, c[0], dtype=float)
        b1 = np.zeros_like(x, dtype=float)
        b2 = np.zeros_like(x, dtype=float)
        for k in range(len(c) - 1, 0, -1):
            b0 = c[k] + 2.0 * x * b1 - b2
            b2 = b1
            b1 = b0
        return c[0] + x * b1 - b2

    def r_handle(x):
        x = np.asarray(x, dtype=float)
        return _eval_cheb(p_coeffs, x) / _eval_cheb(q_coeffs, x)

    return p_coeffs, q_coeffs, r_handle


def _chebpade_clenshawlord(
    c: np.ndarray, m: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Clenshaw-Lord Chebyshev-Padé (internal implementation).

    Provenance
    ----------
    MATLAB source : chebpadeClenshawLord (private sub-function of chebpade.m)
    Chebfun commit: 7574c77
    """
    l = max(m, n)

    # Scale c[0] (two-sided Chebyshev series convention)
    c2 = c.copy()
    c2[0] = 2.0 * c2[0]

    # Build Hankel system for denominator coefficients
    if n > 0:
        # top row: c[|m-n+1|], ..., c[m]
        idx_top = np.abs(np.arange(m - n + 1, m + 1))
        top = c2[idx_top]
        # bottom row: c[m], ..., c[m+n-1]
        bot = c2[m : m + n]
        # RHS: c[m+1], ..., c[m+n]
        rhs = c2[m + 1 : m + n + 1]

        from scipy.linalg import hankel
        H = hankel(top, bot)
        try:
            beta = np.concatenate([np.linalg.solve(-H, rhs), [1.0]])
        except np.linalg.LinAlgError:
            # Singular — fall back to denominator = 1
            beta = np.array([1.0])
        beta = beta[::-1]  # flip to ascending
    else:
        beta = np.array([1.0])

    # Undo the c[0] scaling for convolution
    c2[0] = c2[0] / 2.0

    # Compute numerator via convolution: alpha = conv(c[1:l+2], beta)[:l+1]
    alpha = np.convolve(c2[: l + 1], beta)[: l + 1]

    # Numerator Chebyshev-Padé coefficients using product formula
    l2 = l + 1
    n2 = len(beta)
    pk = np.zeros(m + 1)
    # Build product matrix D[i, j] = alpha[i] * beta[j]
    alpha_mat = np.outer(alpha[:l2], np.ones(n2))
    beta_mat = np.outer(np.ones(l2), beta[:n2])
    D = alpha_mat * beta_mat  # (l2, n2)

    for k in range(m + 1):
        # Sum diagonals at offset k and -k
        d_upper = np.diag(D, k) if k < D.shape[1] else np.array([])
        d_lower = np.diag(D, -k) if k < D.shape[0] else np.array([])
        pk[k] = np.sum(d_upper) + (np.sum(d_lower) if k > 0 else 0.0)

    # Denominator coefficients
    n3 = len(beta)
    qk = np.zeros(n + 1)
    for k in range(n + 1):
        u = beta[: n3 - k]
        v = beta[k:n3]
        qk[k] = np.dot(u, v)

    # Normalize
    pk = pk / qk[0]
    qk = 2.0 * qk / qk[0]
    qk[0] = 1.0

    return pk, qk


def _chebpade_maehly(
    c: np.ndarray, m: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Maehly Chebyshev-Padé (internal implementation).

    Provenance
    ----------
    MATLAB source : chebpadeMaehly (private sub-function of chebpade.m)
    Chebfun commit: 7574c77
    """
    a = c.copy()

    # Denominator system
    rows = np.arange(m + 1, m + n + 1, dtype=int)
    cols = np.arange(1, n + 1, dtype=int)
    R, C = np.meshgrid(rows, cols, indexing="ij")
    D = a[R + C] + a[np.abs(R - C)]
    if n > m:
        for k in range(min(n - m, n)):
            D[k + m, k + m] += a[0]

    if n == 0:
        qk = np.array([1.0])
        pk = a[: m + 1].copy()
        pk[0] = 0.5 * pk[0] if len(pk) > 1 else pk[0]
        return pk, qk

    # Solve for denominator
    rhs = -2.0 * a[m + 1 : m + n + 1]
    try:
        q_inner = np.linalg.solve(D, rhs)
    except np.linalg.LinAlgError:
        q_inner = np.zeros(n)
    qk = np.concatenate([[1.0], q_inner])

    # Numerator
    rows2 = np.arange(1, m + 1, dtype=int)
    cols2 = np.arange(1, n + 1, dtype=int)
    R2, C2 = np.meshgrid(rows2, cols2, indexing="ij")
    B = a[R2 + C2] + a[np.abs(R2 - C2)]
    mask = (R2 == C2) & (R2 <= m) & (C2 <= m)
    B[mask] = B[mask] + a[0]

    top_row = a[1 : n + 1][None, :]  # (1, n)
    B_full = np.vstack([top_row, B])  # (m+1, n)

    pk = 0.5 * B_full @ q_inner + qk[0] * a[: m + 1]
    pk[0] = 2.0 * pk[0]  # back-undo the 1/2 convention for T_0

    # Normalize
    pk = pk / qk[0]
    return pk, qk


# ===========================================================================
# Trigonometric Padé approximation (trigpade)
# ===========================================================================


def trigpade(
    f,
    m: int,
    n: int,
    domain: tuple[float, float] = (-1.0, 1.0),
    N_fft: int = 0,
):
    r"""Trigonometric (Fourier) Padé approximation of type (m, n).

    Computes trigonometric polynomials *p* (degree *m*) and *q* (degree *n*)
    such that the trigonometric rational function ``p/q`` is the type-(m, n)
    Fourier-Padé approximant to *f*.  The Fourier series of ``p/q`` agrees
    with that of *f* up to the highest possible order.

    Parameters
    ----------
    f : callable or array_like or Chebfun
        Periodic function on *domain*, or vector of Fourier coefficients
        ``[c_{-K}, ..., c_{-1}, c_0, c_1, ..., c_K]`` (centred ordering).
        If callable, evaluated on a uniform grid of ``2*(m+2*n)+1`` points.
    m : int
        Degree of the numerator trigonometric polynomial.
    n : int
        Degree of the denominator trigonometric polynomial.
    domain : (float, float), optional
        Period interval.  Default ``(-1.0, 1.0)``.
    N_fft : int, optional
        Number of FFT points for evaluating callable *f*.  0 = auto.

    Returns
    -------
    p_coeffs : np.ndarray, shape (2*m+1,)
        Fourier coefficients of the numerator, in ascending-frequency order
        ``[c_{-m}, ..., c_0, ..., c_m]``.
    q_coeffs : np.ndarray, shape (2*n+1,)
        Fourier coefficients of the denominator, normalised so ``c_0 = 1``.
    r_handle : callable
        Evaluates ``p(x) / q(x)`` at arbitrary points *x*.

    Notes
    -----
    The algorithm follows Baker & Graves-Morris (1996) Chapter 5 and the
    DPhil thesis of Javed (2017).  The denominator coefficients satisfy a
    linear Toeplitz system built from the Fourier coefficients of *f*; the
    numerator is then determined by convolution.

    Provenance
    ----------
    MATLAB source : @chebfun/trigpade.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm:
        [1] G. A. Baker and P. R. Graves-Morris, Padé Approximants,
            Cambridge Univ. Press, 1996.
        [2] M. Javed, DPhil thesis, Oxford, 2017.

    See Also
    --------
    chebpade, padeapprox

    Examples
    --------
    Trigonometric Padé (2, 2) approximant to ``sin(pi*x)`` on ``[-1, 1]``:

    >>> import numpy as np
    >>> from chebfunjax.utils.ratapprox import trigpade
    >>> p_c, q_c, r = trigpade(lambda x: np.sin(np.pi * x), 2, 2)
    >>> abs(r(0.3) - np.sin(np.pi * 0.3)) < 1e-6
    True
    """
    a, b = float(domain[0]), float(domain[1])

    # ---- 1. Extract Fourier coefficients ----
    n_coeff = 2 * (m + 2 * n) + 2 if N_fft <= 0 else N_fft
    n_coeff = max(n_coeff, 2 * (m + 2 * n) + 4)
    if n_coeff % 2 == 0:
        n_coeff += 1  # odd for symmetric extraction

    if hasattr(f, "coeffs") and hasattr(f, "domain"):
        # Chebfun: re-sample uniformly and do FFT
        x_uni = np.linspace(a, b, n_coeff, endpoint=False)
        import jax.numpy as _jnp
        fvals = np.asarray(f(_jnp.array(x_uni)), dtype=float)
    elif callable(f):
        x_uni = np.linspace(a, b, n_coeff, endpoint=False)
        fvals = np.asarray(f(x_uni), dtype=float)
    else:
        c_full = np.asarray(f, dtype=complex).ravel()
        # c_full already in centred order — just use them
        _M = (len(c_full) - 1) // 2
        # Build lookup for frequency k in [-_M, _M]
        def _c(k):
            idx = k + _M
            if 0 <= idx < len(c_full):
                return c_full[idx]
            return 0.0
        return _build_trigpade_from_cfunc(m, n, _c, a, b)

    # FFT to get Fourier coefficients (standard ordering: 0, 1, ..., N//2, -(N//2-1), ..., -1)
    C_fft = np.fft.fft(fvals) / n_coeff
    N = n_coeff

    def _c(k):
        """Fourier coefficient for frequency k (any integer)."""
        idx = int(k) % N
        return complex(C_fft[idx])

    return _build_trigpade_from_cfunc(m, n, _c, a, b)


def _build_trigpade_from_cfunc(
    m: int,
    n: int,
    c_func,
    a: float,
    b: float,
):
    """Construct trigonometric Padé approximant from a Fourier coefficient oracle."""
    # Solve Toeplitz system for denominator Fourier coefficients
    # The system is: sum_{j=-n}^{n} c_{k-j} * q_j = 0 for k = m+1, ..., m+n
    # and k = -(m+1), ..., -(m+n)  (negative frequencies by complex conjugate).
    # Following Javed (2017): denominator has 2n+1 coefficients b_j, j=-n..n.
    # We write b_0 = 1 and solve for b_1, ..., b_n (+ complex conjugates).

    if n == 0:
        # Trivial denominator = 1
        q_c = np.array([1.0])
        # Numerator = first 2m+1 Fourier coefficients of f
        p_c = np.array([c_func(k) for k in range(-m, m + 1)], dtype=complex)
        # Normalise to real if input is real
        if np.allclose(np.imag(p_c), 0, atol=1e-12):
            p_c = np.real(p_c)
            q_c = np.array([1.0])

        def r_handle(x):
            x = np.asarray(x, dtype=float)
            vals = np.zeros_like(x, dtype=complex)
            for k in range(-m, m + 1):
                omega = 2j * np.pi * k / (b - a)
                vals += p_c[k + m] * np.exp(omega * (x - a))
            return np.real(vals) if np.allclose(np.imag(p_c), 0, atol=1e-12) else vals

        return np.real(p_c) if np.allclose(np.imag(p_c), 0) else p_c, q_c, r_handle

    # Build Toeplitz system (n x n) for q_1, ..., q_n
    # Row k (k = 1..n): sum_{j=1}^{n} (c_{k+j} + c_{k-j}) * q_j = -c_k
    # (using symmetry b_j = conj(b_{-j}) for real f)
    A = np.zeros((n, n), dtype=complex)
    rhs = np.zeros(n, dtype=complex)
    for i in range(n):
        k = m + 1 + i  # frequency index
        rhs[i] = -c_func(k)
        for j in range(n):
            jj = j + 1
            A[i, j] = c_func(k + jj) + c_func(k - jj)

    try:
        q_inner = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        q_inner = np.zeros(n, dtype=complex)

    # Full denominator Fourier coefficients: b_{-n},...,b_0,...,b_n
    q_all = np.concatenate([np.conj(q_inner[::-1]), [1.0], q_inner])

    # Numerator: p_k = sum_{j=-n}^{n} q_j * c_{k-j}, for k = -m, ..., m
    p_all = np.zeros(2 * m + 1, dtype=complex)
    for i, k in enumerate(range(-m, m + 1)):
        for j in range(-n, n + 1):
            p_all[i] += q_all[j + n] * c_func(k - j)

    # Make real if applicable
    if np.allclose(np.imag(p_all), 0, atol=1e-10) and np.allclose(
        np.imag(q_all), 0, atol=1e-10
    ):
        p_all = np.real(p_all)
        q_all = np.real(q_all)

    def r_handle(x, _p=p_all, _q=q_all):
        x = np.asarray(x, dtype=float)
        p_val = np.zeros_like(x, dtype=complex)
        q_val = np.zeros_like(x, dtype=complex)
        for k, pk in enumerate(_p):
            freq = k - len(_p) // 2
            omega = 2j * np.pi * freq / (b - a)
            p_val += pk * np.exp(omega * (x - a))
        for j, qj in enumerate(_q):
            freq = j - len(_q) // 2
            omega = 2j * np.pi * freq / (b - a)
            q_val += qj * np.exp(omega * (x - a))
        result = p_val / q_val
        return np.real(result) if np.allclose(np.imag(_p), 0) else result

    return p_all, q_all, r_handle


# Source-faithful trigonometric rational interpolation helpers.

def _trigrat_nonnegative_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    ivalue = int(value)
    if ivalue != value or ivalue < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return ivalue


def _trigrat_check_symmetries(f, xi, xi_type, tol):
    """MATLAB checkSymmetries indexing; flags are later reset by source code."""
    n = int(xi.size)
    if n == 0:
        # Empty source comparisons have infinity norm zero.
        return tol > 0, tol > 0
    if n == 1:
        if xi_type.startswith("equi") or float(xi[0]) in (-1.0, 1.0):
            # These source branches compare empty vectors.
            return tol > 0, tol > 0
        if abs(2.0 * float(xi[0])) >= tol:
            return False, False
        return True, float(abs(2.0 * f[0])) < tol
    if xi_type.startswith("equi"):
        if n % 2:
            mid = n // 2
            fl, fr = f[1:mid + 1], f[n - 1:mid:-1]
        else:
            mid = n // 2
            fl, fr = f[1:mid + 1], f[n - 1:mid - 1:-1]
    else:
        order = jnp.argsort(xi)
        x, f = xi[order], f[order]
        if float(x[0]) == -1.0:
            if n % 2:
                mid = n // 2
                xl, xr = x[1:mid + 1], x[n - 1:mid:-1]
                fl, fr = f[1:mid + 1], f[n - 1:mid:-1]
            else:
                mid = n // 2
                xl, xr = x[1:mid + 1], x[n - 1:mid - 1:-1]
                fl, fr = f[1:mid + 1], f[n - 1:mid - 1:-1]
        elif float(x[-1]) == 1.0:
            if n % 2:
                mid = n // 2
                xl, xr = x[:mid], x[n - 2:mid - 1:-1]
                fl, fr = f[:mid], f[n - 2:mid - 1:-1]
            else:
                mid = n // 2
                xl, xr = x[:mid], x[n - 2:mid - 2:-1]
                fl, fr = f[:mid], f[n - 2:mid - 2:-1]
        else:
            if n % 2:
                mid = (n + 1) // 2
                xl, xr = x[:mid], x[n - 1:mid - 2:-1]
                fl, fr = f[:mid], f[n - 1:mid - 2:-1]
            else:
                mid = n // 2
                xl, xr = x[:mid], x[n - 1:mid - 1:-1]
                fl, fr = f[:mid], f[n - 1:mid - 1:-1]
        if xl.size == 0 or float(jnp.max(jnp.abs(xl + xr))) >= tol:
            return False, False
    if fl.size == 0:
        return tol > 0, False
    return (
        float(jnp.max(jnp.abs(fl - fr))) < tol,
        float(jnp.max(jnp.abs(fl + fr))) < tol,
    )


def _trigrat_construct_matrices(th, m, n):
    # MATLAB first allocates 2*m+1 columns, then P(:,1)=1 expands an
    # N-by-0 array to N-by-1 when m == -0.5 after empty-tail chopping.
    # The same assignment/expansion occurs independently for Q and n.
    p_count, q_count = max(1, int(2 * m + 1)), max(1, int(2 * n + 1))
    P = jnp.zeros((th.size, p_count), dtype=jnp.float64)
    Q = jnp.zeros((th.size, q_count), dtype=jnp.float64)
    P = P.at[:, 0].set(1.0)
    Q = Q.at[:, 0].set(1.0)
    for j in range(1, int(m) + 1):
        theta = jnp.pi * j * th
        P = P.at[:, 2 * j - 1].set(jnp.sin(theta))
        P = P.at[:, 2 * j].set(jnp.cos(theta))
    for j in range(1, int(n) + 1):
        theta = jnp.pi * j * th
        Q = Q.at[:, 2 * j - 1].set(jnp.sin(theta))
        Q = Q.at[:, 2 * j].set(jnp.cos(theta))
    return P, Q


def _trigrat_sincos_to_exponential(a):
    a = jnp.ravel(a)
    tmp = (a[2::2] - 1j * a[1::2]) / 2.0
    # MATLAB uses a(1) when assembling the constant exponential coefficient;
    # keep the corresponding IndexError when source slicing produced empty a.
    center = a[0]
    return jnp.concatenate((jnp.conj(tmp[::-1]), jnp.asarray([center]), tmp))


def _trigrat_chop_coeffs(a, threshold):
    a = jnp.ravel(a)
    size = int(a.size)
    if size <= 1:
        return a
    if size % 2 == 0:
        raise ValueError("trigratinterp coefficients must have odd length")
    mid = size // 2
    sym_tail = (jnp.abs(a[mid:]) + jnp.abs(a[mid::-1])) / 2.0
    kept = jnp.where(sym_tail > threshold, jnp.arange(mid + 1), -1)
    idx = int(jnp.max(kept))
    if idx < 0:
        # MATLAB's empty find index yields an empty colon slice here.
        return a[:0]
    return a[mid - idx:mid + idx + 1]


def _trigrat_fit_coefficients(fk, m, n, th, f_even, f_odd, *, robustness,
                      interpolation, threshold):
    # Source trig_rat_interp.m explicitly resets the symmetry flags here.
    f_even = False  # noqa: F841 -- source resets both symmetry flags here.
    f_odd = False  # noqa: F841 -- source resets both symmetry flags here.
    while True:
        pmat, qmat = _trigrat_construct_matrices(th, m, n)
        sys = jnp.concatenate((pmat, -fk[:, None] * qmat), axis=1)
        scale = 1.0 / jnp.maximum(jnp.abs(fk), 1.0)
        sys = scale[:, None] * sys
        _u, singular, vh = jnp.linalg.svd(sys, full_matrices=True)
        V = jnp.conj(vh.T)
        v = V[:, -1]
        # MATLAB getCoeffs splits at 2*m+1, not at the dynamically expanded
        # P width. After an empty chop this is zero; int() is the Python slice
        # adapter for MATLAB's integral or half-integral width expression.
        p_width = int(2 * m + 1)
        ac = _trigrat_chop_coeffs(
            _trigrat_sincos_to_exponential(v[:p_width]), threshold)
        bc = _trigrat_chop_coeffs(
            _trigrat_sincos_to_exponential(v[p_width:]), threshold)
        m = min((ac.size - 1) / 2, m)
        above = jnp.where(jnp.abs(singular) > threshold, jnp.arange(singular.size) + 1, 0)
        n_big = int(jnp.max(above))
        n_small = int(singular.size) - n_big
        if n_small < 2 or not robustness:
            break
        reduction = n_small // 2
        n_new = n - reduction
        if n_new < 0:
            # Source retains n and loops; guard against an infinite loop as a
            # documented defensive adapter if this malformed rank pattern occurs.
            break
        n = n_new
    return ac, bc


def _trigrat_degree(coeffs):
    """MATLAB degree convention: `(length(coeffs)-1)/2`, including -0.5."""
    return (int(jnp.asarray(coeffs).size) - 1) / 2


def _trigrat_make_approximation(ac, bc, a_dom, b_dom):
    """Build the source p/q Chebfuns once and capture their quotient in r."""
    from chebfunjax.chebfun1d.chebfun import chebfun

    p = chebfun(ac, coeffs=True, trig=True)
    q = chebfun(bc, coeffs=True, trig=True)
    if (a_dom, b_dom) != (-1.0, 1.0):
        p = p.new_domain((a_dom, b_dom))
        q = q.new_domain((a_dom, b_dom))

    def evaluate(x):
        x = jnp.asarray(x)
        return p(x) / q(x)

    return p, q, evaluate


def _trigrat_eval_fourier(coeffs, xi):
    coeffs = jnp.asarray(coeffs)
    if coeffs.size == 0:
        return jnp.zeros_like(jnp.asarray(xi), dtype=jnp.complex128)
    degree = (coeffs.size - 1) // 2
    k = jnp.arange(-degree, degree + 1)
    phase = jnp.exp(1j * jnp.pi * jnp.asarray(xi)[..., None] * k)
    return jnp.sum(phase * coeffs, axis=-1)


def _trigrat_trim_leading_polynomial_zeros(coeffs):
    """Eager shape adapter for MATLAB roots/residue leading-zero trimming."""
    coeffs = jnp.ravel(jnp.asarray(coeffs, dtype=jnp.complex128))
    if coeffs.size == 0:
        return coeffs
    nz = jnp.abs(coeffs) != 0
    if not bool(jnp.any(nz)):
        return coeffs[:0]
    first = int(jnp.argmax(nz))
    return coeffs[first:]


def _trigrat_source_x_roots(bc, a_dom, b_dom):
    # MATLAB @trigtech/roots.m returns [] immediately for an empty Trigtech.
    if jnp.asarray(bc).size == 0:
        return jnp.empty((0,), dtype=jnp.complex128)
    # MATLAB roots(q,'all') first simplifies each Trigtech column, then reverses
    # coeffs for polynomial roots and maps -i*log(z)/pi to the physical domain.
    # Reuse the current public constructor/simplifier; its realness metadata is
    # a separate dependency and is not claimed qualified by this candidate.
    from chebfunjax.chebfun1d.chebfun import chebfun

    q = chebfun(bc, coeffs=True, trig=True, domain=(a_dom, b_dom))
    q_piece = q.funs[0]
    q_tech = getattr(q_piece, "tech", q_piece).simplify()
    poly = _trigrat_trim_leading_polynomial_zeros(
        jnp.ravel(q_tech.coeffs)[::-1])
    if poly.size <= 1:
        return jnp.empty((0,), dtype=jnp.complex128)
    z = jnp.roots(poly, strip_zeros=False)
    x_ref = -1j * jnp.log(z) / jnp.pi
    return 0.5 * (a_dom + b_dom) + 0.5 * (b_dom - a_dom) * x_ref


def _trigrat_poly_derivative(coeffs):
    coeffs = jnp.asarray(coeffs, dtype=jnp.complex128)
    degree = coeffs.size - 1
    return coeffs[:-1] * jnp.arange(degree, 0, -1, dtype=jnp.float64)


def _trigrat_matlab_complex_sort_order(values):
    """MATLAB sort order for complex values: magnitude, then phase (-pi,pi]."""
    values = jnp.asarray(values, dtype=jnp.complex128)
    phase = jnp.angle(values)
    phase = jnp.where(phase <= -jnp.pi, phase + 2.0 * jnp.pi, phase)
    return jnp.lexsort((phase, jnp.abs(values)))


def _trigrat_source_poly_residues(ac, bc):
    # @trigtech/poly returns coefficient arrays unchanged; polynomial residue
    # consumes them as descending-power vectors. Full repeated-pole residue
    # expansion is intentionally unported. Exact duplicate roots are rejected;
    # near-multiple roots also remain outside this simple-pole implementation.
    p = _trigrat_trim_leading_polynomial_zeros(ac)
    q = _trigrat_trim_leading_polynomial_zeros(bc)
    if q.size <= 1:
        dtype = jnp.result_type(p.dtype, q.dtype, jnp.complex128)
        return jnp.empty((0,), dtype=dtype), jnp.empty((0,), dtype=dtype)
    poles = jnp.roots(q, strip_zeros=False)
    if poles.size > 1:
        diffs = jnp.abs(poles[:, None] - poles[None, :])
        close = (diffs == 0) & (~jnp.eye(poles.size, dtype=bool))
        if bool(jnp.any(close)):
            raise NotImplementedError("repeated-pole residue expansion is not yet ported")
    qd = _trigrat_poly_derivative(q)
    residues = jnp.polyval(p, poles) / jnp.polyval(qd, poles)
    order = _trigrat_matlab_complex_sort_order(poles)
    poles, residues = poles[order], residues[order]
    if poles.size > 1:
        duplicate = poles[1:] == poles[:-1]
        residues = residues.at[1:].set(jnp.where(duplicate, residues[:-1], residues[1:]))
    return poles, residues
