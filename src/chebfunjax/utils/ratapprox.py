# uses-numpy: rational interpolation uses numpy/scipy SVD and eigenvalue solvers (not JIT-safe)
"""Rational approximation: Padé, rational interpolation, trig rational interpolation.

Translated from MATLAB Chebfun (commit 7574c77): padeapprox.m, ratinterp.m,
trigratinterp.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

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
            Q, _ = np.linalg.qr((C @ D).T, mode='complete')
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
    poles : np.ndarray
        Real poles of the approximant (on ``domain``).
    residues : np.ndarray
        Residues at those poles.

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
        b_poly = np.zeros(nu + 1, dtype=complex)
        b_poly[:nu+1] = b_coeffs[:nu+1]
        # b_coeffs in Chebyshev basis — find poles via eigenvalues
        poles_ref = _chebyshev_roots(b_poly)
        poles_ref = poles_ref[np.abs(np.imag(poles_ref)) < 1e-10]
        poles_ref = np.real(poles_ref)
        poles = mid + hd * poles_ref

        t = max(tol, 1e-7)
        residues = t * (r_handle(poles + t) - r_handle(poles - t)) / 2.0
    else:
        poles = np.array([])
        residues = np.array([])

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
    else:  # ARBITRARY nodes — build Chebyshev Vandermonde, QR decompose
        C = np.ones((N1, N1))
        xi_real = np.real(xi)
        C[:, 1] = xi_real
        for k in range(2, N1):
            C[:, k] = 2 * xi_real * C[:, k - 1] - C[:, k - 2]
        Q_qr, R_qr = np.linalg.qr(C)
        Z = Q_qr.T @ np.diag(f) @ Q_qr[:, : n + 1]

    return Z, R_qr, Q_qr


def _chebtech2_coeffs2vals_matrix(N):
    """Dense (N x N) matrix: Chebyshev coefficients -> values at 2nd-kind pts."""
    eye_N = np.eye(N)
    result = np.zeros((N, N))
    for j in range(N):
        result[:, j] = np.array(coeffs2vals(jnp.array(eye_N[:, j], dtype=jnp.float64)))
    return result


def _chebtech2_vals2coeffs_matrix_apply(V, N):
    """Apply vals2coeffs column-by-column to build Z."""
    _, ncols = V.shape
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
    """Apply chebtech1 vals2coeffs (DCT-II) column-by-column."""
    _, ncols = V.shape
    result = np.zeros((N, ncols))
    k = np.arange(N)
    j = np.arange(N)
    # DCT-II: c_k = (2/N) * sum_j v_j * cos(k*(2j+1)*pi/(2N))
    # with c_0 halved
    T = np.cos(np.outer(k, (2 * j[::-1] + 1)) * np.pi / (2 * N))
    T_scaled = (2.0 / N) * T
    T_scaled[0, :] /= 2.0
    for j_col in range(ncols):
        result[:, j_col] = T_scaled @ np.real(V[:, j_col])
    return result


def _qr_to_cheb_basis(a_hat, b_hat, R_qr, N1):
    """Convert QR-basis coefficients a_hat, b_hat to Chebyshev basis.

    The QR decomposition gives C = Q * R where C is the Chebyshev Vandermonde.
    Coefficients in QR basis satisfy C @ cheb_coeffs = Q @ hat_coeffs, i.e.
    R @ cheb_coeffs = hat_coeffs (padded to N1).  Solve via back-substitution.
    """
    na = len(a_hat)
    nb = len(b_hat)

    # Pad to N1 and solve R @ a_cheb = a_hat_padded
    a_pad = np.zeros(N1, dtype=complex)
    a_pad[:na] = a_hat
    b_pad = np.zeros(N1, dtype=complex)
    b_pad[:nb] = b_hat

    # Use triangular solve: R is upper triangular (N1 x N1)
    a_cheb_full = np.linalg.solve(R_qr, a_pad)
    b_cheb_full = np.linalg.solve(R_qr, b_pad)

    # Return only the significant coefficients
    a_cheb = np.real(a_cheb_full[:na])
    b_cheb = np.real(b_cheb_full[:nb])

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

    The submatrix C = Z[m+1:N1, :n+1] has:
      - intrinsic null dim = max(0, n_cols - n_rows)   (from underdetermination)
      - 1 additional null direction when the function is exactly rational of type
        (m, n_current) — this is the denominator we seek.

    So the EXPECTED number of near-zero SVs is intrinsic_null_dim + 1.
    We count how many SVs are near the smallest (relative to ts).
    If count > expected_null → more null directions than expected → reduce degree
    by (count - expected_null).
    If count <= expected_null → found our denominator, stop.
    """
    if n == 0:
        return np.array([1.0]), 0

    shift = int(fEven) ^ int(m % 2 == 1)

    if not (fOdd or fEven) or (n > 1):
        n_current = n
        while True:
            if not (fOdd or fEven):
                sub = Z[m + 1: N1, : n_current + 1]
                n_rows, n_cols = sub.shape
                sv = np.linalg.svd(sub, compute_uv=False)
                b = np.linalg.svd(sub, full_matrices=True)[2][-1, :].conj()
            else:
                rows = slice(m + 1 + shift, N1, 2)
                cols = slice(None, n_current + 1, 2)
                sub = Z[rows, cols]
                n_rows, n_cols = sub.shape
                sv = np.linalg.svd(sub, compute_uv=False)
                b_half = np.linalg.svd(sub, full_matrices=True)[2][-1, :].conj()
                b = np.zeros(n_current + 1, dtype=complex)
                b[::2] = b_half

            if n_current <= 0 or len(sv) == 0:
                break

            ssv = sv[-1]

            # Expected null dim: intrinsic (underdetermination) + 1 (rational)
            intrinsic_null_dim = max(0, n_cols - n_rows)
            expected_null = intrinsic_null_dim + 1

            # Count near-zero SVs (those within ts of the smallest)
            count = int(np.sum(sv - ssv <= ts))

            if count <= expected_null:
                # Exactly the expected null space — denominator found, stop
                break

            # More near-zero SVs than expected: degree is too high
            reduce = count - expected_null
            if fEven or fOdd:
                n_current -= 2 * reduce
            else:
                n_current -= reduce

            if n_current <= 0:
                if fEven:
                    b = np.array([1.0, 0.0])
                elif fOdd:
                    b = np.array([0.0, 1.0])
                else:
                    b = np.array([1.0])
                n_current = max(n_current, 0)
                break
    else:
        if fEven:
            b = np.array([1.0, 0.0])
        elif fOdd:
            b = np.array([0.0, 1.0])
        else:
            b = np.array([1.0])
        n_current = n

    return b, n_current


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
            b_vals = _chebtech1_coeffs2vals_matrix(N1) @ np.real(b_pad)
            a_vals = b_vals * np.real(f)
            # Convert back to coefficients
            a = _chebtech1_vals2coeffs_matrix_apply(a_vals[:, None], N1).ravel()
            a = a[: m + 1]
        else:  # TYPE2
            b_pad = np.zeros(N1, dtype=complex)
            b_pad[: len(b)] = b
            b_vals = np.array(
                coeffs2vals(jnp.array(np.real(b_pad), dtype=jnp.float64))
            )
            a_vals = b_vals * np.real(f)
            a = np.array(
                vals2coeffs(jnp.array(a_vals, dtype=jnp.float64))
            )
            a = a[: m + 1]
    else:
        # ARBITRARY nodes: Z = Q'.diag(f).Q  (QR basis)
        # a_hat = Z[:m+1, :n_b] @ b_hat  (still QR basis; converted to Cheb later)
        n_b = len(b)
        a = Z[: m + 1, :n_b] @ b

    if fEven:
        a[1::2] = 0.0
    elif fOdd:
        a[0::2] = 0.0

    return a


def _trim_coeffs(a, b, tol, ts):
    """Trim trailing small coefficients from a and b."""
    at = np.array(a, dtype=complex)
    bt = np.array(b, dtype=complex)

    if tol > 0:
        nna = np.abs(at) > ts
        nnb = np.abs(bt) > tol

        last_a = int(np.where(nna)[0][-1]) if np.any(nna) else 0
        last_b = int(np.where(nnb)[0][-1]) if np.any(nnb) else 0

        at = at[:last_a + 1]
        bt = bt[:last_b + 1]

        # Remove small leading coefficients (both < threshold)
        while len(at) > 0 and len(bt) > 0 and np.abs(at[0]) < ts and np.abs(bt[0]) < ts:
            at = at[1:]
            bt = bt[1:]

    if len(at) == 0:
        at = np.array([0.0 + 0j])
        bt = np.array([1.0 + 0j])

    return np.real(at), np.real(bt)


def _construct_rat_approx(xi_type, R_qr, a, b, mu, nu, a_dom, b_dom):
    """Build the function handle for the rational approximant."""
    mid = 0.5 * (a_dom + b_dom)
    hd = 2.0 / (b_dom - a_dom)  # maps x in [a,b] to t in [-1,1]: t = hd*(x-mid)

    if xi_type.upper().startswith("TYPE"):
        ch = xi_type[4]
        if ch == "0":  # Roots of unity — polynomial in z
            a_rev = a[: mu + 1][::-1]
            b_rev = b[: nu + 1][::-1]
            def r_fn(x):
                x = np.asarray(x, dtype=float)
                t = hd * (x - mid)
                return np.polyval(a_rev, t) / np.polyval(b_rev, t)
        else:  # Chebyshev basis
            def r_fn(x):
                x = np.asarray(x, dtype=float)
                t = hd * (x - mid)
                return _eval_cheb_poly(a, t) / _eval_cheb_poly(b, t)
    else:  # Arbitrary nodes — coefficients in orthogonal basis from QR
        def r_fn(x):
            x = np.asarray(x, dtype=float)
            t = hd * (x - mid)
            return _eval_cheb_poly(a, t) / _eval_cheb_poly(b, t)

    return r_fn


def _eval_cheb_poly(coeffs, x):
    """Evaluate a Chebyshev expansion at x (numpy, scalar or array)."""
    x = np.asarray(x, dtype=float)
    n = len(coeffs)
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.full_like(x, coeffs[0])
    bk2 = np.zeros_like(x)
    bk1 = np.zeros_like(x)
    for k in range(n - 1, 0, -1):
        bk = coeffs[k] + 2.0 * x * bk1 - bk2
        bk2 = bk1
        bk1 = bk
    return coeffs[0] + x * bk1 - bk2


def _chebyshev_roots(coeffs):
    """Find roots of a Chebyshev expansion via colleague matrix."""
    c = np.asarray(coeffs, dtype=complex)
    n = len(c) - 1  # degree
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([-c[0] / c[1]])
    # Colleague matrix
    oh = 0.5 * np.ones(n - 1)
    A = np.diag(oh, 1) + np.diag(oh, -1)
    A[-2, -1] = 1.0
    c_adj = -0.5 * c[:-1] / c[-1]
    c_adj[-2] += 0.5
    A[:, 0] = np.real(c_adj[::-1])
    return np.linalg.eigvals(A)


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
    domain: tuple[float, float] = (-1.0, 1.0),
):
    r"""Robust trigonometric rational interpolation or least-squares.

    Computes a type-(mu, nu) trigonometric rational approximant to a function
    or periodic data, where both numerator and denominator are trigonometric
    polynomials of degree ``m`` and ``n``, respectively.

    Parameters
    ----------
    f : callable or array_like
        Function handle or vector of function values at the nodes.
        If callable, it is sampled at ``NN`` equidistant points on ``domain``.
    m : int
        Desired numerator degree (trig polynomial of degree m has 2m+1 terms).
    n : int
        Desired denominator degree.
    NN : int or None, optional
        Number of nodes.  Defaults to ``2*(m+n)+1`` (minimum for interpolation).
        Must be >= ``2*(m+n)+1``.
    xi : array_like, str, or None, optional
        Nodes.  Defaults to ``NN`` equidistant points on ``domain``.
        Can also be ``'equi'`` or ``'equidistant'`` for the same default.
    tol : float, optional
        Relative tolerance for robustification.  Default: 1e-14.
    domain : (float, float), optional
        Physical domain.  Default: ``(-1, 1)``.

    Returns
    -------
    r_handle : callable
        Function handle for the trigonometric rational approximant.
    a : np.ndarray
        Numerator Fourier coefficients (length 2*mu+1).
    b : np.ndarray
        Denominator Fourier coefficients (length 2*nu+1), normalized b[0]=1.
    mu : int
        Exact numerator degree.
    nu : int
        Exact denominator degree.
    poles : np.ndarray
        Poles in the complex plane.
    residues : np.ndarray
        Residues at those poles.

    Notes
    -----
    Developer notes from MATLAB Chebfun (trigratinterp.m):

    The algorithm is described in the DPhil thesis of Mohsin Javed.  It uses
    a DFT-based approach to assemble the linear system for the Fourier
    coefficients of the numerator and denominator, and robustifies via SVD.

    References
    ----------
    .. [1] M. Javed, "Algorithms for Trigonometric Polynomial and Rational
       Approximation", DPhil thesis, Oxford, 2016.

    Examples
    --------
    Type-(5, 5) approximant to 1/(sin(pi*x) - 0.2):

    >>> r_fn, a, b, mu, nu, poles, res = trigratinterp(
    ...     lambda x: 1.0 / (np.sin(np.pi * x) - 0.2), 5, 5)
    >>> abs(r_fn(0.5) - 1.0/(np.sin(np.pi*0.5) - 0.2)) < 1e-6
    True

    Provenance
    ----------
    MATLAB source : trigratinterp.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: [1] M. Javed, DPhil thesis, Oxford, 2016.

    See Also
    --------
    ratinterp, padeapprox, aaa
    """
    a_dom, b_dom = float(domain[0]), float(domain[1])
    period = b_dom - a_dom

    # ------------------------------------------------------------------
    # 1.  Determine number of nodes
    # ------------------------------------------------------------------
    min_NN = 2 * (m + n) + 1
    # If f is a data vector and NN is not explicitly given, infer from length
    if NN is None and not callable(f) and not isinstance(f, (str,)):
        try:
            NN = len(f)
        except TypeError:
            pass
    if NN is None:
        NN = min_NN
    if NN < min_NN:
        raise ValueError(
            f"trigratinterp: NN={NN} must be >= 2*(m+n)+1 = {min_NN}."
        )

    # ------------------------------------------------------------------
    # 2.  Generate nodes
    # ------------------------------------------------------------------
    if xi is None or (isinstance(xi, str) and xi.upper().startswith("EQUI")):
        # Equidistant points on [a_dom, b_dom)
        th = a_dom + period * np.arange(NN) / NN
    elif isinstance(xi, np.ndarray) or hasattr(xi, "__len__"):
        th = np.asarray(xi, dtype=float).ravel()
        NN = len(th)
    else:
        raise ValueError(f"trigratinterp: unrecognized xi type '{xi}'.")

    # ------------------------------------------------------------------
    # 3.  Sample f
    # ------------------------------------------------------------------
    if callable(f):
        fvals = np.asarray(f(th), dtype=complex).ravel()
    else:
        fvals = np.asarray(f, dtype=complex).ravel()
        if len(fvals) != NN:
            raise ValueError(
                f"trigratinterp: f has {len(fvals)} values but NN={NN}."
            )

    ts = tol * np.linalg.norm(fvals, np.inf)

    # ------------------------------------------------------------------
    # 4.  Check symmetries (even/odd in Fourier sense)
    # ------------------------------------------------------------------
    fEven, fOdd = _trig_check_symmetries(fvals, ts)

    # ------------------------------------------------------------------
    # 5.  Run the trig rational interpolation algorithm
    # ------------------------------------------------------------------
    ac, bc, n_eff = _trig_rat_interp(
        fvals, m, n, th, a_dom, b_dom, fEven, fOdd, tol > 0, True, ts
    )

    mu = (len(ac) - 1) // 2
    nu = (len(bc) - 1) // 2

    # ------------------------------------------------------------------
    # 6.  Build function handle
    # ------------------------------------------------------------------
    r_handle = _construct_trig_rat_approx(ac, bc, a_dom, b_dom, ts)

    # Normalize if pure polynomial (n=0 case)
    if n == 0 or nu == 0:
        q_val = _eval_trig_poly(bc, np.array([0.5 * (a_dom + b_dom)]), a_dom, b_dom)
        if q_val != 0:
            r_old = r_handle
            def r_handle(x, _qv=q_val):
                return r_old(x) / _qv

    # ------------------------------------------------------------------
    # 7.  Poles and residues
    # ------------------------------------------------------------------
    poles = np.array([])
    residues = np.array([])
    if nu > 0:
        try:
            poles = _find_trig_poles(bc, a_dom, b_dom)
            t_eps = max(tol, 1e-7)
            residues = t_eps * (r_handle(poles + t_eps) - r_handle(poles - t_eps)) / 2.0
        except Exception:
            pass

    return r_handle, ac, bc, mu, nu, poles, residues


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
        result = result + coeffs[j] * np.exp(1j * 2 * np.pi * k * (x - a_dom) / period)

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
    x_poles = a_dom + period * np.log(z_roots) / (2j * np.pi)
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
        from chebfunjax.utils.transforms import vals2coeffs
        import jax.numpy as _jnp
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
    tol = 1e-10
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
    period = b - a

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
