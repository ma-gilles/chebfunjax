# uses-numpy: tridiagonal eigenvalue problem and Legendre-to-Chebyshev conversion are not JIT-safe
"""Prolate spheroidal wave functions (PSWFs) and associated quadrature.

Translated from MATLAB Chebfun (commit 7574c77): pswf.m, pswfpts.m.
Original: Copyright 2020 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

References
----------
[1] H. Xiao, V. Rokhlin and N. Yarvin, "Prolate spheroidal wave functions,
    quadrature and interpolation", Inverse Problems, 17 (2001), 805–838.
[2] J. Ma, V. Rokhlin and S. Wandzura, "Generalised Gaussian Quadrature Rules
    for Systems of Arbitrary Functions", SIAM J. Numer. Anal., 1996.
[3] https://dlmf.nist.gov/30.4
"""

from __future__ import annotations

import numpy as np

# ===========================================================================
# Public API
# ===========================================================================


def pswf(
    N: int | np.ndarray,
    c: float,
    domain: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prolate spheroidal wave functions.

    Compute the Nth PSWF(s) with bandwidth parameter C on [-1, 1] via an
    eigenvalue problem on the truncated tridiagonal matrix (Xiao et al. [1]).

    P is the Nth eigenfunction of the prolate differential operator

        [(1 - x^2) P'(x)]' + [lam - c^2 x^2] P(x) = 0,

    normalised so that int_{-1}^{1} P_N(x)^2 dx = 2/(2N+1), with the
    sign convention sign(P(0)) = (-1)^(N/2) for even N and
    sign(P'(0)) = (-1)^((N-1)/2) for odd N (DLMF 30.4.1).

    Parameters
    ----------
    N : int or array_like of int
        PSWF order(s), non-negative integer(s).
    c : float
        Bandwidth parameter (positive scalar).
    domain : (float, float), optional
        Domain for output.  Defaults to (-1, 1).

    Returns
    -------
    x_grid : np.ndarray, shape (M,)
        Chebyshev grid on *domain* at which the functions are evaluated.
    P : np.ndarray, shape (M,) or (M, len(N))
        PSWF values at x_grid.  Single column for scalar N.
    lam : np.ndarray, shape (len(N),) or scalar
        Eigenvalue(s).

    Notes
    -----
    The algorithm expands the PSWF in Legendre polynomials up to degree M,
    where M is chosen adaptively so that the trailing Legendre coefficients
    are below machine precision.  The Legendre coefficients are then evaluated
    pointwise using Clenshaw's algorithm.

    Provenance
    ----------
    MATLAB source : pswf.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2020 by The University of Oxford
        and The Chebfun Developers.

    Examples
    --------
    >>> import numpy as np
    >>> x, p, lam = pswf(0, np.pi)
    >>> abs(p[len(p)//2])   # P_0 at x=0 should be ~ 1
    0.9...

    See Also
    --------
    pswfpts
    """
    N_arr = np.atleast_1d(np.asarray(N, dtype=int)).ravel()
    if not (np.all(N_arr >= 0) and np.all(N_arr == np.round(N_arr))):
        raise ValueError("N must be non-negative integer(s).")
    if not (np.isscalar(c) and c > 0):
        raise ValueError("c must be a positive scalar.")

    V, lam_all = _build_legendre_coeffs(N_arr, float(c))

    # Evaluate on a fine grid via Clenshaw recurrence
    a_dom, b_dom = float(domain[0]), float(domain[1])
    n_grid = max(200, 4 * V.shape[0])
    # Map grid from domain to [-1, 1]
    t = np.linspace(-1.0, 1.0, n_grid)
    x_grid = 0.5 * ((b_dom - a_dom) * t + (a_dom + b_dom))

    P = _legpolyval(V, t)  # shape (n_grid, len(N_arr))

    if np.isscalar(N) or (hasattr(N, '__len__') and len(N) == 1):
        P = P[:, 0]
        lam_all = lam_all[0]

    return x_grid, P, lam_all


def pswfpts(
    N: int,
    c: float,
    domain: tuple[float, float] = (-1.0, 1.0),
    quadtype: str = "roots",
) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes and weights from PSWF roots.

    Parameters
    ----------
    N : int
        Non-negative integer: the *order* of the PSWF whose roots are sought.
        ``X = PSWFPTS(N, C)`` returns the N roots of the Nth PSWF.
    c : float
        Bandwidth parameter (positive scalar).
    domain : (float, float), optional
        Scale nodes and weights to this interval.  Defaults to (-1, 1).
    quadtype : {'roots', 'GGQ'}, optional
        ``'roots'``  — interpolatory quadrature with nodes at PSWF zeros.
        ``'GGQ'``    — generalised Gauss quadrature exact for PSWFs 0…2N-1.

    Returns
    -------
    x : np.ndarray, shape (N,)
        Quadrature nodes in *domain*.
    w : np.ndarray, shape (N,)
        Quadrature weights for *domain*.

    Provenance
    ----------
    MATLAB source : pswfpts.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2020 by The University of Oxford
        and The Chebfun Developers.

    Examples
    --------
    >>> import numpy as np
    >>> x, w = pswfpts(5, np.pi)
    >>> len(x)
    5

    See Also
    --------
    pswf
    """
    N = int(N)
    if N < 0:
        raise ValueError("N must be a non-negative integer.")
    if not (np.isscalar(c) and c > 0):
        raise ValueError("c must be a positive scalar.")
    if N == 0:
        return np.array([]), np.array([])

    if quadtype.lower() == "ggq":
        x, w = _pswf_ggq(N, float(c))
    else:
        x, w = _pswf_roots_quad(N, float(c))

    # Enforce exact symmetry
    x = 0.5 * (x + (-x[::-1]))
    w = 0.5 * (w + w[::-1])

    # Scale to domain
    a, b = float(domain[0]), float(domain[1])
    if not (a == -1.0 and b == 1.0):
        x = b * (x + 1.0) / 2.0 + a * (1.0 - x) / 2.0
        w = (b - a) / 2.0 * w

    return x, w


# ===========================================================================
# Private helpers
# ===========================================================================


def _build_legendre_coeffs(
    N_arr: np.ndarray,
    c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build normalised Legendre coefficient matrix for PSWFs N_arr with bandwidth c."""
    M = int(max(np.ceil(2.0 * np.sqrt(c) * np.max(N_arr) + 1), 2 * c, 20))

    ishappy = False
    tol = np.finfo(float).eps
    count = 0

    while not ishappy:
        j = np.arange(M + 1, dtype=float)
        Asub = c ** 2 * j * (j - 1) / ((2 * j - 1) * np.sqrt(np.maximum((2 * j - 3) * (2 * j + 1), 1e-300)))
        Adia = j * (j + 1) + c ** 2 * (2 * j * (j + 1) - 1) / ((2 * j + 3) * (2 * j - 1) + 1e-300)
        Asup = c ** 2 * (j + 2) * (j + 1) / ((2 * j + 3) * np.sqrt(np.maximum((2 * j + 5) * (2 * j + 1), 1e-300)))

        A = np.diag(Adia) + np.diag(Asub[2:], -2) + np.diag(Asup[:-2], 2)

        # Split into even and odd subproblems for efficiency
        Ae = A[0::2, 0::2]
        Ao = A[1::2, 1::2]

        lame, Ve = np.linalg.eigh(Ae)
        lamo, Vo = np.linalg.eigh(Ao)

        # Sort ascending
        idx_e = np.argsort(lame)
        lame = lame[idx_e]
        Ve = Ve[:, idx_e]
        idx_o = np.argsort(lamo)
        lamo = lamo[idx_o]
        Vo = Vo[:, idx_o]

        # Interleave
        V_full = np.zeros((M + 1, M + 1))
        V_full[0::2, 0::2] = Ve
        V_full[1::2, 1::2] = Vo
        lam_full = np.zeros(M + 1)
        lam_full[0::2] = lame
        lam_full[1::2] = lamo

        # Check convergence on requested columns
        col_idx = N_arr  # 0-based
        tail_sum = np.sum(np.abs(V_full[-4:, col_idx])) / (2.0 * max(len(N_arr), 1))
        ishappy = tail_sum < tol or count > 10
        if not ishappy:
            M *= 2
        count += 1

    # Extract columns for requested N, unnormalise
    V = V_full[:, N_arr] * np.sqrt(np.arange(M + 1)[:, None] + 0.5)
    lam = lam_full[N_arr]

    # Trim trailing small coefficients
    row_max = np.max(np.abs(V), axis=1)
    last_nonzero = np.flatnonzero(row_max > tol)
    if len(last_nonzero) > 0:
        V = V[: last_nonzero[-1] + 1, :]

    # Rescale as per Wolfram/DLMF definition (sqrt(N+0.5) normalisation)
    V = V / np.sqrt(N_arr + 0.5)

    # Enforce sign convention
    _fix_signs(V, N_arr)

    return V, lam


def _fix_signs(V: np.ndarray, N_arr: np.ndarray) -> None:
    """Enforce sign convention in-place: P(0)>0 for even N, P'(0)>0 for odd N."""
    n_rows = V.shape[0]
    # P_k(0) = (-1)^(k/2) * binomial(k,k/2)/2^k   (nonzero only for even k)
    m_even = np.arange((n_rows + 1) // 2)
    # Closed-form value of P_{2m}(0) = (-1)^m * (2m choose m) / 4^m
    L0 = np.zeros(n_rows)
    for mm in m_even:
        binom = 1.0
        for kk in range(1, mm + 1):
            binom *= (mm + kk) / kk
        L0[2 * mm] = ((-1) ** mm) * binom / (4.0 ** mm)

    # P'_{2m+1}(0) = (-1)^m * (2m+1) * (2m choose m) / 4^m ... use recurrence
    L0p_odd = np.zeros(n_rows)
    for mm in m_even:
        if 2 * mm + 1 < n_rows:
            binom = 1.0
            for kk in range(1, mm + 1):
                binom *= (mm + kk) / kk
            L0p_odd[2 * mm + 1] = ((-1) ** mm) * (2 * mm + 1) * binom / (4.0 ** mm)

    for col_idx, n_val in enumerate(N_arr):
        if n_val % 2 == 0:
            v0 = L0 @ V[:, col_idx]
            desired = 1.0 if (n_val // 2) % 2 == 0 else -1.0
            if np.sign(v0) != np.sign(desired) and v0 != 0:
                V[:, col_idx] *= -1.0
        else:
            vp0 = L0p_odd @ V[:, col_idx]
            desired = 1.0 if ((n_val - 1) // 2) % 2 == 0 else -1.0
            if np.sign(vp0) != np.sign(desired) and vp0 != 0:
                V[:, col_idx] *= -1.0


def _legpolyval(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate Legendre expansions at x via modified Clenshaw recurrence.

    Parameters
    ----------
    c : np.ndarray, shape (n_terms, n_funcs)
        Legendre coefficients; c[k, j] is the coefficient of P_k for function j.
    x : np.ndarray, shape (m,)
        Evaluation points in [-1, 1].

    Returns
    -------
    y : np.ndarray, shape (m, n_funcs)
    """
    x = np.asarray(x).ravel()
    c = np.atleast_2d(c) if c.ndim == 1 else c
    n = c.shape[0] - 1
    m = len(x)
    n_funcs = c.shape[1]

    bk1 = np.zeros((m, n_funcs))
    bk2 = np.zeros((m, n_funcs))

    for k in range(n, 0, -1):
        bk = c[k, :][None, :] + (2 * k + 1) / (k + 1) * x[:, None] * bk1 - (k + 1) / (k + 2) * bk2
        bk2 = bk1
        bk1 = bk

    y = c[0, :][None, :] + x[:, None] * bk1 - 0.5 * bk2
    return y


def _pswf_roots_quad(N: int, c: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolatory quadrature with nodes at roots of the Nth PSWF."""
    # Get Legendre coefficients of PSWFs 0..N
    V_all, _ = _build_legendre_coeffs(np.arange(N + 1), c)
    # Roots of the Nth PSWF via generalised Legendre companion matrix
    v_N = V_all[:, N]
    x = _legroots(v_N)

    # Weights: solve P @ w = S where P is the Vandermonde-Legendre matrix
    # and S are the integrals of PSWFs 0..N-1 (= 2*c_{0,k} by orthogonality)
    if N == 0:
        return x, np.array([])

    V_sub = V_all[:, :N]      # Legendre coeffs for PSWFs 0..N-1
    S = 2.0 * V_sub[0, :]     # integrals (using <P_0, pswf_k> = 2*coeff_0)
    P_mat = _legpolyval(V_sub, x)  # shape (N, N)
    if P_mat.shape[0] == N and P_mat.shape[1] == N:
        try:
            w = np.linalg.solve(P_mat.T, S)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(P_mat.T, S, rcond=None)[0]
    else:
        w = np.linalg.lstsq(P_mat.T, S, rcond=None)[0]

    return x, w


def _pswf_ggq(N: int, c: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Gauss quadrature exact for PSWFs 0..2N-1 with bandwidth c."""
    # Initial guess: Gauss-Legendre nodes (scipy)
    from scipy.special import roots_legendre
    x, _ = roots_legendre(N)
    x = x.astype(float)
    # KTE map for better Newton convergence
    a = 0.5
    x = np.arcsin(a * x) / np.arcsin(a)

    # Continuation in c for large c > N
    if c > N:
        c_steps = list(range(N, int(c), 2)) + [int(c), c]
        for cc in c_steps:
            x, w = _pswf_ggq_step(N, float(cc), x)
        return x, w

    return _pswf_ggq_step(N, c, x)


def _pswf_ggq_step(
    N: int, c: float, x_init: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One Newton iteration step for GGQ."""
    V, _ = _build_legendre_coeffs(np.arange(2 * N), c)
    S = 2.0 * V[0, :]   # shape (2N,)

    # Differentiate: use recurrence for Legendre derivative
    # d/dx sum_k c_k P_k(x) = sum_k c_k P_k'(x)
    # P_k'(x) can be computed via ultraspherical: P_k'(x) = (2k-1) C_{k-1}^{3/2}
    # Simple approach: numerical differentiation within the Clenshaw
    Vp = _legderiv(V)

    x = x_init.copy()
    w = np.zeros(N)
    for _ in range(20):
        Pvals = _legpolyval(V, x)    # shape (N, 2N): row i = pswf evals at x[i]
        Ppvals = _legpolyval(Vp, x)  # shape (N, 2N)

        # Build (2N, 2N) system: interleave function and derivative rows
        # A[2k, :] = PSWF values at x[k]; A[2k+1, :] = PSWF derivatives at x[k]
        A = np.zeros((2 * N, 2 * N))
        A[0::2, :] = Pvals    # (N, 2N) -> rows 0,2,4,...
        A[1::2, :] = Ppvals   # (N, 2N) -> rows 1,3,5,...

        try:
            # Solve A @ [w_1, dx_1, w_2, dx_2, ...] = S
            sol = np.linalg.solve(A, S)
        except np.linalg.LinAlgError:
            break

        # sol[2k] = weight for node k, sol[2k+1] = Newton correction for x[k]
        w_iter = sol[0::2]
        dx = sol[1::2] / (np.abs(w_iter) + 1e-300)
        x = x + dx
        if np.linalg.norm(dx, np.inf) < 1e-12:
            break

    # Final weights from N-point quadrature system
    Pvals = _legpolyval(V[:, :N], x)  # use first N PSWFs, shape (N, N)
    S_N = 2.0 * V[0, :N]
    try:
        w = np.linalg.solve(Pvals.T, S_N)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Pvals.T, S_N, rcond=None)[0]

    return x, w


def _legderiv(V: np.ndarray) -> np.ndarray:
    """Differentiate a matrix of Legendre series (in-place coefficients).

    For c = [c_0, c_1, ..., c_n], the derivative has Legendre coefficients
    d_k = (2k+1) * sum_{j=k+1, j+k odd} c_j.
    """
    n, m = V.shape
    Vp = np.zeros_like(V)
    for k in range(n - 1):
        for j in range(k + 1, n):
            if (j + k) % 2 == 1:
                Vp[k, :] += (2 * k + 1) * V[j, :]
    return Vp


def _legroots(v: np.ndarray) -> np.ndarray:
    """Real roots of a Legendre series in [-1, 1] via companion matrix."""
    # Remove trailing zeros
    last = np.flatnonzero(v)
    if len(last) == 0:
        return np.array([])
    v = v[: last[-1] + 1]
    n = len(v) - 1
    if n == 0:
        return np.array([])

    # Legendre recursion: a_k = (k+1)/(2k+1), g_k = k/(2k+1)
    N_arr = np.arange(n)
    a = (N_arr + 1.0) / (2 * N_arr + 1.0)
    g = N_arr / (2 * N_arr + 1.0)

    C = np.diag(a[:n - 1], 1) + np.diag(g[1:n], -1)
    C[-1, :] = -v[:n] / v[n]
    C[-1, n - 2] = -(v[n - 1] / v[n]) + (1.0 - 1.0 / n) * v[n]

    B = np.eye(n)
    B[-1, -1] = (2.0 - 1.0 / n) * v[n]

    try:
        x = np.linalg.eigvals(np.linalg.solve(B, C))
    except np.linalg.LinAlgError:
        x = np.linalg.eigvals(C)

    # Keep real roots in [-1, 1]
    x = x[np.abs(x.imag) < 1e-12].real
    x = x[np.abs(x) <= 1.0 + 1e-10]
    x = np.clip(x, -1.0, 1.0)
    return np.sort(x)
