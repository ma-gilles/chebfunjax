"""Spectral differentiation, integration, and cumulative sum matrices.

Provides ``diffmat``, ``cumsummat``, ``intmat``, ``introw``, and ``diffrow`` —
the core spectral-collocation building blocks.

Translated from MATLAB Chebfun (commit 7574c77): diffmat.m, intmat.m,
cumsummat.m, introw.m, diffrow.m, @chebcolloc/baryDiffMat.m,
@chebcolloc2/diffmat.m, @chebcolloc2/cumsummat.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.utils.quadrature import chebpts, chebweights

# ============================================================================
# Helper: barycentric weights and angles for Chebyshev points
# ============================================================================

def _cheb2_barywts(n: int) -> jnp.ndarray:
    """Barycentric weights for n Chebyshev points of the 2nd kind.

    Alternating +/-1 with half-weight at endpoints. Normalised so that
    ``max(|w|) == 1`` and the last entry is positive.

    Provenance
    ----------
    MATLAB source : @chebtech2/barywts.m
    Chebfun commit: 7574c77
    """
    if n == 0:
        return jnp.array([], dtype=jnp.float64)
    if n == 1:
        return jnp.array([1.0], dtype=jnp.float64)

    # v = [1, 1, ..., 1, 0.5];  v(end-1:-2:1) = -1;  v(1) = 0.5*v(1)
    v = jnp.ones(n, dtype=jnp.float64)
    v = v.at[-1].set(0.5)
    idx = jnp.arange(n, dtype=jnp.int32)
    negate_mask = (idx % 2 == (n % 2)) & (idx <= n - 2)
    v = jnp.where(negate_mask, -1.0, v)
    v = v.at[0].set(v[0] * 0.5)
    return v


def _cheb2_angles(n: int) -> jnp.ndarray:
    """Angles theta_k = acos(x_k) for n 2nd-kind Chebyshev points.

    Returns theta in descending order: theta_k = (n-1-k)*pi/(n-1), k=0..n-1.

    Provenance
    ----------
    MATLAB source : @chebtech2/angles.m
    Chebfun commit: 7574c77
    """
    if n == 0:
        return jnp.array([], dtype=jnp.float64)
    if n == 1:
        return jnp.array([jnp.pi / 2], dtype=jnp.float64)

    m = n - 1
    return jnp.arange(m, -1, -1, dtype=jnp.float64) * jnp.pi / m


def _cheb1_barywts(n: int) -> jnp.ndarray:
    """Barycentric weights for n Chebyshev points of the 1st kind.

    Provenance
    ----------
    MATLAB source : @chebtech1/barywts.m
    Chebfun commit: 7574c77
    """
    if n == 0:
        return jnp.array([], dtype=jnp.float64)
    if n == 1:
        return jnp.array([1.0], dtype=jnp.float64)

    # v = sin(((n-1:-1:0)+0.5)*pi/n)  (MATLAB 1-based: k=n:-1:1 -> 0-based: k=n-1:-1:0)
    k = jnp.arange(n - 1, -1, -1, dtype=jnp.float64)
    v = jnp.sin((k + 0.5) * jnp.pi / n)

    # Flipping trick for symmetry and improved relative accuracy
    half = n // 2
    v = v.at[:half].set(v[n - 1:n - 1 - half:-1])

    # Alternate signs: v(end-1:-2:1) = -v(end-1:-2:1) in MATLAB (0-based: n-2, n-4, ...)
    idx = jnp.arange(n, dtype=jnp.int32)
    negate_mask = (idx % 2 == (n % 2)) & (idx <= n - 2)
    v = jnp.where(negate_mask, -v, v)
    return v


def _cheb1_angles(n: int) -> jnp.ndarray:
    """Angles theta_k = acos(x_k) for n 1st-kind Chebyshev points.

    Returns theta in descending order: theta_k = (n - 0.5 - k)*pi/n, k=0..n-1.

    Provenance
    ----------
    MATLAB source : @chebtech1/angles.m
    Chebfun commit: 7574c77
    """
    # MATLAB: out = (n-.5:-1:.5).'*pi/n;
    return (jnp.arange(n - 0.5, 0, -1, dtype=jnp.float64)) * jnp.pi / n


# ============================================================================
# Core: barycentric differentiation matrix
# ============================================================================

def _bary_diffmat(x: jnp.ndarray, w: jnp.ndarray, k: int = 1,
                  t: jnp.ndarray | None = None) -> jnp.ndarray:
    """Barycentric differentiation matrix.

    Given interpolation nodes ``x`` and barycentric weights ``w``, returns
    the ``k``-th order differentiation matrix ``D`` such that ``D @ f_values``
    approximates the ``k``-th derivative at the same nodes.

    Parameters
    ----------
    x : jnp.ndarray, shape (n,)
        Interpolation nodes.
    w : jnp.ndarray, shape (n,)
        Barycentric weights.
    k : int, default 1
        Order of differentiation.
    t : jnp.ndarray, shape (n,) or None
        Angles ``acos(x)``. When provided, pairwise differences of ``x`` are
        computed more accurately using the trigonometric identity
        ``x_i - x_j = 2 sin((t_i+t_j)/2) sin((t_i-t_j)/2)`` [4].

    Returns
    -------
    D : jnp.ndarray, shape (n, n)
        Differentiation matrix.

    Notes
    -----
    Uses the 'hybrid' formula of Schneider & Werner [1] and Welfert [2]
    proposed by Tee [3] with the accuracy tricks from [4].

    References
    ----------
    .. [1] C. Schneider and W. Werner, "Some new aspects of rational
       interpolation", Math. Comp. 47:285-299, 1986.
    .. [2] B. D. Welfert, "Generation of pseudospectral matrices I",
       SINUM 34:1640-1657, 1997.
    .. [3] T. W. Tee, "An adaptive rational spectral method for differential
       equations with rapidly varying solutions", Oxford DPhil Thesis, 2006.
    .. [4] R. Baltensperger and M. R. Trummer, "Spectral Differencing with
       a Twist", SIAM J. Sci. Comp. 24(5):1465-1487, 2003.

    Provenance
    ----------
    MATLAB source : @chebcolloc/baryDiffMat.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    n = x.shape[0]

    if n == 0:
        return jnp.array([], dtype=jnp.float64).reshape(0, 0)
    if n == 1:
        return jnp.zeros((1, 1), dtype=jnp.float64)
    if k == 0:
        return jnp.eye(n, dtype=jnp.float64)

    ii = jnp.arange(n)  # diagonal indices

    # --- Pairwise differences Dx ---
    if t is not None:
        # Trig identity for improved accuracy [4]:
        # MATLAB flips t before using: t = flipud(t)
        t_flip = t[::-1]
        # Dx[i,j] = 2*sin((t_i + t_j)/2)*sin((t_i - t_j)/2)
        Dx = 2.0 * jnp.sin((t_flip[:, None] + t_flip[None, :]) / 2.0) * \
                    jnp.sin((t_flip[:, None] - t_flip[None, :]) / 2.0)
    else:
        Dx = x[:, None] - x[None, :]

    # Flipping trick [4]: enforce antisymmetry for improved accuracy
    DxRot = jnp.rot90(Dx, 2)
    N = n
    # idxTo = rot90(~triu(ones(N))) : lower-left triangle in the rotated view
    triu_mask = jnp.triu(jnp.ones((N, N), dtype=bool))
    idxTo = jnp.rot90(~triu_mask)
    Dx = jnp.where(idxTo, -DxRot, Dx)

    # Set diagonal to 1 (avoid divide by zero)
    Dx = Dx.at[ii, ii].set(1.0)

    # Reciprocal of Dx
    Dxi = 1.0 / Dx

    # Ratio of weights: Dw[i,j] = w[j] / w[i], with zero diagonal
    Dw = w[None, :] / w[:, None]
    Dw = Dw.at[ii, ii].set(0.0)

    # --- k = 1 ---
    D = Dw * Dxi
    # Negative sum trick: set diagonal so that each row sums to zero
    D = D.at[ii, ii].set(0.0)
    D = D.at[ii, ii].set(-jnp.sum(D, axis=1))

    # Forcing symmetry for even N:
    # D(ii(end:-1:N-floor(N/2)+1)) = -D(ii(1:floor(N/2)))
    # In 0-based: D[n-1, n-1], D[n-2, n-2], ... D[n-floor(n/2), n-floor(n/2)]
    #           = -D[0,0], -D[1,1], ..., -D[floor(n/2)-1, floor(n/2)-1]
    half = n // 2
    diag_vals = jnp.diag(D)
    # Replace the last `half` diagonal entries with negatives of the first `half`
    new_diag = diag_vals.at[n - 1:n - 1 - half:-1].set(-diag_vals[:half])
    D = D.at[ii, ii].set(new_diag)

    if k == 1:
        return D

    # --- k = 2 ---
    D = 2.0 * D * (jnp.diag(D)[:, None] - Dxi)
    D = D.at[ii, ii].set(0.0)
    D = D.at[ii, ii].set(-jnp.sum(D, axis=1))

    if k == 2:
        return D

    # --- k = 3, 4, ... ---
    for order in range(3, k + 1):
        D = order * Dxi * (Dw * jnp.diag(D)[:, None] - D)
        D = D.at[ii, ii].set(0.0)
        D = D.at[ii, ii].set(-jnp.sum(D, axis=1))

    return D


# ============================================================================
# Public: diffmat
# ============================================================================

def diffmat(n: int, p: int = 1,
            domain: tuple[float, float] = (-1.0, 1.0),
            kind: int = 2) -> jnp.ndarray:
    """Spectral differentiation matrix on Chebyshev points.

    Returns the ``n x n`` differentiation matrix ``D`` of order ``p`` such
    that ``D @ f_values`` approximates the ``p``-th derivative of the
    interpolating polynomial at Chebyshev points of the given kind.

    Parameters
    ----------
    n : int
        Number of grid points.
    p : int, default 1
        Order of differentiation (0 returns identity).
    domain : (float, float), default (-1, 1)
        Interval ``[a, b]``. The matrix is scaled by ``(2 / (b - a))^p``.
    kind : {1, 2}, default 2
        1 for Chebyshev points of the 1st kind, 2 for the 2nd kind.

    Returns
    -------
    D : jnp.ndarray, shape (n, n)
        Spectral differentiation matrix.

    Examples
    --------
    >>> D = diffmat(5)
    >>> D.shape
    (5, 5)

    The matrix differentiates exactly for polynomials of degree < n:

    >>> import jax.numpy as jnp
    >>> x = chebpts(5, kind=2)
    >>> vals = x**3
    >>> deriv_vals = D @ vals
    >>> exact = 3 * x**2
    >>> float(jnp.max(jnp.abs(deriv_vals - exact)))  # ~0.0
    0.0

    Provenance
    ----------
    MATLAB source : diffmat.m, @chebcolloc/baryDiffMat.m,
        @chebcolloc2/diffmat.m, @chebcolloc1/diffmat.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm:
        [1] R. Baltensperger and M. R. Trummer, "Spectral Differencing
            with a Twist", SIAM J. Sci. Comp. 24(5):1465-1487, 2003.

    See Also
    --------
    cumsummat, intmat, diffrow, introw
    """
    if n <= 0:
        return jnp.array([], dtype=jnp.float64).reshape(0, 0)
    if p < 0:
        raise ValueError(
            f"Differentiation order p must be non-negative, got p={p}."
        )
    if p == 0 or n == 1:
        D = jnp.eye(n, dtype=jnp.float64)
    elif kind == 2:
        x = chebpts(n, kind=2)
        w = _cheb2_barywts(n)
        t = _cheb2_angles(n)
        D = _bary_diffmat(x, w, p, t)
    elif kind == 1:
        x = chebpts(n, kind=1)
        w = _cheb1_barywts(n)
        t = _cheb1_angles(n)
        D = _bary_diffmat(x, w, p, t)
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")

    # Scale to domain
    a, b = domain
    scl = (2.0 / (b - a)) ** p
    return scl * D


# ============================================================================
# Public: cumsummat
# ============================================================================

def cumsummat(n: int,
              domain: tuple[float, float] = (-1.0, 1.0),
              kind: int = 2) -> jnp.ndarray:
    """Indefinite integration (cumulative sum) matrix on Chebyshev points.

    Returns the ``n x n`` matrix ``Q`` such that ``Q @ f_values`` gives the
    values of the antiderivative of the interpolating polynomial at the same
    Chebyshev points, with the convention that the antiderivative is zero at
    the left endpoint.

    Parameters
    ----------
    n : int
        Number of Chebyshev points.
    domain : (float, float), default (-1, 1)
        Interval ``[a, b]``. The matrix is scaled by ``(b - a) / 2``.
    kind : {1, 2}, default 2
        1 for Chebyshev points of the 1st kind, 2 for the 2nd kind.

    Returns
    -------
    Q : jnp.ndarray, shape (n, n)
        Integration matrix.

    Notes
    -----
    The matrix is constructed via the coefficient-space integration recurrence:

    1. ``T_inv``: values -> Chebyshev coefficients
    2. ``B``: coefficient integration operator (truncating the highest mode)
    3. ``T``: coefficients -> values

    Then ``Q = T @ B @ T_inv``, with boundary correction so that ``Q[0, :] = 0``
    (for kind=2, ensuring the antiderivative vanishes at -1).

    Provenance
    ----------
    MATLAB source : cumsummat.m, @chebcolloc2/cumsummat.m,
        @chebcolloc1/cumsummat.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    diffmat, intmat, introw
    """
    if n <= 0:
        return jnp.array([], dtype=jnp.float64).reshape(0, 0)
    if n == 1:
        # Q maps a single value to its integral from the left endpoint.
        # Convention: the antiderivative at -1 is 0, so Q(1) = 1 for the
        # trivial case. (MATLAB returns 1 for N-1=0.)
        a, b = domain
        scl = 0.5 * (b - a)
        return jnp.array([[1.0]], dtype=jnp.float64) * scl

    # Build in the standard domain [-1, 1], then scale.
    Q = _cumsummat_standard(n, kind)

    # Scale to domain
    a, b = domain
    scl = 0.5 * (b - a)
    return scl * Q


def _cumsummat_standard(n: int, kind: int = 2) -> jnp.ndarray:
    """Build the cumsummat on [-1, 1] for n points of the given kind.

    Follows the MATLAB pattern:
        T = chebtech2.coeffs2vals(eye(N+1))
        Tinv = chebtech2.vals2coeffs(eye(N+1))
        B = integration operator in coefficient space
        Q = T @ B @ Tinv

    Note: MATLAB's chebtech2.coeffs2vals / vals2coeffs handle matrix inputs
    (column-by-column). Our transforms.coeffs2vals / vals2coeffs are 1D only,
    so we build the Chebyshev-Vandermonde matrix directly.
    """
    N = n - 1  # MATLAB convention: N = N - 1

    # Build T (coeffs -> values) and Tinv (values -> coeffs) as matrices.
    # T[i, j] = T_j(x_i) where T_j is the j-th Chebyshev polynomial
    # and x_i are the Chebyshev points.
    T, Tinv = _cheb_vandermonde(n, kind)

    # Integration operator in coefficient space.
    # c_k -> integral coeffs:
    #   B has sub-diagonal 1/(2k) and super-diagonal -1/(2(k-1)),
    #   with special handling of first column and first row.
    k = jnp.arange(1, N + 1, dtype=jnp.float64)
    k2 = 2.0 * (k - 1.0)
    k2 = k2.at[0].set(1.0)  # avoid divide by zero; will be overwritten

    # B is (N+1) x (N+1) acting on coefficients [c_0, ..., c_N]
    # Sub-diagonal: B[j, j-1] = 1/(2*j) for j=1..N
    # Super-diagonal: B[j, j+1] = -1/(2*(j-1)) for j=0..N-1 (but j=0 case is special)
    B = jnp.diag(1.0 / (2.0 * k), -1) - jnp.diag(1.0 / k2, 1)

    # First row: B(1,:) = sum(diag(v)*B(2:N+1,:), 1) in MATLAB
    # v = [1, -1, 1, -1, ...] of length N
    v = jnp.ones(N, dtype=jnp.float64)
    v = v.at[1::2].set(-1.0)
    # MATLAB: B(1,:) = sum(diag(v)*B(2:N+1,:), 1)
    # diag(v) * B[1:N+1, :] means multiply row j of B[1:N+1,:] by v[j]
    B_lower = B[1:, :]  # rows 1..N (0-based)
    B_row0 = jnp.sum(v[:, None] * B_lower, axis=0)
    B = B.at[0, :].set(B_row0)

    # Double the first column: B(:,1) = 2*B(:,1)
    B = B.at[:, 0].set(2.0 * B[:, 0])

    Q = T @ B @ Tinv

    if kind == 2:
        # Make exact: Q(1,:) = 0  (MATLAB 1-based -> 0-based first row)
        Q = Q.at[0, :].set(0.0)

    return Q


def _cheb_vandermonde(n: int, kind: int = 2) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build Chebyshev-Vandermonde matrix T and its inverse Tinv.

    T[i, j] = T_j(x_i), where x_i are Chebyshev points of the given kind.
    Tinv is the inverse: values -> coefficients.

    Returns (T, Tinv) both of shape (n, n).
    """
    x = chebpts(n, kind=kind)
    theta = jnp.arccos(x)
    j_idx = jnp.arange(n, dtype=jnp.float64)
    # T[i, j] = cos(j * theta_i)
    T = jnp.cos(j_idx[None, :] * theta[:, None])
    Tinv = jnp.linalg.inv(T)
    return T, Tinv


# ============================================================================
# Public: intmat
# ============================================================================

def intmat(n: int, p: int = 1,
           domain: tuple[float, float] = (-1.0, 1.0)) -> jnp.ndarray:
    """Spectral integration matrix on 2nd-kind Chebyshev points.

    Returns the ``n x n`` matrix ``K`` such that ``K @ f_values`` gives the
    values of the ``p``-fold integral evaluated at the Chebyshev points.

    ``intmat`` integrates by repeated application of ``cumsummat``:
    ``K = cumsummat^p``.

    Parameters
    ----------
    n : int
        Number of grid points.
    p : int, default 1
        Order of integration.
    domain : (float, float), default (-1, 1)
        Interval ``[a, b]``.

    Returns
    -------
    K : jnp.ndarray, shape (n, n)
        Integration matrix.

    Provenance
    ----------
    MATLAB source : intmat.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2015 by The University of Oxford
        and The Chebfun Developers.
    Note: The original MATLAB code builds the integration matrix using
        chebfun objects; this implementation uses cumsummat directly.

    See Also
    --------
    cumsummat, diffmat, introw
    """
    if n <= 0:
        return jnp.array([], dtype=jnp.float64).reshape(0, 0)
    if p < 0:
        raise ValueError(
            f"Integration order p must be non-negative, got p={p}."
        )
    if p == 0:
        return jnp.eye(n, dtype=jnp.float64)

    Q = cumsummat(n, domain=domain)
    K = Q
    for _ in range(1, p):
        K = Q @ K
    return K


# ============================================================================
# Public: introw
# ============================================================================

def introw(n: int,
           domain: tuple[float, float] = (-1.0, 1.0)) -> jnp.ndarray:
    """Clenshaw-Curtis quadrature weights (last row of integration matrix).

    Returns a row vector of ``n`` Clenshaw-Curtis quadrature coefficients,
    equivalent to the last row of ``intmat(n)``.

    Parameters
    ----------
    n : int
        Number of Chebyshev points.
    domain : (float, float), default (-1, 1)
        Interval ``[a, b]``.

    Returns
    -------
    r : jnp.ndarray, shape (n,)
        Quadrature weights.

    Provenance
    ----------
    MATLAB source : introw.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    diffmat, diffrow, intmat
    """
    # MATLAB: [~, r] = chebpts(n, dom)
    # chebpts returns (points, weights) when nargout=2.
    # The weights are Clenshaw-Curtis weights scaled to the domain.
    a, b = domain
    w = chebweights(n, kind=2)
    scl = (b - a) / 2.0
    return scl * w


# ============================================================================
# Public: diffrow
# ============================================================================

def diffrow(n: int, p: int, x: float,
            domain: tuple[float, float] = (-1.0, 1.0)) -> jnp.ndarray:
    """One row of the spectral differentiation matrix.

    Returns the first row (if ``x == domain[0]``) or the last row
    (if ``x == domain[1]``) of ``diffmat(n, p, domain)``.

    Parameters
    ----------
    n : int
        Number of Chebyshev points.
    p : int
        Order of differentiation.
    x : float
        Evaluation point: must be ``domain[0]`` (left) or ``domain[1]`` (right).
    domain : (float, float), default (-1, 1)
        Interval ``[a, b]``.

    Returns
    -------
    r : jnp.ndarray, shape (n,)
        One row of the differentiation matrix.

    Raises
    ------
    ValueError
        If ``x`` is not an endpoint of the domain.

    Provenance
    ----------
    MATLAB source : diffrow.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    diffmat, intmat, introw
    """
    a, b = domain
    D = diffmat(n, p, domain)

    if x == a:
        return D[0, :]
    elif x == b:
        return D[-1, :]
    else:
        raise ValueError(
            f"x must be an endpoint of the domain [{a}, {b}], got x={x}. "
            f"Only boundary rows are supported. Use diffmat() for interior points."
        )
