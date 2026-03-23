"""Polynomial coefficient and value transforms.

Chebyshev <-> Legendre, Chebyshev <-> Jacobi, and Chebyshev values <-> coefficients.

Translated from MATLAB Chebfun (commit 7574c77): cheb2leg.m, leg2cheb.m,
cheb2jac.m, jac2cheb.m, chebvals2legcoeffs.m, chebcoeffs2legvals.m,
chebcoeffs2chebvals.m, chebvals2chebcoeffs.m, and related files.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import jax.numpy as jnp

# ===========================================================================
# Chebyshev values <-> coefficients (DCT-I based)
# ===========================================================================

def vals2coeffs(values: jnp.ndarray) -> jnp.ndarray:
    """Convert values at 2nd-kind Chebyshev points to Chebyshev coefficients.

    Given values V_k = f(x_k) at Chebyshev points of the 2nd kind
    x_k = cos(k*pi/(n-1)), k=0,...,n-1, returns the Chebyshev coefficients c
    such that f(x) = c[0]*T_0(x) + c[1]*T_1(x) + ... + c[n-1]*T_{n-1}(x).

    This is equivalent to the inverse Discrete Cosine Transform of Type I.

    Parameters
    ----------
    values : jnp.ndarray, shape (n,)
        Function values at 2nd-kind Chebyshev points, ordered from x=1 to x=-1
        (descending order, MATLAB convention) or ascending — the transform is
        symmetric so ordering does not matter for even/odd structure.

    Returns
    -------
    coeffs : jnp.ndarray, shape (n,)
        Chebyshev series coefficients.

    Provenance
    ----------
    MATLAB source : @chebtech2/vals2coeffs.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Inverse DCT-I via FFT, see Mason & Handscomb,
        "Chebyshev Polynomials", Section 4.7 (2003).

    See Also
    --------
    coeffs2vals
    """
    n = values.shape[0]
    if n <= 1:
        return values

    # Mirror the values to fake a DCT-I using an FFT:
    # [v_{n-1}, v_{n-2}, ..., v_1, v_0, v_1, ..., v_{n-2}]
    tmp = jnp.concatenate([values[n - 1:0:-1], values[:n - 1]])

    coeffs = jnp.real(jnp.fft.ifft(tmp))

    # Truncate to first n entries
    coeffs = coeffs[:n]

    # Scale interior coefficients by 2
    coeffs = coeffs.at[1:n - 1].multiply(2.0)

    return coeffs


def coeffs2vals(coeffs: jnp.ndarray) -> jnp.ndarray:
    """Convert Chebyshev coefficients to values at 2nd-kind Chebyshev points.

    Given Chebyshev coefficients c, returns the values
    V_k = c[0]*T_0(x_k) + ... + c[n-1]*T_{n-1}(x_k)
    at Chebyshev points of the 2nd kind x_k = cos(k*pi/(n-1)).

    This is equivalent to the Discrete Cosine Transform of Type I.

    Parameters
    ----------
    coeffs : jnp.ndarray, shape (n,)
        Chebyshev series coefficients.

    Returns
    -------
    values : jnp.ndarray, shape (n,)
        Function values at 2nd-kind Chebyshev points (descending x order).

    Provenance
    ----------
    MATLAB source : @chebtech2/coeffs2vals.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: DCT-I via FFT, see Mason & Handscomb,
        "Chebyshev Polynomials", Sections 4.7 and 6.3 (2003).

    See Also
    --------
    vals2coeffs
    """
    n = coeffs.shape[0]
    if n <= 1:
        return coeffs

    # Scale interior coefficients by 1/2
    c = coeffs.at[1:n - 1].multiply(0.5)

    # Mirror the coefficients: [c_0, c_1, ..., c_{n-1}, c_{n-2}, ..., c_1]
    tmp = jnp.concatenate([c, c[n - 2:0:-1]])

    values = jnp.real(jnp.fft.fft(tmp))

    # Reverse and truncate: values at points cos(0), cos(pi/(n-1)), ..., cos(pi)
    # = [x=1, ..., x=-1] which is descending order
    values = values[n - 1::-1]

    return values


# ===========================================================================
# Chebyshev <-> Legendre (direct O(n^2) method)
# ===========================================================================

def cheb2leg(c_cheb: jnp.ndarray, *, normalize: bool = False) -> jnp.ndarray:
    """Convert Chebyshev coefficients to Legendre coefficients.

    C_LEG = cheb2leg(C_CHEB) converts the vector C_CHEB of Chebyshev
    coefficients to a vector C_LEG of Legendre coefficients such that
        C_CHEB[0]*T_0 + ... + C_CHEB[N-1]*T_{N-1}
      = C_LEG[0]*P_0 + ... + C_LEG[N-1]*P_{N-1},
    where P_k is the degree-k Legendre polynomial normalized so that
    max(|P_k|) = 1 (the standard normalization P_k(1) = 1).

    Parameters
    ----------
    c_cheb : jnp.ndarray, shape (n,)
        Chebyshev coefficients.
    normalize : bool, default False
        If True, use Legendre polynomials normalized to be orthonormal.

    Returns
    -------
    c_leg : jnp.ndarray, shape (n,)
        Legendre coefficients.

    Notes
    -----
    Uses the direct O(n^2) method based on evaluating the Chebyshev expansion
    on a fine grid and projecting onto Legendre polynomials via Clenshaw-Curtis
    quadrature. For N >= 513, the fast O(n log^2 n) algorithm of [1] would be
    preferable; this implementation uses the direct method for all n.

    References
    ----------
    .. [1] A. Townsend, M. Webb, and S. Olver, "Fast polynomial transforms
       based on Toeplitz and Hankel matrices", Math. Comp., 87, 2018.

    Provenance
    ----------
    MATLAB source : cheb2leg.m
    Chebfun commit: 7574c77
    Original authors: Alex Townsend, Nick Hale.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.

    See Also
    --------
    leg2cheb, cheb2jac, jac2cheb
    """
    n = c_cheb.shape[0]
    if n <= 1:
        c_leg = c_cheb
        if normalize and n == 1:
            c_leg = c_leg / jnp.sqrt(0.5)
        return c_leg

    return _cheb2leg_direct(c_cheb, normalize)


def _cheb2leg_direct(c_cheb: jnp.ndarray, normalize: bool) -> jnp.ndarray:
    """Convert Chebyshev to Legendre coefficients using the 3-term recurrence.

    Uses Clenshaw-Curtis quadrature on a 2N+1 grid to compute the Legendre
    projection integrals.
    """
    N = c_cheb.shape[0] - 1  # polynomial degree

    # 2*N+1 Chebyshev grid (descending order, cos(pi*k/(2N)) for k=0..2N)
    k = jnp.arange(2 * N + 1, dtype=jnp.float64)
    x = jnp.cos(0.5 * jnp.pi * k / N)

    # Values of the Chebyshev expansion on the 2N+1 grid via DCT
    # Pad coefficients to length 2N+1
    c_padded = jnp.concatenate([c_cheb, jnp.zeros(N, dtype=jnp.float64)])
    f = _dct1(c_padded)

    # Clenshaw-Curtis quadrature weights for 2N+1 points
    w = _cc_weights(2 * N + 1)

    # Build Legendre Vandermonde matrix via 3-term recurrence
    # L[i, j] = P_j(x_i)
    m = 2 * N + 1
    L = jnp.zeros((m, N + 1), dtype=jnp.float64)
    L = L.at[:, 0].set(1.0)
    L = L.at[:, 1].set(x)
    for j in range(1, N):
        # P_{j+1}(x) = ((2j+1)*x*P_j(x) - j*P_{j-1}(x)) / (j+1)
        L = L.at[:, j + 1].set(
            ((2 * j + 1) * x * L[:, j] - j * L[:, j - 1]) / (j + 1)
        )

    # Legendre coefficients: c_leg[j] = (2j+1)/2 * sum(w * f * P_j)
    scale = (2 * jnp.arange(N + 1, dtype=jnp.float64) + 1) / 2
    c_leg = scale * (L.T @ (f * w))

    if normalize:
        norms = jnp.sqrt(jnp.arange(N + 1, dtype=jnp.float64) + 0.5)
        c_leg = c_leg / norms

    return c_leg


def _dct1(c: jnp.ndarray) -> jnp.ndarray:
    """Compute a (scaled) DCT of type I using FFT.

    Returns T(X)*C where X = cos(pi*k/N) for k=0..N, and
    T(X) = [T_0, T_1, ..., T_N](X).
    N = len(c) - 1.
    """
    n = c.shape[0]
    if n <= 1:
        return c

    # Scale endpoints
    c_scaled = c.at[0].multiply(2.0)
    c_scaled = c_scaled.at[-1].multiply(2.0)

    # Mirror: [c_0, c_1, ..., c_{N}, c_{N-1}, ..., c_1]
    tmp = jnp.concatenate([c_scaled, c_scaled[-2:0:-1]])

    # FFT and take first n entries, then scale
    v = jnp.real(jnp.fft.fft(tmp))
    v = v[:n] / 2.0

    return v


def _idct1(v: jnp.ndarray) -> jnp.ndarray:
    """Inverse DCT-I: convert values on a Chebyshev grid to coefficients.

    Returns T(X)\\V where X = cos(pi*k/N), T(X) = [T_0, ..., T_N](X).
    """
    n = v.shape[0]
    if n <= 1:
        return v

    # Mirror: [v_0, v_1, ..., v_{N}, v_{N-1}, ..., v_1]
    tmp = jnp.concatenate([v, v[-2:0:-1]])

    c = jnp.real(jnp.fft.ifft(tmp))
    c = c[:n]

    # Scale endpoints by 1/2
    c = c.at[0].multiply(0.5)
    c = c.at[-1].multiply(0.5)

    return c


def _cc_weights(n: int) -> jnp.ndarray:
    """Clenshaw-Curtis quadrature weights for n 2nd-kind Chebyshev points.

    Points ordered descending: x_k = cos(k*pi/(n-1)), k=0,...,n-1.
    """
    if n == 1:
        return jnp.array([2.0], dtype=jnp.float64)
    if n == 2:
        return jnp.array([1.0, 1.0], dtype=jnp.float64)

    N = n - 1

    # Chebyshev moments: integral of T_k(x) over [-1,1]
    # = 2/(1-k^2) for even k, 0 for odd k
    c = jnp.zeros(N + 1, dtype=jnp.float64)
    k_even = jnp.arange(0, N + 1, 2, dtype=jnp.float64)
    c = c.at[0::2].set(2.0 / (1.0 - k_even**2))

    # Mirror for IFFT
    v = jnp.concatenate([c, c[N - 1:0:-1]])

    w = 2.0 * jnp.real(jnp.fft.ifft(v))
    w = w[:N + 1]

    # Halve endpoints
    w = w.at[0].set(w[0] / 2.0)
    w = w.at[N].set(w[N] / 2.0)

    return w


def leg2cheb(c_leg: jnp.ndarray, *, normalize: bool = False) -> jnp.ndarray:
    """Convert Legendre coefficients to Chebyshev coefficients.

    C_CHEB = leg2cheb(C_LEG) converts the vector C_LEG of Legendre
    coefficients to a vector C_CHEB of Chebyshev coefficients such that
        C_LEG[0]*P_0 + ... + C_LEG[N-1]*P_{N-1}
      = C_CHEB[0]*T_0 + ... + C_CHEB[N-1]*T_{N-1}.

    Parameters
    ----------
    c_leg : jnp.ndarray, shape (n,)
        Legendre coefficients.
    normalize : bool, default False
        If True, the input uses Legendre polynomials normalized to be
        orthonormal (i.e., multiply by sqrt(k+1/2) to get standard).

    Returns
    -------
    c_cheb : jnp.ndarray, shape (n,)
        Chebyshev coefficients.

    Notes
    -----
    Uses the direct O(n^2) method: evaluate the Legendre expansion at
    Chebyshev points via the Legendre Vandermonde matrix, then convert to
    Chebyshev coefficients via inverse DCT. For N > 512, the fast
    O(n log^2 n) algorithm of [1] would be preferable.

    References
    ----------
    .. [1] A. Townsend, M. Webb, and S. Olver, "Fast polynomial transforms
       based on Toeplitz and Hankel matrices", Math. Comp., 87, 2018.

    Provenance
    ----------
    MATLAB source : leg2cheb.m
    Chebfun commit: 7574c77
    Original authors: Alex Townsend, Nick Hale.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.

    See Also
    --------
    cheb2leg, cheb2jac, jac2cheb
    """
    n = c_leg.shape[0]
    if n <= 1:
        c = c_leg.copy()
        if normalize and n == 1:
            c = c * jnp.sqrt(0.5)
        return c

    # If normalized, undo the normalization: multiply by sqrt(k + 1/2)
    if normalize:
        norms = jnp.sqrt(jnp.arange(n, dtype=jnp.float64) + 0.5)
        c_leg = c_leg * norms

    return _leg2cheb_direct(c_leg)


def _leg2cheb_direct(c_leg: jnp.ndarray) -> jnp.ndarray:
    """Convert Legendre to Chebyshev coefficients using Vandermonde + vals2coeffs."""
    N = c_leg.shape[0] - 1  # degree

    # Chebyshev grid of N+1 points (descending order: cos(0)=1, ..., cos(pi)=-1)
    k = jnp.arange(N + 1, dtype=jnp.float64)
    x = jnp.cos(jnp.pi * k / N) if N > 0 else jnp.array([1.0], dtype=jnp.float64)

    # Legendre Vandermonde: L[i,j] = P_j(x_i) at descending points
    L = _legendre_vandermonde(N, x)

    # Values on Chebyshev grid (descending order)
    v_desc = L @ c_leg

    # Reverse to ascending order (x=-1 to x=1) for vals2coeffs
    v_asc = v_desc[::-1]

    # Convert values to Chebyshev coefficients
    c_cheb = vals2coeffs(v_asc)

    return c_cheb


def _legendre_vandermonde(N: int, x: jnp.ndarray) -> jnp.ndarray:
    """Compute the Legendre-Chebyshev Vandermonde matrix.

    L[i, j] = P_j(x[i]) for j = 0, ..., N using the 3-term recurrence.
    """
    m = x.shape[0]
    L = jnp.zeros((m, N + 1), dtype=jnp.float64)
    L = L.at[:, 0].set(1.0)
    if N >= 1:
        L = L.at[:, 1].set(x)
    for j in range(1, N):
        L = L.at[:, j + 1].set(
            ((2 * j + 1) * x * L[:, j] - j * L[:, j - 1]) / (j + 1)
        )
    return L


# ===========================================================================
# Chebyshev <-> Jacobi (direct method)
# ===========================================================================

def cheb2jac(c_cheb: jnp.ndarray, alpha: float, beta: float) -> jnp.ndarray:
    """Convert Chebyshev coefficients to Jacobi coefficients.

    Converts the Chebyshev expansion
        c_cheb[0]*T_0(x) + ... + c_cheb[N-1]*T_{N-1}(x)
    to a Jacobi expansion
        c_jac[0]*P_0^{(a,b)}(x) + ... + c_jac[N-1]*P_{N-1}^{(a,b)}(x),
    where P_k^{(a,b)} is the degree-k Jacobi polynomial for the weight
    function w(x) = (1-x)^a * (1+x)^b.

    Parameters
    ----------
    c_cheb : jnp.ndarray, shape (n,)
        Chebyshev coefficients.
    alpha : float
        Jacobi parameter alpha (exponent for 1-x).
    beta : float
        Jacobi parameter beta (exponent for 1+x).

    Returns
    -------
    c_jac : jnp.ndarray, shape (n,)
        Jacobi coefficients.

    Provenance
    ----------
    MATLAB source : cheb2jac.m
    Chebfun commit: 7574c77
    Original authors: Alex Townsend, Nick Hale.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.
    Algorithm:
        [1] A. Townsend, M. Webb, and S. Olver, "Fast polynomial transforms
            based on Toeplitz and Hankel matrices", Math. Comp., 87, 2018.

    See Also
    --------
    jac2cheb, cheb2leg, leg2cheb
    """
    n = c_cheb.shape[0]
    if n <= 1:
        return c_cheb

    # Special case: alpha=beta=0 is Legendre
    if alpha == 0.0 and beta == 0.0:
        return cheb2leg(c_cheb)

    # Special case: alpha=beta=-1/2 is Chebyshev T_n (up to scaling)
    if alpha == -0.5 and beta == -0.5:
        # T_n = scl[n] * P_n^{(-1/2,-1/2)}
        # scl[n] = prod_{k=0}^{n-1} (1/2+k)/(1+k)  with scl[0]=1
        nn = jnp.arange(n, dtype=jnp.float64)
        scl = jnp.concatenate([
            jnp.array([1.0], dtype=jnp.float64),
            jnp.cumprod((0.5 + nn[:-1]) / (1.0 + nn[:-1]))
        ])
        return c_cheb / scl

    return _cheb2jac_direct(c_cheb, alpha, beta)


def _cheb2jac_direct(c_cheb: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    """Convert Chebyshev to Jacobi coefficients via Vandermonde solve.

    Evaluates the Chebyshev expansion at N+1 Chebyshev-1st-kind points,
    then solves the Jacobi Vandermonde system to get Jacobi coefficients.
    This mirrors the MATLAB jac2cheb_direct approach (reversed).
    """
    N = c_cheb.shape[0] - 1  # degree
    if N <= 0:
        return c_cheb

    # Evaluate Chebyshev expansion at 1st-kind Chebyshev points
    n_pts = N + 1
    k = jnp.arange(n_pts, 0, -1, dtype=jnp.float64)
    x = jnp.cos((2.0 * k - 1.0) * jnp.pi / (2.0 * n_pts))

    # Evaluate the Chebyshev expansion at these points
    # T_k(x_j) = cos(k * arccos(x_j))
    theta = jnp.arccos(x)
    kk = jnp.arange(N + 1, dtype=jnp.float64)
    T = jnp.cos(kk[None, :] * theta[:, None])  # T[j, k] = T_k(x_j)
    v = T @ c_cheb  # values at the points

    # Jacobi Vandermonde at the same points
    P = _jacobi_vandermonde(N, x, a, b)

    # Solve P @ c_jac = v
    c_jac = jnp.linalg.solve(P, v)

    return c_jac


def jac2cheb(c_jac: jnp.ndarray, alpha: float, beta: float) -> jnp.ndarray:
    """Convert Jacobi coefficients to Chebyshev coefficients.

    Converts the Jacobi expansion
        c_jac[0]*P_0^{(a,b)}(x) + ... + c_jac[N-1]*P_{N-1}^{(a,b)}(x)
    to a Chebyshev expansion
        c_cheb[0]*T_0(x) + ... + c_cheb[N-1]*T_{N-1}(x).

    Parameters
    ----------
    c_jac : jnp.ndarray, shape (n,)
        Jacobi coefficients.
    alpha : float
        Jacobi parameter alpha (exponent for 1-x).
    beta : float
        Jacobi parameter beta (exponent for 1+x).

    Returns
    -------
    c_cheb : jnp.ndarray, shape (n,)
        Chebyshev coefficients.

    Provenance
    ----------
    MATLAB source : jac2cheb.m
    Chebfun commit: 7574c77
    Original authors: Alex Townsend, Nick Hale.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.

    See Also
    --------
    cheb2jac, cheb2leg, leg2cheb
    """
    n = c_jac.shape[0]
    if n <= 1:
        return c_jac

    # Special case: alpha=beta=0 is Legendre
    if alpha == 0.0 and beta == 0.0:
        return leg2cheb(c_jac)

    return _jac2cheb_direct(c_jac, alpha, beta)


def _jac2cheb_direct(c_jac: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    """Convert Jacobi to Chebyshev coefficients using Vandermonde evaluation."""
    N = c_jac.shape[0] - 1  # degree
    if N <= 0:
        return c_jac

    # Evaluate Jacobi expansion at 1st-kind Chebyshev points (like MATLAB)
    n_pts = N + 1
    # 1st-kind Chebyshev points: cos((2k-1)*pi/(2n)), k=1..n
    k = jnp.arange(n_pts, 0, -1, dtype=jnp.float64)
    x = jnp.cos((2.0 * k - 1.0) * jnp.pi / (2.0 * n_pts))

    # Jacobi Vandermonde matrix at these points
    P = _jacobi_vandermonde(N, x, a, b)

    # Values on Chebyshev grid
    v_cheb = P @ c_jac

    # Convert values at 1st-kind Chebyshev points to Chebyshev coefficients
    c_cheb = _vals2coeffs_kind1(v_cheb)

    return c_cheb


def _vals2coeffs_kind1(values: jnp.ndarray) -> jnp.ndarray:
    """Convert values at 1st-kind Chebyshev points to Chebyshev coefficients.

    Uses the relation: c_k = (2/n) sum_{j=0}^{n-1} v_j T_k(x_j), with
    appropriate scaling for k=0.
    """
    n = values.shape[0]
    if n <= 1:
        return values

    # DCT-III: c_k = (2/n) * sum_j v_j * cos(k*(2j+1)*pi/(2n))
    # Use FFT-based approach
    # Rearrange for FFT: we need DCT-III of the values

    # Using the relation to FFT:
    # The 1st-kind Chebyshev points are cos((2j-1)*pi/(2n)), j=1..n
    # values are ordered from j=n down to j=1 (ascending x), so
    # values[i] corresponds to x_i = cos((2*(n-i)-1)*pi/(2n))

    # For DCT-III via FFT:
    # Embed in length 2n DFT
    v = jnp.zeros(2 * n, dtype=jnp.float64)
    v = v.at[:n].set(values[::-1])  # reverse to match cos((2j-1)pi/2n), j=0..n-1
    v = v.at[n:].set(-values)  # antisymmetric extension

    # Shift by half-sample: multiply by exp(-i*pi*k/(2n)) in freq domain
    # Alternative: direct DCT-III computation
    # c_k = (2/n) sum_{j=0}^{n-1} values[j] * T_k(x_j)
    # where x_j = cos((2*(n-1-j)+1)*pi/(2n)) (ascending order)

    # Direct approach using explicit DCT-III formula
    j = jnp.arange(n, dtype=jnp.float64)
    k = jnp.arange(n, dtype=jnp.float64)

    # values[j] at x = cos((2j+1)*pi/(2n)) for j=0..n-1 (after reversing)
    vals_rev = values[::-1]

    # T_k(cos(theta)) = cos(k*theta), theta_j = (2j+1)*pi/(2n)
    theta = (2.0 * j + 1.0) * jnp.pi / (2.0 * n)
    # Cosine matrix: M[k,j] = cos(k * theta_j)
    M = jnp.cos(k[:, None] * theta[None, :])

    coeffs = (2.0 / n) * M @ vals_rev
    coeffs = coeffs.at[0].multiply(0.5)

    return coeffs


def _jacobi_vandermonde(
    N: int, x: jnp.ndarray, a: float, b: float
) -> jnp.ndarray:
    """Jacobi polynomial Vandermonde matrix P[i,j] = P_j^{(a,b)}(x[i]).

    Uses the standard three-term recurrence for Jacobi polynomials.
    """
    m = x.shape[0]
    apb = a + b

    P = jnp.zeros((m, N + 1), dtype=jnp.float64)
    P = P.at[:, 0].set(1.0)

    if N >= 1:
        P = P.at[:, 1].set(0.5 * (2.0 * (a + 1.0) + (apb + 2.0) * (x - 1.0)))

    aa = a * a
    bb = b * b
    for k in range(2, N + 1):
        k2 = 2 * k
        k2apb = k2 + apb
        q1 = k2 * (k + apb) * (k2apb - 2)
        q2 = (k2apb - 1) * (aa - bb)
        q3 = (k2apb - 2) * (k2apb - 1) * k2apb
        q4 = 2 * (k + a - 1) * (k + b - 1) * k2apb
        P = P.at[:, k].set(
            ((q2 + q3 * x) * P[:, k - 1] - q4 * P[:, k - 2]) / q1
        )

    return P


# ===========================================================================
# Convenience wrappers (matching MATLAB naming)
# ===========================================================================

def chebcoeffs2legcoeffs(c_cheb: jnp.ndarray) -> jnp.ndarray:
    """Convert Chebyshev coefficients to Legendre coefficients.

    Wrapper for cheb2leg.

    Provenance
    ----------
    MATLAB source : chebcoeffs2legcoeffs.m
    Chebfun commit: 7574c77
    """
    return cheb2leg(c_cheb)


def legcoeffs2chebcoeffs(c_leg: jnp.ndarray) -> jnp.ndarray:
    """Convert Legendre coefficients to Chebyshev coefficients.

    Wrapper for leg2cheb.

    Provenance
    ----------
    MATLAB source : legcoeffs2chebcoeffs.m
    Chebfun commit: 7574c77
    """
    return leg2cheb(c_leg)


def chebvals2legcoeffs(
    v_cheb: jnp.ndarray, *, kind: int = 2, normalize: bool = False
) -> jnp.ndarray:
    """Convert Chebyshev values to Legendre coefficients.

    First converts values at Chebyshev points to Chebyshev coefficients,
    then converts to Legendre coefficients.

    Parameters
    ----------
    v_cheb : jnp.ndarray, shape (n,)
        Values at Chebyshev points of the specified kind.
    kind : {1, 2}, default 2
        Which Chebyshev points the values come from.
    normalize : bool, default False
        If True, use orthonormal Legendre polynomials.

    Returns
    -------
    c_leg : jnp.ndarray, shape (n,)
        Legendre coefficients.

    Provenance
    ----------
    MATLAB source : chebvals2legcoeffs.m
    Chebfun commit: 7574c77
    """
    if kind == 2:
        c_cheb = vals2coeffs(v_cheb)
    elif kind == 1:
        c_cheb = _vals2coeffs_kind1(v_cheb)
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")
    return cheb2leg(c_cheb, normalize=normalize)
