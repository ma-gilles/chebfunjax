"""Polynomial coefficient and value transforms.

Chebyshev <-> Legendre, Chebyshev <-> Jacobi, Legendre values/coefficients,
ultra-spherical, Jacobi-to-Jacobi, and Chebyshev values <-> coefficients.

Translated from MATLAB Chebfun (commit 7574c77): cheb2leg.m, leg2cheb.m,
cheb2jac.m, jac2cheb.m, jac2jac.m, ultra2ultra.m, ultracoeffs.m,
chebvals2legcoeffs.m, chebcoeffs2legvals.m, legvals2chebcoeffs.m,
legvals2chebvals.m, legvals2legcoeffs.m, legcoeffs2chebvals.m,
legcoeffs2legvals.m, chebvals2legvals.m, chebvals2chebvals.m,
chebcoeffs2chebvals.m, chebvals2chebcoeffs.m, and related files.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.special import gammaln

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


# ===========================================================================
# Legendre values <-> coefficients/values  (DLT/iDLT wrappers)
# ===========================================================================

def _legendre_dlt(c_leg: jnp.ndarray) -> jnp.ndarray:
    """Discrete Legendre Transform (DLT): Legendre coefficients -> values at legpts.

    Uses the Legendre-Vandermonde matrix via the 3-term recurrence and applies
    it to ``c_leg``.  O(n^2); sufficient for moderate n.

    This is the analogue of ``chebfun.dlt`` (MATLAB) for our pure-JAX implementation.
    """
    n = c_leg.shape[0]
    if n == 0:
        return c_leg
    if n == 1:
        return c_leg

    # Gauss-Legendre nodes (ascending order)
    from chebfunjax.utils.quadrature import legpts
    x, _ = legpts(n)

    L = _legendre_vandermonde(n - 1, x)
    return L @ c_leg


def _legendre_idlt(v_leg: jnp.ndarray) -> jnp.ndarray:
    """Inverse DLT: values at Gauss-Legendre points -> Legendre coefficients.

    Uses Gauss-Legendre quadrature:
        c_k = (2k+1)/2 * sum_j w_j * P_k(x_j) * v_j

    This is the analogue of ``chebfun.idlt`` (MATLAB).
    """
    n = v_leg.shape[0]
    if n == 0:
        return v_leg
    if n == 1:
        return v_leg

    from chebfunjax.utils.quadrature import legpts
    x, w = legpts(n)

    L = _legendre_vandermonde(n - 1, x)  # (n, n)

    # c_k = (2k+1)/2 * sum_j w_j * P_k(x_j) * v_j
    scale = (2 * jnp.arange(n, dtype=jnp.float64) + 1) / 2.0
    c_leg = scale * (L.T @ (w * v_leg))
    return c_leg


def _legendre_ndct(c_cheb: jnp.ndarray) -> jnp.ndarray:
    """Non-uniform DCT: Chebyshev coefficients -> values at Gauss-Legendre points.

    The Gauss-Legendre points are NOT uniformly spaced in angle, so this
    requires explicit Chebyshev evaluation at non-uniform points.

    This is the analogue of ``chebfun.ndct`` (MATLAB).
    """
    n = c_cheb.shape[0]
    if n <= 1:
        return c_cheb

    from chebfunjax.utils.quadrature import legpts
    x, _ = legpts(n)

    # theta = arccos(x) for Legendre points
    theta = jnp.arccos(jnp.clip(x, -1.0, 1.0))
    k = jnp.arange(n, dtype=jnp.float64)
    # T_k(x) = cos(k * theta)
    T = jnp.cos(k[None, :] * theta[:, None])  # (n, n)
    return T @ c_cheb


def legvals2legcoeffs(v_leg: jnp.ndarray) -> jnp.ndarray:
    """Convert values at Gauss-Legendre points to Legendre coefficients.

    LEGCOEFFS = LEGVALS2LEGCOEFFS(V_LEG) converts the vector V_LEG of values
    at Gauss-Legendre points (LEGPTS(N)) to Legendre series coefficients
    C such that
        f(x) = C[0]*P_0(x) + C[1]*P_1(x) + ... + C[N-1]*P_{N-1}(x).

    Parameters
    ----------
    v_leg : jnp.ndarray, shape (n,)
        Values at n Gauss-Legendre points.

    Returns
    -------
    c_leg : jnp.ndarray, shape (n,)
        Legendre series coefficients.

    Provenance
    ----------
    MATLAB source : legvals2legcoeffs.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    return _legendre_idlt(v_leg)


def legcoeffs2legvals(c_leg: jnp.ndarray) -> jnp.ndarray:
    """Convert Legendre coefficients to values at Gauss-Legendre points.

    LEGVALS = LEGCOEFFS2LEGVALS(C_LEG) evaluates the Legendre expansion
        f(x) = C_LEG[0]*P_0(x) + ... + C_LEG[N-1]*P_{N-1}(x)
    at LEGPTS(N), i.e., the N Gauss-Legendre nodes.

    Parameters
    ----------
    c_leg : jnp.ndarray, shape (n,)
        Legendre coefficients.

    Returns
    -------
    v_leg : jnp.ndarray, shape (n,)
        Values at Gauss-Legendre points.

    Provenance
    ----------
    MATLAB source : legcoeffs2legvals.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    return _legendre_dlt(c_leg)


def legvals2chebcoeffs(v_leg: jnp.ndarray) -> jnp.ndarray:
    """Convert Legendre values to Chebyshev coefficients.

    C_CHEB = LEGVALS2CHEBCOEFFS(V_LEG) converts values at Gauss-Legendre
    points to Chebyshev coefficients of the interpolating polynomial.

    Parameters
    ----------
    v_leg : jnp.ndarray, shape (n,)
        Values at Gauss-Legendre points.

    Returns
    -------
    c_cheb : jnp.ndarray, shape (n,)
        Chebyshev series coefficients.

    Provenance
    ----------
    MATLAB source : legvals2chebcoeffs.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    c_leg = _legendre_idlt(v_leg)
    return leg2cheb(c_leg)


def legvals2chebvals(v_leg: jnp.ndarray, *, kind: int = 2) -> jnp.ndarray:
    """Convert values at Gauss-Legendre points to values at Chebyshev points.

    CHEBVALS = LEGVALS2CHEBVALS(LEGVALS) converts values of a polynomial at
    Gauss-Legendre points to values at 2nd-kind Chebyshev points.

    Parameters
    ----------
    v_leg : jnp.ndarray, shape (n,)
        Values at Gauss-Legendre points.
    kind : {1, 2}, default 2
        Target Chebyshev grid kind.

    Returns
    -------
    v_cheb : jnp.ndarray, shape (n,)
        Values at Chebyshev points of the given kind.

    Provenance
    ----------
    MATLAB source : legvals2chebvals.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    c_leg = _legendre_idlt(v_leg)
    return legcoeffs2chebvals(c_leg, kind=kind)


def legcoeffs2chebvals(c_leg: jnp.ndarray, *, kind: int = 2) -> jnp.ndarray:
    """Convert Legendre coefficients to values at Chebyshev points.

    V_CHEB = LEGCOEFFS2CHEBVALS(C_LEG) converts Legendre coefficients to
    values at 2nd-kind Chebyshev points.

    Parameters
    ----------
    c_leg : jnp.ndarray, shape (n,)
        Legendre coefficients.
    kind : {1, 2}, default 2
        Target Chebyshev grid kind.  1 = first-kind Chebyshev points,
        2 = second-kind (Clenshaw-Curtis).

    Returns
    -------
    v_cheb : jnp.ndarray, shape (n,)
        Values at Chebyshev points.

    Provenance
    ----------
    MATLAB source : legcoeffs2chebvals.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    c_cheb = leg2cheb(c_leg)
    if kind == 2:
        return coeffs2vals(c_cheb)
    elif kind == 1:
        n = c_cheb.shape[0]
        if n <= 1:
            return c_cheb
        # Evaluate at 1st-kind Chebyshev points via DCT-III
        from chebfunjax.utils.quadrature import chebpts
        x = chebpts(n, kind=1)
        theta = jnp.arccos(jnp.clip(x, -1.0, 1.0))
        k = jnp.arange(n, dtype=jnp.float64)
        T = jnp.cos(k[None, :] * theta[:, None])
        # Scale: c_cheb[0] is full coefficient, rest are halved in coeffs2vals
        c_scaled = c_cheb.at[0].multiply(0.5).at[-1].multiply(0.5)
        return T @ (2.0 * c_scaled)
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")


def chebvals2legvals(v_cheb: jnp.ndarray, *, kind: int = 2) -> jnp.ndarray:
    """Convert Chebyshev values to values at Gauss-Legendre points.

    LEGVALS = CHEBVALS2LEGVALS(CHEBVALS) converts values at 2nd-kind
    Chebyshev points to values at Gauss-Legendre points.

    Parameters
    ----------
    v_cheb : jnp.ndarray, shape (n,)
        Values at Chebyshev points.
    kind : {1, 2}, default 2
        Kind of source Chebyshev points.

    Returns
    -------
    v_leg : jnp.ndarray, shape (n,)
        Values at Gauss-Legendre points.

    Provenance
    ----------
    MATLAB source : chebvals2legvals.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    if kind == 2:
        c_cheb = vals2coeffs(v_cheb)
    elif kind == 1:
        c_cheb = _vals2coeffs_kind1(v_cheb)
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")
    return _legendre_ndct(c_cheb)


def chebvals2chebvals(
    v_in: jnp.ndarray, kind1: int, kind2: int
) -> jnp.ndarray:
    """Convert between first- and second-kind Chebyshev grids.

    V_OUT = CHEBVALS2CHEBVALS(V_IN, KIND1, KIND2) converts values of a
    polynomial on a Chebyshev grid of kind KIND1 to a grid of kind KIND2.

    Parameters
    ----------
    v_in : jnp.ndarray, shape (n,)
        Input values on a Chebyshev grid.
    kind1 : {1, 2}
        Source grid kind.
    kind2 : {1, 2}
        Target grid kind.

    Returns
    -------
    v_out : jnp.ndarray, shape (n,)
        Values on the target grid.

    Provenance
    ----------
    MATLAB source : chebvals2chebvals.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    if kind1 == kind2:
        return v_in
    elif kind1 == 1 and kind2 == 2:
        c = _vals2coeffs_kind1(v_in)
        return coeffs2vals(c)
    elif kind1 == 2 and kind2 == 1:
        c = vals2coeffs(v_in)
        n = c.shape[0]
        if n <= 1:
            return c
        from chebfunjax.utils.quadrature import chebpts
        x = chebpts(n, kind=1)
        theta = jnp.arccos(jnp.clip(x, -1.0, 1.0))
        k = jnp.arange(n, dtype=jnp.float64)
        T = jnp.cos(k[None, :] * theta[:, None])
        c_half = c.at[1:n - 1].multiply(0.5)
        return T @ c_half * 2.0
    else:
        raise ValueError(f"kind1 and kind2 must be 1 or 2, got {kind1}, {kind2}")


def chebcoeffs2chebvals(c_cheb: jnp.ndarray, *, kind: int = 2) -> jnp.ndarray:
    """Convert Chebyshev coefficients to values at Chebyshev points.

    Wrapper for coeffs2vals (kind=2) or evaluation at 1st-kind points.

    Parameters
    ----------
    c_cheb : jnp.ndarray, shape (n,)
        Chebyshev coefficients.
    kind : {1, 2}, default 2

    Returns
    -------
    v : jnp.ndarray, shape (n,)

    Provenance
    ----------
    MATLAB source : chebcoeffs2chebvals.m
    Chebfun commit: 7574c77
    """
    if kind == 2:
        return coeffs2vals(c_cheb)
    elif kind == 1:
        return chebvals2chebvals(coeffs2vals(c_cheb), kind1=2, kind2=1)
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")


def chebvals2chebcoeffs(v: jnp.ndarray, *, kind: int = 2) -> jnp.ndarray:
    """Convert Chebyshev values to Chebyshev coefficients.

    Wrapper for vals2coeffs (kind=2) or 1st-kind variant.

    Parameters
    ----------
    v : jnp.ndarray, shape (n,)
        Values at Chebyshev points.
    kind : {1, 2}, default 2

    Returns
    -------
    c_cheb : jnp.ndarray, shape (n,)

    Provenance
    ----------
    MATLAB source : chebvals2chebcoeffs.m
    Chebfun commit: 7574c77
    """
    if kind == 2:
        return vals2coeffs(v)
    elif kind == 1:
        return _vals2coeffs_kind1(v)
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")


# ===========================================================================
# Jacobi-to-Jacobi transform  (jac2jac)
# ===========================================================================

# uses-numpy: iterative pivoted Cholesky and FFT-based Toeplitz-Hankel multiply

def jac2jac(
    c_jac: jnp.ndarray,
    alpha: float,
    beta: float,
    gam: float,
    delta: float,
) -> jnp.ndarray:
    """Convert Jacobi (alpha, beta) coefficients to Jacobi (gam, delta) coefficients.

    C_OUT = JAC2JAC(C_IN, A, B, G, D) converts the vector C_IN of Jacobi
    P^{(A,B)} coefficients to P^{(G,D)} coefficients such that
        C_IN[0]*P_0^{(A,B)}(x) + ... + C_IN[N-1]*P_{N-1}^{(A,B)}(x)
      = C_OUT[0]*P_0^{(G,D)}(x) + ... + C_OUT[N-1]*P_{N-1}^{(G,D)}(x).

    Parameters
    ----------
    c_jac : jnp.ndarray, shape (n,)
        Jacobi (alpha, beta) coefficients.
    alpha, beta : float
        Source Jacobi parameters.
    gam, delta : float
        Target Jacobi parameters.

    Returns
    -------
    c_out : jnp.ndarray, shape (n,)
        Jacobi (gam, delta) coefficients.

    Notes
    -----
    Uses the algorithm from [1]: the conversion matrix decomposes as
    D1*(T.*H)*D2 where T is Toeplitz and H is a Hankel matrix approximated
    by pivoted Cholesky.  O(n log n) per rank-1 update.

    References
    ----------
    .. [1] A. Townsend, M. Webb, and S. Olver, "Fast polynomial transforms
       based on Toeplitz and Hankel matrices", Math. Comp., 87, 2018.

    Provenance
    ----------
    MATLAB source : jac2jac.m
    Chebfun commit: 7574c77
    Original authors: Alex Townsend, Marcus Webb, Sheehan Olver.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.

    See Also
    --------
    cheb2jac, jac2cheb, ultra2ultra
    """
    # Work in numpy (iterative algorithm, data-dependent branching)
    v = np.array(c_jac, dtype=np.float64)
    v = _jac2jac_np(v, alpha, beta, gam, delta)
    return jnp.array(v)


def _jac2jac_np(
    v: np.ndarray,
    alpha: float,
    beta: float,
    gam: float,
    delta: float,
) -> np.ndarray:
    """Numpy O(n^2) implementation of jac2jac via Gauss-Jacobi quadrature.

    Evaluates the source Jacobi expansion at (gam,delta) Gauss-Jacobi nodes,
    then projects onto the target basis via the Gauss-Jacobi inner products.
    This is O(n^2) but correct and numerically stable for n up to a few hundred.
    """
    N = len(v)
    if N == 0:
        return v
    if N == 1:
        return v

    # If source == target, identity
    if abs(alpha - gam) < 1e-14 and abs(beta - delta) < 1e-14:
        return v.copy()

    # Gauss-Jacobi nodes and weights for (gam, delta) — used for quadrature
    from chebfunjax.utils.quadrature import jacpts
    x, w = jacpts(N, gam, delta)
    x_np = np.array(x, dtype=np.float64)
    w_np = np.array(w, dtype=np.float64)

    # Evaluate source expansion at x_np: f(x) = sum_k v[k] * P_k^{(alpha,beta)}(x)
    P_src = np.array(_jacobi_vandermonde(N - 1, jnp.array(x_np), alpha, beta))  # (N, N)
    f_vals = P_src @ v

    # Project f onto target Jacobi basis using Gauss-Jacobi quadrature:
    # c_k = (2k + gam + delta + 1)/(2^{gam+delta+1}) * B(k+gam+1,k+delta+1)/(k! * ...)
    # Actually: c_k = h_k^{-1} * sum_j w_j * P_k^{(gam,delta)}(x_j) * f(x_j)
    # where h_k = 2^{gam+delta+1} * Gamma(k+gam+1)*Gamma(k+delta+1) / ((2k+gam+delta+1)*Gamma(k+1)*Gamma(k+gam+delta+1))

    P_tgt = np.array(_jacobi_vandermonde(N - 1, jnp.array(x_np), gam, delta))  # (N, N)

    # Normalization constants h_k (squared norm of P_k^{(gam,delta)})
    k = np.arange(N, dtype=np.float64)
    h_k = np.exp(
        (gam + delta + 1) * np.log(2)
        + gammaln(k + gam + 1) + gammaln(k + delta + 1)
        - np.log(2 * k + gam + delta + 1)
        - gammaln(k + 1) - gammaln(k + gam + delta + 1)
    )

    c_out = (P_tgt.T @ (w_np * f_vals)) / h_k
    return c_out


def _jacobi_integer_conversion(
    v: np.ndarray,
    alpha: float,
    beta: float,
    gam: float,
    delta: float,
) -> tuple[np.ndarray, float, float]:
    """Move (alpha,beta) to (A,B) so that |A-gam|<1 and |B-delta|<1."""
    a, b = float(alpha), float(beta)

    while a <= gam - 1:
        v = _right_jacobi(v, a, b)
        a += 1
    while a >= gam + 1:
        v = _left_jacobi(v, a - 1, b)
        a -= 1
    while b <= delta - 1:
        v = _up_jacobi(v, a, b)
        b += 1
    while b >= delta + 1:
        v = _down_jacobi(v, a, b - 1)
        b -= 1

    return v, a, b


def _up_jacobi(v: np.ndarray, a: float, b: float) -> np.ndarray:
    """Convert Jacobi (a,b) -> (a,b+1) in O(n) operations."""
    N, = v.shape
    nn = np.arange(N, dtype=np.float64)
    apb = a + b
    # Diagonal
    d1 = np.empty(N)
    d1[0] = 1.0
    if N > 1:
        d1[1] = (apb + 2) / (apb + 3)
    if N > 2:
        d1[2:] = (apb + 3 + nn[:-2]) / (apb + 5 + 2 * nn[:-2])
    # Super-diagonal
    d2 = (a + 1 + nn[:N - 1]) / (apb + 3 + 2 * nn[:N - 1])
    out = d1 * v
    out[:N - 1] += d2 * v[1:]
    return out


def _down_jacobi(v: np.ndarray, a: float, b: float) -> np.ndarray:
    """Convert Jacobi (a,b+1) -> (a,b) by inverting _up_jacobi."""
    N, = v.shape
    nn = np.arange(N, dtype=np.float64)
    apb = a + b
    # Build topRow (first row of inverse of up-conversion matrix)
    topRow = np.ones(N)
    if N > 1:
        topRow[1] = (a + 1) / (apb + 2)
    for k in range(2, N):
        topRow[k] = topRow[k - 1] * (a + k) / (apb + k + 1)
    signs = (-1.0) ** nn
    topRow *= signs

    # Apply S^{-1} in O(N) via fliplr cumsum
    tmp = topRow[:, None] * v[None, :]  # broadcast: (N,N) ... but we only need cols
    # vecsum[k] = sum_{j>=k} topRow[j] * v[j]
    # Efficient: vecsum = fliplr(cumsum(fliplr(topRow * v)))
    tv = topRow * v
    vecsum = np.cumsum(tv[::-1])[::-1]

    ratios = np.empty(N)
    ratios[0] = 1.0
    if N > 1:
        ratios[1] = -(apb + 3) / (a + 1)
    if N > 2:
        ratios[2:] = ((apb + 5 + 2 * nn[:-2]) / (apb + 3 + nn[:-2])) * (1.0 / topRow[2:])
        # Correct signs: alternating after the first two
        ratios[2:] = ratios[2:] * signs[2:] / signs[2:]  # no-op if already correct

    out = ratios * vecsum
    return out


def _right_jacobi(v: np.ndarray, a: float, b: float) -> np.ndarray:
    """Convert Jacobi (a,b) -> (a+1,b) using reflection formula."""
    v = v.copy()
    v[1::2] = -v[1::2]
    v = _up_jacobi(v, b, a)
    v[1::2] = -v[1::2]
    return v


def _left_jacobi(v: np.ndarray, a: float, b: float) -> np.ndarray:
    """Convert Jacobi (a+1,b) -> (a,b) using reflection formula."""
    v = v.copy()
    v[1::2] = -v[1::2]
    v = _down_jacobi(v, b, a)
    v[1::2] = -v[1::2]
    return v


def _jacobi_fractional_conversion(
    v: np.ndarray,
    alpha: float,
    beta: float,
    gam: float,
) -> np.ndarray:
    """Convert Jacobi (alpha,beta) -> (gam,beta) with |alpha-gam|<1.

    Uses the Toeplitz-Hankel decomposition from Townsend-Webb-Olver [1].
    The conversion matrix A = D1*(T.*H)*D2, where T is Toeplitz and
    H is Hankel approximated by pivoted Cholesky.

    References
    ----------
    .. [1] A. Townsend, M. Webb, and S. Olver, 2018.
    """
    N = len(v)
    if N <= 1:
        return v

    # Log-gamma helpers (using scipy.special.gammaln)
    def Lambda1(z):
        return np.exp(gammaln(z + alpha + beta + 1) - gammaln(z + gam + beta + 2))

    def Lambda2(z):
        return np.exp(gammaln(z + alpha - gam) - gammaln(z + 1))

    def Lambda3(z):
        return np.exp(gammaln(z + gam + beta + 1) - gammaln(z + beta + 1))

    def Lambda4(z):
        return np.exp(gammaln(z + beta + 1) - gammaln(z + alpha + beta + 1))

    nn = np.arange(N, dtype=np.float64)

    # Diagonal matrix D1
    d1 = (2 * nn + gam + beta + 1) * Lambda3(np.concatenate([[1], nn[:-1]]))
    d1[0] = 1.0

    # Diagonal matrix D2
    inv_gam_factor = 1.0 / float(np.exp(gammaln(alpha - gam + 1)))  # 1/gamma(alpha-gam)
    d2 = inv_gam_factor * Lambda4(np.concatenate([[1], nn[:-1]]))
    d2[0] = 0.0

    # Symbol of Hankel part: vals[k] = Lambda1(k+1) for k=0..2N-1, vals[0]=0
    vals = Lambda1(np.arange(1, 2 * N + 1, dtype=np.float64))
    # vals[0] would be Lambda1(1), but MATLAB sets vals(1)=0 (1-indexed)
    vals_h = np.concatenate([[0.0], vals[:-1]])  # vals_h[k] = vals[k-1] for k>=1, 0 for k=0

    # Pivoted Cholesky on the Hankel matrix H with diagonal d = vals_h[0::2]
    d_diag = vals_h[0::2].copy()  # diagonal of H: H[i,i] = vals_h[2i]
    # Note: Hankel H has H[i,j] = vals_h[i+j], so diag = vals_h[0,2,4,...]
    # But vals_h[0]=0, so d_diag[0]=0; actual diagonal is Lambda1([1,3,5,...])
    # Re-derive: H[i,j] = Lambda1(i+j+1), diagonal H[i,i]=Lambda1(2i+1)
    d_diag = Lambda1(2 * nn + 1)

    tol_chol = 1e-14 * np.log(N + 2)
    chol_cols = []
    pivot_vals = []

    mx_idx = int(np.argmax(d_diag))
    mx = d_diag[mx_idx]

    # Full Hankel column extractor: col j of H = Lambda1([j+1, j+2, ..., j+N])
    def hankel_col(j):
        return Lambda1(j + 1 + nn)

    while mx > tol_chol:
        new_col = hankel_col(mx_idx)
        if chol_cols:
            C_arr = np.column_stack(chol_cols)
            pv_arr = np.array(pivot_vals)
            new_col = new_col - C_arr @ (C_arr[mx_idx, :] * pv_arr)

        pivot_vals.append(1.0 / mx)
        chol_cols.append(new_col.copy())
        d_diag = d_diag - new_col ** 2 / mx
        d_diag = np.maximum(d_diag, 0.0)
        mx_idx = int(np.argmax(d_diag))
        mx = d_diag[mx_idx]

    if not chol_cols:
        return v

    sz = len(chol_cols)
    C_arr = np.column_stack(chol_cols) * np.sqrt(np.array(pivot_vals))[None, :]  # (N, sz)

    # Toeplitz row: T_row[k] = Lambda2(k) for k=0..N-1
    T_row = Lambda2(nn)
    T_row[0] = float(np.exp(gammaln(alpha - gam + 1))) / (alpha - gam)

    # Fast Toeplitz-vector product via FFT
    Z = np.concatenate([[T_row[0]], np.zeros(N - 1)])
    a_fft = np.fft.fft(np.concatenate([Z, T_row[N - 1:0:-1]]))

    # c_jac = D2 * v
    c_work = d2 * v

    # Apply D1*(T.*H)*D2: c_work -> sum over Cholesky columns
    tmp = C_arr * c_work[:, None]  # (N, sz)
    f1 = np.fft.fft(tmp, n=2 * N - 1, axis=0)
    tmp2 = f1 * a_fft[:, None]
    b = np.real(np.fft.ifft(tmp2, axis=0))
    result = d1 * np.sum(b[:N, :] * C_arr, axis=1)

    # Fix first entry
    Matrow1 = (np.exp(gammaln(gam + beta + 2)) / np.exp(gammaln(beta + 1))
               * d2 * T_row * Lambda1(nn))
    result[0] = float(np.dot(Matrow1, v)) + v[0]

    return result


# ===========================================================================
# Ultra-spherical transforms
# ===========================================================================

def ultra2ultra(c: jnp.ndarray, lam_in: float, lam_out: float) -> jnp.ndarray:
    """Convert between ultraspherical (Gegenbauer) expansions.

    C_OUT = ULTRA2ULTRA(C_IN, LAM_IN, LAM_OUT) converts the vector C_IN of
    ultraspherical C^{(lam_in)} coefficients to C^{(lam_out)} coefficients.

    Ultraspherical polynomials C_n^{(lambda)} are a special case of Jacobi
    polynomials:  C_n^{(lam)} ∝ P_n^{(lam-1/2, lam-1/2)}.
    This function uses the jac2jac algorithm internally.

    Parameters
    ----------
    c : jnp.ndarray, shape (n,)
        Ultraspherical C^{(lam_in)} coefficients.
    lam_in : float
        Source ultraspherical parameter (must be >= 0).
    lam_out : float
        Target ultraspherical parameter (must be >= 0).

    Returns
    -------
    c_out : jnp.ndarray, shape (n,)
        Ultraspherical C^{(lam_out)} coefficients.

    Notes
    -----
    The scaling from ultraspherical to Jacobi and back follows DLMF Table 18.3.1.
    For lam=0 the polynomial reduces to Legendre / T_n (Chebyshev).

    Provenance
    ----------
    MATLAB source : ultra2ultra.m
    Chebfun commit: 7574c77
    Original authors: Alex Townsend. Copyright 2017 by The University of
        Oxford and The Chebfun Developers.

    See Also
    --------
    jac2jac, ultracoeffs
    """
    n = c.shape[0] - 1

    def _scl(lam: float) -> np.ndarray:
        """Scaling from Jacobi to ultraspherical (DLMF Table 18.3.1)."""
        if lam == 0.0:
            nn_scl = np.arange(n, dtype=np.float64)
            s = np.concatenate([[1.0], np.cumprod((nn_scl + 0.5) / (nn_scl + 1.0))])
        else:
            nn_scl = np.arange(n + 1, dtype=np.float64)
            s = (np.exp(gammaln(2 * lam) - gammaln(lam + 0.5))
                 * np.exp(gammaln(lam + 0.5 + nn_scl) - gammaln(2 * lam + nn_scl)))
        return s

    c_np = np.array(c, dtype=np.float64)

    # Scale from US to Jacobi
    c_np = c_np / _scl(lam_in)

    # Convert Jacobi (lam_in-0.5, lam_in-0.5) -> (lam_out-0.5, lam_out-0.5)
    c_np = _jac2jac_np(c_np, lam_in - 0.5, lam_in - 0.5, lam_out - 0.5, lam_out - 0.5)

    # Scale from Jacobi to US
    c_np = c_np * _scl(lam_out)

    return jnp.array(c_np)


def ultracoeffs(c_cheb: jnp.ndarray, lam: float) -> jnp.ndarray:
    """Compute ultraspherical series coefficients from Chebyshev coefficients.

    A = ULTRACOEFFS(C_CHEB, LAM) converts the Chebyshev coefficients C_CHEB
    to ultraspherical C^{(lam)} coefficients A such that
        f(x) ≈ A[0]*C_0^{(lam)}(x) + A[1]*C_1^{(lam)}(x) + ...

    Parameters
    ----------
    c_cheb : jnp.ndarray, shape (n,)
        Chebyshev coefficients.
    lam : float
        Ultraspherical parameter.  Must be > 0.

    Returns
    -------
    c_ultra : jnp.ndarray, shape (n,)
        Ultraspherical coefficients.

    Notes
    -----
    Internally converts Chebyshev -> Jacobi (lam-0.5, lam-0.5) -> ultraspherical.

    Provenance
    ----------
    MATLAB source : ultracoeffs.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    ultra2ultra, cheb2jac
    """
    if lam <= 0.0:
        raise ValueError("Ultraspherical polynomials require lam > 0.")
    if lam == 0.5:
        # US(0.5) == Legendre
        return cheb2leg(c_cheb)
    if lam == 1.0:
        # US(1) == Chebyshev 2nd kind (up to normalization; just return Cheb-2nd coeffs)
        return c_cheb

    # Convert Chebyshev -> Jacobi (ab, ab) where ab = lam - 0.5
    ab = lam - 0.5
    c_jac = cheb2jac(c_cheb, ab, ab)

    # Scale from Jacobi to ultraspherical
    n = c_jac.shape[0]
    nn = jnp.arange(n, dtype=jnp.float64)
    from scipy.special import gammaln as _gammaln
    scl = jnp.array(
        np.exp(_gammaln(2 * lam) - _gammaln(lam + 0.5))
        * np.exp(
            np.array([float(_gammaln(lam + 0.5 + k)) for k in range(n)])
            - np.array([float(_gammaln(2 * lam + k)) for k in range(n)])
        )
    )
    return c_jac * scl


# ===========================================================================
# Discrete Sine Transform (DST) and inverse (IDST)
# ===========================================================================


# uses-numpy: DST/IDST use scipy.fft which is not JAX-JIT-safe
def dst(u: jnp.ndarray, kind: int = 1) -> jnp.ndarray:
    r"""Discrete Sine Transform (DST) of type *kind* on a vector or matrix.

    Implements the Wikipedia / MATLAB Chebfun convention.  Types 1–4 are
    supported.  If *u* is a 2-D array the transform is applied column-wise.

    The implementation delegates to ``scipy.fft.dst`` for numerical accuracy
    and is therefore NOT JIT-safe.

    Parameters
    ----------
    u : jnp.ndarray, shape (n,) or (n, m)
        Input values (real).
    kind : {1, 2, 3, 4}
        DST type.  Default 1 (consistent with MATLAB PDE toolbox / Chebfun).

    Returns
    -------
    y : jnp.ndarray, same shape as *u*

    Provenance
    ----------
    MATLAB source : @chebfun/dst.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    idst

    Examples
    --------
    Round-trip test (DST-1 is its own inverse up to a scale):

    >>> import jax.numpy as jnp
    >>> from chebfunjax.utils.transforms import dst, idst
    >>> u = jnp.array([1.0, 2.0, 3.0])
    >>> v = dst(u, 1)
    >>> u2 = idst(v, 1)
    >>> float(jnp.max(jnp.abs(u - u2))) < 1e-12
    True
    """
    import scipy.fft as _sfft
    u_np = np.array(u, dtype=np.float64)
    y_np = _sfft.dst(u_np, type=kind, axis=0, norm="backward")
    return jnp.array(y_np, dtype=jnp.float64)


def idst(c: jnp.ndarray, kind: int = 1) -> jnp.ndarray:
    r"""Inverse Discrete Sine Transform of type *kind*.

    Computes the exact inverse of :func:`dst` under the same
    Wikipedia / MATLAB Chebfun scaling convention.  Delegates to
    ``scipy.fft.idst`` — not JIT-safe.

    Parameters
    ----------
    c : jnp.ndarray, shape (n,) or (n, m)
        DST coefficients produced by :func:`dst`.
    kind : {1, 2, 3, 4}
        DST type.

    Returns
    -------
    u : jnp.ndarray, same shape as *c*

    Provenance
    ----------
    MATLAB source : @chebfun/idst.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    dst

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from chebfunjax.utils.transforms import dst, idst
    >>> u = jnp.array([1.0, 2.0, 3.0, 4.0])
    >>> v = dst(u, 2)
    >>> u2 = idst(v, 2)
    >>> float(jnp.max(jnp.abs(u - u2))) < 1e-12
    True
    """
    import scipy.fft as _sfft
    c_np = np.array(c, dtype=np.float64)
    u_np = _sfft.idst(c_np, type=kind, axis=0, norm="backward")
    return jnp.array(u_np, dtype=jnp.float64)
