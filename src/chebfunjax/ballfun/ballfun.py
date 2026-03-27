# uses-numpy: ball domain construction uses numpy for coefficient assembly (not JIT-safe)
# uses-numpy: adaptive construction uses numpy for FFT on doubled-up grid (not JIT-safe)
"""Ballfun — tensor-product approximation of functions on the unit ball.

Represents a real- or complex-valued function f(r, lambda, theta) on the unit
ball  {(x,y,z) : x^2+y^2+z^2 <= 1}  using a Chebyshev-Fourier-Fourier (CFF)
spectral expansion:

    f(r, lambda, theta) = sum_{j,k,l} c_{j,k,l} T_j(r) exp(i*k*lambda) exp(i*l*theta)

where r in [0, 1], lambda in [-pi, pi] (azimuth), theta in [0, pi] (polar angle).

The coefficient tensor ``coeffs`` has shape (m, n, p):
  - axis 0 (size m, odd): Chebyshev coefficients in r on the doubled-up [-1, 1],
  - axis 1 (size n, even): Fourier coefficients in lambda on [-pi, pi),
  - axis 2 (size p, even >= 4): Fourier coefficients in theta on [-pi, pi).

The "doubled-up" BMC-III structure means:
  - f(r=0, ...) is a constant (no lambda/theta dependence at the origin),
  - f(r, ..., theta=0) and f(r, ..., theta=pi) are each constant in lambda
    (poles are regular).

Construction is adaptive: the function is sampled on increasingly fine
Chebyshev-Fourier-Fourier grids until the spectral coefficients decay below
machine precision.

Translated from MATLAB Chebfun class @ballfun (commit 7574c77).
Original: Copyright 2019 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

Reference:
    N. Boullé and A. Townsend, "Computing with Functions on the Ball",
    SIAM J. Sci. Comput., 2019.
"""

from __future__ import annotations

import warnings
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.misc import standard_chop
from chebfunjax.utils.quadrature import chebpts

# Machine epsilon for float64.
_EPS = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# Grid helpers
# ============================================================================


def _cheb_pts(m: int) -> np.ndarray:
    """Return m Chebyshev-2 points on [-1, 1] in ascending order.

    These are the radial evaluation points for the doubled-up grid:
    x_k = cos(k*pi/(m-1)), k = m-1, ..., 0  (ascending from -1 to 1).

    Provenance
    ----------
    MATLAB source : @ballfun/constructor.m  (evaluate subfunction)
    Chebfun commit: 7574c77
    """
    pts = np.array(chebpts(m, kind=2))  # descending from 1 to -1
    return pts[::-1].copy()  # ascending from -1 to 1


def _trig_pts(n: int) -> np.ndarray:
    """Return n equispaced trigonometric points on [-pi, pi).

    x_k = -pi + 2*pi*k/n, k = 0, ..., n-1.

    Provenance
    ----------
    MATLAB source : @ballfun/constructor.m  (evaluate subfunction)
    Chebfun commit: 7574c77
    """
    return np.linspace(-np.pi, np.pi, n, endpoint=False, dtype=np.float64)


# ============================================================================
# BMC-III structure imposition
# ============================================================================


def _impose_bmc(g: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, bool]:
    """Double the function in r and theta and impose BMC-III structure.

    Parameters
    ----------
    g : np.ndarray, shape (m_half, n_half+1, p_half+1)
        Values on [0,1] x [-pi, 0] x [0, pi].
        Sampled at m_half radial pts, (n_half+1) lambda pts from -pi to 0,
        and (p_half+1) theta pts from 0 to pi.
    h : np.ndarray, shape (m_half, n_half+1, p_half+1)
        Values on [0,1] x [0, pi] x [0, pi].
        Same sizes as g (both include lambda=0 and theta=0,pi endpoints).

    Returns
    -------
    vals : np.ndarray, shape (2*m_half-1, 2*n_half, 2*p_half)
        Doubled-up BMC-III values on the full grid.
    is_real : bool
        True if the original values are real.

    Notes
    -----
    In the MATLAB code (ImposeBMC with two arguments):
      - g has shape (m_half, n_g, p_g) with n_g = n//2+1, p_g = p//2+1
      - h has the same shape (n gets doubled: 2*n_g-2 = n, same for p)
      - Doubled sizes: m = 2*m_half-1, n = 2*(n_g-1), p = 2*(p_g-1)

    Provenance
    ----------
    MATLAB source : @ballfun/constructor.m  (ImposeBMC subfunction)
    Chebfun commit: 7574c77
    """
    m_half, n_g, p_g = g.shape

    # Doubled sizes (matching MATLAB: n = 2*n_g-2, p = 2*p_g-2, m = 2*m_half-1)
    n_half = n_g - 1  # half-size in lambda
    p_half = p_g - 1  # half-size in theta
    m = 2 * m_half - 1
    n = 2 * n_half  # full lambda size
    p = 2 * p_half  # full theta size

    is_real = bool(np.isrealobj(g) and np.isrealobj(h))

    # ---- Impose BMC-III structure ----

    # f(r=0, ...) = constant: mean over all angles at r=0
    # MATLAB: g0 = g(1,:,:); h0 = h(1,2:end,:);
    # h0 excludes the lambda=0 duplicate (index 1 in 0-based)
    g0 = g[0, :, :]  # (n_g, p_g)
    h0 = h[0, 1:, :]  # (n_half, p_g) — excludes lambda=0 (already in g)
    m_zero_r = float(np.real(np.mean(np.concatenate([g0.ravel(), h0.ravel()]))))
    g[0, :, :] = m_zero_r
    h[0, :, :] = m_zero_r

    # f(r, ..., theta=0) = constant in lambda (per each r)
    # MATLAB: m_zeroT = mean([mean(g(:,:,1),2), mean(h(:,2:end,1),2)], 2)
    g_th0 = g[:, :, 0]  # (m_half, n_g)
    h_th0 = h[:, 1:, 0]  # (m_half, n_half)
    # mean over all lambda
    all_th0 = np.concatenate([g_th0, h_th0], axis=1)  # (m_half, n_g+n_half)
    m_zero_t = np.mean(all_th0, axis=1)  # (m_half,)
    g[:, :, 0] = m_zero_t[:, np.newaxis]
    h[:, :, 0] = m_zero_t[:, np.newaxis]

    # f(r, ..., theta=pi) = constant in lambda (per each r)
    g_thpi = g[:, :, -1]  # (m_half, n_g)
    h_thpi = h[:, 1:, -1]  # (m_half, n_half)
    all_thpi = np.concatenate([g_thpi, h_thpi], axis=1)
    m_pi_t = np.mean(all_thpi, axis=1)  # (m_half,)
    g[:, :, -1] = m_pi_t[:, np.newaxis]
    h[:, :, -1] = m_pi_t[:, np.newaxis]

    # ---- Flip g and h in radial direction ----
    # MATLAB: flip1g = flip(g(1+mod(m,2):end,:,:), 1)
    # m = 2*m_half-1 is odd, so mod(m,2)=1, and g(2:end,:,:) in MATLAB 1-based
    # = g[1:, :, :] in 0-based (skip r=0 row), then flip radially = m_half-1 entries
    flip1g = g[1:, :, :][::-1, :, :].copy()  # (m_half-1, n_g, p_g)
    flip1h = h[1:, :, :][::-1, :, :].copy()  # (m_half-1, n_g, p_g)

    # Allocate doubled-up tensor
    vals = np.zeros((m, n, p), dtype=np.complex128 if not is_real else np.float64)

    # MATLAB index conventions (1-based -> 0-based):
    # floor(m/2) = m_half-1  (since m=2*m_half-1 odd)
    # n/2 = n_half
    # floor((p+1)/2) = p_half
    fmh = m_half - 1  # floor(m/2) in 0-based
    fph = p_half  # floor((p+1)/2) in 0-based

    # Fill in the 8 blocks.
    # MATLAB convention: flip(X, 3) = X[:, :, ::-1] in Python (flip along theta/axis 2).
    # flip1g/flip1h already account for the radial flip (no additional axis-0 flip needed).
    #
    # 1. [0,1] x [-pi,0] x [0,pi[
    #    MATLAB: vals(fmh+1:m, 1:n/2+1, fph+1:p) = g(:, :, 1:end-1)
    vals[fmh:m, 0 : n_half + 1, fph:p] = g[:, :, :p_half]
    #
    # 2. [0,1] x [0,pi[ x [0,pi[
    #    MATLAB: vals(fmh+1:m, n/2+1:n, fph+1:p) = h(:, 1:end-1, 1:end-1)
    vals[fmh:m, n_half:n, fph:p] = h[:, :n_half, :p_half]
    #
    # 3. [-1,0[ x [-pi,0] x [0,pi[
    #    MATLAB: vals(1:fmh, 1:n/2+1, fph+1:p) = flip(flip1h(:, :, 2:end), 3)
    vals[0:fmh, 0 : n_half + 1, fph:p] = flip1h[:, :, 1:][:, :, ::-1]
    #
    # 4. [-1,0[ x [0,pi[ x [0,pi[
    #    MATLAB: vals(1:fmh, n/2+1:n, fph+1:p) = flip(flip1g(:, 1:end-1, 2:end), 3)
    vals[0:fmh, n_half:n, fph:p] = flip1g[:, :n_half, 1:][:, :, ::-1]
    #
    # 5. [0,1] x [-pi,0] x [-pi,0]
    #    MATLAB: vals(fmh+1:m, 1:n/2+1, 1:fph) = flip(h(:, :, 2:end), 3)
    vals[fmh:m, 0 : n_half + 1, 0:fph] = h[:, :, 1:][:, :, ::-1]
    #
    # 6. [0,1] x [0,pi[ x [-pi,0]
    #    MATLAB: vals(fmh+1:m, n/2+1:n, 1:fph) = flip(g(:, 1:end-1, 2:end), 3)
    vals[fmh:m, n_half:n, 0:fph] = g[:, :n_half, 1:][:, :, ::-1]
    #
    # 7. [-1,0[ x [0,pi[ x [-pi,0]
    #    MATLAB: vals(1:fmh, n/2+1:n, 1:fph) = flip1h(:, 1:end-1, 1:end-1)
    vals[0:fmh, n_half:n, 0:fph] = flip1h[:, :n_half, :p_half]
    #
    # 8. [-1,0[ x [-pi,0] x [-pi,0]
    #    MATLAB: vals(1:fmh, 1:n/2+1, 1:fph) = flip1g(:, :, 1:end-1)
    vals[0:fmh, 0 : n_half + 1, 0:fph] = flip1g[:, :, :p_half]

    # Check if real
    if np.linalg.norm(np.imag(vals.ravel())) < 1e5 * np.finfo(float).eps:
        vals = np.real(vals)
        is_real = True

    return vals, is_real


# ============================================================================
# Spectral transforms: vals <-> coeffs
# ============================================================================


def _even_odd_fix(n: int) -> np.ndarray:
    """Phase correction factors for the Fourier transform on [-pi, pi).

    Returns a 1D array of factors (-1)^k for k = -n//2, ..., n//2-1
    (even n) or k = -(n-1)//2, ..., (n-1)//2 (odd n).

    Provenance
    ----------
    MATLAB source : @ballfun/vals2coeffs.m  (even_odd_fix subfunction)
    Chebfun commit: 7574c77
    """
    if n % 2 == 1:
        ks = np.arange(-(n - 1) // 2, (n - 1) // 2 + 1)
    else:
        ks = np.arange(-n // 2, n // 2)
    return (-1.0) ** ks


def _vals2coeffs_3d(X: np.ndarray) -> np.ndarray:
    """Convert BMC-III values to Chebyshev-Fourier-Fourier coefficients.

    The input X is a real or complex array of shape (m, n, p) sampled on a
    doubled-up radial-azimuthal-polar grid. The output C has the same shape
    and contains the Chebyshev-Fourier-Fourier expansion coefficients.

    - Axis 0 (size m): Chebyshev transform via inverse-FFT trick.
    - Axes 1 and 2 (size n and p): Fourier transforms with fftshift phase fix.

    Provenance
    ----------
    MATLAB source : @ballfun/vals2coeffs.m
    Chebfun commit: 7574c77
    """
    m, n, p = X.shape

    # Radial: Chebyshev (DCT-I via FFT)
    if m > 1:
        # Mirror to get the DCT-I: [X[m-1], X[m-2], ..., X[1], X[0], X[1], ..., X[m-2]]
        # But MATLAB does: ifft([X[m:-1:2]; X], 2*(m-1), 1)
        # Which is: stack X[m-2:0:-1] on top of X, then take ifft of size 2*(m-1)
        np.concatenate([X[m - 1 : 0 : -1, :, :], X], axis=0)  # size 2*(m-1) x n x p
        # Wait: MATLAB does X(m:-1:2,:,:) which is rows m, m-1,...,2 in 1-based = rows m-1,m-2,...,1 in 0-based
        # Then vertcat with X gives [X[m-1:-1:1]; X[0:m]] which has size (m-1) + m = 2m-1... no
        # Re-read: ifft(vertcat(X(m:-1:2,:,:), X), 2*(m-1), 1)
        # X(m:-1:2,:,:) has size m-1 (indices m,m-1,...,2 in MATLAB = m-1 entries in 0-based: m-1,m-2,...,1)
        # vertcat size = (m-1) + m = 2m-1, but ifft with N=2*(m-1)...
        # The MATLAB ifft(..., 2*(m-1), 1) operates on a matrix of size 2*(m-1) x n x p
        # but vertcat gives (2m-1) rows. Actually in MATLAB:
        # X(m:-1:2,:,:) has m-1 rows (rows 2..m in 1-based = 1..m-1 in 0-based, reversed)
        # vertcat with X (m rows) gives 2m-1 rows
        # ifft(..., 2*(m-1), 1) truncates to 2*(m-1) rows before taking FFT
        # So the effective array passed to ifft is: [X[m-2:0:-1]; X[0:m-1]] which has 2*(m-1) rows

        X[m - 2 :: -1, :, :]  # rows m-2, m-3, ..., 0 in 0-based (= MATLAB rows m-1:-1:1)
        # MATLAB: X(m:-1:2,:,:) = rows m, m-1,...,2 in 1-based = 0-indexed: m-1, m-2,...,1
        top2 = X[m - 1 : 0 : -1, :, :]  # rows m-1,m-2,...,1 (size m-1)
        combined = np.concatenate([top2, X], axis=0)  # size 2m-1

        # MATLAB's ifft(..., 2*(m-1)) uses only first 2*(m-1) rows of the concatenated array
        N_dct = 2 * (m - 1)
        arr = combined[:N_dct, :, :]  # rows 0..2m-3

        X_r = np.fft.ifft(arr, axis=0)  # complex, size N_dct x n x p
        # Take first m rows and scale
        X_r = 2.0 * X_r[:m, :, :]
        X_r[0, :, :] /= 2.0
        X_r[m - 1, :, :] /= 2.0
        X = X_r

    # Azimuthal and polar: Fourier transforms with phase correction
    # MATLAB: fftshift(fftshift(fft(fft(X,[],2),[],3),2),3)
    X = np.fft.fft(X, axis=1)
    X = np.fft.fft(X, axis=2)
    X = np.fft.fftshift(X, axes=1)
    X = np.fft.fftshift(X, axes=2)

    # Scale factors
    scl_n = _even_odd_fix(n)  # shape (n,)
    scl_p = (1.0 / n / p) * _even_odd_fix(p)  # shape (p,)
    Enp = (scl_n[:, np.newaxis] * scl_p[np.newaxis, :]).reshape(1, n, p)
    X = X * Enp

    return X


def _coeffs2vals_3d(X: np.ndarray) -> np.ndarray:
    """Convert Chebyshev-Fourier-Fourier coefficients to BMC-III values.

    Inverse of _vals2coeffs_3d.

    Provenance
    ----------
    MATLAB source : @ballfun/coeffs2vals.m
    Chebfun commit: 7574c77
    """
    m, n, p = X.shape

    # Scale factors (inverse of vals2coeffs scaling)
    scl_n = _even_odd_fix(n)  # shape (n,)
    scl_p = (n * p) * _even_odd_fix(p)  # shape (p,)
    Enp = (scl_n[:, np.newaxis] * scl_p[np.newaxis, :]).reshape(1, n, p)
    X = X * Enp

    # Azimuthal and polar: inverse Fourier (with inverse fftshift)
    X = np.fft.ifft(
        np.fft.ifft(np.fft.ifftshift(np.fft.ifftshift(X, axes=2), axes=1), axis=1), axis=2
    )

    # Radial: inverse Chebyshev (DCT-I via FFT)
    if m > 1:
        # Halve interior coefficients
        X[1 : m - 1, :, :] /= 2.0
        # Mirror: [X; X[m-2:-1:1]]
        mirrored = np.concatenate([X, X[m - 2 : 0 : -1, :, :]], axis=0)  # size 2*(m-1)
        X_r = np.fft.fft(mirrored, axis=0)  # size 2*(m-1)
        X = X_r[m - 1 :: -1, :, :]  # first m entries, reversed (MATLAB: X(m:-1:1))

    return X


# ============================================================================
# Happiness check for one-dimensional slice
# ============================================================================


def _is_happy_1d(values: np.ndarray, tol: float) -> tuple[bool, int]:
    """Check if a 1D array of values is resolved; return (happy, cutoff).

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Values at Chebyshev-2 or equispaced points (treated as Chebyshev here).
    tol : float
        Absolute tolerance.

    Returns
    -------
    happy : bool
    cutoff : int
        Index where the coefficients are chopped.

    Provenance
    ----------
    MATLAB source : @ballfun/constructor.m  (ballfunHappiness subfunction)
    Chebfun commit: 7574c77
    """
    from chebfunjax.utils.transforms import vals2coeffs as cheb_vals2coeffs

    v = jnp.asarray(values, dtype=jnp.float64)
    c = cheb_vals2coeffs(v)
    vscale = float(jnp.max(jnp.abs(v)))
    if vscale == 0.0:
        return True, 1
    rel_tol = max(tol / vscale, _EPS)
    cutoff = int(standard_chop(c, rel_tol))
    return cutoff < c.shape[0], cutoff


def _ballfun_happiness(
    vals: np.ndarray,
) -> tuple[int, int, int, list[int], list[bool]]:
    """Check whether the current grid resolves the function.

    Parameters
    ----------
    vals : np.ndarray, shape (m, n, p)
        Function values on the doubled-up BMC-III grid.

    Returns
    -------
    new_m, new_n, new_p : int
        Suggested new grid sizes (doubled if not happy).
    cutoffs : list of int
        Suggested cutoff indices [c_r, c_lam, c_th].
    resolved : list of bool
        Whether each direction is resolved.

    Provenance
    ----------
    MATLAB source : @ballfun/constructor.m  (ballfunHappiness subfunction)
    Chebfun commit: 7574c77
    """
    m, n, p = vals.shape

    # Transform to coefficient space
    cfs = _vals2coeffs_3d(vals)

    # Check radial (Chebyshev) direction: sum over lambda and theta
    np.real(np.sum(np.abs(cfs), axis=(1, 2)))
    # Tol relative to max abs coeff
    cfs_abs = np.abs(cfs)
    vscale = float(np.max(cfs_abs)) if cfs_abs.size > 0 else 1.0
    tol = max(vscale * _EPS, 100.0 * _EPS)

    # For each direction compute standard_chop on the "max projection"
    def _chop_dir(c_arr: np.ndarray) -> tuple[bool, int]:
        """c_arr is real 1D slice of abs-summed coefficients."""
        c_jax = jnp.asarray(c_arr, dtype=jnp.float64)
        vsc = float(jnp.max(jnp.abs(c_jax)))
        if vsc == 0.0:
            return True, 1
        rt = max(tol / vsc, _EPS)
        cut = int(standard_chop(c_jax, rt))
        return cut < c_arr.shape[0], cut

    # Radial: first Chebyshev direction (axis 0)
    r_proj = np.max(np.abs(np.real(cfs)), axis=(1, 2))
    r_happy, c_r = _chop_dir(r_proj)
    c_r = max(c_r, 1)

    # Azimuthal Fourier (axis 1): take row sums of abs
    n_mid = n // 2
    lam_proj = np.max(np.abs(cfs), axis=(0, 2))  # shape (n,)
    # Re-order so DC (index n_mid) is first
    lam_shifted = np.roll(lam_proj, -n_mid)  # DC at index 0
    lam_happy_r, c_lam_shifted = _chop_dir(np.abs(lam_shifted))
    c_lam = c_lam_shifted
    c_lam = max(c_lam, 2)
    if c_lam % 2 != 0:
        c_lam += 1

    # Polar Fourier (axis 2)
    th_proj = np.max(np.abs(cfs), axis=(0, 1))  # shape (p,)
    p_mid = p // 2
    th_shifted = np.roll(th_proj, -p_mid)  # DC at index 0
    th_happy_r, c_th_shifted = _chop_dir(np.abs(th_shifted))
    c_th = c_th_shifted
    c_th = max(c_th, 4)
    if c_th % 2 != 0:
        c_th += 1

    # Suggest new grid sizes (double if unhappy)
    new_m = c_r + 1 - c_r % 2 if r_happy else min(2 * m, m + 16)
    new_n = c_lam if lam_happy_r else min(2 * n, n + 16)
    new_p = c_th if th_happy_r else min(2 * p, p + 16)

    # Ensure parity constraints: m odd, n even, p even >= 4
    new_m = new_m + 1 - new_m % 2  # odd
    new_n = new_n + new_n % 2  # even
    new_p = max(4, new_p + new_p % 2)  # even >= 4

    resolved = [r_happy, lam_happy_r, th_happy_r]
    cutoffs = [c_r, c_lam, c_th]

    return new_m, new_n, new_p, cutoffs, resolved


# ============================================================================
# Evaluate op on Cheb-Fourier-Fourier grid
# ============================================================================


def _evaluate_on_grid(
    op: Callable,
    m: int,
    n: int,
    p: int,
    is_spherical: bool = False,
) -> tuple[np.ndarray, bool]:
    """Sample ``op`` on a Cheb x Fourier x Fourier grid and double up.

    Parameters
    ----------
    op : callable
        Either op(r, lam, th) (spherical) or op(x, y, z) (Cartesian).
    m : int
        Number of Chebyshev points in the doubled radial direction (odd).
        The physical half-grid has m_half = (m+1)//2 = (m+1)/2 points in [0,1].
    n : int
        Number of Fourier points in lambda (even). Physical grid has n//2+1
        points in each half of [-pi, pi].
    p : int
        Number of Fourier points in theta (even, >= 4). Physical grid has
        p//2+1 points in [0, pi].
    is_spherical : bool
        If True, op takes (r, lam, th). If False, op takes Cartesian (x,y,z).

    Returns
    -------
    vals : np.ndarray, shape (m, n, p)
        BMC-III doubled-up values.
    is_real : bool
        True if all sampled values are real.

    Notes
    -----
    Matches MATLAB's evaluate subfunction which uses:
      r = chebpts(m)[floor(m/2)+1:m]   (physical half, m_half = ceil(m/2) points)
      lam = [pi*trigpts(n); pi]          (n+1 points in [-pi, pi])
      th  = [pi*trigpts(p); pi]          (p+1 points in [-pi, pi])
    and samples g on [0,1] x [-pi, 0] x [0, pi] and h on [0,1] x [0, pi] x [0, pi].

    Provenance
    ----------
    MATLAB source : @ballfun/constructor.m  (evaluate subfunction)
    Chebfun commit: 7574c77
    """
    # MATLAB: r = chebpts(m) gives m points in [-1, 1] in ASCENDING order.
    # r(floor(m/2)+1:m) in 1-based = r[m//2:] in 0-based = last m_half points = [0, ..., 1].
    # With m odd, floor(m/2) = (m-1)/2, so m_half = (m+1)/2 points in [0, 1].
    r_asc = np.array(chebpts(m, kind=2))  # ascending from -1 to 1
    m_half = (m + 1) // 2  # floor(m/2)+1 = number of points in [0, 1]
    r = r_asc[m - m_half :].copy()  # ascending from ~0 to 1, size m_half

    # MATLAB: lam = [pi*trigpts(n); pi] gives n+1 points in [-pi, pi]
    # trigpts(n) gives n equispaced points in [-1,1): -1+2k/n for k=0..n-1
    # So lam = pi*trigpts(n) in [-pi, pi) plus pi at the end = n+1 pts
    lam_trig = np.linspace(-np.pi, np.pi, n, endpoint=False)  # n pts in [-pi, pi)
    lam = np.append(lam_trig, np.pi)  # n+1 pts: [-pi, ..., pi)], pi]

    # MATLAB: th = [pi*trigpts(p); pi] = p+1 pts in [-pi, pi]
    th_trig = np.linspace(-np.pi, np.pi, p, endpoint=False)
    th = np.append(th_trig, np.pi)  # p+1 pts

    # g: evaluated on [0,1] x [-pi, 0] x [0, pi]
    # MATLAB: [rrg, llg, ttg] = ndgrid(r(floor(m/2)+1:m), lam(1:n/2+1), th(p/2+1:p+1))
    # lam(1:n/2+1) in 1-based = lam[0:n//2+1] in 0-based = [-pi, ..., 0] (n/2+1 pts)
    # th(p/2+1:p+1) in 1-based = th[p//2:p+1] in 0-based = [0, ..., pi] (p/2+1 pts)
    lam_g = lam[: n // 2 + 1]  # [-pi, ..., 0], size n//2+1
    th_g = th[p // 2 :]  # [0, ..., pi], size p//2+1

    # h: evaluated on [0,1] x [0, pi] x [0, pi]
    # MATLAB: [rrh, llh, tth] = ndgrid(r(floor(m/2)+1:m), lam(n/2+1:end), th(p/2+1:p+1))
    # lam(n/2+1:end) in 1-based = lam[n//2:] in 0-based = [0, ..., pi] (n/2+1 pts)
    lam_h = lam[n // 2 :]  # [0, ..., pi], size n//2+1
    th_h = th_g  # same theta grid

    # Build ndgrid: g shape (m_half, n//2+1, p//2+1), h same shape
    rrg, llg, ttg = np.meshgrid(r, lam_g, th_g, indexing="ij")
    rrh, llh, tth = np.meshgrid(r, lam_h, th_h, indexing="ij")

    if is_spherical:
        g = np.array(
            op(
                jnp.asarray(rrg, dtype=jnp.float64),
                jnp.asarray(llg, dtype=jnp.float64),
                jnp.asarray(ttg, dtype=jnp.float64),
            ),
            dtype=np.complex128,
        )
        h = np.array(
            op(
                jnp.asarray(rrh, dtype=jnp.float64),
                jnp.asarray(llh, dtype=jnp.float64),
                jnp.asarray(tth, dtype=jnp.float64),
            ),
            dtype=np.complex128,
        )
    else:
        # Convert to Cartesian: x = r*sin(th)*cos(lam), y = r*sin(th)*sin(lam), z = r*cos(th)
        xg = rrg * np.sin(ttg) * np.cos(llg)
        yg = rrg * np.sin(ttg) * np.sin(llg)
        zg = rrg * np.cos(ttg)
        xh = rrh * np.sin(tth) * np.cos(llh)
        yh = rrh * np.sin(tth) * np.sin(llh)
        zh = rrh * np.cos(tth)
        g = np.array(
            op(
                jnp.asarray(xg, dtype=jnp.float64),
                jnp.asarray(yg, dtype=jnp.float64),
                jnp.asarray(zg, dtype=jnp.float64),
            ),
            dtype=np.complex128,
        )
        h = np.array(
            op(
                jnp.asarray(xh, dtype=jnp.float64),
                jnp.asarray(yh, dtype=jnp.float64),
                jnp.asarray(zh, dtype=jnp.float64),
            ),
            dtype=np.complex128,
        )

    vals, is_real = _impose_bmc(g, h)
    return vals, is_real


# ============================================================================
# Main class
# ============================================================================


class Ballfun(eqx.Module):
    """Chebyshev-Fourier-Fourier approximation of a function on the unit ball.

    Represents a smooth function on the unit ball
    {(x,y,z) : x^2+y^2+z^2 <= 1} using the BMC-III tensor-product structure.

    The representation uses spherical coordinates (r, lambda, theta):
      - r in [0, 1]:       radial variable
      - lambda in [-pi, pi]: azimuthal (longitude) angle
      - theta in [0, pi]:   polar (colatitude) angle

    Internally the function is doubled up:
      - r is extended to [-1, 1] (odd extension)
      - theta is extended to [-pi, pi] (even extension)
    so that spectral convergence is maintained.

    Attributes
    ----------
    coeffs : jax.Array, shape (m, n, p) complex
        Chebyshev-Fourier-Fourier coefficients. m is odd (Chebyshev in r),
        n is even (Fourier in lambda), p is even >= 4 (Fourier in theta).
        Stored as complex128 to handle the Fourier structure; for real-valued
        functions the Hermitian symmetry holds approximately.
    is_real : bool
        True if the represented function is real-valued.
    domain : tuple of 6 floats
        Always (0, 1, -pi, pi, 0, pi). Static field.

    Notes
    -----
    Construction is NOT JIT-safe (adaptive Python loop).
    Evaluation IS JIT-safe via ``fevalm``.

    Provenance
    ----------
    MATLAB source : @ballfun/ballfun.m, @ballfun/constructor.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2019 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: N. Boullé and A. Townsend, "Computing with Functions on
        the Ball", SIAM J. Sci. Comput., 2019.

    See Also
    --------
    Spherefun, SeparableApprox
    """

    coeffs: jax.Array  # shape (m, n, p) complex128
    is_real: bool = eqx.field(static=True)
    domain: tuple = eqx.field(static=True)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_function(
        cls,
        op: Callable,
        *,
        spherical: bool = False,
        fixed_size: tuple[int, int, int] | None = None,
        tol: float = _EPS,
        max_sample: int = 2**16,
    ) -> "Ballfun":
        """Construct a Ballfun from a callable.

        Parameters
        ----------
        op : callable
            The function to approximate. By default, should accept Cartesian
            coordinates ``(x, y, z)`` as JAX arrays. If ``spherical=True``,
            should accept spherical coordinates ``(r, lambda, theta)`` as JAX
            arrays, where r in [0,1], lambda in [-pi, pi], theta in [0, pi].
            The callable must be vectorized (handle array inputs).
        spherical : bool, optional
            If True, ``op`` is in spherical coordinates (r, lam, th).
            Default False (Cartesian).
        fixed_size : tuple of 3 ints or None, optional
            If given as (m, n, p), use a fixed grid of that size without
            adaptive refinement.
        tol : float, optional
            Target tolerance. Default is machine epsilon (~2.2e-16).
        max_sample : int, optional
            Maximum total grid size m*n*p. Default 2^16.

        Returns
        -------
        Ballfun
            Approximation of ``op`` on the unit ball.

        Raises
        ------
        ValueError
            If ``op`` returns Inf or NaN on the evaluation grid.
        RuntimeWarning
            If the adaptive loop did not converge.

        Notes
        -----
        Construction is NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @ballfun/constructor.m
        Chebfun commit: 7574c77
        """
        # --- Fixed size case ---
        if fixed_size is not None:
            m, n, p = int(fixed_size[0]), int(fixed_size[1]), int(fixed_size[2])
            # Enforce parity constraints
            m = m + 1 - m % 2  # odd
            n = n + n % 2  # even
            p = max(4, p + p % 2)  # even >= 4
            vals, is_real = _evaluate_on_grid(op, m, n, p, is_spherical=spherical)
            cfs = _vals2coeffs_3d(vals)
            cfs_jax = jnp.asarray(cfs, dtype=jnp.complex128)
            return cls(
                coeffs=cfs_jax,
                is_real=bool(is_real),
                domain=(0.0, 1.0, -float(np.pi), float(np.pi), 0.0, float(np.pi)),
            )

        # --- Adaptive construction ---
        # Initial grid sizes (matching MATLAB defaults, minSamples=9)
        m = 9  # odd
        n = 4  # even
        p = 4  # even >= 4

        is_happy = False
        failure = False

        while not is_happy and not failure:
            vals, is_real = _evaluate_on_grid(op, m, n, p, is_spherical=spherical)

            vscale = float(np.max(np.abs(vals)))
            if not np.isfinite(vscale):
                raise ValueError(
                    "Ballfun.from_function: operator returned Inf or NaN "
                    f"on the grid of size ({m}, {n}, {p})."
                )

            if m * n * p > max_sample:
                warnings.warn(
                    f"Ballfun.from_function: grid size ({m}, {n}, {p}) = "
                    f"{m * n * p} exceeded max_sample={max_sample}. "
                    "Returning best approximation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                failure = True
                break

            new_m, new_n, new_p, cutoffs, resolved = _ballfun_happiness(vals)
            is_happy = all(resolved)

            if not is_happy:
                m = new_m
                n = new_n
                p = new_p

        # Final evaluation at correct grid
        c_r, c_lam, c_th = cutoffs if not failure else (m, n, p)

        # Enforce parity constraints on cutoffs
        c_r = c_r + 1 - c_r % 2  # odd
        c_lam = c_lam + c_lam % 2  # even
        c_th = max(4, c_th + c_th % 2)  # even >= 4

        vals_final, is_real = _evaluate_on_grid(op, c_r, c_lam, c_th, is_spherical=spherical)
        cfs = _vals2coeffs_3d(vals_final)

        # Chop to resolved sizes
        if resolved[0]:
            cfs = cfs[:c_r, :, :]
        mf = cfs.shape[1]
        mid_n = mf // 2
        if resolved[1]:
            half_lam = c_lam // 2
            cfs = cfs[:, mid_n - half_lam : mid_n + c_lam - half_lam, :]
        pf = cfs.shape[2]
        mid_p = pf // 2
        if resolved[2]:
            half_th = c_th // 2
            cfs = cfs[:, :, mid_p - half_th : mid_p + c_th - half_th]

        cfs_jax = jnp.asarray(cfs, dtype=jnp.complex128)
        return cls(
            coeffs=cfs_jax,
            is_real=bool(is_real),
            domain=(0.0, 1.0, -float(np.pi), float(np.pi), 0.0, float(np.pi)),
        )

    @classmethod
    def from_coeffs(cls, coeffs: jax.Array, *, is_real: bool = True) -> "Ballfun":
        """Construct a Ballfun directly from CFF coefficients.

        Parameters
        ----------
        coeffs : jax.Array, shape (m, n, p)
            Chebyshev-Fourier-Fourier coefficients. m should be odd, n and p
            should be even.
        is_real : bool, optional
            Whether the function is real-valued. Default True.

        Returns
        -------
        Ballfun

        Provenance
        ----------
        MATLAB source : @ballfun/ballfun.m  (coeffs flag)
        Chebfun commit: 7574c77
        """
        coeffs = jnp.asarray(coeffs, dtype=jnp.complex128)
        return cls(
            coeffs=coeffs,
            is_real=bool(is_real),
            domain=(0.0, 1.0, -float(np.pi), float(np.pi), 0.0, float(np.pi)),
        )

    # ------------------------------------------------------------------
    # Shape / size
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the coefficient tensor (m, n, p)."""
        return tuple(self.coeffs.shape)

    def __len__(self) -> int:
        """Total number of coefficients."""
        m, n, p = self.shape
        return m * n * p

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def fevalm(
        self,
        r: jax.Array,
        lam: jax.Array,
        th: jax.Array,
    ) -> jax.Array:
        """Evaluate f at tensor-product spherical coordinate grids.

        Given 1D arrays r (length Nr), lam (length Nlam), th (length Nth),
        returns a 3D array of shape (Nr, Nlam, Nth) via Clenshaw + Horner.

        Parameters
        ----------
        r : jax.Array, shape (Nr,)
            Radial values in [0, 1].
        lam : jax.Array, shape (Nlam,)
            Azimuthal angles in [-pi, pi].
        th : jax.Array, shape (Nth,)
            Polar angles in [0, pi].

        Returns
        -------
        vals : jax.Array, shape (Nr, Nlam, Nth)
            Evaluated values.

        Notes
        -----
        JIT-safe, vmap-safe.

        Provenance
        ----------
        MATLAB source : @ballfun/fevalm.m
        Chebfun commit: 7574c77
        """
        r = jnp.asarray(r, dtype=jnp.float64)
        lam = jnp.asarray(lam, dtype=jnp.float64)
        th = jnp.asarray(th, dtype=jnp.float64)

        F = self.coeffs.astype(jnp.complex128)  # shape (m, n, p) complex
        m, n, p = F.shape

        Nr = r.shape[0]
        Nlam = lam.shape[0]
        Nth = th.shape[0]

        # --- Step 1: evaluate in r via Clenshaw (complex-safe) ---
        # F_2d has shape (m, n*p); evaluate at each r_i to get (Nr, n*p)
        F_2d = F.reshape(m, n * p)  # (m, n*p) complex

        def _clenshaw_cmplx(c: jax.Array, x: jax.Array) -> jax.Array:
            """Clenshaw for complex coefficients, scalar x. Returns complex scalar."""
            # c shape (m,) complex, x scalar float
            n_local = c.shape[0]
            if n_local == 0:
                return jnp.zeros((), dtype=jnp.complex128)
            if n_local == 1:
                return c[0].astype(jnp.complex128)
            x2 = 2.0 * x

            # Run recurrence: bk1, bk2 start at 0
            # b_{k} = c[k] + 2x*b_{k+1} - b_{k+2}
            def body(carry, k):
                bk1, bk2 = carry
                bk = c[k] + x2 * bk1 - bk2
                return (bk, bk1), None

            init = (jnp.zeros((), dtype=jnp.complex128), jnp.zeros((), dtype=jnp.complex128))
            (bk1, bk2), _ = jax.lax.scan(body, init, jnp.arange(n_local - 1, 0, -1))
            return c[0] + x * bk1 - bk2

        # Evaluate each column of F_2d at all r points
        # G[i, j] = sum_k F_2d[k, j] * T_k(r[i])
        # Use matrix-vector product via explicit Chebyshev evaluation
        # More efficient: build Chebyshev matrix T of shape (Nr, m)
        # T[i, k] = T_k(r[i])
        def _cheb_matrix(r_pts: jax.Array, m_local: int) -> jax.Array:
            """Build Chebyshev matrix T[i, k] = T_k(r[i]), shape (Nr, m_local)."""
            if m_local == 0:
                return jnp.zeros((r_pts.shape[0], 0), dtype=jnp.float64)
            if m_local == 1:
                return jnp.ones((r_pts.shape[0], 1), dtype=jnp.float64)
            # T_0=1, T_1=r, T_{k+1} = 2r*T_k - T_{k-1}
            T_prev = jnp.ones_like(r_pts)  # T_0
            T_curr = r_pts  # T_1
            cols = [T_prev, T_curr]
            for k in range(2, m_local):
                T_next = 2.0 * r_pts * T_curr - T_prev
                cols.append(T_next)
                T_prev = T_curr
                T_curr = T_next
            return jnp.stack(cols, axis=1)  # (Nr, m_local)

        T_mat = _cheb_matrix(r, m)  # (Nr, m) float64
        # G = T_mat @ F_2d: (Nr, m) x (m, n*p) -> (Nr, n*p) complex
        G = jnp.dot(T_mat.astype(jnp.complex128), F_2d)  # (Nr, n*p)
        G = G.reshape(Nr, n, p)  # (Nr, n, p)

        # --- Step 2: evaluate in lambda via DFT Horner ---
        # Fourier coefficients stored in fftshift order: k = -n//2, ..., n//2-1
        # f(lam) = sum_{k=-n//2}^{n//2-1} C[k+n//2] * exp(i*k*lam)
        #        = exp(-i*(n//2)*lam) * sum_{j=0}^{n-1} C[j] * exp(i*j*lam)
        # Use matrix multiplication: E[i,j] = exp(i * j * lam[i]), then
        # H = E @ C_shifted, where C_shifted corrects the phase.

        # Build Fourier matrix for lambda
        # shape (Nlam, n): E[i, j] = exp(i * (j - n//2) * lam[i])
        n_mid = n // 2
        ks_lam = jnp.arange(n, dtype=jnp.float64) - n_mid  # wavenumbers
        # E_lam[i, k] = exp(i * k * lam[i])
        E_lam = jnp.exp(1j * jnp.outer(lam, ks_lam))  # (Nlam, n)

        # G has shape (Nr, n, p); for each (r_idx, th_idx) evaluate at all lam
        # Reshape G to (Nr*p, n), multiply by E_lam^T to get (Nr*p, Nlam), reshape to (Nr, p, Nlam)
        G_rp = G.transpose(0, 2, 1).reshape(Nr * p, n)  # (Nr*p, n)
        H_rp = jnp.dot(G_rp, E_lam.T)  # (Nr*p, Nlam)
        H = H_rp.reshape(Nr, p, Nlam)  # (Nr, p, Nlam)

        # --- Step 3: evaluate in theta via DFT Horner ---
        p_mid = p // 2
        ks_th = jnp.arange(p, dtype=jnp.float64) - p_mid
        E_th = jnp.exp(1j * jnp.outer(th, ks_th))  # (Nth, p)

        # H has shape (Nr, p, Nlam); for each (r_idx, lam_idx) evaluate at all th
        H_rl = H.transpose(0, 2, 1).reshape(Nr * Nlam, p)  # (Nr*Nlam, p)
        vals_rl = jnp.dot(H_rl, E_th.T)  # (Nr*Nlam, Nth)
        vals = vals_rl.reshape(Nr, Nlam, Nth)  # (Nr, Nlam, Nth)

        if self.is_real:
            vals = jnp.real(vals)
        return vals

    def __call__(
        self,
        r: jax.Array,
        lam: jax.Array,
        th: jax.Array,
    ) -> jax.Array:
        """Evaluate f at spherical coordinates (r, lam, th).

        Accepts scalar or array inputs of the same shape, or 1D arrays
        (in which case a tensor-product grid is used via ``fevalm``).

        Parameters
        ----------
        r : jax.Array
            Radial coordinate(s) in [0, 1].
        lam : jax.Array
            Azimuthal angle(s) in [-pi, pi].
        th : jax.Array
            Polar angle(s) in [0, pi].

        Returns
        -------
        jax.Array
            Function value(s). Same shape as the broadcast of inputs, or
            (len(r), len(lam), len(th)) for 1D array inputs.

        Provenance
        ----------
        MATLAB source : @ballfun/feval.m
        Chebfun commit: 7574c77
        """
        r = jnp.asarray(r, dtype=jnp.float64)
        lam = jnp.asarray(lam, dtype=jnp.float64)
        th = jnp.asarray(th, dtype=jnp.float64)

        # Scalar case
        if r.ndim == 0 and lam.ndim == 0 and th.ndim == 0:
            vals = self.fevalm(r[jnp.newaxis], lam[jnp.newaxis], th[jnp.newaxis])
            return vals[0, 0, 0]

        # 1D arrays: tensor product grid
        if r.ndim == 1 and lam.ndim == 1 and th.ndim == 1:
            return self.fevalm(r, lam, th)

        # Point-by-point (flat arrays of same shape)
        r_flat = r.ravel()
        lam_flat = lam.ravel()
        th_flat = th.ravel()

        def _eval_one(ri: jax.Array, li: jax.Array, ti: jax.Array) -> jax.Array:
            return self.fevalm(ri[jnp.newaxis], li[jnp.newaxis], ti[jnp.newaxis])[0, 0, 0]

        vals_flat = jax.vmap(_eval_one)(r_flat, lam_flat, th_flat)
        vals = vals_flat.reshape(r.shape)
        if self.is_real:
            vals = jnp.real(vals)
        return vals

    # ------------------------------------------------------------------
    # Arithmetic (immutable: return new Ballfun)
    # ------------------------------------------------------------------

    def __neg__(self) -> "Ballfun":
        """Negate: -f."""
        return Ballfun(coeffs=-self.coeffs, is_real=self.is_real, domain=self.domain)

    def __pos__(self) -> "Ballfun":
        """Unary plus: +f."""
        return self

    def __add__(self, other: "Ballfun | float | int") -> "Ballfun":
        """Add two Ballfun objects or add a scalar.

        Provenance
        ----------
        MATLAB source : @ballfun/plus.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, (int, float, complex)):
            # Add scalar to DC coefficient (index [0, n//2, p//2])
            new_coeffs = self.coeffs
            m, n, p = self.shape
            dc_idx_n = n // 2
            dc_idx_p = p // 2
            new_coeffs = new_coeffs.at[0, dc_idx_n, dc_idx_p].add(complex(other))
            return Ballfun(coeffs=new_coeffs, is_real=self.is_real, domain=self.domain)
        if isinstance(other, Ballfun):
            # Pad to common size
            c1, c2 = self.coeffs, other.coeffs
            m1, n1, p1 = c1.shape
            m2, n2, p2 = c2.shape
            m = max(m1, m2)
            n = max(n1, n2)
            p = max(p1, p2)
            # Make m odd, n/p even
            m = m + 1 - m % 2
            n = n + n % 2
            p = max(4, p + p % 2)

            def _pad_coeffs(c: jax.Array, target_m: int, target_n: int, target_p: int) -> jax.Array:
                cm, cn, cp = c.shape
                # Pad m (append zeros at end)
                if cm < target_m:
                    c = jnp.concatenate(
                        [c, jnp.zeros((target_m - cm, cn, cp), dtype=c.dtype)], axis=0
                    )
                # Pad n (insert zeros symmetrically in Fourier)
                if cn < target_n:
                    dn = target_n - cn
                    left = dn // 2
                    right = dn - left
                    c = jnp.concatenate(
                        [
                            jnp.zeros(
                                (target_m, left, target_p if cp == target_p else cp), dtype=c.dtype
                            ),
                            c,
                            jnp.zeros(
                                (target_m, right, target_p if cp == target_p else cp), dtype=c.dtype
                            ),
                        ],
                        axis=1,
                    )
                # Pad p (insert zeros symmetrically in Fourier)
                cp_new = c.shape[2]
                if cp_new < target_p:
                    dp = target_p - cp_new
                    low = dp // 2
                    high = dp - low
                    c = jnp.concatenate(
                        [
                            jnp.zeros((target_m, target_n, low), dtype=c.dtype),
                            c,
                            jnp.zeros((target_m, target_n, high), dtype=c.dtype),
                        ],
                        axis=2,
                    )
                return c

            c1_pad = _pad_coeffs(c1, m, n, p)
            c2_pad = _pad_coeffs(c2, m, n, p)
            new_coeffs = c1_pad + c2_pad
            new_is_real = self.is_real and other.is_real
            return Ballfun(coeffs=new_coeffs, is_real=new_is_real, domain=self.domain)
        return NotImplemented

    def __radd__(self, other: "float | int") -> "Ballfun":
        return self.__add__(other)

    def __sub__(self, other: "Ballfun | float | int") -> "Ballfun":
        """Subtract: f - g or f - scalar.

        Provenance
        ----------
        MATLAB source : @ballfun/minus.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Ballfun):
            return self.__add__(other.__neg__())
        return self.__add__(-other)

    def __rsub__(self, other: "float | int") -> "Ballfun":
        return self.__neg__().__add__(other)

    def __mul__(self, other: "Ballfun | float | int | complex") -> "Ballfun":
        """Pointwise multiply: f .* g or f .* scalar.

        Provenance
        ----------
        MATLAB source : @ballfun/times.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, (int, float, complex)):
            return Ballfun(
                coeffs=self.coeffs * complex(other),
                is_real=self.is_real and isinstance(other, (int, float)),
                domain=self.domain,
            )
        if isinstance(other, Ballfun):
            # Multiply via physical space (inverse transform, multiply, transform)
            c1 = np.array(self.coeffs)
            c2 = np.array(other.coeffs)
            m1, n1, p1 = c1.shape
            m2, n2, p2 = c2.shape
            # Use common size (at least the sum for convolution accuracy)
            m = max(m1 + m2 - 1, m1, m2)
            n = n1 + n2
            p = p1 + p2
            m = m + 1 - m % 2
            n = n + n % 2
            p = max(4, p + p % 2)

            # Pad both to (m, n, p)
            def _pad_np(c: np.ndarray, tm: int, tn: int, tp: int) -> np.ndarray:
                cm, cn, cp = c.shape
                out = np.zeros((tm, tn, tp), dtype=complex)
                r_start = 0
                n_start = (tn - cn) // 2
                p_start = (tp - cp) // 2
                out[r_start : r_start + cm, n_start : n_start + cn, p_start : p_start + cp] = c
                return out

            c1p = _pad_np(c1, m, n, p)
            c2p = _pad_np(c2, m, n, p)

            v1 = _coeffs2vals_3d(c1p)
            v2 = _coeffs2vals_3d(c2p)
            v_prod = v1 * v2
            c_prod = _vals2coeffs_3d(v_prod)
            new_is_real = self.is_real and other.is_real
            return Ballfun(
                coeffs=jnp.asarray(c_prod, dtype=jnp.complex128),
                is_real=new_is_real,
                domain=self.domain,
            )
        return NotImplemented

    def __rmul__(self, other: "float | int | complex") -> "Ballfun":
        return self.__mul__(other)

    def __truediv__(self, other: "float | int | complex") -> "Ballfun":
        """Divide by scalar: f / c.

        Provenance
        ----------
        MATLAB source : @ballfun/mrdivide.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, (int, float, complex)):
            return self.__mul__(1.0 / other)
        return NotImplemented

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def sum(self) -> float:
        """Triple integral of f over the unit ball.

        Computes integral_ball f dV = integral_0^1 integral_{-pi}^{pi}
        integral_0^pi f(r,lam,th) r^2 sin(th) dr dlam dth.

        Returns
        -------
        float
            The triple integral.

        Notes
        -----
        For a constant function f = c, sum() = c * 4*pi/3.

        Provenance
        ----------
        MATLAB source : @ballfun/sum3.m, @ballfun/integral.m
        Chebfun commit: 7574c77
        """
        cfs = np.array(self.coeffs)
        m_orig, n, p_orig = cfs.shape

        # Step 1: zero-pad coefficients by 2 in r and theta (matches MATLAB coeffs3(f,m+2,n,p+2))
        # r axis (Chebyshev): append zeros at the end (high-degree coefficients → zero-pad)
        # theta axis (Fourier, fftshift order): insert zeros at both ends to keep DC at center
        m = m_orig + 2
        p = p_orig + 2
        F_big = np.zeros((m, n, p), dtype=complex)
        # For Fourier (fftshift), the DC mode shifts from p_orig//2 to p//2.
        # Place original wavenumbers at the correct positions in the padded array.
        theta_offset = p // 2 - p_orig // 2  # = 1 for even p_orig
        F_big[:m_orig, :, theta_offset : theta_offset + p_orig] = cfs

        # Step 2: extract DC lambda slice (0-th Fourier mode, index n//2)
        dc_lam = n // 2
        F_rth = F_big[:, dc_lam, :]  # shape (m, p)

        # Step 3: pad F_rth to (m+2, p+2) with one zero column on each theta side
        # and two zero rows appended in r (MATLAB: [zeros(m,1),F,zeros(m,1);zeros(2,p+2)])
        m2 = m + 2  # = m_orig + 4
        p2 = p + 2  # = p_orig + 4
        F_pad = np.zeros((m2, p2), dtype=complex)
        F_pad[:m, 1 : 1 + p] = F_rth  # embed with one-zero padding on theta sides
        # last two r rows remain zero (already initialized)

        # Step 4: build multiplication matrix for r^2 in Chebyshev-T basis.
        # r^2 = (T_0 + T_2)/2.  T_in * T_0 = T_in, T_in * T_2 = (T_{in+2} + T_{|in-2|})/2.
        # So [Mr2]_{out, in} = 0.5 * delta(out,in)
        #                    + 0.25 * delta(out, in+2)  [if in+2 < m2]
        #                    + 0.25 * delta(out, |in-2|)
        # Special cases: for in=0, |in-2|=2 = in+2, so the two T_2 terms coincide → coeff = 0.5
        Mr2 = np.zeros((m2, m2))
        for i in range(m2):
            Mr2[i, i] += 0.5  # T_0 term
            if i + 2 < m2:
                Mr2[i + 2, i] += 0.25  # upper diagonal from T_2
            j_low = abs(i - 2)
            Mr2[j_low, i] += 0.25  # lower diagonal from T_2

        # Step 5: build multiplication matrix for sin(theta) in Fourier basis.
        # Fourier coefficients stored as k = -p2//2, ..., p2//2-1 (fftshift order).
        # sin(th) = (exp(ith) - exp(-ith))/(2i) = (e^{ith} terms: coeff +1/(2i) at k=+1, -1/(2i) at k=-1)
        # In the fftshift ordering with p2 modes, k=+1 is at index p2//2+1 and k=-1 at index p2//2-1.
        # Multiplication by e^{ith}: shifts k → k+1, i.e., [Mplus]_{k+1, k} = 1.
        # Multiplication by e^{-ith}: shifts k → k-1, i.e., [Mminus]_{k-1, k} = 1.
        # Msin = (1/(2i)) * Mplus + (-1/(2i)) * Mminus = -0.5j * Mplus + 0.5j * Mminus
        # MATLAB trigspec.multmat(p, [0.5i; 0; -0.5i]) uses fftshift ordering.
        # The Fourier coeff vector [0.5i, 0, -0.5i] corresponds to:
        #   k=-1 → 0.5i, k=0 → 0, k=+1 → -0.5i
        # which is: f(theta) = 0.5i*exp(-ith) - 0.5i*exp(ith) = sin(th).  ✓
        # The Toeplitz multiplication matrix: [Msin]_{out, in} = coeff[out - in]
        # where coeff[k] is the Fourier coefficient at wavenumber k.
        Msin = np.zeros((p2, p2), dtype=complex)
        # In fftshift order, index j corresponds to wavenumber j - p2//2.
        # [Msin]_{out, in} = c_{(out - p2//2) - (in - p2//2)} = c_{out - in}
        # sin(th): c_{-1} = 0.5i, c_{0} = 0, c_{+1} = -0.5i
        for out_idx in range(p2):
            for in_idx in range(p2):
                dk = out_idx - in_idx  # wavenumber shift
                if dk == -1:
                    Msin[out_idx, in_idx] = 0.5j
                elif dk == 1:
                    Msin[out_idx, in_idx] = -0.5j

        # Step 6: apply Jacobian multiplication F = Mr2 * F_pad * Msin.T
        F_jac = Mr2 @ F_pad @ Msin.T

        # Step 7: integration weight vectors
        # int_0^1 T_j(r) dr (Chebyshev T on [-1,1] but we only want [0,1])
        # Using: int_0^1 T_j(r) dr from MATLAB sum3 formula:
        #   mod(j,4)==0: -1/(j^2-1)  [special: j=0 → 1]
        #   mod(j,4)==1: 1/(j+1)
        #   mod(j,4)==2: -1/(j^2-1)
        #   mod(j,4)==3: -1/(j-1)
        int_cheb = np.zeros(m2, dtype=float)
        for j in range(m2):
            r = j % 4
            if j == 0:
                int_cheb[j] = 1.0
            elif r == 0:
                int_cheb[j] = -1.0 / (j * j - 1)
            elif r == 1:
                int_cheb[j] = 1.0 / (j + 1)
            elif r == 2:
                int_cheb[j] = -1.0 / (j * j - 1)
            else:  # r == 3
                int_cheb[j] = -1.0 / (j - 1)

        # int_0^pi exp(i*k*th) dth (Fourier on [0, pi])
        # = pi           if k=0
        # = -i*((-1)^k - 1)/k   if k != 0
        # (from MATLAB: Listp = (1:p2).' - floor(p2/2)-1 gives k from -(p2//2) to p2//2-1
        #  IntTheta(k==0) = pi, else = -1i*((-1)^k - 1)/k)
        int_theta = np.zeros(p2, dtype=complex)
        p2_mid = p2 // 2  # index of k=0 in fftshift ordering
        for idx in range(p2):
            k = idx - p2_mid  # wavenumber
            if k == 0:
                int_theta[idx] = np.pi
            else:
                int_theta[idx] = -1j * ((-1.0) ** k - 1.0) / k

        # Step 8: integrate over lambda (multiply by 2*pi for the DC lambda mode)
        int_theta *= 2.0 * np.pi

        # Step 9: I = int_cheb @ F_jac @ int_theta
        I = int_cheb @ F_jac @ int_theta

        if self.is_real:
            I = float(np.real(I))
        return I

    def integral(self) -> float:
        """Triple integral of f over the unit ball.

        Alias for ``sum()``.

        Returns
        -------
        float
            The triple integral.

        Provenance
        ----------
        MATLAB source : @ballfun/integral.m
        Chebfun commit: 7574c77
        """
        return self.sum()

    # ------------------------------------------------------------------
    # Simplify / reduce
    # ------------------------------------------------------------------

    def simplify(self, tol: float | None = None) -> "Ballfun":
        """Remove negligible coefficients.

        Parameters
        ----------
        tol : float or None, optional
            Tolerance for coefficient removal. Defaults to machine epsilon.

        Returns
        -------
        Ballfun
            Simplified representation.

        Provenance
        ----------
        MATLAB source : @ballfun/simplify.m
        Chebfun commit: 7574c77
        """
        if tol is None:
            tol = _EPS
        cfs = np.array(self.coeffs)
        vscale = float(np.max(np.abs(cfs)))
        if vscale == 0.0:
            return self
        threshold = tol * vscale
        # Keep coefficients above threshold
        mask = np.abs(cfs) < threshold
        cfs_clean = cfs.copy()
        cfs_clean[mask] = 0.0
        return Ballfun(
            coeffs=jnp.asarray(cfs_clean, dtype=jnp.complex128),
            is_real=self.is_real,
            domain=self.domain,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def plot(self, n_pts: int = 50, ax=None, **kwargs):
        """Plot the Ballfun on a spherical slice (r=1 surface).

        Shows the function values on the unit sphere boundary as a
        pseudocolor plot in Mollweide projection.

        Parameters
        ----------
        n_pts : int
            Number of grid points per angular direction.
        ax : matplotlib axes, optional
            Axes to plot on. If None, a new figure is created.
        **kwargs
            Extra keyword arguments passed to ``pcolormesh``.

        Returns
        -------
        ax : matplotlib axes
        """
        import matplotlib.pyplot as plt

        lam = np.linspace(-np.pi, np.pi, 2 * n_pts)
        theta = np.linspace(0, np.pi, n_pts)
        LAM, THETA = np.meshgrid(lam, theta, indexing="ij")
        R = np.ones_like(LAM)

        eval_fn = jax.vmap(lambda ri, li, ti: self(ri, li, ti))
        vals = np.asarray(eval_fn(
            jnp.array(R.ravel()),
            jnp.array(LAM.ravel()),
            jnp.array(THETA.ravel()),
        )).reshape(LAM.shape)

        # Convert to Cartesian for 3D surface plot
        X = np.sin(THETA) * np.cos(LAM)
        Y = np.sin(THETA) * np.sin(LAM)
        Z = np.cos(THETA)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(
            (vals - vals.min()) / (vals.max() - vals.min() + 1e-16)
        ), rstride=1, cstride=1, shade=False)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return ax

    def plot_slice(self, axis: str = "z", level: float = 0.0,
                   n_pts: int = 80, ax=None, **kwargs):
        """Plot a cross-sectional slice through the ball.

        Parameters
        ----------
        axis : str
            Which axis to slice: 'x', 'y', or 'z'.
        level : float
            The value at which to slice (default 0 = midplane).
        n_pts : int
            Grid resolution.
        ax : matplotlib axes, optional
        **kwargs
            Passed to ``pcolormesh``.

        Returns
        -------
        ax : matplotlib axes
        """
        import matplotlib.pyplot as plt

        t1 = np.linspace(-1, 1, n_pts)
        t2 = np.linspace(-1, 1, n_pts)
        T1, T2 = np.meshgrid(t1, t2, indexing="ij")

        if axis == "z":
            X, Y, Z = T1, T2, np.full_like(T1, level)
        elif axis == "y":
            X, Z, Y = T1, T2, np.full_like(T1, level)
        else:
            Y, Z, X = T1, T2, np.full_like(T1, level)

        R = np.sqrt(X**2 + Y**2 + Z**2)
        mask = R <= 1.0
        LAM = np.arctan2(Y, X)
        THETA = np.where(R > 0, np.arccos(np.clip(Z / np.maximum(R, 1e-16), -1, 1)), 0.0)

        vals = np.full(R.shape, np.nan)
        idx = mask.ravel()
        if idx.any():
            r_pts = jnp.array(R.ravel()[idx])
            l_pts = jnp.array(LAM.ravel()[idx])
            t_pts = jnp.array(THETA.ravel()[idx])
            # Use vmap for pointwise evaluation (not tensor-product)
            eval_fn = jax.vmap(lambda ri, li, ti: self(ri, li, ti))
            vals_valid = np.asarray(eval_fn(r_pts, l_pts, t_pts)).ravel()
            flat_vals = vals.ravel()
            flat_vals[idx] = vals_valid
            vals = flat_vals.reshape(R.shape)

        if ax is None:
            fig, ax = plt.subplots()

        pcm = ax.pcolormesh(T1, T2, vals.T, shading="auto", **kwargs)
        ax.set_aspect("equal")
        circle = plt.Circle((0, 0), 1, fill=False, color="k", lw=1)
        ax.add_patch(circle)
        plt.colorbar(pcm, ax=ax)
        return ax

    def __repr__(self) -> str:
        """Compact display like MATLAB Chebfun.

        Examples
        --------
        >>> f = Ballfun.from_function(lambda x, y, z: x**2 + y**2 + z**2)
        >>> repr(f)
        'Ballfun(shape=(m, n, p), domain=[0,1]x[-pi,pi]x[0,pi], is_real=True)'
        """
        m, n, p = self.shape
        return (
            f"Ballfun(shape=({m}, {n}, {p}), domain=[0,1]x[-pi,pi]x[0,pi], is_real={self.is_real})"
        )
