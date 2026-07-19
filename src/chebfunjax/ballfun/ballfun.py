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
    from chebfunjax.tech.chebtech import Chebtech2, _coeffs_to_values
    from chebfunjax.tech.trigtech import (
        Trigtech,
        _chop_cutoff_to_ncoeffs,
        trig_coeffs2vals,
    )

    m, n, p = vals.shape
    vals_np = np.asarray(vals)

    # Transform to coefficient space
    cfs = _vals2coeffs_3d(vals_np)

    # MATLAB: vscl = max(1, max(abs(vals(:)))) — passed as the global
    # vscale into every per-direction happiness check.
    vscl = max(1.0, float(np.max(np.abs(vals_np))))

    # Per-direction coefficient envelopes (complex modulus, as in MATLAB:
    # max(max(abs(cfs),[],dim_a),[],dim_b)).
    r_cfs = jnp.asarray(np.max(np.abs(cfs), axis=(1, 2)), dtype=jnp.float64)
    l_cfs = jnp.asarray(np.max(np.abs(cfs), axis=(0, 2)), dtype=jnp.complex128)
    t_cfs = jnp.asarray(np.max(np.abs(cfs), axis=(0, 1)), dtype=jnp.complex128)

    # MATLAB builds a chebtech2 from the radial envelope and trigtechs from
    # the two Fourier envelopes and runs each tech's happinessCheck. The
    # trig check folds the two-sided spectrum (k, -k pairs) into a decaying
    # envelope — running standard_chop directly on a DC-rolled spectrum
    # never resolves because the negative-frequency tail does not decay.
    r_happy, c_r = Chebtech2.happiness_check(
        r_cfs, _coeffs_to_values(r_cfs), vscale=vscl
    )
    l_happy, l_cut = Trigtech.happiness_check(
        l_cfs, trig_coeffs2vals(l_cfs), vscale=vscl
    )
    t_happy, t_cut = Trigtech.happiness_check(
        t_cfs, trig_coeffs2vals(t_cfs), vscale=vscl
    )

    c_r = max(int(c_r), 1)
    c_lam = _chop_cutoff_to_ncoeffs(int(l_cut), n) if l_happy else n
    c_th = _chop_cutoff_to_ncoeffs(int(t_cut), p) if t_happy else p
    c_lam = max(c_lam, 2)
    if c_lam % 2 != 0:
        c_lam += 1
    c_th = max(c_th, 4)
    if c_th % 2 != 0:
        c_th += 1

    # Suggest new grid sizes. MATLAB (@ballfun/constructor.m,
    # ballfunHappiness): a RESOLVED direction keeps its current grid size —
    # shrinking it to the cutoff mid-loop makes it unhappy again on the next
    # pass (standard_chop needs >= 17 points) and the loop seesaws forever
    # (constant functions used to hang here). Unresolved directions grow by
    # 1.5x. Final chop to the cutoffs happens after the loop.
    new_m = m if r_happy else round(1.5 * m)
    new_n = n if l_happy else round(1.5 * n)
    new_p = p if t_happy else round(1.5 * p)

    # Ensure parity constraints: m odd, n even, p even >= 4
    new_m = new_m + 1 - new_m % 2  # odd
    new_n = new_n + new_n % 2  # even
    new_p = max(4, new_p + new_p % 2)  # even >= 4

    resolved = [r_happy, l_happy, t_happy]
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

    @classmethod
    def empty(cls) -> "Ballfun":
        """The empty Ballfun (MATLAB ballfun()): no data; isempty() is
        True and operations on it are undefined.

        Provenance
        ----------
        MATLAB source : @ballfun/isempty.m
        Chebfun commit: 7574c77
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "_is_empty_object", True)
        return obj

    def isempty(self) -> bool:
        """True for the empty Ballfun (MATLAB isempty).

        Provenance
        ----------
        MATLAB source : @ballfun/isempty.m
        Chebfun commit: 7574c77
        """
        return getattr(self, "_is_empty_object", False)

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
        # Initial grid sizes: MATLAB @ballfun/constructor.m starts every
        # direction at tpref.minSamples (17) with parity adjustments
        # (grid1 odd; grid2, grid3 even). Starting below standard_chop's
        # 17-point minimum makes every direction unhappy regardless of
        # the function.
        m = 17  # odd
        n = 18  # even
        p = 18  # even >= 4

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

            # MATLAB failure test: any PAIRWISE grid product exceeding
            # maxSample (not the triple product, which grows too fast to
            # bound each direction meaningfully).
            if max(m * n, n * p, m * p) > max_sample:
                warnings.warn(
                    f"Ballfun.from_function: grid size ({m}, {n}, {p}) "
                    f"exceeded max_sample={max_sample}. "
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

    def __pow__(self, n) -> "Ballfun":
        """Pointwise power (MATLAB power).  Integer n uses repeated
        products; fractional n re-approximates.

        Provenance
        ----------
        MATLAB source : @ballfun/power.m
        Chebfun commit: 7574c77
        """
        if isinstance(n, int) or (isinstance(n, float)
                                  and float(n).is_integer()):
            n = int(n)
            if n < 0:
                return Ballfun.from_function(
                    lambda r, lam, th: self(r, lam, th) ** n,
                    spherical=True)
            out = None
            base = self
            k = n
            if k == 0:
                return Ballfun.from_function(
                    lambda x, y, z: 1.0 + 0.0 * x)
            while k:
                if k & 1:
                    out = base if out is None else out * base
                base = base * base if k > 1 else base
                k >>= 1
            return out
        return Ballfun.from_function(
            lambda r, lam, th: self(r, lam, th) ** n, spherical=True)

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

    def diff(self, dim: int = 1, k: int = 1,
             coord: str = "cartesian") -> "Ballfun":
        """k-th partial derivative in direction ``dim``.

        Parameters
        ----------
        dim : int, default 1
            With ``coord='cartesian'`` (default): 1 -> d/dx, 2 -> d/dy,
            3 -> d/dz.  With ``coord='spherical'``: 1 -> d/dr, 2 -> d/dlam,
            3 -> d/dtheta (MATLAB convention).
        k : int, default 1
            Order (applied by iterated first derivatives).
        coord : {'cartesian', 'spherical'}, default 'cartesian'
            Coordinate frame of the derivative.  ``'spherical'`` mirrors
            MATLAB ``diff(f, dim, 'spherical')``.

        Returns
        -------
        Ballfun

        Provenance
        ----------
        MATLAB source : @ballfun/diff.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2019 by The University of Oxford
            and The Chebfun Developers.
        """
        F = np.asarray(self.coeffs, dtype=np.complex128)
        if str(coord).lower().startswith("spher"):
            for _ in range(int(k)):
                F = _ballfun_spherical_diff(F, dim)
        else:
            for _ in range(int(k)):
                F = _ballfun_onediff_cart(F, dim)
        return Ballfun(coeffs=jnp.asarray(F), is_real=self.is_real,
                       domain=self.domain)

    def compose(self, op) -> "Ballfun":
        """Re-approximate op(f) with the constructor (MATLAB compose;
        added by Claude Fable 5)."""
        return Ballfun.from_function(
            lambda r, lam, th: op(self(r, lam, th)), spherical=True)

    def exp(self):
        return self.compose(jnp.exp)

    def sin(self):
        return self.compose(jnp.sin)

    def cos(self):
        return self.compose(jnp.cos)

    def sqrt(self):
        return self.compose(jnp.sqrt)

    def log(self):
        return self.compose(jnp.log)

    def tan(self):
        return self.compose(jnp.tan)

    def tanh(self):
        return self.compose(jnp.tanh)

    def sinh(self):
        return self.compose(jnp.sinh)

    def cosh(self):
        return self.compose(jnp.cosh)

    def real(self) -> "Ballfun":
        """Real part of f (MATLAB real): re-approximates ``Re f``.

        Provenance
        ----------
        MATLAB source : @ballfun/real.m
        Chebfun commit: 7574c77
        """
        return self.compose(jnp.real)

    def imag(self) -> "Ballfun":
        """Imaginary part of f (MATLAB imag).

        Provenance
        ----------
        MATLAB source : @ballfun/imag.m
        Chebfun commit: 7574c77
        """
        return self.compose(jnp.imag)

    def conj(self) -> "Ballfun":
        """Complex conjugate of f (MATLAB conj).

        Provenance
        ----------
        MATLAB source : @ballfun/conj.m
        Chebfun commit: 7574c77
        """
        return self.compose(jnp.conj)

    def abs(self) -> "Ballfun":
        """Absolute value of f (MATLAB abs): re-approximates ``|f|``.
        Assumes f does not change sign / pass through zero.

        Provenance
        ----------
        MATLAB source : @ballfun/abs.m
        Chebfun commit: 7574c77
        """
        return self.compose(jnp.abs)

    def __abs__(self) -> "Ballfun":
        return self.abs()

    def iszero(self) -> bool:
        """True iff f is exactly the zero function (MATLAB iszero:
        ``nnz(coeffs) == 0``).

        Provenance
        ----------
        MATLAB source : @ballfun/iszero.m
        Chebfun commit: 7574c77
        """
        return bool(np.count_nonzero(np.asarray(self.coeffs)) == 0)

    def isequal(self, other: "Ballfun") -> bool:
        """True iff f == g, i.e. ``iszero(f - g)`` (MATLAB isequal).

        Provenance
        ----------
        MATLAB source : @ballfun/isequal.m
        Chebfun commit: 7574c77
        """
        if not isinstance(other, Ballfun):
            return False
        return (self - other).iszero()

    def laplacian(self) -> "Ballfun":
        """Scalar Laplacian: f_xx + f_yy + f_zz.

        Provenance
        ----------
        MATLAB source : @ballfun/laplacian.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2019 by The University of Oxford
            and The Chebfun Developers.
        """
        return self.diff(1, 2) + self.diff(2, 2) + self.diff(3, 2)

    def grad(self) -> tuple["Ballfun", "Ballfun", "Ballfun"]:
        """Cartesian gradient (f_x, f_y, f_z).

        Provenance
        ----------
        MATLAB source : @ballfun/grad.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2019 by The University of Oxford
            and The Chebfun Developers.
        """
        return self.diff(1), self.diff(2), self.diff(3)

    def norm(self) -> float:
        """L2 norm of f over the unit ball: sqrt(integral of |f|^2).

        Provenance
        ----------
        MATLAB source : @ballfun/norm.m
        Chebfun commit: 7574c77
        """
        c = np.array(self.coeffs)
        m, n, p = c.shape
        # MATLAB pads to (2m, 2n, 2p) before forming |f|^2 to avoid aliasing.
        mp, np_, pp = 2 * m + 1 - (2 * m) % 2, 2 * n + (2 * n) % 2, max(4, 2 * p)
        big = np.zeros((mp, np_, pp), dtype=complex)
        n_off = np_ // 2 - n // 2
        p_off = pp // 2 - p // 2
        big[:m, n_off : n_off + n, p_off : p_off + p] = c
        v = _coeffs2vals_3d(big)
        f2 = Ballfun(
            coeffs=jnp.asarray(_vals2coeffs_3d(v * np.conj(v)), dtype=jnp.complex128),
            is_real=True,
            domain=self.domain,
        )
        return float(np.sqrt(abs(f2.sum())))

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

    def plot(self, *args, **kwargs):
        """Plot this Ballfun using the shared MATLAB-faithful renderer."""
        from chebfunjax.plotting import plot_ball_slices

        style = "ball"
        if args:
            if len(args) > 1 or not isinstance(args[0], str):
                raise ValueError(
                    "Ballfun.plot accepts at most one positional argument: "
                    "'WedgeAz' or 'WedgePol'."
                )
            style = args[0]
        return plot_ball_slices(self, style=style, **kwargs)

    def surf(self, **kwargs):
        """Surface/slice plot of this Ballfun."""
        from chebfunjax.plotting import surf_ball

        return surf_ball(self, **kwargs)

    def isosurface(self, levels=None, **kwargs):
        """Isosurface plot of this Ballfun (calls :func:`chebfunjax.plotting.isosurface_ball`)."""
        from chebfunjax.plotting import isosurface_ball
        return isosurface_ball(self, levels=levels, **kwargs)

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

    @staticmethod
    def poisson(f, lmax: int = 8, nr: int = 24) -> "Ballfun":
        r"""Solve the Poisson equation :math:`\\nabla^2 u = f` on the ball.

        Homogeneous Dirichlet boundary condition ``u = 0`` on ``r = 1``.
        Solved spectrally: the right-hand side is expanded in real
        spherical harmonics at each radius, and each mode's radial
        function is found by Chebyshev collocation of

        .. math::
            u_{lm}'' + \\frac{2}{r} u_{lm}'
            - \\frac{l(l+1)}{r^2} u_{lm} = f_{lm},

        with ``u_{lm}(1)=0`` and pole regularity at the origin
        (``u_{lm}(0)=0`` for ``l>0``, ``u_{lm}'(0)=0`` for ``l=0``).
        Implemented and verified by Claude Opus 4.8 against manufactured
        solutions ``u = (1-r^2) r^l Y_l^m`` (task #17).

        Parameters
        ----------
        f : Ballfun or callable
            Right-hand side ``f(r, lambda, theta)`` in spherical coords.
        lmax : int, default 8
            Spherical-harmonic bandwidth.
        nr : int, default 24
            Radial Chebyshev resolution.

        Returns
        -------
        Ballfun

        Provenance
        ----------
        MATLAB source : @ballfun/poisson.m (result-equivalent).
        Chebfun commit: 7574c77
        """
        ev = _ballfun_poisson_evaluator(f, lmax, nr)
        return Ballfun.from_function(
            lambda x, y, z: jnp.asarray(ev(x, y, z)), spherical=True)

    def rotate(self, phi: float = 0.0, theta: float = 0.0,
               psi: float = 0.0) -> "Ballfun":
        """Rotate by Euler angles (ZYZ, same convention as
        Spherefun.rotate): radius is unchanged, angular coordinates
        are pulled back through R = Rz(phi) Ry(theta) Rz(psi).

        Provenance
        ----------
        MATLAB source : @ballfun/rotate.m
        Chebfun commit: 7574c77
        """
        ca, sa = np.cos(phi), np.sin(phi)
        cb, sb = np.cos(theta), np.sin(theta)
        cc, sc = np.cos(psi), np.sin(psi)
        Rz1 = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        Rz2 = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
        R = jnp.asarray(Rz1 @ Ry @ Rz2)

        def g(r, lam, th):
            x = jnp.cos(lam) * jnp.sin(th)
            y = jnp.sin(lam) * jnp.sin(th)
            z = jnp.cos(th)
            xp = R[0, 0] * x + R[1, 0] * y + R[2, 0] * z
            yp = R[0, 1] * x + R[1, 1] * y + R[2, 1] * z
            zp = R[0, 2] * x + R[1, 2] * y + R[2, 2] * z
            return self(r, jnp.arctan2(yp, xp),
                        jnp.arccos(jnp.clip(zp, -1.0, 1.0)))

        return Ballfun.from_function(g, spherical=True)

    def mean(self, dim: int = 1):
        """Average over one spherical coordinate (MATLAB mean(f, dim)):
        dim=1 averages over r and returns a Spherefun; dim=2 averages
        over lambda and dim=3 over theta, each returning a Diskfun in
        the surviving coordinates (doubled-angle convention for the
        colatitude).

        Provenance
        ----------
        MATLAB source : @ballfun/mean.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        xg, wg = _np.polynomial.legendre.leggauss(48)

        def _avg(r_, l_, t_):
            # Ballfun.__call__ grids 1D args; broadcast to a common
            # shape and evaluate pointwise via 2D reshape instead.
            r_, l_, t_ = jnp.broadcast_arrays(r_, l_, t_)
            shp = r_.shape
            vals = self(r_.reshape(-1, 1), l_.reshape(-1, 1),
                        t_.reshape(-1, 1))
            return jnp.asarray(vals).reshape(shp)

        if dim == 1:
            from chebfunjax.spherefun.spherefun import Spherefun
            rq = jnp.asarray((xg + 1.0) / 2.0)
            wq = jnp.asarray(wg / 2.0)

            def g(lam, th):
                v = _avg(rq, lam[..., None], th[..., None])
                return jnp.sum(wq * v, axis=-1)

            return Spherefun.from_function(g)

        from chebfunjax.diskfun.diskfun import Diskfun
        if dim == 2:
            lq = jnp.asarray(_np.pi * xg)
            wq = jnp.asarray(wg / 2.0)

            def g(t, rr):
                v = _avg(rr[..., None], lq, jnp.abs(t)[..., None])
                return jnp.sum(wq * v, axis=-1)

            return Diskfun.from_function(g)
        if dim == 3:
            tq = jnp.asarray(_np.pi * (xg + 1.0) / 2.0)
            wq = jnp.asarray(wg / 2.0)

            def g(t, rr):
                v = _avg(rr[..., None], t[..., None], tq)
                return jnp.sum(wq * v, axis=-1)

            return Diskfun.from_function(g)
        raise ValueError("dim must be 1, 2, or 3")

    def mean2(self, dims=(1, 2)):
        """Average over two spherical coordinates, returning a 1D
        Chebfun in the survivor (MATLAB mean2(f, dims)): r on [0, 1],
        lambda on [-pi, pi] (trig), theta on [0, pi].

        Provenance
        ----------
        MATLAB source : @ballfun/mean2.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import chebfun
        xg, wg = _np.polynomial.legendre.leggauss(48)
        grids = {
            1: (jnp.asarray((xg + 1.0) / 2.0),
                jnp.asarray(wg / 2.0)),
            2: (jnp.asarray(_np.pi * xg), jnp.asarray(wg / 2.0)),
            3: (jnp.asarray(_np.pi * (xg + 1.0) / 2.0),
                jnp.asarray(wg / 2.0)),
        }
        doms = {1: (0.0, 1.0), 2: (-_np.pi, _np.pi),
                3: (0.0, _np.pi)}
        d1, d2 = int(dims[0]), int(dims[1])
        surv = ({1, 2, 3} - {d1, d2}).pop()
        (q1, w1), (q2, w2) = grids[d1], grids[d2]

        def g(t):
            args = [None, None, None]
            args[surv - 1] = t[..., None, None]
            args[d1 - 1] = q1[:, None]
            args[d2 - 1] = q2[None, :]
            r_, l_, t_ = jnp.broadcast_arrays(*args)
            shp = r_.shape
            vals = jnp.asarray(self(
                r_.reshape(-1, 1), l_.reshape(-1, 1),
                t_.reshape(-1, 1))).reshape(shp)
            return jnp.sum(w1[:, None] * w2[None, :] * vals,
                           axis=(-2, -1))

        return chebfun(g, domain=doms[surv], trig=(surv == 2))

    def mean3(self) -> float:
        """Average over the whole ball: sum(f) / (4 pi / 3)
        (MATLAB mean3).

        Provenance
        ----------
        MATLAB source : @ballfun/mean3.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        return float(self.sum()) / (4.0 * _np.pi / 3.0)

    @staticmethod
    def solharm(l: int, m: int) -> "Ballfun":
        r"""Solid harmonic :math:`R_{lm} = \sqrt{2l+3}\, r^l\,
        Y_l^m(\lambda, \theta)` , normalized to unit L2 norm on the
        ball (MATLAB ballfun.solharm).

        Provenance
        ----------
        MATLAB source : @ballfun/solharm.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.spherefun.spherefun import _real_ylm_values
        c = float(np.sqrt(2 * l + 3))

        def g(r, lam, th):
            return c * r ** l * jnp.asarray(
                _real_ylm_values(l, m, jnp.asarray(lam).ravel(),
                                 jnp.asarray(th).ravel())
            ).reshape(jnp.broadcast_shapes(
                jnp.asarray(r).shape, jnp.asarray(lam).shape,
                jnp.asarray(th).shape))

        return Ballfun.from_function(g, spherical=True)

    @staticmethod
    def helmholtz(f, K: float, bc=None, m: int = 39,
                  n: int = 40, p: int = 41,
                  bc_type: str = "dirichlet") -> "Ballfun":
        r"""Solve the Helmholtz equation
        :math:`\nabla^2 u + K^2 u = f` on the ball with Dirichlet or
        Neumann boundary data at ``r = 1`` (MATLAB ballfun helmholtz).

        Spectral solve in coefficient space: the PDE decouples in the
        lambda-Fourier direction, and each mode is a 2D (r, theta)
        generalized Sylvester equation solved by Bartels-Stewart, with a
        special Legendre-basis branch for the DC mode of the
        Poisson-Neumann (``K = 0``) problem.

        Parameters
        ----------
        f : Ballfun or callable
            Right-hand side ``f(r, lam, theta)`` (spherical coordinates).
        K : float
            Wavenumber (K = 0 reduces to Poisson).
        bc : callable, float, ndarray, or None
            Boundary data.  ``None`` (or the zero array) is homogeneous; a
            callable ``bc(lam, th)`` or scalar is sampled onto the trig
            grid; a 2D array is taken as an ``n x p`` Fourier(lambda) x
            Fourier(theta) coefficient matrix.  For ``bc_type='dirichlet'``
            this is ``u(1,.,.)``; for ``bc_type='neumann'`` it is the radial
            derivative ``du/dr(1,.,.)``.
        m, n, p : int
            Discretization sizes (Chebyshev in r, Fourier in lambda and
            theta).  ``m`` is made odd; ``n`` and ``p`` even.
        bc_type : {'dirichlet', 'neumann'}, default 'dirichlet'
            Boundary condition type imposed at ``r = 1``.

        Provenance
        ----------
        MATLAB source : @ballfun/helmholtz.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2019 by The University of Oxford
            and The Chebfun Developers.
        """
        isNeumann = str(bc_type).lower().startswith("neu")
        m, n, p = int(m), int(n), int(p)
        # Parity: m odd (radial), n and p even (doubled Fourier structure).
        m = m + 1 - m % 2
        n = n + n % 2
        p = p + p % 2

        # Right-hand side coefficients on the solve grid.
        if isinstance(f, Ballfun):
            f_is_real = f.is_real
            Fc = _resize_coeffs3_ball(
                np.asarray(f.coeffs, dtype=np.complex128), m, n, p)
        else:
            fb = Ballfun.from_function(f, spherical=True)
            f_is_real = fb.is_real
            Fc = _resize_coeffs3_ball(
                np.asarray(fb.coeffs, dtype=np.complex128), m, n, p)

        # Boundary coefficients (n x p Fourier-Fourier matrix or None).
        if bc is None:
            BC1 = None
        elif callable(bc):
            BC1 = _sample_boundary_coeffs(bc, n, p)
        elif np.ndim(bc) >= 2:
            BC1 = _resize_fourier2(np.asarray(bc, dtype=np.complex128), n, p)
        else:
            cval = complex(bc)
            BC1 = _sample_boundary_coeffs(
                lambda ll, tt: cval + 0.0 * ll, n, p)

        CFS = _ballfun_helmholtz_spectral(Fc, float(K), BC1, isNeumann)
        return Ballfun.from_coeffs(
            jnp.asarray(CFS, dtype=jnp.complex128), is_real=bool(f_is_real))

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


# ============================================================================
# Cartesian spectral differentiation (translated from @ballfun/diff.m)
# ============================================================================


def _fourier_wavenumbers_1d(n: int) -> np.ndarray:
    """Wavenumbers -floor(n/2) .. ceil(n/2)-1 (coefficient storage order)."""
    return np.arange(-(n // 2), -(n // 2) + n, dtype=np.float64)


def _fourier_multmat_3band(n: int, a_m1: complex, a_p1: complex) -> np.ndarray:
    """Fourier multiplication matrix for a_m1*e^{-i t} + a_p1*e^{i t}.

    In coefficient space multiplication is convolution, so the matrix has
    a_p1 on the first subdiagonal (mode k -> k+1) and a_m1 on the first
    superdiagonal (mode k -> k-1).
    """
    M = np.zeros((n, n), dtype=np.complex128)
    idx = np.arange(n - 1)
    M[idx + 1, idx] = a_p1
    M[idx, idx + 1] = a_m1
    return M


def _ballfun_spherical_diff(F: np.ndarray, dim: int) -> np.ndarray:
    """First derivative in the spherical variable dim (1=r, 2=lambda, 3=theta)."""
    from chebfunjax.discretization.ultras import convertmat, diffmat

    m, n, p = F.shape
    if dim == 1:
        DC1 = np.asarray(diffmat(m, 1))
        S01 = np.asarray(convertmat(m, 0, 0))
        out = np.linalg.solve(S01, DC1 @ F.reshape(m, -1))
        return out.reshape(m, n, p)
    if dim == 2:
        ks = _fourier_wavenumbers_1d(n)
        return F * (1j * ks)[None, :, None]
    ks = _fourier_wavenumbers_1d(p)
    return F * (1j * ks)[None, None, :]


def _ballfun_onediff_cart(F: np.ndarray, dim: int) -> np.ndarray:
    """One Cartesian derivative of a CFF coefficient tensor.

    Follows @ballfun/diff.m: expand the tensor by two wavenumbers in each
    direction (the variable-coefficient chain rule increases bandwidth by
    one), take the three spherical derivatives, and combine them with
    multiplication operators for r, sin/cos(lambda), sin/cos(theta).
    """
    from chebfunjax.discretization.ultras import multmat

    m, n, p = F.shape
    m_t = m + (m % 2) + 2
    n_t = n + 2
    p_t = p + (p % 2) + 2

    Fexp = np.zeros((m_t, n_t, p_t), dtype=np.complex128)
    n_off = n_t // 2 - n // 2
    p_off = p_t // 2 - p // 2
    Fexp[:m, n_off:n_off + n, p_off:p_off + p] = F

    dR = _ballfun_spherical_diff(Fexp, 1)
    dLam = _ballfun_spherical_diff(Fexp, 2)
    dTh = _ballfun_spherical_diff(Fexp, 3)

    Mr = np.asarray(multmat(m_t, jnp.array([0.0, 1.0]), 0))
    MsinL = _fourier_multmat_3band(n_t, 0.5j, -0.5j)
    McosL = _fourier_multmat_3band(n_t, 0.5, 0.5)
    MsinT = _fourier_multmat_3band(p_t, 0.5j, -0.5j)
    McosT = _fourier_multmat_3band(p_t, 0.5, 0.5)

    def solve_r(X):
        return np.linalg.solve(Mr, X.reshape(m_t, -1)).reshape(m_t, n_t, p_t)

    def mul_lam(X, M):
        # apply the multiplication operator along the lambda axis
        return np.tensordot(X, M, axes=([1], [1])).transpose(0, 2, 1)

    def mul_th(X, M):
        # apply the multiplication operator along the theta axis
        return np.tensordot(X, M, axes=([2], [1]))

    def solve_th(X, M):
        # X such that (result multiplied by M along theta) = X
        return np.linalg.solve(
            M, X.reshape(-1, p_t).T).T.reshape(m_t, n_t, p_t)

    if dim == 1:
        dR = mul_lam(dR, McosL)
        dLam = -mul_lam(solve_r(dLam), MsinL)
        dTh = mul_lam(solve_r(dTh), McosL)
        out = mul_th(dR, MsinT) + solve_th(dLam, MsinT) + mul_th(dTh, McosT)
    elif dim == 2:
        dR = mul_lam(dR, MsinL)
        dLam = mul_lam(solve_r(dLam), McosL)
        dTh = mul_lam(solve_r(dTh), MsinL)
        out = mul_th(dR, MsinT) + solve_th(dLam, MsinT) + mul_th(dTh, McosT)
    else:
        dTh = solve_r(dTh)
        out = mul_th(dR, McosT) - mul_th(dTh, MsinT)
    return out


# ============================================================================
# Spectral ball Helmholtz / Poisson solver (translated from
# @ballfun/helmholtz.m).  Coefficient space Chebyshev(r) x Fourier(lambda)
# x Fourier(theta); the PDE decouples in lambda so each Fourier mode gives a
# 2D (r, theta) generalized Sylvester problem solved by Bartels-Stewart.
# Supports Dirichlet and Neumann boundary data at r = 1, with a special
# Legendre-basis branch for the DC lambda mode of the Neumann-Poisson (K=0)
# problem.
# ============================================================================


def _trigspec_multmat(n: int, vec) -> np.ndarray:
    """Fourier (trigspec) multiplication matrix (MATLAB trigspec.multmat).

    ``vec`` is the centred trigonometric coefficient list
    ``[c_{-K}, ..., c_0, ..., c_K]``; the returned ``n x n`` Toeplitz matrix
    satisfies ``M[out, in] = c_{out-in}`` in fftshift ordering.
    """
    vec = np.asarray(vec, dtype=np.complex128)
    L = len(vec)
    K = L // 2
    M = np.zeros((n, n), dtype=np.complex128)
    for dk in range(-K, K + 1):
        c = vec[K + dk]
        if c == 0:
            continue
        if dk >= 0:
            rows = np.arange(dk, n)
            cols = np.arange(0, n - dk)
        else:
            rows = np.arange(0, n + dk)
            cols = np.arange(-dk, n)
        M[rows, cols] = c
    return M


def _trigspec_diffmat(n: int, k: int) -> np.ndarray:
    """Fourier (trigspec) k-th derivative matrix ``diag((1i*wavenumber)^k)``."""
    w = _fourier_wavenumbers_1d(n)
    return np.diag((1j * w) ** k)


def _resize_coeffs3_ball(F: np.ndarray, m: int, n: int, p: int) -> np.ndarray:
    """Resize a CFF coefficient tensor to ``(m, n, p)`` (MATLAB coeffs3).

    Chebyshev (axis 0) is truncated/zero-padded at the tail; the two Fourier
    axes are truncated/zero-padded symmetrically about the DC mode.
    """
    F = np.asarray(F, dtype=np.complex128)
    mf, nf, pf = F.shape
    out = np.zeros((m, n, p), dtype=np.complex128)
    mc = min(m, mf)

    def _fourier_slices(dst, src):
        lo = min(src // 2, dst // 2)
        hi = min(src - src // 2, dst - dst // 2)
        d0 = dst // 2 - lo
        s0 = src // 2 - lo
        length = lo + hi
        return slice(d0, d0 + length), slice(s0, s0 + length)

    dn, sn = _fourier_slices(n, nf)
    dp, sp = _fourier_slices(p, pf)
    out[:mc, dn, dp] = F[:mc, sn, sp]
    return out


def _resize_fourier2(mat: np.ndarray, n: int, p: int) -> np.ndarray:
    """Centred zero-pad / truncate an ``(nf, pf)`` Fourier-Fourier matrix to
    ``(n, p)`` (both axes about their DC mode)."""
    mat = np.asarray(mat, dtype=np.complex128)
    nf, pf = mat.shape
    out = np.zeros((n, p), dtype=np.complex128)

    def _slices(dst, src):
        lo = min(src // 2, dst // 2)
        hi = min(src - src // 2, dst - dst // 2)
        d0 = dst // 2 - lo
        s0 = src // 2 - lo
        length = lo + hi
        return slice(d0, d0 + length), slice(s0, s0 + length)

    dn, sn = _slices(n, nf)
    dp, sp = _slices(p, pf)
    out[dn, dp] = mat[sn, sp]
    return out


def _sample_boundary_coeffs(g, n: int, p: int) -> np.ndarray:
    """Sample boundary data ``g(lambda, theta)`` on the ``n x p`` trig grid
    and return its Fourier(lambda) x Fourier(theta) coefficients in the
    ballfun (fftshift + even/odd fix) convention."""
    lam = np.linspace(-np.pi, np.pi, n, endpoint=False)
    th = np.linspace(-np.pi, np.pi, p, endpoint=False)
    ll, tt = np.meshgrid(lam, th, indexing="ij")  # (n, p)
    vals = np.asarray(g(jnp.asarray(ll), jnp.asarray(tt)), dtype=np.complex128)
    if vals.shape != (n, p):
        vals = np.broadcast_to(vals, (n, p)).astype(np.complex128).copy()
    # Angular transform only (m = 1 skips the radial Chebyshev step).
    return np.asarray(_vals2coeffs_3d(vals[None, :, :])[0])


def _compute_boundary_rows(BC1, m: int, n: int, p: int, isNeumann: bool):
    """Build the boundary coefficient matrices ``BC1, BC2`` and the two
    boundary-condition rows ``bc`` (MATLAB ComputeBoundary).

    ``BC1`` is the ``n x p`` Fourier(lambda) x Fourier(theta) coefficient
    matrix of the boundary data (``None`` means homogeneous).  ``BC2`` is the
    symmetry-implied companion; ``bc`` is the ``2 x m`` matrix of radial
    boundary functionals (values for Dirichlet, derivatives for Neumann).
    """
    if BC1 is None:
        BC1 = np.zeros((n, p), dtype=np.complex128)
    BC1 = np.asarray(BC1, dtype=np.complex128)

    th_idx = np.arange(p)
    if not isNeumann:
        scale = (-1.0) ** (th_idx - p // 2)
        BC2 = scale[None, :] * BC1
        bc1 = np.ones(m, dtype=np.complex128)
        bc2 = (-1.0) ** np.arange(m)
    else:
        scale = (-1.0) ** (th_idx - p // 2 + 1)
        BC2 = scale[None, :] * BC1
        j = np.arange(m, dtype=np.float64)
        bc1 = j ** 2
        bc2 = ((-1.0) ** np.arange(1, m + 1)) * bc1
    bc = np.vstack([bc1, bc2]).astype(np.complex128)
    return BC1, BC2, bc


def _bs_solve(A, B, C, D, E) -> np.ndarray:
    """Solve the generalized Sylvester equation ``A X B^T + C X D^T = E``.

    Solved directly through the Kronecker system
    ``(B (x) A + D (x) C) vec(X) = vec(E)`` using column-major vectorization
    (``vec(A X B^T) = (B (x) A) vec(X)``).  Correct but O((mp)^3); used only
    as a reference — the ball Helmholtz solve uses :func:`_gen_sylv_reduced`,
    which amortizes a single QZ of the (fixed) left pencil across all Fourier
    modes.
    """
    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    C = np.asarray(C, dtype=np.complex128)
    D = np.asarray(D, dtype=np.complex128)
    E = np.asarray(E, dtype=np.complex128)
    if np.linalg.norm(E) == 0.0:
        return np.zeros_like(E)
    my, nx = E.shape
    M = np.kron(B, A) + np.kron(D, C)
    x = np.linalg.solve(M, E.flatten(order="F"))
    return x.reshape((my, nx), order="F")


def _gen_sylv_reduced(AA, CC, Q, Z, B, D, E) -> np.ndarray:
    """Solve ``A X B^T + C X D^T = E`` given the complex QZ of ``(A, C)``.

    ``AA, CC, Q, Z`` come from ``scipy.linalg.qz(A, C, output='complex')``
    (``A = Q AA Z^H``, ``C = Q CC Z^H``, with ``AA, CC`` upper triangular).
    With ``Y = Z^H X`` and ``F = Q^H E`` the system reduces to
    ``AA Y B^T + CC Y D^T = F``; since ``AA, CC`` are triangular the rows of
    ``Y`` are recovered bottom-up, each from a single ``p x p`` solve.  This
    is the Bartels-Stewart reduction, exploiting that the radial (left) pencil
    is identical for every lambda mode.
    """
    E = np.asarray(E, dtype=np.complex128)
    if np.linalg.norm(E) == 0.0:
        return np.zeros_like(E)
    m2, p = E.shape
    F = Q.conj().T @ E
    Y = np.zeros((m2, p), dtype=np.complex128)
    YBt = np.zeros((m2, p), dtype=np.complex128)
    YDt = np.zeros((m2, p), dtype=np.complex128)
    for i in range(m2 - 1, -1, -1):
        rhs = F[i, :].copy()
        for j in range(i + 1, m2):
            rhs -= AA[i, j] * YBt[j, :] + CC[i, j] * YDt[j, :]
        Mi = AA[i, i] * B + CC[i, i] * D
        yi = np.linalg.solve(Mi, rhs)
        Y[i, :] = yi
        YBt[i, :] = yi @ B.T
        YDt[i, :] = yi @ D.T
    return Z @ Y


def _apply_cols(fn, mat: np.ndarray) -> np.ndarray:
    """Apply a 1D transform ``fn`` (real-input) column-wise to a complex
    matrix, splitting the real and imaginary parts."""
    mat = np.asarray(mat, dtype=np.complex128)
    out = []
    for jj in range(mat.shape[1]):
        col = mat[:, jj]
        cr = np.asarray(fn(jnp.asarray(np.real(col), dtype=jnp.float64)))
        ci = np.asarray(fn(jnp.asarray(np.imag(col), dtype=jnp.float64)))
        out.append(cr + 1j * ci)
    return np.stack(out, axis=1)


def _ballfun_helmholtz_spectral(Fc: np.ndarray, K: float, BC1,
                                isNeumann: bool) -> np.ndarray:
    r"""Solve :math:`\nabla^2 u + K^2 u = f` on the unit ball in coefficient
    space and return the CFF coefficient tensor of ``u``.

    Parameters
    ----------
    Fc : np.ndarray, shape (m, n, p)
        Chebyshev(r) x Fourier(lambda) x Fourier(theta) coefficients of the
        right-hand side, already resized to the solve grid.
    K : float
        Helmholtz frequency (``K = 0`` is the Poisson problem).
    BC1 : np.ndarray or None
        ``n x p`` Fourier-Fourier coefficient matrix of the boundary data at
        ``r = 1`` (``None`` is homogeneous).  Dirichlet: ``u(1, .)``;
        Neumann: ``du/dr(1, .)``.
    isNeumann : bool
        Whether the boundary condition is Neumann.

    Returns
    -------
    np.ndarray, shape (m, n, p)
        CFF coefficients of the solution.

    Provenance
    ----------
    MATLAB source : @ballfun/helmholtz.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2019 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: N. Boulle and A. Townsend, "Computing with Functions on
        the Ball", SIAM J. Sci. Comput., 2019.  Per-mode Sylvester solves via
        Gardiner et al. (Bartels-Stewart), ACM TOMS 18(2), 1992.
    """
    from chebfunjax.discretization.ultras import convertmat, diffmat, multmat
    from chebfunjax.tech.trigtech import trig_coeffs2vals, trig_vals2coeffs
    from chebfunjax.utils.transforms import (
        chebvals2legcoeffs,
        legcoeffs2chebvals,
    )

    m, n, p = Fc.shape
    K = float(K)

    # The code is written with variables in order (r, theta, lambda).
    F = np.transpose(Fc, (0, 2, 1))  # permute [1 3 2] -> (m, p, n)

    # Useful spectral matrices in r (ultraspherical) and theta (trigspec).
    DC2 = np.asarray(diffmat(m, 2), dtype=np.complex128)
    S12 = np.asarray(convertmat(m, 1, 1), dtype=np.complex128)
    Mr = np.asarray(multmat(m, jnp.array([0.0, 1.0]), 1), dtype=np.complex128)
    Mr2 = np.asarray(multmat(m, jnp.array([0.5, 0.0, 0.5]), 2),
                     dtype=np.complex128)
    DC1 = np.asarray(diffmat(m, 1), dtype=np.complex128)
    Msin2 = _trigspec_multmat(p, [-0.25, 0.0, 0.5, 0.0, -0.25])
    Ip = np.eye(p, dtype=np.complex128)
    S02 = np.asarray(convertmat(m, 0, 1), dtype=np.complex128)
    DF2 = _trigspec_diffmat(p, 2)
    Mcossin = _trigspec_multmat(p, [0.25j, 0.0, 0.0, 0.0, -0.25j])
    DF1 = _trigspec_diffmat(p, 1)

    BC1m, BC2m, bc = _compute_boundary_rows(BC1, m, n, p, isNeumann)

    if abs(K) > 1:
        Lr = Mr2 @ DC2 / K ** 2 + 2 * S12 @ Mr @ DC1 / K ** 2 + Mr2 @ S02
    else:
        Lr = Mr2 @ DC2 + 2 * S12 @ Mr @ DC1 + K * K * Mr2 @ S02

    Lth = Msin2 @ DF2 + Mcossin @ DF1

    # Normalize bc so that bc(:, 2:3) is the identity (columns 1:3 0-based).
    D = bc[:, 1:3].copy()
    bc = np.linalg.solve(D, bc)  # (2, m)

    # Use the boundary conditions to remove two radial degrees of freedom.
    myS02 = S02.copy()
    c1 = myS02[:, 1:3].copy()
    c2 = Lr[:, 1:3].copy()
    myS02 = myS02 - myS02[:, 1:3] @ bc
    Lr = Lr - Lr[:, 1:3] @ bc

    CFS = np.zeros((m, p, n), dtype=np.complex128)
    n_dc = n // 2                       # DC lambda mode (shift = floor(n/2)+1)
    cols = np.array([0] + list(range(3, m)))   # MATLAB [1 4:end] (0-based)

    # The left (radial) pencil is identical for every lambda mode, so its QZ
    # is computed once and reused by the reduced Sylvester solver.
    import scipy.linalg
    Lr_sub = np.real(Lr[:m - 2][:, cols]).astype(np.float64)
    S02_sub = np.real(myS02[:m - 2][:, cols]).astype(np.float64)
    AA_qz, CC_qz, Q_qz, Z_qz = scipy.linalg.qz(Lr_sub, S02_sub,
                                               output="complex")

    for kk in range(n):
        if (np.max(np.abs(F[:, :, kk])) <= 1e-16
                and np.max(np.abs(BC1m[kk, :])) <= 1e-16
                and np.max(np.abs(BC2m[kk, :])) <= 1e-16):
            continue

        # Eliminating boundary conditions changes the rhs.
        BCk = np.linalg.solve(D, np.vstack([BC1m[kk, :], BC2m[kk, :]]))  # (2, p)

        if kk == n_dc and K == 0 and isNeumann:
            # Special Legendre-basis branch for the Poisson-Neumann DC mode.
            ff = Mr2 @ S02 @ F[:, :, kk]                # (m, p)

            p_tilde = max(2 * p - 2, 1)
            lo = p_tilde // 2 - p // 2

            Xc = np.zeros((p_tilde, m), dtype=np.complex128)
            Xc[lo:lo + p, :] = ff.T
            Xc = np.asarray(trig_coeffs2vals(jnp.asarray(Xc)))
            ff = _apply_cols(chebvals2legcoeffs, Xc[:p, :])   # (p, m)

            Xc2 = np.zeros((p_tilde, 2), dtype=np.complex128)
            Xc2[lo:lo + p, :] = BCk.T
            Xc2 = np.asarray(trig_coeffs2vals(jnp.asarray(Xc2)))
            BCLeg = _apply_cols(chebvals2legcoeffs, Xc2[:p, :])  # (p, 2)

            XLeg = np.zeros((p, m), dtype=np.complex128)
            for jdx in range(p):
                fac = (jdx + 1) * jdx                 # l(l+1), l = jdx
                A = Lr - fac * myS02
                c3 = c2 - fac * c1
                ff[jdx, :] = ff[jdx, :] - BCLeg[jdx, :] @ c3.T

                if jdx > 0:
                    Asub = A[:m - 2][:, cols]
                    X = np.linalg.solve(Asub, ff[jdx, :m - 2])
                else:
                    Asub = A[:m - 3][:, 3:]
                    Xs = np.linalg.solve(Asub, ff[jdx, :m - 3])
                    X = np.concatenate([[0.0 + 0.0j], Xs])
                col = BCLeg[jdx, :] - X @ bc[:, cols].T   # (2,)
                XLeg[jdx, :] = np.concatenate([[X[0]], col, X[1:]])

            Xcheb = _apply_cols(legcoeffs2chebvals, XLeg)     # (p, m)
            ext = np.vstack([Xcheb, Xcheb[1:-1][::-1]])       # (p_tilde, m)
            Xf = np.asarray(trig_vals2coeffs(jnp.asarray(ext)))
            CFS[:, :, kk] = Xf[lo:lo + p, :].T
        else:
            A = Lth - (kk - n_dc) ** 2 * Ip
            ff = Mr2 @ S02 @ F[:, :, kk] @ Msin2.T
            if abs(K) > 1:
                A = A / K ** 2
                ff = ff / K ** 2
            ff = ff - c1 @ BCk @ A.T - c2 @ BCk @ Msin2.T
            X = _gen_sylv_reduced(AA_qz, CC_qz, Q_qz, Z_qz,
                                  Msin2, A, ff[:m - 2, :])
            col = BCk - bc[:, cols] @ X               # (2, p)
            CFS[:, :, kk] = np.vstack([X[0:1, :], col, X[1:, :]])

    return np.transpose(CFS, (0, 2, 1))  # permute back [1 3 2] -> (m, n, p)


def _ballfun_poisson_evaluator(f, lmax: int, nr: int,
                               K: float = 0.0, bc=None,
                               bc_type: str = "dirichlet"):
    """Spectral ball Poisson/Helmholtz solver -> ev(r, lam, theta).

    Per real spherical-harmonic mode, solve the radial ODE by Chebyshev
    collocation with u(1) = bc-mode and pole regularity at the origin.
    (Poisson by Claude Opus 4.8, task #17; the K^2 term and
    non-homogeneous Dirichlet data added in the Fable 5 audit.)
    """
    from numpy.polynomial import chebyshev as _C

    from chebfunjax.diskfun.diskfun import _cheb_diff_matrix
    from chebfunjax.spherefun.spherefun import _real_ylm_values

    # Element-wise RHS evaluator (a Ballfun's __call__ grids 1D arrays,
    # so reshape to 2D to force point-by-point evaluation).
    if isinstance(f, Ballfun):
        def feval(rr, lam, th):
            rr = jnp.asarray(rr).reshape(-1, 1)
            lam = jnp.asarray(lam).reshape(-1, 1)
            th = jnp.asarray(th).reshape(-1, 1)
            return jnp.asarray(f(rr, lam, th)).reshape(-1)
    else:
        feval = f

    neumann = str(bc_type).lower().startswith("neu")

    D, x = _cheb_diff_matrix(nr)
    r = (x + 1.0) / 2.0
    Dr = 2.0 * D
    Drr = Dr @ Dr
    npts = len(r)
    last = npts - 1                      # r = 0 index (x = -1)

    nth = 2 * lmax + 2
    nph = 2 * lmax + 2
    xg, wg = np.polynomial.legendre.leggauss(nth)
    theta = np.arccos(xg)
    phi = np.linspace(-np.pi, np.pi, nph, endpoint=False)
    dph = 2.0 * np.pi / nph
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    lam_j = jnp.asarray(PH.ravel())
    th_j = jnp.asarray(TH.ravel())

    lm_list = [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]
    ycache = {(l, m): np.asarray(
        _real_ylm_values(l, m, lam_j, th_j)).reshape(TH.shape)
        for (l, m) in lm_list}

    modes = {k: np.zeros(npts) for k in lm_list}
    for ir, rr in enumerate(r):
        if rr < 1e-12:
            continue
        F = np.asarray(feval(jnp.full(lam_j.shape, float(rr)), lam_j, th_j)
                       ).reshape(TH.shape)
        for k, Y in ycache.items():
            modes[k][ir] = np.sum(F * Y * wg[:, None] * dph)

    # Dirichlet boundary data expanded in real spherical harmonics
    bc_modes = {k: 0.0 for k in lm_list}
    if bc is not None:
        if callable(bc):
            B = np.asarray(bc(lam_j, th_j), dtype=float).reshape(TH.shape)
        else:
            B = np.full(TH.shape, float(bc))
        for k, Y in ycache.items():
            bc_modes[k] = float(np.sum(B * Y * wg[:, None] * dph))

    rs = r.copy()
    rs[rs < 1e-12] = 1e-12
    vand = _C.chebvander(x, nr)
    coefs = {}
    for (l, m), flm in modes.items():
        if np.max(np.abs(flm)) < 1e-11 \
                and abs(bc_modes[(l, m)]) < 1e-13:
            continue
        Lm = Drr + np.diag(2.0 / rs) @ Dr \
            - l * (l + 1) * np.diag(1.0 / rs ** 2) \
            + (K * K) * np.eye(len(r))
        A = Lm.astype(float).copy()
        rhs = flm.astype(float).copy()
        if neumann:
            A[0, :] = Dr[0, :]            # u'(1) = bc_lm  (Neumann data)
            rhs[0] = bc_modes[(l, m)]
            # u(0) = 0 pins the origin value (regularity for l>0 and the
            # otherwise-free additive constant of the l=0 Neumann mode)
            A[last, :] = 0.0
            A[last, last] = 1.0
            rhs[last] = 0.0
        else:
            A[0, :] = 0.0
            A[0, 0] = 1.0
            rhs[0] = bc_modes[(l, m)]         # u(1) = bc_lm
            if l == 0:
                A[last, :] = Dr[last, :]      # u'(0) = 0
                rhs[last] = 0.0
            else:
                A[last, :] = 0.0
                A[last, last] = 1.0           # u(0) = 0
                rhs[last] = 0.0
        u = np.linalg.solve(A, rhs)
        coefs[(l, m)] = np.linalg.lstsq(vand, u, rcond=None)[0]

    def ev(r_, lam_, th_):
        r_ = np.atleast_1d(np.asarray(r_, dtype=float))
        lam_ = np.atleast_1d(np.asarray(lam_, dtype=float))
        th_ = np.atleast_1d(np.asarray(th_, dtype=float))
        shape = np.broadcast(r_, lam_, th_).shape
        s = 2.0 * np.broadcast_to(r_, shape).ravel() - 1.0
        lamf = jnp.asarray(np.broadcast_to(lam_, shape).ravel())
        thf = jnp.asarray(np.broadcast_to(th_, shape).ravel())
        out = np.zeros(s.shape)
        for (l, m), c in coefs.items():
            out = out + _C.chebval(s, c) * np.asarray(
                _real_ylm_values(l, m, lamf, thf))
        return out.reshape(shape)

    return ev
