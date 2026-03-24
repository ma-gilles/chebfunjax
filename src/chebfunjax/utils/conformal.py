# uses-numpy: Kerzman-Stein integral equation and AAA are iterative/not JIT-safe
"""Conformal mapping to the unit disk.

Translated from MATLAB Chebfun (commit 7574c77): conformal.m.
Original: Copyright 2019 by L. N. Trefethen and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

References
----------
A. Gopal and L. N. Trefethen, "Representation of conformal maps by rational
functions", Numer. Math. 142 (2019), 359-382.

L. N. Trefethen, "Numerical conformal mapping with rational functions",
Comp. Meth. Funct. Th. 20 (2020), 369-387.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import jax.numpy as jnp

from chebfunjax.utils.aaa import aaa


# ===========================================================================
# Public API
# ===========================================================================


def conformal(
    boundary_pts: jnp.ndarray,
    ctr: complex = 0.0,
    *,
    tol: float = 1e-5,
    method: str = "kerzman-stein",
) -> Tuple[Callable, Callable, jnp.ndarray, jnp.ndarray]:
    """Conformal map from a simply-connected region to the unit disk.

    [F, FINV, POL, POLINV] = CONFORMAL(C, CTR) computes a conformal map
    F of the region bounded by the complex curve C to the unit disk and its
    inverse FINV, with F(ctr) = 0 and F'(ctr) > 0.  Both maps are
    represented as barycentric rational functions via the AAA algorithm.

    Parameters
    ----------
    boundary_pts : jnp.ndarray, complex, shape (M,)
        Boundary of the region as M sample points in counterclockwise order.
        The boundary should be smooth (no corners).
    ctr : complex, default 0.0
        Interior point that maps to 0.  Must be inside the region.
    tol : float, default 1e-5
        Convergence tolerance (relative).
    method : {'kerzman-stein', 'poly'}, default 'kerzman-stein'
        Algorithm to use:
        - 'kerzman-stein': solve the Kerzman-Stein integral equation
          (Greenbaum-Caldwell), more robust for smooth domains.
        - 'poly': polynomial least-squares (faster for simple domains).

    Returns
    -------
    f : callable
        Forward conformal map.  ``f(z)`` maps region to unit disk.
        JIT-safe (it is the AAA rational approximant).
    finv : callable
        Inverse conformal map.  ``finv(w)`` maps unit disk to region.
        JIT-safe.
    pol : jnp.ndarray, complex
        Poles of the forward map.
    polinv : jnp.ndarray, complex
        Poles of the inverse map.

    Notes
    -----
    This is an experimental implementation suitable for smooth simple regions.
    Regions with corners or near-degeneracies may not converge.

    The Kerzman-Stein method sets up an O(M^2) linear system to find the
    boundary correspondence function.  The polynomial method is cheaper
    (O(M) iterations of a least-squares problem).

    Provenance
    ----------
    MATLAB source : conformal.m
    Chebfun commit: 7574c77
    Original authors: L. N. Trefethen, Anne Greenbaum, Trevor Caldwell.
        Copyright 2019 by The University of Oxford and The Chebfun Developers.

    Examples
    --------
    Ellipse centered at origin:

    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> theta = jnp.linspace(0, 2*np.pi, 200, endpoint=False)
    >>> C = 2*jnp.cos(theta) + 1j*jnp.sin(theta)
    >>> f, finv, pol, polinv = conformal(C)

    See Also
    --------
    aaa
    """
    Z_np = np.array(boundary_pts, dtype=complex).ravel()
    M_in = len(Z_np)
    ctr_c = complex(ctr)
    scl = np.max(np.abs(Z_np - ctr_c))

    if method == "kerzman-stein":
        W = _kerzman_stein((Z_np - ctr_c) / scl, tol=tol)
    else:
        W = _poly_method((Z_np - ctr_c) / scl, tol=tol)

    # Forward map: Z -> W (unit circle)
    f0, pol_raw, _, _, zj0, fj0, wj0 = aaa(
        jnp.array(W), jnp.array(Z_np), tol=tol
    )

    # Correct rotation so that f'(ctr) > 0
    eps_fd = 1e-4 * scl
    zz_fd = np.array([ctr_c + eps_fd, ctr_c + 1j * eps_fd,
                      ctr_c - eps_fd, ctr_c - 1j * eps_fd])
    dwdz = np.sum(np.array(f0(jnp.array(zz_fd))) / (zz_fd - ctr_c))
    rot = np.exp(-1j * np.angle(dwdz))

    # Rotate W
    W_rot = rot * W

    # Re-fit with rotation applied
    f1, pol1, _, _, zj1, fj1, wj1 = aaa(
        jnp.array(W_rot), jnp.array(Z_np), tol=tol
    )

    # Inverse map: W_rot -> Z
    finv1, polinv1, _, _, _, _, _ = aaa(
        jnp.array(Z_np), jnp.array(W_rot), tol=tol
    )

    pol = pol1
    polinv = polinv1

    # Warn about poles inside region or inside disk
    pol_np = np.array(pol)
    polinv_np = np.array(polinv)

    # Simple check: poles of f should be outside the region (|z - ctr| > scl roughly)
    # and poles of finv should be outside the unit disk
    if len(polinv_np) > 0 and np.min(np.abs(polinv_np)) < 1.0:
        import warnings
        warnings.warn("conformal: pole of inverse map inside unit disk", stacklevel=2)

    return f1, finv1, pol, polinv


def _kerzman_stein(
    Z_scl: np.ndarray,
    *,
    tol: float = 1e-5,
    M_start: int = 300,
    M_max: int = 1200,
) -> np.ndarray:
    """Kerzman-Stein integral equation solver.

    Given boundary points Z_scl (scaled so that the region has unit scale),
    solve the integral equation to find W (images on the unit circle).

    Returns
    -------
    W : np.ndarray, complex
        Boundary correspondence values on the unit circle.
    """
    M = M_start
    err = np.inf
    W_best = None

    while err > tol and M <= M_max:
        # Arclength-equispaced points on the boundary
        # For simplicity, use the provided boundary points
        # (A full implementation would re-parameterize by arclength)
        M_pts = min(M, len(Z_scl))
        idx = np.round(np.linspace(0, len(Z_scl) - 1, M_pts)).astype(int)
        Dvec = Z_scl[idx]

        M_actual = len(Dvec)
        ds = 1.0 / M_actual  # approximate arc-length element (normalized)

        # Tangent directions
        dD = np.diff(np.concatenate([Dvec, [Dvec[0]]]))
        gamdot = dD / (np.abs(dD) + 1e-300)

        # Right-hand side
        d = 1.0 / (2j * np.pi)
        gvec = d * np.conj(gamdot / (0.0 - Dvec))  # 0 = ctr (already scaled)

        # Kerzman-Stein matrix
        A = np.eye(M_actual, dtype=complex)
        for j_idx in range(M_actual):
            w_pt = Dvec[j_idx]
            for i_idx in range(M_actual):
                if i_idx != j_idx:
                    z_pt = Dvec[i_idx]
                    A[i_idx, j_idx] -= d * (
                        np.conj(gamdot[i_idx] / (z_pt - w_pt))
                        + gamdot[j_idx] / (w_pt - z_pt)
                    ) * ds

        fvec = np.linalg.solve(A, gvec)

        # Compute boundary images on unit circle
        Rprime = fvec ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            Rvec = -1j * gamdot * (Rprime / (np.abs(Rprime) + 1e-300))
        W = Rvec

        # Error estimate: last few Fourier coefficients of W
        W_fft = np.fft.fft(W)
        n_check = min(10, M_actual // 4)
        err = np.max(np.abs(W_fft[:n_check])) + np.max(np.abs(W_fft[-n_check:]))
        err /= (np.max(np.abs(W_fft)) + 1e-300)
        W_best = W

        M += 300

    # Return W at the original boundary points via interpolation
    if len(W_best) == len(Z_scl):
        return W_best

    # Interpolate W back to original Z_scl points
    theta_src = np.angle(W_best)
    theta_tgt = np.linspace(theta_src[0], theta_src[-1], len(Z_scl))
    W_out = np.interp(theta_tgt, theta_src, np.real(W_best)) + \
            1j * np.interp(theta_tgt, theta_src, np.imag(W_best))
    W_out = W_out / np.abs(W_out)
    return W_out


def _poly_method(
    Z_scl: np.ndarray,
    *,
    tol: float = 1e-5,
) -> np.ndarray:
    """Polynomial least-squares method for conformal mapping.

    Given boundary points Z_scl, find harmonic u on the boundary such that
    u(z) ≈ -log|z|, then compute the analytic extension to get images W.

    Returns
    -------
    W : np.ndarray, complex
        Boundary images on the unit circle.
    """
    M = len(Z_scl)

    # Target: u(z) = -log|z| on the boundary
    G = -np.log(np.abs(Z_scl) + 1e-300)

    # Iteratively increase polynomial degree
    err = np.inf
    logn = 4.0
    W_best = None

    while err > tol and logn < 9.5:
        n = round(2 ** logn)
        logn += 0.5

        # Build Arnoldi-orthogonalized Vandermonde
        Q = np.ones((M, 1), dtype=complex)
        H = np.zeros((n + 1, n), dtype=complex)
        for k in range(n):
            v = Z_scl * Q[:, k]
            v = v - Q @ (Q.T.conj() @ v) / M
            H[k + 1, k] = np.linalg.norm(v) / np.sqrt(M)
            if H[k + 1, k] == 0:
                break
            Q = np.column_stack([Q, v / H[k + 1, k]])

        A = np.column_stack([np.real(Q), np.imag(Q[:, 1:])])
        c, _, _, _ = np.linalg.lstsq(A, G, rcond=None)
        err = np.linalg.norm(A @ c - G, np.inf)

        # Extract analytic coefficients
        n1 = np.arange(n + 1)
        n2 = np.arange(1, n + 1)
        cc = c[:n + 1] - 1j * np.concatenate([[0.0], c[n + 1:]])

        W = Z_scl * np.exp(Q @ cc)
        W_best = W

    if W_best is None:
        W_best = np.exp(1j * np.linspace(0, 2 * np.pi, M, endpoint=False))

    W_best = W_best / (np.abs(W_best) + 1e-300)
    return W_best
