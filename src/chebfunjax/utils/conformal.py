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

import jax.numpy as jnp
import numpy as np

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
    ctr_c = complex(ctr)
    scl = np.max(np.abs(Z_np - ctr_c))

    if method == "kerzman-stein":
        # MATLAB loop: M = 600, 900, 1200 sample points equispaced in
        # arclength; err is the norm of the 20 extreme trig
        # coefficients of the boundary correspondence function.
        M = 300
        err = np.inf
        while err > tol:
            M += 300
            W, Z_ks = _kerzman_stein((Z_np - ctr_c) / scl, M)
            Z_ks = Z_ks * scl + ctr_c
            gc = np.fft.fftshift(np.fft.fft(W)) / M
            err = np.linalg.norm(np.concatenate([gc[:10], gc[-10:]]))
            if err > tol and M >= 1200:
                import warnings
                warnings.warn("conformal did not converge", stacklevel=2)
                break
        Z_fit = Z_ks
    else:
        W = _poly_method((Z_np - ctr_c) / scl, tol=tol)
        Z_fit = Z_np

    # Forward map: Z -> W (unit circle)
    f0, pol, _, _, zj0, fj0, wj0 = aaa(
        jnp.array(W), jnp.array(Z_fit), tol=tol
    )

    # Correct rotation so that f'(ctr) > 0 (finite differences at ctr)
    zz_fd = 1e-4 * scl * np.array([1, 1j, -1, -1j])
    dwdz = np.sum(np.array(f0(jnp.array(ctr_c + zz_fd))) / zz_fd)
    rot = np.exp(-1j * np.angle(dwdz))
    f1 = lambda z: rot * f0(z)  # noqa: E731
    W_rot = rot * W

    # Inverse map: W_rot -> Z
    finv1, polinv, _, _, _, _, _ = aaa(
        jnp.array(Z_fit), jnp.array(W_rot), tol=tol
    )
    Z_np = np.asarray(Z_fit)

    # Warn about poles inside region or inside disk
    pol_np = np.array(pol)
    polinv_np = np.array(polinv)

    # MATLAB: inC = inpolygon(real(pol), imag(pol), real(Z), imag(Z)) —
    # a true point-in-polygon test of the forward-map poles against the
    # discretized boundary curve; poles of finv must lie outside the unit disk.
    import warnings
    if len(pol_np) > 0:
        from matplotlib.path import Path as _MplPath

        region = _MplPath(np.column_stack([Z_np.real, Z_np.imag]))
        pole_pts = np.column_stack([pol_np.real, pol_np.imag])
        if region.contains_points(pole_pts).any():
            warnings.warn("conformal: pole of forward map inside region",
                          stacklevel=2)
    if len(polinv_np) > 0 and np.min(np.abs(polinv_np)) < 1.0:
        warnings.warn("conformal: pole of inverse map inside unit disk", stacklevel=2)

    return f1, finv1, pol, polinv


def _kerzman_stein(
    Z_scl: np.ndarray,
    M: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Kerzman-Stein integral equation solver (MATLAB kerzstein).

    The boundary samples Z_scl (assumed equispaced in the curve
    parameter, tracing the closed curve once counterclockwise) are
    first reparametrized by arclength via trigonometric interpolation;
    the integral equation is then discretized at M points equispaced
    in arclength with the trapezoid rule, exactly as in kerzstein.

    Returns
    -------
    W : np.ndarray, complex, shape (M,)
        Boundary correspondence values on the unit circle.
    Z : np.ndarray, complex, shape (M,)
        The arclength-equispaced boundary points used.

    Provenance
    ----------
    MATLAB source : conformal.m (kerzstein subfunction; original code
        by Anne Greenbaum and Trevor Caldwell)
    Chebfun commit: 7574c77
    """
    N = len(Z_scl)
    # trig-series representation of the curve and its derivative
    coeffs = np.fft.fft(Z_scl) / N
    k = np.fft.fftfreq(N, d=1.0 / N)  # integer wavenumbers

    def C_eval(t):  # t in [0, 2*pi)
        return np.exp(1j * np.outer(t, k)) @ coeffs

    def Cp_eval(t):
        return np.exp(1j * np.outer(t, k)) @ (1j * k * coeffs)

    # cumulative arclength on a fine grid, then invert at equispaced s
    fine = max(8 * N, 4096)
    tf = 2 * np.pi * np.arange(fine + 1) / fine
    speed = np.abs(Cp_eval(tf))
    s_fine = np.concatenate([
        [0.0],
        np.cumsum((speed[1:] + speed[:-1]) / 2) * (2 * np.pi / fine)])
    S = s_fine[-1]
    svec = S * np.arange(M) / M
    tvec = np.interp(svec, s_fine, tf)
    # one Newton step: s(t) - target = 0, s'(t) = |C'(t)|
    sv = np.interp(tvec, tf, s_fine)
    tvec = tvec - (sv - svec) / np.maximum(np.abs(Cp_eval(tvec)), 1e-300)

    Dvec = C_eval(tvec)
    gamdot = Cp_eval(tvec)
    gamdot = gamdot / np.abs(gamdot)  # unit tangents

    ds = S / M
    d = 1.0 / (2j * np.pi)
    gvec = d * np.conj(gamdot / (0.0 - Dvec))  # ctr = 0 (pre-shifted)

    # A = I - d*ds*( conj(gamdot_i/(z_i - w_j)) + gamdot_j/(w_j - z_i) )
    Zi = Dvec[:, None]
    Wj = Dvec[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        K = (np.conj(gamdot[:, None] / (Zi - Wj))
             + gamdot[None, :] / (Wj - Zi))
    np.fill_diagonal(K, 0.0)
    A = np.eye(M, dtype=complex) - d * ds * K

    fvec = np.linalg.solve(A, gvec)
    Rprime = fvec ** 2
    W = -1j * gamdot * (Rprime / np.abs(Rprime))
    return W, Dvec


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
        cc = c[:n + 1] - 1j * np.concatenate([[0.0], c[n + 1:]])

        W = Z_scl * np.exp(Q @ cc)
        W_best = W

    if W_best is None:
        W_best = np.exp(1j * np.linspace(0, 2 * np.pi, M, endpoint=False))

    W_best = W_best / (np.abs(W_best) + 1e-300)
    return W_best
