# uses-numpy: ETDRK4 phi-function contour integration uses numpy (setup phase, not JIT-safe)
"""ETDRK4 time-stepping solvers for 3-D periodic and sphere semilinear PDEs.

Implements:
  - ``spin3`` : 3-D periodic PDE on [x0,x1]x[y0,y1]x[z0,z1] via 3-D FFT
  - ``spinsphere`` : PDE on the unit sphere via the doubled-up Fourier (DFS) method

Both use the fourth-order ETDRK4 scheme of Cox & Matthews (2002) with the
Kassam-Trefethen (2005) contour-integral method for phi-functions.

Translated from MATLAB Chebfun @expinteg/computeCoeffs.m, @expinteg/oneStep.m,
@spinoperator/solvepde.m, @spinop3/discretize.m, @spinopsphere/discretize.m
(commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from chebfunjax.spin.solver import (
    _compute_contour,
    _phi_eval_contour,
    _psi_eval_contour,
)
from chebfunjax.spin.spinop3 import SpinOp3
from chebfunjax.spin.spinopsphere import SpinOpSphere

# ---------------------------------------------------------------------------
# 3-D ETDRK4 coefficient computation (diagonal L)
# ---------------------------------------------------------------------------


def _compute_etdrk4_coeffs_3d(
    dt: float, L_tensor: np.ndarray, M: int = 32
) -> dict:
    """Compute ETDRK4 coefficients for the 3-D diagonal linear operator.

    Extends :func:`~chebfunjax.spin.solver._compute_etdrk4_coeffs` to 3-D by
    flattening the N×N×N diagonal tensor, computing coefficients, then
    reshaping back.

    Parameters
    ----------
    dt : float
        Time-step.
    L_tensor : np.ndarray, shape (N, N, N), complex
        Diagonal of the 3-D linear operator in Fourier space.
    M : int
        Number of contour points.

    Returns
    -------
    coeffs : dict
        Keys: ``E_half``, ``E_full``, ``psi12``, ``B2``, ``B3``, ``B4``,
        each shape (N, N, N).

    Provenance
    ----------
    MATLAB source : @expinteg/computeCoeffs.m (etdrk4 branch)
    Chebfun commit: 7574c77
    """
    shape = L_tensor.shape   # (N, N, N)
    L_flat = L_tensor.ravel()  # (N^3,)

    LR = _compute_contour(dt, L_flat, M)   # (N^3, M)

    phi1 = _phi_eval_contour(1, LR)        # (N^3,)
    phi2 = _phi_eval_contour(2, LR)        # (N^3,)
    phi3 = _phi_eval_contour(3, LR)        # (N^3,)
    psi12 = _psi_eval_contour(1, 0.5, LR)  # (N^3,)

    if np.isrealobj(L_flat):
        phi1, phi2, phi3, psi12 = (
            np.real(phi1), np.real(phi2), np.real(phi3), np.real(psi12)
        )

    E_half = np.exp(0.5 * dt * L_flat).reshape(shape)
    E_full = np.exp(dt * L_flat).reshape(shape)
    psi12 = (dt * psi12).reshape(shape)
    B2 = (dt * (2.0 * phi2 - 4.0 * phi3)).reshape(shape)
    B3 = B2.copy()
    B4 = (dt * (-phi2 + 4.0 * phi3)).reshape(shape)

    return {
        "E_half": E_half,
        "E_full": E_full,
        "psi12": psi12,
        "B2": B2,
        "B3": B3,
        "B4": B4,
    }


# ---------------------------------------------------------------------------
# 3-D ETDRK4 single time-step
# ---------------------------------------------------------------------------


def _etdrk4_step_3d(
    u_hat: np.ndarray,
    nonlin_vals,
    coeffs: dict,
    dealias: Optional[np.ndarray],
) -> np.ndarray:
    """Advance 3-D Fourier coefficients by one ETDRK4 step.

    The nonlinear part has no differentiation (Nc = 1), as in Chebfun's
    spinop3 implementation.

    Parameters
    ----------
    u_hat : np.ndarray, shape (N, N, N), complex
        Current 3-D Fourier coefficients (output of fftn).
    nonlin_vals : callable(u_vals) -> array
        Nonlinear operator in physical space.
    coeffs : dict
        Output of :func:`_compute_etdrk4_coeffs_3d`.
    dealias : np.ndarray or None, shape (N, N, N), bool
        3-D dealiasing mask. True = keep mode. None = no dealiasing.

    Returns
    -------
    u_new : np.ndarray, shape (N, N, N), complex

    Provenance
    ----------
    MATLAB source : @expinteg/oneStep.m
    Chebfun commit: 7574c77
    """
    E_half = coeffs["E_half"]
    E_full = coeffs["E_full"]
    psi12 = coeffs["psi12"]
    B2 = coeffs["B2"]
    B3 = coeffs["B3"]
    B4 = coeffs["B4"]

    def _nonlin_hat(c_hat):
        u_v = np.fft.ifftn(c_hat)
        nv = nonlin_vals(u_v)
        return np.fft.fftn(nv)

    # Stage 2: a = E_half * u
    a_hat = E_half * u_hat
    Na = _nonlin_hat(a_hat)

    # Stage 3: b = E_half * u + psi12 * N(a)
    b_hat = E_half * u_hat + psi12 * Na
    Nb = _nonlin_hat(b_hat)

    # Stage 4: c = E_full * u + 2*psi12 * N(b)
    c_hat = E_full * u_hat + 2.0 * psi12 * Nb
    Nc_hat = _nonlin_hat(c_hat)

    # Solution update
    u_new = E_full * u_hat + B2 * Na + B3 * Nb + B4 * Nc_hat

    if dealias is not None:
        u_new = np.where(dealias, u_new, 0.0 + 0.0j)

    return u_new


# ---------------------------------------------------------------------------
# Sphere ETDRK4: matrix-exponential phi-functions (non-diagonal L)
# ---------------------------------------------------------------------------


def _compute_etdrk4_coeffs_sphere(
    dt: float, L_mat: np.ndarray, M: int = 32
) -> dict:
    """Compute ETDRK4 coefficients for the sphere's block-diagonal linear operator.

    Since L is block-diagonal (one N x N block per lambda-wavenumber), we
    compute the matrix exponentials block-by-block using scipy.linalg.expm
    and the phi-functions via the formula (Sidje 1998):

        exp(t*A) = sum_k t^k/k! * A^k  (for small A, use Padé)
        phi_1(t*A)*v = integral_0^1 exp((1-s)*t*A) v ds

    For the phi-functions in the ETDRK4 scheme, we use the augmented-matrix
    approach (Al-Mohy & Higham 2011) which computes exp of an augmented
    matrix to extract phi_1, phi_2, phi_3 simultaneously.

    For each N×N block B_m (lambda-wavenumber m), we compute:
        E_half_m = expm(dt/2 * B_m)
        E_full_m = expm(dt   * B_m)
        psi12_m  = (dt/2) * phi_1(dt/2 * B_m)
        B2_m     = dt * (2*phi_2(dt*B_m) - 4*phi_3(dt*B_m))
        B4_m     = dt * (-phi_2(dt*B_m) + 4*phi_3(dt*B_m))

    using the relation expm([A, v; 0, 0]) = [expm(A), phi_1(A)*v; 0, 1].

    Parameters
    ----------
    dt : float
        Time-step.
    L_mat : np.ndarray, shape (N^2, N^2), complex
        Block-diagonal Laplace-Beltrami matrix scaled by lin_scale.
    M : int
        Unused (kept for API compatibility; matrix exponentials are exact).

    Returns
    -------
    coeffs : dict
        Keys: ``E_half``, ``E_full``, ``psi12``, ``B2``, ``B4``,
        each shape (N^2, N^2) (block-diagonal matrices).

    Notes
    -----
    The augmented-matrix approach for phi-functions is described in:
    Al-Mohy, A.H. and Higham, N.J. (2011). Computing the action of the
    matrix exponential, with an application to exponential integrators.
    SIAM J. Sci. Comput. 33(2), 488-511.

    Provenance
    ----------
    MATLAB source : @expinteg/computeCoeffs.m, @spinopsphere/discretize.m
    Chebfun commit: 7574c77
    """
    from scipy.linalg import expm

    N2 = L_mat.shape[0]
    N = int(np.round(np.sqrt(N2)))

    E_half = np.zeros_like(L_mat)
    E_full = np.zeros_like(L_mat)
    psi12 = np.zeros_like(L_mat)
    B2 = np.zeros_like(L_mat)
    B4 = np.zeros_like(L_mat)

    for j in range(N):
        sl = slice(j * N, (j + 1) * N)
        Bj = L_mat[sl, sl]                   # (N, N) block for lambda-mode j

        # Matrix exponentials
        E_half[sl, sl] = expm(0.5 * dt * Bj)
        E_full[sl, sl] = expm(dt * Bj)

        # Full-step phi-functions via column-by-column augmented matrix approach
        phi2_mat = _compute_phi_matrix(dt * Bj, l=2)
        phi3_mat = _compute_phi_matrix(dt * Bj, l=3)

        B2[sl, sl] = dt * (2.0 * phi2_mat - 4.0 * phi3_mat)
        B4[sl, sl] = dt * (-phi2_mat + 4.0 * phi3_mat)

        # Half-step phi_1 for psi12
        phi1_half = _compute_phi_matrix(0.5 * dt * Bj, l=1)
        psi12[sl, sl] = 0.5 * dt * phi1_half

    B3 = B2.copy()

    return {
        "E_half": E_half,
        "E_full": E_full,
        "psi12": psi12,
        "B2": B2,
        "B3": B3,
        "B4": B4,
    }


def _compute_phi_matrix(A: np.ndarray, l: int) -> np.ndarray:
    """Compute the phi_l matrix function phi_l(A) for a dense matrix A.

    Uses the augmented-matrix approach: for each standard basis vector e_j,
    compute phi_l(A) * e_j via expm of a (n+l) x (n+l) augmented matrix,
    then assemble the columns into the full matrix.

    The augmented system for computing phi_l(A) * v is:

        expm([[A,  v,  0, ..., 0],
              [0,  0,  1, ..., 0],
              [0,  0,  0, ..., 0],
              ...
              [0,  0,  0, ..., 0]])[:n, n+l-1]  = phi_l(A) * v

    where the (n+l) x (n+l) augmented matrix has A in the upper-left, v as
    the (n+1)-th column, and a bidiagonal shift (ones on the superdiagonal)
    in the extra block.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Dense matrix (one N x N block of the sphere Laplacian).
    l : int
        Index of the phi-function (1, 2, or 3 for ETDRK4).

    Returns
    -------
    phi_l_A : np.ndarray, shape (n, n), complex

    Provenance
    ----------
    Standard algorithm from Higham (2008), "Functions of Matrices", Ch. 10.
    Al-Mohy & Higham (2011), SIAM J. Sci. Comput. 33(2), 488-511.
    """
    from scipy.linalg import expm

    n = A.shape[0]
    phi_mat = np.zeros((n, n), dtype=complex)
    for j in range(n):
        ej = np.zeros(n, dtype=complex)
        ej[j] = 1.0
        # Build (n+l) x (n+l) augmented matrix for phi_l(A) * ej
        aug = np.zeros((n + l, n + l), dtype=complex)
        aug[:n, :n] = A
        aug[:n, n] = ej
        # Bidiagonal shift in the extra l x l block
        for k in range(1, l):
            aug[n + k - 1, n + k] = 1.0
        phi_mat[:, j] = expm(aug)[:n, n + l - 1]

    return phi_mat


# ---------------------------------------------------------------------------
# Sphere: FFT/IFFT on N x N grid
# ---------------------------------------------------------------------------


def _sphere_fft2(u_vals: np.ndarray) -> np.ndarray:
    """2-D FFT on the sphere's doubled grid (N, N) -> Fourier coefficients.

    The DFS method uses the standard 2-D FFT.  The output u_hat[j_th, j_lam]
    corresponds to (theta-mode, lambda-mode).  In MATLAB's convention the
    Fourier coefficients are stored in fftshift order; here we use standard
    FFT ordering and keep track consistently.

    Parameters
    ----------
    u_vals : np.ndarray, shape (N, N)
        Physical values on the doubled (theta, lambda) grid.

    Returns
    -------
    u_hat : np.ndarray, shape (N^2,), complex
        Flattened Fourier coefficients (row-major: theta index first, i.e.,
        u_hat[j_th * N + j_lam]).
    """
    # fft2 with axes=(0,1): axis 0 = theta, axis 1 = lambda
    return np.fft.fft2(u_vals).ravel()


def _sphere_ifft2(u_hat: np.ndarray, N: int) -> np.ndarray:
    """2-D IFFT on the sphere's doubled grid.

    Parameters
    ----------
    u_hat : np.ndarray, shape (N^2,), complex
        Flattened Fourier coefficients.
    N : int
        Grid size per direction.

    Returns
    -------
    u_vals : np.ndarray, shape (N, N)
    """
    return np.fft.ifft2(u_hat.reshape(N, N))


# ---------------------------------------------------------------------------
# Sphere ETDRK4 single time-step (non-diagonal L)
# ---------------------------------------------------------------------------


def _etdrk4_step_sphere(
    u_hat: np.ndarray,
    nonlin_vals,
    coeffs: dict,
    N: int,
    dealias: Optional[np.ndarray],
) -> np.ndarray:
    """Advance sphere Fourier coefficients by one ETDRK4 step.

    The linear part is applied as matrix-vector products (block-diagonal
    matrix exponentials); the nonlinear part is evaluated in physical space.

    Parameters
    ----------
    u_hat : np.ndarray, shape (N^2,), complex
        Current Fourier coefficients (2-D FFT, flattened row-major).
    nonlin_vals : callable(u_vals) -> array
        Nonlinear operator in physical space.
    coeffs : dict
        Output of :func:`_compute_etdrk4_coeffs_sphere`.
    N : int
        Grid size per direction.
    dealias : np.ndarray or None, shape (N, N), bool
        2-D dealiasing mask. None = no dealiasing.

    Returns
    -------
    u_new : np.ndarray, shape (N^2,), complex

    Provenance
    ----------
    MATLAB source : @expinteg/oneStep.m
    Chebfun commit: 7574c77
    """
    E_half = coeffs["E_half"]   # (N^2, N^2)
    E_full = coeffs["E_full"]   # (N^2, N^2)
    psi12 = coeffs["psi12"]     # (N^2, N^2)
    B2 = coeffs["B2"]
    B3 = coeffs["B3"]
    B4 = coeffs["B4"]

    def _nonlin_hat(c_hat):
        u_v = _sphere_ifft2(c_hat, N)   # (N, N)
        nv = nonlin_vals(u_v)
        return _sphere_fft2(np.asarray(nv))

    def _apply_mat(M_mat, v):
        return M_mat @ v

    # Stage 2
    a_hat = _apply_mat(E_half, u_hat)
    Na = _nonlin_hat(a_hat)

    # Stage 3
    b_hat = _apply_mat(E_half, u_hat) + _apply_mat(psi12, Na)
    Nb = _nonlin_hat(b_hat)

    # Stage 4
    c_hat = _apply_mat(E_full, u_hat) + 2.0 * _apply_mat(psi12, Nb)
    Nc_hat = _nonlin_hat(c_hat)

    # Solution
    u_new = (_apply_mat(E_full, u_hat) + _apply_mat(B2, Na)
             + _apply_mat(B3, Nb) + _apply_mat(B4, Nc_hat))

    if dealias is not None:
        mask_flat = dealias.ravel()
        u_new = np.where(mask_flat, u_new, 0.0 + 0.0j)

    return u_new


# ---------------------------------------------------------------------------
# Public API: spin3()
# ---------------------------------------------------------------------------


def spin3(
    op: Union[SpinOp3, str],
    N: Optional[int] = None,
    dt: Optional[float] = None,
    *,
    dealias: bool = True,
    M: int = 32,
    verbose: bool = False,
) -> Tuple[tuple, float, np.ndarray]:
    """Solve a 3-D periodic semilinear PDE via ETDRK4.

    Given a :class:`SpinOp3` or a PDE name string, integrates

        u_t = L[u] + N[u]

    on a 3-D periodic domain from ``t0`` to ``tf`` using the fourth-order
    exponential time-differencing Runge-Kutta (ETDRK4) scheme with 3-D
    Fourier spectral discretization.

    Parameters
    ----------
    op : SpinOp3 or str
        The PDE operator.  Pass a string (e.g. ``'GL'``) to use a built-in
        3-D example.
    N : int, optional
        Number of Fourier modes per dimension.  Defaults to the built-in
        value or 32.
    dt : float, optional
        Time-step.  Defaults to the built-in value or 1e-2.
    dealias : bool, default True
        Apply 2/3-rule dealiasing in 3-D (zero top-third of modes in each
        dimension).
    M : int, default 32
        Number of contour points for phi-function evaluation.
    verbose : bool, default False
        Print progress every 10% of the integration.

    Returns
    -------
    grids : tuple of 3 np.ndarray, each shape (N, N, N)
        ``(xx, yy, zz)`` — 3-D spatial grid from ``meshgrid``.
    t : float
        Final time reached.
    u_final : np.ndarray, shape (N, N, N), float or complex
        Solution at the final time in physical space.

    Examples
    --------
    3-D Allen-Cahn:

    >>> grids, t, u = spin3('AC', N=32, dt=5e-2)
    >>> import numpy as np
    >>> np.max(np.abs(u))  # solution stays in [-1, 1]

    Notes
    -----
    The ETDRK4 scheme uses a 3-D FFT to evaluate the nonlinear term.  The
    linear part is diagonal in 3-D Fourier space (stored as N×N×N tensor).

    References
    ----------
    .. [1] S. M. Cox and P. C. Matthews, "Exponential time differencing for
       stiff systems", J. Comput. Phys. 176 (2002), 430-455.
    .. [2] H. Montanelli and N. Bootland, "Solving periodic semilinear stiff
       PDEs in 1D/2D/3D with exponential integrators", submitted (2017).

    Provenance
    ----------
    MATLAB source : @spinoperator/solvepde.m, @spinop3/discretize.m,
                    @expinteg/computeCoeffs.m, @expinteg/oneStep.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and The
        Chebfun Developers.

    See Also
    --------
    SpinOp3
    """
    # ---- Parse operator ----
    pde_name = None
    if isinstance(op, str):
        pde_name = op
        op = SpinOp3.from_name(op)

    # ---- Grid size and time-step ----
    if N is None:
        N = op.default_N(pde_name)
    if dt is None:
        dt = op.default_dt(pde_name)

    x0, x1, y0, y1, z0, z1 = [float(c) for c in op.domain]
    t0, tf = float(op.tspan[0]), float(op.tspan[1])

    # ---- 3-D spatial grid (uniform, periodic, N points per dim) ----
    x = np.linspace(x0, x1, N, endpoint=False)
    y = np.linspace(y0, y1, N, endpoint=False)
    z = np.linspace(z0, z1, N, endpoint=False)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")  # (N, N, N)

    # ---- Initial condition ----
    u0_vals = np.asarray(op.u0(xx, yy, zz), dtype=complex)

    # ---- Fourier eigenvalues of the linear operator ----
    L_tensor = op.build_linear_eigenvalues(N)   # (N, N, N)

    # ---- ETDRK4 coefficients ----
    coeffs = _compute_etdrk4_coeffs_3d(dt, L_tensor, M=M)

    # ---- Dealiasing mask ----
    dmask = op.dealias_mask(N) if dealias else None

    # ---- Initial Fourier coefficients ----
    u_hat = np.fft.fftn(u0_vals)
    if dealias and dmask is not None:
        u_hat = np.where(dmask, u_hat, 0.0 + 0.0j)

    nonlin_vals_fn = op.nonlin_vals

    # ---- Time-stepping loop ----
    t = t0
    nsteps = int(round((tf - t0) / dt))
    if nsteps <= 0:
        raise ValueError(
            f"tspan [{t0}, {tf}] with dt={dt} gives no steps. "
            "Check tspan and dt."
        )

    report_every = max(1, nsteps // 10)

    for step in range(nsteps):
        u_hat = _etdrk4_step_3d(u_hat, nonlin_vals_fn, coeffs, dmask)
        if np.any(np.isnan(u_hat)):
            raise RuntimeError(
                f"3D solution blew up at step {step}, t={t + dt:.6g}. "
                "Try a smaller time-step."
            )
        t += dt
        if verbose and (step + 1) % report_every == 0:
            pct = 100.0 * (step + 1) / nsteps
            print(f"  spin3: {pct:.0f}%  t={t:.4g}")

    # ---- Convert back to physical space ----
    u_final = np.fft.ifftn(u_hat)
    if op.is_real:
        u_final = np.real(u_final)

    return (xx, yy, zz), t, u_final


# ---------------------------------------------------------------------------
# Public API: spinsphere()
# ---------------------------------------------------------------------------


def spinsphere(
    op: Union[SpinOpSphere, str],
    N: Optional[int] = None,
    dt: Optional[float] = None,
    *,
    dealias: bool = True,
    M: int = 32,
    verbose: bool = False,
) -> Tuple[tuple, float, np.ndarray]:
    """Solve a semilinear PDE on the unit sphere via ETDRK4.

    Uses the doubled-up Fourier spectral (DFS) method: the sphere is
    represented via the periodic doubled domain lambda in [-pi, pi],
    theta in [-pi, pi].

    Parameters
    ----------
    op : SpinOpSphere or str
        The PDE operator.  Pass a string (e.g. ``'AC'``) to use a built-in
        example.
    N : int, optional
        Number of doubled-grid points per direction.  Defaults to 32.
    dt : float, optional
        Time-step.  Defaults to the built-in value or 1e-3.
    dealias : bool, default True
        Apply 2/3-rule dealiasing.
    M : int, default 32
        Unused; kept for API compatibility (matrix exponentials are exact).
    verbose : bool, default False
        Print progress.

    Returns
    -------
    grids : tuple of 2 np.ndarray, each shape (N, N)
        ``(ll, tt)`` — longitude and (doubled) colatitude grid.
    t : float
        Final time reached.
    u_final : np.ndarray, shape (N, N), float or complex
        Solution on the doubled grid at the final time.

    Examples
    --------
    Allen-Cahn on the sphere:

    >>> grids, t, u = spinsphere('AC', N=32, dt=5e-3)

    Notes
    -----
    The DFS method doubles the colatitude domain to [-pi, pi] so that the
    sphere PDE becomes a 2-D periodic PDE on a torus.  The Laplace-Beltrami
    operator in the doubled Fourier basis is block-tridiagonal (not diagonal).
    Matrix exponentials are computed via ``scipy.linalg.expm`` for each
    N×N block.

    References
    ----------
    .. [1] H. Montanelli and N. Bootland, "Solving periodic semilinear stiff
       PDEs in 1D/2D/3D with exponential integrators", submitted (2017).

    Provenance
    ----------
    MATLAB source : @spinoperator/solvepde.m, @spinopsphere/discretize.m,
                    @expinteg/computeCoeffs.m, @expinteg/oneStep.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and The
        Chebfun Developers.

    See Also
    --------
    SpinOpSphere
    """
    # ---- Parse operator ----
    pde_name = None
    if isinstance(op, str):
        pde_name = op
        op = SpinOpSphere.from_name(op)

    # ---- Grid size and time-step ----
    if N is None:
        N = op.default_N(pde_name)
    if dt is None:
        dt = op.default_dt(pde_name)

    t0, tf = float(op.tspan[0]), float(op.tspan[1])

    # ---- DFS grid: lambda in [-pi, pi], theta in [-pi, pi] (doubled) ----
    lam_pts = np.linspace(-np.pi, np.pi, N, endpoint=False)   # lambda
    th_pts = np.linspace(-np.pi, np.pi, N, endpoint=False)    # theta (doubled)
    ll, tt = np.meshgrid(lam_pts, th_pts, indexing="xy")      # (N, N)

    # ---- Initial condition on doubled grid ----
    u0_vals = np.asarray(op.u0(ll, tt), dtype=complex)        # (N, N)

    # ---- Build block-diagonal Laplace-Beltrami matrix ----
    L_mat = op.build_linear_matrix(N)      # (N^2, N^2)

    # ---- ETDRK4 coefficients (block-diagonal matrix exponentials) ----
    coeffs = _compute_etdrk4_coeffs_sphere(dt, L_mat, M=M)

    # ---- Dealiasing mask ----
    dmask = op.dealias_mask(N) if dealias else None

    # ---- Initial Fourier coefficients (flattened 2-D FFT) ----
    u_hat = _sphere_fft2(u0_vals)    # (N^2,)
    if dealias and dmask is not None:
        u_hat = np.where(dmask.ravel(), u_hat, 0.0 + 0.0j)

    nonlin_vals_fn = op.nonlin_vals

    # ---- Time-stepping loop ----
    t = t0
    nsteps = int(round((tf - t0) / dt))
    if nsteps <= 0:
        raise ValueError(
            f"tspan [{t0}, {tf}] with dt={dt} gives no steps."
        )

    report_every = max(1, nsteps // 10)

    for step in range(nsteps):
        u_hat = _etdrk4_step_sphere(
            u_hat, nonlin_vals_fn, coeffs, N, dmask
        )
        if np.any(np.isnan(u_hat)):
            raise RuntimeError(
                f"Sphere solution blew up at step {step}, t={t + dt:.6g}. "
                "Try a smaller time-step."
            )
        t += dt
        if verbose and (step + 1) % report_every == 0:
            pct = 100.0 * (step + 1) / nsteps
            print(f"  spinsphere: {pct:.0f}%  t={t:.4g}")

    # ---- Convert back to physical space ----
    u_final = _sphere_ifft2(u_hat, N)
    if op.is_real:
        u_final = np.real(u_final)

    return (ll, tt), t, u_final
