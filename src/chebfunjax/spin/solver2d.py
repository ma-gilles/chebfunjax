# uses-numpy: ETDRK4 phi-function contour integration uses numpy (setup phase, not JIT-safe)
"""ETDRK4 time-stepping solver for 2D periodic semilinear PDEs.

Implements the ``spin2`` function: given a ``SpinOp2``, a grid size N, and a
time-step dt, integrates u_t = L[u] + N[u] on a 2D periodic domain using the
fourth-order exponential time-differencing Runge-Kutta (ETDRK4) scheme of
Cox and Matthews (2002) with the Kassam-Trefethen (2005) contour-integral
method for evaluating the phi-functions stably.

The 2D generalization mirrors the 1D scheme exactly: the linear operator is
diagonal in 2D Fourier space (N x N eigenvalues), and the ETDRK4 stages use
2D FFT/IFFT.

Translated from MATLAB Chebfun @spinop2 and @expinteg (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

Provenance
----------
MATLAB source : @spinoperator/solvepde.m, @expinteg/computeCoeffs.m,
                @expinteg/oneStep.m, @spinop2/spinop2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from chebfunjax.spin.solver import (
    _compute_contour,
    _phi_eval_contour,
    _psi_eval_contour,
)
from chebfunjax.spin.spinop2 import (
    SpinOp2,
    _dealias_mask_2d,
    build_linear_eigenvalues_2d,
)

# ---------------------------------------------------------------------------
# ETDRK4 coefficient computation for 2D (flat N^2 eigenvalue arrays)
# ---------------------------------------------------------------------------


def _compute_etdrk4_coeffs_2d(
    dt: float,
    L_flat: np.ndarray,
    is_real: bool,
    M: int = 32,
) -> dict:
    """Compute ETDRK4 coefficients for a flattened 2D diagonal operator.

    This is identical to the 1D version in :mod:`chebfunjax.spin.solver` but
    operates on the flattened N^2 eigenvalue vector.

    Parameters
    ----------
    dt : float
        Time-step.
    L_flat : np.ndarray, shape (N*N,), complex
        Flattened diagonal of the linear operator in 2D Fourier space.
    is_real : bool
        If True the eigenvalues are real (purely diffusive operator) and the
        upper-half contour is used; coefficients are taken as real.
    M : int
        Number of contour points.

    Returns
    -------
    coeffs : dict — same key set as :func:`chebfunjax.spin.solver._compute_etdrk4_coeffs`.

    Provenance
    ----------
    MATLAB source : @expinteg/computeCoeffs.m
    Chebfun commit: 7574c77
    """
    # For purely diffusive (real) operators use the upper-half circle
    if is_real:
        L_contour = np.real(L_flat)  # ensure real dtype for contour logic
    else:
        L_contour = L_flat

    LR = _compute_contour(dt, L_contour, M)  # (N^2, M)

    phi1 = _phi_eval_contour(1, LR)
    phi2 = _phi_eval_contour(2, LR)
    phi3 = _phi_eval_contour(3, LR)
    psi12 = _psi_eval_contour(1, 0.5, LR)

    if is_real:
        phi1, phi2, phi3, psi12 = (
            np.real(phi1),
            np.real(phi2),
            np.real(phi3),
            np.real(psi12),
        )

    E_half = np.exp(0.5 * dt * L_flat)
    E_full = np.exp(dt * L_flat)

    B2 = dt * (2.0 * phi2 - 4.0 * phi3)
    B3 = dt * (2.0 * phi2 - 4.0 * phi3)
    B4 = dt * (-phi2 + 4.0 * phi3)

    A32 = dt * psi12
    A43 = dt * psi12

    return {
        "E_half": E_half,
        "E_full": E_full,
        "A32": A32,
        "A43": A43,
        "B2": B2,
        "B3": B3,
        "B4": B4,
        "phi1": dt * phi1,
        "psi12": dt * psi12,
    }


# ---------------------------------------------------------------------------
# ETDRK4 single time-step for a single-component 2D PDE
# ---------------------------------------------------------------------------


def _etdrk4_step_2d_scalar(
    u_hat: np.ndarray,
    nonlin_vals_fn,
    coeffs: dict,
    dealias: Optional[np.ndarray],
    N: int,
) -> np.ndarray:
    """Advance the 2D Fourier coefficients by one ETDRK4 step (scalar PDE).

    The 2D Fourier array ``u_hat`` is flattened to a 1D vector to reuse the
    1D ETDRK4 coefficient structure, then reshaped back.

    Parameters
    ----------
    u_hat : np.ndarray, shape (N, N), complex
        Current 2D Fourier coefficients.
    nonlin_vals_fn : callable(u_vals) -> array
        Nonlinear operator in physical space.
    coeffs : dict
        Output of :func:`_compute_etdrk4_coeffs_2d`.
    dealias : np.ndarray or None, shape (N, N), bool
        2D dealiasing mask.
    N : int
        Number of Fourier modes per direction.

    Returns
    -------
    u_new : np.ndarray, shape (N, N), complex

    Provenance
    ----------
    MATLAB source : @expinteg/oneStep.m
    Chebfun commit: 7574c77
    """
    E_half = coeffs["E_half"]  # (N*N,)
    E_full = coeffs["E_full"]
    psi12 = coeffs["psi12"]
    B2 = coeffs["B2"]
    B3 = coeffs["B3"]
    B4 = coeffs["B4"]

    def _nonlin_coeff(c_hat_flat):
        """Compute fft2(nonlin_vals(ifft2(c_hat)))."""
        c_hat = c_hat_flat.reshape(N, N)
        u_vals = np.fft.ifft2(c_hat)
        nv = nonlin_vals_fn(u_vals)
        return np.fft.fft2(nv).ravel()

    u_flat = u_hat.ravel()

    # Stage 2: a = E_half * u
    a_flat = E_half * u_flat
    Na = _nonlin_coeff(a_flat)

    # Stage 3: b = E_half * u + psi12 * N(a)
    b_flat = E_half * u_flat + psi12 * Na
    Nb = _nonlin_coeff(b_flat)

    # Stage 4: c = E_full * u + 2*psi12 * N(b)
    c_flat = E_full * u_flat + 2.0 * psi12 * Nb
    Nc_flat = _nonlin_coeff(c_flat)

    # Solution update
    u_new_flat = E_full * u_flat + B2 * Na + B3 * Nb + B4 * Nc_flat

    u_new = u_new_flat.reshape(N, N)

    if dealias is not None:
        u_new = np.where(dealias, u_new, 0.0 + 0.0j)

    return u_new


# ---------------------------------------------------------------------------
# ETDRK4 single time-step for multi-component 2D PDEs
# ---------------------------------------------------------------------------


def _etdrk4_step_2d_multi(
    u_hats: list,
    nonlin_fns: list,
    coeffs_list: list,
    dealias: Optional[np.ndarray],
    N: int,
) -> list:
    """Advance a multi-component 2D PDE by one ETDRK4 step.

    For n_vars components, each component uses its own linear operator
    coefficients (stored in ``coeffs_list``), but shares the same nonlinear
    coupling through ``nonlin_fns``.

    Parameters
    ----------
    u_hats : list of np.ndarray, each shape (N, N)
        Current 2D Fourier coefficients per component.
    nonlin_fns : list of callables
        One callable per component: ``nonlin_fns[i](u0_vals, u1_vals, ...)``
        returns the nonlinear forcing for component i.
    coeffs_list : list of dicts
        ETDRK4 coefficient dicts (one per component).
    dealias : np.ndarray or None, shape (N, N)
    N : int

    Returns
    -------
    u_new_hats : list of np.ndarray, each shape (N, N)

    Provenance
    ----------
    MATLAB source : @expinteg/oneStep.m
    Chebfun commit: 7574c77
    """
    n_vars = len(u_hats)

    def _vals_from_hats(hats):
        return [np.fft.ifft2(h) for h in hats]

    def _nonlin_coeffs(hats):
        """Compute fft2(N_i(ifft2(h0), ifft2(h1), ...)) for each component."""
        vals = _vals_from_hats(hats)
        return [np.fft.fft2(nonlin_fns[i](*vals)) for i in range(n_vars)]

    # ---- Stage 2 (a stages) per component ----
    a_hats = [coeffs_list[i]["E_half"] * u_hats[i].ravel() for i in range(n_vars)]
    a_hats = [a.reshape(N, N) for a in a_hats]
    Na = _nonlin_coeffs(a_hats)

    # ---- Stage 3 (b stages) per component ----
    b_hats = [
        (
            coeffs_list[i]["E_half"] * u_hats[i].ravel()
            + coeffs_list[i]["psi12"] * Na[i].ravel()
        ).reshape(N, N)
        for i in range(n_vars)
    ]
    Nb = _nonlin_coeffs(b_hats)

    # ---- Stage 4 (c stages) per component ----
    c_hats = [
        (
            coeffs_list[i]["E_full"] * u_hats[i].ravel()
            + 2.0 * coeffs_list[i]["psi12"] * Nb[i].ravel()
        ).reshape(N, N)
        for i in range(n_vars)
    ]
    Nc = _nonlin_coeffs(c_hats)

    # ---- Solution update per component ----
    u_new_hats = []
    for i in range(n_vars):
        cf = coeffs_list[i]
        u_new_flat = (
            cf["E_full"] * u_hats[i].ravel()
            + cf["B2"] * Na[i].ravel()
            + cf["B3"] * Nb[i].ravel()
            + cf["B4"] * Nc[i].ravel()
        )
        u_new = u_new_flat.reshape(N, N)
        if dealias is not None:
            u_new = np.where(dealias, u_new, 0.0 + 0.0j)
        u_new_hats.append(u_new)

    return u_new_hats


# ---------------------------------------------------------------------------
# Public API: spin2()
# ---------------------------------------------------------------------------


def spin2(
    op: Union[SpinOp2, str],
    N: Optional[int] = None,
    dt: Optional[float] = None,
    *,
    dealias: bool = True,
    M: int = 32,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, object]:
    """Solve a 2D periodic semilinear PDE via ETDRK4.

    Given a :class:`SpinOp2` or a PDE name string, integrates

        u_t = L[u] + N[u]

    from ``t0`` to ``tf`` using the exponential time-differencing Runge-Kutta
    scheme of order 4 (ETDRK4) with 2D Fourier spectral discretization.

    Parameters
    ----------
    op : SpinOp2 or str
        The 2D PDE operator.  Pass a string (e.g. ``'GL'``) to use a built-in
        example, or construct a :class:`SpinOp2` manually.
    N : int, optional
        Number of Fourier modes per direction (grid is N x N).
        Defaults to the built-in value for named PDEs or 64 otherwise.
    dt : float, optional
        Time-step.  Defaults to the built-in value for named PDEs or 1e-3.
    dealias : bool, default True
        Apply 2D 2/3-rule dealiasing at each step.
    M : int, default 32
        Number of contour points for computing phi-functions.
    verbose : bool, default False
        If True, print progress every 10% of the integration.

    Returns
    -------
    xx : np.ndarray, shape (N, N), float64
        2D grid of x-coordinates (rows).
    yy : np.ndarray, shape (N, N), float64
        2D grid of y-coordinates (cols).
    t : float
        Final time reached.
    u_final : np.ndarray, shape (N, N) or list of (N, N) arrays
        Solution at the final time in physical (value) space.
        For scalar PDEs: a single (N, N) array.
        For multi-component PDEs: a list of n_vars (N, N) arrays.

    Examples
    --------
    Ginzburg-Landau (complex scalar):

    >>> xx, yy, t, u = spin2('GL', N=64, dt=5e-3)

    Allen-Cahn 2D (real scalar):

    >>> xx, yy, t, u = spin2('AC2', N=128, dt=1e-2)

    Notes
    -----
    The ETDRK4 scheme reads (Cox & Matthews 2002, Kassam & Trefethen 2005):

        a = exp(dt/2 * L) * u_n
        b = exp(dt/2 * L) * u_n + psi_1(dt/2*L) * N(a)
        c = exp(dt   * L) * u_n + 2*psi_1(dt/2*L) * N(b)
        u_{n+1} = exp(dt*L) * u_n + dt*(2*phi_2-4*phi_3)*(N(a)+N(b))
                                   + dt*(-phi_2+4*phi_3)*N(c)

    In 2D, the 1D eigenvalue array is replaced by an N x N matrix (or its
    flattened N^2-length version) and the 1D FFT/IFFT by 2D FFT/IFFT.

    References
    ----------
    .. [1] S. M. Cox and P. C. Matthews, "Exponential time differencing for
       stiff systems", J. Comput. Phys. 176 (2002), 430-455.
    .. [2] A.-K. Kassam and L. N. Trefethen, "Fourth-order time-stepping for
       stiff PDEs", SIAM J. Sci. Comput. 26 (2005), 1214-1233.
    .. [3] H. Montanelli and N. Bootland, "Solving periodic semilinear stiff
       PDEs in 1D/2D/3D with exponential integrators", submitted (2017).

    Provenance
    ----------
    MATLAB source : @spinoperator/solvepde.m, @expinteg/computeCoeffs.m,
                    @expinteg/oneStep.m, @spinop2/spinop2.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and The
        Chebfun Developers.

    See Also
    --------
    SpinOp2
    """
    # ---- Parse operator ----
    pde_name = None
    if isinstance(op, str):
        pde_name = op
        op = SpinOp2.from_name(op)

    # ---- Grid size and time-step ----
    if N is None:
        N = op.default_N(pde_name)
    if dt is None:
        dt = op.default_dt(pde_name)

    ax, bx, ay, by = op.domain
    t0, tf = op.tspan

    # ---- Spatial grids ----
    x = np.linspace(float(ax), float(bx), N, endpoint=False)
    y = np.linspace(float(ay), float(by), N, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")  # (N, N), row=x, col=y

    # ---- Number of steps ----
    nsteps = int(round((float(tf) - float(t0)) / dt))
    if nsteps <= 0:
        raise ValueError(
            f"tspan [{t0}, {tf}] with dt={dt} gives no steps. "
            "Check tspan and dt."
        )

    report_every = max(1, nsteps // 10)
    t = float(t0)

    # ---- Dealiasing mask ----
    dmask = _dealias_mask_2d(N) if dealias else None

    # ================================================================
    # Scalar PDE
    # ================================================================
    if op.n_vars == 1:
        # ---- Linear operator eigenvalues (N x N -> flat N^2) ----
        L_mat = build_linear_eigenvalues_2d(op.lin_coeffs, N, (ax, bx, ay, by))
        L_flat = L_mat.ravel()

        # Determine if the operator is purely real (diffusive)
        _is_real_op = np.allclose(np.imag(L_flat), 0.0)

        # ---- ETDRK4 coefficients ----
        coeffs = _compute_etdrk4_coeffs_2d(dt, L_flat, is_real=_is_real_op, M=M)

        # ---- Initial condition ----
        u0_vals = np.asarray(op.u0(xx, yy), dtype=complex)
        u_hat = np.fft.fft2(u0_vals)
        if dealias and dmask is not None:
            u_hat = np.where(dmask, u_hat, 0.0 + 0.0j)

        nonlin_vals_fn = op.nonlin_vals

        # ---- Time-stepping loop ----
        for step in range(nsteps):
            u_hat = _etdrk4_step_2d_scalar(
                u_hat, nonlin_vals_fn, coeffs, dmask, N
            )
            if np.any(np.isnan(u_hat)):
                raise RuntimeError(
                    f"Solution blew up at step {step}, t={t + dt:.6g}. "
                    "Try a smaller time-step."
                )
            t += dt
            if verbose and (step + 1) % report_every == 0:
                pct = 100.0 * (step + 1) / nsteps
                print(f"  spin2: {pct:.0f}%  t={t:.4g}")

        # ---- Back to physical space ----
        u_final = np.fft.ifft2(u_hat)
        if op.is_real:
            u_final = np.real(u_final)

        return xx, yy, t, u_final

    # ================================================================
    # Multi-component PDE
    # ================================================================
    n_vars = op.n_vars

    # ---- Linear operator eigenvalues per component ----
    L_mats = [
        build_linear_eigenvalues_2d(op.lin_coeffs[i], N, (ax, bx, ay, by))
        for i in range(n_vars)
    ]
    L_flats = [L.ravel() for L in L_mats]

    # Determine real/complex per component
    _real_ops = [np.allclose(np.imag(lf), 0.0) for lf in L_flats]

    # ---- ETDRK4 coefficients per component ----
    coeffs_list = [
        _compute_etdrk4_coeffs_2d(dt, L_flats[i], is_real=_real_ops[i], M=M)
        for i in range(n_vars)
    ]

    # ---- Initial conditions ----
    u_hats = []
    for i in range(n_vars):
        u0_vals = np.asarray(op.u0[i](xx, yy), dtype=complex)
        u_hat = np.fft.fft2(u0_vals)
        if dealias and dmask is not None:
            u_hat = np.where(dmask, u_hat, 0.0 + 0.0j)
        u_hats.append(u_hat)

    # ---- Time-stepping loop ----
    for step in range(nsteps):
        u_hats = _etdrk4_step_2d_multi(
            u_hats, op.nonlin_vals, coeffs_list, dmask, N
        )
        if any(np.any(np.isnan(h)) for h in u_hats):
            raise RuntimeError(
                f"Solution blew up at step {step}, t={t + dt:.6g}. "
                "Try a smaller time-step."
            )
        t += dt
        if verbose and (step + 1) % report_every == 0:
            pct = 100.0 * (step + 1) / nsteps
            print(f"  spin2: {pct:.0f}%  t={t:.4g}")

    # ---- Back to physical space ----
    u_finals = []
    for i in range(n_vars):
        u_val = np.fft.ifft2(u_hats[i])
        if op.is_real:
            u_val = np.real(u_val)
        u_finals.append(u_val)

    return xx, yy, t, u_finals
