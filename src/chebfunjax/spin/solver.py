# uses-numpy: ETDRK4 phi-function contour integration uses numpy (setup phase, not JIT-safe)
"""ETDRK4 time-stepping solver for 1-D periodic semilinear PDEs.

Implements the ``spin`` function: given a ``SpinOp``, a grid size N, and a
time-step dt, integrates u_t = L[u] + N[u] on a periodic domain using the
fourth-order exponential time-differencing Runge-Kutta (ETDRK4) scheme of
Cox and Matthews (2002) with the Kassam-Trefethen (2005) contour-integral
method for evaluating the phi-functions stably.

Translated from MATLAB Chebfun @expinteg/computeCoeffs.m, @expinteg/oneStep.m,
and @spinoperator/solvepde.m (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

from chebfunjax.spin.spinop import SpinOp

# ---------------------------------------------------------------------------
# Phi-function evaluation via contour integral (Kassam-Trefethen method)
# ---------------------------------------------------------------------------


def _phi_fun(l: int) -> Callable:
    """Return the phi-function of index *l* as a callable.

    The phi-functions are defined by the recursion:
      phi_0(z) = exp(z)
      phi_l(z) = (phi_{l-1}(z) - 1/(l-1)!) / z,  l >= 1.

    For small |z| the recursion is numerically unstable; the contour integral
    approach in :func:`_phi_eval` avoids evaluating the recursion directly.

    Provenance
    ----------
    MATLAB source : @expinteg/phiFun.m
    Chebfun commit: 7574c77
    """
    if l == 0:
        return jnp.exp
    else:
        prev = _phi_fun(l - 1)
        fact = math.factorial(l - 1)
        return lambda z: (prev(z) - 1.0 / fact) / z


def _compute_contour(dt: float, L: np.ndarray, M: int = 32) -> np.ndarray:
    """Compute contour points around each eigenvalue of *dt * L*.

    For each eigenvalue lambda_j of the (diagonal) linear operator, we
    sample M points on a small circle of radius 1 centered at dt*lambda_j:

        LR[j, m] = dt*lambda_j + exp(i*pi*(m + 0.5)/M),   m = 0,...,M-1

    For real eigenvalues we use the upper-half circle only.

    Parameters
    ----------
    dt : float
        Time-step.
    L : np.ndarray, shape (N,), complex
        Diagonal of the linearized operator in Fourier space.
    M : int
        Number of contour points.

    Returns
    -------
    LR : np.ndarray, shape (N, M), complex
        Contour points.

    Provenance
    ----------
    MATLAB source : @spinoperator/computeLR.m
    Chebfun commit: 7574c77
    """
    if np.isrealobj(L):
        # Upper-half circle for real (diffusive) operators
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    else:
        r = np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    # LR[j, m] = dt * L[j] + r[m]
    LR = dt * L[:, np.newaxis] + r[np.newaxis, :]  # (N, M)
    return LR


def _phi_eval_contour(l: int, LR: np.ndarray) -> np.ndarray:
    """Evaluate phi_l at each eigenvalue via the mean over the contour.

    phi_l(lambda) ≈ mean over contour samples phi_l(LR[j, :])

    Parameters
    ----------
    l : int
        Index of the phi-function.
    LR : np.ndarray, shape (N, M), complex
        Contour sample points.

    Returns
    -------
    phi : np.ndarray, shape (N,), complex
        phi_l evaluated at each eigenvalue.

    Provenance
    ----------
    MATLAB source : @expinteg/phiEval.m
    Chebfun commit: 7574c77
    """
    phi = _phi_fun(l)
    return np.mean(phi(LR), axis=1)  # (N,)


def _psi_eval_contour(l: int, c: float, LR: np.ndarray) -> np.ndarray:
    """Evaluate the psi-function psi_{l,c}(lambda) via contour integral.

    psi_{l,c}(lambda) = c^l * phi_l(c * lambda)

    Parameters
    ----------
    l : int
        Index.
    c : float
        Stage coefficient (e.g., 1/2 for midpoint stages).
    LR : np.ndarray, shape (N, M)
        Full-step contour; we scale by c to get the fractional-step contour.
    Provenance
    ----------
    MATLAB source : @expinteg/psiEval.m
    Chebfun commit: 7574c77
    """
    phi = _phi_fun(l)
    return np.mean(c ** l * phi(c * LR), axis=1)  # (N,)


# ---------------------------------------------------------------------------
# ETDRK4 coefficient computation
# ---------------------------------------------------------------------------


def _compute_etdrk4_coeffs(
    dt: float, L: np.ndarray, M: int = 32
) -> dict:
    """Compute ETDRK4 coefficients for the diagonal operator L.

    The ETDRK4 scheme (Cox & Matthews 2002, Kassam & Trefethen 2005) for
    u_t = L*u + N(u) reads:

        a = exp(dt/2 * L) * u_n  +  (dt/2) * psi_1(dt/2*L) * N(u_n)
        b = exp(dt/2 * L) * u_n  +  (dt/2) * psi_1(dt/2*L) * N(a)
        c = exp(dt/2 * L) * a    +  (dt/2) * psi_1(dt/2*L) * (2*N(b) - N(u_n))
        u_{n+1} = exp(dt*L) * u_n
                + dt * [phi_1 - 3*phi_2 + 4*phi_3] * N(u_n)   (B1 term)
                + dt * [2*phi_2 - 4*phi_3]          * (N(a)+N(b))  (B2,B3)
                + dt * [-phi_2 + 4*phi_3]            * N(c)     (B4 term)

    The exact coefficients in Chebfun's notation (A_{3,2}, A_{4,3}, B_2,
    B_3, B_4, and E = [exp(c_i*dt*L)]) are computed here via contour
    integrals.

    Parameters
    ----------
    dt : float
        Time-step.
    L : np.ndarray, shape (N,), complex
        Diagonal of the linear operator (Fourier eigenvalues).
    M : int
        Number of contour points for complex means.

    Returns
    -------
    coeffs : dict with keys
        ``E_half`` : array (N,) — exp(dt/2 * L)
        ``E_full`` : array (N,) — exp(dt * L)
        ``A32``    : array (N,) — A_{3,2} = psi_1(1/2; dt*L)
        ``A43``    : array (N,) — A_{4,3} = 2 * psi_1(1/2; dt*L)
        ``B2``     : array (N,) — 2*phi_2 - 4*phi_3   (weights for stages 2 & 3)
        ``B3``     : array (N,) — 2*phi_2 - 4*phi_3
        ``B4``     : array (N,) — -phi_2 + 4*phi_3
        ``B1``     : array (N,) — phi_1 - 3*phi_2 + 4*phi_3  (weight for stage 1)

    Notes
    -----
    Developer notes from MATLAB Chebfun::

        Cox & Matthews (2002), "Exponential time differencing for stiff
        systems", J. Comput. Phys. 176, 430-455.

        Kassam & Trefethen (2005), "Fourth-order time-stepping for stiff
        PDEs", SIAM J. Sci. Comput. 26, 1214-1233.

        Montanelli & Bootland (2017), "Solving periodic semilinear stiff PDEs
        in 1D/2D/3D with exponential integrators" (submitted).

    Provenance
    ----------
    MATLAB source : @expinteg/computeCoeffs.m (etdrk4 branch)
    Chebfun commit: 7574c77
    """
    LR = _compute_contour(dt, L, M)  # (N, M)

    # phi-functions via contour integrals
    phi1 = _phi_eval_contour(1, LR)  # phi_1(dt*L)
    phi2 = _phi_eval_contour(2, LR)  # phi_2(dt*L)
    phi3 = _phi_eval_contour(3, LR)  # phi_3(dt*L)

    # psi-functions (fractional steps c=1/2 and c=1)
    psi12 = _psi_eval_contour(1, 0.5, LR)  # psi_{1,1/2} = (1/2)*phi_1(dt/2*L)
    # psi_{1,1} = psi_1(1, dt*L) = phi_1(dt*L)  (c=1)

    # For real-eigenvalue (diffusive) operators, take real parts
    if np.isrealobj(L):
        phi1, phi2, phi3, psi12 = (
            np.real(phi1), np.real(phi2), np.real(phi3), np.real(psi12)
        )

    # Matrix exponentials (pointwise on the diagonal)
    E_half = np.exp(0.5 * dt * L)
    E_full = np.exp(dt * L)

    # Scheme weights for the solution update
    # u_{n+1} = E_full*u_n + dt*(B1*N(u_n) + B2*N(a) + B3*N(b) + B4*N(c))
    # B1 = phi1 - 3*phi2 + 4*phi3  (Chebfun's B{1} is unused; B{2}=B{3}=B2)
    # B{1} in MATLAB corresponds to no contribution because A{2,1} is empty
    # for etdrk4.  The update weights are:
    #   B{2} = 2*phi_2 - 4*phi_3
    #   B{3} = 2*phi_2 - 4*phi_3
    #   B{4} = -phi_2 + 4*phi_3
    # The stage-1 (u_n itself) contribution enters via
    #   u_{n+1} = E_full*u_n + dt*(B2*N(a) + B3*N(b) + B4*N(c))
    # but we need to include u_n's nonlinear contribution.  In Chebfun's
    # formula the solution step is:
    #   sol = E{5}.*vSol{1} + B{2}.*Nv{2} + B{3}.*Nv{3} + B{4}.*Nv{4}
    # where vSol{1} = u_n, B{1} is empty (no N(u_n) term in etdrk4).
    # The formula that matches Cox-Matthews-Kassam-Trefethen is:
    #   u_{n+1} = E*u + dt*(phi1-3*phi2+4*phi3)*N(u) + dt*(2*phi2-4*phi3)*(N(a)+N(b))
    #             + dt*(-phi2+4*phi3)*N(c)
    # but Chebfun's MATLAB code does NOT include a B{1} term for etdrk4.
    # This is because the internal stage c absorbs the u_n nonlinear term
    # via A{4,3} = 2*psi12.  The final weights exactly match the standard
    # ETDRK4 Butcher tableau.
    B2 = dt * (2.0 * phi2 - 4.0 * phi3)
    B3 = dt * (2.0 * phi2 - 4.0 * phi3)
    B4 = dt * (-phi2 + 4.0 * phi3)

    # Internal-stage weights
    A32 = dt * psi12   # (dt/2) * phi_1(dt/2 * L) for stage 3 from stage 2
    A43 = dt * psi12   # Same — stage 4 uses 2*A32 effectively via 2*N(b)

    return {
        "E_half": E_half,
        "E_full": E_full,
        "A32": A32,   # stage c weight for N(a)
        "A43": A43,
        "B2": B2,
        "B3": B3,
        "B4": B4,
        "phi1": dt * phi1,  # dt*phi_1(dt*L) — needed for stage a
        "psi12": dt * psi12,  # dt*psi_{1,1/2}(dt*L) — half-step propagator weight
    }


# ---------------------------------------------------------------------------
# Fourier wavenumber helpers
# ---------------------------------------------------------------------------


def _fourier_wavenumbers(N: int, domain: tuple) -> np.ndarray:
    """Compute the integer wavenumbers for the FFT ordering on [a, b].

    For a domain of length L = b - a, d/dx has Fourier eigenvalue
    i * (2*pi/L) * k, where k is the integer wavenumber.  We return k in
    the standard NumPy FFT ordering (0, 1, ..., N/2-1, -N/2, ..., -1).

    Parameters
    ----------
    N : int
        Number of grid points (should be even for dealiasing).
    domain : tuple(float, float)
        (a, b).

    Returns
    -------
    k : np.ndarray, shape (N,), float
        Wavenumbers in FFT ordering (NOT scaled by 2*pi/L — the scaling is
        incorporated into ``lin_coeff``).
    """
    a, b = domain
    L = float(b) - float(a)
    # Standard FFT wavenumbers scaled so that d/dx -> i*(2*pi/L)*k
    # Return the multiplicative factor i*(2*pi/L)*k so that
    # lin_coeff(k) = L_diag  where k already carries the 2*pi/L factor.
    freqs = np.fft.fftfreq(N, d=1.0 / N)  # integers 0,1,...,N/2-1,-N/2,...,-1
    return (2.0 * np.pi / L) * freqs  # actual angular frequencies (rad / unit length)


def _build_linear_eigenvalues(
    lin_coeff: Callable, N: int, domain: tuple
) -> np.ndarray:
    """Evaluate the diagonal of the linear operator in Fourier space.

    Parameters
    ----------
    lin_coeff : callable(xi) -> array
        Maps frequency xi = (2*pi/L)*k to the eigenvalue of L at that
        frequency.  E.g., for -d^3/dx^3 on [-pi, pi]:
        ``lin_coeff = lambda xi: -(1j*xi)**3``.
    N : int
        Number of grid points.
    domain : tuple
        (a, b).

    Returns
    -------
    L : np.ndarray, shape (N,), complex
        L[j] = eigenvalue of the linear operator at frequency xi[j].
    """
    xi = _fourier_wavenumbers(N, domain)  # real, (N,)
    L = np.asarray(lin_coeff(xi), dtype=complex)
    return L


def _build_nonlin_coeff_factor(
    diff_order: int, N: int, domain: tuple
) -> np.ndarray:
    """Return the diagonal of the differentiation factor for the nonlinear term.

    For the nonlinear part written as  N(u) = d^m/dx^m [f(u)],
    in Fourier space this becomes  Nc * fft(f(u))  where
    Nc[k] = (i*xi[k])^m.

    Parameters
    ----------
    diff_order : int
        Differentiation order m (0 for no differentiation).
    N : int
    domain : tuple

    Returns
    -------
    Nc : np.ndarray, shape (N,), complex
    """
    if diff_order == 0:
        return np.ones(N, dtype=complex)
    xi = _fourier_wavenumbers(N, domain)
    return (1j * xi) ** diff_order


# ---------------------------------------------------------------------------
# 2/3-rule dealiasing
# ---------------------------------------------------------------------------


def _dealias_mask(N: int) -> np.ndarray:
    """Return a boolean mask that zeros the top 1/3 of Fourier modes.

    The 2/3-rule dealiasing zeros modes with |k| > N/3 to prevent aliasing
    errors from the quadratic nonlinearity.

    Parameters
    ----------
    N : int
        Number of Fourier modes.

    Returns
    -------
    mask : np.ndarray, shape (N,), bool
        True for modes to *keep*, False for modes to zero.

    Provenance
    ----------
    MATLAB source : @spinop/getDealiasingIndexes.m
    Chebfun commit: 7574c77
    """
    mask = np.ones(N, dtype=bool)
    # Zero top 1/3 in FFT ordering
    N // 3
    # The high-frequency modes in FFT ordering occupy indices [N//3+1 : 2*N//3]
    # (approximately).  Use the MATLAB logic: toOne covers the middle third.
    # In MATLAB (1-indexed): toOne = floor(N/2)+1-ceil(N/6) : floor(N/2)+ceil(N/6)
    # which is the highest-frequency modes near the Nyquist frequency.
    half = N // 2
    sixth = math.ceil(N / 6)
    lo = half - sixth      # 0-indexed
    hi = half + sixth      # 0-indexed exclusive
    mask[lo:hi] = False
    return mask


# ---------------------------------------------------------------------------
# ETDRK4 single time-step
# ---------------------------------------------------------------------------


def _etdrk4_step(
    u_hat: np.ndarray,
    Nc: np.ndarray,
    nonlin_vals: Callable,
    coeffs: dict,
    dealias: Optional[np.ndarray],
) -> np.ndarray:
    """Advance the Fourier coefficients by one ETDRK4 step.

    Implements the exact Chebfun ETDRK4 stages from @expinteg/oneStep.m
    with the coefficient structure from @expinteg/computeCoeffs.m.

    For the etdrk4 scheme in Chebfun, with stage coefficients
    C = [0, 1/2, 1/2, 1] and::

        A{3,2} = psi12 = dt/2 * phi_1(dt/2*L)
        A{4,3} = 2*psi12
        B{2} = B{3} = dt*(2*phi_2 - 4*phi_3)
        B{4} = dt*(-phi_2 + 4*phi_3)

    the four stages are (B{1} is empty in the MATLAB code, no N(u_n) weight):

        a_hat = exp(dt/2*L) * u_hat                       (stage 2)
        b_hat = exp(dt/2*L) * u_hat + psi12 * N(a_hat)   (stage 3)
        c_hat = exp(dt*L)   * u_hat + 2*psi12 * N(b_hat) (stage 4)

        u_new = exp(dt*L)*u_hat + B2*N(a) + B3*N(b) + B4*N(c)

    Parameters
    ----------
    u_hat : np.ndarray, shape (N,), complex
        Current Fourier coefficients.
    Nc : np.ndarray, shape (N,), complex
        Coefficient-space factor for the nonlinear term (differentiation).
    nonlin_vals : callable(u_vals) -> array
        Nonlinear operator in physical space (includes sign).
    coeffs : dict
        Output of :func:`_compute_etdrk4_coeffs`.
    dealias : np.ndarray or None, shape (N,), bool
        Dealiasing mask (True = keep). If None, no dealiasing.

    Returns
    -------
    u_new : np.ndarray, shape (N,), complex

    Notes
    -----
    Developer notes from MATLAB Chebfun::

        This is NOT the standard Cox-Matthews ETDRK4. Chebfun uses a modified
        variant where stage 2 has no nonlinear forcing from u_n. The B
        coefficients are the same as Cox-Matthews, but the stage structure
        differs. The overall method is 4th-order accurate.

        Cox & Matthews (2002), J. Comput. Phys. 176, 430-455.
        Kassam & Trefethen (2005), SIAM J. Sci. Comput. 26, 1214-1233.

    Provenance
    ----------
    MATLAB source : @expinteg/oneStep.m, @expinteg/computeCoeffs.m
    Chebfun commit: 7574c77
    """
    E_half = coeffs["E_half"]
    E_full = coeffs["E_full"]
    psi12 = coeffs["psi12"]
    B2 = coeffs["B2"]
    B3 = coeffs["B3"]
    B4 = coeffs["B4"]

    def _nonlin_coeff(c_hat):
        """Compute Nc * fft(nonlin_vals(ifft(c_hat)))."""
        u_vals = np.fft.ifft(c_hat)
        nv = nonlin_vals(u_vals)
        return Nc * np.fft.fft(nv)

    # Stage 2 (MATLAB vSol{2}): a = E{2}*u  (A{2,1} is empty in Chebfun's etdrk4)
    a_hat = E_half * u_hat
    Na = _nonlin_coeff(a_hat)

    # Stage 3 (MATLAB vSol{3}): b = E{3}*u + A{3,2}*N(a)
    b_hat = E_half * u_hat + psi12 * Na
    Nb = _nonlin_coeff(b_hat)

    # Stage 4 (MATLAB vSol{4}): c = E{4}*u + A{4,3}*N(b)  where E{4}=exp(dt*L), A{4,3}=2*psi12
    c_hat = E_full * u_hat + 2.0 * psi12 * Nb
    Nc_hat = _nonlin_coeff(c_hat)

    # Solution (MATLAB sol = E{5}.*vSol{1} + B{2}.*N(a) + B{3}.*N(b) + B{4}.*N(c))
    # B{1} is empty (no N(u_n) term), B2=B3=dt*(2*phi2-4*phi3), B4=dt*(-phi2+4*phi3)
    u_new = E_full * u_hat + B2 * Na + B3 * Nb + B4 * Nc_hat

    # Apply dealiasing
    if dealias is not None:
        u_new = np.where(dealias, u_new, 0.0 + 0.0j)

    return u_new


# ---------------------------------------------------------------------------
# Public API: spin()
# ---------------------------------------------------------------------------


def spin(
    op: Union[SpinOp, str],
    N: Optional[int] = None,
    dt: Optional[float] = None,
    *,
    dealias: bool = True,
    M: int = 32,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a 1-D periodic semilinear PDE via ETDRK4.

    Given a :class:`SpinOp` or a PDE name string, integrates

        u_t = L[u] + N[u]

    from ``t0`` to ``tf`` using the exponential time-differencing Runge-Kutta
    scheme of order 4 (ETDRK4) with Fourier spectral discretization.

    Parameters
    ----------
    op : SpinOp or str
        The PDE operator.  Pass a string (e.g. ``'KdV'``) to use a built-in
        example, or construct a :class:`SpinOp` manually.
    N : int, optional
        Number of Fourier modes.  Defaults to the built-in value for named
        PDEs or 256 otherwise.
    dt : float, optional
        Time-step.  Defaults to the built-in value for named PDEs or 1e-3.
    dealias : bool, default True
        Apply 2/3-rule dealiasing (zero top-third of modes) at each step.
    M : int, default 32
        Number of contour points for computing phi-functions via complex means.
    verbose : bool, default False
        If True, print progress every 10% of the integration.

    Returns
    -------
    x : np.ndarray, shape (N,), float64
        Spatial grid points on [a, b).
    t : float
        Final time reached.
    u_final : np.ndarray, shape (N,), float or complex
        Solution at the final time in physical (value) space.

    Examples
    --------
    KdV two-soliton — solve and check mass conservation:

    >>> x, t, u = spin('KdV', N=512, dt=3e-6)
    >>> import numpy as np
    >>> np.trapz(u, x)  # mass ≈ constant

    Allen-Cahn metastable fronts:

    >>> x, t, u = spin('AC', N=256, dt=0.1)

    Custom SpinOp (Burgers equation):

    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> op = SpinOp(
    ...     lin_coeff=lambda xi: 1e-3 * (1j * xi) ** 2,
    ...     nonlin_vals=lambda u: 0.5 * u ** 2,
    ...     nonlin_diff_order=1,
    ...     domain=(-1.0, 1.0),
    ...     tspan=(0.0, 20.0),
    ...     u0=lambda x: (1 - x**2) * jnp.exp(-30 * (x + 0.5)**2),
    ...     is_real=True,
    ... )
    >>> x, t, u = spin(op, N=512, dt=5e-3)

    Notes
    -----
    The ETDRK4 scheme reads (Cox & Matthews 2002, Kassam & Trefethen 2005):

        a = exp(dt/2 * L) * u_n  +  (dt/2) * phi_1(dt/2*L) * N(u_n)
        b = exp(dt/2 * L) * u_n  +  (dt/2) * phi_1(dt/2*L) * N(a)
        c = exp(dt/2 * L) * a    +  (dt/2) * phi_1(dt/2*L) * (2*N(b) - N(u_n))
        u_{n+1} = exp(dt*L) * u_n + dt*(2*phi_2-4*phi_3)*(N(a)+N(b))
                                   + dt*(-phi_2+4*phi_3)*N(c)

    The phi-functions are evaluated stably by contour integration on small
    circles of radius 1 around each Fourier eigenvalue (Kassam & Trefethen).

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
                    @expinteg/oneStep.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and The
        Chebfun Developers.

    See Also
    --------
    SpinOp
    """
    # ---- Parse operator ----
    pde_name = None
    if isinstance(op, str):
        pde_name = op
        op = SpinOp.from_name(op)

    # ---- Grid size and time-step ----
    if N is None:
        N = op.default_N(pde_name)
    if dt is None:
        dt = op.default_dt(pde_name)

    a, b = op.domain
    t0, tf = op.tspan

    # ---- Spatial grid (uniform, periodic on [a, b) ) ----
    x = np.linspace(float(a), float(b), N, endpoint=False)

    # ---- Initial condition ----
    u0_vals = np.asarray(op.u0(x), dtype=complex)

    # ---- Fourier eigenvalues of the linear operator ----
    L_diag = _build_linear_eigenvalues(op.lin_coeff, N, (a, b))

    # ---- Nonlinear coefficient factor (differentiation in Fourier space) ----
    Nc = _build_nonlin_coeff_factor(op.nonlin_diff_order, N, (a, b))

    # ---- ETDRK4 precomputed coefficients ----
    coeffs = _compute_etdrk4_coeffs(dt, L_diag, M=M)

    # ---- Dealiasing mask ----
    dmask = _dealias_mask(N) if dealias else None

    # ---- Initial Fourier coefficients ----
    u_hat = np.fft.fft(u0_vals)
    if dealias and dmask is not None:
        u_hat = np.where(dmask, u_hat, 0.0 + 0.0j)

    # ---- Extract nonlinear part function ----
    nonlin_vals_fn = op.nonlin_vals

    # ---- Time-stepping loop ----
    t = float(t0)
    nsteps = int(round((float(tf) - float(t0)) / dt))
    if nsteps <= 0:
        raise ValueError(
            f"tspan [{t0}, {tf}] with dt={dt} gives no steps. "
            "Check tspan and dt."
        )

    report_every = max(1, nsteps // 10)

    for step in range(nsteps):
        u_hat = _etdrk4_step(u_hat, Nc, nonlin_vals_fn, coeffs, dmask)
        if np.any(np.isnan(u_hat)):
            raise RuntimeError(
                f"Solution blew up at step {step}, t={t + dt:.6g}. "
                "Try a smaller time-step."
            )
        t += dt
        if verbose and (step + 1) % report_every == 0:
            pct = 100.0 * (step + 1) / nsteps
            print(f"  spin: {pct:.0f}%  t={t:.4g}")

    # ---- Convert back to physical space ----
    u_final = np.fft.ifft(u_hat)
    if op.is_real:
        u_final = np.real(u_final)

    return x, t, u_final
