"""IMEX (implicit-explicit) time-stepping schemes for stiff semi-linear PDEs.

Implements two IMEX schemes for problems of the form

    u_t = L u + N(u),

where L is a stiff linear operator (treated implicitly) and N is a non-stiff
nonlinear term (treated explicitly).  Both schemes work in Fourier space on
a periodic domain: L is diagonal with eigenvalues ``L_diag``, and N is
evaluated in physical space then transformed to Fourier space.

Available schemes
-----------------
``imex_euler``
    First-order IMEX-Euler (backward Euler on L, forward Euler on N).
    Stability: A-stable for the linear part; explicit for nonlinear.

``imex_sbdf2``
    Second-order implicit-explicit BDF2 (also called CNAB or SBDF2).
    Two-step method: requires one startup step via IMEX-Euler.
    Stability: A(alpha)-stable for alpha ≈ 90°; superior to Euler for
    diffusion-dominated problems.

Notes
-----
For comparison, MATLAB Chebfun's @imex class supports lirk4 (4th-order IMEX
Runge-Kutta) and imexbdf4 (4th-order BDF).  This module provides the two
fundamental lower-order building blocks that are used as starters and for
moderate-accuracy problems.

Provenance
----------
MATLAB source : @imex/imex.m, @imex/computeCoeffs.m, @imex/oneStep.m,
                @imex/startMultistep.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

References
----------
.. [1] U. Ascher, S. Ruuth, B. Wetton, "Implicit-explicit methods for
   time-dependent PDEs", SIAM J. Numer. Anal. 32 (1995), 797-823.
.. [2] G. Akrivis, M. Crouzeix, C. Makridakis, "Implicit-explicit multistep
   finite element methods for nonlinear parabolic problems", Math. Comp.
   67 (1998), 457-477.
.. [3] H. Montanelli, Y. Nakatsukasa, "Fourth-order time-stepping for stiff
   PDEs on the sphere", SIAM J. Sci. Comput. 40 (2018).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

__all__ = ["imex_euler", "imex_sbdf2"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nonlin_eval(
    u_hat: np.ndarray,
    Nc: np.ndarray,
    nonlin_vals: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Evaluate the nonlinear term in Fourier space.

    Computes ``Nc * fft(nonlin_vals(ifft(u_hat)))``.

    Parameters
    ----------
    u_hat : np.ndarray, shape (N,), complex
        Current Fourier coefficients.
    Nc : np.ndarray, shape (N,), complex
        Coefficient-space differentiation factor for the nonlinear term.
        ``Nc[k] = (i*xi[k])^m`` for a nonlinearity that needs *m* derivatives
        in Fourier space; ``Nc = ones(N)`` for no differentiation.
    nonlin_vals : callable
        Physical-space nonlinear operator: maps values array to values array.

    Returns
    -------
    Nu_hat : np.ndarray, shape (N,), complex
        Fourier coefficients of the evaluated nonlinear term.

    Provenance
    ----------
    MATLAB source : @imex/oneStep.m (inner coeffs2vals / vals2coeffs pattern)
    Chebfun commit: 7574c77
    """
    u_vals = np.fft.ifft(u_hat)
    Nu_vals = nonlin_vals(u_vals)
    return Nc * np.fft.fft(Nu_vals)


# ---------------------------------------------------------------------------
# IMEX-Euler  (1st order)
# ---------------------------------------------------------------------------


def imex_euler(
    u0_hat: np.ndarray,
    L_diag: np.ndarray,
    nonlin_vals: Callable[[np.ndarray], np.ndarray],
    Nc: np.ndarray,
    dt: float,
    nsteps: int,
    *,
    dealias: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """Integrate u_t = L u + N(u) using first-order IMEX-Euler.

    The update formula is:

        u^{n+1} = (I - dt * L)^{-1} (u^n + dt * N(u^n))

    In Fourier space (L diagonal with eigenvalues ``L_diag``):

        u_hat^{n+1}[k] = (u_hat^n[k] + dt * Nu_hat^n[k]) / (1 - dt * L_diag[k])

    Parameters
    ----------
    u0_hat : np.ndarray, shape (N,), complex
        Initial Fourier coefficients.
    L_diag : np.ndarray, shape (N,), complex
        Diagonal of the linear operator in Fourier space.
    nonlin_vals : callable
        Physical-space nonlinear function: ``f(u_vals) -> Nu_vals``.
    Nc : np.ndarray, shape (N,), complex
        Coefficient-space factor for the nonlinear term.  Use
        ``np.ones(N, dtype=complex)`` for no differentiation.
    dt : float
        Time-step.
    nsteps : int
        Number of time-steps to take.
    dealias : np.ndarray or None, shape (N,), bool
        Dealiasing mask (True = keep mode).  Applied after each step.
        ``None`` means no dealiasing.
    verbose : bool, default False
        Print progress every 10 % of the integration.

    Returns
    -------
    u_hat : np.ndarray, shape (N,), complex
        Fourier coefficients at the final time ``t0 + nsteps * dt``.
    t_final : float
        Final time reached (``nsteps * dt``, measured from 0).

    Notes
    -----
    Stability constraint: the scheme is A-stable for the *implicit* linear
    part, i.e. for ``real(L_diag[k]) <= 0`` every mode is stable regardless
    of ``dt``.  The *explicit* nonlinear part imposes the standard
    ``dt * |N'| < 1`` CFL condition.

    Provenance
    ----------
    MATLAB source : @imex/oneStep.m (imexeuler branch — not in the released
        Chebfun code but a standard building block)
    Chebfun commit: 7574c77

    See Also
    --------
    imex_sbdf2
    """
    # Pre-compute the implicit solve factor: 1 / (1 - dt * L)
    # (works element-wise since L is diagonal)
    impl_factor = 1.0 / (1.0 - dt * L_diag)  # shape (N,)

    u_hat = np.array(u0_hat, dtype=complex)
    t = 0.0
    report_every = max(1, nsteps // 10)

    for step in range(nsteps):
        # Explicit nonlinear evaluation at current state
        Nu_hat = _nonlin_eval(u_hat, Nc, nonlin_vals)

        # Implicit update: (I - dt*L)^{-1} * (u + dt*N(u))
        u_hat = impl_factor * (u_hat + dt * Nu_hat)

        # Dealiasing
        if dealias is not None:
            u_hat = np.where(dealias, u_hat, 0.0 + 0.0j)

        t += dt

        if verbose and (step + 1) % report_every == 0:
            pct = 100.0 * (step + 1) / nsteps
            print(f"  imex_euler: {pct:.0f}%  t={t:.4g}")

    return u_hat, t


# ---------------------------------------------------------------------------
# IMEX-SBDF2  (2nd order)
# ---------------------------------------------------------------------------


def imex_sbdf2(
    u0_hat: np.ndarray,
    L_diag: np.ndarray,
    nonlin_vals: Callable[[np.ndarray], np.ndarray],
    Nc: np.ndarray,
    dt: float,
    nsteps: int,
    *,
    dealias: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """Integrate u_t = L u + N(u) using second-order IMEX-SBDF2.

    SBDF2 (Second-order semi-implicit BDF, Ascher-Ruuth-Wetton 1995) is a
    two-step method.  The update formula is:

        (3/2) u^{n+1} - 2 u^n + (1/2) u^{n-1}
            = dt * L u^{n+1} + dt * (2 N(u^n) - N(u^{n-1}))

    Rearranged in Fourier space:

        u_hat^{n+1} = [(3/2 - dt*L)]^{-1}
                      * [2 u_hat^n - 1/2 u_hat^{n-1}
                         + dt * (2 Nu_hat^n - Nu_hat^{n-1})]

    The first step uses IMEX-Euler (first-order) as a startup step to obtain
    ``u^1`` from ``u^0``.

    Parameters
    ----------
    u0_hat : np.ndarray, shape (N,), complex
        Initial Fourier coefficients.
    L_diag : np.ndarray, shape (N,), complex
        Diagonal of the linear operator in Fourier space.
    nonlin_vals : callable
        Physical-space nonlinear function: ``f(u_vals) -> Nu_vals``.
    Nc : np.ndarray, shape (N,), complex
        Coefficient-space factor for the nonlinear term.
    dt : float
        Time-step.
    nsteps : int
        Number of time-steps to take (including the startup step).
    dealias : np.ndarray or None, shape (N,), bool
        Dealiasing mask.  ``None`` means no dealiasing.
    verbose : bool, default False
        Print progress every 10 % of the integration.

    Returns
    -------
    u_hat : np.ndarray, shape (N,), complex
        Fourier coefficients at the final time ``nsteps * dt``.
    t_final : float
        Final time reached.

    Notes
    -----
    SBDF2 is the "canonical" second-order IMEX method; it is sometimes
    written as CNAB (Crank-Nicolson Adams-Bashforth) in the literature,
    though the coefficients differ slightly.  The scheme in [1] uses the BDF2
    discretization for the implicit part.

    The implicit solve factor ``(3/2 - dt*L)^{-1}`` is pre-computed once
    (setup cost O(N), then O(N) per step), making each step very cheap.

    Provenance
    ----------
    MATLAB source : @imex/oneStep.m (imexbdf4 is the 4-step BDF version;
        SBDF2 is the 2-step variant from the same family)
    Chebfun commit: 7574c77

    References
    ----------
    .. [1] U. Ascher, S. Ruuth, B. Wetton, "Implicit-explicit methods for
       time-dependent PDEs", SIAM J. Numer. Anal. 32 (1995), 797-823.

    See Also
    --------
    imex_euler
    """
    if nsteps < 1:
        return np.array(u0_hat, dtype=complex), 0.0

    # Pre-compute implicit solve factor: 1 / (3/2 - dt*L)
    impl_factor = 1.0 / (1.5 - dt * L_diag)  # shape (N,)

    # --- Startup step (IMEX-Euler to get u^1 from u^0) ---
    u0 = np.array(u0_hat, dtype=complex)
    Nu0 = _nonlin_eval(u0, Nc, nonlin_vals)

    # IMEX-Euler: u^1 = (I - dt*L)^{-1} * (u^0 + dt*N(u^0))
    euler_factor = 1.0 / (1.0 - dt * L_diag)
    u1 = euler_factor * (u0 + dt * Nu0)
    if dealias is not None:
        u1 = np.where(dealias, u1, 0.0 + 0.0j)
    Nu1 = _nonlin_eval(u1, Nc, nonlin_vals)

    t = dt

    if nsteps == 1:
        return u1, t

    # Sliding window: (u_prev, Nu_prev, u_curr, Nu_curr)
    u_prev = u0
    Nu_prev = Nu0
    u_curr = u1
    Nu_curr = Nu1

    report_every = max(1, nsteps // 10)

    for step in range(1, nsteps):
        # SBDF2 update:
        # u_new = impl_factor * (2*u_curr - 0.5*u_prev
        #                        + dt*(2*Nu_curr - Nu_prev))
        rhs = (2.0 * u_curr - 0.5 * u_prev
               + dt * (2.0 * Nu_curr - Nu_prev))
        u_new = impl_factor * rhs

        if dealias is not None:
            u_new = np.where(dealias, u_new, 0.0 + 0.0j)

        Nu_new = _nonlin_eval(u_new, Nc, nonlin_vals)

        # Advance the window
        u_prev = u_curr
        Nu_prev = Nu_curr
        u_curr = u_new
        Nu_curr = Nu_new

        t += dt

        if verbose and (step + 1) % report_every == 0:
            pct = 100.0 * (step + 1) / nsteps
            print(f"  imex_sbdf2: {pct:.0f}%  t={t:.4g}")

    return u_curr, t
