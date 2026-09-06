# uses-numpy: exponential-integrator time-stepping is a one-shot host-side
# solve (contour-integral coefficients + FFT stepping loop); numpy keeps the
# per-step Python loop free of JIT dispatch overhead, as in spinop.py.
"""Generic exponential-integrator engine (MATLAB ``expinteg`` class).

Port of the scheme-agnostic machinery of Chebfun's exponential
integrators: the contour-integral evaluation of the phi/psi/gamma
functions, the scheme coefficient tables (``computeCoeffs`` +
``computeMissingCoeffs``), the generic Runge-Kutta/multistep step
(``oneStep``) and the multistep start-up (``startMultistep``).

States are arrays of shape ``(nVars*N, ...)``: the coefficient blocks of
the ``nVars`` unknowns stacked along the first axis, exactly MATLAB's
layout, so the same code serves the 1D (``(nVars*N,)``) and 2D
(``(nVars*N, N)``) drivers.  ``c2v``/``v2c`` are the per-block
coefficient/value transforms and ``Nv`` the value-space nonlinearity
acting on the stacked value array; ``Nc`` multiplies the resulting
coefficients (MATLAB's ``Nc``; the dealiasing mask here).

Provenance
----------
MATLAB source : @expinteg/{expinteg,computeCoeffs,oneStep,
    startMultistep,phiEval,psiEval,gammaEval,gammaFun,phiFun}.m,
    @spinoperator/computeLR.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and The
    Chebfun Developers.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

# scheme -> (order, stages, steps), MATLAB @expinteg/expinteg.m
SCHEMES = {
    "abnorsett4": (4, 1, 4),
    "abnorsett5": (5, 1, 5),
    "abnorsett6": (6, 1, 6),
    "etdrk2": (2, 2, 1),
    "etdrk4": (4, 4, 1),
    "exprk5s8": (5, 8, 1),
    "friedli": (4, 4, 1),
    "hochbruck-ostermann": (4, 5, 1),
    "krogstad": (4, 4, 1),
    "minchev": (4, 4, 1),
    "strehmel-weiner": (4, 4, 1),
    "ablawson4": (4, 1, 4),
    "lawson4": (4, 4, 1),
    "genlawson41": (4, 4, 1),
    "genlawson42": (4, 4, 2),
    "genlawson43": (4, 4, 3),
    "genlawson44": (4, 4, 4),
    "genlawson45": (4, 4, 5),
    "modgenlawson41": (4, 4, 1),
    "modgenlawson42": (4, 4, 2),
    "modgenlawson43": (4, 4, 3),
    "modgenlawson44": (4, 4, 4),
    "modgenlawson45": (4, 4, 5),
    "pec423": (4, 2, 3),
    "pecec433": (4, 3, 3),
    "pec524": (5, 2, 4),
    "pecec534": (5, 3, 4),
    "pec625": (6, 2, 5),
    "pecec635": (6, 3, 5),
    "pec726": (7, 2, 6),
    "pecec736": (7, 3, 6),
}


class Expinteg:
    """MATLAB ``expinteg(schemeName)``: order/stages/steps of a scheme."""

    def __init__(self, scheme: str):
        key = str(scheme).lower()
        if key not in SCHEMES:
            raise ValueError(
                "Unrecognized time-stepping scheme. See HELP/EXPINTEG and "
                "HELP/IMEX for a list of available schemes.")
        self.scheme = key
        self.order, self.stages, self.steps = SCHEMES[key]


# ---------------------------------------------------------------------------
# phi / psi / gamma functions on the contour
# ---------------------------------------------------------------------------

def phi_fun(l: int):
    """``phi_0 = exp``, ``phi_l(z) = (phi_{l-1}(z) - 1/(l-1)!) / z``."""
    if int(l) == 0:
        return np.exp
    f = phi_fun(int(l) - 1)
    c = 1.0 / math.factorial(int(l) - 1)
    return lambda z: (f(z) - c) / z


def gamma_fun(j: int, k: int):
    """MATLAB ``expinteg.gammaFun(j, k)`` (multistep start-up weights)."""
    if j == 0:
        return lambda z: (np.exp(k * z) - 1.0) / z

    parts = [(((-1.0) ** (m - 1)) / m, gamma_fun(j - m, k))
             for m in range(1, j + 1)]

    def g(z):
        out = 0.0 * z
        for cm, gm in parts:
            out = out + cm * gm(z)
        return out

    if j <= k:
        c = float(math.comb(k, j))
        return lambda z: (g(z) - c) / z
    return lambda z: g(z) / z


def compute_lr(dt: float, L: np.ndarray, M: int, is_real: bool) -> np.ndarray:
    """MATLAB ``computeLR``: ``dt*L(:)`` plus the contour points ``r``
    (upper half circle for a real operator, full circle otherwise)."""
    m = np.arange(1, M + 1)
    if is_real:
        r = np.exp(1j * np.pi * (m - 0.5) / M)
    else:
        r = np.exp(2j * np.pi * (m - 0.5) / M)
    return dt * np.asarray(L).reshape(-1, 1) + r[None, :]


def _mean_on_contour(fn, LR, shape):
    return np.mean(fn(LR), axis=1).reshape(shape)


def phi_eval(l: int, LR, shape):
    """MATLAB ``phiEval``: contour mean of ``phi_l``."""
    return _mean_on_contour(phi_fun(l), LR, shape)


def psi_eval(l: int, C: float, LR, shape):
    """MATLAB ``psiEval``: contour mean of ``C^l phi_l(C z)``."""
    f = phi_fun(l)
    return np.mean((C ** l) * f(C * LR), axis=1).reshape(shape)


def gamma_eval(j: int, k: int, LR, shape):
    """MATLAB ``gammaEval``: contour mean of ``gamma_{j,k}``."""
    return _mean_on_contour(gamma_fun(j, k), LR, shape)


# ---------------------------------------------------------------------------
# Scheme coefficients (MATLAB computeCoeffs + computeMissingCoeffs)
# ---------------------------------------------------------------------------

def compute_coeffs(K: Expinteg, dt: float, L: np.ndarray, M: int,
                   is_real: bool) -> dict:
    """Coefficients ``E, A, B, U, V, C`` of the scheme ``K`` for the
    diagonal linear operator ``L`` (MATLAB ``computeCoeffs``).  ``A`` is
    an ``s x s`` list, ``B`` length ``s``, ``U`` ``s x (q-1)``, ``V``
    length ``q-1``; absent entries are ``None``; all are already
    multiplied by ``dt``.
    """
    s, q = K.stages, K.steps
    L = np.asarray(L)
    shape = L.shape
    LR = compute_lr(dt, L, M, is_real)
    A = [[None] * s for _ in range(s)]
    B = [None] * s
    C = np.zeros(s)
    U = [[None] * max(q - 1, 0) for _ in range(s)]
    V = [None] * max(q - 1, 0)
    name = K.scheme

    def phi(l):
        v = phi_eval(l, LR, shape)
        return np.real(v) if is_real else v

    def psi(l, c):
        v = psi_eval(l, c, LR, shape)
        return np.real(v) if is_real else v

    from chebfunjax.operators.expinteg_tables import fill_tables
    p, ps = fill_tables(name, C, A, B, U, V, phi, psi)
    # MATLAB: computeMissingCoeffs for every scheme except the Lawson
    # pair and ETDRK2 (their first-stage weights are given explicitly).
    if name not in ("lawson4", "ablawson4", "etdrk2"):
        _missing(A, B, U, V, s, q, p, ps)
    return _finish(dt, L, C, A, B, U, V, s)


def _missing(A, B, U, V, s, q, phis, psis):
    """MATLAB ``computeMissingCoeffs``: the first-stage weights follow
    from consistency (row sums of psi / phi_1)."""
    for i in range(1, s):
        a = psis[(1, i + 1)].copy()
        for j in range(1, i):
            if A[i][j] is not None:
                a = a - A[i][j]
        for j in range(q - 1):
            if U[i][j] is not None:
                a = a - U[i][j]
        A[i][0] = a
    b = phis[1].copy()
    for i in range(1, s):
        if B[i] is not None:
            b = b - B[i]
    for j in range(q - 1):
        if V[j] is not None:
            b = b - V[j]
    B[0] = b


def _finish(dt, L, C, A, B, U, V, s):
    E = [np.exp(C[i] * dt * L) for i in range(s)] + [np.exp(dt * L)]
    scale = lambda X: None if X is None else dt * X  # noqa: E731
    return {
        "E": E,
        "A": [[scale(a) for a in row] for row in A],
        "B": [scale(b) for b in B],
        "U": [[scale(u) for u in row] for row in U],
        "V": [scale(v) for v in V],
        "C": C,
    }


# ---------------------------------------------------------------------------
# Generic step and multistep start-up
# ---------------------------------------------------------------------------

def _nonlin_coeffs(u, Nc, Nv, c2v, v2c, n_vars):
    """``Nc .* v2c(Nv(c2v(u)))`` block-wise over the stacked unknowns."""
    N = u.shape[0] // n_vars
    vals = np.concatenate([c2v(u[k * N:(k + 1) * N]) for k in range(n_vars)],
                          axis=0)
    vals = Nv(vals)
    coeffs = np.concatenate([v2c(vals[k * N:(k + 1) * N])
                             for k in range(n_vars)], axis=0)
    return Nc * coeffs


def one_step(K: Expinteg, coeffs: dict, Nc, Nv, c2v, v2c, n_vars: int,
             u_sol: list, Nu_sol: list):
    """One step of the scheme (MATLAB ``oneStep``).  ``u_sol``/``Nu_sol``
    hold the last ``q`` states, newest first; returns the updated lists."""
    s, q = K.stages, K.steps
    A, B, E, U, V = (coeffs["A"], coeffs["B"], coeffs["E"], coeffs["U"],
                     coeffs["V"])
    v = [None] * s
    Nvs = [None] * s
    v[0] = u_sol[0]
    Nvs[0] = Nu_sol[0]
    for i in range(1, s):
        vi = E[i] * v[0]
        for j in range(i):
            if A[i][j] is not None:
                vi = vi + A[i][j] * Nvs[j]
        for j in range(q - 1):
            if U[i][j] is not None:
                vi = vi + U[i][j] * Nu_sol[j + 1]
        v[i] = vi
        Nvs[i] = _nonlin_coeffs(vi, Nc, Nv, c2v, v2c, n_vars)
    sol = E[s] * v[0]
    for i in range(s):
        if B[i] is not None:
            sol = sol + B[i] * Nvs[i]
    for j in range(q - 1):
        if V[j] is not None:
            sol = sol + V[j] * Nu_sol[j + 1]
    Nsol = _nonlin_coeffs(sol, Nc, Nv, c2v, v2c, n_vars)
    if q == 1:
        return [sol], [Nsol]
    return [sol] + list(u_sol[:-1]), [Nsol] + list(Nu_sol[:-1])


def start_multistep(K: Expinteg, dt: float, L: np.ndarray, M: int,
                    is_real: bool, Nc, Nv, c2v, v2c, n_vars: int,
                    u_init, Nu_init):
    """MATLAB ``startMultistep``: the ``q-1`` start-up states of a
    multistep scheme -- ETDRK2 predictions refined by a fixed-point
    iteration on the exact variation-of-constants formula with the
    ``gamma`` weights.  Returns ``(u_sol, Nu_sol)`` newest first."""
    q = K.steps
    err_tol = min(1e-2, max(1e-10, dt ** q))
    L = np.asarray(L)
    shape = L.shape
    u_sol = [None] * q
    Nu_sol = [None] * q
    u_sol[q - 1] = u_init
    Nu_sol[q - 1] = Nu_init
    K2 = Expinteg("etdrk2")
    c2 = compute_coeffs(K2, dt, L, M, is_real)
    u_old, Nu_old = [u_init], [Nu_init]
    for j in range(1, q):
        u_new, Nu_new = one_step(K2, c2, Nc, Nv, c2v, v2c, n_vars,
                                 u_old, Nu_old)
        u_sol[q - 1 - j] = u_new[0]
        Nu_sol[q - 1 - j] = Nu_new[0]
        u_old, Nu_old = u_new, Nu_new
    LR = compute_lr(dt, L, M, is_real)
    g0 = [gamma_eval(0, j, LR, shape) for j in range(1, q)]
    g = [[gamma_eval(j, k, LR, shape) for k in range(1, q)]
         for j in range(1, q)]
    if is_real:
        g0 = [np.real(x) for x in g0]
        g = [[np.real(x) for x in row] for row in g]
    E = [np.exp(j * dt * L) for j in range(1, q)]
    err = 1.0
    u_old, Nu_old = list(u_sol), list(Nu_sol)
    it = 0
    max_iter = 100
    u_new, Nu_new = list(u_old), list(Nu_old)
    while err > err_tol and it < max_iter:
        it += 1
        err = 0.0
        u_new = [None] * q
        Nu_new = [None] * q
        for j in range(1, q):
            unj = E[j - 1] * u_old[q - 1] + dt * g0[j - 1] * Nu_old[q - 1]
            for l in range(1, q):
                temp = 0.0
                for i in range(l + 1):
                    temp = temp + ((-1.0) ** i) * math.comb(l, i) \
                        * Nu_old[q - 1 - l + i]
                unj = unj + dt * g[l - 1][j - 1] * temp
            err = max(err, float(np.max(np.abs(u_old[q - 1 - j] - unj))))
            u_new[q - 1 - j] = unj
            Nu_new[q - 1 - j] = _nonlin_coeffs(unj, Nc, Nv, c2v, v2c, n_vars)
        u_new[q - 1] = u_old[q - 1]
        Nu_new[q - 1] = Nu_old[q - 1]
        u_old, Nu_old = u_new, Nu_new
    if it >= max_iter:
        warnings.warn("Fixed-point iteration might not have converged.")
    if np.any(np.isnan(u_new[0])):
        raise ValueError(
            "Fixed-point iteration diverged. Try a smaller time-step.")
    return u_new, Nu_new


def integrate(K: Expinteg, dt: float, L: np.ndarray, M: int, is_real: bool,
              Nc, Nv, c2v, v2c, n_vars: int, u0, nsteps: int,
              post=None):
    """Drive ``nsteps`` steps of scheme ``K`` from the coefficient state
    ``u0`` (MATLAB solvepde's stepping loop, without the plotting).
    ``post`` (optional) is applied to the newest state after each step
    (e.g. re-Hermitianisation)."""
    coeffs = compute_coeffs(K, dt, L, M, is_real)
    Nu0 = _nonlin_coeffs(u0, Nc, Nv, c2v, v2c, n_vars)
    q = K.steps
    if q == 1:
        u_sol, Nu_sol = [u0], [Nu0]
        done = 0
    else:
        u_sol, Nu_sol = start_multistep(K, dt, L, M, is_real, Nc, Nv, c2v,
                                        v2c, n_vars, u0, Nu0)
        done = q - 1
    for _ in range(max(nsteps - done, 0)):
        u_sol, Nu_sol = one_step(K, coeffs, Nc, Nv, c2v, v2c, n_vars,
                                 u_sol, Nu_sol)
        if post is not None:
            u_sol[0] = post(u_sol[0])
            # the stored nonlinear term must belong to the projected state
            Nu_sol[0] = _nonlin_coeffs(u_sol[0], Nc, Nv, c2v, v2c, n_vars)
    return u_sol[0]
