# uses-numpy: DFS Fourier-Fourier IMEX time-stepping is a one-shot host-side
# solve per call (banded per-lambda-mode sparse LU factor/solve via SuperLU,
# mirroring MATLAB's sparse lu), matching the numpy pattern of
# operators/spinop.py; no JIT, no device management.
"""Spinopsphere -- stiff semilinear PDEs on the unit sphere,
u_t = L u + N(u) with L = A*lap (Laplace-Beltrami), solved by
spinsphere() with implicit-explicit schemes (IMEX-BDF4 when A is real,
LIRK4 when A is complex) in the doubled-Fourier-sphere (DFS)
Fourier-Fourier coefficient space.

Added by Claude Fable 5 (spinsphere port).

Provenance
----------
MATLAB source : spinsphere.m, @spinopsphere/spinopsphere.m,
    @spinopsphere/discretize.m, @spinoperator/solvepde.m,
    @imex/{imex,computeCoeffs,oneStep,startMultistep}.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers.
Algorithm: H. Montanelli and Y. Nakatsukasa, "Fourth-order
    time-stepping for stiff PDEs on the sphere", SISC 40(1), 2018.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.linalg import toeplitz
from scipy.sparse.linalg import splu

from chebfunjax.spherefun.spherefun import Spherefun

__all__ = ["Spinopsphere", "spinsphere", "func2str"]


# ---------------------------------------------------------------------------
# MATLAB-function-handle shim
# ---------------------------------------------------------------------------


class FuncHandle:
    """A minimal stand-in for a MATLAB function handle.

    Wraps a Python callable together with the exact string that MATLAB's
    ``func2str`` would return, so ports can assert on the operator's
    textual form (as the spinopsphere test does) while still evaluating
    it.

    Provenance
    ----------
    MATLAB source : builtin func2str / anonymous function handles
    Chebfun commit: 7574c77
    """

    def __init__(self, func, string: str):
        self._func = func
        self.string = string

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return self.string


def func2str(handle) -> str:
    """Return the MATLAB ``func2str`` string of a :class:`FuncHandle`.

    Provenance
    ----------
    MATLAB source : builtin func2str
    Chebfun commit: 7574c77
    """
    return str(handle)


# ---------------------------------------------------------------------------
# Built-in PDE catalogue
# ---------------------------------------------------------------------------


def _cart(lam, th):
    """Cartesian coordinates on the unit sphere from (lam, theta)."""
    x = jnp.sin(th) * jnp.cos(lam)
    y = jnp.sin(th) * jnp.sin(lam)
    z = jnp.cos(th)
    return x, y, z


def _preset(pdechar: str):
    """Return (lin_scale, lin_handle, nonlin_handle, tspan, init) for a
    named PDE, mirroring @spinopsphere/spinopsphere.m parseInputs.

    Provenance
    ----------
    MATLAB source : @spinopsphere/spinopsphere.m (parseInputs)
    Chebfun commit: 7574c77
    """
    key = pdechar.upper()
    if key == "AC":
        lin = FuncHandle(None, "@(u)1e-2*lap(u)")
        nonlin = FuncHandle(lambda u: u - u ** 3, "@(u)u-u.^3")

        def ac_u0(lam, th):
            x, y, z = _cart(lam, th)
            return jnp.cos(jnp.cosh(5.0 * x * z) - 10.0 * y)

        init = Spherefun.from_function(ac_u0)
        return 1e-2, lin, nonlin, (0.0, 60.0), init

    if key == "GL":
        lin = FuncHandle(None, "@(u)1e-3*lap(u)")
        # abs() (builtin) dispatches to numpy for the numpy value arrays
        # used in the time-stepping hot loop -- keeping it out of JAX.
        nonlin = FuncHandle(
            lambda u: u - (1.0 + 1.5j) * u * (abs(u) ** 2),
            "@(u)u-(1+1.5i)*u.*(abs(u).^2)",
        )
        # MATLAB seeds a randnfunsphere(.1); chebfunjax has no
        # randnfunsphere, so init is left unset (the demo/test supplies
        # its own S.init before solving).
        return 1e-3, lin, nonlin, (0.0, 100.0), None

    if key == "NLS":
        lin = FuncHandle(None, "@(u)1i*lap(u)")
        nonlin = FuncHandle(
            lambda u: 1j * u * (abs(u) ** 2),
            "@(u)1i*u.*abs(u).^2",
        )
        return 1j, lin, nonlin, (0.0, 3.0), None

    if key == "GM":
        raise NotImplementedError(
            "spinopsphere('GM') is a system (nVars=2); spinsphere only "
            "supports scalar equations in chebfunjax."
        )

    raise ValueError(f"Unrecognized PDE {pdechar!r}. Options: AC, GL, NLS.")


# ---------------------------------------------------------------------------
# Spinopsphere
# ---------------------------------------------------------------------------


class Spinopsphere:
    """Spatial part S of a time-dependent PDE ``u_t = S(u) = L u + N(u)``
    on the unit sphere, with ``L = A*lap`` the Laplace-Beltrami operator
    scaled by a constant ``A``.

    Construct from a named preset (``Spinopsphere('AC')``,
    ``Spinopsphere('GL')``, ``Spinopsphere('NLS')``) or from a time
    interval (``Spinopsphere([0, 1])``) and then set ``.lin_scale``,
    ``.nonlin`` and ``.init`` directly.

    Attributes
    ----------
    domain : tuple of four floats
        ``(-pi, pi, 0, pi)`` (longitude x colatitude), MATLAB convention.
    lin : FuncHandle
        The linear part ``@(u) A*lap(u)`` (textual form + placeholder).
    lin_scale : complex
        The constant ``A`` in front of the Laplacian.
    nonlin : FuncHandle
        The nonlinear part ``@(u) f(u)`` acting elementwise in value
        space; ``func2str(S.nonlin)`` returns its MATLAB string.
    tspan : tuple of floats
        Time interval ``(t0, tf)``.
    init : Spherefun or None
        Initial condition.

    Provenance
    ----------
    MATLAB source : @spinopsphere/spinopsphere.m, @spinoperator/spinoperator.m
    Chebfun commit: 7574c77
    """

    def __init__(self, arg=None):
        self.domain = (-float(np.pi), float(np.pi), 0.0, float(np.pi))
        self.lin = None
        self.lin_scale = None
        self.nonlin = None
        self.tspan = None
        self.init = None
        if arg is None:
            return
        if isinstance(arg, str):
            A, lin, nonlin, tspan, init = _preset(arg)
            self.lin_scale = A
            self.lin = lin
            self.nonlin = nonlin
            self.tspan = tspan
            self.init = init
        else:
            # tspan constructor: Spinopsphere([t0, ..., tf])
            self.tspan = tuple(float(v) for v in arg)

    @property
    def numVars(self) -> int:
        """Number of unknown functions (1 for the supported scalar PDEs).

        Provenance
        ----------
        MATLAB source : @spinoperator/spinoperator.m (get.numVars)
        Chebfun commit: 7574c77
        """
        return 1

    def __repr__(self) -> str:
        return (f"Spinopsphere(lin_scale={self.lin_scale!r}, "
                f"tspan={self.tspan!r})")


# ---------------------------------------------------------------------------
# DFS Fourier-Fourier discretization of the Laplace-Beltrami operator
# ---------------------------------------------------------------------------


def _v2c_1d(v):
    """Values -> Fourier coeffs along axis 0 (trigtech convention)."""
    n = v.shape[0]
    c = np.fft.fftshift(np.fft.fft(v, axis=0), axes=0) / n
    half = n // 2
    ks = np.arange(-half, half)
    fix = ((-1.0 + 0j) ** ks).reshape((n,) + (1,) * (v.ndim - 1))
    return c * fix


def _c2v_1d(c):
    """Fourier coeffs -> values along axis 0 (inverse of _v2c_1d)."""
    n = c.shape[0]
    half = n // 2
    ks = np.arange(-half, half)
    fix = ((-1.0 + 0j) ** ks).reshape((n,) + (1,) * (c.ndim - 1))
    c = c * fix
    return np.fft.ifft(np.fft.ifftshift(n * c, axes=0), axis=0)


def _vals2coeffs2(u):
    """2-D trigtech vals->coeffs (theta along axis 0, lambda along axis 1).

    Mirrors @spinopsphere/getVals2CoeffsTransform.m.
    """
    return _v2c_1d(_v2c_1d(u).T).T


def _coeffs2vals2(c):
    """2-D trigtech coeffs->vals (inverse of :func:`_vals2coeffs2`).

    Mirrors @spinopsphere/getCoeffs2ValsTransform.m.
    """
    return _c2v_1d(_c2v_1d(c).T).T


def _discretize(N):
    """Build the DFS Fourier-Fourier building blocks of the surface
    Laplacian (premultiplied by sin^2 theta).

    Returns ``(Tsin2, B, kn)`` where ``Tsin2`` and ``B`` are sparse
    (CSC) N x N matrices: the block-diagonal linear operator ``L =
    A*lapmat`` acts on the (theta x lambda) coefficient matrix as
    ``A * (B @ C - C * kn**2)`` and the preconditioner is ``Tsin2 @ C``.
    ``Tsin2`` multiplies by ``sin^2(theta)`` and ``B = Tsin2*D2m +
    Tcossin*Dm``.  Verified against ``spherefun.laplacian`` to ~5e-12.

    The per-lambda-mode blocks are pentadiagonal (plus a few pole/mean
    corner entries), so they are kept sparse and factored with SuperLU,
    mirroring MATLAB's sparse ``lu`` and avoiding dense-BLAS overhead.

    Provenance
    ----------
    MATLAB source : @spinopsphere/discretize.m
    Chebfun commit: 7574c77
    """
    m = N
    n = N
    # d/dtheta: 1i*[0, -m/2+1 : m/2-1]  (Nyquist derivative zeroed)
    dm = np.concatenate([[0.0], np.arange(-m // 2 + 1, m // 2)]) * 1j
    Dm = np.diag(dm)
    # d^2/dtheta^2: -(-m/2 : m/2-1)^2
    d2m = -(np.arange(-m // 2, m // 2).astype(float)) ** 2
    D2m = np.diag(d2m)
    # Multiplication by sin^2(theta): Toeplitz(1/2 diag, -1/4 on +/-2)
    # with the pole/mean-preserving endpoint operators P, Q.
    cs = np.zeros(m + 5)
    cs[0] = 0.5
    cs[2] = -0.25
    Msin2 = toeplitz(np.conj(cs), cs)[:, 2:m + 3]
    # Multiplication by cos(theta)*sin(theta): 1i/4 on +/-2 (Hermitian).
    cc = np.zeros(m + 5, dtype=complex)
    cc[2] = 1j / 4.0
    Mcossin = toeplitz(np.conj(cc), cc)[:, 2:m + 3]
    # Endpoint operators (MATLAB discretize.m).
    P = np.zeros((m + 1, m))
    for i in range(m):
        P[i, i] = 1.0
    P[0, 0] = 0.5
    P[m, 0] = 0.5
    Q = np.zeros((m, m + 5))
    for r in range(m):
        Q[r, r + 2] = 1.0
    Q[0, 2] = 1.0
    Q[0, m + 2] = 1.0
    Tsin2 = np.round(Q @ Msin2 @ P, 15)
    Tcossin = np.round(Q @ Mcossin @ P, 15)
    B = Tsin2 @ D2m + Tcossin @ Dm
    kn = np.arange(-n // 2, n // 2).astype(float)  # lambda wavenumbers
    return sp.csc_matrix(Tsin2), sp.csc_matrix(B), kn


# ---------------------------------------------------------------------------
# spinsphere solver
# ---------------------------------------------------------------------------


def _dfs_grid(N):
    """Doubled-up DFS computational grid (lambda, theta) on [-pi, pi]^2.

    Mirrors @spinopsphere/getGrid.m: theta is imposed on [-pi, pi] (the
    doubled latitude), not on the spherefun domain [0, pi].
    """
    lam = -np.pi + 2.0 * np.pi * np.arange(N) / N
    th = -np.pi + 2.0 * np.pi * np.arange(N) / N
    return np.meshgrid(lam, th)  # ll[i,j]=lam_j, tt[i,j]=th_i


def _make_output_spherefun(Cfinal, N):
    """Build the real-valued output Spherefun from the final
    Fourier-Fourier coefficient matrix.

    The 2-D trig interpolant ``u(lam,theta) = Re sum_{p,q} C[p,q]
    e^{i kp theta} e^{i kq lam}`` is band-limited to N modes; taking its
    real part reproduces @spinoperator/reshapeData.m + spherefun(...,
    'trig').  The evaluator is pure NumPy so the adaptive Spherefun
    constructor samples it without JAX tracing/compilation.  A relaxed
    tolerance keeps the pivoting from chasing the ~1e-13 non-BMC roundoff
    left by the time-stepping (far below any tested accuracy).
    """
    kp = np.arange(-N // 2, N // 2).astype(float)
    C = np.asarray(Cfinal, dtype=complex)

    def ev(lam, theta):
        lam = np.asarray(lam, dtype=float)
        theta = np.asarray(theta, dtype=float)
        # Fast separable path for meshgrid inputs (theta constant along
        # rows, lam constant along columns), which the Spherefun
        # constructor's phase one always supplies: value = Ath @ C @
        # Alam.T, O(npts * N) instead of O(npts * N^2).
        if (lam.ndim == 2 and theta.ndim == 2 and lam.shape == theta.shape
                and bool(np.all(lam == lam[0:1, :]))
                and bool(np.all(theta == theta[:, 0:1]))):
            ath = np.exp(1j * theta[:, 0][:, None] * kp[None, :])
            alam = np.exp(1j * lam[0, :][:, None] * kp[None, :])
            return np.real(ath @ C @ alam.T)
        # General per-point path (phase-two slices are small 1-D lines).
        shape = np.broadcast_shapes(lam.shape, theta.shape)
        lamf = np.broadcast_to(lam, shape).reshape(-1)
        thf = np.broadcast_to(theta, shape).reshape(-1)
        ath = np.exp(1j * thf[:, None] * kp[None, :])
        alam = np.exp(1j * lamf[:, None] * kp[None, :])
        vals = np.einsum("np,pq,nq->n", ath, C, alam)
        return np.real(vals).reshape(shape)

    # The field is an exact degree-N/2 trig polynomial, so its slices
    # resolve on any grid >= N (coefficients vanish beyond the band).
    # Cap the constructor grid at ~2N -- enough to capture the full band
    # exactly -- and silence the "not resolved" note that fires when the
    # solution is grid-limited (its Nyquist-edge coefficients sit at a
    # finite floor, exactly as in MATLAB's fixed-grid spherefun(vals,
    # 'trig'), which does no adaptive refinement at all).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*slices not resolved.*",
            category=RuntimeWarning,
        )
        return Spherefun.from_function(ev, tol=1e-9, max_sample=2 * N)


def spinsphere(S: Spinopsphere, N: int, dt: float, *args, **kwargs):
    """Solve the sphere PDE specified by ``S`` with ``N`` grid points in
    each direction and time-step ``dt`` (MATLAB ``spinsphere(S, N, dt,
    'plot', 'off')``); returns the solution at ``tspan(end)`` as a real
    :class:`Spherefun`.

    Diffusive PDEs (real Laplacian constant) are stepped with IMEX-BDF4
    started by three LIRK4 steps; dispersive PDEs (complex constant, e.g.
    NLS) use LIRK4 throughout.  Plotting arguments are accepted and
    ignored.

    Provenance
    ----------
    MATLAB source : spinsphere.m, @spinoperator/solvepde.m,
        @imex/{computeCoeffs,oneStep,startMultistep}.m
    Chebfun commit: 7574c77
    Algorithm: Montanelli & Nakatsukasa, SISC 40(1), 2018.
    """
    if S.numVars != 1:
        raise NotImplementedError(
            "spinsphere only supports scalar equations (nVars=1).")
    if S.init is None:
        raise ValueError(
            "Spinopsphere has no initial condition (set S.init).")

    A = complex(S.lin_scale)
    is_real = A.imag == 0.0
    nonlin = S.nonlin
    t0, tf = S.tspan
    nsteps = int(round((tf - t0) / dt))

    Tsin2, B, kn = _discretize(N)
    Im = sp.identity(N, format="csc")
    kn2 = (kn ** 2)[None, :]

    def apply_lap(C):
        return A * (B @ C - C * kn2)

    def apply_P(C):
        return Tsin2 @ C

    def nhat(C):
        return _vals2coeffs2(nonlin(_coeffs2vals2(C)))

    # Per-lambda-mode sparse LU factorizations (block-diagonal
    # decoupling; each block is pentadiagonal, so SuperLU factors and
    # solves it in linear time -- mirroring MATLAB's sparse lu).  Factor
    # in complex arithmetic so the (generally complex) state can be
    # back-solved directly.
    def _splu_c(M):
        return splu(sp.csc_matrix(M, dtype=complex))

    lu_P = _splu_c(Tsin2)
    lu_La = [_splu_c(Tsin2 - 0.25 * dt * A * (B - kn[q] ** 2 * Im))
             for q in range(N)]

    def solve_cols(lu_list, R):
        X = np.empty_like(R, dtype=complex)
        for q in range(N):
            X[:, q] = lu_list[q].solve(np.ascontiguousarray(R[:, q]))
        return X

    def solve_P(C):
        X = np.empty_like(C, dtype=complex)
        for q in range(N):
            X[:, q] = lu_P.solve(np.ascontiguousarray(C[:, q]))
        return X

    def lirk4_step(v, Nv):
        # @imex/oneStep.m, lirk4 branch (scalar, Nc=1).
        w = apply_P(v)
        wa = w + dt * apply_P(0.25 * Nv)
        a = solve_cols(lu_La, wa)
        Na = nhat(a)
        wb = w + dt * apply_lap(0.5 * a) + dt * apply_P(-0.25 * Nv + Na)
        b = solve_cols(lu_La, wb)
        Nb = nhat(b)
        wc = (w + dt * apply_lap(17 / 50 * a - 1 / 25 * b)
              + dt * apply_P(-13 / 100 * Nv + 43 / 75 * Na + 8 / 75 * Nb))
        c = solve_cols(lu_La, wc)
        Nc = nhat(c)
        wd = (w + dt * apply_lap(371 / 1360 * a - 137 / 2720 * b + 15 / 544 * c)
              + dt * apply_P(-6 / 85 * Nv + 42 / 85 * Na + 179 / 1360 * Nb
                             - 15 / 272 * Nc))
        d = solve_cols(lu_La, wd)
        Nd = nhat(d)
        we = (w + dt * apply_lap(25 / 24 * a - 49 / 48 * b + 125 / 16 * c
                                 - 85 / 12 * d)
              + dt * apply_P(79 / 24 * Na - 5 / 8 * Nb + 25 / 2 * Nc
                             - 85 / 6 * Nd))
        e = solve_cols(lu_La, we)
        Ne = nhat(e)
        v_new = (v + dt * solve_P(apply_lap(25 / 24 * a - 49 / 48 * b
                                            + 125 / 16 * c - 85 / 12 * d
                                            + 1 / 4 * e))
                 + dt * (25 / 24 * Na - 49 / 48 * Nb + 125 / 16 * Nc
                         - 85 / 12 * Nd + 1 / 4 * Ne))
        return v_new, nhat(v_new)

    # Initial condition on the DFS grid -> Fourier-Fourier coefficients.
    ll, tt = _dfs_grid(N)
    V0 = np.asarray(S.init(jnp.asarray(ll), jnp.asarray(tt)), dtype=complex)
    C0 = _vals2coeffs2(V0)
    Nc0 = nhat(C0)

    if is_real:
        # IMEX-BDF4 (q=4), bootstrapped with three LIRK4 steps.
        q = 4
        lu_bdf = [_splu_c(25 * Tsin2 - 12 * dt * A * (B - kn[j] ** 2 * Im))
                  for j in range(N)]
        us = [None] * q
        Ns = [None] * q
        us[q - 1] = C0
        Ns[q - 1] = Nc0
        cur, curN = C0, Nc0
        for j in range(1, q):
            cur, curN = lirk4_step(cur, curN)
            us[q - 1 - j] = cur
            Ns[q - 1 - j] = curN
        for _ in range(nsteps - (q - 1)):
            R = apply_P(48 * us[0] - 36 * us[1] + 16 * us[2] - 3 * us[3]
                        + 48 * dt * Ns[0] - 72 * dt * Ns[1]
                        + 48 * dt * Ns[2] - 12 * dt * Ns[3])
            v = solve_cols(lu_bdf, R)
            Nv = nhat(v)
            us = [v, us[0], us[1], us[2]]
            Ns = [Nv, Ns[0], Ns[1], Ns[2]]
        Cfinal = us[0]
    else:
        # LIRK4 throughout (dispersive PDEs).
        cur, curN = C0, Nc0
        for _ in range(nsteps):
            cur, curN = lirk4_step(cur, curN)
        Cfinal = cur

    return _make_output_spherefun(Cfinal, N)
