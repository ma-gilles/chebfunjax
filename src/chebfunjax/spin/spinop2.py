# uses-numpy: contour-integral phi-function evaluation uses numpy (setup phase, not JIT-safe)
"""SpinOp2 — 2D periodic PDE operator for the Spin framework.

Translated from MATLAB Chebfun class @spinop2 (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

The 2D linear operator is restricted to the form

    L = A * lap + B * biharm + C * triharm + D * quadharm + E * quintharm

where ``lap`` is the 2D Laplacian and higher harmonics are powers of ``lap``.
In Fourier space the Laplacian on [ax, bx] x [ay, by] with N modes per
dimension has eigenvalues

    lambda_{j,k} = -(xi_j^2 + eta_k^2)

where xi_j = 2*pi/(bx-ax) * j  and  eta_k = 2*pi/(by-ay) * k  are the
angular wavenumbers.  The N x N eigenvalue matrix is::

    L_diag = A * lap_mat + B * lap_mat**2 + ...

where lap_mat[j, k] = -(xi_j^2 + eta_k^2).

Provenance
----------
MATLAB source : @spinop2/spinop2.m, @spinop2/discretize.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and The
    Chebfun Developers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Built-in PDE definitions
# ---------------------------------------------------------------------------


def _make_builtin_pdes2() -> dict:
    """Return the catalogue of built-in 2D periodic PDEs.

    Each entry maps a (case-insensitive) key to a dict with keys:

      lin_coeffs   : (A, B, C, D, E) — coefficients of lap, biharm, triharm,
                     quadharm, quintharm in the linear part.
      nonlin_vals  : callable(u_vals) -> array — nonlinear part in value space.
                     For multi-component PDEs this is a list of callables, one
                     per component.
      n_vars       : int — number of PDE components (1 for scalar PDEs).
      domain       : (ax, bx, ay, by)
      tspan        : (t0, tf)
      u0           : callable(x, y) -> array, or list of callables for
                     multi-component PDEs.
      N            : default number of Fourier modes per direction.
      dt           : default time-step.
      is_real      : True if the solution is real-valued.

    Notes
    -----
    The linear part in MATLAB Chebfun 2D is restricted to isotropic operators:
    multiples of the Laplacian, biharmonic, etc.  We store the scalar
    coefficients (A, B, C, D, E) for lap, biharm, triharm, quadharm, quintharm.

    Provenance
    ----------
    MATLAB source : @spinop2/spinop2.m  (parseInputs function)
    Chebfun commit: 7574c77
    """
    # ---- Allen-Cahn 2D ----
    # u_t = eps * lap(u) + u - u^3,  eps = 1e-2
    # lin = eps * lap   -> lin_coeffs = (eps, 0, 0, 0, 0)
    # nonlin(u) = u - u^3  (no differentiation)
    _ac2_eps = 1e-2

    def _ac2_u0(x, y):
        return jnp.tanh(
            (jnp.sin(y) - jnp.sin(x)) / jnp.sqrt(2 * _ac2_eps)
        )

    # ---- Ginzburg-Landau 2D (GL) ----
    # u_t = lap(u) + u - (1 + 1.5i)*u*|u|^2
    # lin = lap  -> A=1
    # nonlin(u) = u - (1+1.5i)*u*|u|^2
    def _gl_u0(x, y):
        # Random-like smooth initial condition: use a sum of low-frequency modes
        # (MATLAB uses randnfun2 normalized to inf-norm 1; we use a deterministic
        # approximation that exercises the same dynamics).
        return (
            0.5 * jnp.cos(2 * jnp.pi * x / 100.0)
            + 0.5j * jnp.sin(2 * jnp.pi * y / 100.0)
            + 0.3 * jnp.cos(4 * jnp.pi * (x + y) / 100.0)
        )

    # ---- Gray-Scott equations (stripes) ----
    # u_t = 2e-5*lap(u) + F*(1-u) - u*v^2
    # v_t = 1e-5*lap(v) - (F+K)*v + u*v^2
    # lin = [(2e-5, 0, ...), (1e-5, 0, ...)]  (per component)
    _gs_F = 0.030
    _gs_K = 0.057

    def _gs_u0(x, y):
        return 1.0 - jnp.exp(-100.0 * ((x - 0.5) ** 2 + (y - 0.505) ** 2))

    def _gs_v0(x, y):
        return jnp.exp(-100.0 * ((x - 0.5) ** 2 + 2.0 * (y - 0.5) ** 2))

    # ---- Swift-Hohenberg 2D ----
    # u_t = -2*lap(u) - biharm(u) - 0.9*u - u^3
    # lin = -2*lap - biharm  -> lin_coeffs = (-2, -1, 0, 0, 0)
    # nonlin(u) = -0.9*u - u^3
    def _sh_u0(x, y):
        # Deterministic smooth "random-like" initial condition
        return 0.1 * jnp.cos(x / 50.0) * jnp.sin(y / 50.0)

    return {
        "ac2": dict(
            lin_coeffs=(_ac2_eps, 0.0, 0.0, 0.0, 0.0),
            nonlin_vals=lambda u: u - u ** 3,
            n_vars=1,
            domain=(0.0, 2 * float(jnp.pi), 0.0, 2 * float(jnp.pi)),
            tspan=(0.0, 50.0),
            u0=_ac2_u0,
            N=128,
            dt=1e-2,
            is_real=True,
        ),
        "gl": dict(
            lin_coeffs=(1.0, 0.0, 0.0, 0.0, 0.0),
            nonlin_vals=lambda u: u - (1.0 + 1.5j) * u * jnp.abs(u) ** 2,
            n_vars=1,
            domain=(0.0, 100.0, 0.0, 100.0),
            tspan=(0.0, 100.0),
            u0=_gl_u0,
            N=64,
            dt=5e-3,
            is_real=False,
        ),
        "gs": dict(
            # Multi-component: (u, v), lin_coeffs stored as list per component
            lin_coeffs=[(2e-5, 0.0, 0.0, 0.0, 0.0), (1e-5, 0.0, 0.0, 0.0, 0.0)],
            nonlin_vals=[
                lambda u, v: _gs_F * (1.0 - u) - u * v ** 2,
                lambda u, v: -(_gs_F + _gs_K) * v + u * v ** 2,
            ],
            n_vars=2,
            domain=(0.0, 1.0, 0.0, 1.0),
            tspan=(0.0, 2000.0),
            u0=[_gs_u0, _gs_v0],
            N=64,
            dt=2.0,
            is_real=True,
        ),
        "sh": dict(
            lin_coeffs=(-2.0, -1.0, 0.0, 0.0, 0.0),
            nonlin_vals=lambda u: -0.9 * u - u ** 3,
            n_vars=1,
            domain=(0.0, 50.0, 0.0, 50.0),
            tspan=(0.0, 200.0),
            u0=_sh_u0,
            N=64,
            dt=1e-2,
            is_real=True,
        ),
    }


_BUILTIN_PDES2 = _make_builtin_pdes2()


# ---------------------------------------------------------------------------
# SpinOp2 class
# ---------------------------------------------------------------------------


class SpinOp2:
    """Operator for a 2D periodic semilinear PDE u_t = L[u] + N[u].

    The PDE is discretized on a uniform 2D Fourier grid of N x N points on
    the periodic domain [ax, bx] x [ay, by].  The linear part L must be an
    isotropic operator — a polynomial in the 2D Laplacian::

        L = A * lap + B * biharm + C * triharm + D * quadharm + E * quintharm

    In Fourier space the eigenvalues form an N x N diagonal matrix.

    For multi-component PDEs (e.g. Gray-Scott), pass lists of length n_vars
    for ``lin_coeffs``, ``nonlin_vals``, and ``u0``.

    Attributes
    ----------
    lin_coeffs : tuple[float, float, float, float, float] or list thereof
        Coefficients (A, B, C, D, E) for lap, biharm, triharm, quadharm,
        quintharm.  For multi-component PDEs: list of n_vars such tuples.
    nonlin_vals : callable(u_vals) -> array, or list of callables
        Nonlinear part in physical space.  For multi-component PDEs this is
        a list of callables, one per equation, each taking n_vars arrays.
    n_vars : int
        Number of PDE components.
    domain : tuple[float, float, float, float]
        Spatial domain (ax, bx, ay, by).
    tspan : tuple[float, float]
        Time interval (t0, tf).
    u0 : callable(x, y) -> array, or list of callables
        Initial condition(s).
    is_real : bool
        If True the solution is real-valued.

    Notes
    -----
    Sign convention: the PDE is ``u_t = L(u) + N(u)`` where both L and N
    include their signs.  For Allen-Cahn 2D the user writes::

        L = A * lap,  N(u) = u - u^3

    For the Swift-Hohenberg equation::

        L = -2*lap - biharm,  N(u) = -0.9*u - u^3

    Provenance
    ----------
    MATLAB source : @spinop2/spinop2.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and The
        Chebfun Developers.

    See Also
    --------
    spin2
    """

    def __init__(
        self,
        lin_coeffs,
        nonlin_vals,
        n_vars: int,
        domain: Tuple[float, float, float, float],
        tspan: Tuple[float, float],
        u0,
        is_real: bool = True,
    ) -> None:
        self.lin_coeffs = lin_coeffs
        self.nonlin_vals = nonlin_vals
        self.n_vars = n_vars
        self.domain = domain
        self.tspan = tspan
        self.u0 = u0
        self.is_real = is_real

    @classmethod
    def from_name(cls, name: str) -> "SpinOp2":
        """Construct a SpinOp2 from a built-in PDE name.

        Parameters
        ----------
        name : str
            Case-insensitive PDE identifier.  Supported values:

            * ``'AC2'`` — Allen-Cahn equation (2D)
            * ``'GL'``  — Ginzburg-Landau equation
            * ``'GS'``  — Gray-Scott equations (stripes)
            * ``'SH'``  — Swift-Hohenberg equation

        Returns
        -------
        SpinOp2

        Raises
        ------
        ValueError
            If *name* is not recognised.

        Examples
        --------
        >>> op = SpinOp2.from_name('GL')

        Provenance
        ----------
        MATLAB source : @spinop2/spinop2.m (parseInputs)
        Chebfun commit: 7574c77
        """
        key = name.lower()
        if key not in _BUILTIN_PDES2:
            supported = ", ".join(_BUILTIN_PDES2.keys())
            raise ValueError(
                f"Unrecognised 2D PDE name {name!r}. "
                f"Supported names (case-insensitive): {supported}."
            )
        d = _BUILTIN_PDES2[key]
        return cls(
            lin_coeffs=d["lin_coeffs"],
            nonlin_vals=d["nonlin_vals"],
            n_vars=d["n_vars"],
            domain=d["domain"],
            tspan=d["tspan"],
            u0=d["u0"],
            is_real=d["is_real"],
        )

    def default_N(self, name: Optional[str] = None) -> int:
        """Default number of Fourier modes per direction (if built-in)."""
        if name is not None:
            key = name.lower()
            if key in _BUILTIN_PDES2:
                return _BUILTIN_PDES2[key]["N"]
        return 64

    def default_dt(self, name: Optional[str] = None) -> float:
        """Default time-step for this PDE (if built-in)."""
        if name is not None:
            key = name.lower()
            if key in _BUILTIN_PDES2:
                return _BUILTIN_PDES2[key]["dt"]
        return 1e-3

    def __repr__(self) -> str:
        ax, bx, ay, by = self.domain
        t0, tf = self.tspan
        return (
            f"SpinOp2(domain=[{float(ax):.4g},{float(bx):.4g}]"
            f"x[{float(ay):.4g},{float(by):.4g}], "
            f"tspan=[{t0}, {tf}], n_vars={self.n_vars})"
        )


# ---------------------------------------------------------------------------
# Helpers: Fourier eigenvalues for 2D Laplacian
# ---------------------------------------------------------------------------


def _fourier_wavenumbers_2d(
    N: int, domain: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the 2D angular wavenumber grids for an N x N Fourier grid.

    For a domain [ax, bx] x [ay, by] of sizes Lx and Ly, the angular
    wavenumbers are:

        xi[j]  = (2*pi/Lx) * k_j,   k in FFT ordering
        eta[k] = (2*pi/Ly) * k_k

    Returns
    -------
    XI : np.ndarray, shape (N, N)
        2D grid of x-direction wavenumbers (rows constant along y).
    ETA : np.ndarray, shape (N, N)
        2D grid of y-direction wavenumbers (cols constant along x).

    Provenance
    ----------
    MATLAB source : @spinop2/discretize.m  (D2 construction via trigspec)
    Chebfun commit: 7574c77
    """
    ax, bx, ay, by = domain
    Lx = float(bx) - float(ax)
    Ly = float(by) - float(ay)
    kx = np.fft.fftfreq(N, d=1.0 / N)  # integers
    ky = np.fft.fftfreq(N, d=1.0 / N)
    xi = (2.0 * np.pi / Lx) * kx  # angular freq in x
    eta = (2.0 * np.pi / Ly) * ky  # angular freq in y
    XI, ETA = np.meshgrid(xi, eta, indexing="ij")  # (N, N), row=x, col=y
    return XI, ETA


def build_laplacian_eigenvalues_2d(
    N: int, domain: Tuple[float, float, float, float]
) -> np.ndarray:
    """Build the N x N matrix of 2D Laplacian eigenvalues.

    The eigenvalue at mode (j, k) is

        lap_jk = -(xi_j^2 + eta_k^2)

    which is the eigenvalue of the continuous operator ``d^2/dx^2 + d^2/dy^2``
    on the periodic domain.

    Parameters
    ----------
    N : int
        Number of Fourier modes per direction.
    domain : tuple(float, float, float, float)
        (ax, bx, ay, by).

    Returns
    -------
    lap_mat : np.ndarray, shape (N, N), float
        Laplacian eigenvalues.

    Provenance
    ----------
    MATLAB source : @spinop2/discretize.m  (lapmat construction)
    Chebfun commit: 7574c77
    """
    XI, ETA = _fourier_wavenumbers_2d(N, domain)
    return -(XI ** 2 + ETA ** 2)


def build_linear_eigenvalues_2d(
    lin_coeffs,
    N: int,
    domain: Tuple[float, float, float, float],
) -> np.ndarray:
    """Build the N x N diagonal of the linear operator for one component.

    The linear operator is a polynomial in the Laplacian:

        L = A*lap + B*biharm + C*triharm + D*quadharm + E*quintharm

    In Fourier space this is::

        L_mat = A*lap_mat + B*lap_mat**2 + C*lap_mat**3 + ...

    Parameters
    ----------
    lin_coeffs : sequence of five floats
        (A, B, C, D, E).
    N : int
    domain : tuple

    Returns
    -------
    L_mat : np.ndarray, shape (N, N), complex
        Eigenvalues of the linear operator.

    Provenance
    ----------
    MATLAB source : @spinop2/discretize.m
    Chebfun commit: 7574c77
    """
    A, B, C, D, E = lin_coeffs
    lap = build_laplacian_eigenvalues_2d(N, domain)
    L_mat = (
        A * lap
        + B * lap ** 2
        + C * lap ** 3
        + D * lap ** 4
        + E * lap ** 5
    )
    return np.asarray(L_mat, dtype=complex)


def _dealias_mask_2d(N: int) -> np.ndarray:
    """Return a 2D boolean mask that zeros the top 1/3 of 2D Fourier modes.

    In 2D, the 2/3-rule zeros modes with |j| > N/3 OR |k| > N/3 in the FFT
    ordering.  This is the direct 2D generalization of the 1D mask.

    Parameters
    ----------
    N : int
        Number of Fourier modes per direction.

    Returns
    -------
    mask : np.ndarray, shape (N, N), bool
        True for modes to *keep*.

    Provenance
    ----------
    MATLAB source : @spinop2/getDealiasingIndexes.m
    Chebfun commit: 7574c77
    """
    # MATLAB: toOne = floor(N/2)+1-ceil(N/6) : floor(N/2)+ceil(N/6)  (1-indexed)
    # In 0-indexed Python: lo = floor(N/2) - ceil(N/6),  hi = floor(N/2) + ceil(N/6)
    import math as _math
    half = N // 2
    sixth = _math.ceil(N / 6)
    lo = half - sixth
    hi = half + sixth  # exclusive upper bound
    mask = np.ones((N, N), dtype=bool)
    mask[lo:hi, :] = False
    mask[:, lo:hi] = False
    return mask
