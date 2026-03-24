"""SpinOp3 — 3D periodic PDE operator for the Spin framework.

Translated from MATLAB Chebfun class @spinop3 (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Built-in 3-D periodic PDE definitions
# ---------------------------------------------------------------------------


def _make_builtin_pdes_3d() -> dict:
    """Return the catalogue of built-in 3-D periodic PDEs.

    Each entry maps a (case-insensitive) key to a dict with keys:
      lin_type  : string encoding the operator type (e.g. 'lap', 'biharm')
      lin_scale : float or complex — scalar A in front of the Laplacian,
                  or a tuple (A, B, ...) for lap + biharm + ...
      nonlin    : callable(u_vals) -> nonlinear part in value space
      domain    : (x0, x1, y0, y1, z0, z1)
      tspan     : (t0, tf)
      u0        : callable(x, y, z) -> initial condition values
      N         : default number of grid points per dimension
      dt        : default time-step
      is_real   : True if solution is real-valued

    The linear part is restricted to combinations of Laplacian, biharmonic,
    triharmonic, quadharmonic and quintharmonic operators (as in Chebfun).
    """
    # ------------------------------------------------------------------
    # Ginzburg-Landau: u_t = lap(u) + u - (1+1.5i)*u*|u|^2
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)

    def _gl_u0(x, y, z):
        vals = 0.1 * rng.standard_normal(x.shape)
        return vals.astype(complex)

    # ------------------------------------------------------------------
    # Swift-Hohenberg: u_t = -2*lap(u) - biharm(u) - 0.9*u - u^3
    # ------------------------------------------------------------------
    def _sh_u0(x, y, z):
        rng2 = np.random.default_rng(1)
        return 0.1 * rng2.standard_normal(x.shape)

    return {
        "gl": dict(
            # L = lap(u)  =>  lin_coeff(xi_x, xi_y, xi_z) = -(xi_x^2 + xi_y^2 + xi_z^2)
            lin_scales=(1.0,),           # (A,) for A*lap
            lin_ops=("lap",),
            nonlin_vals=lambda u: u - (1.0 + 1.5j) * u * jnp.abs(u) ** 2,
            domain=(0.0, 50.0, 0.0, 50.0, 0.0, 50.0),
            tspan=(0.0, 100.0),
            u0=_gl_u0,
            N=32,
            dt=1e-1,
            is_real=False,
        ),
        "sh": dict(
            # L = -2*lap(u) - biharm(u)
            lin_scales=(-2.0, -1.0),     # (A, B) for A*lap + B*biharm
            lin_ops=("lap", "biharm"),
            nonlin_vals=lambda u: -0.9 * u - u ** 3,
            domain=(0.0, 25.0, 0.0, 25.0, 0.0, 25.0),
            tspan=(0.0, 800.0),
            u0=_sh_u0,
            N=32,
            dt=5e-2,
            is_real=True,
        ),
        "ac": dict(
            # 3D Allen-Cahn: u_t = 5e-3*lap(u) + u - u^3
            lin_scales=(5e-3,),
            lin_ops=("lap",),
            nonlin_vals=lambda u: u - u ** 3,
            domain=(0.0, 2.0 * jnp.pi, 0.0, 2.0 * jnp.pi, 0.0, 2.0 * jnp.pi),
            tspan=(0.0, 20.0),
            u0=lambda x, y, z: jnp.sin(x) * jnp.cos(y) * jnp.cos(z),
            N=32,
            dt=5e-2,
            is_real=True,
        ),
    }


_BUILTIN_PDES_3D = _make_builtin_pdes_3d()


# ---------------------------------------------------------------------------
# SpinOp3 class
# ---------------------------------------------------------------------------


class SpinOp3:
    """Operator for a 3-D periodic semilinear PDE u_t = L[u] + N[u].

    The PDE is discretized on a uniform 3-D Fourier grid of N×N×N points on
    the periodic domain [x0,x1]×[y0,y1]×[z0,z1].  The linear part L is
    diagonal in Fourier space (stored as an N×N×N tensor).

    Attributes
    ----------
    lin_scales : tuple of float or complex
        Scalar prefactors (A, B, C, ...) for the linear operator.
        The linear part has the form
        ``A*lap + B*biharm + C*triharm + D*quadharm + E*quintharm``.
    lin_ops : tuple of str
        Operator names (``'lap'``, ``'biharm'``, ``'triharm'``,
        ``'quadharm'``, ``'quintharm'``) corresponding to ``lin_scales``.
        Must have the same length as ``lin_scales``.
    nonlin_vals : callable(u_vals) -> array
        Nonlinear part evaluated in physical (value) space.
        For GL: ``lambda u: u - (1+1.5j)*u*jnp.abs(u)**2``.
    domain : tuple of 6 floats
        ``(x0, x1, y0, y1, z0, z1)`` — the spatial domain.
    tspan : tuple[float, float]
        Time interval ``(t0, tf)``.
    u0 : callable(x, y, z) -> array
        Initial condition as a function of a 3-D grid.
    is_real : bool
        If True the solution is real-valued.

    Notes
    -----
    The linear operator is restricted to combinations of Laplacian and its
    iterated powers (biharmonic = lap^2, triharmonic = lap^3, etc.), so that
    the diagonal is:

        L_k = A * lap_k + B * lap_k^2 + C * lap_k^3 + D * lap_k^4 + E * lap_k^5

    where ``lap_k = -(xi_x^2 + xi_y^2 + xi_z^2)`` and ``xi_x``, ``xi_y``,
    ``xi_z`` are the scaled Fourier angular frequencies in each direction.

    The nonlinear part N(u) must be a pointwise (zero-differentiation) operator
    in physical space.

    Provenance
    ----------
    MATLAB source : @spinop3/spinop3.m, @spinop3/discretize.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    spin3
    """

    # Supported operator names (in order of polynomial degree in lap)
    _OP_DEGREE: dict = {
        "lap": 1,
        "biharm": 2,
        "biharmonic": 2,
        "triharm": 3,
        "triharmonic": 3,
        "quadharm": 4,
        "quadharmonic": 4,
        "quintharm": 5,
        "quintharmonic": 5,
    }

    def __init__(
        self,
        lin_scales: Sequence,
        lin_ops: Sequence[str],
        nonlin_vals: Callable,
        domain: Tuple[float, float, float, float, float, float],
        tspan: Tuple[float, float],
        u0: Callable,
        is_real: bool = True,
    ) -> None:
        self.lin_scales = tuple(lin_scales)
        self.lin_ops = tuple(op.lower() for op in lin_ops)
        self.nonlin_vals = nonlin_vals
        self.domain = domain
        self.tspan = tspan
        self.u0 = u0
        self.is_real = is_real

        if len(self.lin_scales) != len(self.lin_ops):
            raise ValueError(
                "lin_scales and lin_ops must have the same length."
            )
        for op in self.lin_ops:
            if op not in self._OP_DEGREE:
                raise ValueError(
                    f"Unknown linear operator {op!r}. "
                    f"Supported: {sorted(self._OP_DEGREE)}."
                )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_name(cls, name: str) -> "SpinOp3":
        """Construct a SpinOp3 from a built-in PDE name.

        Parameters
        ----------
        name : str
            Case-insensitive PDE identifier.  Supported values:

            * ``'GL'``  — complex Ginzburg-Landau equation
            * ``'SH'``  — Swift-Hohenberg equation
            * ``'AC'``  — Allen-Cahn equation (3D)

        Returns
        -------
        SpinOp3

        Raises
        ------
        ValueError
            If *name* is not recognised.

        Provenance
        ----------
        MATLAB source : @spinop3/spinop3.m (parseInputs)
        Chebfun commit: 7574c77
        """
        key = name.lower()
        if key not in _BUILTIN_PDES_3D:
            supported = ", ".join(k.upper() for k in _BUILTIN_PDES_3D)
            raise ValueError(
                f"Unrecognised 3D PDE name {name!r}. "
                f"Supported names (case-insensitive): {supported}."
            )
        d = _BUILTIN_PDES_3D[key]
        return cls(
            lin_scales=d["lin_scales"],
            lin_ops=d["lin_ops"],
            nonlin_vals=d["nonlin_vals"],
            domain=d["domain"],
            tspan=d["tspan"],
            u0=d["u0"],
            is_real=d["is_real"],
        )

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def default_N(self, name: Optional[str] = None) -> int:
        """Default number of grid points per dimension (if built-in)."""
        if name is not None:
            key = name.lower()
            if key in _BUILTIN_PDES_3D:
                return _BUILTIN_PDES_3D[key]["N"]
        return 32

    def default_dt(self, name: Optional[str] = None) -> float:
        """Default time-step for this PDE (if built-in)."""
        if name is not None:
            key = name.lower()
            if key in _BUILTIN_PDES_3D:
                return _BUILTIN_PDES_3D[key]["dt"]
        return 1e-2

    # ------------------------------------------------------------------
    # Discretization helpers
    # ------------------------------------------------------------------

    def build_linear_eigenvalues(self, N: int) -> np.ndarray:
        """Compute the diagonal of the linear operator in 3-D Fourier space.

        Returns an N×N×N array L_tensor where

            L_tensor[i,j,k] = A*lap_{ijk} + B*lap_{ijk}^2 + ...

        and ``lap_{ijk} = -(xi_x[j]^2 + xi_y[i]^2 + xi_z[k]^2)``

        (following MATLAB Chebfun's Kronecker structure: the x-direction
        varies along the *column* axis in ``meshgrid`` convention, matching
        trigpts → meshgrid ordering).

        Parameters
        ----------
        N : int
            Number of grid points per dimension.

        Returns
        -------
        L_tensor : np.ndarray, shape (N, N, N), complex
            Diagonal of the linear operator.

        Provenance
        ----------
        MATLAB source : @spinop3/discretize.m
        Chebfun commit: 7574c77
        """
        x0, x1, y0, y1, z0, z1 = [float(c) for c in self.domain]
        Lx = x1 - x0
        Ly = y1 - y0
        Lz = z1 - z0

        # Fourier angular frequencies (FFT ordering, 0-indexed)
        freqs = np.fft.fftfreq(N, d=1.0 / N)  # integer wavenumbers
        xi_x = (2.0 * np.pi / Lx) * freqs   # (N,)
        xi_y = (2.0 * np.pi / Ly) * freqs   # (N,)
        xi_z = (2.0 * np.pi / Lz) * freqs   # (N,)

        # Broadcast to N×N×N (axis 0=y, axis 1=x, axis 2=z, matching meshgrid)
        # lap_tensor[i,j,k] = -(xi_y[i]^2 + xi_x[j]^2 + xi_z[k]^2)
        xi_y3 = xi_y[:, np.newaxis, np.newaxis]   # (N,1,1)
        xi_x3 = xi_x[np.newaxis, :, np.newaxis]   # (1,N,1)
        xi_z3 = xi_z[np.newaxis, np.newaxis, :]   # (1,1,N)
        lap_tensor = -(xi_x3 ** 2 + xi_y3 ** 2 + xi_z3 ** 2)  # (N,N,N)

        L_tensor = np.zeros((N, N, N), dtype=complex)
        for scale, op in zip(self.lin_scales, self.lin_ops):
            deg = self._OP_DEGREE[op]
            L_tensor += scale * (lap_tensor ** deg)

        return L_tensor

    def dealias_mask(self, N: int) -> np.ndarray:
        """Return the 3-D 2/3-rule dealiasing mask.

        Zeros Fourier modes with |k| in the top 1/3 in each dimension
        (following MATLAB Chebfun's getDealiasingIndexes for spinop3).

        Parameters
        ----------
        N : int
            Number of grid points per dimension.

        Returns
        -------
        mask : np.ndarray, shape (N, N, N), bool
            True for modes to keep, False for modes to zero.

        Provenance
        ----------
        MATLAB source : @spinop3/getDealiasingIndexes.m
        Chebfun commit: 7574c77
        """
        import math
        half = N // 2
        sixth = math.ceil(N / 6)
        # MATLAB (1-indexed): toOne = floor(N/2)+1-ceil(N/6) : floor(N/2)+ceil(N/6)
        # Convert to 0-indexed Python: lo = half - sixth, hi = half + sixth
        lo = half - sixth
        hi = half + sixth

        mask1d = np.ones(N, dtype=bool)
        mask1d[lo:hi] = False

        # 3D mask: zero if ANY dimension falls in the aliased band
        # (MATLAB uses idx(toOne, toOne, toOne) = 1, i.e. zero in all three)
        mask3d = np.ones((N, N, N), dtype=bool)
        mask3d[lo:hi, :, :] = False
        mask3d[:, lo:hi, :] = False
        mask3d[:, :, lo:hi] = False
        return mask3d

    def __repr__(self) -> str:
        d = self.domain
        t0, tf = self.tspan
        ops = " + ".join(
            f"{s}*{o}" for s, o in zip(self.lin_scales, self.lin_ops)
        )
        return (
            f"SpinOp3(domain=[{float(d[0]):.3g},{float(d[1]):.3g}]^3, "
            f"tspan=[{t0},{tf}], lin={ops})"
        )
