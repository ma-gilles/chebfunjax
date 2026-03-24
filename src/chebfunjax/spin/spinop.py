"""SpinOp — 1D periodic PDE operator for the Spin framework.

Translated from MATLAB Chebfun class @spinop (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Built-in PDE definitions
# ---------------------------------------------------------------------------

def _make_builtin_pdes() -> dict:
    """Return the catalogue of built-in 1-D periodic PDEs.

    Each entry maps a (case-insensitive) key to a dict with keys:
      lin      : callable(u_hat, k) -> linear part applied to Fourier coeffs,
                 here stored as a function that returns the *diagonal of the
                 linear operator* as a function of wavenumber vector k.
      nonlin   : callable(u_vals) -> nonlinear part evaluated in value space
      nonlin_k : callable(k) -> diagonal of the differentiation factor for the
                 nonlinear part in coefficient space (or None for no diff)
      domain   : (a, b)
      tspan    : (t0, tf)
      u0       : callable(x) -> initial condition values
      N        : default number of Fourier modes
      dt       : default time-step
      is_real  : True if the solution is real-valued
    """
    # Allen-Cahn: u_t = 5e-3*u_xx + u - u^3
    # lin = 5e-3 * (ik)^2 = -5e-3 * k^2
    # nonlin (vals) = u - u^3   (no diff, so nonlin_k = None)
    def _ac_u0(x):
        return (1.0 / 3.0 * jnp.tanh(2.0 * jnp.sin(x))
                - jnp.exp(-23.5 * (x - jnp.pi / 2.0) ** 2)
                + jnp.exp(-27.0 * (x - 4.2) ** 2)
                + jnp.exp(-38.0 * (x - 5.4) ** 2))

    # KdV: u_t = -u_xxx - 0.5*(u^2)_x
    # lin = -(ik)^3  = -i*k^3 (complex)  ->  eigenvalue of d^3/dx^3 on [-pi,pi]
    # nonlin (vals) = 0.5*u^2   (nonlin_k = first-order diff)
    def _kdv_u0(x):
        A, B = 25.0, 16.0
        return (3.0 * A ** 2 * (1.0 / jnp.cosh(0.5 * A * (x + 2.0))) ** 2
                + 3.0 * B ** 2 * (1.0 / jnp.cosh(0.5 * B * (x + 1.0))) ** 2)

    # NLS: u_t = i*u_xx + i*|u|^2*u
    def _nls_u0(x):
        A, B = 1.0, 1.0
        return (2.0 * B ** 2 / (2.0 - jnp.sqrt(2.0) * jnp.sqrt(2.0 - B ** 2)
                                  * jnp.cos(A * B * x)) - 1.0) * A + 0j

    return {
        # key       : (lin_diag_fn, nonlin_vals_fn, nonlin_diff_order,
        #              domain, tspan, u0, N, dt, is_real)
        "ac": dict(
            # lin part: L_k = 5e-3 * (i*pi*k)^2  (domain scaling: k*pi/L)
            # Below, k is the wavenumber index array on domain [0, 2*pi],
            # so the Fourier frequencies are integer wavenumbers and d/dx
            # has eigenvalue i*k for the standard trig convention.
            lin_coeff=lambda k: 5e-3 * (1j * k) ** 2,
            nonlin_vals=lambda u: u - u ** 3,
            nonlin_diff_order=0,  # no extra diff on nonlinear term
            domain=(0.0, 2 * jnp.pi),
            tspan=(0.0, 500.0),
            u0=_ac_u0,
            N=256,
            dt=1e-1,
            is_real=True,
        ),
        "kdv": dict(
            # KdV: u_t = -u_xxx + N(u) = -u_xxx - 0.5*(u^2)_x
            # L_k = -(i*xi)^3 = i*xi^3  (dispersive, purely imaginary)
            # nonlin_vals(u) = -0.5*u^2 (value-space factor, includes sign)
            # Nc_k = (i*xi)^1  (first-order differentiation in coeff space)
            # Total: Nc * fft(nonlin_vals(u)) = (i*xi)*fft(-0.5*u^2) = -0.5*(u^2)_x
            lin_coeff=lambda k: -(1j * k) ** 3,
            nonlin_vals=lambda u: -0.5 * u ** 2,
            nonlin_diff_order=1,  # differentiate once in coefficient space
            domain=(-jnp.pi, jnp.pi),
            tspan=(0.0, 0.03015),
            u0=_kdv_u0,
            N=512,
            dt=3e-6,
            is_real=True,
        ),
        "nls": dict(
            lin_coeff=lambda k: 1j * (1j * k) ** 2,
            nonlin_vals=lambda u: 1j * jnp.abs(u) ** 2 * u,
            nonlin_diff_order=0,
            domain=(-jnp.pi, jnp.pi),
            tspan=(0.0, 20.0),
            u0=_nls_u0,
            N=256,
            dt=1e-3,
            is_real=False,
        ),
        "ks": dict(
            # Kuramoto-Sivashinsky: u_t = -u_xx - u_xxxx - 0.5*(u^2)_x
            lin_coeff=lambda k: -(1j * k) ** 2 - (1j * k) ** 4,
            nonlin_vals=lambda u: -0.5 * u ** 2,
            nonlin_diff_order=1,
            domain=(0.0, 32 * jnp.pi),
            tspan=(0.0, 200.0),
            u0=lambda x: jnp.cos(x / 16.0) * (1.0 + jnp.sin((x - 1.0) / 16.0)),
            N=256,
            dt=1e-2,
            is_real=True,
        ),
    }


_BUILTIN_PDES = _make_builtin_pdes()


# ---------------------------------------------------------------------------
# SpinOp class
# ---------------------------------------------------------------------------


class SpinOp:
    """Operator for a 1-D periodic semilinear PDE u_t = L[u] + N[u].

    The PDE is discretized on a uniform Fourier grid of N points on the
    periodic domain [a, b]. The linear part L is diagonal in Fourier space.

    Attributes
    ----------
    lin_coeff : callable(k) -> array
        Returns the diagonal of the linear operator as a function of the
        wavenumber array k (integer wavenumbers in the FFT ordering).
        Example: ``lambda k: -(1j*k)**3`` for the KdV dispersive term.
    nonlin_vals : callable(u_vals) -> array
        Nonlinear part evaluated in physical (value) space.
        For KdV: ``lambda u: 0.5 * u**2``.
    nonlin_diff_order : int
        Differentiation order to apply to ``nonlin_vals(u)`` in Fourier space.
        For KdV the nonlinearity is ``-0.5*(u^2)_x``, so ``nonlin_diff_order=1``
        and ``nonlin_vals`` returns ``0.5*u^2``.
        For Allen-Cahn ``u - u^3`` (no diff): ``nonlin_diff_order=0``.
    domain : tuple[float, float]
        Spatial domain ``(a, b)``.
    tspan : tuple[float, float]
        Time interval ``(t0, tf)``.
    u0 : callable(x) -> array
        Initial condition as a function of x.
    is_real : bool
        If True the solution is real-valued; imaginary parts are numerical
        noise and will be discarded in the output.

    Notes
    -----
    The sign convention matches Chebfun's ``@spinop``:
    the PDE is ``u_t = L(u) + N(u)`` where both L and N include their signs.
    For KdV, the user writes ``L = @(u) -diff(u,3)`` and
    ``N = @(u) -0.5*diff(u.^2)``, so both terms are supplied with minus signs.

    Provenance
    ----------
    MATLAB source : @spinop/spinop.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    spin
    """

    def __init__(
        self,
        lin_coeff: Callable,
        nonlin_vals: Callable,
        nonlin_diff_order: int,
        domain: Tuple[float, float],
        tspan: Tuple[float, float],
        u0: Callable,
        is_real: bool = True,
    ) -> None:
        self.lin_coeff = lin_coeff
        self.nonlin_vals = nonlin_vals
        self.nonlin_diff_order = nonlin_diff_order
        self.domain = domain
        self.tspan = tspan
        self.u0 = u0
        self.is_real = is_real

    @classmethod
    def from_name(cls, name: str) -> "SpinOp":
        """Construct a SpinOp from a built-in PDE name.

        Parameters
        ----------
        name : str
            Case-insensitive PDE identifier.  Supported values:

            * ``'AC'``  — Allen-Cahn equation
            * ``'KdV'`` — Korteweg-de Vries equation
            * ``'NLS'`` — nonlinear Schrödinger equation
            * ``'KS'``  — Kuramoto-Sivashinsky equation

        Returns
        -------
        SpinOp

        Raises
        ------
        ValueError
            If *name* is not recognised.

        Examples
        --------
        >>> op = SpinOp.from_name('KdV')

        Provenance
        ----------
        MATLAB source : @spinop/spinop.m (parseInputs)
        Chebfun commit: 7574c77
        """
        key = name.lower()
        if key not in _BUILTIN_PDES:
            supported = ", ".join(_BUILTIN_PDES.keys())
            raise ValueError(
                f"Unrecognised PDE name {name!r}. "
                f"Supported names (case-insensitive): {supported}."
            )
        d = _BUILTIN_PDES[key]
        return cls(
            lin_coeff=d["lin_coeff"],
            nonlin_vals=d["nonlin_vals"],
            nonlin_diff_order=d["nonlin_diff_order"],
            domain=d["domain"],
            tspan=d["tspan"],
            u0=d["u0"],
            is_real=d["is_real"],
        )

    def default_N(self, name: Optional[str] = None) -> int:
        """Default number of Fourier modes for this PDE (if built-in)."""
        if name is not None:
            key = name.lower()
            if key in _BUILTIN_PDES:
                return _BUILTIN_PDES[key]["N"]
        return 256

    def default_dt(self, name: Optional[str] = None) -> float:
        """Default time-step for this PDE (if built-in)."""
        if name is not None:
            key = name.lower()
            if key in _BUILTIN_PDES:
                return _BUILTIN_PDES[key]["dt"]
        return 1e-3

    def __repr__(self) -> str:
        a, b = self.domain
        t0, tf = self.tspan
        return (
            f"SpinOp(domain=[{float(a):.4g}, {float(b):.4g}], "
            f"tspan=[{t0}, {tf}], "
            f"nonlin_diff_order={self.nonlin_diff_order})"
        )
