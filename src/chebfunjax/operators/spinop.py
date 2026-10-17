"""Spinop -- stiff semilinear periodic PDEs u_t = L u + N(u), solved
by spin() with the ETDRK4 exponential integrator on a Fourier grid.

Added by Claude Fable 5 (Big-Three directive, spinop family).

Provenance
----------
MATLAB source : @spinop/spinop.m, spin.m (Kassam & Trefethen
    ETDRK4 with contour-integral phi functions)
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers.
"""

from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp

from chebfunjax.operators._spin2_expinteg import (
    Expinteg,
    compute_coeffs,
    one_step,
    start_multistep,
)

__all__ = ["Spinop", "spin"]

# This operator API shares the JAX scheme tables, multistep startup, and
# stepping engine used by the 2D/3D public solvers. The named preset
# callbacks below are JAX-compatible; custom callbacks should be too.


# MATLAB @spinop/spinop.m preset operators (func2str form), used by
# ``Spinop.lin_str`` / ``Spinop.nonlin_str`` and by :meth:`Spinop.lin`.
_LIN_STRINGS = {
    "AC": "@(u)5e-3*diff(u,2)",
    "BURG": "@(u)1e-3*diff(u,2)",
    "CH": "@(u)-1e-2*(diff(u,2)+1e-3*diff(u,4))",
    "KDV": "@(u)-diff(u,3)",
    "KS": "@(u)-diff(u,2)-diff(u,4)",
    "NIKO": "@(u).1*diff(u,2)+diff(u,4)+diff(u,6)",
    "NLS": "@(u)1i*diff(u,2)",
}
_NONLIN_STRINGS = {
    "AC": "@(u)u-u.^3",
    "BURG": "@(u)-.5*diff(u.^2)",
    "CH": "@(u)1e-2*diff(u.^3,2)",
    "KDV": "@(u)-.5*diff(u.^2)",
    "KS": "@(u)-.5*diff(u.^2)",
    "NIKO": "@(u)-.5*diff(u.^2)",
    "NLS": "@(u)1i*abs(u).^2.*u",
}


class Spinop:
    """Semilinear PDE specification u_t = L u + N(u) on a periodic
    domain.

    Construct from a named preset (``Spinop('AC')``, ``Spinop('CH')``,
    ``Spinop('KdV')``, ``Spinop('Burg')``) or from explicit parts:
    ``Spinop(domain=..., tspan=..., lin_symbol=..., nonlin=...,
    init=...)`` where ``lin_symbol(omega)`` returns the Fourier
    symbol of L at (angular) wavenumbers ``omega`` and
    ``nonlin(values, omega, dx_op)`` maps grid values of u to grid
    values of N(u) (``dx_op(values, order)`` applies spectral
    derivatives).

    Provenance
    ----------
    MATLAB source : @spinop/spinop.m
    Chebfun commit: 7574c77
    """

    def __init__(self, pdechar: str | None = None, *, domain=None,
                 tspan=None, lin_symbol=None, nonlin=None,
                 init=None):
        self.preset = None
        self.lin_str = None
        self.nonlin_str = None
        if pdechar is not None:
            self.preset = pdechar
            self._preset(pdechar)
            return
        self.domain = tuple(float(v) for v in domain)
        self.tspan = tuple(float(v) for v in tspan)
        self.lin_symbol = lin_symbol
        self.nonlin = nonlin
        self.init = init

    def lin(self, u):
        """The linear part applied to a (periodic) Chebfun -- MATLAB
        ``S.lin(u)`` -- for the named presets."""
        name = (self.preset or "").upper()
        if name == "AC":
            return 5e-3 * u.diff(2)
        if name == "BURG":
            return 1e-3 * u.diff(2)
        if name == "CH":
            return -1e-2 * (u.diff(2) + 1e-3 * u.diff(4))
        if name == "KDV":
            return -u.diff(3)
        if name == "KS":
            return -u.diff(2) - u.diff(4)
        if name == "NIKO":
            return 0.1 * u.diff(2) + u.diff(4) + u.diff(6)
        if name == "NLS":
            return 1j * u.diff(2)
        raise ValueError("Spinop.lin: no preset linear operator")

    def _preset(self, name: str) -> None:
        name = name.upper()
        if name == "AC":
            # Allen-Cahn: u_t = 5e-3 u_xx + u - u^3 on [0, 2 pi]
            self.domain = (0.0, 2.0 * math.pi)
            self.tspan = (0.0, 500.0)
            self.lin_symbol = lambda om: -5e-3 * om ** 2
            self.nonlin = lambda v, om, dx: v - v ** 3
            self.init = lambda x: (
                jnp.tanh(2 * jnp.sin(x)) / 3
                - jnp.exp(-23.5 * (x - jnp.pi / 2) ** 2)
                + jnp.exp(-27 * (x - 4.2) ** 2)
                + jnp.exp(-38 * (x - 5.4) ** 2))
        elif name == "CH":
            # Cahn-Hilliard: u_t = -1e-2 (u_xx + 1e-3 u_xxxx)
            #                      + 1e-2 (u^3)_xx  on [-1, 1]
            self.domain = (-1.0, 1.0)
            self.tspan = (0.0, 100.0)
            self.lin_symbol = lambda om: -1e-2 * (
                -om ** 2 + 1e-3 * om ** 4)
            self.nonlin = lambda v, om, dx: 1e-2 * dx(v ** 3, 2)
            self.init = lambda x: (
                jnp.sin(4 * jnp.pi * x) ** 5 / 5
                - 4.0 / 5.0 * jnp.sin(jnp.pi * x))
        elif name == "BURG":
            # Viscous Burgers: u_t = 1e-3 u_xx - (u^2/2)_x on [-1, 1]
            self.domain = (-1.0, 1.0)
            self.tspan = (0.0, 20.0)
            self.lin_symbol = lambda om: -1e-3 * om ** 2
            self.nonlin = lambda v, om, dx: -0.5 * dx(v ** 2, 1)
            self.init = lambda x: (
                (1 - x ** 2) * jnp.exp(-30 * (x + 0.5) ** 2))
        elif name == "KDV":
            # Korteweg-de Vries: u_t = -u_xxx - (u^2/2)_x
            self.domain = (-math.pi, math.pi)
            self.tspan = (0.0, 0.03)
            self.lin_symbol = lambda om: 1j * om ** 3
            self.nonlin = lambda v, om, dx: -0.5 * dx(v ** 2, 1)
            A = 25.0
            B = 16.0
            self.init = lambda x: (
                3 * A ** 2 / jnp.cosh(0.5 * A * (x + 2)) ** 2
                + 3 * B ** 2 / jnp.cosh(0.5 * B * (x + 1)) ** 2)
        elif name == "KS":
            # Kuramoto-Sivashinsky: u_t = -u_xx - u_xxxx - (u^2/2)_x on [0, 32pi]
            self.domain = (0.0, 32.0 * math.pi)
            self.tspan = (0.0, 300.0)
            self.lin_symbol = lambda om: om ** 2 - om ** 4
            self.nonlin = lambda v, om, dx: -0.5 * dx(v ** 2, 1)
            self.init = lambda x: jnp.cos(x / 16) * (1 + jnp.sin(x / 16))
        elif name == "NIKO":
            # Nikolaevskiy: u_t = .1 u_xx + u_xxxx + u_xxxxxx - (u^2/2)_x
            self.domain = (0.0, 32.0 * math.pi)
            self.tspan = (0.0, 300.0)
            self.lin_symbol = lambda om: -0.1 * om ** 2 + om ** 4 - om ** 6
            self.nonlin = lambda v, om, dx: -0.5 * dx(v ** 2, 1)
            self.init = lambda x: jnp.cos(x / 16) * (1 + jnp.sin(x / 16))
        elif name == "NLS":
            # Nonlinear Schroedinger: u_t = i u_xx + i |u|^2 u on [-pi, pi]
            self.domain = (-math.pi, math.pi)
            self.tspan = (0.0, 20.0)
            self.lin_symbol = lambda om: -1j * om ** 2
            self.nonlin = lambda v, om, dx: 1j * jnp.abs(v) ** 2 * v
            A = 2.0
            B = 1.0
            self.init = lambda x: ((2 * B ** 2 / (2 + A ** 2 * jnp.cos(B * x)
                                                 / (2 * B ** 2)))
                                   * (1 + 0j) - 1.0)
        else:
            raise ValueError(f"unknown spinop preset {name!r}")
        self.lin_str = _LIN_STRINGS.get(name)
        self.nonlin_str = _NONLIN_STRINGS.get(name)


def _parse_scheme(args, kwargs, scheme):
    """MATLAB-style trailing ('plot', 'off', 'scheme', name) pairs or the
    ``scheme=`` keyword; plotting options are accepted and ignored."""
    if scheme is not None:
        return str(scheme).lower()
    if "scheme" in kwargs and kwargs["scheme"] is not None:
        return str(kwargs["scheme"]).lower()
    a = list(args)
    for i in range(len(a) - 1):
        if isinstance(a[i], str) and a[i].lower() == "scheme":
            return str(a[i + 1]).lower()
    return None


def spin(S: Spinop, n: int, dt: float, *args, scheme: str | None = None,
         M: int = 32, return_computing_time: bool = False, **kwargs):
    """Solve the Spinop's PDE on an n-point Fourier grid (MATLAB
    ``spin(S, N, dt, 'plot', 'off')``): returns a trig chebfun of the
    solution at ``tspan(end)``.

    Options (MATLAB trailing pairs or keywords): ``scheme=`` any
    expinteg scheme name (default ETDRK4; see the JAX engine in
    :mod:`chebfunjax.operators._spin2_expinteg`), ``dealias=`` the
    2/3-rule projection (chebfunjax default on; MATLAB's default is
    ``'off'``), ``M`` contour points for the phi functions, and
    ``return_computing_time=True`` to also return the stepping-loop time.
    Custom initial, linear-symbol and nonlinear callbacks should use
    JAX-compatible operations.

    Provenance
    ----------
    MATLAB source : spin.m / @spinoperator/solvepde.m
    Chebfun commit: 7574c77
    """
    a, b = (float(value) for value in S.domain)
    t0, tf = (float(value) for value in S.tspan)
    if n < 1 or dt <= 0 or b <= a or tf <= t0:
        raise ValueError("spin requires positive N/dt and increasing domain/tspan.")

    x = a + (b - a) * jnp.arange(n, dtype=jnp.float64) / n
    k = jnp.fft.fftfreq(n, d=1.0 / n)
    omega = 2.0 * jnp.pi * k / (b - a)
    initial = jnp.asarray(S.init(x) if callable(S.init) else S.init)
    if initial.shape == ():
        initial = jnp.broadcast_to(initial, (n,))
    if initial.shape != (n,):
        raise ValueError("Initial-condition callback must return shape (N,).")

    linear = jnp.asarray(S.lin_symbol(omega), dtype=jnp.complex128)
    if linear.shape == ():
        linear = jnp.broadcast_to(linear, (n,))
    if linear.shape != (n,):
        raise ValueError("Linear-symbol callback must return shape (N,).")
    if n % 2 == 0:
        nyquist = n // 2
        even_symbol = 0.5 * (linear[nyquist] +
                             jnp.asarray(S.lin_symbol(-omega))[nyquist])
        linear = linear.at[nyquist].set(even_symbol)
    is_real = bool(jnp.all(jnp.imag(linear) == 0))
    if is_real:
        linear = jnp.real(linear)

    def dx_op(values, order):
        factor = (1j * omega) ** order
        if n % 2 == 0 and order % 2:
            factor = factor.at[n // 2].set(0.0)
        return jnp.fft.ifft(factor * jnp.fft.fft(values))

    def nonlinear(values):
        result = jnp.asarray(S.nonlin(values, omega, dx_op))
        if result.shape == ():
            result = jnp.broadcast_to(result, values.shape)
        if result.shape != values.shape:
            raise ValueError("Nonlinear callback must return shape (N,).")
        return result

    def c2v(coefficients):
        return jnp.fft.ifft(coefficients)

    def v2c(values):
        return jnp.fft.fft(values)

    # Preserve the Python API's historical dealias default. MATLAB's
    # spincomp creates a fresh spinpref (dealias off) and copies only M.
    dealias = kwargs.get("dealias", True)
    for index in range(len(args) - 1):
        if isinstance(args[index], str) and args[index].lower() == "dealias":
            dealias = str(args[index + 1]).lower() in ("on", "true", "1")
    output_mask = ((jnp.abs(k) < n / 3.0) if dealias
                   else jnp.ones(n, dtype=bool))

    def N_v(values):
        return nonlinear(values)

    selected_scheme = _parse_scheme(args, kwargs, scheme) or "etdrk4"
    K = Expinteg(selected_scheme)
    initial_coefficients = v2c(initial)
    nonlinear_coefficients = v2c(N_v(initial))
    if K.steps == 1:
        states, nonlinear_states = [initial_coefficients], [nonlinear_coefficients]
    else:
        states, nonlinear_states = start_multistep(
            K, dt, linear, M, is_real, 1, N_v, c2v, v2c, 1,
            initial_coefficients, nonlinear_coefficients)
    coefficients = compute_coeffs(K, dt, linear, M, is_real)

    # Native solvepde keeps the initial values as the first output checkpoint.
    # Multistep startup states are internal and replace this output only when
    # the literal time loop reaches a requested tspan checkpoint.
    u_vals = initial
    step = K.steps - 1
    t = step * dt
    endpoint_saved = False
    if return_computing_time:
        for leaf in jax.tree_util.tree_leaves((states, nonlinear_states,
                                               coefficients)):
            block = getattr(leaf, "block_until_ready", None)
            if block is not None:
                block()
        loop_start = time.perf_counter()
    while t < tf:
        states, nonlinear_states = one_step(
            K, coefficients, 1, N_v, c2v, v2c, 1,
            states, nonlinear_states)
        if bool(jnp.any(jnp.isnan(states[0]))):
            raise RuntimeError("The solution blew up. Try a smaller time-step.")
        step += 1
        t = step * dt
        if abs(t - tf) < 1e-10:
            # Native dealiasing masks only the saved output after retaining
            # the unfiltered state for the next step.
            u_coefficients = jnp.where(output_mask, states[0], 0.0)
            u_vals = c2v(u_coefficients)
            if (jnp.max(jnp.abs(jnp.imag(u_vals)))
                    < jnp.max(jnp.abs(u_vals)) * 1e-10):
                u_vals = jnp.real(u_vals)
            endpoint_saved = True
            break
    if return_computing_time and tf > t0 and not endpoint_saved:
        # spincomp requests solvepde's third output; native solvepde then
        # indexes tOut(2), which is absent unless the endpoint was saved.
        raise IndexError("native solvepde has no saved tOut(2) checkpoint")
    if return_computing_time:
        for leaf in jax.tree_util.tree_leaves((states, u_vals)):
            block = getattr(leaf, "block_until_ready", None)
            if block is not None:
                block()
        computing_time = time.perf_counter() - loop_start
    if (jnp.max(jnp.abs(jnp.imag(u_vals)))
            < jnp.max(jnp.abs(u_vals)) * 1e-10):
        u_vals = jnp.real(u_vals)

    from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, _Piece
    from chebfunjax.tech.trigtech import Trigtech
    tech = Trigtech.from_values(jnp.asarray(u_vals)).simplify()
    piece = _Piece(tech=tech, interval=(a, b))
    result = Chebfun(funs=[piece], domain=Domain((a, b)))
    return (result, computing_time) if return_computing_time else result
