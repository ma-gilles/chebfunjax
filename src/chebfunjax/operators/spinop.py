# uses-numpy: FFT time-stepping is one-shot numpy per solve
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

import jax.numpy as jnp
import numpy as np

__all__ = ["Spinop", "spin"]

# RESOLVED (Fable 5 audit): the 'CH' preset used to blow up at t ~ 30.
# Root cause was NOT the integrator, resolution, dealiasing, or the
# transcription (all verified correct against @spinop/spinop.m): the
# ETDRK4 loop propagates the full complex Fourier state, and CH's
# linear part has a growth band (L = 1e-2 om^2 - 1e-5 om^4 > 0 for
# small om).  Because Nhat evaluates np.real(ifft(.)), the nonlinear
# saturation never sees the conjugate-antisymmetric part of the state
# (the imaginary part of u); seeded at roundoff it grows as exp(2.5 t)
# under the bare exponential propagator until it corrupts the real
# solution near t ~ 30.  Fixed by re-Hermitianizing the state each
# step (v = fft(real(ifft(v)))) -- a no-op in exact arithmetic.  See
# the time-stepping loop below and the spinop port test.


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
        if pdechar is not None:
            self._preset(pdechar)
            return
        self.domain = tuple(float(v) for v in domain)
        self.tspan = tuple(float(v) for v in tspan)
        self.lin_symbol = lin_symbol
        self.nonlin = nonlin
        self.init = init

    def _preset(self, name: str) -> None:
        name = name.upper()
        if name == "AC":
            # Allen-Cahn: u_t = 5e-3 u_xx + u - u^3 on [0, 2 pi]
            self.domain = (0.0, 2.0 * np.pi)
            self.tspan = (0.0, 500.0)
            self.lin_symbol = lambda om: -5e-3 * om ** 2
            self.nonlin = lambda v, om, dx: v - v ** 3
            self.init = lambda x: (
                np.tanh(2 * np.sin(x)) / 3
                - np.exp(-23.5 * (x - np.pi / 2) ** 2)
                + np.exp(-27 * (x - 4.2) ** 2)
                + np.exp(-38 * (x - 5.4) ** 2))
        elif name == "CH":
            # Cahn-Hilliard: u_t = -1e-2 (u_xx + 1e-3 u_xxxx)
            #                      + 1e-2 (u^3)_xx  on [-1, 1]
            self.domain = (-1.0, 1.0)
            self.tspan = (0.0, 100.0)
            self.lin_symbol = lambda om: -1e-2 * (
                -om ** 2 + 1e-3 * om ** 4)
            self.nonlin = lambda v, om, dx: 1e-2 * dx(v ** 3, 2)
            self.init = lambda x: (
                np.sin(4 * np.pi * x) ** 5 / 5
                - 4.0 / 5.0 * np.sin(np.pi * x))
        elif name == "BURG":
            # Viscous Burgers: u_t = 1e-3 u_xx - (u^2/2)_x on [-1, 1]
            self.domain = (-1.0, 1.0)
            self.tspan = (0.0, 20.0)
            self.lin_symbol = lambda om: -1e-3 * om ** 2
            self.nonlin = lambda v, om, dx: -0.5 * dx(v ** 2, 1)
            self.init = lambda x: (
                (1 - x ** 2) * np.exp(-30 * (x + 0.5) ** 2))
        elif name == "KDV":
            # Korteweg-de Vries: u_t = -u_xxx - (u^2/2)_x
            self.domain = (-np.pi, np.pi)
            self.tspan = (0.0, 0.03)
            self.lin_symbol = lambda om: 1j * om ** 3
            self.nonlin = lambda v, om, dx: -0.5 * dx(v ** 2, 1)
            A = 25.0
            B = 16.0
            self.init = lambda x: (
                3 * A ** 2 / np.cosh(0.5 * A * (x + 2)) ** 2
                + 3 * B ** 2 / np.cosh(0.5 * B * (x + 1)) ** 2)
        else:
            raise ValueError(f"unknown spinop preset {name!r}")


def spin(S: Spinop, n: int, dt: float):
    """Solve the Spinop's PDE with ETDRK4 on an n-point Fourier grid
    (MATLAB spin(S, N, dt, 'plot', 'off')): returns a trig chebfun
    of the solution at tspan(end).

    Provenance
    ----------
    MATLAB source : spin.m / @spinoperator/solvepde.m
        (ETDRK4, Kassam & Trefethen SISC 2005 contour-integral phi
        functions)
    Chebfun commit: 7574c77
    """
    a, b = S.domain
    P = b - a
    x = a + P * np.arange(n) / n
    k = np.fft.fftfreq(n, d=1.0 / n)          # integer wavenumbers
    om = 2.0 * np.pi * k / P                  # angular wavenumbers

    def dx_op(vals, order):
        return np.real(np.fft.ifft(
            (1j * om) ** order * np.fft.fft(vals)))

    u0 = S.init(x) if callable(S.init) else np.asarray(
        S.init(jnp.asarray(x)))
    v = np.fft.fft(np.asarray(u0, dtype=float))

    L = S.lin_symbol(om).astype(complex)
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)

    # contour-integral phi weights (Kassam-Trefethen)
    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * np.real(np.mean(
        (-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3,
        axis=1))
    f2 = dt * np.real(np.mean(
        (2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3,
        axis=1))

    # 2/3-rule dealiasing mask (MATLAB spin dealiases the
    # nonlinear evaluation; without it stiff nonlinearities like
    # Cahn-Hilliard's (u^3)_xx alias and blow up)
    dealias = np.abs(k) < n / 3.0

    def Nhat(vhat):
        vals = np.real(np.fft.ifft(vhat))
        out = np.fft.fft(S.nonlin(vals, om, dx_op))
        return np.where(dealias, out, 0.0)

    t0, t1 = S.tspan
    nsteps = int(round((t1 - t0) / dt))
    for _ in range(nsteps):
        Nv = Nhat(v)
        av = E2 * v + Q * Nv
        Na = Nhat(av)
        bv = E2 * v + Q * Na
        Nb = Nhat(bv)
        cv = E2 * av + Q * (2 * Nb - Nv)
        Nc = Nhat(cv)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        # Re-Hermitianize the Fourier state so u stays real.  This is a
        # no-op in exact arithmetic (v is conjugate-symmetric because the
        # symbol L and the nonlinear map are real and even), but it is
        # ESSENTIAL for PDEs whose linear part has a growth band, e.g.
        # Cahn-Hilliard's spinodal instability (L = 1e-2 om^2 - 1e-5 om^4
        # is positive for |om| < ~31).  The nonlinear term evaluates
        # np.real(ifft(.)), so it never sees the conjugate-ANTISYMMETRIC
        # component of v (the imaginary part of u).  Seeded at roundoff,
        # that component is therefore propagated by the bare linear
        # operator exp(dt L) with NO nonlinear saturation and grows like
        # exp(2.5 t) until it overflows into the real solution near t~30.
        # Projecting onto the real subspace each step removes it and
        # leaves the physical (already-real) solution untouched.
        v = np.fft.fft(np.real(np.fft.ifft(v)))

    u_vals = np.real(np.fft.ifft(v))

    from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, _Piece
    from chebfunjax.tech.trigtech import Trigtech
    tech = Trigtech.from_values(jnp.asarray(u_vals)).simplify()
    piece = _Piece(tech=tech, interval=(a, b))
    return Chebfun(funs=[piece], domain=Domain((a, b)))
