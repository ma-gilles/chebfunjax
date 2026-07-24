# uses-numpy: ETDRK4 Fourier time-stepping is a one-shot host-side solve per
# call (delegated to chebfunjax.spin.solver2d, which runs on numpy), and the
# output trig interpolant is evaluated with numpy FFT weights -- matching the
# numpy pattern of operators/spinop.py and operators/spinopsphere.py; no JIT,
# no device management.
"""Spinop2 -- stiff semilinear PDEs on a 2D periodic domain,
u_t = L u + N(u) with L a polynomial in the Laplacian, solved by
spin2() with the ETDRK4 exponential integrator on a 2D Fourier grid.

Added by Claude Fable 5 (spinop2 port).  The constructor surface
(named presets, ``func2str`` of the nonlinear part, domain/tspan
plumbing) mirrors @spinop2/spinop2.m; the actual time-stepping is the
golden-ref-tested ETDRK4 solver in :mod:`chebfunjax.spin.solver2d`.

Provenance
----------
MATLAB source : @spinop2/spinop2.m, spin2.m, @spinoperator/solvepde.m
    (ETDRK4, Kassam & Trefethen SISC 2005 contour-integral phi
    functions; Montanelli & Bootland 2D/3D exponential integrators)
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.spinopsphere import FuncHandle, func2str
from chebfunjax.spin.solver2d import spin2 as _core_spin2
from chebfunjax.spin.spinop2 import SpinOp2 as _CoreSpinOp2

__all__ = ["Spinop2", "spin2", "func2str", "FuncHandle"]


# ---------------------------------------------------------------------------
# Built-in PDE catalogue (constructor surface)
# ---------------------------------------------------------------------------


def _gl_init(x, y):
    """Deterministic smooth initial condition standing in for MATLAB's
    ``randnfun2(4, dom, 'trig')`` on the GL domain ``[0, 100]^2``.

    MATLAB seeds a band-limited random trig field; chebfunjax has no
    ``randnfun2``, so a deterministic low-frequency field is used
    instead.  The spin2 self-convergence test only compares two solves
    that share this same init, so determinism (not randomness) is what
    the port requires.
    """
    return (0.5 * jnp.cos(2 * jnp.pi * x / 100.0)
            + 0.5j * jnp.sin(2 * jnp.pi * y / 100.0)
            + 0.3 * jnp.cos(4 * jnp.pi * (x + y) / 100.0))


def _preset(pdechar: str):
    """Return the constructor fields for a named 2D PDE, mirroring
    @spinop2/spinop2.m parseInputs.

    Returns ``(domain, tspan, lin, nonlin, init, lin_coeffs,
    nonlin_vals, n_vars, is_real)`` where ``lin``/``nonlin`` are
    :class:`FuncHandle` objects carrying the MATLAB ``func2str`` text
    and the remaining fields feed the :mod:`chebfunjax.spin.solver2d`
    ETDRK4 numerics.

    Provenance
    ----------
    MATLAB source : @spinop2/spinop2.m (parseInputs)
    Chebfun commit: 7574c77
    """
    key = pdechar.upper()
    if key == "GL":
        # Ginzburg-Landau: u_t = lap(u) + u - (1+1.5i) u |u|^2 on [0,100]^2.
        lin = FuncHandle(None, "@(u)lap(u)")
        nonlin = FuncHandle(
            lambda u: u - (1.0 + 1.5j) * u * (abs(u) ** 2),
            "@(u)u-(1+1.5i)*u.*(abs(u).^2)",
        )
        def nonlin_vals(u):
            return u - (1.0 + 1.5j) * u * jnp.abs(u) ** 2
        return (
            (0.0, 100.0, 0.0, 100.0),   # domain
            (0.0, 100.0),               # tspan
            lin, nonlin, _gl_init,
            (1.0, 0.0, 0.0, 0.0, 0.0),  # lin_coeffs (A*lap)
            nonlin_vals, 1, False,
        )
    raise ValueError(
        f"Unrecognized PDE {pdechar!r}. Options: GL.")


# ---------------------------------------------------------------------------
# Spinop2
# ---------------------------------------------------------------------------


class Spinop2:
    """Spatial part S of a time-dependent 2D periodic PDE
    ``u_t = S(u) = L u + N(u)``.

    Construct from a named preset (``Spinop2('GL')``) or from an
    explicit domain and time interval (``Spinop2(dom, tspan)`` with
    ``dom = [ax, bx, ay, by]``) and then set ``.lin``, ``.nonlin`` and
    ``.init`` directly.

    Attributes
    ----------
    domain : tuple of four floats
        ``(ax, bx, ay, by)``.  Recursive indexing (MATLAB
        ``S.domain([2 4])``) is available via ordinary tuple indexing.
    lin : FuncHandle
        The linear part ``@(u) ...`` (textual form + placeholder).
    nonlin : FuncHandle
        The nonlinear part ``@(u) f(u)`` acting elementwise in value
        space; ``func2str(S.nonlin)`` returns its MATLAB string.
    tspan : tuple of floats
        Time interval ``(t0, tf)``.
    init : callable or None
        Initial condition ``u0(x, y)``.

    Provenance
    ----------
    MATLAB source : @spinop2/spinop2.m, @spinoperator/spinoperator.m
    Chebfun commit: 7574c77
    """

    def __init__(self, arg=None, tspan=None):
        self.domain = None
        self.tspan = None
        self.lin = None
        self.nonlin = None
        self.init = None
        # Numerics fields consumed by spin2() (private).
        self._lin_coeffs = None
        self._nonlin_vals = None
        self._n_vars = 1
        self._is_real = None
        if arg is None:
            return
        if isinstance(arg, str):
            (dom, tsp, lin, nonlin, init, lin_coeffs,
             nonlin_vals, n_vars, is_real) = _preset(arg)
            self.domain = dom
            self.tspan = tsp
            self.lin = lin
            self.nonlin = nonlin
            self.init = init
            self._lin_coeffs = lin_coeffs
            self._nonlin_vals = nonlin_vals
            self._n_vars = n_vars
            self._is_real = is_real
        else:
            # (dom, tspan) constructor: Spinop2([ax, bx, ay, by], [t0, tf]).
            self.domain = tuple(float(v) for v in arg)
            if tspan is not None:
                self.tspan = tuple(float(v) for v in tspan)

    @property
    def numVars(self) -> int:
        """Number of unknown functions (1 for the supported scalar PDEs).

        Provenance
        ----------
        MATLAB source : @spinoperator/spinoperator.m (get.numVars)
        Chebfun commit: 7574c77
        """
        return self._n_vars

    def __repr__(self) -> str:
        return f"Spinop2(domain={self.domain!r}, tspan={self.tspan!r})"


# ---------------------------------------------------------------------------
# Output trig interpolant
# ---------------------------------------------------------------------------


def _make_trig_interp(V, domain_pairs):
    """Build a callable periodic 2D trig interpolant reproducing the
    value grid ``V`` at its nodes.

    ``V[i, j]`` samples the (generally complex) solution on the
    equispaced periodic grid ``x_i = a_x + L_x i/N``,
    ``y_j = a_y + L_y j/N``.  The returned callable ``u(x, y)``
    evaluates the band-limited trigonometric interpolant

        u(x, y) = N^-2 sum_{p,q} C[p,q]
                    exp(i xi_p (x - a_x)) exp(i eta_q (y - a_y))

    with ``C = fft2(V)`` and ``xi_p = 2 pi k_p / L_x`` the FFT angular
    wavenumbers.  This mirrors @spinoperator/reshapeData.m building a
    ``chebfun2(..., 'trig')`` from the final value grid.
    """
    V = np.asarray(V)
    N = V.shape[0]
    C = np.fft.fftn(V)
    ks = np.fft.fftfreq(N, d=1.0 / N)  # integer wavenumbers
    (ax, bx), (ay, by) = domain_pairs
    xi = 2.0 * np.pi * ks / (bx - ax)
    eta = 2.0 * np.pi * ks / (by - ay)
    scale = 1.0 / (N * N)

    def ev(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        shape = np.broadcast_shapes(x.shape, y.shape)
        xf = np.broadcast_to(x, shape).reshape(-1)
        yf = np.broadcast_to(y, shape).reshape(-1)
        Ex = np.exp(1j * (xf[:, None] - ax) * xi[None, :])   # (npts, N)
        Ey = np.exp(1j * (yf[:, None] - ay) * eta[None, :])  # (npts, N)
        vals = np.einsum("np,pq,nq->n", Ex, C, Ey) * scale
        return vals.reshape(shape)

    return ev


# ---------------------------------------------------------------------------
# spin2 solver
# ---------------------------------------------------------------------------


def spin2(S: Spinop2, N: int, dt: float, *args, **kwargs):
    """Solve the 2D periodic PDE specified by ``S`` with ``N`` grid
    points per direction and time-step ``dt`` (MATLAB
    ``spin2(S, N, dt, 'plot', 'off')``); returns the solution at
    ``tspan(end)`` as a callable periodic trig interpolant ``u(x, y)``.

    The heavy lifting is the golden-ref-tested ETDRK4 solver in
    :func:`chebfunjax.spin.solver2d.spin2`; this wrapper adapts the
    :class:`Spinop2` constructor surface to it and wraps the returned
    value grid in a trigonometric interpolant.  Plotting arguments
    (``'plot', 'off'``) are accepted and ignored.

    Provenance
    ----------
    MATLAB source : spin2.m, @spinoperator/solvepde.m,
        @expinteg/{computeCoeffs,oneStep}.m
    Chebfun commit: 7574c77
    Algorithm: Kassam & Trefethen, SISC 26 (2005); Montanelli &
        Bootland, 2D/3D exponential integrators.
    """
    if S._lin_coeffs is None:
        raise ValueError(
            "Spinop2 has no numerics (construct from a preset, e.g. "
            "Spinop2('GL'), or set the linear/nonlinear parts).")
    if S.init is None:
        raise ValueError("Spinop2 has no initial condition (set S.init).")

    core = _CoreSpinOp2(
        lin_coeffs=S._lin_coeffs,
        nonlin_vals=S._nonlin_vals,
        n_vars=S._n_vars,
        domain=tuple(float(v) for v in S.domain),
        tspan=tuple(float(v) for v in S.tspan),
        u0=S.init,
        is_real=S._is_real,
    )
    _xx, _yy, _t, u_final = _core_spin2(core, N, dt)
    ax, bx, ay, by = core.domain
    return _make_trig_interp(u_final, [(ax, bx), (ay, by)])
