# uses-numpy: ETDRK4 Fourier time-stepping is a one-shot host-side solve per
# call (delegated to chebfunjax.spin.solver3, which runs on numpy), and the
# output trig interpolant is evaluated with numpy FFT weights -- matching the
# numpy pattern of operators/spinop.py and operators/spinopsphere.py; no JIT,
# no device management.
"""Spinop3 -- stiff semilinear PDEs on a 3D periodic domain,
u_t = L u + N(u) with L a polynomial in the Laplacian, solved by
spin3() with the ETDRK4 exponential integrator on a 3D Fourier grid.

Added by Claude Fable 5 (spinop3 port).  The constructor surface
(named presets, ``func2str`` of the nonlinear part, domain/tspan
plumbing) mirrors @spinop3/spinop3.m; the actual time-stepping is the
golden-ref-tested ETDRK4 solver in :mod:`chebfunjax.spin.solver3`.

Provenance
----------
MATLAB source : @spinop3/spinop3.m, spin3.m, @spinoperator/solvepde.m
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
from chebfunjax.spin.solver3 import spin3 as _core_spin3
from chebfunjax.spin.spinop3 import SpinOp3 as _CoreSpinOp3

__all__ = ["Spinop3", "spin3", "func2str", "FuncHandle"]


# ---------------------------------------------------------------------------
# Built-in PDE catalogue (constructor surface)
# ---------------------------------------------------------------------------


def _gl_init(x, y, z):
    """Deterministic smooth initial condition standing in for MATLAB's
    random ``chebfun3(vals, dom, 'trig')`` on the GL domain
    ``[0, 50]^3``.

    chebfunjax has no random 3D trig field constructor, so a
    deterministic low-frequency field is used.  The spin3
    self-convergence test only compares two solves that share this same
    init, so determinism (not randomness) is what the port requires.
    """
    return (0.5 * jnp.cos(2 * jnp.pi * x / 50.0)
            + 0.5j * jnp.sin(2 * jnp.pi * y / 50.0)
            + 0.3 * jnp.cos(4 * jnp.pi * z / 50.0))


def _preset(pdechar: str):
    """Return the constructor fields for a named 3D PDE, mirroring
    @spinop3/spinop3.m parseInputs.

    Returns ``(domain, tspan, lin, nonlin, init, lin_scales,
    lin_ops, nonlin_vals, is_real)``.

    Provenance
    ----------
    MATLAB source : @spinop3/spinop3.m (parseInputs)
    Chebfun commit: 7574c77
    """
    key = pdechar.upper()
    if key == "GL":
        # Ginzburg-Landau: u_t = lap(u) + u - (1+1.5i) u |u|^2 on [0,50]^3.
        lin = FuncHandle(None, "@(u)lap(u)")
        nonlin = FuncHandle(
            lambda u: u - (1.0 + 1.5j) * u * (abs(u) ** 2),
            "@(u)u-(1+1.5i)*u.*(abs(u).^2)",
        )
        def nonlin_vals(u):
            return u - (1.0 + 1.5j) * u * jnp.abs(u) ** 2
        return (
            (0.0, 50.0, 0.0, 50.0, 0.0, 50.0),   # domain
            (0.0, 100.0),                         # tspan
            lin, nonlin, _gl_init,
            (1.0,), ("lap",),                     # lin_scales, lin_ops (A*lap)
            nonlin_vals, False,
        )
    raise ValueError(
        f"Unrecognized PDE {pdechar!r}. Options: GL.")


# ---------------------------------------------------------------------------
# Spinop3
# ---------------------------------------------------------------------------


class Spinop3:
    """Spatial part S of a time-dependent 3D periodic PDE
    ``u_t = S(u) = L u + N(u)``.

    Construct from a named preset (``Spinop3('GL')``) or from an
    explicit domain and time interval (``Spinop3(dom, tspan)`` with
    ``dom = [ax, bx, ay, by, az, bz]``) and then set ``.lin``,
    ``.nonlin`` and ``.init`` directly.

    Attributes
    ----------
    domain : tuple of six floats
        ``(ax, bx, ay, by, az, bz)``.  Recursive indexing (MATLAB
        ``S.domain([2 4 6])``) is available via ordinary tuple
        indexing.
    lin : FuncHandle
        The linear part ``@(u) ...`` (textual form + placeholder).
    nonlin : FuncHandle
        The nonlinear part ``@(u) f(u)`` acting elementwise in value
        space; ``func2str(S.nonlin)`` returns its MATLAB string.
    tspan : tuple of floats
        Time interval ``(t0, tf)``.
    init : callable or None
        Initial condition ``u0(x, y, z)``.

    Provenance
    ----------
    MATLAB source : @spinop3/spinop3.m, @spinoperator/spinoperator.m
    Chebfun commit: 7574c77
    """

    def __init__(self, arg=None, tspan=None):
        self.domain = None
        self.tspan = None
        self.lin = None
        self.nonlin = None
        self.init = None
        # Numerics fields consumed by spin3() (private).
        self._lin_scales = None
        self._lin_ops = None
        self._nonlin_vals = None
        self._is_real = None
        if arg is None:
            return
        if isinstance(arg, str):
            (dom, tsp, lin, nonlin, init, lin_scales, lin_ops,
             nonlin_vals, is_real) = _preset(arg)
            self.domain = dom
            self.tspan = tsp
            self.lin = lin
            self.nonlin = nonlin
            self.init = init
            self._lin_scales = lin_scales
            self._lin_ops = lin_ops
            self._nonlin_vals = nonlin_vals
            self._is_real = is_real
        else:
            # (dom, tspan) constructor.
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
        return 1

    def __repr__(self) -> str:
        return f"Spinop3(domain={self.domain!r}, tspan={self.tspan!r})"


# ---------------------------------------------------------------------------
# Output trig interpolant
# ---------------------------------------------------------------------------


def _make_trig_interp(V, domain_triples):
    """Build a callable periodic 3D trig interpolant reproducing the
    value grid ``V`` at its nodes.

    ``V`` samples the (generally complex) solution on the equispaced
    periodic grid of the cube; the returned ``u(x, y, z)`` evaluates the
    band-limited trigonometric interpolant

        u = N^-3 sum_{p,q,r} C[p,q,r]
              exp(i xi_p (x-a)) exp(i eta_q (y-b)) exp(i zeta_r (z-c))

    with ``C = fftn(V)``.  Mirrors @spinoperator/reshapeData.m building
    a ``chebfun3(..., 'trig')`` from the final value grid.
    """
    V = np.asarray(V)
    N = V.shape[0]
    C = np.fft.fftn(V)
    ks = np.fft.fftfreq(N, d=1.0 / N)
    (ax, bx), (ay, by), (az, bz) = domain_triples
    xi = 2.0 * np.pi * ks / (bx - ax)
    eta = 2.0 * np.pi * ks / (by - ay)
    zeta = 2.0 * np.pi * ks / (bz - az)
    scale = 1.0 / (N ** 3)

    def ev(x, y, z):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        shape = np.broadcast_shapes(x.shape, y.shape, z.shape)
        xf = np.broadcast_to(x, shape).reshape(-1)
        yf = np.broadcast_to(y, shape).reshape(-1)
        zf = np.broadcast_to(z, shape).reshape(-1)
        Ex = np.exp(1j * (xf[:, None] - ax) * xi[None, :])    # (npts, N)
        Ey = np.exp(1j * (yf[:, None] - ay) * eta[None, :])
        Ez = np.exp(1j * (zf[:, None] - az) * zeta[None, :])
        vals = np.einsum("np,pqr,nq,nr->n", Ex, C, Ey, Ez,
                         optimize=True) * scale
        return vals.reshape(shape)

    return ev


# ---------------------------------------------------------------------------
# spin3 solver
# ---------------------------------------------------------------------------


def spin3(S: Spinop3, N: int, dt: float, *args, **kwargs):
    """Solve the 3D periodic PDE specified by ``S`` with ``N`` grid
    points per direction and time-step ``dt`` (MATLAB
    ``spin3(S, N, dt, 'plot', 'off')``); returns the solution at
    ``tspan(end)`` as a callable periodic trig interpolant
    ``u(x, y, z)``.

    The heavy lifting is the golden-ref-tested ETDRK4 solver in
    :func:`chebfunjax.spin.solver3.spin3`; this wrapper adapts the
    :class:`Spinop3` constructor surface to it and wraps the returned
    value grid in a trigonometric interpolant.  Plotting arguments
    (``'plot', 'off'``) are accepted and ignored.

    Provenance
    ----------
    MATLAB source : spin3.m, @spinoperator/solvepde.m,
        @expinteg/{computeCoeffs,oneStep}.m
    Chebfun commit: 7574c77
    Algorithm: Kassam & Trefethen, SISC 26 (2005); Montanelli &
        Bootland, 2D/3D exponential integrators.
    """
    if S._lin_scales is None:
        raise ValueError(
            "Spinop3 has no numerics (construct from a preset, e.g. "
            "Spinop3('GL'), or set the linear/nonlinear parts).")
    if S.init is None:
        raise ValueError("Spinop3 has no initial condition (set S.init).")

    core = _CoreSpinOp3(
        lin_scales=S._lin_scales,
        lin_ops=S._lin_ops,
        nonlin_vals=S._nonlin_vals,
        domain=tuple(float(v) for v in S.domain),
        tspan=tuple(float(v) for v in S.tspan),
        u0=S.init,
        is_real=S._is_real,
    )
    _grids, _t, u_final = _core_spin3(core, N, dt)
    ax, bx, ay, by, az, bz = core.domain
    return _make_trig_interp(u_final, [(ax, bx), (ay, by), (az, bz)])
