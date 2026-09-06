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
            nonlin_vals, False, 1,
        )
    if key == "GS":
        # Gray-Scott on [0,0.5]^3 (MATLAB spinop3 'GS').
        F, K = 0.030, 0.057
        G = 0.5
        lin = FuncHandle(None, "@(u,v)[2e-5*lap(u);1e-5*lap(v)]")
        nonlin = FuncHandle(
            lambda u, v: (F * (1 - u) - u * v ** 2, -(F + K) * v + u * v ** 2),
            "@(u,v)[F*(1-u)-u.*v.^2;-(F+K)*v+u.*v.^2]",
        )

        def n1(u, v):
            return F * (1.0 - u) - u * v ** 2

        def n2(u, v):
            return -(F + K) * v + u * v ** 2

        def u01(x, y, z):
            return 1.0 - jnp.exp(-500.0 * ((x - G / 2.05) ** 2 + (y - G / 2.05) ** 2
                                           + (z - G / 2.15) ** 2))

        def u02(x, y, z):
            return jnp.exp(-500.0 * ((x - G / 2) ** 2 + 2 * (y - G / 2) ** 2
                                     + 2 * (z - G / 2) ** 2))

        return ((0.0, G, 0.0, G, 0.0, G), (0.0, 5000.0), lin, nonlin, [u01, u02],
                [(2e-5,), (1e-5,)], [("lap",), ("lap",)], [n1, n2], True, 2)
    if key == "SCHNAK":
        # Schnakenberg on [0,25]^3 (MATLAB spinop3 'SCHNAK').
        G = 25.0
        lin = FuncHandle(None, "@(u,v)[lap(u);10*lap(v)]")
        nonlin = FuncHandle(
            lambda u, v: (3 * (.1 - u + u ** 2 * v), 3 * (.9 - u ** 2 * v)),
            "@(u,v)[3*(.1-u+u.^2.*v);3*(.9-u.^2.*v)]",
        )

        def n1(u, v):
            return 3.0 * (0.1 - u + u ** 2 * v)

        def n2(u, v):
            return 3.0 * (0.9 - u ** 2 * v)

        def u01(x, y, z):
            return 1.0 - jnp.exp(-2.0 * ((x - G / 2.15) ** 2 + (y - G / 2.15) ** 2
                                         + (z - G / 2.15) ** 2))

        def u02(x, y, z):
            return (0.9 / (0.1 + 0.9) ** 2
                    + jnp.exp(-2.0 * ((x - G / 2) ** 2 + 2 * (y - G / 2) ** 2
                                      + 2 * (z - G / 2) ** 2)))

        return ((0.0, G, 0.0, G, 0.0, G), (0.0, 500.0), lin, nonlin, [u01, u02],
                [(1.0,), (10.0,)], [("lap",), ("lap",)], [n1, n2], True, 2)
    if key == "SH":
        # Swift-Hohenberg on [0,25]^3: u_t = -2 lap(u) - biharm(u) - .9 u - u^3.
        # MATLAB seeds u0 = chebfun3(.1*randn(32,32,32), dom, 'trig'); the
        # trig interpolant of a fixed-seed normal field stands in.
        G = 25.0
        dom = (0.0, G, 0.0, G, 0.0, G)
        lin = FuncHandle(None, "@(u)-2*lap(u)-biharm(u)")
        nonlin = FuncHandle(lambda u: -.9 * u - u ** 3, "@(u)-.9*u-u.^3")

        def nonlin_vals(u):
            return -0.9 * u - u ** 3

        vals = 0.1 * np.random.RandomState(0).randn(32, 32, 32)
        u0 = _make_trig_interp(vals, [(0.0, G), (0.0, G), (0.0, G)])
        return (dom, (0.0, 800.0), lin, nonlin, u0, (-2.0, -1.0),
                ("lap", "biharm"), nonlin_vals, True, 1)
    raise ValueError(
        f"Unrecognized PDE {pdechar!r}. Options: GL, GS, Schnak, SH.")


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
        self._n_vars = 1
        if arg is None:
            return
        if isinstance(arg, str):
            (dom, tsp, lin, nonlin, init, lin_scales, lin_ops,
             nonlin_vals, is_real, n_vars) = _preset(arg)
            self._n_vars = n_vars
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
        return self._n_vars

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

    if getattr(S, "_n_vars", 1) > 1:
        from chebfunjax.operators.spinop import _parse_scheme as _ps
        _da0 = kwargs.get("dealias", True)
        _al0 = list(args)
        for _i in range(len(_al0) - 1):
            if isinstance(_al0[_i], str) and _al0[_i].lower() == "dealias":
                _da0 = str(_al0[_i + 1]).lower() in ("on", "true", "1")
        return _spin3_system(S, N, dt, _ps(args, kwargs, None), bool(_da0))
    core = _CoreSpinOp3(
        lin_scales=S._lin_scales,
        lin_ops=S._lin_ops,
        nonlin_vals=S._nonlin_vals,
        domain=tuple(float(v) for v in S.domain),
        tspan=tuple(float(v) for v in S.tspan),
        u0=S.init,
        is_real=S._is_real,
    )
    from chebfunjax.operators.spinop import _parse_scheme
    _da = kwargs.get("dealias", True)
    _al = list(args)
    for i in range(len(_al) - 1):
        if isinstance(_al[i], str) and _al[i].lower() == "dealias":
            _da = str(_al[i + 1]).lower() in ("on", "true", "1")
    _grids, _t, u_final = _core_spin3(core, N, dt, dealias=bool(_da),
                                      scheme=_parse_scheme(args, kwargs, None))
    ax, bx, ay, by, az, bz = core.domain
    # The solver's value grid is meshgrid-'xy' (axes y, x, z); the
    # interpolant builder takes ndgrid layout (x, y, z).  (Before this
    # transpose spin3's output was x/y-swapped relative to its initial
    # condition -- MATLAB's is not.)
    return _make_trig_interp(np.transpose(np.asarray(u_final), (1, 0, 2)),
                             [(ax, bx), (ay, by), (az, bz)])


def _spin3_system(S: "Spinop3", N: int, dt: float, scheme, dealias: bool,
                  M: int = 32):
    """spin3 for a system of unknowns (MATLAB spinop3 'GS'/'Schnak'):
    the coefficient blocks are stacked along the first axis
    ``(nVars*N, N, N)`` and stepped by the generic expinteg engine
    (any scheme, ETDRK4 by default)."""
    from chebfunjax.operators.expinteg_engine import Expinteg, integrate
    from chebfunjax.operators.spinop2 import _TrigInterpVector

    n_vars = S._n_vars
    dom = tuple(float(v) for v in S.domain)
    x0, x1, y0, y1, z0, z1 = dom
    cores = [_CoreSpinOp3(lin_scales=S._lin_scales[i], lin_ops=S._lin_ops[i],
                          nonlin_vals=S._nonlin_vals[i], domain=dom,
                          tspan=tuple(float(v) for v in S.tspan),
                          u0=S.init[i], is_real=S._is_real)
             for i in range(n_vars)]
    x = np.linspace(x0, x1, N, endpoint=False)
    y = np.linspace(y0, y1, N, endpoint=False)
    z = np.linspace(z0, z1, N, endpoint=False)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")
    L = np.concatenate([np.asarray(c.build_linear_eigenvalues(N), dtype=complex)
                        for c in cores], axis=0)
    is_real = bool(np.allclose(np.imag(L), 0.0))
    if is_real:
        L = np.real(L)
    u0 = np.concatenate([np.asarray(np.fft.fftn(np.asarray(
        S.init[i](xx, yy, zz), dtype=complex)), dtype=complex)
        for i in range(n_vars)], axis=0)
    mask = None
    if dealias:
        m3 = np.asarray(cores[0].dealias_mask(N), dtype=float)
        mask = np.concatenate([m3] * n_vars, axis=0)
        u0 = u0 * mask
    nonlin = S._nonlin_vals

    def _Nv(vals):
        parts = [vals[k * N:(k + 1) * N] for k in range(n_vars)]
        return np.concatenate([np.asarray(nonlin[k](*parts))
                               for k in range(n_vars)], axis=0)

    t0, tf = (float(v) for v in S.tspan)
    nsteps = int(round((tf - t0) / dt))
    post = (lambda u: u * mask) if mask is not None else None
    u = integrate(Expinteg(scheme or "etdrk4"), dt, L, M, is_real,
                  np.ones_like(L, dtype=float), _Nv, np.fft.ifftn, np.fft.fftn,
                  n_vars, u0, nsteps, post=post)
    comps = []
    for k in range(n_vars):
        val = np.fft.ifftn(u[k * N:(k + 1) * N])
        val = np.transpose(np.real(val) if S._is_real else val, (1, 0, 2))
        comps.append(_make_trig_interp(val, [(x0, x1), (y0, y1), (z0, z1)]))
    return _TrigInterpVector(comps)
