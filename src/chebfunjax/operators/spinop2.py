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
    if key in ("GS", "GSSPOTS"):
        # Gray-Scott: u_t = 2e-5 lap(u) + F(1-u) - u v^2,
        #             v_t = 1e-5 lap(v) - (F+K) v + u v^2 on [0,1]^2.
        F, K = (0.030, 0.057) if key == "GS" else (0.026, 0.059)
        G = 1.0
        lin = FuncHandle(None, "@(u,v)[2e-5*lap(u);1e-5*lap(v)]")
        nonlin = FuncHandle(
            lambda u, v: (F * (1 - u) - u * v ** 2, -(F + K) * v + u * v ** 2),
            "@(u,v)[F*(1-u)-u.*v.^2;-(F+K)*v+u.*v.^2]",
        )

        def n1(u, v):
            return F * (1.0 - u) - u * v ** 2

        def n2(u, v):
            return -(F + K) * v + u * v ** 2

        def u01(x, y):
            return 1.0 - jnp.exp(-100.0 * ((x - G / 2.05) ** 2
                                           + (y - G / 2.05) ** 2))

        def u02(x, y):
            return jnp.exp(-100.0 * ((x - G / 2) ** 2 + 2 * (y - G / 2) ** 2))

        return (
            (0.0, G, 0.0, G), (0.0, 5000.0), lin, nonlin, [u01, u02],
            [(2e-5, 0.0, 0.0, 0.0, 0.0), (1e-5, 0.0, 0.0, 0.0, 0.0)],
            [n1, n2], 2, True,
        )
    if key == "SCHNAK":
        # Schnakenberg: u_t = lap(u) + 3(.1 - u + u^2 v),
        #               v_t = 10 lap(v) + 3(.9 - u^2 v) on [0,50]^2.
        G = 50.0
        lin = FuncHandle(None, "@(u,v)[lap(u);10*lap(v)]")
        nonlin = FuncHandle(
            lambda u, v: (3 * (.1 - u + u ** 2 * v), 3 * (.9 - u ** 2 * v)),
            "@(u,v)[3*(.1-u+u.^2.*v);3*(.9-u.^2.*v)]",
        )

        def n1(u, v):
            return 3.0 * (0.1 - u + u ** 2 * v)

        def n2(u, v):
            return 3.0 * (0.9 - u ** 2 * v)

        def u01(x, y):
            return 1.0 - jnp.exp(-2.0 * ((x - G / 2.15) ** 2
                                         + (y - G / 2.15) ** 2))

        def u02(x, y):
            return (0.9 / (0.1 + 0.9) ** 2
                    + jnp.exp(-2.0 * ((x - G / 2) ** 2 + 2 * (y - G / 2) ** 2)))

        return (
            (0.0, G, 0.0, G), (0.0, 500.0), lin, nonlin, [u01, u02],
            [(1.0, 0.0, 0.0, 0.0, 0.0), (10.0, 0.0, 0.0, 0.0, 0.0)],
            [n1, n2], 2, True,
        )
    if key == "SH":
        # Swift-Hohenberg: u_t = -2 lap(u) - biharm(u) - .9 u - u^3 on [0,50]^2.
        G = 50.0
        dom = (0.0, G, 0.0, G)
        lin = FuncHandle(None, "@(u)-2*lap(u)-biharm(u)")
        nonlin = FuncHandle(lambda u: -.9 * u - u ** 3, "@(u)-.9*u-u.^3")

        def nonlin_vals(u):
            return -0.9 * u - u ** 3

        from chebfunjax.utils.random import randnfun2
        f0 = randnfun2(4.0, dom, trig=True)
        gx, gy = jnp.meshgrid(jnp.linspace(0.0, G, 201), jnp.linspace(0.0, G, 201))
        scale = float(jnp.max(jnp.abs(f0(gx, gy))))

        def u0(x, y):
            return f0(x, y) / scale

        return (dom, (0.0, 800.0), lin, nonlin, u0,
                (-2.0, -1.0, 0.0, 0.0, 0.0), nonlin_vals, 1, True)
    raise ValueError(
        f"Unrecognized PDE {pdechar!r}. Options: GL, GS, GSspots, Schnak, SH.")


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
        self._lin_handle = None
        self._nonlin_handle = None
        self._init_value = None
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


    # ------------------------------------------------------------------
    # MATLAB-style operator handles: S.lin = '@(u,v) [lap(u); 10*lap(v)]'
    # ------------------------------------------------------------------
    @property
    def lin(self):
        """Linear part (a :class:`FuncHandle`; assign a MATLAB anonymous
        function string to define a custom operator)."""
        return self._lin_handle

    @lin.setter
    def lin(self, value):
        if value is None or isinstance(value, FuncHandle):
            self._lin_handle = value
            return
        if isinstance(value, str):
            from chebfunjax.operators.spinop_parse import (
                func2str_text,
                parse_lin_handle,
            )
            names, coeffs = parse_lin_handle(value)
            self._lin_handle = FuncHandle(None, func2str_text(value))
            self._set_lin_coeffs(coeffs)
            self._n_vars = len(coeffs)
            self._is_real = all(complex(c).imag == 0.0
                                for row in coeffs for c in row)
            return
        raise TypeError(
            "S.lin must be a MATLAB anonymous-function string such as "
            "'@(u) lap(u) - biharm(u)' or '@(u,v) [lap(u); 10*lap(v)]'.")

    @property
    def nonlin(self):
        """Nonlinear part (a :class:`FuncHandle`; assign a MATLAB string
        such as ``'@(u) u - u.^3'`` or a Python callable of the values)."""
        return self._nonlin_handle

    @nonlin.setter
    def nonlin(self, value):
        if value is None or isinstance(value, FuncHandle):
            self._nonlin_handle = value
            return
        if isinstance(value, str):
            from chebfunjax.operators.spinop_parse import (
                func2str_text,
                parse_nonlin_handle,
            )
            names, fns = parse_nonlin_handle(value)
            self._nonlin_vals = fns[0] if len(fns) == 1 else list(fns)
            self._nonlin_handle = FuncHandle(
                fns[0] if len(fns) == 1 else (lambda *a: tuple(f(*a) for f in fns)),
                func2str_text(value))
            return
        if callable(value):
            self._nonlin_vals = value
            self._nonlin_handle = FuncHandle(value, "@(u)<python callable>")
            return
        if isinstance(value, (list, tuple)) and all(callable(f) for f in value):
            self._nonlin_vals = list(value)
            self._nonlin_handle = FuncHandle(
                lambda *a: tuple(f(*a) for f in value), "@(u,...)<python callables>")
            return
        raise TypeError("S.nonlin must be a MATLAB string, a callable or a "
                        "list of callables.")

    @property
    def init(self):
        """Initial condition: a callable of the grid (or a list of them
        for a system)."""
        return self._init_value

    @init.setter
    def init(self, value):
        self._init_value = value

    def _set_lin_coeffs(self, coeffs):
        self._lin_coeffs = tuple(coeffs[0]) if len(coeffs) == 1 else [
            tuple(c) for c in coeffs]

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
        is_real=bool(S._is_real) and _init_is_real(S.init, S.domain),
    )
    from chebfunjax.operators.spinop import _parse_scheme
    # MATLAB spinpref2 defaults: dealias 'off'; chebfunjax's ETDRK4 path
    # dealiases by default (stability of the stiff presets) -- pass
    # dealias=False for MATLAB's default behaviour.
    _da = kwargs.get("dealias", True)
    _al = list(args)
    for i in range(len(_al) - 1):
        if isinstance(_al[i], str) and _al[i].lower() == "dealias":
            _da = str(_al[i + 1]).lower() in ("on", "true", "1")
    _xx, _yy, _t, u_final = _core_spin2(core, N, dt, dealias=bool(_da),
                                        scheme=_parse_scheme(args, kwargs, None))
    ax, bx, ay, by = core.domain
    if isinstance(u_final, (list, tuple)):
        # MATLAB returns the chebmatrix [u; v]: a vector of interpolants.
        return _TrigInterpVector([_make_trig_interp(v, [(ax, bx), (ay, by)])
                                  for v in u_final])
    return _make_trig_interp(u_final, [(ax, bx), (ay, by)])


class _TrigInterpVector:
    """System solution of spin2 (MATLAB chebmatrix ``[u; v]``): the
    per-unknown periodic interpolants in ``components``."""

    def __init__(self, components):
        self.components = list(components)

    def __len__(self):
        return len(self.components)

    def __getitem__(self, k):
        return self.components[k]

    def __iter__(self):
        return iter(self.components)


def _init_is_real(init, domain) -> bool:
    """True when the initial condition(s) evaluate real at the domain
    centre (a real operator with a complex initial condition keeps a
    complex solution)."""
    mids = [0.5 * (float(domain[2 * k]) + float(domain[2 * k + 1]))
            for k in range(len(domain) // 2)]
    fns = init if isinstance(init, (list, tuple)) else [init]
    for f in fns:
        try:
            v = np.asarray(f(*[jnp.asarray(m) for m in mids]))
        except Exception:
            return True
        if np.iscomplexobj(v) and np.any(np.imag(v) != 0.0):
            return False
    return True
