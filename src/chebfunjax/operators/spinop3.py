"""Spinop3 -- stiff semilinear PDEs on a 3D periodic domain,
u_t = L u + N(u) with L a polynomial in the Laplacian, solved by
spin3() with the ETDRK4 exponential integrator on a 3D Fourier grid.

Added by Claude Fable 5 (spinop3 port).  The constructor surface
(named presets, ``func2str`` of the nonlinear part, domain/tspan
plumbing) mirrors @spinop3/spinop3.m; public time-stepping and saved
outputs use the shared JAX exponential-integrator engine.

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

from chebfunjax.operators._spin3_public import Spin3SolutionMatrix
from chebfunjax.operators.spinopsphere import FuncHandle, func2str

__all__ = ["Spinop3", "spin3", "Spin3SolutionMatrix", "func2str", "FuncHandle"]


# ---------------------------------------------------------------------------
# Built-in PDE catalogue (constructor surface)
# ---------------------------------------------------------------------------


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
        from chebfunjax.chebfun3d.chebfun3 import chebfun3
        from chebfunjax.utils import _randnfun

        domain = (0.0, 50.0, 0.0, 50.0, 0.0, 50.0)
        # Source uses .1*randn(32,32,32). The shared JAX normal stream
        # advances once per preset; its samples are not MATLAB stream parity.
        values = .1 * _randnfun._normal_draw(
            _randnfun._next_key(), 32**3, 1).reshape((32, 32, 32), order="F")
        initial = chebfun3(values, domain, trig=True)
        return (
            domain,                              # domain
            (0.0, 100.0),                         # tspan
            lin, nonlin, initial,
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
        # Native SH uses .1*randn(32,32,32) followed by numeric trig
        # Chebfun3 construction. Reuse the shared JAX normal stream, matching
        # the GL adapter; MATLAB randn samples are not claimed bit-identical.
        G = 25.0
        dom = (0.0, G, 0.0, G, 0.0, G)
        lin = FuncHandle(None, "@(u)-2*lap(u)-biharm(u)")
        nonlin = FuncHandle(lambda u: -.9 * u - u ** 3, "@(u)-.9*u-u.^3")

        def nonlin_vals(u):
            return -0.9 * u - u ** 3

        from chebfunjax.chebfun3d.chebfun3 import chebfun3
        from chebfunjax.utils import _randnfun
        vals = 0.1 * _randnfun._normal_draw(
            _randnfun._next_key(), 32**3, 1).reshape((32, 32, 32), order="F")
        u0 = chebfun3(vals, dom, trig=True)
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
        self._lin_handle = None
        self._nonlin_handle = None
        self._init_value = None
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
        ops = ("lap", "biharm", "triharm", "quadharm", "quintharm")
        scales, names = [], []
        for row in coeffs:
            sc = tuple(c for c in row if c != 0) or (0.0,)
            nm = tuple(o for c, o in zip(row, ops) if c != 0) or ("lap",)
            scales.append(sc)
            names.append(nm)
        if len(coeffs) == 1:
            self._lin_scales, self._lin_ops = scales[0], names[0]
        else:
            self._lin_scales, self._lin_ops = scales, names

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
    """Construct native numeric periodic Chebfun3 output from ndgrid values.

    Provenance
    ----------
    MATLAB source: @spinoperator/solvepde.m, final output construction.
    Chebfun commit: 7574c77.
    Numeric trig factors retain the sampled grid and symmetric Nyquist mode.
    """
    from chebfunjax.chebfun3d.chebfun3 import chebfun3

    domain = tuple(endpoint for interval in domain_triples for endpoint in interval)
    return chebfun3(jnp.asarray(V), domain, trig=True)


# ---------------------------------------------------------------------------
# spin3 solver
# ---------------------------------------------------------------------------


_SPIN3_PRESET_DEFAULTS = {
    "GL": (32, 0.1, (-1.0, 1.0), 2, 64),
    "GS": (32, 5.0, (0.25, 0.8, 0.0, 0.35), 10, 64),
    "SCHNAK": (32, 0.5, (0.7, 1.7, 0.65, 1.05), 10, 64),
    "SH": (32, 1.0, (-0.4, 0.5), 4, 64),
}


def _normalize_spin3_call(S, N, dt, args, kwargs):
    """Normalize source-supported named and explicit ``spin3`` call forms.

    The named forms are ``spin3(name)``, ``spin3(name,key,value)``, and
    ``spin3(name,key,value,...)``. Explicit operators retain
    ``spin3(S,N,dt[,pref-or-pairs...])``.

    Provenance
    ----------
    MATLAB source: spin3.m input dispatch and local parseInputs.
    Chebfun commit: 7574c77.
    """
    from chebfunjax.operators.spinpref import SpinPref3

    if isinstance(S, str):
        key = S.upper()
        bare_preset = N is None and dt is None and not args and not kwargs
        if key not in _SPIN3_PRESET_DEFAULTS:
            if bare_preset:
                try:
                    Spinop3(S)
                except Exception as error:
                    raise ValueError(
                        "Unrecognized PDE. See HELP/SPINSPHERE for the list of PDEs."
                    ) from error
            # Spinop3 owns the source-shaped unknown-preset diagnostic.
            raise ValueError(
                f"Unrecognized PDE {S!r}. Options: GL, GS, Schnak, SH."
            )
        n, step, clim, iterplot, nplot = _SPIN3_PRESET_DEFAULTS[key]
        if N is None and dt is None:
            pairs = args
        elif N is not None and dt is not None:
            pairs = (N, dt, *args)
        else:
            raise TypeError("Named spin3 calls require complete preference pairs.")
        if len(pairs) % 2:
            raise TypeError("Named spin3 calls require preference name/value pairs.")
        pref = SpinPref3(Clim=clim, iterplot=iterplot, Nplot=nplot)
        names = {name.lower(): name for name in pref._defaults}
        for name, value in zip(pairs[::2], pairs[1::2]):
            if not isinstance(name, str) or name.lower() not in names:
                raise ValueError(f"Unrecognized spin3 preference: {name!r}")
            setattr(pref, names[name.lower()], value)
        for name, value in kwargs.items():
            if name.lower() not in names:
                raise ValueError(f"Unrecognized spin3 preference: {name!r}")
            setattr(pref, names[name.lower()], value)
        if bare_preset:
            # Native spin3 validates the preset in its single-argument try
            # block, then parseInputs constructs it again. This advances the
            # JAX normal stream twice for GL and SH, matching that call path.
            Spinop3(S)
        S = Spinop3(S)
        return S, n, step, (pref,), {}

    if N is None or dt is None:
        raise TypeError("Explicit spin3 calls require S, N, and dt.")
    return S, N, dt, args, kwargs


def spin3(S: Spinop3 | str, N: int | str | None = None,
          dt: float | str | None = None, *args, return_times=False,
          return_computing_time=False, **kwargs):
    """Solve a periodic 3D PDE with source checkpoint and output semantics.

    Scalar two-endpoint solves return the final periodic Chebfun3. Systems
    or multiple requested times return a bounded Spin3SolutionMatrix with
    unknown rows and saved-time columns. ``return_times=True`` expresses
    MATLAB's second output. Only step-aligned requested times are saved;
    checkpoints come from one continuous integration. Preferences accept
    SpinPref3, native name/value pairs, or keywords. Headless numerical
    execution uses the shared JAX exponential-integrator engine; plotting
    preferences are accepted without displaying a movie.

    Provenance
    ----------
    MATLAB source: spin3.m, @spinpref3/spinpref3.m,
        @spinoperator/solvepde.m, @spinop3/discretize.m,
        @expinteg/{computeCoeffs,oneStep,startMultistep}.m
    Chebfun commit: 7574c77.
    """
    from chebfunjax.operators._spin3_public import solve_public

    S, N, dt, args, kwargs = _normalize_spin3_call(S, N, dt, args, kwargs)
    return solve_public(
        S, N, dt, args, kwargs, return_times=return_times,
        return_computing_time=return_computing_time)
