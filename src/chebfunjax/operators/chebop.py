"""User-friendly nonlinear operator constructor for ODEs and BVPs.

:class:`Chebop` mirrors MATLAB Chebfun's ``chebop`` class: the user specifies
the differential operator as a Python callable and attaches boundary conditions
as scalars, callables, or strings.  For *linear* problems, :class:`Chebop`
delegates to :class:`~chebfunjax.operators.linop.Linop` (purely spectral
solve).  For *nonlinear* problems, Newton iteration is used, with each Newton
step solved by a :class:`Linop`.

Typical use::

    import jax.numpy as jnp
    from chebfunjax.operators.chebop import Chebop

    # u'' + u = 0,  u(0) = 0,  u(pi) = 0   =>  u = sin(x)
    N = Chebop(lambda x, u: u.diff(2) + u, domain=(0.0, jnp.pi))
    N.lbc = 0.0    # u(0) = 0
    N.rbc = 0.0    # u(pi) = 0
    u = N.solve(0.0)

    # Linear problem (same thing):
    u = N \\ 0.0   # or  N.solve(0.0)

Translated from MATLAB Chebfun class ``@chebop`` (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import warnings
from typing import Callable

import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.operators.blocks import (
    ChebColloc2Disc,
    FunctionalBlock,
    OperatorBlock,
    eval_at,
)
from chebfunjax.operators.linop import Linop
from chebfunjax.utils.quadrature import chebpts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chebfun_from_values(values, domain: tuple[float, float]):
    """Wrap collocation values as a Chebfun."""
    from chebfunjax.chebfun1d.chebfun import Chebfun
    dom = Domain(domain)
    return Chebfun.from_values(jnp.asarray(values, dtype=jnp.float64), dom)


def _chebfun_zeros(domain: tuple[float, float]):
    """Return the zero Chebfun on domain."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: jnp.zeros_like(x), domain=tuple(domain), n=2)


def _chebfun_identity(domain: tuple[float, float]):
    """Return the identity Chebfun f(x) = x on domain."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: x, domain=tuple(domain), n=2)


def _eval_chebfun_at(u, x0: float) -> float:
    """Evaluate u at the physical point x0."""
    if isinstance(u, (int, float)):
        return float(u)
    arr = u(jnp.array(x0, dtype=jnp.float64))
    return float(arr)


class _TrigX:
    """Array wrapper exposing chebfun-style elementwise methods.

    The periodic discretizations sample the independent variable as a
    plain array; ops written chebfun-style (``x.cos().sin()``) need the
    method chain to keep working, while mixed arithmetic with the
    solver's linear-operator proxies must hand over the raw array (via
    the proxy's reflected operation).
    """

    def __init__(self, v):
        self.v = jnp.asarray(v)

    def _w(self, v):
        return _TrigX(v)

    def sin(self):
        return self._w(jnp.sin(self.v))

    def cos(self):
        return self._w(jnp.cos(self.v))

    def tan(self):
        return self._w(jnp.tan(self.v))

    def exp(self):
        return self._w(jnp.exp(self.v))

    def log(self):
        return self._w(jnp.log(self.v))

    def sqrt(self):
        return self._w(jnp.sqrt(self.v))

    def sinh(self):
        return self._w(jnp.sinh(self.v))

    def cosh(self):
        return self._w(jnp.cosh(self.v))

    def tanh(self):
        return self._w(jnp.tanh(self.v))

    def abs(self):
        return self._w(jnp.abs(self.v))

    def __abs__(self):
        return self.abs()

    def __neg__(self):
        return self._w(-self.v)

    def __pow__(self, p):
        return self._w(self.v ** p)

    @staticmethod
    def _plain(o):
        return isinstance(o, (int, float, complex)) or hasattr(o, "dtype")

    def _bin(self, o, fwd, refl_name):
        if isinstance(o, _TrigX):
            return self._w(fwd(self.v, o.v))
        if self._plain(o):
            return self._w(fwd(self.v, o))
        # Proxy operand: hand the raw array to its reflected op.
        return getattr(o, refl_name)(self.v)

    def __add__(self, o):
        return self._bin(o, lambda a, b: a + b, "__radd__")

    def __radd__(self, o):
        return self._bin(o, lambda a, b: b + a, "__add__")

    def __sub__(self, o):
        return self._bin(o, lambda a, b: a - b, "__rsub__")

    def __rsub__(self, o):
        return self._bin(o, lambda a, b: b - a, "__sub__")

    def __mul__(self, o):
        return self._bin(o, lambda a, b: a * b, "__rmul__")

    def __rmul__(self, o):
        return self._bin(o, lambda a, b: b * a, "__mul__")

    def __truediv__(self, o):
        return self._bin(o, lambda a, b: a / b, "__rtruediv__")

    def __rtruediv__(self, o):
        return self._bin(o, lambda a, b: b / a, "__truediv__")


class SystemSolution(list):
    """Solution container for systems: indexable list of chebfuns
    with MATLAB-ish u{1} semantics via u[0] / u.blocks.

    Provenance
    ----------
    MATLAB source : @chebmatrix (solution container role)
    Chebfun commit: 7574c77
    """

    @property
    def blocks(self):
        return list(self)


# ===========================================================================
# Chebop
# ===========================================================================


class Chebop:
    """User-facing operator constructor for ODEs and BVPs.

    :class:`Chebop` mirrors MATLAB Chebfun's ``chebop``: the user defines an
    operator (possibly nonlinear) as a Python callable and attaches boundary
    conditions.  Calling :meth:`solve` (or using ``N \\ f``) dispatches to
    either:

    - **Linear** problems: direct spectral solve via :class:`Linop`.
    - **Nonlinear** problems: Newton iteration with linearised :class:`Linop`
      solves at each step.

    Parameters
    ----------
    op : callable or None
        The differential operator.  For a scalar problem, the signature is
        one of:

        * ``lambda x, u: ...``  — explicit ``x`` (identity Chebfun) + ``u``
        * ``lambda u: ...``     — autonomous (no explicit ``x``)

        The callable must accept :class:`~chebfunjax.chebfun1d.Chebfun`
        objects and return a :class:`~chebfunjax.chebfun1d.Chebfun` (or a
        scalar ``0`` for a zero RHS).
    domain : (float, float), default (-1, 1)
        Physical domain.
    lbc : scalar or callable or None
        Left boundary condition.  Interpreted as:

        * scalar ``c``      → ``u(a) = c``  (Dirichlet)
        * callable ``g(u)`` → ``g(u) = 0`` at the left endpoint
    rbc : scalar or callable or None
        Right boundary condition.  Same conventions as ``lbc``.

    Attributes
    ----------
    op : callable or None
    lbc : scalar or callable or None
    rbc : scalar or callable or None
    domain : (float, float)

    Examples
    --------
    **Linear BVP** — u'' = -1, u(±1) = 0:

    >>> from chebfunjax.operators.chebop import Chebop
    >>> N = Chebop(lambda x, u: u.diff(2), domain=(-1.0, 1.0))
    >>> N.lbc = 0.0
    >>> N.rbc = 0.0
    >>> u = N.solve(-1.0)          # RHS = -1 (constant)
    >>> import jax.numpy as jnp
    >>> abs(float(u(0.0)) - 0.5) < 1e-12
    True

    **Eigenvalues** of u'' with Dirichlet BCs on [-1,1]:

    >>> lam = N.eigs(k=4)
    >>> # Should be -(1*pi/2)^2, -(2*pi/2)^2, ...

    Notes
    -----
    Operator construction and solve are NOT JIT-safe (Python-level adaptive
    loops).  Evaluation of the returned :class:`~chebfunjax.chebfun1d.Chebfun`
    *is* JIT-safe.

    The nonlinear Newton iteration requires the operator to be applied to a
    :class:`~chebfunjax.chebfun1d.Chebfun` object.  Linearization is performed
    by finite differences (of Chebfun objects) rather than automatic
    differentiation, which is consistent with Chebfun's ``adchebfun`` approach
    but simpler.

    Provenance
    ----------
    MATLAB source : @chebop/chebop.m, @chebop/mldivide.m,
        @chebop/solvebvpLinear.m, @chebop/solvebvpNonlinear.m,
        @chebop/linearize.m, @chebop/newtonBVP.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Linop, OperatorBlock, FunctionalBlock
    """

    def __init__(
        self,
        op: Callable | None = None,
        domain: tuple[float, float] = (-1.0, 1.0),
        lbc=None,
        rbc=None,
        bc=None,
    ) -> None:
        self.op = op
        self.domain: tuple[float, float] = tuple(float(v) for v in domain)
        self._lbc_raw = None
        self._rbc_raw = None
        self._bc_show = None
        #: General constraint callable set via ``N.bc = @(x, u, ...) [...]``
        #: (MATLAB's "other" boundary conditions).  Unlike lbc/rbc it is a
        #: single callable returning a list of scalar residual conditions
        #: that the user evaluates at arbitrary points (e.g. ``u(0)``,
        #: ``feval(diff(u), 0) - p``), including interior points.  It is
        #: additive to any lbc/rbc and can reference unknown parameters.
        self._bc_general = None
        self._periodic = False
        #: Initial guess for nonlinear solves (Chebfun, callable, or None) —
        #: MATLAB's N.init. Previously assigning N.init was silently ignored.
        self.init = None
        #: Deflation data set by :meth:`deflate` — ``(roots, p, alp, type)``
        #: where ``roots`` is a list of previously found solutions (Chebfun).
        #: When non-None the operator carries a multiplicative deflation
        #: factor ``M(u; roots)`` (see :meth:`deflate`); it is genuinely
        #: nonlinear and its Newton Jacobian is formed by finite differences
        #: so the deflation term is captured (MATLAB uses AD for this).
        self._deflation = None
        if lbc is not None:
            self.lbc = lbc
        if rbc is not None:
            self.rbc = rbc
        if bc is not None:
            self.bc = bc

    # ------------------------------------------------------------------
    # BC setters (properties for MATLAB-style assignment)
    # ------------------------------------------------------------------

    @property
    def lbc(self):
        """Left boundary condition (scalar, callable, or None)."""
        return self._lbc_raw

    @lbc.setter
    def lbc(self, val):
        self._lbc_raw = val

    @property
    def rbc(self):
        """Right boundary condition (scalar, callable, or None)."""
        return self._rbc_raw

    @rbc.setter
    def rbc(self, val):
        self._rbc_raw = val

    @property
    def bc(self):
        """Boundary conditions applied at BOTH endpoints.

        MATLAB semantics (@chebop set.bc): a numeric value or callable
        sets ``lbc = rbc = value``; the strings ``'dirichlet'`` and
        ``'neumann'`` set u = 0 / u' = 0 at both ends. Before this
        property existed, ``L.bc = 0`` silently created a dead attribute
        and the operator solved with NO boundary conditions — wrong
        eigenvalues with no warning.
        """
        return self._bc_show

    @bc.setter
    def bc(self, val):
        self._bc_show = val
        self._periodic = False
        self._bc_general = None
        if val is None:
            self._lbc_raw = None
            self._rbc_raw = None
            return
        if isinstance(val, str):
            key = val.lower()
            if key == "dirichlet":
                self._lbc_raw = 0.0
                self._rbc_raw = 0.0
            elif key == "neumann":
                self._lbc_raw = lambda u: u.diff()
                self._rbc_raw = lambda u: u.diff()
            elif key == "periodic":
                # Periodic problems are solved by Fourier collocation in
                # solve(); no endpoint constraints are attached.
                # Implemented by Claude Opus 4.8 (task #24).
                self._lbc_raw = None
                self._rbc_raw = None
                self._periodic = True
                return
            else:
                raise ValueError(
                    f"Unknown bc keyword {val!r}: expected 'dirichlet', "
                    f"'neumann', or 'periodic'."
                )
        elif callable(val):
            # MATLAB N.bc = @(x, u, ...) [...] : a general constraint
            # function returning scalar conditions (u evaluated at arbitrary
            # points, possibly interior; may reference unknown parameters).
            # Additive to lbc/rbc rather than overwriting them.
            self._bc_general = val
        elif isinstance(val, (int, float)):
            self._lbc_raw = val
            self._rbc_raw = val
        else:
            raise TypeError(
                f"Chebop.bc must be a number, callable, or keyword string; "
                f"got {type(val).__name__}."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        f=0.0,
        n: int | None = None,
        n_min: int = 8,
        n_max: int = 2048,
        tol: float = 1e-10,
        max_iter: int = 15,
        newton_tol: float = 5e-13,
        discretization: str | None = None,
    ):
        if (discretization is not None
                and str(discretization).lower() in ("chebcolloc2", "colloc2")
                and getattr(self, "_periodic", False)):
            # MATLAB: pref.discretization = @chebcolloc2 on a periodic
            # problem solves by Chebyshev collocation with wrap-around
            # rows u^(d)(a) = u^(d)(b), d < order, instead of a Fourier
            # discretization (FourierCollocation example).
            out = self._solve_periodic_colloc(f, n=n, n_min=n_min,
                                              n_max=n_max, tol=tol)
            return self._simplify_solution(out)
        out = self._solve_impl(f, n=n, n_min=n_min, n_max=n_max, tol=tol,
                               max_iter=max_iter, newton_tol=newton_tol)
        # MATLAB's linsolve/solvebvp returns SIMPLIFIED chebfuns; an
        # unchopped 1e-14 coefficient tail is amplified ~n^(2m) by
        # diff(m) in residual checks (test_promote_functional measured
        # 2e-9 from exactly this).
        return self._simplify_solution(out)

    @staticmethod
    def _simplify_solution(out):
        from chebfunjax.chebfun1d.chebfun import Chebfun
        if isinstance(out, Chebfun):
            return out.simplify()
        if isinstance(out, (list, tuple)):
            return type(out)(Chebop._simplify_solution(v) for v in out)
        return out

    def _solve_impl(
        self,
        f=0.0,
        n: int | None = None,
        n_min: int = 8,
        n_max: int = 2048,
        tol: float = 1e-10,
        max_iter: int = 15,
        newton_tol: float = 5e-13,
    ):
        """Solve the BVP ``N[u] = f`` with the attached boundary conditions.

        For linear operators this calls :meth:`Linop.solve` directly.
        For nonlinear operators, Newton iteration is used.

        Parameters
        ----------
        f : scalar, callable, or Chebfun, default 0.0
            Right-hand side.  If a scalar, treated as a constant function.
            If callable, called at the collocation points.
        n : int or None
            Fixed discretization size (``None`` = adaptive).
        n_min : int, default 8
            Minimum size for adaptive loop.
        n_max : int, default 2048
            Maximum size for adaptive loop.
        tol : float, default 1e-10
            Convergence tolerance for the adaptive size loop.
        max_iter : int, default 15
            Maximum Newton iterations (for nonlinear problems).
        newton_tol : float, default 1e-10
            Newton convergence tolerance (max absolute correction).

        Returns
        -------
        u : Chebfun
            Solution.

        Raises
        ------
        RuntimeError
            If Newton iteration does not converge.

        Provenance
        ----------
        MATLAB source : @chebop/mldivide.m, @chebop/solvebvp.m,
            @chebop/solvebvpLinear.m, @chebop/solvebvpNonlinear.m
        Chebfun commit: 7574c77
        """
        if self.op is None:
            raise ValueError(
                "Chebop.solve: operator is not set. Assign N.op = lambda x, u: ...  "
                "before solving."
            )

        # Piecewise domains (>= 1 interior breakpoint): collocate each
        # unknown piece-by-piece and glue with continuity conditions at the
        # breaks.  Handles both scalar and system BVPs; single-interval
        # domains fall through to the established spectral paths unchanged.
        if len(self.domain) > 2:
            # Periodic problems on piecewise domains solve with the
            # piecewise collocation too — the endpoint slots carry
            # wrap-around rows instead of lbc/rbc (PeriodicSystem).
            return self._solve_piecewise(f, n=n, max_iter=max_iter)

        # A discontinuous coefficient (e.g. ``(x>=0)*diff(u)``) injects
        # breakpoints the single-interval spectral path cannot resolve —
        # it grinds to n_max unhappy and loses digits.  Probe the op once
        # on a smooth chebfun; interior breakpoints in the output route
        # the problem to the piecewise solver (which re-detects and
        # unions them into its grid).  MATLAB does the equivalent while
        # building the piecewise chebmatrix.
        if (not getattr(self, "_periodic", False)
                and self._n_vars() == 1 and self._bc_general is None):
            try:
                import inspect as _inspect

                from chebfunjax.chebfun1d.chebfun import Chebfun as _Cf
                _dom = Domain(tuple(float(v) for v in self.domain))
                _nargs = len(_inspect.signature(self.op).parameters)
                _probe = _Cf.identity(_dom)
                _out = (self.op(_probe, _probe) if _nargs > 1
                        else self.op(_probe))
                if not isinstance(_out, (list, tuple)):
                    _out = [_out]
                _bps = [float(v) for v in self.domain]
                _has_break = any(
                    hasattr(o, "domain")
                    and any(all(abs(float(b) - e) > 1e-12 for e in _bps)
                            for b in o.domain.breakpoints)
                    for o in _out)
                # A piecewise RHS (e.g. sign-function initial data in
                # the ContourExpm heat solves) also demands a piecewise
                # solution grid: sampling it on a smooth grid Gibbses.
                _fbreaks = []
                if hasattr(f, "domain"):
                    _fbreaks = [float(b) for b in f.domain.breakpoints
                                if all(abs(float(b) - e) > 1e-12
                                       for e in _bps)]
                if _has_break or _fbreaks:
                    return self._solve_piecewise(
                        f, n=n, max_iter=max_iter,
                        cont_breaks=_fbreaks or None)
            except Exception:
                pass

        # Interior jump / one-sided conditions in a general .bc make the
        # solution discontinuous at the referenced points: detect those
        # breakpoints and solve piecewise, imposing the .bc conditions in
        # place of continuity there.
        if (self._bc_general is not None and self._n_vars() < 2
                and not getattr(self, "_periodic", False)):
            jbps = self._detect_jump_breakpoints()
            if jbps:
                return self._solve_piecewise(
                    f, n=n, max_iter=max_iter, extra_breaks=jbps)

        # Systems of ODEs: op signature (x, u, v, ...) with >= 2
        # unknowns dispatches to block collocation (linear) or Newton
        # on top of it (nonlinear); periodic systems use a Fourier
        # (equispaced/trig) discretization with no BC rows.
        if self._n_vars() >= 2:
            if getattr(self, "_periodic", False):
                return self._solve_periodic_system(
                    f, n=n, max_iter=max_iter)
            # First-order explicit IVP systems (all BCs at one end)
            # time-march like MATLAB routes to ode113.  A general .bc
            # constraint or an unknown parameter means the problem is a
            # genuine (coupled) BVP, not an initial-value march.
            if (
                self._bc_general is None
                and self._n_params() == 0
                and (self._lbc_raw is None) != (self._rbc_raw is None)
            ):
                try:
                    return self._solve_ivp_system(f)
                except Exception:
                    pass
            if self._system_is_linear():
                return self._solve_linear_system(f, n=n)
            return self._solve_nonlinear_system(
                f, n=n, max_iter=max_iter)

        # Periodic BVPs use Fourier collocation (task #24, Opus 4.8).
        if getattr(self, "_periodic", False):
            return self._solve_periodic(f, n=n, n_max=n_max, tol=tol)

        # Deflated operators (G = M(u; r) N(u)) use a dedicated globalized
        # solve: the multiplicative factor makes plain undamped Newton from a
        # zero guess diverge, so a damped Newton with a boundary-condition
        # satisfying default initial guess is required to reach a NEW root.
        if self._deflation is not None:
            return self._solve_deflated(
                f, n=n, n_min=n_min, n_max=n_max, tol=tol,
                max_iter=max_iter, newton_tol=newton_tol,
            )

        # A scalar problem carrying a general .bc constraint (conditions the
        # user evaluates at arbitrary points, e.g. an interior u(0)) is
        # assembled by the block collocation solver, which supports the
        # general-constraint row placement.  A single-block system reduces to
        # the scalar collocation matrix; unwrap to a plain Chebfun.
        if self._bc_general is not None:
            if self._system_is_linear():
                sol = self._solve_linear_system(f, n=n)
            else:
                sol = self._solve_nonlinear_system(f, n=n, max_iter=max_iter)
            return sol[0] if len(sol) == 1 else sol

        # IVPs (all BCs at one endpoint) time-march like MATLAB (#24).
        # Fall back to collocation if the extraction fails.
        if n is None and self._is_ivp():
            try:
                return self.solve_ivp(f)
            except Exception:
                pass

        # Try to detect linearity by checking if operator is an OperatorBlock
        if self._is_linear():
            return self._solve_linear(f, n=n, n_min=n_min, n_max=n_max, tol=tol)
        else:
            return self._solve_nonlinear(
                f, n=n, n_min=n_min, n_max=n_max, tol=tol,
                max_iter=max_iter, newton_tol=newton_tol,
            )

    def deflate(self, r, p, alp, type="L2"):
        """Deflate known solutions ``r`` from this operator.

        Returns a new :class:`Chebop` whose operator ``G`` is the original
        operator ``N`` scaled by the deflation factor ``M``::

            G(u) = M(u; r) * N(u)

        with, for a single deflated root and ``type='L2'``,

        .. math::

            M(u; r) = \\frac{1}{\\|u - r\\|^{p}} + \\alpha,

        and for several deflated roots :math:`r_1, \\dots, r_n`

        .. math::

            M(u; r_1, \\dots, r_n)
                = \\frac{1}{\\|u - r_1\\|^{p} \\cdots \\|u - r_n\\|^{p}}
                  + \\alpha .

        Solving ``G(u) = 0`` with Newton then converges to a solution of the
        original problem *distinct* from every deflated root: the singularity
        of ``M`` at each known root repels the iteration.  Only the returned
        operator is modified; the boundary conditions, initial guess and
        domain are carried over unchanged, so ``Ndef.solve(0)`` (MATLAB's
        ``Ndef \\ 0``) computes a new solution.

        Only scalar problems are supported, matching MATLAB @chebop/deflate.

        Parameters
        ----------
        r : Chebfun or list of Chebfun or SystemSolution
            Previously found solution(s) to deflate.
        p : float
            Power coefficient of the deflation scheme.
        alp : float
            Shift coefficient of the deflation scheme.
        type : {'L2', 'H1'}, default 'L2'
            Norm used for deflation.  ``'H1'`` adds the derivative's L2 norm.

        Returns
        -------
        Chebop
            A new operator with the deflation factor applied.

        Provenance
        ----------
        MATLAB source : @chebop/deflate.m, @chebmatrix/deflationFun.m,
            @chebfun/deflationFun.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        References
        ----------
        P. E. Farrell, A. Birkisson, S. W. Funke, "Deflation techniques for
        finding distinct solutions of nonlinear partial differential
        equations", SIAM J. Sci. Comput. 37 (2015).
        """
        if self.op is None:
            raise ValueError("Chebop.deflate: operator is not set.")
        if self._n_vars() > 1:
            raise ValueError(
                "Chebop.deflate: only scalar problems are supported "
                "(mirrors MATLAB @chebop/deflate)."
            )
        if type not in ("L2", "H1"):
            raise ValueError(
                f"Chebop.deflate: unknown norm type {type!r} "
                "(expected 'L2' or 'H1')."
            )

        roots = _normalize_deflation_roots(r)

        N2 = Chebop(domain=self.domain)
        N2.op = _make_deflated_op(self.op, roots, float(p), float(alp), type)
        # Carry over the boundary conditions, initial guess and periodicity.
        N2._lbc_raw = self._lbc_raw
        N2._rbc_raw = self._rbc_raw
        N2._bc_general = self._bc_general
        N2._bc_show = self._bc_show
        N2._periodic = self._periodic
        N2.init = self.init
        N2._deflation = (self.op, roots, float(p), float(alp), type)
        return N2

    def _n_vars(self) -> int:
        """Number of unknown functions (op arity minus the x arg)."""
        import inspect
        try:
            params = [
                q for q in
                inspect.signature(self.op).parameters.values()
                if q.kind in (q.POSITIONAL_ONLY,
                              q.POSITIONAL_OR_KEYWORD)
                and q.default is inspect.Parameter.empty
            ]
            return max(1, len(params) - 1)
        except (TypeError, ValueError):
            return 1

    def _n_equations(self) -> int:
        """Number of differential equations the operator returns.

        Probed by evaluating ``op`` on zero functions and counting outputs.
        When this is fewer than :meth:`_n_vars`, the trailing unknowns are
        scalar *parameters* (MATLAB @chebop treats extra unknowns pinned by
        boundary conditions as constants determined by the linearization).
        """
        m = self._n_vars()
        try:
            from chebfunjax.chebfun1d.chebfun import Chebfun
            x_fun = Chebfun.identity(Domain(self.domain))
            zero = _chebfun_from_values(jnp.zeros(2), self.domain)
            out = self.op(x_fun, *([zero] * m))
        except Exception:
            return m
        if isinstance(out, (list, tuple)):
            return len(out)
        return 1

    def _n_params(self) -> int:
        """Number of unknown scalar parameters (m - number of equations).

        A parameter is an extra trailing argument of ``op`` for which the
        operator returns no differential equation; it is carried in the
        square system as a constant unknown (``p' = 0``) pinned by a boundary
        condition, matching MATLAB @chebop's parameter treatment.
        """
        if self._n_vars() < 2:
            return 0
        return max(0, self._n_vars() - self._n_equations())

    def linop(self):
        """Linearize this CHEBOP and return its typed block operator.

        Probes the operator with one system-aware linearization variable per
        unknown (see :class:`_LinopVar`) and assembles a
        :class:`~chebfunjax.operators.chebmatrix.ChebMatrix` whose ``blocks``
        are *typed* exactly as MATLAB ``linop(N).blocks``:

        * a block is an :class:`~chebfunjax.operators.blocks.OperatorBlock`
          when its unknown is differentiated or integrated *somewhere* in the
          system (or when nothing in the system is differentiated/integrated at
          all -- then every unknown is a genuine function, hence an operator);
        * otherwise the unknown is a *parameter* (never differentiated while
          something else is) and its block collapses to a
          :class:`~chebfunjax.chebfun1d.chebfun.Chebfun` -- the multiplicative
          coefficient of that unknown in the equation.

        This mirrors the ``isParam`` bookkeeping of MATLAB
        ``@chebop/linearize.m``::

            isParam = any(any(~isNotDiffOrInt)) & all(isNotDiffOrInt, 1);

        Returns
        -------
        ChebMatrix
            ``n_eq x m`` block matrix (``n_eq`` equations, ``m`` unknowns).
            For a scalar equation it is ``1 x m`` and linear indexing
            (``L.linop()[j]``) recovers block ``j``.

        Provenance
        ----------
        MATLAB source : @chebop/linop.m, @chebop/linearize.m, @linop/linop.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        ChebMatrix, OperatorBlock
        """
        import inspect

        from chebfunjax.operators.chebmatrix import ChebMatrix

        m = self._n_vars()
        domain = self.domain

        if self.op is None:
            raise ValueError("Chebop.linop: operator is not set.")

        # Seed one system-linearization variable per unknown: variable j has an
        # identity block in column j and structural zeros elsewhere.
        seeds = []
        for j in range(m):
            jac = [None] * m
            jac[j] = _I_block(domain)
            seeds.append(_LinopVar(jac, domain))

        from chebfunjax.chebfun1d.chebfun import Chebfun
        x_fun = Chebfun.identity(Domain(domain))
        try:
            nargs = len(inspect.signature(self.op).parameters)
        except (TypeError, ValueError):
            nargs = m + 1
        out = self.op(x_fun, *seeds) if nargs > m else self.op(*seeds)

        # Normalize the output into a list of equation rows.
        if isinstance(out, _LinopVar):
            equations = [out]
        elif isinstance(out, (list, tuple)):
            equations = list(out)
        else:
            # An equation that is a bare affine part (no unknown) -> zero row.
            equations = [out]

        rows = []
        for eq in equations:
            if isinstance(eq, _LinopVar):
                rows.append(list(eq.jac))
            else:
                rows.append([None] * m)

        # Column classification (MATLAB isParam): a variable is a parameter iff
        # something in the system is differentiated/integrated AND that variable
        # never is.
        def _is_diffint(blk):
            return blk is not None and getattr(blk, "order", 0) != 0

        any_diffint = any(_is_diffint(blk) for row in rows for blk in row)
        is_param = [
            any_diffint and not any(_is_diffint(row[j]) for row in rows)
            for j in range(m)
        ]

        one_fun = _chebfun_ones(domain)
        typed_rows = []
        for row in rows:
            typed_row = []
            for j, blk in enumerate(row):
                if is_param[j]:
                    typed_row.append(_block_to_coefficient(blk, one_fun, domain))
                else:
                    typed_row.append(blk if blk is not None
                                     else _zero_operator(domain))
            typed_rows.append(typed_row)

        return ChebMatrix(typed_rows, domain=domain)

    def _eigs_system(self, k: int = 6, n: int = 64):
        """Eigenvalues of a linear system of ODEs (MATLAB eigs for
        chebmatrix operators): the block collocation matrix with BC
        rows enforced through a generalized eigenproblem
        A U = lam B U, where B is the identity with zero rows at the
        BC positions (spurious infinite eigenvalues filtered out).

        Returns ``(V, lam)``: a list of SystemSolution eigenfunctions
        and the k eigenvalues closest to zero (by magnitude).

        Provenance
        ----------
        MATLAB source : @chebop/eigs.m (system branch), @linop/eigs.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        import scipy.linalg as _sla

        from chebfunjax.chebfun1d.chebfun import Chebfun
        m = self._n_vars()
        a, b = self.domain
        kk = _np.arange(n)
        xg = _np.cos(_np.pi * kk / (n - 1))[::-1]
        xp = a + (b - a) * (xg + 1.0) / 2.0
        x_fun = Chebfun.identity(Domain(self.domain))
        zero = _chebfun_from_values(jnp.zeros(2), self.domain)

        def apply_op(us):
            out = self._call_op(x_fun, us)
            if not isinstance(out, (list, tuple)):
                out = [out]
            return [zero + o if isinstance(o, (int, float)) else o
                    for o in out]

        def rows_at(fun_list, pts):
            return _np.concatenate([
                _np.asarray(g(jnp.asarray(pts))) for g in fun_list])

        zeros_list = [zero for _ in range(m)]
        base = rows_at(apply_op(zeros_list), xp)
        A = _np.zeros((m * n, m * n))
        for var in range(m):
            for j in range(n):
                vals = _np.zeros(n)
                vals[j] = 1.0
                probe = list(zeros_list)
                probe[var] = _chebfun_from_values(
                    jnp.asarray(vals), self.domain)
                A[:, var * n + j] = \
                    rows_at(apply_op(probe), xp) - base

        def bc_rows(bc_raw, x0):
            if bc_raw is None:
                return []
            def bc_list(us):
                out = bc_raw(*us)
                if not isinstance(out, (list, tuple)):
                    out = [out]
                return [zero + o
                        if isinstance(o, (int, float)) else o
                        for o in out]
            base_bc = _np.array([
                _eval_chebfun_at(g, x0)
                for g in bc_list(zeros_list)])
            R = _np.zeros((len(base_bc), m * n))
            for var in range(m):
                for j in range(n):
                    vals = _np.zeros(n)
                    vals[j] = 1.0
                    probe = list(zeros_list)
                    probe[var] = _chebfun_from_values(
                        jnp.asarray(vals), self.domain)
                    R[:, var * n + j] = _np.array([
                        _eval_chebfun_at(g, x0)
                        for g in bc_list(probe)]) - base_bc
            return list(R)

        B = _np.eye(m * n)
        ridx = 0
        for row in bc_rows(self._lbc_raw, a):
            r = (ridx % m) * n
            A[r, :] = row
            B[r, :] = 0.0
            ridx += 1
        ridx = 0
        for row in bc_rows(self._rbc_raw, b):
            r = (ridx % m) * n + n - 1
            A[r, :] = row
            B[r, :] = 0.0
            ridx += 1

        lam, W = _sla.eig(A, B)
        finite = _np.isfinite(lam) & (_np.abs(lam) < 1e8)
        lam, W = lam[finite], W[:, finite]
        # Two-resolution agreement filter: spurious discrete
        # eigenvalues move when n changes; genuine ones stay.
        lam2 = self._system_eig_values(n + 17)
        keep = []
        used = set()
        for i in _np.argsort(_np.abs(lam)):
            d = _np.abs(lam2 - lam[i])
            j = int(_np.argmin(d))
            if d[j] < 1e-6 * max(1.0, abs(lam[i])) and j not in used:
                keep.append(i)
                used.add(j)
            if len(keep) >= k:
                break
        order = _np.asarray(keep, dtype=int)
        lam, W = lam[order], W[:, order]
        V = []
        for i in range(len(lam)):
            w = W[:, i]
            wmax = w[_np.argmax(_np.abs(w))]
            w = _np.real(w / wmax) if _np.max(
                _np.abs(_np.imag(w / wmax))) < 1e-8 else w
            comps = []
            for vi in range(m):
                vals = w[vi * n: (vi + 1) * n]
                comps.append(_chebfun_from_values(
                    jnp.asarray(_np.real(vals)), self.domain))
            V.append(SystemSolution(comps))
        return V, jnp.asarray(lam)

    def _system_eig_values(self, n: int):
        """Eigenvalues only, at resolution n (helper for the
        two-resolution spurious-mode filter in _eigs_system)."""
        import numpy as _np
        import scipy.linalg as _sla

        from chebfunjax.chebfun1d.chebfun import Chebfun
        m = self._n_vars()
        a, b = self.domain
        kk = _np.arange(n)
        xg = _np.cos(_np.pi * kk / (n - 1))[::-1]
        xp = a + (b - a) * (xg + 1.0) / 2.0
        x_fun = Chebfun.identity(Domain(self.domain))
        zero = _chebfun_from_values(jnp.zeros(2), self.domain)

        def apply_op(us):
            out = self._call_op(x_fun, us)
            if not isinstance(out, (list, tuple)):
                out = [out]
            return [zero + o if isinstance(o, (int, float)) else o
                    for o in out]

        def rows_at(fun_list, pts):
            return _np.concatenate([
                _np.asarray(g(jnp.asarray(pts))) for g in fun_list])

        zeros_list = [zero for _ in range(m)]
        base = rows_at(apply_op(zeros_list), xp)
        A = _np.zeros((m * n, m * n))
        for var in range(m):
            for j in range(n):
                vals = _np.zeros(n)
                vals[j] = 1.0
                probe = list(zeros_list)
                probe[var] = _chebfun_from_values(
                    jnp.asarray(vals), self.domain)
                A[:, var * n + j] = \
                    rows_at(apply_op(probe), xp) - base

        def bc_rows(bc_raw, x0):
            if bc_raw is None:
                return []

            def bc_list(us):
                out = bc_raw(*us)
                if not isinstance(out, (list, tuple)):
                    out = [out]
                return [zero + o
                        if isinstance(o, (int, float)) else o
                        for o in out]

            base_bc = _np.array([
                _eval_chebfun_at(g, x0)
                for g in bc_list(zeros_list)])
            R = _np.zeros((len(base_bc), m * n))
            for var in range(m):
                for j in range(n):
                    vals = _np.zeros(n)
                    vals[j] = 1.0
                    probe = list(zeros_list)
                    probe[var] = _chebfun_from_values(
                        jnp.asarray(vals), self.domain)
                    R[:, var * n + j] = _np.array([
                        _eval_chebfun_at(g, x0)
                        for g in bc_list(probe)]) - base_bc
            return list(R)

        B = _np.eye(m * n)
        ridx = 0
        for row in bc_rows(self._lbc_raw, a):
            r = (ridx % m) * n
            A[r, :] = row
            B[r, :] = 0.0
            ridx += 1
        ridx = 0
        for row in bc_rows(self._rbc_raw, b):
            r = (ridx % m) * n + n - 1
            A[r, :] = row
            B[r, :] = 0.0
            ridx += 1
        lam = _sla.eigvals(A, B)
        return lam[_np.isfinite(lam) & (_np.abs(lam) < 1e8)]

    def _solve_periodic_system(self, f=0.0, n: int | None = None,
                               max_iter: int = 25):
        """Periodic systems of ODEs: Fourier collocation on an
        equispaced grid (periodicity built into the trig
        representation; no boundary rows), Newton if the op fails the
        superposition probe.  Least-squares solve absorbs the
        singular mean modes.

        Provenance
        ----------
        MATLAB source : @chebop/solvebvp.m ('periodic' + trigcolloc)
        Chebfun commit: 7574c77
        """
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
        from chebfunjax.tech.trigtech import Trigtech

        m = self._n_vars()
        a, b = self.domain
        if n is None:
            n = 64
        L = b - a
        xp = a + L * _np.arange(n) / n
        x_fun = Chebfun.identity(Domain(self.domain))

        def trig_fun(values):
            tech = Trigtech.from_values(
                jnp.asarray(values, dtype=jnp.float64))
            piece = _Piece(tech=tech, interval=(a, b))
            return Chebfun(funs=[piece], domain=Domain((a, b)))

        def to_funs(U):
            return [trig_fun(U[i * n: (i + 1) * n])
                    for i in range(m)]

        if isinstance(f, (int, float)):
            f_vals = _np.full(m * n, float(f))
        elif isinstance(f, (list, tuple)):
            f_vals = _np.concatenate([
                _np.full(n, float(fi))
                if isinstance(fi, (int, float))
                else _np.asarray(fi(jnp.asarray(xp))) for fi in f])
        else:
            f_vals = _np.tile(_np.asarray(f(jnp.asarray(xp))), m)

        def residual(U):
            us = to_funs(U)
            out = self._call_op(x_fun, us)
            if not isinstance(out, (list, tuple)):
                out = [out]
            R = _np.concatenate([
                _np.full(n, float(o))
                if isinstance(o, (int, float))
                else _np.asarray(o(jnp.asarray(xp))) for o in out])
            return R - f_vals

        U = _np.zeros(m * n)
        if self.init is not None:
            init = self.init if isinstance(self.init, (list, tuple)) \
                else [self.init] * m
            U = _np.concatenate([
                _np.full(n, float(gi))
                if isinstance(gi, (int, float))
                else _np.asarray(gi(jnp.asarray(xp)))
                for gi in init])
        R = residual(U)
        Rn = R
        for _it in range(max_iter):
            nrm = _np.max(_np.abs(R))
            if nrm < 1e-11:
                break
            J = _np.zeros((m * n, m * n))
            h = 1e-7 * max(1.0, _np.max(_np.abs(U)))
            for j in range(m * n):
                Up = U.copy()
                Up[j] += h
                J[:, j] = (residual(Up) - R) / h
            step = _np.linalg.lstsq(J, R, rcond=None)[0]
            lam = 1.0
            for _d in range(30):
                Rn = residual(U - lam * step)
                if _np.max(_np.abs(Rn)) < nrm or lam < 1e-4:
                    break
                lam *= 0.5
            U = U - lam * step
            R = Rn
        else:
            import warnings as _w
            _w.warn("chebop periodic system: max Newton iterations "
                    f"reached (residual {_np.max(_np.abs(R)):.2e})",
                    RuntimeWarning, stacklevel=2)
        return SystemSolution(to_funs(U))

    def _solve_ivp_system(self, f=0.0):
        """Time-march a first-order explicit IVP system (MATLAB
        routes these to ode113): equations of the form
        u_i' - g_i(t, u) = f_i with all conditions at one endpoint.
        The RHS is recovered by evaluating the op on CONSTANT
        chebfuns (derivatives vanish), and the initial values by
        solving the affine boundary residuals.

        Provenance
        ----------
        MATLAB source : @chebop/solveivp.m (system branch)
        Chebfun commit: 7574c77
        """
        import numpy as _np
        from scipy.integrate import solve_ivp as _sivp

        from chebfunjax.chebfun1d.chebfun import Chebfun
        m = self._n_vars()
        a, b = self.domain
        x_fun = Chebfun.identity(Domain(self.domain))
        forward = self._lbc_raw is not None
        t0, t1 = (a, b) if forward else (b, a)
        bc_raw = self._lbc_raw if forward else self._rbc_raw

        def const_funs(y):
            return [_chebfun_from_values(
                jnp.full(2, float(yj)), self.domain) for yj in y]

        if isinstance(f, (int, float)):
            fvals = [float(f)] * m
            f_of_t = None
        elif isinstance(f, (list, tuple)):
            fvals = f
            f_of_t = None
        else:
            fvals = None
            f_of_t = f

        def op_at(t, y):
            us = const_funs(y)
            out = self._call_op(x_fun, us)
            if not isinstance(out, (list, tuple)):
                out = [out]
            tt = jnp.asarray(t)
            vals = _np.array([
                float(o) if isinstance(o, (int, float))
                else float(o(tt)) for o in out])
            if fvals is not None:
                vals -= _np.array([
                    float(v) if isinstance(v, (int, float))
                    else float(v(tt)) for v in fvals])
            elif f_of_t is not None:
                vals -= float(f_of_t(tt))
            return vals

        def _sub_f(vals, t):
            if fvals is not None:
                vals -= _np.array([
                    float(v) if isinstance(v, (int, float))
                    else float(v(jnp.asarray(t))) for v in fvals])
            elif f_of_t is not None:
                vals -= float(f_of_t(jnp.asarray(t)))
            return vals

        import inspect as _inspect
        try:
            _onargs = len(_inspect.signature(self.op).parameters)
        except (TypeError, ValueError):
            _onargs = m + 1

        def op_at_fast(t, y):
            """Scalar-proxy evaluation of the op: thousands of times
            cheaper per time step than building constant chebfuns."""
            towers = [
                _IVPProxy([jnp.asarray(float(yj)), jnp.asarray(0.0)])
                for yj in y]
            targ = jnp.asarray(float(t))
            try:
                out = (self.op(targ, *towers) if _onargs > m
                       else self.op(*towers))
            except AttributeError:
                out = self.op(_TrigX(targ), *towers)
            if not isinstance(out, (list, tuple)):
                out = [out]
            vals = _np.array([
                float(_np.asarray(o.v if isinstance(o, _TrigX)
                                  else (o._v if isinstance(o, _IVPProxy)
                                        else o)))
                for o in out])
            return _sub_f(vals, t)

        # Use the scalar fast path when it reproduces the chebfun-based
        # evaluation at a probe point; exotic ops fall back.
        _yprobe = _np.linspace(0.3, 1.1, m)
        try:
            _fast_ok = bool(_np.allclose(
                op_at_fast(t0, _yprobe), op_at(t0, _yprobe),
                rtol=1e-9, atol=1e-9))
        except Exception:
            _fast_ok = False
        op_eval = op_at_fast if _fast_ok else op_at

        # verify first-order explicit form: residual affine in y'
        # with unit coefficient => R(t, y, y') = y' + op_at-part
        def rhs(t, y):
            return -op_eval(t, y)

        # initial values: solve the affine bc residuals bc(y0) = 0
        def bc_res(y):
            us = const_funs(y)
            out = bc_raw(*us)
            if not isinstance(out, (list, tuple)):
                out = [out]
            return _np.array([
                float(o) if isinstance(o, (int, float))
                else _eval_chebfun_at(o, t0) for o in out])

        r0 = bc_res(_np.zeros(m))
        if len(r0) != m:
            raise ValueError("ivp system needs m conditions")
        J = _np.zeros((m, m))
        for j in range(m):
            e = _np.zeros(m)
            e[j] = 1.0
            J[:, j] = bc_res(e) - r0
        y0 = _np.linalg.solve(J, -r0)

        sol = _sivp(rhs, (t0, t1), y0, method="LSODA",
                    rtol=1e-11, atol=1e-12, dense_output=True)
        if not sol.success:
            raise RuntimeError(f"ivp system: {sol.message}")

        # Build each component as a PIECEWISE chebfun on the solver's
        # own time mesh (MATLAB @chebfun/constructODEsol).  A single
        # global polynomial is accurate only relative to its global
        # vscale, so a trajectory spanning many orders of magnitude
        # (e.g. two Lorenz orbits separating from 1e-9) evaluates to
        # cancellation noise early on; local pieces keep local accuracy.
        from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece

        ts = _np.asarray(sol.t, dtype=float)
        if ts[0] > ts[-1]:
            ts = ts[::-1]
        ts = _np.unique(_np.clip(ts, min(a, b), max(a, b)))
        # Group solver steps into pieces (a piece per step would give
        # thousands of tiny funs); ~16 steps per piece keeps each fun
        # low-degree while bounding their number.
        step = max(1, int(_np.ceil(ts.size / 256.0)), 8)
        idx = list(range(0, ts.size - 1, step))
        breaks = [float(a)] + [float(ts[i]) for i in idx[1:]] + [float(b)]
        breaks = sorted(set(breaks))
        span = abs(float(b) - float(a))
        breaks = [breaks[0]] + [
            v for k, v in enumerate(breaks[1:], 1)
            if v - breaks[k - 1] > 1e-13 * span]
        if breaks[-1] != float(b):
            breaks[-1] = float(b)

        def _component(i):
            def _ev(t, _i=i):
                tt = _np.atleast_1d(_np.asarray(t, dtype=float))
                vals = sol.sol(tt)[_i]
                return jnp.asarray(
                    vals.reshape(_np.shape(t)) if _np.ndim(t) else vals[0],
                    dtype=jnp.float64)
            return _ev

        try:
            out = []
            for i in range(m):
                ev = _component(i)
                funs = [_Piece.from_function(ev, breaks[k], breaks[k + 1])
                        for k in range(len(breaks) - 1)]
                out.append(Chebfun(funs=funs,
                                   domain=Domain(tuple(breaks))))
            return SystemSolution(out)
        except Exception:
            # Fall back to a single global representation.
            nn = int(min(8193, max(257, 4 * sol.t.size)))
            kk = _np.arange(nn)
            xg = _np.cos(_np.pi * kk / (nn - 1))[::-1]
            xp = a + (b - a) * (xg + 1.0) / 2.0
            Y = sol.sol(xp)
            return SystemSolution([
                _chebfun_from_values(
                    jnp.asarray(Y[i]), self.domain).simplify()
                for i in range(m)
            ])

    def eigs_generalized(self, B: "Chebop", k: int = 6,
                         n: int = 96, sort: str = "SM"):
        """Generalized eigenvalue problem A u = lambda B u for two
        linear chebops (MATLAB eigs(A, B, k)).  Both operators are
        assembled by basis probing on the collocation grid (complex
        operators keep complex matrices); A carries the boundary rows,
        B is zeroed there; spurious modes are removed with a
        two-resolution agreement filter.

        Boundary conditions may be 'dirichlet'/'neumann' keywords,
        scalars, or callables returning one or MORE conditions (e.g.
        MATLAB's clamped ``@(u) [u; diff(u)]``); each condition
        replaces one collocation row from the corresponding end.

        ``sort='SM'`` keeps the smallest-magnitude modes (default);
        ``'LR'`` the largest real part (MATLAB eigs(..., 'LR')).

        Provenance
        ----------
        MATLAB source : @chebop/eigs.m (generalized branch)
        Chebfun commit: 7574c77
        """
        import numpy as _np
        import scipy.linalg as _sla

        # Piecewise-coefficient operators (e.g. a mass term B with a
        # discontinuous coefficient F(x)) carry interior breakpoints that a
        # single Chebyshev grid cannot resolve -- the eigenvalues converge
        # too slowly for the agreement filter.  MATLAB breaks the domain at
        # the coefficient discontinuities and collocates piece-by-piece; we
        # mirror that with a block-diagonal pencil plus continuity rows in
        # BOTH matrices when interior breaks are detected.
        _bks = self._generalized_breakpoints(B)
        if len(_bks) > 2:
            # 32 points per piece matches MATLAB's default eigs stopping
            # dimension for this class of piecewise-smooth eigenfunctions, so
            # the returned eigenvalues carry the same discretization error as
            # the reference values (over-refining moves away from them).
            return self._eigs_generalized_piecewise(
                B, _bks, k=k, nn=(int(n) if (n and 16 <= n <= 40) else 32),
                sort=sort)

        def assemble(nn):
            from chebfunjax.chebfun1d.chebfun import Chebfun
            a, b = self.domain
            kk = _np.arange(nn)
            xg = _np.cos(_np.pi * kk / (nn - 1))[::-1]
            xp = a + (b - a) * (xg + 1.0) / 2.0
            x_fun = Chebfun.identity(Domain(self.domain))

            def basis(j):
                vals = _np.zeros(nn)
                vals[j] = 1.0
                return _chebfun_from_values(
                    jnp.asarray(vals), self.domain)

            def probe_matrix(op):
                M = _np.zeros((nn, nn), dtype=complex)
                for j in range(nn):
                    out = op(x_fun, basis(j))
                    M[:, j] = _np.asarray(out(jnp.asarray(xp)))
                if _np.max(_np.abs(M.imag)) == 0.0:
                    return M.real
                return M

            Am = probe_matrix(self.op)
            Bm = probe_matrix(B.op)
            dt = _np.result_type(Am.dtype, Bm.dtype)
            Am = Am.astype(dt)
            Bm = Bm.astype(dt)
            # boundary rows: keywords, scalars, or callables that may
            # return several conditions (clamped etc.)
            Du = None

            def bc_rows(kind, endpoint_x, idx):
                nonlocal Du
                if callable(kind):
                    # Probe each condition: rows[i, j] = c_i(e_j) at
                    # the endpoint (MATLAB bc functional evaluation).
                    rows = None
                    xe = jnp.asarray(float(endpoint_x))
                    for j in range(nn):
                        out = kind(basis(j))
                        if not isinstance(out, (list, tuple)):
                            out = [out]
                        if rows is None:
                            rows = _np.zeros((len(out), nn),
                                             dtype=complex)
                        for i, o in enumerate(out):
                            rows[i, j] = complex(
                                _np.asarray(o(xe), dtype=complex))
                    if _np.max(_np.abs(rows.imag)) == 0.0:
                        rows = rows.real
                    return rows
                row = _np.zeros(nn)
                if kind in ("dirichlet", 0.0, None):
                    row[idx] = 1.0
                    return row[None, :]
                if kind == "neumann":
                    if Du is None:
                        Duloc = _np.zeros((nn, nn))
                        for j in range(nn):
                            Duloc[:, j] = _np.asarray(
                                basis(j).diff()(jnp.asarray(xp)))
                        Du = Duloc
                    return Du[idx][None, :]
                raise ValueError(f"unsupported bc {kind!r}")

            lb = self._lbc_raw
            rb = self._rbc_raw
            lb_kind = lb if isinstance(lb, str) or callable(lb) else (
                "dirichlet" if isinstance(lb, (int, float))
                or lb is None else lb)
            rb_kind = rb if isinstance(rb, str) or callable(rb) else (
                "dirichlet" if isinstance(rb, (int, float))
                or rb is None else rb)
            if lb is not None:
                rows = bc_rows(lb_kind, a, 0)
                if _np.iscomplexobj(rows) and not _np.iscomplexobj(Am):
                    Am = Am.astype(complex)
                for i in range(rows.shape[0]):
                    Am[i, :] = rows[i]
                    Bm[i, :] = 0.0
            if rb is not None:
                rows = bc_rows(rb_kind, b, nn - 1)
                if _np.iscomplexobj(rows) and not _np.iscomplexobj(Am):
                    Am = Am.astype(complex)
                for i in range(rows.shape[0]):
                    Am[nn - 1 - i, :] = rows[i]
                    Bm[nn - 1 - i, :] = 0.0
            lam, W = _sla.eig(Am, Bm)
            fin = _np.isfinite(lam) & (_np.abs(lam) < 1e10)
            return lam[fin], W[:, fin], xp

        if sort == "LR":
            def rank_order(lams):
                return _np.argsort(-_np.real(lams))
        else:
            def rank_order(lams):
                return _np.argsort(_np.abs(lams))

        lam, W, xp = assemble(n)
        lam2, _, _ = assemble(n + 17)
        # Agreement filter: 1e-4 relative (higher-order operators
        # converge slowly enough that 1e-6 rejected GENUINE modes --
        # e.g. the clamped beam's beta_1^4 moves by ~1e-6 relative
        # between n and n+17); spurious modes disagree by orders of
        # magnitude.  The FINER resolution's eigenvalue is returned.
        keep, used, fine = [], set(), []
        for i in rank_order(lam):
            d = _np.abs(lam2 - lam[i])
            j = int(_np.argmin(d))
            if d[j] < 1e-4 * max(1.0, abs(lam[i])) and j not in used:
                keep.append(i)
                used.add(j)
                fine.append(lam2[j])
            if len(keep) >= k:
                break
        order = _np.asarray(keep, dtype=int)
        lam = _np.asarray(fine)
        V = []
        for i in order:
            w = W[:, i]
            if _np.max(_np.abs(w.imag)) > 1e-13 * _np.max(_np.abs(w)):
                V.append(
                    _chebfun_from_values(
                        jnp.asarray(w.real), self.domain)
                    + 1j * _chebfun_from_values(
                        jnp.asarray(w.imag), self.domain))
            else:
                V.append(_chebfun_from_values(
                    jnp.asarray(w.real), self.domain))
        return V, jnp.asarray(lam)

    def _generalized_breakpoints(self, B: "Chebop") -> list[float]:
        """Union of the domain endpoints with any coefficient-induced
        interior breakpoints of the pencil ``(A, B)``.

        Probes ``self.op`` and ``B.op`` on a smooth identity input over the
        base domain and reads back the breakpoints of the outputs, so a
        discontinuous coefficient (e.g. an indicator mass term) is detected
        and its jump locations are returned as breakpoints.

        Provenance
        ----------
        MATLAB source : @linop/eigs.m (piecewise discretization setup)
        Chebfun commit: 7574c77
        """
        import inspect

        from chebfunjax.chebfun1d.chebfun import Chebfun
        a, b = self.domain
        bps = [float(a), float(b)]
        dom0 = Domain((a, b))
        xf = Chebfun.identity(dom0)
        probe = Chebfun.identity(dom0)
        for op in (self.op, B.op):
            if op is None:
                continue
            try:
                nargs = len(inspect.signature(op).parameters)
                out = op(xf, probe) if nargs >= 2 else op(probe)
                outs = out if isinstance(out, (list, tuple)) else [out]
                for o in outs:
                    if not hasattr(o, "domain"):
                        continue
                    a0, b0 = float(o.domain.a), float(o.domain.b)
                    dlt = 1e-9 * (b0 - a0)
                    for bb in o.domain.breakpoints:
                        bf = float(bb)
                        if not (a0 < bf < b0):
                            continue
                        if any(abs(bf - e) <= 1e-12 for e in bps):
                            continue
                        # Keep a breakpoint only where the operator output is
                        # genuinely discontinuous (value or slope jump): the
                        # root of |x| at 0 injects a removable break where the
                        # coefficient is smooth, which MATLAB merges away.
                        vl = float(o(jnp.asarray(bf - dlt)))
                        vr = float(o(jnp.asarray(bf + dlt)))
                        dl = (vl - float(o(jnp.asarray(bf - 2 * dlt)))) / dlt
                        dr = (float(o(jnp.asarray(bf + 2 * dlt))) - vr) / dlt
                        scale = max(abs(vl), abs(vr), 1.0)
                        dscale = max(abs(dl), abs(dr), 1.0)
                        if (abs(vl - vr) > 1e-7 * scale
                                or abs(dl - dr) > 1e-6 * dscale):
                            bps.append(bf)
            except Exception:
                pass
        return sorted(bps)

    def _eigs_generalized_piecewise(self, B: "Chebop", bps: list[float],
                                    k: int = 6, nn: int = 48,
                                    sort: str = "SM"):
        """Generalized eigenproblem ``A u = lambda B u`` on a domain with
        interior breakpoints (piecewise-coefficient operators).

        Each unknown is represented by ``P`` Chebyshev pieces of ``nn``
        values.  Both ``A`` and ``B`` are assembled block-diagonally by
        probing per-piece delta bases; the boundary conditions and the
        continuity of ``u`` and its derivatives ``0..order-1`` across every
        interior break replace collocation rows in ``A`` (the corresponding
        rows are zeroed in ``B``, exactly as MATLAB enforces constraints on
        the left operator only).  A two-resolution agreement filter removes
        the spurious modes that a singular ``B`` (zero mass on a piece)
        introduces.

        Provenance
        ----------
        MATLAB source : @linop/eigs.m, @chebop/eigs.m (piecewise branch)
        Chebfun commit: 7574c77
        """
        import inspect

        import numpy as _np
        import scipy.linalg as _sla

        from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece

        order = int(self._piecewise_orders(1)[0])
        if order not in (1, 2):
            raise NotImplementedError(
                "piecewise generalized eigs supports operators of "
                f"differential order 1 or 2; got order {order}.")

        def _apply(op, u, dom, xf):
            try:
                nargs = len(inspect.signature(op).parameters)
            except (TypeError, ValueError):
                nargs = 2
            return op(xf, u) if nargs >= 2 else op(u)

        def assemble(m):
            P = len(bps) - 1
            ints = [(bps[p], bps[p + 1]) for p in range(P)]
            dom = Domain(tuple(bps))
            xf = Chebfun.identity(dom)
            kk = _np.arange(m)
            tref = _np.cos(_np.pi * kk / (m - 1))[::-1]
            xps = [ints[p][0] + (ints[p][1] - ints[p][0]) * (tref + 1.0) / 2.0
                   for p in range(P)]
            Pn = P * m

            def basis(p, j):
                funs = []
                for q in range(P):
                    v = _np.zeros(m)
                    if q == p:
                        v[j] = 1.0
                    funs.append(_Piece.from_values(
                        jnp.asarray(v), ints[q][0], ints[q][1]))
                return Chebfun(funs=funs, domain=dom)

            def probe(op):
                M = _np.zeros((Pn, Pn), dtype=complex)
                for p in range(P):
                    for j in range(m):
                        o = _apply(op, basis(p, j), dom, xf)
                        if hasattr(o, "domain"):
                            col = _np.concatenate([
                                _np.asarray(o(jnp.asarray(xps[q])),
                                            dtype=complex)
                                for q in range(P)])
                        else:
                            col = _np.full(Pn, complex(o))
                        M[:, p * m + j] = col
                if _np.max(_np.abs(M.imag)) == 0.0:
                    return M.real
                return M

            Am = probe(self.op)
            Bm = probe(B.op)
            dt = _np.result_type(Am.dtype, Bm.dtype)
            Am = Am.astype(dt)
            Bm = Bm.astype(dt)

            # Endpoint/derivative functional of piece p as a full-length row.
            def frow(p, deriv, xpt):
                r = _np.zeros(Pn)
                for j in range(m):
                    v = _np.zeros(m)
                    v[j] = 1.0
                    pc = _Piece.from_values(
                        jnp.asarray(v), ints[p][0], ints[p][1])
                    xj = jnp.asarray(xpt)
                    val = pc.diff(deriv)(xj) if deriv > 0 else pc(xj)
                    r[p * m + j] = float(val)
                return r

            def bc_row(kind, p, xpt):
                if callable(kind):
                    return frow(p, 0, xpt)   # value functional (homogeneous)
                if kind == "neumann":
                    return frow(p, 1, xpt)
                return frow(p, 0, xpt)       # scalar / None / dirichlet

            constraints = []                 # (row_index, row_vector)
            if self._lbc_raw is not None:
                constraints.append((0, bc_row(self._lbc_raw, 0, bps[0])))
            if self._rbc_raw is not None:
                constraints.append(
                    (Pn - 1, bc_row(self._rbc_raw, P - 1, bps[-1])))
            for p in range(1, P):
                for d in range(order):
                    row = frow(p - 1, d, bps[p]) - frow(p, d, bps[p])
                    ridx = (p - 1) * m + (m - 1) if d == 0 else p * m
                    constraints.append((ridx, row))

            for ridx, row in constraints:
                Am[ridx, :] = row
                Bm[ridx, :] = 0.0

            lam, W = _sla.eig(Am, Bm)
            fin = _np.isfinite(lam) & (_np.abs(lam) < 1e10)
            return lam[fin], W[:, fin]

        if sort == "LR":
            def rank_order(lams):
                return _np.argsort(-_np.real(lams))
        else:
            def rank_order(lams):
                return _np.argsort(_np.abs(lams))

        lam, W = assemble(nn)
        lam2, _ = assemble(nn + 16)
        # Two-resolution agreement removes the spurious modes a singular B
        # (zero mass on the outer pieces) injects, but the *coarse* value is
        # returned: MATLAB's eigs stops at a finite per-piece dimension, so
        # its reference eigenvalues carry the corresponding discretization
        # error -- over-refining to the true value would move away from it.
        keep, used, coarse = [], set(), []
        for i in rank_order(lam):
            d = _np.abs(lam2 - lam[i])
            j = int(_np.argmin(d))
            if d[j] < 1e-4 * max(1.0, abs(lam[i])) and j not in used:
                keep.append(i)
                used.add(j)
                coarse.append(lam[i])
            if len(keep) >= k:
                break
        lam_out = _np.asarray(coarse)
        V = []
        dom = Domain(tuple(bps))
        for i in keep:
            w = W[:, i]
            wr = w.real if _np.max(_np.abs(w.real)) >= _np.max(
                _np.abs(w.imag)) else w.imag
            P = len(bps) - 1
            funs = [_Piece.from_values(
                jnp.asarray(wr[p * nn:(p + 1) * nn]),
                bps[p], bps[p + 1]) for p in range(P)]
            V.append(Chebfun(funs=funs, domain=dom))
        return V, jnp.asarray(lam_out)

    def _system_is_linear(self) -> bool:
        """Superposition check for system ops:
        op(u+v) == op(u) + op(v) - op(0) at random probes."""
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import Chebfun
        m = self._n_vars()
        a, b = self.domain
        x_fun = Chebfun.identity(Domain(self.domain))
        rng = _np.random.default_rng(7)
        xs = jnp.asarray(a + (b - a) * rng.random(7))

        def mk(seed):
            r = _np.random.default_rng(seed)
            return [
                _chebfun_from_values(
                    jnp.asarray(r.standard_normal(6)), self.domain)
                for _ in range(m)
            ]

        import inspect as _inspect
        try:
            _nargs = len(_inspect.signature(self.op).parameters)
        except (TypeError, ValueError):
            _nargs = m + 1

        def ev(us):
            out = (self.op(x_fun, *us) if _nargs > m
                   else self.op(*us))
            if not isinstance(out, (list, tuple)):
                out = [out]
            return _np.concatenate([
                _np.full(7, float(o)) if isinstance(o, (int, float))
                else _np.asarray(o(xs)) for o in out])

        u1, u2 = mk(1), mk(2)
        z = [u * 0.0 for u in u1]
        lhs = ev([p + q for p, q in zip(u1, u2)])
        rhs = ev(u1) + ev(u2) - ev(z)
        scale = max(_np.max(_np.abs(lhs)), 1.0)
        return bool(_np.max(_np.abs(lhs - rhs)) < 1e-9 * scale)

    def _solve_nonlinear_system(self, f=0.0, n: int | None = None,
                                max_iter: int = 25):
        """Newton iteration for nonlinear systems: at each iterate the
        residual Jacobian is assembled by finite-difference probing of
        the op (same block layout as the linear solver), with damped
        updates.

        Provenance
        ----------
        MATLAB source : @chebop/solvebvpNonlinear.m (system branch)
        Chebfun commit: 7574c77
        """
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import Chebfun
        m = self._n_vars()
        a, b = self.domain
        if n is None:
            n = 48
        k = _np.arange(n)
        xg = _np.cos(_np.pi * k / (n - 1))[::-1]
        xp = a + (b - a) * (xg + 1.0) / 2.0
        x_fun = Chebfun.identity(Domain(self.domain))

        def to_funs(U):
            return [
                _chebfun_from_values(
                    jnp.asarray(U[i * n: (i + 1) * n]), self.domain)
                for i in range(m)
            ]

        if isinstance(f, (int, float)):
            f_vals = _np.full(m * n, float(f))
        elif isinstance(f, (list, tuple)):
            f_vals = _np.concatenate([
                _np.full(n, float(fi)) if isinstance(fi, (int, float))
                else _np.asarray(fi(jnp.asarray(xp))) for fi in f])
        else:
            f_vals = _np.tile(_np.asarray(f(jnp.asarray(xp))), m)

        def bc_list(bc_raw, us):
            if bc_raw is None:
                return []
            out = bc_raw(*us)
            if not isinstance(out, (list, tuple)):
                out = [out]
            return list(out)

        import inspect as _inspect
        try:
            _gnargs = (len(_inspect.signature(self._bc_general).parameters)
                       if self._bc_general is not None else 0)
        except (TypeError, ValueError):
            _gnargs = m + 1

        def gen_list(us):
            if self._bc_general is None:
                return []
            out = (self._bc_general(x_fun, *us) if _gnargs > m
                   else self._bc_general(*us))
            if not isinstance(out, (list, tuple)):
                out = [out]
            mid = 0.5 * (a + b)
            vals = []
            for o in out:
                if isinstance(o, (int, float)):
                    vals.append(float(o))
                elif hasattr(o, "funs"):
                    vals.append(float(_np.asarray(o(jnp.asarray(mid)))))
                else:
                    vals.append(float(_np.asarray(o).reshape(())))
            return vals

        n_g = len(gen_list(to_funs(_np.zeros(m * n))))
        # General .bc rows take slots alternating inward from the two
        # ends (rows not already claimed by lbc/rbc slots at 0, n-1).
        g_slots = []
        off_l, off_r = 1, 1
        for i in range(n_g):
            if i % 2 == 0:
                g_slots.append((i % m) * n + n - 1 - off_r)
                off_r += 1
            else:
                g_slots.append((i % m) * n + off_l)
                off_l += 1

        def residual(U):
            us = to_funs(U)
            out = self._call_op(x_fun, us)
            if not isinstance(out, (list, tuple)):
                out = [out]
            R = _np.concatenate([
                _np.full(n, float(o)) if isinstance(o, (int, float))
                else _np.asarray(o(jnp.asarray(xp))) for o in out])
            R = R - f_vals
            for i, g in enumerate(bc_list(self._lbc_raw, us)):
                R[(i % m) * n] = _eval_chebfun_at(g, a)
            for i, g in enumerate(bc_list(self._rbc_raw, us)):
                R[(i % m) * n + n - 1] = _eval_chebfun_at(g, b)
            for i, v in enumerate(gen_list(us)):
                R[g_slots[i]] = v
            return R

        U = _np.zeros(m * n)
        if self.init is not None:
            init = self.init if isinstance(self.init, (list, tuple)) \
                else [self.init] * m
            U = _np.concatenate([
                _np.asarray(gi(jnp.asarray(xp))) for gi in init])
        import scipy.linalg as _sla
        R = residual(U)
        for _it in range(max_iter):
            nrm = _np.max(_np.abs(R))
            if nrm < 1e-11:
                break
            # finite-difference Jacobian by column probing
            J = _np.zeros((m * n, m * n))
            h = 1e-7 * max(1.0, _np.max(_np.abs(U)))
            for j in range(m * n):
                Up = U.copy()
                Up[j] += h
                J[:, j] = (residual(Up) - R) / h
            try:
                lu = _sla.lu_factor(J)
            except (ValueError, _np.linalg.LinAlgError):
                break
            step = _sla.lu_solve(lu, R)
            nd = _np.linalg.norm(step)
            # Affine-invariant (Deuflhard) damping: monotone decrease of
            # the SIMPLIFIED Newton step keeps the iteration in the
            # initial guess's basin (see _solve_periodic_nonlinear).
            lam = 1.0
            Rn = R
            for _d in range(30):
                Rn = residual(U - lam * step)
                if _np.all(_np.isfinite(Rn)):
                    step_bar = _sla.lu_solve(lu, Rn)
                    if _np.linalg.norm(step_bar) < nd or lam < 1e-6:
                        break
                lam *= 0.5
            U = U - lam * step
            R = Rn
        else:
            import warnings as _w
            _w.warn("chebop system Newton: max iterations reached "
                    f"(residual {_np.max(_np.abs(R)):.2e})",
                    RuntimeWarning, stacklevel=2)
        return SystemSolution(to_funs(U))

    def _solve_linear_system(self, f=0.0, n: int | None = None):
        """Solve a LINEAR system of coupled ODEs by block collocation
        (MATLAB solvebvpLinear for chebmatrix operators).

        The op has signature (x, u1, ..., um) and returns a list of m
        expression chebfuns.  Each unknown is discretized by its
        values at n Chebyshev points; the (m*n) x (m*n) block matrix
        is built by probing the op with coordinate basis functions
        (columns are op(e_k) - op(0), exact for linear ops).  Boundary
        residual rows from lbc/rbc replace the boundary-point rows of
        successive block equations.

        Provenance
        ----------
        MATLAB source : @chebop/solvebvp.m (chebmatrix branch),
            @linop/mldivide.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        m = self._n_vars()
        a, b = self.domain
        if n is None:
            n = 64
        k = _np.arange(n)
        xg = _np.cos(_np.pi * k / (n - 1))[::-1]      # ascending
        xp = a + (b - a) * (xg + 1.0) / 2.0
        from chebfunjax.chebfun1d.chebfun import Chebfun
        x_fun = Chebfun.identity(Domain(self.domain))
        zero = _chebfun_from_values(jnp.zeros(2), self.domain)

        def apply_op(us):
            out = self._call_op(x_fun, us)
            if not isinstance(out, (list, tuple)):
                out = [out]
            out = list(out)
            # Parameter augmentation: trailing unknowns for which the operator
            # returns no equation are unknown scalar parameters.  Carry each as
            # a constant field by appending the equation ``p' = 0`` (MATLAB
            # @chebop treats parameters as extra unknowns in the linearization,
            # pinned by a boundary condition).
            for i in range(len(out), m):
                out.append(us[i].diff())
            return [zero + o if isinstance(o, (int, float)) else o
                    for o in out]

        def rows_at(fun_list, pts):
            return _np.concatenate([
                _np.asarray(g(jnp.asarray(pts))) for g in fun_list])

        zeros_list = [zero for _ in range(m)]
        base = rows_at(apply_op(zeros_list), xp)       # (m*n,)

        A = _np.zeros((m * n, m * n))
        for var in range(m):
            for j in range(n):
                vals = _np.zeros(n)
                vals[j] = 1.0
                probe = list(zeros_list)
                probe[var] = _chebfun_from_values(
                    jnp.asarray(vals), self.domain)
                A[:, var * n + j] = rows_at(apply_op(probe), xp) - base

        # RHS
        if isinstance(f, (int, float)):
            rhs_list = [float(f)] * m
        elif isinstance(f, (list, tuple)):
            rhs_list = list(f)
        else:
            rhs_list = [f]
        bvec = _np.zeros(m * n)
        for i in range(m):
            fi = rhs_list[i] if i < len(rhs_list) else 0.0
            if isinstance(fi, (int, float)):
                bvec[i * n: (i + 1) * n] = float(fi)
            else:
                bvec[i * n: (i + 1) * n] = _np.asarray(
                    fi(jnp.asarray(xp)))
        bvec = bvec - base

        # Boundary condition rows (probed the same way)
        def bc_rows(bc_raw, x0):
            if bc_raw is None:
                return [], []
            if isinstance(bc_raw, (int, float)):
                if m != 1:
                    raise ValueError(
                        "system BCs must be callables of (u1, ..., um)")
                # Scalar Dirichlet on the single unknown: u(x0) = value
                # (reached when a scalar lbc/rbc combines with a general
                # .bc constraint, e.g. NonstandardBCs).
                val = float(bc_raw)
                bc_raw = lambda u, _v=val: u - _v  # noqa: E731

            def bc_list(us):
                out = bc_raw(*us)
                if not isinstance(out, (list, tuple)):
                    out = [out]
                return [zero + o if isinstance(o, (int, float)) else o
                        for o in out]

            base_bc = _np.array([
                _eval_chebfun_at(g, x0) for g in bc_list(zeros_list)])
            nbc = len(base_bc)
            R = _np.zeros((nbc, m * n))
            for var in range(m):
                for j in range(n):
                    vals = _np.zeros(n)
                    vals[j] = 1.0
                    probe = list(zeros_list)
                    probe[var] = _chebfun_from_values(
                        jnp.asarray(vals), self.domain)
                    col = _np.array([
                        _eval_chebfun_at(g, x0)
                        for g in bc_list(probe)]) - base_bc
                    R[:, var * n + j] = col
            return list(R), list(-base_bc)

        # General .bc constraint rows: a single callable op(x, u1, ..., um)
        # returning a list of scalar conditions the user evaluated at
        # arbitrary points (u(0), feval(diff(u), 0) - p, ...), possibly
        # interior and possibly referencing unknown parameters.  Each is a
        # linear functional of the unknowns, probed the same way.
        def general_rows():
            if self._bc_general is None:
                return [], []

            def cond_list(us):
                out = self._bc_general(x_fun, *us)
                if not isinstance(out, (list, tuple)):
                    out = [out]
                return out

            mid = 0.5 * (a + b)

            def scalarize(c):
                if isinstance(c, (int, float)):
                    return float(c)
                if hasattr(c, "funs"):
                    # a constant Chebfun (e.g. scalar minus a const parameter)
                    return float(c(jnp.asarray(mid)))
                return float(_np.asarray(c).reshape(()))

            base_g = _np.array([scalarize(c) for c in cond_list(zeros_list)])
            ng = len(base_g)
            R = _np.zeros((ng, m * n))
            for var in range(m):
                for j in range(n):
                    vals = _np.zeros(n)
                    vals[j] = 1.0
                    probe = list(zeros_list)
                    probe[var] = _chebfun_from_values(
                        jnp.asarray(vals), self.domain)
                    col = _np.array([
                        scalarize(c) for c in cond_list(probe)]) - base_g
                    R[:, var * n + j] = col
            return list(R), list(-base_g)

        rows_l, vals_l = bc_rows(self._lbc_raw, a)
        rows_r, vals_r = bc_rows(self._rbc_raw, b)
        rows_g, vals_g = general_rows()

        used_rows: set[int] = set()

        def _place(ridx, row, val):
            A[ridx, :] = row
            bvec[ridx] = val
            used_rows.add(ridx)

        if self._bc_general is None and self._n_params() == 0:
            # Balanced system: keep the established lbc-left / rbc-right
            # placement that the coupled-system ports rely on.
            for i, (row, val) in enumerate(zip(rows_l, vals_l)):
                _place((i % m) * n, row, val)          # x = a row of eq i%m
            for i, (row, val) in enumerate(zip(rows_r, vals_r)):
                _place((i % m) * n + n - 1, row, val)  # x = b row
        else:
            # Parameter / general-constraint problems.  Drop exactly
            # ``order_i`` boundary rows from block i and fill the freed slots
            # with the constraints.  ``order_i`` comes from the augmented
            # system: a parameter block carries the appended equation
            # ``p' = 0`` (order 1).  Which constraint lands in which freed
            # slot is immaterial to the linear system -- only the set of
            # retained collocation rows matters -- so all conditions (lbc,
            # rbc, then general) are laid into the freed slots in order,
            # boundary-first, guaranteeing every parameter block receives a
            # pinning row.
            n_eq = self._n_equations()
            orders = self._piecewise_orders(m)
            for i in range(min(n_eq, m), m):
                orders[i] = 1
            orders = [max(1, int(o)) for o in orders]
            drop_slots: list[int] = []
            for blk in range(m):
                seq: list[int] = []
                for depth in range(n):
                    seq.append(blk * n + depth)
                    seq.append(blk * n + n - 1 - depth)
                drop_slots.extend(seq[: orders[blk]])
            conditions = (
                list(zip(rows_l, vals_l))
                + list(zip(rows_r, vals_r))
                + list(zip(rows_g, vals_g))
            )
            if len(conditions) != len(drop_slots):
                raise ValueError(
                    "chebop parameter/general-bc solve: number of boundary "
                    f"conditions ({len(conditions)}) does not match the "
                    f"system's total differential order ({len(drop_slots)}); "
                    "the problem is over- or under-determined."
                )
            for (row, val), ridx in zip(conditions, drop_slots):
                _place(ridx, row, val)

        sol = _np.linalg.solve(A, bvec)
        out = [
            _chebfun_from_values(
                jnp.asarray(sol[i * n: (i + 1) * n]), self.domain)
            for i in range(m)
        ]
        return SystemSolution(out)

    def _piecewise_orders(self, m: int) -> list[int]:
        """Differential order of each of the ``m`` unknowns.

        Probes the operator with one :class:`_SysOrderSniffer` per variable
        and reads back the highest ``diff`` order applied to each.  The order
        ``k_j`` fixes how many continuity conditions (derivatives 0..k_j-1)
        variable ``j`` needs at every interior breakpoint.

        Provenance
        ----------
        MATLAB source : @linop/getDiffOrder.m
        Chebfun commit: 7574c77
        """
        import inspect

        orders = [0] * m
        sniffers = [_SysOrderSniffer(orders, (i,)) for i in range(m)]
        x_sniff = _SysOrderSniffer(orders, ())
        try:
            nargs = len(inspect.signature(self.op).parameters)
        except (TypeError, ValueError):
            nargs = m + 1
        try:
            if nargs > m:
                self.op(x_sniff, *sniffers)
            else:
                self.op(*sniffers)
        except Exception:
            # Probe failed (unusual op): assume every unknown is 2nd order,
            # the common case, so continuity is still imposed.
            orders = [2] * m
        return orders

    def _piecewise_eq_orders(self, m: int) -> "list[int]":
        """Differential order of each EQUATION of the system operator.

        Used to distribute the continuity/wrap condition rows across the
        equation blocks in proportion to each equation's order — the
        naive ``eq = c % m`` round-robin starves low-order equations of
        collocation rows in mixed-order systems.
        """
        import inspect

        sniffers = [_EqOrderSniffer({(i, 0)}) for i in range(m)]
        x_sniff = _EqOrderSniffer()
        try:
            nargs = len(inspect.signature(self.op).parameters)
        except (TypeError, ValueError):
            nargs = m + 1
        try:
            out = (self.op(x_sniff, *sniffers) if nargs > m
                   else self.op(*sniffers))
            if not isinstance(out, (list, tuple)):
                out = [out]
            eq_orders = []
            for o in out:
                pairs = getattr(o, "pairs", frozenset())
                eq_orders.append(max((k for _v, k in pairs),
                                     default=0))
            if len(eq_orders) == m:
                return eq_orders
        except Exception:
            pass
        return [2] * m

    def _detect_jump_breakpoints(self) -> list[float]:
        """Interior points a general ``.bc`` refers to via ``jump`` / one-sided
        evaluation.

        Runs the ``.bc`` callable once on a smooth probe with one-sided
        evaluation recording active (see
        :func:`chebfunjax.chebfun1d.chebfun.start_side_eval_record`), so every
        ``u(x0, 'left')`` and ``jump(u, x0)`` reports its point ``x0``.  The
        distinct interior points become breakpoints of a piecewise solve, at
        which the ``.bc`` conditions replace continuity.

        Provenance
        ----------
        MATLAB source : @chebop/mldivide.m, @linop/matrix.m (jump handling)
        Chebfun commit: 7574c77
        """
        if self._bc_general is None:
            return []
        import inspect

        from chebfunjax.chebfun1d.chebfun import (
            Chebfun,
            start_side_eval_record,
            stop_side_eval_record,
        )

        m = self._n_vars()
        dom0 = Domain(self.domain)
        xf = Chebfun.identity(dom0)
        us = [Chebfun.identity(dom0) for _ in range(m)]
        try:
            nargs = len(inspect.signature(self._bc_general).parameters)
        except (TypeError, ValueError):
            nargs = m + 1
        a, b = self.domain
        start_side_eval_record()
        try:
            if nargs > m:
                self._bc_general(xf, *us)
            else:
                self._bc_general(*us)
        except Exception:
            pass
        pts = stop_side_eval_record()
        out: list[float] = []
        for p in pts:
            if (a + 1e-10 < p < b - 1e-10
                    and all(abs(p - q) > 1e-10 for q in out)):
                out.append(float(p))
        return sorted(out)

    def _piecewise_linear_matrix(self, m, P, nn, Pn, bps, ints, xps,
                                 orders, to_funs, apply_op, bc_res,
                                 residual, l_rows, r_rows, c_rows,
                                 f_vals, wrap_conds=None,
                                 use_cache=True):
        """Direct collocation matrix for a LINEAR piecewise problem.

        Returns ``(A, R0)`` with ``residual(U) == A @ U + R0`` for linear
        operators; the caller verifies affineness on a random vector
        before trusting the matrix (nonlinear ops fail that check and
        fall back to FD-Newton).

        Provenance
        ----------
        MATLAB source : @chebop/linearize.m (coefficient extraction),
            @linop/continuity.m (interface rows)
        Chebfun commit: 7574c77
        """
        import math as _math

        import numpy as _np

        from chebfunjax.utils.diffmat import diffmat as _diffmat

        # Per-piece differentiation matrices up to the max order.
        max_ord = max(orders) if orders else 0
        Dmats = [[_np.asarray(_diffmat(nn, k, domain=ints[p]))
                  for k in range(max_ord + 1)] for p in range(P)]

        pts = [_np.asarray(xps[p]) for p in range(P)]
        A = _np.zeros((m * Pn, m * Pn))

        # Operator block via monomial probes: place x^k/k! in variable
        # `var` (zeros elsewhere), apply the op, and forward-substitute
        # for the coefficient FUNCTIONS (kept as evaluable callables and
        # cached across adaptive refinement levels — re-applying the op
        # at every level costs seconds of chebfun arithmetic per call,
        # while re-evaluating the extracted coefficients is millisecs).
        cache_key = (m, tuple(bps), id(self.op),
                     id(self._lbc_raw), id(self._rbc_raw))
        cache = (getattr(self, "_pw_lin_cache", None)
                 if use_cache else None)
        fresh = cache is None or cache.get("key") != cache_key
        if fresh:
            def _as_fun(o):
                if isinstance(o, (int, float)):
                    return (lambda x, _v=float(o):
                            _np.full(_np.shape(x), _v))
                if isinstance(o, complex):
                    return (lambda x, _v=o:
                            _np.full(_np.shape(x), _v,
                                     dtype=_np.complex128))
                return lambda x, _o=o: _np.asarray(_o(jnp.asarray(x)))

            out0 = apply_op(to_funs(_np.zeros(m * Pn)))
            op0_funs = [_as_fun(o) for o in out0]
            c_funs = [[None] * m for _ in range(m)]   # [eq][var] -> list_k
            for var in range(m):
                kmax = orders[var] if orders else 0
                outs = []
                for k in range(kmax + 1):
                    U_probe = _np.zeros(m * Pn)
                    for p in range(P):
                        U_probe[var * Pn + p * nn:
                                var * Pn + (p + 1) * nn] = \
                            pts[p] ** k / _math.factorial(k)
                    outs.append([_as_fun(o)
                                 for o in apply_op(to_funs(U_probe))])
                for eq in range(m):
                    ck_list = []
                    for k in range(kmax + 1):
                        def _ck(x, _k=k, _eq=eq, _outs=outs,
                                _prev=tuple(ck_list),
                                _op0=op0_funs[eq]):
                            v = _outs[_k][_eq](x) - _op0(x)
                            xx = _np.asarray(x)
                            for j, cj in enumerate(_prev):
                                v = v - cj(x) * (
                                    xx ** (_k - j)
                                    / _math.factorial(_k - j))
                            return v
                        ck_list.append(_ck)
                    c_funs[eq][var] = ck_list
            cache = {"key": cache_key, "c_funs": c_funs,
                     "op0_funs": op0_funs}
            if use_cache:
                self._pw_lin_cache = cache

        # R0 = residual at zero, built from the cached op0 (one op
        # application total, not one per refinement level): op rows are
        # op0(x) - f, continuity rows vanish at zero, boundary rows come
        # from bc_res on the zero function.
        op0_funs = cache["op0_funs"]
        R0 = _np.zeros(m * Pn)
        for eq in range(m):
            for p in range(P):
                sl = slice(eq * Pn + p * nn, eq * Pn + (p + 1) * nn)
                v0 = _np.asarray(op0_funs[eq](pts[p])).reshape(-1)
                if v0.size == 1:
                    v0 = (_np.full(nn, complex(v0))
                          if _np.iscomplexobj(v0)
                          else _np.full(nn, float(v0)))
                if _np.iscomplexobj(v0) and not _np.iscomplexobj(R0):
                    R0 = R0.astype(_np.complex128)
                R0[sl] = v0
        # Subtract the RHS exactly as residual() does.
        R0 = R0 - f_vals
        us_zero_l = to_funs(_np.zeros(m * Pn))
        if not wrap_conds:
            for i, v in enumerate(bc_res(self._lbc_raw, us_zero_l,
                                         bps[0])):
                R0[l_rows[i]] = v
            for i, v in enumerate(bc_res(self._rbc_raw, us_zero_l,
                                         bps[-1])):
                R0[r_rows[i]] = v
        for (rr, _j, _d, _bp, _pl, _pr) in c_rows:
            R0[rr] = 0.0

        c_funs = cache["c_funs"]
        for var in range(m):
            kmax = orders[var] if orders else 0
            for eq in range(m):
                for k in range(kmax + 1):
                    for p in range(P):
                        rs = slice(eq * Pn + p * nn,
                                   eq * Pn + (p + 1) * nn)
                        cs = slice(var * Pn + p * nn,
                                   var * Pn + (p + 1) * nn)
                        ckv = _np.asarray(
                            c_funs[eq][var][k](pts[p])).reshape(-1)
                        if ckv.size == 1:
                            ckv = (_np.full(nn, complex(ckv))
                                   if _np.iscomplexobj(ckv)
                                   else _np.full(nn, float(ckv)))
                        if (_np.iscomplexobj(ckv)
                                and not _np.iscomplexobj(A)):
                            A = A.astype(_np.complex128)
                        A[rs, cs] += ckv[:, None] * Dmats[p][k]

        # Continuity rows: u_j^(d) matches across each interior break.
        # The grids are ascending, so the left piece's endpoint is its
        # last collocation row and the right piece's is its first.
        for (rr, j, d, bp, p_l, p_r) in c_rows:
            A[rr, :] = 0.0
            A[rr, j * Pn + p_l * nn: j * Pn + (p_l + 1) * nn] = \
                Dmats[p_l][d][-1, :] if d > 0 else _np.eye(nn)[-1]
            A[rr, j * Pn + p_r * nn: j * Pn + (p_r + 1) * nn] -= \
                Dmats[p_r][d][0, :] if d > 0 else _np.eye(nn)[0]

        # Boundary rows: probe bc_res with unit vectors over the endpoint
        # pieces only (an lbc/rbc functional depends on its endpoint
        # piece; anything more exotic fails the caller's affine check).
        def _bc_rows_probe(bc_raw, x0, rows_idx, piece):
            if not rows_idx:
                return
            # Scalar Dirichlet (the common case): the row is the unit
            # endpoint-evaluation row of variable 0 — no probing needed.
            if isinstance(bc_raw, (int, float)):
                rr = rows_idx[0]
                A[rr, :] = 0.0
                pt = 0 if abs(x0 - bps[0]) <= abs(x0 - bps[-1]) else nn - 1
                A[rr, 0 * Pn + piece * nn + pt] = 1.0
                return
            base = _np.asarray(bc_res(bc_raw, to_funs(_np.zeros(m * Pn)),
                                      x0), dtype=float)
            for rr in rows_idx:
                A[rr, :] = 0.0
            for var in range(m):
                for i in range(nn):
                    U_probe = _np.zeros(m * Pn)
                    U_probe[var * Pn + piece * nn + i] = 1.0
                    vals = _np.asarray(
                        bc_res(bc_raw, to_funs(U_probe), x0), dtype=float)
                    col = vals - base
                    for ri, rr in enumerate(rows_idx):
                        A[rr, var * Pn + piece * nn + i] = col[ri]

        if wrap_conds:
            for i, (j, dd) in enumerate(wrap_conds):
                rr = l_rows[i]
                A[rr, :] = 0.0
                A[rr, j * Pn + 0 * nn: j * Pn + 1 * nn] = \
                    (Dmats[0][dd][0, :] if dd > 0 else _np.eye(nn)[0])
                A[rr, j * Pn + (P - 1) * nn: j * Pn + P * nn] -= \
                    (Dmats[P - 1][dd][-1, :] if dd > 0
                     else _np.eye(nn)[-1])
                R0[rr] = 0.0
        else:
            _bc_rows_probe(self._lbc_raw, bps[0], l_rows, 0)
            _bc_rows_probe(self._rbc_raw, bps[-1], r_rows, P - 1)

        return A, R0, fresh

    def _solve_piecewise(self, f=0.0, n: int | None = None,
                         max_iter: int = 40, extra_breaks=None,
                         cont_breaks=None):
        """Adaptive piecewise solve: refine the per-piece resolution until
        the solution's Chebyshev coefficients decay.  The fixed default
        (n=32 per piece) silently under-resolved e.g. the ParameterODE
        nearly-singular-coefficient problem at O(1) error; with the
        direct linear assembly in :meth:`_piecewise_linear_matrix` each
        refinement level costs one linear solve, so adapting is cheap.
        Nonlinear problems (FD-Newton path) refine only to 64 to bound
        the cost of the column-by-column Jacobian.
        """
        # Fresh coefficient extraction per solve: a reused id() after
        # garbage collection must not resurrect a stale cache from a
        # previous op assignment.
        self._pw_lin_cache = None
        if n is not None:
            return self._solve_piecewise_at(
                f, n=n, max_iter=max_iter, extra_breaks=extra_breaks,
                cont_breaks=cont_breaks)

        from chebfunjax.utils.misc import standard_chop

        def _happy(us) -> bool:
            for u in us:
                for piece in u.funs:
                    coef = jnp.abs(jnp.asarray(piece.tech.coeffs))
                    npts = coef.shape[0]
                    if npts < 8:
                        continue
                    if int(standard_chop(coef, tol=1e-11)) >= npts - 1:
                        return False
            return True

        sol = None
        saved_init = self.init
        try:
            for nn in (32, 64, 128, 256, 512):
                sol = self._solve_piecewise_at(
                    f, n=nn, max_iter=max_iter,
                    extra_breaks=extra_breaks,
                    cont_breaks=cont_breaks)
                # Seed the next refinement level with this solution
                # (grid-continuation; restarting Newton from the line
                # guess at every level rediscovers the wrong basin on
                # hard nonlinear problems).
                if not getattr(self, "_pw_linear_used", False):
                    self.init = (list(sol)
                                 if isinstance(sol, SystemSolution)
                                 else sol)
                used_linear = getattr(self, "_pw_linear_used", False)
                candidates = (list(sol)
                              if isinstance(sol, SystemSolution)
                              else [sol])
                try:
                    if _happy(candidates):
                        break
                except Exception:
                    break
                # FD-Newton refinement is O((mPn)^2) residual
                # evaluations; cap it where MATLAB's default dimension
                # also stops.  The monomial-probe Jacobian escalates
                # further at low cost.
                fastjac = getattr(self, "_pw_fastjac_used", False)
                if not used_linear and not fastjac and nn >= 64:
                    break
                if not used_linear and fastjac and nn >= 256:
                    break
        finally:
            self.init = saved_init
        return sol

    def _solve_piecewise_at(self, f=0.0, n: int | None = None,
                            max_iter: int = 40, extra_breaks=None,
                            cont_breaks=None):
        """Solve a BVP on a domain with interior breakpoints by piecewise
        collocation (MATLAB @chebop with a ``chebcolloc2`` discretisation on
        a multi-interval domain).

        Each unknown ``u_j`` is represented by ``P`` Chebyshev pieces (one
        per sub-interval) of ``n`` values each, so the piecewise chebfun has
        ``P`` funs.  The operator residual is evaluated piece-by-piece
        (``P*n`` rows per equation); boundary rows are then overwritten to
        impose

        1. the global boundary conditions at the domain endpoints, exactly as
           the single-interval solver does; and
        2. continuity of ``u_j`` and its derivatives ``0..k_j-1`` across every
           interior breakpoint, where ``k_j`` is the differential order of
           variable ``j`` (from :meth:`_piecewise_orders`).

        The duplicated breakpoint values (piece ``p``'s right endpoint and
        piece ``p+1``'s left endpoint are separate unknowns at the same
        physical point) provide the natural row slots for the continuity
        conditions; they are distributed over the equation blocks
        (``eq = c mod m``) alternating the two coincident collocation points,
        a pattern that keeps the collocation matrix well conditioned.  A
        damped Newton iteration with a finite-difference Jacobian solves the
        coupled system; linear problems converge in one full step.

        Handles both scalar (``m = 1``) and system (``m >= 2``) problems; the
        return value is a :class:`~chebfunjax.chebfun1d.Chebfun` for a scalar
        unknown and a :class:`SystemSolution` otherwise.

        Provenance
        ----------
        MATLAB source : @chebop/solvebvp.m (piecewise chebcolloc2 branch),
            @linop/mldivide.m, @linop/continuity.m,
            @chebop/solvebvpNonlinear.m
        Chebfun commit: 7574c77
        """
        import inspect

        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece

        m = self._n_vars()
        bps = [float(v) for v in self.domain]

        # Coefficient-induced breakpoints: a discontinuous coefficient such
        # as ``sign(x)`` injects a kink the collocation grid cannot resolve
        # unless a breakpoint is placed there.  MATLAB detects this while
        # building the piecewise chebmatrix; we mirror it by applying the
        # operator to a smooth probe over the current domain and unioning any
        # interior breakpoints of the output into ``bps``.
        try:
            import inspect as _inspect
            _nargs0 = len(_inspect.signature(self.op).parameters)
            _dom0 = Domain(tuple(bps))
            _xf0 = Chebfun.identity(_dom0)
            _probe = [Chebfun.identity(_dom0) for _ in range(m)]
            _out0 = (self.op(_xf0, *_probe) if _nargs0 > m
                     else self.op(*_probe))
            if not isinstance(_out0, (list, tuple)):
                _out0 = [_out0]
            _extra = set()
            for _o in _out0:
                if hasattr(_o, "domain"):
                    for _b in _o.domain.breakpoints:
                        _bf = float(_b)
                        if all(abs(_bf - _e) > 1e-12 for _e in bps):
                            _extra.add(_bf)
            if _extra:
                bps = sorted(set(bps) | _extra)
        except Exception:
            pass

        # Breakpoints where the solution stays C^{k-1} but the RHS (or a
        # coefficient) is only piecewise-smooth: add to the grid with the
        # usual continuity rows.
        for cb in [float(v) for v in (cont_breaks or [])]:
            if all(abs(cb - e) > 1e-12 for e in bps):
                bps = sorted(bps + [cb])

        # Interior breakpoints introduced by jump / one-sided conditions in a
        # general .bc.  At these the solution may be discontinuous, so the
        # usual continuity rows are replaced by the .bc conditions.
        jump_xs = [float(v) for v in (extra_breaks or [])]
        for jx in jump_xs:
            if all(abs(jx - e) > 1e-12 for e in bps):
                bps = sorted(bps + [jx])
        jump_break_ps = {
            p for p in range(1, len(bps) - 1)
            if any(abs(bps[p] - jx) <= 1e-12 for jx in jump_xs)}

        P = len(bps) - 1
        nn = 32 if n is None else int(n)
        dom = Domain(tuple(bps))
        ints = [(bps[p], bps[p + 1]) for p in range(P)]
        kk = _np.arange(nn)
        tref = _np.cos(_np.pi * kk / (nn - 1))[::-1]        # ascending cheb2
        xps = [ints[p][0] + (ints[p][1] - ints[p][0]) * (tref + 1.0) / 2.0
               for p in range(P)]
        x_fun = Chebfun.identity(dom)
        Pn = P * nn

        orders = self._piecewise_orders(m)
        K = int(sum(orders))                       # conditions per breakpoint
        if K > 2 * m:
            raise NotImplementedError(
                "chebop piecewise collocation currently supports systems "
                f"whose total differential order ({K}) is at most twice the "
                f"number of unknowns ({m}); got order {K}.")

        def to_funs(U):
            us = []
            for j in range(m):
                funs = [_Piece.from_values(
                    jnp.asarray(U[j * Pn + p * nn: j * Pn + (p + 1) * nn]),
                    ints[p][0], ints[p][1]) for p in range(P)]
                us.append(Chebfun(funs=funs, domain=dom))
            return us

        try:
            nargs = len(inspect.signature(self.op).parameters)
        except (TypeError, ValueError):
            nargs = m + 1

        def apply_op(us):
            out = self.op(x_fun, *us) if nargs > m else self.op(*us)
            if not isinstance(out, (list, tuple)):
                out = [out]
            return list(out)

        def bc_res(bc_raw, us, x0):
            if bc_raw is None:
                return []
            if isinstance(bc_raw, (int, float)):
                # scalar Dirichlet on the (single) unknown
                return [float(us[0](jnp.asarray(x0))) - float(bc_raw)]
            out = bc_raw(*us)
            if not isinstance(out, (list, tuple)):
                out = [out]
            vals = []
            for o in out:
                vals.append(float(o) if isinstance(o, (int, float))
                            else float(o(jnp.asarray(x0))))
            return vals

        def row(eq, p, pt):
            return eq * Pn + p * nn + pt

        us_zero = to_funs(_np.zeros(m * Pn))
        # Distribute condition rows across equation blocks in proportion
        # to each equation's differential order: a first-order equation
        # in a mixed-order system must not lose as many collocation rows
        # as a second-order one (that starvation broke the periodic 2x2
        # PeriodicSystem case).
        eq_orders = (self._piecewise_eq_orders(m) if m > 1
                     else [orders[0]])
        # Which equation block a condition is charged to.  With equal
        # differential orders this is plain round-robin (each equation
        # gives up the same number of rows); with MIXED orders the pool
        # is weighted by order, since a first-order equation must not
        # surrender as many collocation rows as a second-order one.
        if len(set(eq_orders)) <= 1:
            eq_pool = list(range(m))
        else:
            eq_pool = [e for e in range(m)
                       for _ in range(max(eq_orders[e], 1))]
        if not eq_pool:
            eq_pool = list(range(m))

        def _pool_eq(c):
            return eq_pool[c % len(eq_pool)]

        # Balanced slot allocator: each (piece, equation) block gives up
        # exactly eq_orders[e] collocation rows, taken from its ends.
        # ``_take_row`` NEVER returns a row twice -- an earlier version
        # fell back to a fixed index when a block was exhausted, which
        # silently overwrote a previously placed condition (two
        # continuity rows landed on the same row of the 2x2 piecewise
        # system in test_nonlinSysDampingBreaks_C2, losing continuity).
        _quota = {(p_, e_): max(eq_orders[e_], 1)
                  for p_ in range(P) for e_ in range(m)}
        _used = set()

        def _take_in(p_, e_, prefer_right):
            """Absolute row index inside block (p_, e_), or None."""
            if _quota[(p_, e_)] <= 0:
                return None
            pts = (range(nn - 1, -1, -1) if prefer_right
                   else range(nn))
            for pt in pts:
                rr = row(e_, p_, pt)
                if rr not in _used:
                    _used.add(rr)
                    _quota[(p_, e_)] -= 1
                    return rr
            return None

        def _take_row(p_, e_, prefer_right):
            """Absolute row index, preferring block (p_, e_) and
            falling back to any block that still has quota (the total
            quota always equals the number of conditions)."""
            rr = _take_in(p_, e_, prefer_right)
            if rr is not None:
                return rr
            for cand in ([(p_, e) for e in range(m)]
                         + [(p, e_) for p in range(P)]
                         + [(p, e) for p in range(P)
                            for e in range(m)]):
                rr = _take_in(cand[0], cand[1], prefer_right)
                if rr is not None:
                    return rr
            raise RuntimeError(
                "chebop piecewise: no free condition row (conditions "
                "exceed the available collocation rows)")

        # Periodic problems on piecewise domains: the endpoint condition
        # slots hold the wrap-around rows u_j^(d)(a) = u_j^(d)(b),
        # d < order_j (MATLAB's periodic chebcolloc continuity).
        wrap_conds = []
        if getattr(self, "_periodic", False):
            wrap_conds = [(j, d) for j in range(m)
                          for d in range(orders[j])]
            n_l = len(wrap_conds)
            n_r = 0
        else:
            n_l = len(bc_res(self._lbc_raw, us_zero, bps[0]))
            n_r = len(bc_res(self._rbc_raw, us_zero, bps[-1]))
        if wrap_conds:
            # One wrap row per (var, deriv); alternate between the two
            # end pieces so neither end is starved of collocation rows.
            l_rows = []
            for i in range(n_l):
                e_ = _pool_eq(i)
                p_pref, pref_right = ((0, False) if i % 2 == 0
                                      else (P - 1, True))
                l_rows.append(_take_row(p_pref, e_, pref_right))
            r_rows = []
        else:
            l_rows = [_take_row(0, _pool_eq(i), False)
                      for i in range(n_l)]
            r_rows = [_take_row(P - 1, _pool_eq(i), True)
                      for i in range(n_r)]

        # Continuity: derivatives 0..k_j-1 of each variable j at every
        # interior break.  Each condition takes one of the two collocation
        # rows that coincide at the break (left piece right end / right piece
        # left end), distributed across equations for good conditioning.
        conds = [(j, d) for j in range(m) for d in range(orders[j])]
        c_rows = []
        g_slots = []
        for p in range(1, P):
            for c, (j, d) in enumerate(conds):
                eq = _pool_eq(c)
                # Charge the condition to the left piece when it still
                # has quota, otherwise the right piece (or any block
                # with room) -- never reusing a row.
                rr = _take_in(p - 1, eq, True)
                if rr is None:
                    rr = _take_row(p, eq, False)
                if p in jump_break_ps:
                    # A jump breakpoint: the .bc supplies the interface
                    # conditions, so this row slot is reserved for them.
                    g_slots.append(rr)
                else:
                    c_rows.append((rr, j, d, bps[p], p - 1, p))

        # General .bc conditions (jump / one-sided) fill the reserved
        # interface rows at the jump breakpoints.
        try:
            gen_nargs = (len(inspect.signature(self._bc_general).parameters)
                         if self._bc_general is not None else 0)
        except (TypeError, ValueError):
            gen_nargs = m + 1

        def gen_res(us):
            if self._bc_general is None:
                return []
            out = (self._bc_general(x_fun, *us) if gen_nargs > m
                   else self._bc_general(*us))
            if not isinstance(out, (list, tuple)):
                out = [out]
            mid = 0.5 * (bps[0] + bps[-1])
            vals = []
            for o in out:
                if isinstance(o, (int, float)):
                    vals.append(float(o))
                elif hasattr(o, "funs"):
                    vals.append(float(o(jnp.asarray(mid))))
                else:
                    vals.append(float(_np.asarray(o).reshape(())))
            return vals

        n_g = len(gen_res(us_zero))
        if n_g != len(g_slots):
            raise ValueError(
                "chebop piecewise jump solve: number of general .bc "
                f"conditions ({n_g}) does not match the interface rows freed "
                f"at the jump breakpoints ({len(g_slots)}).")

        # RHS values laid out per equation / piece (complex-preserving).
        f_vals = _np.zeros(m * Pn)
        for eq in range(m):
            fi = f[eq] if isinstance(f, (list, tuple)) and eq < len(f) else (
                f if not isinstance(f, (list, tuple)) else 0.0)
            for p in range(P):
                sl = slice(eq * Pn + p * nn, eq * Pn + (p + 1) * nn)
                fv = (float(fi) if isinstance(fi, (int, float))
                      else _np.asarray(fi(jnp.asarray(xps[p]))))
                if _np.iscomplexobj(fv) and not _np.iscomplexobj(f_vals):
                    f_vals = f_vals.astype(_np.complex128)
                f_vals[sl] = fv

        def residual(U):
            us = to_funs(U)
            out = apply_op(us)
            R = _np.zeros(m * Pn,
                          dtype=(_np.complex128
                                 if _np.iscomplexobj(f_vals)
                                 or _np.iscomplexobj(U) else _np.float64))
            for eq in range(m):
                o = out[eq]
                for p in range(P):
                    sl = slice(eq * Pn + p * nn, eq * Pn + (p + 1) * nn)
                    # Evaluate the operator output through its own __call__
                    # rather than indexing ``o.funs[p]``: a discontinuous
                    # coefficient (e.g. ``sign(x)``) can inject an extra
                    # breakpoint interior to piece ``p``, so ``o`` may carry
                    # more funs than the ``P`` solver pieces and positional
                    # indexing would read the wrong sub-interval.
                    if isinstance(o, (int, float)):
                        ov = _np.full(nn, float(o))
                    else:
                        ov = _np.asarray(o(jnp.asarray(xps[p])))
                        # At interior duplicated nodes take THIS piece's
                        # one-sided limit: __call__'s convention at a
                        # breakpoint may return the other piece's value,
                        # which desynchronizes the residual from the
                        # direct assembly (its rows are per-piece).
                        ov = _np.array(ov)
                        try:
                            if p > 0:
                                ov[0] = float(_np.asarray(o(
                                    jnp.asarray(xps[p][0]), "right")))
                            if p < P - 1:
                                ov[-1] = float(_np.asarray(o(
                                    jnp.asarray(xps[p][-1]), "left")))
                        except Exception:
                            pass
                    if _np.iscomplexobj(ov) and not _np.iscomplexobj(R):
                        R = R.astype(_np.complex128)
                    R[sl] = ov
            R = R - f_vals
            if wrap_conds:
                for i, (j, dd) in enumerate(wrap_conds):
                    ua = (us[j].funs[0].diff(dd)(jnp.asarray(bps[0]))
                          if dd > 0 else
                          us[j].funs[0](jnp.asarray(bps[0])))
                    ub = (us[j].funs[-1].diff(dd)(jnp.asarray(bps[-1]))
                          if dd > 0 else
                          us[j].funs[-1](jnp.asarray(bps[-1])))
                    R[l_rows[i]] = float(ua) - float(ub)
            else:
                for i, v in enumerate(bc_res(self._lbc_raw, us,
                                             bps[0])):
                    R[l_rows[i]] = v
                for i, v in enumerate(bc_res(self._rbc_raw, us,
                                             bps[-1])):
                    R[r_rows[i]] = v
            for (rr, j, d, bp, p_l, p_r) in c_rows:
                xb = jnp.asarray(bp)
                left = us[j].funs[p_l].diff(d)(xb) if d > 0 \
                    else us[j].funs[p_l](xb)
                right = us[j].funs[p_r].diff(d)(xb) if d > 0 \
                    else us[j].funs[p_r](xb)
                R[rr] = float(left) - float(right)
            for i, v in enumerate(gen_res(us)):
                R[g_slots[i]] = v
            return R

        # ---- Linear fast path -------------------------------------------
        # For a linear operator the residual is affine, R(U) = A U + R(0).
        # Build A directly instead of column-by-column finite differences:
        # the operator block is recovered with (order+1) op applications
        # per variable via monomial probes (L[x^k/k!] = sum_{j<=k}
        # c_j(x) x^{k-j}/(k-j)!, forward-substituted for the coefficient
        # functions c_j), continuity rows are barycentric-endpoint rows of
        # per-piece differentiation matrices, and boundary rows are probed
        # only over the endpoint pieces.  Affineness is verified against
        # residual() on a random vector; any mismatch (nonlinear op,
        # nonlocal BC, ...) falls back to the damped FD-Newton below.
        # Jump/general .bc problems keep the Newton path.
        self._pw_linear_used = False
        if not g_slots and self.init is None:
            try:
                lin = self._piecewise_linear_matrix(
                    m, P, nn, Pn, bps, ints, xps, orders,
                    to_funs, apply_op, bc_res, residual,
                    l_rows, r_rows, c_rows, f_vals,
                    wrap_conds=wrap_conds)
            except Exception:
                lin = None
            if lin is not None:
                A_lin, R0, fresh = lin
                ok = True
                if fresh:
                    # Verify affineness once per extraction (each check
                    # costs a full op application); refinement levels
                    # reuse the validated coefficient functions.
                    rng = _np.random.RandomState(0)
                    U_t = rng.randn(m * Pn)
                    R_t = residual(U_t)
                    scale = max(1.0, float(_np.max(_np.abs(R_t))))
                    ok = (_np.max(_np.abs(A_lin @ U_t + R0 - R_t))
                          <= 1e-6 * scale)
                    if not ok:
                        self._pw_lin_cache = None
                if ok:
                    try:
                        U_lin = _np.linalg.solve(A_lin, -R0)
                        self._pw_linear_used = True
                        us = to_funs(U_lin)
                        return us[0] if m == 1 else SystemSolution(us)
                    except _np.linalg.LinAlgError:
                        pass

        # Initial iterate: user's N.init, else a boundary-condition
        # satisfying straight line for scalar Dirichlet data (MATLAB's
        # chebop default guess) — a zero start makes u*u'-type Newton
        # iterations stall on a degenerate Jacobian — else zero.
        U = _np.zeros(m * Pn)
        if (self.init is None and m == 1
                and isinstance(self._lbc_raw, (int, float))
                and isinstance(self._rbc_raw, (int, float))):
            la, lb = float(self._lbc_raw), float(self._rbc_raw)
            for p_ in range(P):
                U[p_ * nn: (p_ + 1) * nn] = la + (lb - la) * (
                    (_np.asarray(xps[p_]) - bps[0])
                    / (bps[-1] - bps[0]))
        if self.init is not None:
            init = self.init if isinstance(self.init, (list, tuple)) \
                else [self.init] * m
            blocks = []
            for gi in init:
                if callable(gi):
                    blocks.append(_np.concatenate([
                        _np.asarray(gi(jnp.asarray(xps[p]))) for p in range(P)]))
                else:
                    blocks.append(_np.full(Pn, float(gi)))
            U = _np.concatenate(blocks)

        # Damped Newton (linear problems converge in a single full step).
        self._pw_fastjac_used = False

        def _fast_jacobian(U_k):
            """Frechet-derivative matrix at U_k via monomial probes:
            J w = (N(u_k + h w) - N(u_k))/h is LINEAR in w, so the same
            coefficient extraction as the linear fast path assembles it
            with (order+1)*m op applications instead of m*P*n finite-
            difference columns."""
            us_k = to_funs(U_k)
            hh = 1e-7 * max(1.0, float(_np.max(_np.abs(U_k))))
            out_base = apply_op(us_k)

            def apply_op_lin(ws):
                pert = [us_k[j] + hh * ws[j] for j in range(m)]
                out_p = apply_op(pert)
                return [(a - b) * (1.0 / hh)
                        for a, b in zip(out_p, out_base)]

            lin = self._piecewise_linear_matrix(
                m, P, nn, Pn, bps, ints, xps, orders,
                to_funs, apply_op_lin, bc_res, residual,
                l_rows, r_rows, c_rows,
                _np.zeros_like(f_vals), wrap_conds=wrap_conds,
                use_cache=False)
            return _np.asarray(lin[0])

        R = residual(U)
        Rn = R
        for _it in range(max_iter):
            nrm = _np.max(_np.abs(R))
            if nrm < 1e-12:
                break
            J = None
            try:
                J = _fast_jacobian(U)
                # Directional sanity check against the true residual.
                rng_j = _np.random.RandomState(1)
                w_t = rng_j.randn(m * Pn)
                h_t = 1e-6 * max(1.0, float(_np.max(_np.abs(U))))
                d_true = (residual(U + h_t * w_t) - R) / h_t
                d_lin = J @ w_t
                scl = max(1.0, float(_np.max(_np.abs(d_true))))
                if (float(_np.max(_np.abs(d_lin - d_true)))
                        > 1e-3 * scl):
                    J = None
            except Exception:
                J = None
            if J is not None:
                self._pw_fastjac_used = True
            else:
                J = _np.zeros((m * Pn, m * Pn))
                h = 1e-7 * max(1.0, _np.max(_np.abs(U)))
                for jc in range(m * Pn):
                    Up = U.copy()
                    Up[jc] += h
                    J[:, jc] = (residual(Up) - R) / h
            try:
                step = _np.linalg.solve(J, R)
            except _np.linalg.LinAlgError:
                break
            lam = 1.0
            for _d in range(40):
                Rn = residual(U - lam * step)
                if _np.max(_np.abs(Rn)) < nrm or lam < 1e-6:
                    break
                lam *= 0.5
            new_nrm = _np.max(_np.abs(Rn))
            U = U - lam * step
            R = Rn
            # Stagnation: the damped step no longer reduces the residual --
            # we have reached the finite-difference/conditioning floor (a
            # linear problem plateaus here after one full step), so stop.
            if new_nrm >= nrm * (1.0 - 1e-8):
                break

        final_res = float(_np.max(_np.abs(R)))
        if final_res > 1e-8:
            import warnings as _w
            _w.warn("chebop piecewise Newton: did not converge "
                    f"(residual {final_res:.2e})",
                    RuntimeWarning, stacklevel=2)

        us = to_funs(U)
        return us[0] if m == 1 else SystemSolution(us)

    def expm(self, t: float, u0, n: int = 128):
        """exp(t*L) applied to u0 for the linearised operator.

        See :meth:`Linop.expm`.
        """
        return self._build_linop(value_shift=0.0).expm(t, u0, n=n)

    def matrix(self, n: int):
        """The n x n discretization matrix with BC rows (MATLAB matrix(L,n))."""
        return self._build_linop(value_shift=0.0).matrix(n)

    def eigs(
        self,
        *,
        n: int | None = None,
        k: int = 6,
        n_default: int = 64,
        sigma=None,
        return_eigenfunctions: bool = False,
    ):
        """Eigenvalues of the (linearised) operator.

        Constructs the :class:`Linop` corresponding to the (linearised)
        operator and calls :meth:`Linop.eigs`.

        Parameters
        ----------
        n : int or None
            Discretization size.
        k : int, default 6
            Number of eigenvalues to return.
        n_default : int, default 64
            Default size when ``n`` is ``None``.
        sigma : scalar or str or None
            Target eigenvalue or string selector (see :meth:`Linop.eigs`).

        Returns
        -------
        lam : jnp.ndarray, shape (k,)
            Selected eigenvalues.

        Provenance
        ----------
        MATLAB source : @chebop/eigs.m
        Chebfun commit: 7574c77
        """
        if getattr(self, "_periodic", False) and self._n_vars() < 2:
            return self._eigs_periodic(
                k=k, n=n, sigma=sigma,
                return_eigenfunctions=return_eigenfunctions)
        if self._n_vars() >= 2:
            return self._eigs_system(k=k, n=n or n_default)
        linop = self._build_linop(value_shift=0.0)
        return linop.eigs(n=n, k=k, n_default=n_default, sigma=sigma,
                          return_eigenfunctions=return_eigenfunctions)

    def _eigs_periodic(self, k: int = 6, n: int | None = None,
                       sigma=None, return_eigenfunctions: bool = False):
        """Eigenvalues of a linear periodic operator by Fourier collocation.

        Assembles the operator matrix on an equispaced periodic grid with
        the Fourier differentiation matrix (the same :class:`_FourierProxy`
        machinery :meth:`_solve_periodic` uses), so ``N[u] = lambda u``
        becomes the algebraic eigenproblem ``A v = lambda v``.  The ``k``
        modes selected by ``sigma`` are returned (default: algebraically
        smallest, matching MATLAB's ``eigs(L, k)`` for periodic
        Sturm-Liouville problems).

        ``sigma`` accepts the same selectors as :meth:`Linop.eigs`:
        ``None``/``'SM'`` (smallest magnitude), ``'LR'``/``'SR'`` (largest
        / smallest real part), ``'LM'`` (largest magnitude), or a scalar
        target (nearest to ``sigma``).  MATLAB's periodic Mathieu example
        uses ``'LR'`` to pick the smooth low-order elliptic modes.

        With ``return_eigenfunctions=True`` returns ``(V, lam)`` where each
        eigenfunction is a trigonometric :class:`Chebfun`, L2-normalized
        with the MATLAB sign convention (real part positive just right of
        the domain midpoint); otherwise returns ``lam`` alone.

        Provenance
        ----------
        MATLAB source : @chebop/eigs.m, @linop/eigs.m (trigcolloc branch)
        Chebfun commit: 7574c77
        """
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import chebfun
        a, b = self.domain
        Lp = float(b - a)
        N = 64 if n is None else int(n)
        x = a + Lp * _np.arange(N) / N
        proxy = _FourierProxy(N, Lp, _np.eye(N), grid=x)
        out = self._apply_op(jnp.asarray(x), proxy)
        if not isinstance(out, _FourierProxy):
            raise TypeError(
                "Chebop._eigs_periodic: operator is not linear in u.")
        A = _np.asarray(out.mat)
        lam, W = _np.linalg.eig(A)

        # Selection key: mirror Linop.eigs so the periodic path honours the
        # same ``sigma`` selectors (default: algebraically smallest, i.e.
        # smallest real part -- the lowest periodic Sturm-Liouville modes).
        if sigma is None:
            key = _np.real(lam)
        elif sigma == "SM":
            key = _np.abs(lam)
        elif sigma == "LM":
            key = -_np.abs(lam)
        elif sigma == "LR":
            key = -_np.real(lam)
        elif sigma == "SR":
            key = _np.real(lam)
        elif sigma == "LI":
            key = -_np.imag(lam)
        elif sigma == "SI":
            key = _np.imag(lam)
        elif isinstance(sigma, (int, float, complex)):
            key = _np.abs(lam - sigma)
        else:
            raise ValueError(
                f"Chebop._eigs_periodic: unrecognised sigma={sigma!r}.")
        order = _np.argsort(key, kind="stable")[:k]
        lam_sel = lam[order]
        if _np.max(_np.abs(_np.imag(lam_sel))) < 1e-9 * max(
                1.0, _np.max(_np.abs(lam_sel))):
            lam_sel = _np.real(lam_sel)
        lam_out = jnp.asarray(lam_sel)
        if not return_eigenfunctions:
            return lam_out

        xs = jnp.asarray(x, dtype=jnp.float64)
        # MATLAB @linop/eigs.m fixes the sign of each eigenfunction from the
        # real part just right of the domain midpoint.
        x_sign = a + Lp * 0.500023981
        V = []
        for idx in order:
            w = W[:, idx]
            # Real trig eigenfunction: for a real operator eigenvectors are
            # real up to a global phase; take whichever of real/imag part
            # carries the amplitude.
            wr = w.real if _np.max(_np.abs(w.real)) >= _np.max(
                _np.abs(w.imag)) else w.imag
            vals = jnp.asarray(wr, dtype=jnp.float64)

            def _interp(t, vals=vals):
                return _fourier_interp(xs, vals, jnp.asarray(t), a, Lp)

            u = chebfun(_interp, domain=(a, b), trig=True)
            nrm = float(u.norm(2))
            if nrm > 0:
                u = u * (1.0 / nrm)
            s = float(u(jnp.asarray(x_sign)))
            if s < 0:
                u = -u
            V.append(u)
        return V, lam_out

    def __truediv__(self, f):
        """``N \\ f`` syntax — solve N[u] = f."""
        return self.solve(f)

    def __repr__(self) -> str:
        """MATLAB-style linear-operator display (@chebop/display).

        Reproduces the published format::

               Linear operator:
                  u |--> diff(u,2)+u
               operating on chebfun objects defined on:
                  [-1,1]
               with
                left boundary condition(s):
                  u = 0

        The op string comes from ``_disp_op_str`` when set (e.g. by
        :func:`~chebfunjax.operators.adjoint.adjoint`) or is recovered
        from the lambda source with chebfun-method-to-MATLAB rewriting
        (``u.diff(2)`` -> ``diff(u,2)``, ``*`` -> ``.*``).

        Provenance
        ----------
        MATLAB source : @chebop/display.m
        Chebfun commit: 7574c77
        """
        import inspect
        import re

        a, b = (float(self.domain[0]), float(self.domain[-1]))

        var = getattr(self, "_disp_var", None)
        op_str = getattr(self, "_disp_op_str", None)
        if op_str is None and self.op is not None:
            try:
                params = list(
                    inspect.signature(self.op).parameters.keys())
                var = params[-1]
                src = inspect.getsource(self.op)
                body = src.split(":", 1)[1]
                # Cut at the first comma/paren at lambda-arg depth 0
                # (the lambda is usually inline in a call).
                depth = 0
                out_chars = []
                for ch in body:
                    if ch in "([{":
                        depth += 1
                    elif ch in ")]}":
                        if depth == 0:
                            break
                        depth -= 1
                    elif ch == "," and depth == 0:
                        break
                    elif ch == "\n":
                        break
                    out_chars.append(ch)
                body = "".join(out_chars).strip()
                body = re.sub(r"(\w+)\.diff\((\d+)\)", r"diff(\1,\2)",
                              body)
                body = re.sub(r"(\w+)\.diff\(\)", r"diff(\1)", body)
                body = re.sub(
                    r"(\w+)\.(sin|cos|exp|tan|sinh|cosh|tanh|sqrt|log)"
                    r"\(\)", r"\2(\1)", body)
                body = body.replace(" ", "").replace("**", "^")
                body = body.replace("*", ".*").replace("/", "./")
                op_str = body
            except Exception:
                op_str = "<op>"
        if var is None:
            var = "u"

        def _fmt_num(v):
            fv = float(v)
            return str(int(fv)) if fv == int(fv) else f"{fv:g}"

        def _bc_lines(spec, disp_list):
            if disp_list:
                if len(disp_list) == 1:
                    return [f"{disp_list[0]} = 0"]
                return ["[" + ";".join(disp_list) + "] = 0"]
            if spec is None:
                return []
            if isinstance(spec, (int, float)):
                return [f"{var} = {_fmt_num(spec)}"]
            if isinstance(spec, (list, tuple)):
                primes = [var + "'" * i for i in range(len(spec))]
                w = max(len(p) for p in primes)
                return [f"{p:<{w}} = {_fmt_num(v)}"
                        for p, v in zip(primes, spec)]
            return [f"{var} = 0"]

        def _dom_str(v):
            fv = float(v)
            return str(int(fv)) if fv == int(fv) else f"{fv:g}"

        lines = ["   Linear operator:",
                 f"      {var} |--> {op_str}",
                 "   operating on chebfun objects defined on:",
                 f"      [{_dom_str(a)},{_dom_str(b)}]"]
        lbc = _bc_lines(self._lbc_raw, getattr(self, "_disp_lbc", None))
        rbc = _bc_lines(self._rbc_raw, getattr(self, "_disp_rbc", None))
        if lbc or rbc or getattr(self, "_periodic", False):
            lines.append("   with")
        if lbc:
            lines.append("    left boundary condition(s):")
            lines.extend(f"      {ln}" for ln in lbc)
        if rbc:
            lines.append("    right boundary condition(s):")
            lines.extend(f"      {ln}" for ln in rbc)
        if getattr(self, "_periodic", False):
            lines.append("    periodic boundary conditions.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Linearity detection
    # ------------------------------------------------------------------

    def _is_linear(self) -> bool:
        """Linearity detection using ADChebfun symbolic AD.

        First tries the exact ``detect_linearity`` approach using ADChebfun
        (which checks whether the Fréchet derivative is constant).  Falls back
        to the numerical probe if symbolic detection fails or raises.

        This is conservative: if in doubt, returns ``False``.
        """
        # A deflated operator carries a nonlinear multiplicative factor
        # M(u; r) and is never linear (probing it with AD would trip over the
        # norm reductions inside M).
        if self._deflation is not None:
            return False
        # If op is already an OperatorBlock, definitely linear
        if isinstance(self.op, OperatorBlock):
            return True
        # Try exact symbolic linearity detection via ADChebfun
        try:
            from chebfunjax.autodiff.adchebfun import detect_linearity
            from chebfunjax.chebfun1d.chebfun import chebfun as _chebfun
            a, b = self.domain
            u_probe = _chebfun(
                lambda x: jnp.sin(jnp.pi * (x - a) / (b - a)),
                domain=self.domain,
                n=8,
            )
            return detect_linearity(self.op, u_probe, domain=self.domain)
        except Exception:
            pass
        # Fall back to numerical probe
        try:
            return self._probe_linearity()
        except Exception:
            return False

    def _probe_linearity(self) -> bool:
        """Numerical linearity probe.

        Evaluates the operator on three Chebfuns: the zero function, a
        sinusoidal probe ``p``, and ``2*p``.  If::

            op(2*p) - op(0) ≈ 2*(op(p) - op(0))

        then the operator is (approximately) affine, hence linear for
        zero-offset operations.
        """
        from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
        dom = Domain(self.domain)
        a, b = self.domain
        # chebfun() factory expects a sequence for domain, not a Domain object
        dom_tup = (a, b)

        # Use a simple harmonic probe
        probe = chebfun(lambda x: jnp.sin(jnp.pi * (x - a) / (b - a)), domain=dom_tup, n=8)
        zero_fun = Chebfun.from_values(jnp.zeros(8, dtype=jnp.float64), dom)
        x_fun = Chebfun.identity(dom)

        try:
            op0 = self._apply_op(x_fun, zero_fun)
            op1 = self._apply_op(x_fun, probe)
            op2 = self._apply_op(x_fun, 2.0 * probe)
        except Exception:
            return False

        # Evaluate at a test point
        mid = 0.5 * (a + b)
        x_mid = jnp.array(mid, dtype=jnp.float64)

        v0 = float(_safe_eval(op0, x_mid))
        v1 = float(_safe_eval(op1, x_mid))
        v2 = float(_safe_eval(op2, x_mid))

        diff = abs(v2 - v0 - 2.0 * (v1 - v0))
        scale = max(abs(v0), abs(v1), abs(v2), 1e-10)
        return diff / scale < 1e-6

    def _solve_periodic(self, f=0.0, n=None, n_max: int = 2048,
                        tol: float = 1e-10):
        """Solve a linear periodic BVP by Fourier collocation (Opus 4.8).

        Discretises with the Fourier differentiation matrix on N
        equispaced points; periodicity is built into the basis, so no
        endpoint constraints are needed.  Adaptively doubles N until the
        solution's trailing Fourier coefficients decay below ``tol``.
        Returns the solution as a trig Chebfun.
        """
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import chebfun
        a, b = self.domain
        L = float(b - a)

        def rhs_at(pts):
            if callable(f):
                return _np.asarray(f(jnp.asarray(pts)), dtype=float)
            return _np.full(pts.shape, float(f))

        N = 16 if n is None else int(n)
        u_vals = None
        while True:
            x = a + L * _np.arange(N) / N          # equispaced, periodic
            try:
                proxy = _FourierProxy(N, L, _np.eye(N), grid=x)
                out = self._apply_op(jnp.asarray(x), proxy)
                if not isinstance(out, _FourierProxy):
                    raise TypeError("nonlinear")
            except (TypeError, AttributeError, ValueError):
                # Nonlinear periodic operator: Newton on the grid.
                return self._solve_periodic_nonlinear(
                    f, n=n, n_max=n_max, tol=tol)
            A = out.mat
            rhs = rhs_at(x)
            u_vals = _np.linalg.solve(A, rhs)
            if n is not None:
                break
            # resolution check: trailing Fourier coeffs small?
            coeffs = _np.fft.fft(u_vals) / N
            tail = _np.max(_np.abs(coeffs[N // 4: 3 * N // 4]))
            scale = max(_np.max(_np.abs(coeffs)), 1e-14)
            if tail / scale < tol or N >= n_max:
                break
            N *= 2

        # Build a trig Chebfun that interpolates the periodic samples.
        vals = jnp.asarray(u_vals, dtype=jnp.float64)
        xs = jnp.asarray(x, dtype=jnp.float64)

        def interp(t):
            return _fourier_interp(xs, vals, jnp.asarray(t), a, L)

        return chebfun(interp, domain=(a, b), trig=True)

    def _solve_periodic_nonlinear(self, f=0.0, n=None,
                                  n_max: int = 1024,
                                  tol: float = 1e-10,
                                  max_iter: int = 30):
        """Nonlinear periodic BVP by damped Newton on the Fourier grid.

        The operator is evaluated through the value-space
        :class:`_TrigVals` proxy; the Jacobian is built column-by-column
        by finite differences (each evaluation is an O(n^2) matvec
        chain, cheap at the sizes involved), and levels are seeded with
        the interpolated previous solution (grid continuation).

        Provenance
        ----------
        MATLAB source : @chebop/solvebvpNonlinear.m (trigcolloc branch)
        Chebfun commit: 7574c77
        """
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import chebfun
        a, b = self.domain
        L = float(b - a)

        def rhs_at(pts):
            if callable(f):
                return _np.asarray(f(jnp.asarray(pts)), dtype=float)
            return _np.full(pts.shape, float(f))

        def op_vals(U, N):
            pr = _TrigVals(U, N, L)
            xg = a + L * _np.arange(N) / N
            out = self._apply_op(_TrigX(jnp.asarray(xg)), pr)
            if isinstance(out, _TrigVals):
                return _np.asarray(out.v, dtype=float)
            return _np.asarray(out, dtype=float)

        N = 32 if n is None else int(n)
        U = None
        x = None
        while True:
            x = a + L * _np.arange(N) / N
            if U is None:
                U = (_np.asarray(self.init(jnp.asarray(x)), dtype=float)
                     if self.init is not None else _np.zeros(N))
            import scipy.linalg as _sla
            g = rhs_at(x)
            R = op_vals(U, N) - g
            for _it in range(max_iter):
                nrm = _np.max(_np.abs(R))
                if nrm < 1e-11:
                    break
                J = _np.zeros((N, N))
                h = 1e-7 * max(1.0, _np.max(_np.abs(U)))
                for jc in range(N):
                    Up = U.copy()
                    Up[jc] += h
                    J[:, jc] = (op_vals(Up, N) - g - R) / h
                try:
                    lu = _sla.lu_factor(J)
                except (ValueError, _np.linalg.LinAlgError):
                    break
                dU = _sla.lu_solve(lu, R)
                nd = _np.linalg.norm(dU)
                # Affine-invariant (Deuflhard) damping: require the
                # SIMPLIFIED Newton step to shrink.  Residual-monotone
                # backtracking pulls iterates out of the init's basin
                # toward globally-dominant solutions — MATLAB's
                # solvebvpNonlinear keeps the basin, and so does this.
                lam = 1.0
                Rn = R
                for _d in range(25):
                    Rn = op_vals(U - lam * dU, N) - g
                    if not _np.all(_np.isfinite(Rn)):
                        lam *= 0.5
                        continue
                    dU_bar = _sla.lu_solve(lu, Rn)
                    if _np.linalg.norm(dU_bar) < nd or lam < 1e-8:
                        break
                    lam *= 0.5
                U = U - lam * dU
                if _np.max(_np.abs(Rn)) >= nrm * (1.0 - 1e-12) \
                        and lam >= 1.0:
                    R = Rn
                    break
                R = Rn
            if n is not None:
                break
            coeffs = _np.fft.fft(U) / N
            tail = _np.max(_np.abs(coeffs[N // 4: 3 * N // 4]))
            scale = max(_np.max(_np.abs(coeffs)), 1e-14)
            if tail / scale < tol or N >= n_max:
                break
            # interpolate onto the doubled grid (grid continuation)
            vals = jnp.asarray(U, dtype=jnp.float64)
            xs = jnp.asarray(x, dtype=jnp.float64)
            N *= 2
            x2 = a + L * _np.arange(N) / N
            U = _np.asarray(_fourier_interp(
                xs, vals, jnp.asarray(x2), a, L), dtype=float)

        vals = jnp.asarray(U, dtype=jnp.float64)
        xs = jnp.asarray(x, dtype=jnp.float64)

        def interp(t):
            return _fourier_interp(xs, vals, jnp.asarray(t), a, L)

        return chebfun(interp, domain=(a, b), trig=True)

    def _is_ivp(self) -> bool:
        """True if all boundary conditions sit at a single endpoint."""
        if getattr(self, "_periodic", False):
            return False
        has_l = self._lbc_raw is not None
        has_r = self._rbc_raw is not None
        return (has_l and not has_r) or (has_r and not has_l)

    def _op_order(self) -> int:
        import inspect
        sniff = _OrderSniffer()
        a, b = self.domain
        x = jnp.asarray(0.5 * (a + b))
        nargs = len(inspect.signature(self.op).parameters)
        try:
            _ = self.op(x, sniff) if nargs == 2 else self.op(sniff)
        except AttributeError:
            # chebfun-style op (x.cos() ...): retry with the wrapper.
            _ = self.op(_TrigX(x), sniff)
        return sniff.order

    def _bc_values(self, bc_raw, k: int):
        """Extract the initial values [c0, ..., c_{k-1}] from a BC spec.

        Robust version (Fable 5 audit of the Opus 4.8 original): instead
        of assuming the BC list is ordered ``[u, u', ...]`` with unit
        coefficients, evaluate the residuals at one-hot derivative towers
        to build the Jacobian ``B`` and solve ``B @ ic = -r0``.  Raises
        if the BCs are not k independent affine constraints (caller then
        falls back to collocation).
        """
        import numpy as _np
        if bc_raw is None:
            return None
        if not callable(bc_raw):
            return [float(bc_raw)]           # scalar Dirichlet value

        def _residuals(tower_vals):
            tower = [jnp.asarray(v) for v in tower_vals] + [jnp.array(0.0)]
            res = bc_raw(_IVPProxy(tower))
            if not isinstance(res, (list, tuple)):
                res = [res]
            out = []
            for r in res:
                if isinstance(r, _IVPProxy):
                    r = r._v
                out.append(float(_np.asarray(r)))
            return out

        r0 = _np.asarray(_residuals([0.0] * k))
        m = r0.shape[0]
        if m != k:
            raise ValueError(
                f"IVP needs {k} conditions at one endpoint, got {m}")
        B = _np.zeros((m, k))
        for j in range(k):
            e = [0.0] * k
            e[j] = 1.0
            B[:, j] = _np.asarray(_residuals(e)) - r0
        # affinity check: residuals at 2*e_j must extrapolate linearly
        e2 = [0.0] * k
        e2[0] = 2.0
        r2 = _np.asarray(_residuals(e2))
        if not _np.allclose(r2, r0 + 2.0 * B[:, 0], rtol=1e-9, atol=1e-9):
            raise ValueError("boundary conditions are not affine")
        return list(_np.linalg.solve(B, -r0))

    def _solve_periodic_colloc(self, f=0.0, n=None, n_min: int = 8,
                               n_max: int = 4096, tol: float = 1e-10):
        """Periodic BVP by Chebyshev collocation with wrap-around rows.

        Provenance
        ----------
        MATLAB source : @chebcolloc2 discretization of a periodic linop
            (continuity rows via @linop/continuity.m)
        Chebfun commit: 7574c77
        """
        from chebfunjax.operators.blocks import FunctionalBlock
        from chebfunjax.operators.linop import Linop
        from chebfunjax.utils.diffmat import diffmat as _diffmat

        k = self._op_order()
        op_block = self._linearize_op()
        dom = tuple(float(v) for v in self.domain)

        def _wrap_row(order_d):
            def _fn(disc):
                nloc = disc.n
                if order_d == 0:
                    row = jnp.zeros(nloc, dtype=jnp.float64)
                    return row.at[0].set(1.0).at[nloc - 1].set(-1.0)
                D = _diffmat(nloc, order_d, domain=disc.domain)
                return D[0, :] - D[nloc - 1, :]
            return FunctionalBlock(_fn, domain=dom)

        wraps = [_wrap_row(d) for d in range(k)]
        linop = Linop(op_block, bcs=wraps, domain=dom,
                      bc_values=[0.0] * k)
        rhs = _make_rhs_callable(f)
        return linop.solve(rhs, n=n, n_min=n_min, n_max=n_max, tol=tol)

    def solve_ivp(self, f=0.0, rtol: float = 1e-11, atol: float = 1e-12):
        """Solve an initial-value problem by time marching (task #24).

        Applicable when all boundary conditions sit at one endpoint.  The
        operator is assumed affine in its highest derivative (true for
        essentially all ODEs): the k-th derivative is extracted as
        ``u^{(k)} = (f - L|_{u^{(k)}=0}) / (L|_{u^{(k)}=1} - L|_{u^{(k)}=0})``
        and the resulting first-order system is integrated with
        ``scipy.integrate.solve_ivp`` (Dormand--Prince).  Returns the
        solution ``u`` as a Chebfun.  Implemented by Claude Opus 4.8.

        Provenance
        ----------
        MATLAB source : @chebop/solveivp.m (routing of one-sided BCs).
        Chebfun commit: 7574c77
        """
        import inspect

        import numpy as _np
        from scipy.integrate import solve_ivp as _solve_ivp

        from chebfunjax.chebfun1d.chebfun import chebfun
        a, b = self.domain
        k = self._op_order()
        nargs = len(inspect.signature(self.op).parameters)

        def L(x, u):
            return self.op(x, u) if nargs == 2 else self.op(u)

        def fval(x):
            if callable(f):
                return float(_np.asarray(f(jnp.asarray(x))))
            return float(f)

        # initial conditions and marching direction
        left = self._lbc_raw is not None
        bc_raw = self._lbc_raw if left else self._rbc_raw
        ic = self._bc_values(bc_raw, k)
        if ic is None or len(ic) != k:
            raise ValueError(
                f"IVP requires exactly {k} initial conditions")
        x0, x1 = (a, b) if left else (b, a)

        def _op_at(x, y, s):
            tower = [jnp.asarray(v) for v in list(y) + [s]]
            try:
                return float(_np.asarray(
                    L(jnp.asarray(x), _IVPProxy(tower))))
            except AttributeError:
                # chebfun-style op (x.cos() ...): wrap the scalar x so
                # the method chain works (see _TrigX / _apply_op).
                r = L(_TrigX(jnp.asarray(x)), _IVPProxy(tower))
                if isinstance(r, _TrigX):
                    r = r.v
                return float(_np.asarray(r))

        # Verify the operator is affine in the highest derivative before
        # trusting the extraction (Fable 5 audit: e.g. (u'')^2 would
        # otherwise be silently mis-extracted).  Checked at the initial
        # point; non-affine ops raise -> caller falls back to collocation.
        r0 = _op_at(x0, ic, 0.0)
        r1 = _op_at(x0, ic, 1.0)
        r2 = _op_at(x0, ic, 2.0)
        if not _np.isclose(r2 - r1, r1 - r0,
                           rtol=1e-8, atol=1e-8 * (abs(r1 - r0) + 1.0)):
            raise ValueError(
                "operator is not affine in its highest derivative")

        def rhs(x, y):
            lower = _op_at(x, y, 0.0)
            ak = _op_at(x, y, 1.0) - lower
            ukk = (fval(x) - lower) / ak
            return list(y[1:]) + [ukk]

        # LSODA switches between stiff/non-stiff automatically, so a
        # stiff problem cannot grind RK45 into a CI timeout.
        sol = _solve_ivp(rhs, [x0, x1], ic, dense_output=True,
                         method="LSODA", rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        def u_eval(x):
            xn = _np.atleast_1d(_np.asarray(x, dtype=float))
            vals = sol.sol(xn)[0]
            return jnp.asarray(vals.reshape(_np.shape(x)) if _np.ndim(x)
                               else vals[0], dtype=jnp.float64)

        return chebfun(lambda x: u_eval(x), domain=(a, b))

    def __call__(self, u):
        """Apply the operator to a chebfun (MATLAB N(u) / N*u).

        An integer argument realizes the dense collocation (differentiation)
        matrix at that grid size, matching MATLAB's deprecated ``D(n)`` /
        ``feval(D, n)`` syntax (ATAP chapter 21).  It is equivalent to
        :meth:`matrix`.

        Provenance
        ----------
        MATLAB source : @chebop/feval.m, @chebop/mtimes.m
        Chebfun commit: 7574c77
        """
        import numbers
        if isinstance(u, numbers.Integral) and not isinstance(u, bool):
            return self.matrix(int(u))
        from chebfunjax.chebfun1d.chebfun import Chebfun
        x_fun = Chebfun.identity(Domain(self.domain))
        if isinstance(u, (list, tuple)) or (
                self._n_vars() >= 2 and hasattr(u, "__getitem__")
                and not isinstance(u, Chebfun)):
            out = self.op(x_fun, *list(u))
            return SystemSolution(list(out)) \
                if isinstance(out, (list, tuple)) else out
        return self._apply_op(x_fun, u)

    def _call_op(self, x_fun, us):
        """Call self.op with or without the leading x argument, matching
        the op's arity (ops of one unknown may be written either as
        lambda u: ... or lambda x, u: ...)."""
        import inspect
        try:
            nargs = len(inspect.signature(self.op).parameters)
        except (TypeError, ValueError):
            nargs = len(us) + 1
        return (self.op(x_fun, *us) if nargs > len(us)
                else self.op(*us))

    def _apply_op(self, x_fun, u_fun):
        """Evaluate self.op(x_fun, u_fun) or self.op(u_fun).

        The periodic discretizations pass ``x_fun`` as a raw value array,
        so an op written chebfun-style (``x.cos()``, as MATLAB's
        ``cos(x)`` allows on either type) raises AttributeError; retry
        with a thin wrapper that supports the chebfun-style elementwise
        methods while degrading to plain arrays in mixed arithmetic.
        """
        import inspect
        try:
            n = len(inspect.signature(self.op).parameters)
        except (TypeError, ValueError):
            n = 2  # default: assume (x, u)

        if n == 1:
            return self.op(u_fun)
        try:
            return self.op(x_fun, u_fun)
        except AttributeError:
            if hasattr(x_fun, "dtype"):
                out = self.op(_TrigX(x_fun), u_fun)
                if isinstance(out, _TrigX):
                    out = out.v
                elif isinstance(out, (list, tuple)):
                    out = type(out)(o.v if isinstance(o, _TrigX) else o
                                    for o in out)
                return out
            raise

    # ------------------------------------------------------------------
    # Linear solve
    # ------------------------------------------------------------------

    def _build_linop(self, value_shift: float = 0.0) -> Linop:
        """Build a Linop from the operator and BCs.

        The operator ``self.op`` is called on a symbolic proxy (Chebfun
        wrapping the columns of the differentiation matrix) to extract the
        :class:`OperatorBlock`.  For non-symbolic ops (pure callables), a
        numerical linearization approach is used.

        For the simple case where ``self.op`` *is* an ``OperatorBlock``, it
        is used directly.

        Parameters
        ----------
        value_shift : float
            Not used in the linear case; kept for API symmetry with Newton.

        Returns
        -------
        Linop
        """
        a, b = self.domain

        # Case 1: op is already an OperatorBlock
        if isinstance(self.op, OperatorBlock):
            op_block = self.op
        else:
            # Case 2: op is a callable — we need to extract the OperatorBlock.
            # We do this by calling op on an "adchebfun-style" proxy that tracks
            # linear operator composition.  We use a simpler approach:
            # finite-difference linearization at the zero function to get the
            # operator matrix, then wrap it in a generic OperatorBlock.
            op_block = self._linearize_op()

        bcs, bc_vals = self._parse_bcs()

        return Linop(op_block, bcs=bcs, domain=self.domain, bc_values=bc_vals)

    def _linearize_op(self) -> OperatorBlock:
        """Linearize self.op around the zero function.

        Returns an OperatorBlock whose matrix at discretization n is the
        Frechet derivative of self.op at u=0, computed by finite differences
        on Chebfun column vectors.

        For linear operators, this is exact.  For nonlinear operators it is
        the Jacobian at u=0 (used as a starting Linop for Newton iteration).

        Notes
        -----
        The Frechet derivative is approximated as::

            J_ij ≈ [op(e_j) - op(0)]_i / h

        where e_j is the j-th standard basis vector (unit values at the
        j-th Chebyshev point) and h is a small perturbation.  In the linear
        case h=1 and the formula is exact.

        This is a Python-level operation and NOT JIT-safe.

        Assembly strategy
        -----------------
        A variable-coefficient linear operator has the form

            L[u] = c_0(x) u + c_1(x) u' + ... + c_m(x) u^(m).

        The collocation matrix is therefore

            A = sum_k diag(c_k(nodes)) @ D^k

        where ``D^k = diffmat(n, k)``.  We extract the coefficient functions
        ``c_k`` with only ``m + 1`` operator applications (independent of
        ``n``) by probing ``L`` on the scaled monomials ``y^k/k!`` and
        forward-substituting.  This replaces the previous
        column-by-column probe (``n`` operator applications assembled into a
        single ``jnp.stack``), whose XLA graph grew as ``O(n)`` and, for the
        large / stiff variable-coefficient BVPs of guide chapter 7, blew up
        the compile (an ~11 min ``jit_stack`` compile followed by an
        out-of-memory segfault at ``n`` ~ 1024).

        For operators the structured form cannot represent (e.g. integral /
        nonlocal terms) the monomial probe either fails or the assembled
        matrix does not reproduce ``L`` on a test function; in that case we
        fall back to the general column-by-column probe, materialised
        eagerly column-by-column (never a monolithic ``jnp.stack``) so the
        fallback stays bounded in memory.
        """
        domain = self.domain
        max_order = 8

        def _op_fn(disc: ChebColloc2Disc) -> jnp.ndarray:
            import numpy as _np

            from chebfunjax.chebfun1d.chebfun import Chebfun
            n = disc.n
            a, b = disc.domain
            dom = Domain((a, b))
            x_fun = Chebfun.identity(dom)

            def _op_at(vals):
                """L applied to the Chebfun with these nodal values -> nodal
                values of L[u] at the collocation nodes (the affine constant
                L[0] is NOT removed here; callers subtract ``op0``)."""
                uf = Chebfun.from_values(jnp.asarray(vals, dtype=jnp.float64),
                                         dom)
                # Preserve complex outputs: a complex-shifted operator
                # (e.g. Talbot-contour Helmholtz solves, zk complex) was
                # previously silently real-cast to WRONG values.
                out = _np.asarray(_chebfun_to_values(
                    self._apply_op(x_fun, uf), disc))
                if not _np.iscomplexobj(out):
                    out = out.astype(_np.float64)
                return out

            # Constant part L[0] (nonzero for affine operators).
            try:
                op0 = _op_at(_np.zeros(n))
            except Exception:
                op0 = _np.zeros(n)

            # Physical Chebyshev-2 nodes and the domain-scaled variable y.
            t_ref = _np.asarray(chebpts(n, kind=2))
            nodes = 0.5 * (b - a) * t_ref + 0.5 * (a + b)
            x0 = 0.5 * (a + b)
            h = 0.5 * (b - a)
            y = (nodes - x0) / h

            # ---- Structured assembly (fast: m+1 operator applications). ----
            m = self._sniff_order(x_fun, max_order)
            if m is not None:
                try:
                    from math import factorial

                    from chebfunjax.utils.diffmat import diffmat
                    # L[y^k/k!] at the nodes (constant part removed).
                    Lp = [_op_at((y ** k) / factorial(k)) - op0
                          for k in range(m + 1)]
                    # Forward-substitute for c_k:  c_k/h^k = L[p_k] -
                    #   sum_{j<k} c_j y^{k-j} / (h^j (k-j)!).
                    coeffs = []
                    for k in range(m + 1):
                        acc = Lp[k].copy()
                        for j in range(k):
                            acc = acc - coeffs[j] * (
                                (y ** (k - j))
                                / (h ** j * factorial(k - j)))
                        coeffs.append(acc * (h ** k))
                    dt = _np.result_type(_np.float64,
                                         *[c.dtype for c in coeffs])
                    A = _np.zeros((n, n), dtype=dt)
                    for k in range(m + 1):
                        A = A + coeffs[k][:, None] * _np.asarray(
                            diffmat(n, k, domain=(a, b)))
                    if self._assembly_ok(A, op0, _op_at, y):
                        return jnp.asarray(A)
                except Exception:
                    pass

            # ---- Fallback: general column probe, eager per column. ----
            A = _np.empty((n, n),
                          dtype=(_np.complex128 if _np.iscomplexobj(op0)
                                 else _np.float64))
            for j in range(n):
                e_j = _np.zeros(n)
                e_j[j] = 1.0
                col = _op_at(e_j) - op0
                if _np.iscomplexobj(col) and not _np.iscomplexobj(A):
                    A = A.astype(_np.complex128)
                A[:, j] = col
            return jnp.asarray(A)

        return OperatorBlock(_op_fn, order=2, domain=domain)

    def _sniff_order(self, x_fun, max_order: int):
        """Highest derivative order the operator applies to ``u``.

        Probes ``self.op`` with an :class:`_OrderSniffer` in the ``u`` slot
        (and the real identity Chebfun in the ``x`` slot, so variable
        coefficients such as ``sin(x)`` evaluate normally).  Returns the
        order, or ``None`` if it cannot be determined or exceeds
        ``max_order`` (e.g. integral / nonlocal operators), signalling the
        caller to use the general column probe instead.
        """
        try:
            sniff = _OrderSniffer()
            self._apply_op(x_fun, sniff)
            m = int(sniff.order)
        except Exception:
            return None
        if m < 0 or m > max_order:
            return None
        return m

    @staticmethod
    def _assembly_ok(A, op0, op_at, y) -> bool:
        """True if the structured matrix ``A`` reproduces ``L`` on a test
        function -- guards against operators the differential form cannot
        represent (integral / nonlocal terms), for which the caller falls
        back to the general column probe."""
        import numpy as _np
        tv = (_np.cos(2.3 * y) + 0.4 * _np.sin(1.7 * y)
              + 0.2 * _np.cos(4.1 * y))
        lhs = op_at(tv) - op0
        rhs = A @ tv
        scale = max(float(_np.max(_np.abs(lhs))), 1e-30)
        return bool(_np.max(_np.abs(lhs - rhs)) / scale < 1e-6)

    def _solve_linear(
        self,
        f,
        n: int | None,
        n_min: int,
        n_max: int,
        tol: float,
    ):
        """Direct spectral solve for linear operators."""
        linop = self._build_linop()

        # RHS callable
        rhs = _make_rhs_callable(f)

        return linop.solve(rhs, n=n, n_min=n_min, n_max=n_max, tol=tol)

    # ------------------------------------------------------------------
    # Nonlinear solve (Newton iteration)
    # ------------------------------------------------------------------

    def _deflation_default_init_vals(self, x_pts):
        """Boundary-condition satisfying default initial guess values.

        MATLAB's chebop uses a default initial guess that satisfies the
        boundary conditions; from a plain zero guess the deflated Newton
        iteration for a non-homogeneous problem (e.g. Painleve, ``u(L) =
        sqrt(L)``) stalls.  For scalar Dirichlet endpoints ``u(a)=la`` and
        ``u(b)=lb`` the guess is the straight line through them; a single
        scalar endpoint gives a constant; callable (e.g. Neumann) endpoints
        fall back to zero.
        """
        a, b = self.domain
        la = self._lbc_raw if isinstance(self._lbc_raw, (int, float)) else None
        lb = self._rbc_raw if isinstance(self._rbc_raw, (int, float)) else None
        if la is not None and lb is not None:
            return la + (lb - la) * (jnp.asarray(x_pts) - a) / (b - a)
        if la is not None:
            return jnp.full_like(jnp.asarray(x_pts, dtype=jnp.float64),
                                 float(la))
        if lb is not None:
            return jnp.full_like(jnp.asarray(x_pts, dtype=jnp.float64),
                                 float(lb))
        return jnp.zeros_like(jnp.asarray(x_pts, dtype=jnp.float64))

    def _deflated_newton_once(
        self, disc, x_fun, bcs, bc_vals, f_vals, init_vals,
        max_iter, newton_tol,
    ):
        """One fixed-size damped-Newton solve of the deflated system.

        A monotone backtracking line search on the residual norm globalises
        the iteration.  Returns ``(u_fun, converged, r_norm, finite)`` where
        ``converged`` means the Newton step fell below tolerance and
        ``r_norm`` is the final collocation-residual infinity norm (small only
        at a genuine root).
        """
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import Chebfun

        dom = Domain(disc.domain)
        sz = disc.n
        n_bc = len(bcs)

        def _residual(uv):
            ufun = Chebfun.from_values(jnp.asarray(uv, dtype=jnp.float64), dom)
            Nu_fun = self._apply_op(x_fun, ufun)
            Nu_v = _np.array(_chebfun_to_values(Nu_fun, disc))
            rv = Nu_v - _np.asarray(f_vals)
            for i, (bc, bc_val) in enumerate(zip(bcs, bc_vals)):
                bc_row = _np.asarray(bc.matrix(disc))
                rv[sz - n_bc + i] = float(bc_row @ uv) - float(bc_val)
            return rv, Nu_v, ufun

        u_np = _np.asarray(init_vals, dtype=_np.float64).copy()
        r_np, Nu_v, ufun = _residual(u_np)
        r_norm = float(_np.max(_np.abs(r_np)))
        converged = False
        for _it in range(max_iter):
            J_mat = self._jacobian_matrix(disc, x_fun, ufun, jnp.asarray(Nu_v))
            J_np = _np.array(J_mat)
            for i, bc in enumerate(bcs):
                J_np[sz - n_bc + i, :] = _np.asarray(bc.matrix(disc))
            try:
                delta = _np.linalg.solve(J_np, -r_np)
            except _np.linalg.LinAlgError:
                break
            if not _np.all(_np.isfinite(delta)):
                break

            lam_damp = 1.0
            accepted = False
            for _ls in range(40):
                u_try = u_np + lam_damp * delta
                r_try, Nu_try, ufun_try = _residual(u_try)
                r_try_norm = float(_np.max(_np.abs(r_try)))
                if _np.isfinite(r_try_norm) and (
                    r_try_norm <= (1.0 - 1e-4 * lam_damp) * r_norm
                    or r_try_norm < newton_tol
                ):
                    accepted = True
                    break
                lam_damp *= 0.5
            if not accepted:
                break
            u_np, r_np, Nu_v, ufun = u_try, r_try, Nu_try, ufun_try
            r_norm = r_try_norm

            step_norm = float(_np.max(_np.abs(lam_damp * delta)))
            u_scale = max(1.0, float(_np.max(_np.abs(u_np))))
            if step_norm < newton_tol * u_scale:
                converged = True
                break

        u_vals = jnp.asarray(u_np, dtype=jnp.float64)
        finite = bool(jnp.isfinite(u_vals).all())
        u_fun = Chebfun.from_values(u_vals, dom) if finite else None
        return u_fun, converged, r_norm, finite

    def _deflation_const_candidates(self):
        """Constant initial-guess levels for the multi-start fallback.

        A single boundary-condition-satisfying guess is enough for most
        deflated problems, but when several solutions have already been
        deflated the remaining root can lie in a basin the default guess does
        not reach (e.g. Herceg's third solution sits near a *negative*
        constant while the deflated ones are positive).  Following the
        Farrell--Birkisson--Funke practice of pairing deflation with varied
        initial guesses, we sweep constants spanning -- and extending well
        beyond -- the value range of the already-found solutions.
        """
        import numpy as _np
        roots = self._deflation[1]
        a, b = self.domain
        t_ref = chebpts(32, kind=2)
        x_pts = 0.5 * (b - a) * t_ref + 0.5 * (a + b)
        vals = _np.concatenate([
            _np.asarray(r_k(x_pts), dtype=float) for r_k in roots])
        lo, hi = float(vals.min()), float(vals.max())
        span = max(hi - lo, 1.0)
        return list(_np.linspace(lo - 2.0 * span, hi + 2.0 * span, 9))

    def _solve_deflated(
        self,
        f,
        n: int | None,
        n_min: int,
        n_max: int,
        tol: float,
        max_iter: int,
        newton_tol: float,
    ):
        """Globalized damped-Newton solve for a deflated operator.

        Solves ``G(u) = M(u; r) N(u) = f`` (usually ``f = 0``) with the
        attached scalar boundary conditions and returns a solution of the
        *undeflated* problem distinct from every deflated root.  Three
        deflation-specific robustness features are layered on top of the
        adaptive-resolution strategy of :meth:`_solve_nonlinear`:

        * **Damped Newton from the start.**  A monotone backtracking line
          search keeps the iteration in the basin of a new solution; plain
          undamped Newton from a zero guess diverges on these problems.
        * **Boundary-condition satisfying default guess.**  When ``N.init`` is
          unset the iteration starts from
          :meth:`_deflation_default_init_vals` (matching MATLAB's default
          initial guess -- essential for non-homogeneous BCs such as
          Painleve's ``u(L) = sqrt(L)``).
        * **Multi-start fallback.**  If the default guess stalls at a
          non-root, or converges onto an already-deflated solution, the solve
          retries from a spread of constant guesses
          (:meth:`_deflation_const_candidates`) until a genuine *distinct*
          root is found.  A candidate that stalls or rediscovers a known root
          is abandoned immediately rather than refined, so the fallback stays
          cheap.

        The Jacobian is the exact product-rule Jacobian assembled by
        :meth:`_jacobian_matrix_deflated`.

        Provenance
        ----------
        MATLAB source : @chebop/deflate.m, @chebop/solvebvpNonlinear.m,
            @chebop/newtonBVP.m
        Chebfun commit: 7574c77

        References
        ----------
        P. E. Farrell, A. Birkisson, S. W. Funke, "Deflation techniques for
        finding distinct solutions of nonlinear partial differential
        equations", SIAM J. Sci. Comput. 37 (2015).
        """
        from chebfunjax.chebfun1d.chebfun import Chebfun
        from chebfunjax.tech.chebtech import Chebtech2

        a, b = self.domain
        dom = Domain(self.domain)
        rhs = _make_rhs_callable(f)
        bcs, bc_vals = self._parse_bcs()
        roots = self._deflation[1]

        fixed_size = n is not None
        start_sz = int(n) if fixed_size else max(n_min, 16)
        genuine_tol = 1e-6

        def _distinct(u_fun) -> bool:
            scale = max(1.0, float(u_fun.norm(2)))
            return all(
                float((u_fun - r_k).norm(2)) > 1e-3 * scale for r_k in roots)

        def _setup(sz):
            disc = ChebColloc2Disc(sz, self.domain)
            t_ref = chebpts(sz, kind=2)
            x_pts = 0.5 * (b - a) * t_ref + 0.5 * (a + b)
            f_vals = jnp.asarray(rhs(x_pts), dtype=jnp.float64)
            return disc, Chebfun.identity(dom), x_pts, f_vals

        def _run_adaptive(init_from):
            """Adaptive damped Newton from ``init_from(x_pts) -> values``.

            Returns ``(u_fun, ok)``.  ``ok`` is True only for a genuine,
            distinct, resolved root.  Aborts (without growing the grid) as
            soon as Newton stalls or lands on an already-deflated root, since
            refining resolution cannot fix a basin/identity problem.
            """
            sz = start_sz
            warm = None
            best = None
            while True:
                disc, x_fun, x_pts, f_vals = _setup(sz)
                init_vals = (warm(x_pts) if warm is not None
                             else init_from(x_pts))
                u_fun, converged, r_norm, finite = self._deflated_newton_once(
                    disc, x_fun, bcs, bc_vals, f_vals,
                    jnp.asarray(init_vals, dtype=jnp.float64),
                    max_iter, newton_tol)
                if not finite or not converged:
                    return best, False
                if not _distinct(u_fun):
                    return u_fun, False
                best = u_fun
                tech = u_fun.funs[0].tech
                resolved, _cut = Chebtech2.happiness_check(
                    tech.coeffs, tech.values)
                genuine = r_norm < genuine_tol
                if resolved or fixed_size or 2 * sz > n_max:
                    return u_fun, genuine
                warm = u_fun
                sz = 2 * sz

        # Primary attempt: N.init if provided, else BC-satisfying default.
        if self.init is not None:
            init_f = self.init

            def primary(xp):
                return init_f(xp) if callable(init_f) else init_f
        else:
            primary = self._deflation_default_init_vals

        u_fun, ok = _run_adaptive(primary)
        if ok:
            return u_fun
        best = u_fun

        # Multi-start fallback over constant guesses.
        for c in self._deflation_const_candidates():
            def const_init(xp, c=c):
                return jnp.full(len(jnp.asarray(xp)), float(c),
                                dtype=jnp.float64)
            u_c, ok_c = _run_adaptive(const_init)
            if ok_c:
                return u_c
            if best is None and u_c is not None:
                best = u_c

        warnings.warn(
            "Chebop.solve (deflated Newton): could not locate a new solution "
            "distinct from the deflated roots; returning best approximation.",
            stacklevel=3,
        )
        if best is not None:
            return best
        return Chebfun.from_values(
            jnp.zeros(start_sz, dtype=jnp.float64), dom)

    def _solve_nonlinear(
        self,
        f,
        n: int | None,
        n_min: int,
        n_max: int,
        tol: float,
        max_iter: int,
        newton_tol: float,
    ):
        """Newton iteration with adaptive discretization refinement.

        The previous implementation ran Newton at a FIXED grid (16 points
        unless n was given; n_max was accepted but never used) and returned
        whatever the coarse collocation system converged to — for stiff
        problems that solution satisfies the BCs but grossly violates the
        ODE (residual O(100)). Following MATLAB solvebvpNonlinear's
        adaptive strategy:

        1. Newton-solve the collocation system at size sz (warm-started by
           interpolating the previous size's iterate; the user's N.init is
           honoured at the first size).
        2. Check that the solution is RESOLVED at sz via the Chebyshev
           coefficient-tail happiness check (an under-resolved solution has
           a fat tail).
        3. If unresolved (or Newton failed), double sz and continue — the
           coarse solution warm-starts the next level, acting as a natural
           continuation strategy for stiff problems.

        Provenance
        ----------
        MATLAB source : @chebop/solvebvpNonlinear.m, @chebop/newtonBVP.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import Chebfun
        from chebfunjax.tech.chebtech import Chebtech2

        a, b = self.domain
        dom = Domain(self.domain)
        rhs = _make_rhs_callable(f)
        bcs, bc_vals = self._parse_bcs()
        n_bc = len(bcs)

        fixed_size = n is not None
        sz = int(n) if fixed_size else max(n_min, 16)
        u_fun_prev = None
        correction_norm = float("inf")

        while True:
            disc = ChebColloc2Disc(sz, self.domain)
            t_ref = chebpts(sz, kind=2)
            x_pts = 0.5 * (b - a) * t_ref + 0.5 * (a + b)
            f_vals = jnp.asarray(rhs(x_pts), dtype=jnp.float64)

            # Initial iterate: warm start from the previous size, else the
            # user-provided N.init, else zero.
            if u_fun_prev is not None:
                u_vals = jnp.asarray(u_fun_prev(x_pts), dtype=jnp.float64)
            elif self.init is not None:
                init_f = self.init
                u_vals = jnp.asarray(
                    init_f(x_pts) if callable(init_f) else init_f,
                    dtype=jnp.float64,
                )
            else:
                u_vals = jnp.zeros(sz, dtype=jnp.float64)

            x_fun = Chebfun.identity(dom)
            import numpy as _np

            def _residual(uv):
                """Collocation residual with BC rows replaced by BC errors."""
                ufun = Chebfun.from_values(jnp.asarray(uv, dtype=jnp.float64), dom)
                Nu_fun = self._apply_op(x_fun, ufun)
                Nu_v = _np.asarray(_chebfun_to_values(Nu_fun, disc))
                rv = Nu_v - _np.asarray(f_vals)
                for i, (bc, bc_val) in enumerate(zip(bcs, bc_vals)):
                    bc_row = _np.asarray(bc.matrix(disc))
                    rv[sz - n_bc + i] = float(bc_row @ uv) - float(bc_val)
                return rv, Nu_v, ufun

            # Work in materialized numpy per iteration: chaining lazy JAX
            # graphs across Newton iterations made the eventual float()
            # materialization at n~1024 segfault XLA's CPU backend.
            u0_np = _np.asarray(u_vals, dtype=_np.float64)

            def _newton_run(damped):
                """Run Newton from u0; return (u, r, Nu, ufun, converged)."""
                u_np = u0_np.copy()
                r_np, Nu_v, ufun = _residual(u_np)
                r_norm = float(_np.max(_np.abs(r_np)))
                converged = False
                for _it in range(max_iter):
                    J_mat = self._jacobian_matrix(
                        disc, x_fun, ufun, jnp.asarray(Nu_v)
                    )
                    J_np = _np.array(J_mat)  # copy: jax buffers read-only
                    for i, bc in enumerate(bcs):
                        J_np[sz - n_bc + i, :] = _np.asarray(bc.matrix(disc))
                    try:
                        delta = _np.linalg.solve(J_np, -r_np)
                    except _np.linalg.LinAlgError:
                        break
                    if not _np.all(_np.isfinite(delta)):
                        break

                    if damped:
                        # Monotone backtracking on the residual norm
                        # (MATLAB solvebvpNonlinear damps for global
                        # convergence on stiff problems).
                        lam_damp = 1.0
                        for _ls in range(8):
                            u_try = u_np + lam_damp * delta
                            r_try, Nu_try, ufun_try = _residual(u_try)
                            r_try_norm = float(_np.max(_np.abs(r_try)))
                            if _np.isfinite(r_try_norm) and (
                                r_try_norm <= (1.0 - 0.25 * lam_damp) * r_norm
                                or r_try_norm < newton_tol
                            ):
                                break
                            lam_damp *= 0.5
                    else:
                        lam_damp = 1.0
                        u_try = u_np + delta
                        r_try, Nu_try, ufun_try = _residual(u_try)
                        r_try_norm = float(_np.max(_np.abs(r_try)))

                    if not _np.isfinite(r_try_norm):
                        break
                    u_np, r_np, Nu_v, ufun = u_try, r_try, Nu_try, ufun_try
                    r_norm = r_try_norm

                    step_norm = float(_np.max(_np.abs(lam_damp * delta)))
                    u_scale = max(1.0, float(_np.max(_np.abs(u_np))))
                    if step_norm < newton_tol * u_scale:
                        converged = True
                        break
                return u_np, ufun, converged

            # Plain Newton first (the fast path, and what stiff-but-benign
            # problems like the van der Pol IVP-as-BVP need); fall back to
            # a damped run from the same start only if it diverges.
            u_np, u_fun, newton_converged = _newton_run(damped=False)
            if not newton_converged and not _np.all(
                _np.isfinite(u_np)
            ) or (not newton_converged and float(
                _np.max(_np.abs(u_np))
            ) > 1e8):
                u_np, u_fun, newton_converged = _newton_run(damped=True)

            u_vals = jnp.asarray(u_np, dtype=jnp.float64)

            finite = bool(jnp.isfinite(u_vals).all())
            if finite:
                u_fun = Chebfun.from_values(u_vals, dom)
                tech = u_fun.funs[0].tech
                resolved, _cut = Chebtech2.happiness_check(
                    tech.coeffs, tech.values
                )
            else:
                resolved = False

            if newton_converged and resolved:
                return u_fun

            if fixed_size:
                if not newton_converged:
                    warnings.warn(
                        f"Chebop.solve (Newton): did not converge at fixed "
                        f"n={sz} (last correction {correction_norm:.2e}).",
                        stacklevel=3,
                    )
                return u_fun

            if 2 * sz > n_max:
                warnings.warn(
                    "Chebop.solve (Newton): solution not resolved at "
                    f"n_max={n_max} (converged={newton_converged}, "
                    f"resolved={resolved}). Returning best approximation.",
                    stacklevel=3,
                )
                return u_fun if finite else Chebfun.from_values(
                    jnp.zeros(sz, dtype=jnp.float64), dom
                )

            # Refine and warm-start (drop a diverged iterate).
            u_fun_prev = u_fun if finite else None
            sz = 2 * sz

    def _jacobian_matrix(self, disc, x_fun, u_fun, Nu_vals):
        """Compute the Jacobian of self.op at u.

        Tries to use ADChebfun symbolic linearization first (exact, faster).
        Falls back to finite differences if symbolic linearization fails.
        """
        # A deflated operator G(u) = M(u; r) * N(u) needs the product-rule
        # Jacobian  J_G = M * J_N + N (x) dM/du.  The (dM/du) * N term is what
        # steers Newton away from the deflated roots and must not be dropped.
        # J_N is linearized exactly with ADChebfun and dM/du is obtained by
        # reverse-mode autodiff of the scalar deflation factor, so the whole
        # Jacobian is exact (matching MATLAB, which builds it via AD).
        if self._deflation is not None:
            try:
                return self._jacobian_matrix_deflated(
                    disc, x_fun, u_fun, Nu_vals)
            except Exception:
                # If the underlying operator cannot be linearized symbolically,
                # fall back to a finite-difference Jacobian of the full
                # deflated residual (slower, but still captures both terms).
                return self._jacobian_matrix_fd(disc, x_fun, u_fun, Nu_vals)
        # Try symbolic linearization via ADChebfun
        try:
            return self._jacobian_matrix_ad(disc, u_fun)
        except Exception:
            pass
        # Fall back to finite differences
        return self._jacobian_matrix_fd(disc, x_fun, u_fun, Nu_vals)

    def _jacobian_matrix_ad(self, disc, u_fun):
        """Compute Jacobian using ADChebfun symbolic differentiation (exact)."""
        from chebfunjax.autodiff.adchebfun import linearize_op
        J_op = linearize_op(self.op, u_fun, domain=disc.domain)
        return J_op.matrix(disc)

    def _jacobian_matrix_deflated(self, disc, x_fun, u_fun, Nu_vals):
        """Exact collocation Jacobian of a deflated operator ``M(u; r) N(u)``.

        Uses the product rule ``J_G = M J_N + N (x) dM/du`` where

        * ``J_N`` is the exact Jacobian of the *undeflated* operator ``N``,
          obtained by ADChebfun symbolic linearization;
        * ``M`` is the scalar deflation factor at ``u``;
        * ``dM/du`` is the gradient of that scalar with respect to the
          collocation values of ``u``, obtained by reverse-mode autodiff;
        * ``N`` is the vector of undeflated residual values at the collocation
          points (recovered as ``Nu_vals / M`` — ``Nu_vals`` already carries
          the factor ``M`` from the residual assembly).

        The outer-product term is what drives Newton away from the deflated
        roots; near a *new* root ``N`` vanishes and the Jacobian reduces to
        ``M J_N``, restoring quadratic convergence (mirroring MATLAB, which
        forms the same derivative through ADchebfun).

        Provenance
        ----------
        MATLAB source : @chebop/deflate.m, @chebmatrix/deflationFun.m,
            @chebop/linearize.m
        Chebfun commit: 7574c77
        """
        import jax
        import numpy as _np

        from chebfunjax.autodiff.adchebfun import linearize_op
        from chebfunjax.chebfun1d.chebfun import Chebfun

        orig_op, roots, p, alp, norm_type = self._deflation

        # Exact Jacobian of the undeflated operator N at u.
        JN = _np.asarray(
            linearize_op(orig_op, u_fun, domain=disc.domain).matrix(disc)
        )

        dom = Domain(disc.domain)
        u_vals = jnp.asarray(u_fun.funs[0].values, dtype=jnp.float64)
        # Sample each deflated root on the collocation nodes so that
        # ``u - r_k`` is the interpolant of the value difference (same length).
        a, b = disc.domain
        t_ref = chebpts(disc.n, kind=2)
        node_x = 0.5 * (b - a) * t_ref + 0.5 * (a + b)
        roots_vals = [jnp.asarray(r_k(node_x), dtype=jnp.float64)
                      for r_k in roots]

        def _M_of(uv):
            prod = jnp.asarray(1.0, dtype=jnp.float64)
            for rv in roots_vals:
                g = Chebfun.from_values(uv - rv, dom)
                s_k = jnp.real(g.inner(g))
                if norm_type != "L2":  # H1
                    gd = g.diff()
                    s_k = s_k + jnp.real(gd.inner(gd))
                prod = prod * s_k
            norm_fun = prod ** (p / 2.0)
            return 1.0 / norm_fun + alp

        M = float(_M_of(u_vals))
        dM = _np.asarray(jax.grad(_M_of)(u_vals), dtype=_np.float64)

        # Undeflated residual values N = (M N) / M.
        N_vals = _np.asarray(Nu_vals, dtype=_np.float64) / M
        return M * JN + _np.outer(N_vals, dM)

    def _jacobian_matrix_fd(self, disc, x_fun, u_fun, Nu_vals):
        """Compute the Jacobian of self.op at u by forward finite differences."""
        import numpy as _np

        from chebfunjax.chebfun1d.chebfun import Chebfun

        n = disc.n
        dom = Domain(disc.domain)
        h = max(1e-6, 1e-6 * float(jnp.max(jnp.abs(u_fun.funs[0].values))))

        # Jacobian columns -- materialised eagerly column-by-column into a
        # numpy array so the finite-difference probe never accumulates a
        # single O(n)-sized ``jnp.stack`` graph (which blows up the XLA
        # compile at large n; see _linearize_op).
        base_vals = u_fun.funs[0].values
        Nu_np = _np.asarray(Nu_vals, dtype=_np.float64)
        J = _np.empty((n, n), dtype=_np.float64)
        for j in range(n):
            e_j = jnp.zeros(n, dtype=jnp.float64).at[j].set(h)
            u_pert = Chebfun.from_values(base_vals + e_j, dom)
            Nu_pert = self._apply_op(x_fun, u_pert)
            Nu_pert_vals = _np.asarray(_chebfun_to_values(Nu_pert, disc),
                                       dtype=_np.float64)
            J[:, j] = (Nu_pert_vals - Nu_np) / h

        return jnp.asarray(J, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # BC parsing
    # ------------------------------------------------------------------

    def _parse_bcs(self) -> tuple[list[FunctionalBlock], list[float]]:
        """Parse lbc and rbc into FunctionalBlock objects and values.

        Supported forms:
        - scalar ``c``         → ``u(endpoint) = c``  (simple Dirichlet)
        - callable ``g(u)``    → linearized at u=0 (Neumann etc.)
        - None                 → no BC at that end

        Returns
        -------
        bcs : list[FunctionalBlock]
        bc_vals : list[float]
        """
        bcs: list[FunctionalBlock] = []
        bc_vals: list[float] = []

        a, b = self.domain

        # Left BC
        if self._lbc_raw is not None:
            lbc_blocks, lbc_vals = self._bc_to_functionals(
                self._lbc_raw, endpoint=a
            )
            bcs.extend(lbc_blocks)
            bc_vals.extend(lbc_vals)

        # Right BC
        if self._rbc_raw is not None:
            rbc_blocks, rbc_vals = self._bc_to_functionals(
                self._rbc_raw, endpoint=b
            )
            bcs.extend(rbc_blocks)
            bc_vals.extend(rbc_vals)

        return bcs, bc_vals

    def _bc_to_functionals(
        self, bc_spec, endpoint: float
    ) -> tuple[list[FunctionalBlock], list[float]]:
        """Convert a BC specification to a list of (FunctionalBlock, value).

        Parameters
        ----------
        bc_spec : scalar, list, or callable
            BC specification.
        endpoint : float
            The domain endpoint (a for left, b for right).

        Returns
        -------
        blocks : list[FunctionalBlock]
        values : list[float]
        """
        domain = self.domain

        # Scalar: simple Dirichlet u(endpoint) = bc_spec
        if isinstance(bc_spec, (int, float)):
            fb = eval_at(endpoint, domain=domain)
            return [fb], [float(bc_spec)]

        # List or tuple: one condition per entry
        # e.g. [0, 1] means u(a) = 0, u'(a) = 1
        if isinstance(bc_spec, (list, tuple)):
            blocks = []
            vals = []
            for i, val in enumerate(bc_spec):
                # i-th entry: value of the i-th derivative
                if i == 0:
                    fb = eval_at(endpoint, domain=domain)
                else:
                    # Build eval_at ∘ D^i as a FunctionalBlock
                    fb = _derivative_eval_at(endpoint, domain=domain, order=i)
                blocks.append(fb)
                vals.append(float(val))
            return blocks, vals

        # callable g(u) — linearize at u=0
        # Evaluate g on identity and zero to extract the linear row
        if callable(bc_spec):
            return self._callable_bc_to_functional(bc_spec, endpoint)

        raise TypeError(
            f"Chebop: unsupported BC type {type(bc_spec).__name__}. "
            f"Use a scalar, list of scalars, or a callable g(u)."
        )

    def _callable_bc_to_functional(
        self, bc_fn, endpoint: float
    ) -> tuple[list[FunctionalBlock], list[float]]:
        """Linearize a callable BC g(u) into a FunctionalBlock.

        Computes the Jacobian row of g evaluated at u=0 by finite differences.

        Returns
        -------
        blocks : list[FunctionalBlock] (length 1)
        values : list[float] (length 1 — the g(0) value negated)
        """
        from chebfunjax.chebfun1d.chebfun import Chebfun

        domain = self.domain
        a, b = domain
        Domain(domain)

        # Evaluate g at zero
        zero_vals = jnp.zeros(8, dtype=jnp.float64)
        u0 = Chebfun.from_values(zero_vals, Domain(domain))

        # The BC value is -g(u0) (we enforce g(u) = 0, so rhs = -g(0))
        try:
            g0 = bc_fn(u0)
            # Evaluate at endpoint
            g0_val = float(_safe_eval(g0, jnp.array(endpoint, dtype=jnp.float64)))
        except Exception:
            g0_val = 0.0

        # Build a generic FunctionalBlock that numerically evaluates the BC row
        ep = endpoint

        def _fn(disc: ChebColloc2Disc) -> jnp.ndarray:
            n = disc.n
            dom_inner = Domain(disc.domain)

            x0 = Chebfun.from_values(jnp.zeros(n, dtype=jnp.float64), dom_inner)
            try:
                g_at_zero = bc_fn(x0)
                g0_pt = float(_safe_eval(g_at_zero, jnp.array(ep, dtype=jnp.float64)))
            except Exception:
                g0_pt = 0.0

            def _probe_row(h: float) -> jnp.ndarray:
                r = jnp.zeros(n, dtype=jnp.float64)
                for j in range(n):
                    e_j = jnp.zeros(n, dtype=jnp.float64).at[j].set(h)
                    u_pert = Chebfun.from_values(e_j, dom_inner)
                    try:
                        g_pert = bc_fn(u_pert)
                        g_pert_pt = float(_safe_eval(
                            g_pert, jnp.array(ep, dtype=jnp.float64)))
                    except Exception:
                        g_pert_pt = g0_pt
                    r = r.at[j].set((g_pert_pt - g0_pt) / h)
                return r

            # Probe with h=1 first: for a LINEAR functional (the common
            # Neumann/Robin case) this is exact — no finite-difference
            # cancellation, which at h=1e-6 leaves ~eps/h ≈ 1e-10 noise
            # per entry and (via near-resonant modes) 1e-9-level BVP
            # errors where MATLAB's AD-linearized BCs give 1e-14.
            row = _probe_row(1.0)
            # Linearity check on a combined direction: g(sum e_j) must
            # equal g0 + sum(row).  Nonlinear BCs fail this and fall
            # back to a small-h Jacobian at u=0.
            try:
                v = Chebfun.from_values(
                    jnp.ones(n, dtype=jnp.float64), dom_inner)
                g_v = float(_safe_eval(
                    bc_fn(v), jnp.array(ep, dtype=jnp.float64)))
                scale = max(1.0, float(jnp.max(jnp.abs(row))), abs(g0_pt))
                is_linear = abs(g_v - g0_pt - float(jnp.sum(row))) \
                    <= 1e-8 * scale * n
            except Exception:
                is_linear = False
            if not is_linear:
                row = _probe_row(1e-6)
            return row

        fb = FunctionalBlock(_fn, domain=domain)
        return [fb], [-g0_val]


# ===========================================================================
# Module-level private helpers
# ===========================================================================


def _derivative_eval_at(
    x: float,
    domain: tuple[float, float],
    order: int,
) -> FunctionalBlock:
    """Evaluation functional for the *order*-th derivative at a point.

    Returns a FunctionalBlock whose row vector ``r`` satisfies::

        r @ u_vals ≈ u^(order)(x)

    Parameters
    ----------
    x : float
        Physical evaluation point.
    domain : (float, float)
        Physical domain.
    order : int
        Derivative order (0 = function value, 1 = first derivative, etc.).

    Returns
    -------
    FunctionalBlock
    """
    from chebfunjax.operators.blocks import D as diff_op
    from chebfunjax.operators.blocks import eval_at as eval_fb

    dom = domain

    def _fn(disc: ChebColloc2Disc) -> jnp.ndarray:
        # Differentiation matrix of the given order
        D_mat = diff_op(dom, order).matrix(disc)       # (n, n)
        # Evaluation row at x
        E_row = eval_fb(x, dom).matrix(disc)            # (n,)
        # Composed row: E @ D
        return E_row @ D_mat                            # (n,)

    return FunctionalBlock(_fn, domain=domain)


def _make_rhs_callable(f):
    """Convert scalar / callable / Chebfun to a callable of physical pts."""
    if callable(f) and not isinstance(f, (int, float)):
        # Check if it's a Chebfun
        if hasattr(f, "funs"):
            return lambda x: f(jnp.asarray(x, dtype=jnp.float64))
        return f
    val = float(f)
    return lambda x: jnp.full(x.shape, val, dtype=jnp.float64)


def _safe_eval(f, x):
    """Safely evaluate f at x; return 0 if f is not callable."""
    if hasattr(f, "__call__"):
        return f(x)
    return jnp.asarray(float(f), dtype=jnp.float64)


def _chebfun_to_values(f, disc: ChebColloc2Disc) -> jnp.ndarray:
    """Evaluate a Chebfun at the disc's collocation points.

    Parameters
    ----------
    f : Chebfun or scalar
    disc : ChebColloc2Disc

    Returns
    -------
    vals : jnp.ndarray, shape (n,)
    """
    if isinstance(f, (int, float)):
        return jnp.full(disc.n, float(f), dtype=jnp.float64)
    if isinstance(f, complex):
        return jnp.full(disc.n, f, dtype=jnp.complex128)
    # Compute physical Chebyshev-2 points from the disc descriptor
    a, b = disc.domain
    t_ref = chebpts(disc.n, kind=2)
    x_pts = 0.5 * (b - a) * t_ref + 0.5 * (a + b)
    vals = jnp.asarray(f(x_pts))
    # Preserve complex operator outputs (complex-shifted problems).
    if not jnp.iscomplexobj(vals):
        vals = vals.astype(jnp.float64)
    return vals


# ============================================================================
# Fourier-collocation helpers for periodic BVPs (Claude Opus 4.8, task #24)
# ============================================================================


def _fourier_diffmat(n: int, length: float, order: int):
    """Order-``order`` Fourier differentiation matrix on ``n`` equispaced
    periodic points over an interval of length ``length``.

    Built spectrally: ``D = Re( IDFT @ diag((i*k*2pi/L)^order) @ DFT )``,
    with the Nyquist mode zeroed for odd orders (even n) so the matrix
    stays real.
    """
    import numpy as _np

    k = _np.fft.fftfreq(n, d=1.0 / n)          # integer wavenumbers
    mult = (1j * k * (2.0 * _np.pi / length)) ** order
    if n % 2 == 0 and order % 2 == 1:
        mult[n // 2] = 0.0                      # kill Nyquist for odd order
    j = _np.arange(n)
    dft = _np.exp(-2j * _np.pi * _np.outer(j, j) / n)
    idft = _np.exp(2j * _np.pi * _np.outer(j, j) / n) / n
    return _np.real(idft @ (mult[:, None] * dft))


class _TrigVals:
    """Value-space proxy on the equispaced periodic grid.

    Carries the grid VALUES of an expression; ``diff`` multiplies by the
    Fourier differentiation matrix and elementwise functions apply
    pointwise, so a NONLINEAR periodic operator can be evaluated on a
    candidate solution vector (used by the periodic Newton iteration).
    """

    def __init__(self, v, n, length):
        self.v = jnp.asarray(v, dtype=jnp.float64)
        self.n = n
        self.length = length

    def _w(self, v):
        return _TrigVals(v, self.n, self.length)

    def diff(self, k: int = 1):
        import numpy as _np
        D = _np.asarray(_fourier_diffmat(self.n, self.length, k))
        return self._w(jnp.asarray(D) @ self.v)

    def cumsum(self):
        raise TypeError("cumsum unsupported on the periodic grid proxy")

    def sum(self):
        return float(jnp.sum(self.v)) * (self.length / self.n)

    def __call__(self, *a, **k):
        raise TypeError("evaluation unsupported on the grid proxy")

    def _c(self, o):
        return o.v if isinstance(o, _TrigVals) else o

    def __add__(self, o):
        return self._w(self.v + self._c(o))

    __radd__ = __add__

    def __sub__(self, o):
        return self._w(self.v - self._c(o))

    def __rsub__(self, o):
        return self._w(self._c(o) - self.v)

    def __mul__(self, o):
        return self._w(self.v * self._c(o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self._w(self.v / self._c(o))

    def __rtruediv__(self, o):
        return self._w(self._c(o) / self.v)

    def __pow__(self, p):
        return self._w(self.v ** p)

    def __neg__(self):
        return self._w(-self.v)

    def __abs__(self):
        return self._w(jnp.abs(self.v))

    def __getattr__(self, name):
        fn = getattr(jnp, name, None)
        if fn is None:
            raise AttributeError(name)

        def _apply():
            return self._w(fn(self.v))
        return _apply


class _FourierProxy:
    """Linear-operator proxy for Fourier collocation.

    Wraps an (n, n) matrix ``mat`` describing the linear action on the
    grid values of ``u``.  Supports ``diff``, addition/subtraction, and
    multiplication by scalars or grid-sampled variable coefficients, so
    that evaluating ``op(x_grid, proxy)`` assembles the operator matrix.
    """

    def __init__(self, n: int, length: float, mat, grid=None):
        self.n = n
        self.length = length
        self.mat = mat
        self.grid = grid

    def _wrap(self, mat):
        return _FourierProxy(self.n, self.length, mat, grid=self.grid)

    def diff(self, order: int = 1):
        import numpy as _np
        d = _fourier_diffmat(self.n, self.length, order)
        return self._wrap(_np.asarray(d) @ self.mat)

    def __add__(self, other):
        if isinstance(other, _FourierProxy):
            return self._wrap(self.mat + other.mat)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, _FourierProxy):
            return self._wrap(self.mat - other.mat)
        return NotImplemented

    def __neg__(self):
        return self._wrap(-self.mat)

    def _scale(self, other):
        import numpy as _np
        try:
            arr = _np.asarray(other, dtype=float)
        except (TypeError, ValueError):
            # Variable coefficient given as a chebfun (MATLAB
            # ``a1.*diff(u)``): sample it on the collocation grid.
            if callable(other) and self.grid is not None:
                arr = _np.asarray(other(jnp.asarray(self.grid)),
                                  dtype=float)
            else:
                raise
        if arr.ndim == 0:
            return self._wrap(float(arr) * self.mat)
        return self._wrap(arr[:, None] * self.mat)   # diag(coeff) @ mat

    def __mul__(self, other):
        return self._scale(other)

    def __rmul__(self, other):
        return self._scale(other)


def _fourier_interp(nodes, vals, t, a: float, length: float):
    """Evaluate the trigonometric interpolant of periodic samples ``vals``
    at nodes ``nodes`` (equispaced) at points ``t``.  Returns real values.
    """
    n = nodes.shape[0]
    k = jnp.fft.fftfreq(n, d=1.0 / n)
    coeffs = jnp.fft.fft(vals) / n
    if n % 2 == 0:
        # split Nyquist so the real interpolant is symmetric
        coeffs = coeffs.at[n // 2].multiply(0.5)
    theta = 2.0 * jnp.pi * (jnp.asarray(t) - a) / length
    # sum_k coeffs[k] e^{i k theta}  (+ conjugate Nyquist term)
    phase = jnp.exp(1j * theta[..., None] * k[None, :])
    out = jnp.real(phase @ coeffs)
    if n % 2 == 0:
        out = out + jnp.real(
            jnp.exp(-1j * theta * (n // 2)) * coeffs[n // 2])
    return out


# ============================================================================
# IVP time-marching (all BCs at one endpoint) — task #24, Claude Opus 4.8
# ============================================================================


class _IVPProxy:
    """Proxy that behaves as u (= its 0-th derivative) in arithmetic and
    returns the j-th derivative from a supplied tower on ``diff(j)``.
    Used to extract the ODE right-hand side from a Chebop operator."""

    def __init__(self, tower):
        self._d = tower                       # [u, u', ..., u^(k-1), probe]

    @property
    def _v(self):
        return self._d[0]

    def diff(self, j: int = 1):
        return self._d[j]

    def __add__(self, o):
        return self._v + (o._v if isinstance(o, _IVPProxy) else o)

    __radd__ = __add__

    def __sub__(self, o):
        return self._v - (o._v if isinstance(o, _IVPProxy) else o)

    def __rsub__(self, o):
        return (o._v if isinstance(o, _IVPProxy) else o) - self._v

    def __mul__(self, o):
        return self._v * (o._v if isinstance(o, _IVPProxy) else o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self._v / (o._v if isinstance(o, _IVPProxy) else o)

    def __rtruediv__(self, o):
        return (o._v if isinstance(o, _IVPProxy) else o) / self._v

    def __pow__(self, o):
        return self._v ** (o._v if isinstance(o, _IVPProxy) else o)

    def __neg__(self):
        return -self._v


class _OrderSniffer:
    """Records the highest derivative order requested from an operator."""

    def __init__(self):
        self.order = 0

    def diff(self, j: int = 1):
        self.order = max(self.order, j)
        return self

    def __add__(self, o):
        return self

    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __add__
    __truediv__ = __rtruediv__ = __pow__ = __rpow__ = __add__

    def __neg__(self):
        return self


class _EqOrderSniffer:
    """Deferred per-equation order sniffer.

    Expressions carry the set of ``(var, order)`` pairs they depend on;
    reading the pairs of each OUTPUT of a system operator gives that
    equation's differential order in every variable (the shared-list
    :class:`_SysOrderSniffer` cannot attribute orders to outputs since
    it records at ``diff``-call time).
    """

    def __init__(self, pairs=frozenset()):
        object.__setattr__(self, "pairs", frozenset(pairs))

    def diff(self, k: int = 1):
        return _EqOrderSniffer({(v, o + int(k)) for v, o in self.pairs})

    def _comb(self, o):
        pr = set(self.pairs)
        if isinstance(o, _EqOrderSniffer):
            pr |= o.pairs
        return _EqOrderSniffer(pr)

    def __add__(self, o):
        return self._comb(o)

    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __add__
    __truediv__ = __rtruediv__ = __pow__ = __rpow__ = __add__

    def __neg__(self):
        return self

    def __abs__(self):
        return self

    def sum(self):
        return self

    def cumsum(self, *a):
        return self

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, name):
        def _fn(*a, **k):
            return self
        return _fn


class _SysOrderSniffer:
    """Per-variable differential-order sniffer for system operators.

    Each unknown variable is given a sniffer carrying its index; the
    highest ``diff`` order applied to it is recorded in a shared mutable
    ``orders`` list.  Arithmetic propagates the set of variable indices an
    expression depends on, so ``diff`` applied to a compound expression is
    recorded against every contributing variable.  Elementwise callables
    (``sin``, ``cos``, ``exp``, ...) are absorbed via ``__getattr__`` and
    leave the recorded order unchanged.

    Used to decide, for a piecewise-collocation system, how many continuity
    conditions (derivatives 0..k-1) each variable needs at every interior
    breakpoint — mirroring the diffOrder bookkeeping of MATLAB @chebop.

    Provenance
    ----------
    MATLAB source : @chebop/getDiffOrder.m, @linop/diffOrder
    Chebfun commit: 7574c77
    """

    def __init__(self, orders, idx=()):
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "idx", frozenset(idx))

    def diff(self, k: int = 1):
        for i in self.idx:
            self.orders[i] = max(self.orders[i], int(k))
        return self

    def _combine(self, o):
        idx = set(self.idx)
        if isinstance(o, _SysOrderSniffer):
            idx |= o.idx
        return _SysOrderSniffer(self.orders, idx)

    def __add__(self, o):
        return self._combine(o)

    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __add__
    __truediv__ = __rtruediv__ = __pow__ = __rpow__ = __matmul__ = __add__

    def __neg__(self):
        return self

    def __abs__(self):
        return self

    def __getattr__(self, name):
        if name in ("orders", "idx"):
            raise AttributeError(name)

        def _method(*a, **k):
            return self

        return _method


# ===========================================================================
# linop introspection — system linearization producing typed blocks
# ===========================================================================


def _chebfun_ones(domain: tuple[float, float]):
    """Return the constant Chebfun ``f(x) = 1`` on domain."""
    from chebfunjax.chebfun1d.chebfun import Chebfun
    return Chebfun.from_values(jnp.ones(2, dtype=jnp.float64), Domain(domain))


def _I_block(domain: tuple[float, float]) -> OperatorBlock:
    """Identity OperatorBlock on domain (the self-derivative of an unknown)."""
    from chebfunjax.operators.blocks import I as _I
    return _I(domain)


def _block_to_coefficient(blk, one_fun, domain: tuple[float, float]):
    """Collapse a pure-multiplication Jacobian block to its coefficient Chebfun.

    A parameter unknown appears only multiplicatively, so its block is an
    order-0 operator ``diag(c)`` (a composition of ``I`` / ``diag`` / scalar
    scalings); applying it to the constant ``1`` recovers the coefficient
    ``c(x)`` as a Chebfun.  ``None`` (absent unknown) yields the zero Chebfun.
    """
    if blk is None:
        return _chebfun_zeros(domain)
    try:
        return blk.apply(one_fun)
    except Exception:
        # Fall back to reading the coefficient off the collocation diagonal.
        import numpy as _np
        n = 9
        mat = _np.asarray(blk.matrix(ChebColloc2Disc(n, domain)))
        return _chebfun_from_values(_np.diag(mat), domain)


def _zero_operator(domain: tuple[float, float]) -> OperatorBlock:
    """Zero operator block (an unknown that is absent from an equation).

    Represents the ``0`` Frechet-derivative entry of a block Jacobian; its
    matrix is the ``n x n`` zero matrix and its function-space action maps any
    function to the zero function.  Differential order 0 (multiplicative).
    """

    def _fn(disc: ChebColloc2Disc):
        return jnp.zeros((disc.n, disc.n), dtype=jnp.float64)

    return OperatorBlock(
        _fn, order=0, domain=domain,
        apply_fn=lambda u: _chebfun_zeros(domain))


class _LinopVar:
    """System-aware linearization variable for :meth:`Chebop.linop`.

    Carries only the Jacobian *row* of an expression: ``jac[k]`` is the
    :class:`~chebfunjax.operators.blocks.OperatorBlock` giving the Frechet
    derivative of the expression with respect to unknown ``k`` (``None`` marks
    a structurally-zero entry).  The primal value is not tracked: for a linear
    operator the Jacobian blocks are constant, so linearizing around the zero
    function is exact and the affine part (constants / ``x``-only terms) simply
    leaves the Jacobian unchanged.

    Arithmetic mirrors the Frechet-derivative rules used by
    :class:`~chebfunjax.autodiff.adchebfun.ADChebfun`, generalized to a row of
    per-variable blocks.  Multiplying/dividing two unknowns, or applying a
    nonlinear elementwise function to one, raises -- such an operator is not a
    linop.

    Provenance
    ----------
    MATLAB source : @chebop/linearize.m, @linop/linop.m, @adchebfun
    Chebfun commit: 7574c77
    """

    __slots__ = ("jac", "domain")

    def __init__(self, jac, domain):
        self.jac = jac
        self.domain = domain

    # -- additive structure: block-wise combination of Jacobian rows --------

    def __add__(self, other):
        if isinstance(other, _LinopVar):
            return _LinopVar(
                [_op_add(a, b) for a, b in zip(self.jac, other.jac)],
                self.domain)
        # Adding a constant / x-only chebfun is affine: Jacobian unchanged.
        return self

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, _LinopVar):
            return _LinopVar(
                [_op_sub(a, b) for a, b in zip(self.jac, other.jac)],
                self.domain)
        return self

    def __rsub__(self, other):
        # (affine) - self  ->  Jacobian is negated.
        return self.__neg__()

    def __neg__(self):
        return _LinopVar(
            [None if a is None else -a for a in self.jac], self.domain)

    def __pos__(self):
        return self

    # -- scaling by a scalar or a chebfun coefficient -----------------------

    def _scale(self, other):
        if isinstance(other, _LinopVar):
            raise ValueError(
                "Chebop.linop: nonlinear term (product of two unknowns); "
                "the operator is not linear.")
        if isinstance(other, (int, float)):
            c = float(other)
            return _LinopVar(
                [None if a is None else c * a for a in self.jac], self.domain)
        # Chebfun (variable) coefficient: pre-compose with multiplication.
        from chebfunjax.operators.blocks import diag
        M = diag(other, self.domain)
        return _LinopVar(
            [None if a is None else M * a for a in self.jac], self.domain)

    def __mul__(self, other):
        return self._scale(other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self._scale(1.0 / float(other))
        if isinstance(other, _LinopVar):
            raise ValueError(
                "Chebop.linop: nonlinear term (division by an unknown); "
                "the operator is not linear.")
        return self._scale(1.0 / other)

    # -- differential / integral operators ----------------------------------

    def diff(self, k: int = 1):
        from chebfunjax.operators.blocks import D
        Dk = D(self.domain, order=int(k))
        return _LinopVar(
            [None if a is None else Dk * a for a in self.jac], self.domain)

    def cumsum(self):
        def _cumsum_fn(disc: ChebColloc2Disc):
            from chebfunjax.utils.diffmat import diffmat as _dm
            return jnp.linalg.pinv(_dm(disc.n, 1, domain=disc.domain))

        C = OperatorBlock(_cumsum_fn, order=-1, domain=self.domain)
        return _LinopVar(
            [None if a is None else C * a for a in self.jac], self.domain)

    sum = cumsum

    def __getattr__(self, name):
        if name in ("jac", "domain"):
            raise AttributeError(name)
        # Any elementwise nonlinear method (sin, cos, exp, ...) applied to an
        # unknown makes the operator nonlinear -> not a linop.
        def _method(*a, **k):
            raise ValueError(
                f"Chebop.linop: nonlinear operation '{name}' on an unknown; "
                "the operator is not linear.")

        return _method


def _normalize_deflation_roots(r):
    """Normalize a deflation-root argument to a plain list of Chebfun.

    Accepts a single Chebfun, a list/tuple of Chebfun, or any object exposing
    a ``blocks`` sequence (e.g. :class:`SystemSolution` or a ChebMatrix of
    solution columns, mirroring MATLAB's ``chebmatrix(r)`` promotion in
    @chebop/deflate).
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun
    if isinstance(r, Chebfun):
        return [r]
    if hasattr(r, "blocks"):
        return list(r.blocks)
    return list(r)


def _deflation_factor(u, roots, p, alp, norm_type):
    """Scalar deflation factor ``M(u; roots)`` at the current guess ``u``.

    Faithful port of the norm construction in @chebmatrix/deflationFun.m::

        normFun = prod_k ||u - r_k||^2            (L2)
        normFun = prod_k (||u-r_k||^2 + ||d(u-r_k)/dx||^2)   (H1)
        normFun = normFun^(p/2)
        M       = 1 / normFun + alp

    Returned as a Python float; the residual multiplies the (undeflated)
    operator output by this scalar.

    Provenance
    ----------
    MATLAB source : @chebmatrix/deflationFun.m
    Chebfun commit: 7574c77
    """
    norm_fun = 1.0
    for r_k in roots:
        ur = u - r_k
        if norm_type == "L2":
            s_k = float(ur.norm(2)) ** 2
        else:  # H1
            s_k = float(ur.norm(2)) ** 2 + float(ur.diff().norm(2)) ** 2
        norm_fun = norm_fun * s_k
    norm_fun = norm_fun ** (p / 2.0)
    return 1.0 / norm_fun + alp


def _make_deflated_op(orig_op, roots, p, alp, norm_type):
    """Build the deflated operator ``G(u) = M(u; roots) * N(u)``.

    Mirrors @chebop/deflate.m, which wraps ``N.op`` in ``deflationFun``.  The
    returned callable always has an ``(x, u)`` signature; it dispatches on the
    original operator's arity to support both ``@(u)`` and ``@(x, u)`` forms.
    """
    import inspect
    try:
        nargs = len(inspect.signature(orig_op).parameters)
    except (TypeError, ValueError):
        nargs = 2

    def deflated_op(x, u):
        nu = orig_op(u) if nargs == 1 else orig_op(x, u)
        factor = _deflation_factor(u, roots, p, alp, norm_type)
        return nu * factor

    return deflated_op


def deflate(N, r, p, alp, type="L2"):
    """Deflate known solutions ``r`` from a :class:`Chebop`.

    Free-function form of :meth:`Chebop.deflate`, matching MATLAB's
    ``deflate(N, r, p, alp)`` calling sequence.  See :meth:`Chebop.deflate`
    for the full description.

    Provenance
    ----------
    MATLAB source : @chebop/deflate.m
    Chebfun commit: 7574c77
    """
    return N.deflate(r, p, alp, type)


def _op_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _op_sub(a, b):
    if a is None:
        return None if b is None else -b
    if b is None:
        return a
    return a - b
