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
    return chebfun(lambda x: jnp.zeros_like(x), domain=Domain(domain), n=2)


def _chebfun_identity(domain: tuple[float, float]):
    """Return the identity Chebfun f(x) = x on domain."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: x, domain=Domain(domain), n=2)


def _eval_chebfun_at(u, x0: float) -> float:
    """Evaluate u at the physical point x0."""
    if isinstance(u, (int, float)):
        return float(u)
    arr = u(jnp.array(x0, dtype=jnp.float64))
    return float(arr)


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
        self._periodic = False
        #: Initial guess for nonlinear solves (Chebfun, callable, or None) —
        #: MATLAB's N.init. Previously assigning N.init was silently ignored.
        self.init = None
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
        elif isinstance(val, (int, float)) or callable(val):
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
        if len(self.domain) > 2 and not getattr(self, "_periodic", False):
            return self._solve_piecewise(f, n=n, max_iter=max_iter)

        # Systems of ODEs: op signature (x, u, v, ...) with >= 2
        # unknowns dispatches to block collocation (linear) or Newton
        # on top of it (nonlinear); periodic systems use a Fourier
        # (equispaced/trig) discretization with no BC rows.
        if self._n_vars() >= 2:
            if getattr(self, "_periodic", False):
                return self._solve_periodic_system(
                    f, n=n, max_iter=max_iter)
            # First-order explicit IVP systems (all BCs at one end)
            # time-march like MATLAB routes to ode113.
            if (self._lbc_raw is None) != (self._rbc_raw is None):
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
            out = self.op(x_fun, *us)
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
            out = self.op(x_fun, *us)
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
            out = self.op(x_fun, *us)
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
            out = self.op(x_fun, *us)
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

        # verify first-order explicit form: residual affine in y'
        # with unit coefficient => R(t, y, y') = y' + op_at-part
        def rhs(t, y):
            return -op_at(t, y)

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

        # sample onto a Chebyshev grid and wrap each component
        nn = 256
        kk = _np.arange(nn)
        xg = _np.cos(_np.pi * kk / (nn - 1))[::-1]
        xp = a + (b - a) * (xg + 1.0) / 2.0
        Y = sol.sol(xp)
        return SystemSolution([
            _chebfun_from_values(jnp.asarray(Y[i]), self.domain)
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

        def ev(us):
            out = self.op(x_fun, *us)
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

        def residual(U):
            us = to_funs(U)
            out = self.op(x_fun, *us)
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
            return R

        U = _np.zeros(m * n)
        if self.init is not None:
            init = self.init if isinstance(self.init, (list, tuple)) \
                else [self.init] * m
            U = _np.concatenate([
                _np.asarray(gi(jnp.asarray(xp))) for gi in init])
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
            step = _np.linalg.solve(J, R)
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
            out = self.op(x_fun, *us)
            if not isinstance(out, (list, tuple)):
                out = [out]
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
                raise ValueError(
                    "system BCs must be callables of (u1, ..., um)")

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

        rows_l, vals_l = bc_rows(self._lbc_raw, a)
        rows_r, vals_r = bc_rows(self._rbc_raw, b)

        # Replace boundary-point rows of successive equations
        for i, (row, val) in enumerate(zip(rows_l, vals_l)):
            ridx = (i % m) * n           # x = a row of equation i%m
            A[ridx, :] = row
            bvec[ridx] = val
        for i, (row, val) in enumerate(zip(rows_r, vals_r)):
            ridx = (i % m) * n + n - 1   # x = b row
            A[ridx, :] = row
            bvec[ridx] = val

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

    def _solve_piecewise(self, f=0.0, n: int | None = None,
                         max_iter: int = 40):
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
        n_l = len(bc_res(self._lbc_raw, us_zero, bps[0]))
        n_r = len(bc_res(self._rbc_raw, us_zero, bps[-1]))
        l_rows = [row(i % m, 0, 0) for i in range(n_l)]
        r_rows = [row(i % m, P - 1, nn - 1) for i in range(n_r)]

        # Continuity: derivatives 0..k_j-1 of each variable j at every
        # interior break.  Each condition takes one of the two collocation
        # rows that coincide at the break (left piece right end / right piece
        # left end), distributed across equations for good conditioning.
        conds = [(j, d) for j in range(m) for d in range(orders[j])]
        c_rows = []
        for p in range(1, P):
            for c, (j, d) in enumerate(conds):
                eq = c % m
                rr = (row(eq, p - 1, nn - 1) if (c // m) % 2 == 0
                      else row(eq, p, 0))
                c_rows.append((rr, j, d, bps[p], p - 1, p))

        # RHS values laid out per equation / piece.
        f_vals = _np.zeros(m * Pn)
        for eq in range(m):
            fi = f[eq] if isinstance(f, (list, tuple)) and eq < len(f) else (
                f if not isinstance(f, (list, tuple)) else 0.0)
            for p in range(P):
                sl = slice(eq * Pn + p * nn, eq * Pn + (p + 1) * nn)
                f_vals[sl] = (float(fi) if isinstance(fi, (int, float))
                              else _np.asarray(fi(jnp.asarray(xps[p]))))

        def residual(U):
            us = to_funs(U)
            out = apply_op(us)
            R = _np.zeros(m * Pn)
            for eq in range(m):
                o = out[eq]
                for p in range(P):
                    sl = slice(eq * Pn + p * nn, eq * Pn + (p + 1) * nn)
                    R[sl] = (float(o) if isinstance(o, (int, float))
                             else _np.asarray(o.funs[p](jnp.asarray(xps[p]))))
            R = R - f_vals
            for i, v in enumerate(bc_res(self._lbc_raw, us, bps[0])):
                R[l_rows[i]] = v
            for i, v in enumerate(bc_res(self._rbc_raw, us, bps[-1])):
                R[r_rows[i]] = v
            for (rr, j, d, bp, p_l, p_r) in c_rows:
                xb = jnp.asarray(bp)
                left = us[j].funs[p_l].diff(d)(xb) if d > 0 \
                    else us[j].funs[p_l](xb)
                right = us[j].funs[p_r].diff(d)(xb) if d > 0 \
                    else us[j].funs[p_r](xb)
                R[rr] = float(left) - float(right)
            return R

        # Initial iterate: user's N.init, else zero.
        U = _np.zeros(m * Pn)
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
        R = residual(U)
        Rn = R
        for _it in range(max_iter):
            nrm = _np.max(_np.abs(R))
            if nrm < 1e-12:
                break
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
        if self._n_vars() >= 2:
            return self._eigs_system(k=k, n=n or n_default)
        linop = self._build_linop(value_shift=0.0)
        return linop.eigs(n=n, k=k, n_default=n_default, sigma=sigma,
                          return_eigenfunctions=return_eigenfunctions)

    def __truediv__(self, f):
        """``N \\ f`` syntax — solve N[u] = f."""
        return self.solve(f)

    def __repr__(self) -> str:
        a, b = self.domain
        return (
            f"Chebop(domain=({a}, {b}), lbc={self._lbc_raw!r}, "
            f"rbc={self._rbc_raw!r})"
        )

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
            proxy = _FourierProxy(N, L, _np.eye(N))
            out = self._apply_op(jnp.asarray(x), proxy)
            if not isinstance(out, _FourierProxy):
                raise TypeError(
                    "Chebop._solve_periodic: operator is not linear in u "
                    "(nonlinear periodic problems are not supported).")
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
        _ = self.op(x, sniff) if nargs == 2 else self.op(sniff)
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
            return float(_np.asarray(L(jnp.asarray(x), _IVPProxy(tower))))

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

        Provenance
        ----------
        MATLAB source : @chebop/feval.m, @chebop/mtimes.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import Chebfun
        x_fun = Chebfun.identity(Domain(self.domain))
        if isinstance(u, (list, tuple)) or (
                self._n_vars() >= 2 and hasattr(u, "__getitem__")
                and not isinstance(u, Chebfun)):
            out = self.op(x_fun, *list(u))
            return SystemSolution(list(out)) \
                if isinstance(out, (list, tuple)) else out
        return self._apply_op(x_fun, u)

    def _apply_op(self, x_fun, u_fun):
        """Evaluate self.op(x_fun, u_fun) or self.op(u_fun)."""
        import inspect
        try:
            n = len(inspect.signature(self.op).parameters)
        except (TypeError, ValueError):
            n = 2  # default: assume (x, u)

        if n == 1:
            return self.op(u_fun)
        else:
            return self.op(x_fun, u_fun)

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
        """
        domain = self.domain

        def _op_fn(disc: ChebColloc2Disc) -> jnp.ndarray:
            n = disc.n
            a, b = disc.domain

            from chebfunjax.chebfun1d.chebfun import Chebfun
            dom = Domain((a, b))
            x_fun = Chebfun.identity(dom)

            # Evaluate op at zero
            zero_vals = jnp.zeros(n, dtype=jnp.float64)
            u0 = Chebfun.from_values(zero_vals, dom)
            try:
                op0 = self._apply_op(x_fun, u0)
                op0_vals = _chebfun_to_values(op0, disc)
            except Exception:
                op0_vals = jnp.zeros(n, dtype=jnp.float64)

            # Build Jacobian column by column
            cols = []
            for j in range(n):
                e_j = jnp.zeros(n, dtype=jnp.float64).at[j].set(1.0)
                u_j = Chebfun.from_values(e_j, dom)
                op_j = self._apply_op(x_fun, u_j)
                op_j_vals = _chebfun_to_values(op_j, disc)
                cols.append(op_j_vals - op0_vals)

            # Each column corresponds to the j-th basis action
            J = jnp.stack(cols, axis=1)
            return J

        return OperatorBlock(_op_fn, order=2, domain=domain)

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

    def _jacobian_matrix_fd(self, disc, x_fun, u_fun, Nu_vals):
        """Compute the Jacobian of self.op at u by forward finite differences."""
        from chebfunjax.chebfun1d.chebfun import Chebfun

        n = disc.n
        dom = Domain(disc.domain)
        h = max(1e-6, 1e-6 * float(jnp.max(jnp.abs(u_fun.funs[0].values))))

        # Jacobian columns
        J_cols = []
        for j in range(n):
            e_j = jnp.zeros(n, dtype=jnp.float64).at[j].set(h)
            u_pert = Chebfun.from_values(u_fun.funs[0].values + e_j, dom)
            Nu_pert = self._apply_op(x_fun, u_pert)
            Nu_pert_vals = _chebfun_to_values(Nu_pert, disc)
            J_cols.append((Nu_pert_vals - Nu_vals) / h)

        return jnp.stack(J_cols, axis=1)

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

            # Finite-difference Jacobian row
            h = 1e-6
            row = jnp.zeros(n, dtype=jnp.float64)
            for j in range(n):
                e_j = jnp.zeros(n, dtype=jnp.float64).at[j].set(h)
                u_pert = Chebfun.from_values(e_j, dom_inner)
                try:
                    g_pert = bc_fn(u_pert)
                    g_pert_pt = float(_safe_eval(g_pert, jnp.array(ep, dtype=jnp.float64)))
                except Exception:
                    g_pert_pt = g0_pt
                row = row.at[j].set((g_pert_pt - g0_pt) / h)
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
    # Compute physical Chebyshev-2 points from the disc descriptor
    a, b = disc.domain
    t_ref = chebpts(disc.n, kind=2)
    x_pts = 0.5 * (b - a) * t_ref + 0.5 * (a + b)
    return jnp.asarray(f(x_pts), dtype=jnp.float64)


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


class _FourierProxy:
    """Linear-operator proxy for Fourier collocation.

    Wraps an (n, n) matrix ``mat`` describing the linear action on the
    grid values of ``u``.  Supports ``diff``, addition/subtraction, and
    multiplication by scalars or grid-sampled variable coefficients, so
    that evaluating ``op(x_grid, proxy)`` assembles the operator matrix.
    """

    def __init__(self, n: int, length: float, mat):
        self.n = n
        self.length = length
        self.mat = mat

    def _wrap(self, mat):
        return _FourierProxy(self.n, self.length, mat)

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
        arr = _np.asarray(other, dtype=float)
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
