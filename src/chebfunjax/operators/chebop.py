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
    ) -> None:
        self.op = op
        self.domain: tuple[float, float] = tuple(float(v) for v in domain)
        self._lbc_raw = None
        self._rbc_raw = None
        if lbc is not None:
            self.lbc = lbc
        if rbc is not None:
            self.rbc = rbc

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

        # Try to detect linearity by checking if operator is an OperatorBlock
        if self._is_linear():
            return self._solve_linear(f, n=n, n_min=n_min, n_max=n_max, tol=tol)
        else:
            return self._solve_nonlinear(
                f, n=n, n_min=n_min, n_max=n_max, tol=tol,
                max_iter=max_iter, newton_tol=newton_tol,
            )

    def eigs(
        self,
        n: int | None = None,
        k: int = 6,
        n_default: int = 64,
        sigma=None,
    ) -> jnp.ndarray:
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
        linop = self._build_linop(value_shift=0.0)
        return linop.eigs(n=n, k=k, n_default=n_default, sigma=sigma)

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
        """Newton iteration for nonlinear operators.

        Uses a simple fixed-size Newton iteration:

        1. Start from u0 = 0.
        2. At each step, compute the residual r = N[u] - f.
        3. Linearise N around u to get a Linop J.
        4. Solve J * delta = -r.
        5. Update u ← u + delta.
        6. Repeat until ||delta|| < newton_tol.

        The discretization size is chosen as the first size in the adaptive
        ladder that gives convergence of the *linear* solve (or the fixed n).

        Provenance
        ----------
        MATLAB source : @chebop/newtonBVP.m, @chebop/solvebvpNonlinear.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import Chebfun

        a, b = self.domain
        dom = Domain(self.domain)

        # Choose a fixed n for Newton iteration
        # (adaptive sizing for nonlinear is more complex; use n_min or n)
        sz = n if n is not None else max(n_min, 16)

        disc = ChebColloc2Disc(sz, self.domain)
        # Compute physical Chebyshev-2 points
        a, b = self.domain
        t_ref = chebpts(sz, kind=2)
        x_pts = 0.5 * (b - a) * t_ref + 0.5 * (a + b)

        # RHS at collocation points
        rhs = _make_rhs_callable(f)
        f_vals = jnp.asarray(rhs(x_pts), dtype=jnp.float64)

        # Initial guess: zero
        u_vals = jnp.zeros(sz, dtype=jnp.float64)

        bcs, bc_vals = self._parse_bcs()

        for it in range(max_iter):
            u_fun = Chebfun.from_values(u_vals, dom)
            x_fun = Chebfun.identity(dom)

            # Evaluate residual r = N[u] - f
            Nu_fun = self._apply_op(x_fun, u_fun)
            Nu_vals = _chebfun_to_values(Nu_fun, disc)
            r_vals = Nu_vals - f_vals

            # Replace BC rows in residual with BC errors
            n_bc = len(bcs)
            for i, (bc, bc_val) in enumerate(zip(bcs, bc_vals)):
                bc_row = bc.matrix(disc)
                bc_err = float(jnp.dot(bc_row, u_vals)) - float(bc_val)
                r_idx = sz - n_bc + i
                r_vals = r_vals.at[r_idx].set(bc_err)

            # Linearise around u: build Jacobian matrix by finite differences
            J_mat = self._jacobian_matrix(disc, x_fun, u_fun, Nu_vals)

            # Impose BCs on Jacobian rows
            for i, bc in enumerate(bcs):
                bc_row = bc.matrix(disc)
                row_idx = sz - n_bc + i
                J_mat = J_mat.at[row_idx, :].set(bc_row)

            # Solve J * delta = -r
            delta_vals = jnp.linalg.solve(J_mat, -r_vals)

            # Update
            u_vals = u_vals + delta_vals

            # Convergence check
            correction_norm = float(jnp.max(jnp.abs(delta_vals)))
            if correction_norm < newton_tol:
                break
        else:
            warnings.warn(
                f"Chebop.solve (Newton): did not converge in {max_iter} iterations "
                f"(last correction norm = {correction_norm:.2e}).",
                stacklevel=3,
            )

        return Chebfun.from_values(u_vals, dom)

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
