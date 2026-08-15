"""Block linear operators with side constraints (MATLAB ``@linop``).

A :class:`BlockLinop` is a :class:`~chebfunjax.operators.chebmatrix.ChebMatrix`
of operator/functional/chebfun/scalar blocks together with

- ``constraint`` -- boundary and side conditions, and
- ``continuity`` -- automatic continuity conditions at interior breakpoints,

exactly as MATLAB's ``linop`` class carries two ``linopConstraint``
properties.  Discretization uses rectangular Chebyshev collocation: block
column ``k`` is discretized with ``n + r_k`` points per subinterval (where
``r_k`` is the maximal differential order appearing in that column) and the
resulting operator rows are projected down onto ``n`` first-kind points, so
that stacking the constraint rows on top restores squareness.

Translated from MATLAB Chebfun ``@linop`` (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import math
from typing import Sequence

import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.operators.blocks import (
    ChebColloc2Disc,
    D,
    FunctionalBlock,
    OperatorBlock,
    _bary_row,
    _blkdiag,
    _const_chebfun,
    eval_at,
    zero_functional,
)
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.utils.quadrature import chebpts_ab

_DomainT = tuple[float, ...]

_DEFAULT_DIMS = (32, 64, 128, 256, 512, 1024)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_block(blk) -> bool:
    return isinstance(blk, (OperatorBlock, FunctionalBlock))


def _is_zero_entry(blk) -> bool:
    """Whether a constraint-row entry contributes nothing to its variable."""
    if isinstance(blk, (int, float, complex)):
        return blk == 0
    return bool(getattr(blk, "iszero", False))


def _piece_from_values(values, a: float, b: float):
    """A single Chebfun piece carrying the given second-kind values."""
    from chebfunjax.chebfun1d.chebfun import Chebfun
    v = jnp.asarray(values)
    if not jnp.iscomplexobj(v):
        v = v.astype(jnp.float64)
    return Chebfun.from_values(v, Domain((float(a), float(b)))).funs[0]


def _chebfun_from_pieces(pieces, bps: _DomainT):
    from chebfunjax.chebfun1d.chebfun import Chebfun
    return Chebfun(funs=list(pieces), domain=Domain(tuple(float(v)
                                                         for v in bps)))


def _projection(sizes: Sequence[int], r: int,
                dom: _DomainT) -> jnp.ndarray:
    """Barycentric down-projection from ``n_i + r`` second-kind points to
    ``n_i`` first-kind points on each subinterval.

    Provenance
    ----------
    MATLAB source : @chebcolloc/reduce.m
    Chebfun commit: 7574c77
    """
    blocks = []
    for ni, (a, b) in zip(sizes, [(dom[k], dom[k + 1])
                                  for k in range(len(dom) - 1)]):
        x_out = chebpts_ab(ni, a, b, kind=1)
        rows = [_bary_row(ni + r, float((2.0 * xo - (a + b)) / (b - a)))
                for xo in x_out]
        blocks.append(jnp.stack(rows, axis=0))
    return _blkdiag(blocks)


def _equation_points(sizes: Sequence[int], dom: _DomainT) -> jnp.ndarray:
    """Concatenated first-kind collocation points where equations are
    enforced (MATLAB ``chebcolloc2/equationPoints``)."""
    return jnp.concatenate(
        [jnp.asarray(chebpts_ab(ni, dom[k], dom[k + 1], kind=1),
                     dtype=jnp.float64)
         for k, ni in enumerate(sizes)])


# ===========================================================================
# BlockLinop
# ===========================================================================


class BlockLinop:
    """A block linear operator with boundary/side and continuity conditions.

    Parameters
    ----------
    blocks : ChebMatrix, OperatorBlock, FunctionalBlock, or 2-D list
        The operator.  A bare block is wrapped in a 1x1 ChebMatrix.
    domain : tuple of float or None
        Override the domain (used to introduce breakpoints).

    Attributes
    ----------
    A : ChebMatrix
        The block matrix.
    constraint : list of (row, value)
        Boundary/side conditions; ``row`` is a list of blocks, one per
        variable, ``value`` the prescribed scalar.
    continuity : list of (row, value)
        Continuity conditions at interior breakpoints.

    Examples
    --------
    >>> from chebfunjax.operators.blocks import D, eval_at
    >>> L = BlockLinop(D((-1.0, 1.0), 2))
    >>> L = L.addbc(eval_at(-1.0), 0.0).addbc(eval_at(1.0), 0.0)

    Provenance
    ----------
    MATLAB source : @linop/linop.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    ChebMatrix, OperatorBlock, FunctionalBlock
    """

    def __init__(self, blocks, domain: _DomainT | None = None) -> None:
        if isinstance(blocks, BlockLinop):
            A = blocks.A
            self.constraint = list(blocks.constraint)
            self.continuity = list(blocks.continuity)
        else:
            if isinstance(blocks, ChebMatrix):
                A = blocks
            elif _is_block(blocks):
                A = ChebMatrix([[blocks]], domain=domain)
            else:
                A = ChebMatrix(blocks, domain=domain)
            self.constraint = []
            self.continuity = []
        if domain is not None:
            A = ChebMatrix(A.blocks, domain=tuple(float(v) for v in domain))
        self.A = A
        self.periodic = False

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    @property
    def domain(self) -> _DomainT:
        """Breakpoints of the operator."""
        return self.A.domain

    @property
    def nrows(self) -> int:
        """Number of block rows."""
        return self.A.nrows

    @property
    def ncols(self) -> int:
        """Number of block columns (variables)."""
        return self.A.ncols

    def is_fun_variable(self) -> list[bool]:
        """Which variables are functions (as opposed to scalars).

        Provenance
        ----------
        MATLAB source : @chebmatrix/chebmatrix.m (``isFunVariable``)
        Chebfun commit: 7574c77
        """
        return self.A.is_fun_variable()

    def proj_order(self) -> list[int]:
        """Down-projection order of each block column.

        Provenance
        ----------
        MATLAB source : @linop/getProjOrder.m
        Chebfun commit: 7574c77
        """
        isfun = self.is_fun_variable()
        out = []
        for j in range(self.ncols):
            if not isfun[j]:
                out.append(0)
                continue
            orders = [blk.order for blk in
                      (self.A.blocks[i][j] for i in range(self.nrows))
                      if _is_block(blk)]
            out.append(max(0, max(orders) if orders else 0))
        return out

    def _row_is_functional(self, i: int) -> bool:
        for blk in self.A.blocks[i]:
            if isinstance(blk, OperatorBlock):
                return False
            if isinstance(blk, FunctionalBlock):
                return True
        # A row of chebfuns/scalars: a function row iff any block is a Chebfun.
        return all(isinstance(blk, (int, float, complex))
                   for blk in self.A.blocks[i])

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _copy(self) -> "BlockLinop":
        out = BlockLinop(self.A)
        out.constraint = list(self.constraint)
        out.continuity = list(self.continuity)
        out.periodic = self.periodic
        return out

    @staticmethod
    def _as_row(row) -> list:
        if isinstance(row, ChebMatrix):
            return list(row.blocks[0])
        if isinstance(row, (list, tuple)):
            return list(row)
        return [row]

    def add_constraint(self, row, value: float = 0.0) -> "BlockLinop":
        """Attach a side condition ``row * u = value``.

        Provenance
        ----------
        MATLAB source : @linop/addConstraint.m
        Chebfun commit: 7574c77
        """
        out = self._copy()
        out.constraint.append((self._as_row(row), value))
        return out

    def addbc(self, row, value: float = 0.0) -> "BlockLinop":
        """Attach a boundary condition, or make the problem periodic.

        ``addbc(L, 'periodic')`` derives periodic side conditions from the
        differential orders of the variables.

        Provenance
        ----------
        MATLAB source : @linop/addbc.m
        Chebfun commit: 7574c77
        """
        if isinstance(row, str):
            if row.lower() != "periodic":
                raise ValueError(
                    "CHEBFUN:LINOP:addbc:periodic -- the only string "
                    "boundary condition is 'periodic'.")
            return self._make_periodic()
        return self.add_constraint(row, value)

    def add_continuity(self, row, value: float = 0.0) -> "BlockLinop":
        """Attach a continuity (jump) condition.

        Provenance
        ----------
        MATLAB source : @linop/addContinuity.m
        Chebfun commit: 7574c77
        """
        out = self._copy()
        out.continuity.append((self._as_row(row), value))
        return out

    def _var_orders(self) -> list[int]:
        """Maximum differential order of each variable (0 for scalars)."""
        return self.proj_order()

    def _zero_row(self, dom: _DomainT) -> list:
        isfun = self.is_fun_variable()
        return [zero_functional(dom) if isfun[j] else 0.0
                for j in range(self.ncols)]

    def _make_periodic(self) -> "BlockLinop":
        dom = self.domain
        orders = self._var_orders()
        out = self._copy()
        out.periodic = True
        for var, d in enumerate(orders):
            if d <= 0:
                continue
            for m in range(d):
                row = out._zero_row(dom)
                base = (eval_at(dom[-1], dom, -1)
                        - eval_at(dom[0], dom, 1))
                row[var] = base * D(dom, m) if m > 0 else base
                out.constraint.append((row, 0.0))
        return out

    def derive_continuity(self, dom: _DomainT) -> "BlockLinop":
        """Continuity conditions at every interior breakpoint of ``dom``.

        Provenance
        ----------
        MATLAB source : @linop/deriveContinuity.m
        Chebfun commit: 7574c77
        """
        out = self._copy()
        if out.continuity or len(dom) <= 2:
            return out
        orders = out._var_orders()
        interior = list(dom[1:-1])
        for var, d in enumerate(orders):
            if d <= 0:
                continue
            for m in range(d):
                for loc in interior:
                    row = out._zero_row(dom)
                    base = (eval_at(loc, dom, -1) - eval_at(loc, dom, 1))
                    row[var] = base * D(dom, m) if m > 0 else base
                    out.continuity.append((row, 0.0))
        return out

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def __mul__(self, u):
        """Apply the operator to a ChebMatrix / list of functions."""
        return self.A * u

    def __rmul__(self, other):
        return NotImplemented

    def __matmul__(self, u):
        return self.A * u

    # ------------------------------------------------------------------
    # Discretization
    # ------------------------------------------------------------------

    def _sizes(self, n, dom: _DomainT) -> list[int]:
        n_int = len(dom) - 1
        if isinstance(n, (list, tuple)):
            sizes = [int(v) for v in n]
            if len(sizes) != n_int:
                raise ValueError(
                    f"BlockLinop: got {len(sizes)} dimensions for "
                    f"{n_int} subintervals.")
            return sizes
        return [int(n)] * n_int

    def _block_row_matrix(self, row: list, sizes, dom, r, isfun,
                          project: bool):
        """Discretize one block row, optionally projecting operator blocks."""
        parts = []
        for j, blk in enumerate(row):
            disc = ChebColloc2Disc([ni + r[j] for ni in sizes], dom)
            if isinstance(blk, OperatorBlock):
                M = blk.matrix(disc)
                parts.append(_projection(sizes, r[j], dom) @ M
                             if project else M)
            elif isinstance(blk, FunctionalBlock):
                parts.append(blk.matrix(disc)[None, :])
            elif isinstance(blk, (int, float, complex)):
                if isfun[j]:
                    raise TypeError(
                        "BlockLinop: a scalar block cannot occupy a "
                        "function column.")
                parts.append(jnp.asarray([[blk]], dtype=jnp.float64)
                             if not isinstance(blk, complex)
                             else jnp.asarray([[blk]]))
            else:
                # A Chebfun block: a column mapping a scalar to a function.
                if project:
                    vals = jnp.ravel(jnp.asarray(
                        blk(_equation_points(sizes, dom))))
                    parts.append(vals[:, None])
                else:
                    pts = ChebColloc2Disc(sizes, dom).points()
                    parts.append(jnp.ravel(jnp.asarray(blk(pts)))[:, None])
        widths = {p.shape[0] for p in parts}
        if len(widths) > 1:
            # Scalar entries in a function row need broadcasting.
            h = max(widths)
            parts = [jnp.tile(p, (h, 1)) if p.shape[0] == 1 else p
                     for p in parts]
        return jnp.concatenate(parts, axis=1)

    def _constraint_rows(self, sizes, dom, r, isfun):
        """Discretized continuity rows on top of constraint rows."""
        rows, vals = [], []
        for row, val in self.continuity:
            rows.append(self._block_row_matrix(row, sizes, dom, r, isfun,
                                               project=False))
            vals.append(val)
        for row, val in self.constraint:
            rows.append(self._block_row_matrix(row, sizes, dom, r, isfun,
                                               project=False))
            vals.append(val)
        return rows, vals

    def matrix(self, n, dom: _DomainT | None = None):
        """Assemble the square discretization matrix (constraints on top).

        Parameters
        ----------
        n : int or sequence of int
            Collocation dimension per subinterval.
        dom : tuple of float or None
            Override the domain.

        Returns
        -------
        jnp.ndarray

        Provenance
        ----------
        MATLAB source : @opDiscretization/matrix.m, @chebcolloc/reduce.m
        Chebfun commit: 7574c77
        """
        M, _, _ = self._assemble(n, dom)
        return M

    def _assemble(self, n, dom: _DomainT | None = None, r=None):
        dom = tuple(float(v) for v in (dom if dom is not None
                                       else self.domain))
        L = self.derive_continuity(dom) if not self.continuity else self
        sizes = self._sizes(n, dom)
        isfun = L.is_fun_variable()
        r = L.proj_order() if r is None else list(r)
        rows, vals = L._constraint_rows(sizes, dom, r, isfun)
        for i in range(L.nrows):
            rows.append(L._block_row_matrix(L.A.blocks[i], sizes, dom, r,
                                            isfun, project=True))
        M = jnp.concatenate(rows, axis=0)
        col_dims = [sum(ni + r[j] for ni in sizes) if isfun[j] else 1
                    for j in range(L.ncols)]
        return M, vals, (sizes, r, isfun, col_dims, dom, len(vals))

    def matrix_oldschool(self, n) -> jnp.ndarray:
        """Square discretization with the side conditions replacing rows.

        This is the pre-rectangular-collocation ("old school") assembly: the
        operator is discretized square at dimension ``n``, then the first
        ``ceil(k/2)`` rows and the last ``k - ceil(k/2)`` rows are replaced
        by the ``k`` side-condition rows.

        Parameters
        ----------
        n : int or sequence of int
            Collocation dimension per subinterval.

        Returns
        -------
        jnp.ndarray

        Examples
        --------
        >>> from chebfunjax.operators.blocks import D, eval_at
        >>> L = linop(D((-1.0, 1.0), 2)).addbc(eval_at(-1.0), 0.0)
        >>> L = L.addbc(eval_at(1.0), 0.0)
        >>> L.matrix_oldschool(12).shape
        (12, 12)

        Provenance
        ----------
        MATLAB source : @linop/feval.m (the ``oldschool`` branch)
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        if self.nrows != 1 or self.ncols != 1:
            raise ValueError(
                "CHEBFUN:LINOP:feval:multivariable -- the 'oldschool' "
                "assembly is not available for multivariable systems.")
        dom = self.domain
        sizes = self._sizes(n, dom)
        L = self.derive_continuity(dom) if not self.continuity else self
        isfun = L.is_fun_variable()
        zeros = [0] * L.ncols          # dimAdjust = 0
        rows, _ = L._constraint_rows(sizes, dom, zeros, isfun)
        A = L.A.dense(n, domain=dom)
        if not rows:
            return A
        B = jnp.concatenate(rows, axis=0)
        k = B.shape[0]
        k2 = -(-k // 2)
        A = A.at[:k2, :].set(B[:k2, :])
        if k - k2 > 0:
            A = A.at[A.shape[0] - (k - k2):, :].set(B[k2:, :])
        return A

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def _rhs(self, entries, sizes, dom, vals):
        """Right-hand side vector: condition values, then the equations
        sampled at the first-kind collocation points.

        Provenance
        ----------
        MATLAB source : @valsDiscretization/rhs.m
        Chebfun commit: 7574c77
        """
        x_out = _equation_points(sizes, dom)
        parts = [jnp.asarray(vals, dtype=jnp.float64).reshape(-1)] \
            if vals else [jnp.zeros(0, dtype=jnp.float64)]
        for i, entry in enumerate(entries):
            if isinstance(entry, (int, float, complex)):
                if self._row_is_functional(i):
                    parts.append(jnp.asarray([entry], dtype=jnp.float64))
                else:
                    parts.append(jnp.full(x_out.shape, float(entry),
                                          dtype=jnp.float64))
            else:
                parts.append(jnp.ravel(jnp.asarray(entry(x_out),
                                                   dtype=jnp.float64)))
        return jnp.concatenate(parts)

    @staticmethod
    def _happy(coeffs, tol: float) -> bool:
        """Whether the Chebyshev coefficients have decayed into the noise.

        Provenance
        ----------
        MATLAB source : @opDiscretization/testConvergence.m
        Chebfun commit: 7574c77
        """
        c = jnp.abs(jnp.asarray(coeffs))
        if c.size < 8:
            return False
        scale = float(jnp.max(c))
        if scale == 0.0:
            return True
        tail = int(max(5, c.size // 8))
        return bool(float(jnp.max(c[-tail:])) < tol * scale)

    def linsolve(self, f=0.0, n=None, tol: float = 1e-12,
                 dom: _DomainT | None = None,
                 discretization: str = "chebcolloc2") -> ChebMatrix:
        """Solve ``L*u = f`` subject to the attached conditions.

        Parameters
        ----------
        f : ChebMatrix, list, Chebfun, or scalar
            Right-hand side, one entry per block row.
        n : int, sequence of int, or None
            Fixed discretization dimension; ``None`` adapts.
        tol : float, default 1e-12
            Relative coefficient-decay tolerance for the adaptive loop.
        dom : tuple of float or None
            Override the domain (used to introduce breakpoints).

        Returns
        -------
        ChebMatrix
            One entry per variable: a Chebfun for function variables, a
            float for scalar variables.

        Provenance
        ----------
        MATLAB source : @linop/linsolve.m, @linop/mldivide.m
        Chebfun commit: 7574c77
        """
        if discretization != "chebcolloc2":
            return self._linsolve_altdisc(f, n, discretization)
        rhs = self._normalize_rhs(f)
        base_dom = tuple(float(v) for v in (dom if dom is not None
                                            else self.domain))
        merged = set(base_dom)
        for entry in rhs:
            edom = getattr(entry, "domain", None)
            if edom is not None:
                bps = (edom.breakpoints if hasattr(edom, "breakpoints")
                       else edom)
                merged.update(float(v) for v in bps)
        use_dom = tuple(sorted(v for v in merged
                               if base_dom[0] <= v <= base_dom[-1]))

        dims = list(_DEFAULT_DIMS) if n is None else [n]
        out = None
        for nn in dims:
            out = self._solve_at(rhs, nn, use_dom)
            if n is not None:
                break
            if all(self._happy(u.coeffs, tol) for u in out
                   if hasattr(u, "coeffs")):
                break
        return ChebMatrix([[u] for u in out], domain=use_dom)

    def _normalize_rhs(self, f) -> list:
        if isinstance(f, ChebMatrix):
            return [f.blocks[i][0] for i in range(f.nrows)]
        if isinstance(f, (list, tuple)):
            return list(f)
        if isinstance(f, (int, float, complex)):
            return [f] * self.nrows
        return [f]

    def _solve_at(self, rhs, n, dom):
        M, vals, info = self._assemble(n, dom)
        sizes, r, isfun, col_dims, dom, _ = info
        b = self._rhs(rhs, sizes, dom, vals)
        if M.shape[0] != M.shape[1]:
            raise ValueError(
                "CHEBFUN:LINOP:linsolve:notSquare -- operator may not have "
                f"the correct number of boundary conditions (matrix is "
                f"{M.shape[0]}x{M.shape[1]}).")
        v = jnp.linalg.solve(M, b)
        return self._partition(v, sizes, r, isfun, col_dims, dom)

    @staticmethod
    def _partition(v, sizes, r, isfun, col_dims, dom):
        out = []
        pos = 0
        for j, width in enumerate(col_dims):
            seg = v[pos:pos + width]
            pos += width
            if not isfun[j]:
                out.append(float(jnp.real(seg[0])) if not
                           jnp.iscomplexobj(seg) else complex(seg[0]))
                continue
            pieces = []
            q = 0
            for i, ni in enumerate(sizes):
                m = ni + r[j]
                pieces.append(_piece_from_values(seg[q:q + m], dom[i],
                                                 dom[i + 1]))
                q += m
            out.append(_chebfun_from_pieces(pieces, dom))
        return out

    def __truediv__(self, f):
        return self.linsolve(f)

    def __rtruediv__(self, f):
        raise TypeError("Use L.linsolve(f) or L / f to solve L*u = f.")

    def solve(self, f=0.0, **kwargs) -> ChebMatrix:
        """Alias of :meth:`linsolve` (MATLAB ``A\\f``).

        Provenance
        ----------
        MATLAB source : @linop/mldivide.m
        Chebfun commit: 7574c77
        """
        return self.linsolve(f, **kwargs)

    # ------------------------------------------------------------------
    # Operator exponential
    # ------------------------------------------------------------------

    def expm(self, t: float, u0, n=None,
             dom: _DomainT | None = None,
             tol: float = 1e-12,
             discretization: str = "chebcolloc2") -> ChebMatrix:
        """Propagate ``u0`` by the operator exponential ``exp(t*L)``.

        The side conditions attached to ``L`` are enforced homogeneously:
        they are used to lift the reduced (projected) discretization back to
        full size, exactly as MATLAB's ``valsDiscretization/expm`` does.

        Parameters
        ----------
        t : float
            Propagation time.  ``t = 0`` returns ``u0`` unchanged.
        u0 : Chebfun, list, or ChebMatrix
            Initial condition, one entry per variable.
        n : int, sequence of int, or None
            Collocation dimension per subinterval; ``None`` adapts.
        dom : tuple of float or None
            Override the domain.
        tol : float, default 1e-12
            Relative coefficient-decay tolerance for the adaptive loop.

        Returns
        -------
        ChebMatrix

        Examples
        --------
        >>> from chebfunjax.operators.blocks import D, eval_at
        >>> import chebfunjax as cj
        >>> L = linop(D((-1.0, 1.0), 2))
        >>> L = L.addbc(eval_at(-1.0), 0.0).addbc(eval_at(1.0), 0.0)
        >>> u = L.expm(0.01, cj.chebfun(lambda x: 1 - x ** 2))

        Provenance
        ----------
        MATLAB source : @linop/expm.m, @valsDiscretization/expm.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        if discretization != "chebcolloc2":
            return self._expm_altdisc(t, u0, n, discretization)

        entries = self._normalize_rhs(u0)
        base = tuple(float(v) for v in (dom if dom is not None
                                        else self.domain))
        merged = set(base)
        for entry in entries:
            edom = getattr(entry, "domain", None)
            if edom is not None:
                bps = (edom.breakpoints if hasattr(edom, "breakpoints")
                       else edom)
                merged.update(float(v) for v in bps)
        use_dom = tuple(sorted(v for v in merged
                               if base[0] <= v <= base[-1]))

        dims = list(_DEFAULT_DIMS) if n is None else [n]
        out = None
        for nn in dims:
            out = self._expm_at(t, entries, nn, use_dom)
            if n is not None:
                break
            if all(self._happy(u.coeffs, tol) for u in out
                   if hasattr(u, "coeffs")):
                break
        out = [u.simplify() if hasattr(u, "simplify") else u for u in out]
        return ChebMatrix([[u] for u in out], domain=use_dom)

    def _expm_at(self, t, entries, n, use_dom):
        import numpy as np  # uses-numpy: dense matrix exponential
        import scipy.linalg as sla

        from chebfunjax.operators.blocks import _to_values

        M, _, info = self._assemble(n, use_dom)
        sizes, r, isfun, col_dims, use_dom, n_con = info

        v0 = jnp.concatenate([
            _to_values(entry, ChebColloc2Disc(
                [ni + r[k] for ni in sizes], use_dom))
            if isfun[k] else jnp.asarray([float(entry)], dtype=jnp.float64)
            for k, entry in enumerate(entries)])

        if t == 0:
            return self._partition(v0, sizes, r, isfun, col_dims, use_dom)

        P = _blkdiag([_projection(sizes, r[k], use_dom) if isfun[k]
                      else jnp.ones((1, 1), dtype=jnp.float64)
                      for k in range(self.ncols)])
        B = M[:n_con, :]
        PA = M[n_con:, :]

        m_red, m_orig = P.shape
        lhs = np.asarray(jnp.concatenate([B, P], axis=0))
        rhs = np.zeros((m_orig, m_red))
        rhs[m_orig - m_red:, :] = np.eye(m_red)
        Q = np.linalg.solve(lhs, rhs)

        E = sla.expm(float(t) * np.asarray(PA) @ Q)
        v = Q @ (E @ (np.asarray(P) @ np.asarray(v0)))
        return self._partition(jnp.asarray(v), sizes, r, isfun,
                               col_dims, use_dom)

    # ------------------------------------------------------------------
    # Initial guesses
    # ------------------------------------------------------------------

    def fit_bcs(self) -> ChebMatrix:
        """Low-degree polynomial satisfying the side and continuity conditions.

        Returns
        -------
        ChebMatrix
            One entry per variable; used as the initial guess for a Newton
            iteration.

        Examples
        --------
        >>> from chebfunjax.operators.blocks import D, eval_at
        >>> L = linop(D((-1.0, 1.0), 2)).addbc(eval_at(-1.0), -1.0)
        >>> u0 = L.fit_bcs()

        Provenance
        ----------
        MATLAB source : @linop/fitBCs.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        import numpy as np  # uses-numpy: dense least-squares on the BC rows
        dom = self.domain
        n_int = len(dom) - 1
        L = self.derive_continuity(dom) if not self.continuity else self
        isfun = L.is_fun_variable()

        if all(getattr(blk, "isnotdiffint", True)
               for row in L.A.blocks for blk in row):
            zero = _const_chebfun(0.0, dom)
            return ChebMatrix([[zero if isfun[k] else 0.0]
                               for k in range(L.ncols)], domain=dom)

        # Degree of each unknown: the number of conditions it appears in.
        deg = [0] * L.ncols
        for row, _ in L.constraint:
            for k, blk in enumerate(row):
                if not _is_zero_entry(blk):
                    deg[k] += 1
        if L.continuity:
            var_ord = L.proj_order()
            deg = [max(deg[k], var_ord[k] - 1) for k in range(L.ncols)]
        deg = [1 if not isfun[k] else max(1, deg[k]) for k in range(L.ncols)]

        rows, vals = [], []
        for row, val in L.continuity:
            rows.append(row)
            vals.append(val)
        for row, val in L.constraint:
            rows.append(row)
            vals.append(val)

        sizes = [0] * n_int
        B = b = None
        for _ in range(5):
            mats = [L._block_row_matrix(row, sizes, dom, deg, isfun,
                                        project=False) for row in rows]
            B = np.asarray(jnp.concatenate(mats, axis=0))
            b = np.asarray(vals, dtype=np.float64)
            keep = np.any(B != 0.0, axis=1)
            B, b = B[keep], b[keep]
            if B.shape[0] == 0 or np.linalg.matrix_rank(B) >= B.shape[0]:
                break
            sizes = [s + 1 for s in sizes]

        if B.shape[0] == 0:
            u = np.zeros(sum(sum(sizes) + deg[k] if isfun[k] else 1
                             for k in range(L.ncols)))
        else:
            u = np.linalg.lstsq(B, -b, rcond=None)[0]

        col_dims = [sum(s + deg[k] for s in sizes) if isfun[k] else 1
                    for k in range(L.ncols)]
        parts = self._partition(jnp.asarray(u), sizes,
                                deg, isfun, col_dims, dom)
        return ChebMatrix([[p] for p in parts], domain=dom)

    # ------------------------------------------------------------------
    # Eigenvalues
    # ------------------------------------------------------------------

    def _eigs_altdisc(self, k, sigma, n, discretization, B=None):
        """Eigenproblem under the ultraS / chebcolloc1 backends.

        Provenance
        ----------
        MATLAB source : @linop/eigs.m with prefs.discretization
        Chebfun commit: 7574c77
        """
        import numpy as np  # uses-numpy: dense generalized eigensolve
        import scipy.linalg as sla

        from chebfunjax.operators.altdisc import system_matrices
        if B is not None and not isinstance(B, BlockLinop):
            B = BlockLinop(B)
        rmin = None
        if B is not None:
            rmin = [max((blk.order for blk in B.A.blocks[i]
                         if isinstance(blk, OperatorBlock)), default=0)
                    for i in range(B.nrows)]
        sd = system_matrices(self, n, discretization,
                             row_order_min=rmin)
        A, Bm = sd.A, sd.mass(B)
        lam = sla.eig(A, Bm, right=False)
        finite = np.isfinite(lam) & (np.abs(lam) < 1e8)
        lam = lam[finite]
        if np.max(np.abs(lam.imag)) < 1e-8 * max(
                np.max(np.abs(lam.real)), 1e-300):
            lam = lam.real
        if sigma is None or (isinstance(sigma, str)
                             and sigma.upper() == "SM") or sigma == 0:
            order = np.argsort(np.abs(lam))
        else:
            order = np.argsort(np.abs(lam - sigma))
        lam = lam[order[:k]]
        lam = np.sort_complex(lam.astype(complex)) if np.iscomplexobj(
            lam) else np.sort(lam)
        # Eigenvectors for the selected eigenvalues, one shifted solve
        # each (inverse iteration on the discrete pencil).
        vecs = []
        for lv in lam:
            M = A - lv * Bm
            _u, _s, vh = np.linalg.svd(M)
            v = vh[-1].conj()
            entries = sd.recover(v)
            nrm = max((float(e.norm()) for e in entries
                       if hasattr(e, "norm")), default=1.0)
            if nrm > 0:
                entries = [e * (1.0 / nrm) if hasattr(e, "norm")
                           else e / nrm for e in entries]
            vecs.append(ChebMatrix([[e] for e in entries],
                                   domain=self.domain))
        return jnp.asarray(lam), vecs

    def _linsolve_altdisc(self, f, n, discretization):
        """Solve under the ultraS / chebcolloc1 backends.

        Provenance
        ----------
        MATLAB source : @linop/linsolve.m with prefs.discretization
        Chebfun commit: 7574c77
        """
        import numpy as np  # uses-numpy: dense solve

        from chebfunjax.operators.altdisc import system_matrices
        nn = int(n) if n is not None else 128
        entries = self._normalize_rhs(f)
        use_dom = self._merged_domain(entries)
        sd = system_matrices(self, nn, discretization, dom=use_dom)
        b = sd.rhs(entries)
        v = np.linalg.solve(np.asarray(sd.A), b)
        return ChebMatrix([[u] for u in sd.recover(v)],
                          domain=use_dom)

    def _merged_domain(self, entries, dom=None):
        """The operator domain enriched with the breakpoints of the
        given functions (MATLAB ``domain.merge``)."""
        base = tuple(float(v) for v in (dom if dom is not None
                                        else self.domain))
        merged = set(base)
        for entry in entries:
            edom = getattr(entry, "domain", None)
            if edom is not None:
                bps = (edom.breakpoints if hasattr(edom, "breakpoints")
                       else edom)
                merged.update(float(v) for v in bps)
        return tuple(sorted(v for v in merged
                            if base[0] <= v <= base[-1]))

    def _expm_altdisc(self, t, u0, n, discretization):
        """Operator exponential under the ultraS / chebcolloc1 backends:
        reduce with the side conditions, exponentiate the reduced
        generator, and lift back (``u(t) = Q expm(t P A Q) P_s u0``).

        Provenance
        ----------
        MATLAB source : @linop/expm.m with prefs.discretization
        Chebfun commit: 7574c77
        """
        import numpy as np  # uses-numpy: dense expm
        import scipy.linalg as sla

        from chebfunjax.operators.altdisc import system_matrices
        entries = self._normalize_rhs(u0)
        use_dom = self._merged_domain(entries)
        nn = int(n) if n is not None else 96
        sd = system_matrices(self, nn, discretization, dom=use_dom)
        ncon = sd.con_rows.shape[0]
        T_A = np.asarray(sd.A)[ncon:]
        T_S = np.asarray(sd.mass(None))[ncon:]
        Bc = np.asarray(sd.con_rows)
        width = T_A.shape[1]
        nred = T_S.shape[0]
        # Lift: full trial vector from reduced dof, satisfying Bc u = 0.
        Q = np.linalg.solve(
            np.vstack([Bc, T_S]),
            np.vstack([np.zeros((ncon, nred)), np.eye(nred)]))
        G = T_A @ Q
        w0 = T_S @ sd.trial_vector(entries)
        v_full = Q @ (sla.expm(t * G) @ w0)
        if v_full.shape != (width,):
            v_full = v_full.ravel()
        return ChebMatrix([[u] for u in sd.recover(v_full)],
                          domain=use_dom)

    def eigs(self, k: int = 6, sigma=None, B: "BlockLinop | None" = None,
             n: int = 65, dom: _DomainT | None = None,
             rayleigh: bool = False, discretization: str = "chebcolloc2"):
        """Eigenvalues and eigenfunctions of ``L`` (or of ``L*u = lam*B*u``).

        Parameters
        ----------
        k : int, default 6
            Number of eigenvalues.
        sigma : float, complex, 'LM', 'SM', or None
            Target; ``None`` and ``'SM'`` mean nearest zero.
        B : BlockLinop or None
            Mass operator for a generalized problem.
        n : int, default 65
            Collocation dimension per subinterval.
        dom : tuple of float or None
            Override the domain.
        rayleigh : bool, default False
            Perform one step of Rayleigh quotient iteration on the computed
            eigenpairs to improve their accuracy.

        Returns
        -------
        lams : jnp.ndarray, shape (k,)
            Eigenvalues, sorted ascending as in MATLAB.
        vecs : list of ChebMatrix
            The corresponding eigenfunctions.

        Provenance
        ----------
        MATLAB source : @linop/eigs.m
        Chebfun commit: 7574c77
        """
        import numpy as np  # uses-numpy: dense generalized eigensolve
        if discretization != "chebcolloc2":
            return self._eigs_altdisc(k, sigma, n, discretization, B=B)
        r_use = self.proj_order()
        if B is not None:
            r_b = (B.proj_order() if isinstance(B, BlockLinop)
                   else BlockLinop(B).proj_order())
            r_use = [max(a, b) for a, b in zip(r_use, r_b)]
        MA, _, info = self._assemble(n, dom, r=r_use)
        sizes, r, isfun, col_dims, use_dom, n_con = info
        rhs_src = self.A.identity() if B is None else (
            B.A if isinstance(B, BlockLinop) else B)
        rhs_rows = [self._block_row_matrix(rhs_src.blocks[i], sizes,
                                           use_dom, r, isfun, project=True)
                    for i in range(rhs_src.nrows)]
        MB = jnp.concatenate(rhs_rows, axis=0)
        MB = jnp.concatenate(
            [jnp.zeros((n_con, MB.shape[1]), dtype=MB.dtype), MB], axis=0)

        lam, vec = _geig(np.asarray(MA), np.asarray(MB))
        finite = np.isfinite(lam)
        lam, vec = lam[finite], vec[:, finite]
        target = 0.0 if sigma is None or (
            isinstance(sigma, str) and sigma.upper() == "SM") else sigma
        if isinstance(target, str):
            if target.upper() != "LM":
                raise ValueError(f"eigs: unknown sigma {sigma!r}.")
            idx = np.argsort(-np.abs(lam))[:k]
        else:
            idx = np.argsort(np.abs(lam - complex(target)))[:k]
        lam, vec = lam[idx], vec[:, idx]
        # MATLAB's SORT on a real vector is ascending; on a complex vector it
        # orders by magnitude and then by phase angle.
        scale = float(np.max(np.abs(lam))) if lam.size else 1.0
        if lam.size and np.max(np.abs(lam.imag)) <= 1e-12 * max(scale, 1.0):
            order = np.argsort(lam.real)
        else:
            # Magnitudes are rounded before sorting so that a multiple
            # eigenvalue is not split by rounding noise.
            mag = np.round(np.abs(lam) / max(scale, 1e-300), 9)
            order = np.lexsort((np.angle(lam), mag))
        lam, vec = lam[order], vec[:, order]
        funs = []
        for j in range(vec.shape[1]):
            col = jnp.asarray(vec[:, j])
            parts = self._partition(col, sizes, r, isfun, col_dims, use_dom)
            funs.append(ChebMatrix([[p] for p in parts], domain=use_dom))
        if rayleigh:
            lam, funs = self._rayleigh_qi(jnp.asarray(lam), funs, B)
        return jnp.asarray(lam), funs

    def _rayleigh_qi(self, lam, funs, B):
        """One step of Rayleigh quotient iteration on the eigenpairs.

        Provenance
        ----------
        MATLAB source : @linop/eigs.m (``rayleighQI``)
        Chebfun commit: 7574c77
        """
        import numpy as np  # uses-numpy: scalar Rayleigh quotients
        Bmat = self.A.identity() if B is None else (
            B.A if isinstance(B, BlockLinop) else B)
        new_lam, new_funs = [], []
        for j in range(len(funs)):
            u = funs[j]
            val = complex(lam[j])
            shifted = BlockLinop(self.A - Bmat * val)
            shifted.constraint = list(self.constraint)
            shifted.continuity = list(self.continuity)
            rhs = Bmat * u
            v = shifted.linsolve([rhs[i] for i in range(rhs.nrows)])
            nrm = math.sqrt(sum(float(abs(v[i].norm())) ** 2
                                for i in range(v.nrows)))
            v = ChebMatrix([[v[i] * (1.0 / nrm)] for i in range(v.nrows)],
                           domain=v.domain)
            Av, Bv = self.A * v, Bmat * v
            num = sum(complex(v[i].inner(Av[i])) for i in range(v.nrows))
            den = sum(complex(v[i].inner(Bv[i])) for i in range(v.nrows))
            new_lam.append(num / den)
            new_funs.append(v)
        arr = np.asarray(new_lam)
        if np.max(np.abs(arr.imag)) <= 1e-12 * max(
                float(np.max(np.abs(arr))), 1.0):
            order = np.argsort(arr.real)
        else:
            order = np.lexsort((np.angle(arr), np.abs(arr)))
        return (jnp.asarray(arr[order]), [new_funs[i] for i in order])

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"BlockLinop({self.nrows}x{self.ncols}, "
                f"domain={self.domain}, "
                f"{len(self.constraint)} constraints)")


def _geig(MA, MB):
    """Dense generalized eigenproblem ``MA v = lam MB v``."""
    import numpy as np  # uses-numpy: LAPACK generalized eigensolve
    import scipy.linalg as sla
    lam, vec = sla.eig(np.asarray(MA), np.asarray(MB))
    return lam, vec


def linop(blocks, domain: _DomainT | None = None) -> BlockLinop:
    """Build a :class:`BlockLinop` from blocks (MATLAB ``linop``).

    Parameters
    ----------
    blocks : ChebMatrix, OperatorBlock, FunctionalBlock, or 2-D list
        The operator.
    domain : tuple of float or None
        Override the domain.

    Returns
    -------
    BlockLinop

    Examples
    --------
    >>> from chebfunjax.operators.blocks import D
    >>> L = linop(D((-1.0, 1.0), 2))

    Provenance
    ----------
    MATLAB source : @linop/linop.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    BlockLinop
    """
    return BlockLinop(blocks, domain=domain)


def addbc(L: BlockLinop, row, value: float = 0.0) -> BlockLinop:
    """Functional form of :meth:`BlockLinop.addbc`.

    Parameters
    ----------
    L : BlockLinop
    row : list of blocks, ChebMatrix, FunctionalBlock, or 'periodic'
    value : float, default 0

    Returns
    -------
    BlockLinop

    Examples
    --------
    >>> from chebfunjax.operators.blocks import D, eval_at
    >>> L = addbc(linop(D((-1.0, 1.0), 2)), eval_at(-1.0), 0.0)

    Provenance
    ----------
    MATLAB source : @linop/addbc.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    BlockLinop.addbc
    """
    return L.addbc(row, value)
