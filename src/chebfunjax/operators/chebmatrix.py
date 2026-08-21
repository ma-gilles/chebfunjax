"""ChebMatrix — block matrix of operators and functionals for ODE systems.

A ``ChebMatrix`` holds a 2-D list of blocks, where each block is one of:

- :class:`~chebfunjax.operators.blocks.OperatorBlock` (function -> function)
- :class:`~chebfunjax.operators.blocks.FunctionalBlock` (function -> scalar)
- a scalar ``float`` or ``int`` (acts as a constant in a scalar row/column)

The ``matrix(n)`` method assembles the full dense matrix at a given
discretization size ``n``.

Typical use: building the linear system for a boundary-value problem.

    >>> from chebfunjax.operators.blocks import D, I, eval_at
    >>> from chebfunjax.operators.chebmatrix import ChebMatrix
    >>> import jax.numpy as jnp
    >>> # u'' + u = 0, u(-1) = 0, u(1) = 0  (2nd-order BVP)
    >>> D2 = D(order=2) + I()           # D^2 + I: OperatorBlock
    >>> bc_left  = eval_at(-1.0)
    >>> bc_right = eval_at( 1.0)
    >>> cm = ChebMatrix([[D2], [bc_left], [bc_right]])
    >>> A, row_sizes = cm.matrix(16)
    >>> A.shape
    (18, 16)

Translated from MATLAB Chebfun class ``@chebmatrix`` (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp

from chebfunjax.operators.blocks import (
    _DEFAULT_DOMAIN,
    ChebColloc2Disc,
    FunctionalBlock,
    OperatorBlock,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

_Block = Union[OperatorBlock, FunctionalBlock, int, float]
_DomainT = tuple[float, float]


# ===========================================================================
# ChebMatrix
# ===========================================================================


class ChebMatrix:
    """Block matrix of operators, functionals, and scalars.

    A ``ChebMatrix`` is a 2-D Python list of blocks.  Each row contains
    blocks that share the same *output* type (function space or scalar).
    When discretized, operator rows contribute ``n`` rows to the assembled
    matrix and functional rows contribute 1 row each.

    Parameters
    ----------
    blocks : list[list[_Block]]
        2-D list (list of rows, each row is a list of blocks).  All blocks
        in a column must have the same physical domain.
    domain : (float, float) or None
        Override domain for all blocks.  If ``None``, the domain is inferred
        from the first block that has a ``domain`` attribute; defaults to
        ``(-1, 1)`` if none is found.

    Attributes
    ----------
    blocks : list[list[_Block]]
        The 2-D list of blocks.
    nrows, ncols : int
        Number of block-rows and block-columns.
    domain : (float, float)
        Common physical domain.

    Examples
    --------
    BVP  u'' + u = sin(x),  u(-1) = u(1) = 0:

    >>> from chebfunjax.operators.blocks import D, I, eval_at
    >>> from chebfunjax.operators.chebmatrix import ChebMatrix
    >>> L   = D(order=2) + I()
    >>> bc0 = eval_at(-1.0)
    >>> bc1 = eval_at(1.0)
    >>> cm  = ChebMatrix([[L], [bc0], [bc1]])
    >>> A, rszs = cm.matrix(12)   # (14, 12) system

    Notes
    -----
    - Each row in ``blocks`` must contain exactly one block (the current
      implementation handles 1-column systems, i.e. scalar ODEs).  For
      multi-component systems, use one column per unknown.
    - For a BVP with ``m`` boundary conditions and operator interior, the
      assembled matrix has shape ``(n - m + m, n) = (n, n)`` when the
      caller replaces the last ``m`` interior rows with the BC rows.
      ``ChebMatrix.matrix`` returns the *raw* stacked matrix; the caller
      decides how to impose BCs.

    Provenance
    ----------
    MATLAB source : @chebmatrix/chebmatrix.m, @chebmatrix/matrix.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    OperatorBlock, FunctionalBlock, blocks.D, blocks.I, blocks.eval_at
    """

    def __init__(
        self,
        blocks: list[list[_Block]],
        domain: _DomainT | None = None,
    ) -> None:
        if not blocks:
            raise ValueError("ChebMatrix: blocks list must be non-empty.")
        # Validate rectangular structure
        ncols = len(blocks[0])
        for i, row in enumerate(blocks):
            if len(row) != ncols:
                raise ValueError(
                    f"ChebMatrix: all rows must have the same number of columns. "
                    f"Row 0 has {ncols} columns, row {i} has {len(row)}."
                )
        self.blocks: list[list[_Block]] = blocks
        self.nrows: int = len(blocks)
        self.ncols: int = ncols

        # Infer domain
        if domain is not None:
            self.domain: _DomainT = domain
        else:
            self.domain = _infer_domain(blocks)

    # ------------------------------------------------------------------
    # Container API (MATLAB chebmatrix parity, Fable 5 audit)
    # ------------------------------------------------------------------

    @classmethod
    def from_array(cls, arr, domain=None) -> "ChebMatrix":
        """Build from a 2-D array of scalars, or a 2-D list mixing
        scalars/chebfuns/blocks (MATLAB chebmatrix(r) /
        chebmatrix(num2cell(r))).

        Provenance
        ----------
        MATLAB source : @chebmatrix/chebmatrix.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        if hasattr(arr, "ndim"):
            a = _np.atleast_2d(_np.asarray(arr))
            blocks = [[float(a[i, j]) for j in range(a.shape[1])]
                      for i in range(a.shape[0])]
            return cls(blocks, domain=domain)
        rows = arr if isinstance(arr[0], (list, tuple)) else [arr]
        for row in rows:
            for blk in row:
                if hasattr(blk, "ndim") and getattr(
                        blk, "ndim", 0) > 0:
                    raise ValueError(
                        "chebmatrix: cell entries must be scalars, "
                        "chebfuns, or blocks (nonscalarcell)")
        return cls([list(r) for r in rows], domain=domain)

    @staticmethod
    def _as_rows(item) -> list[list]:
        """Normalize a vertcat item into a list of block-rows."""
        import numpy as _np
        if isinstance(item, ChebMatrix):
            return [list(r) for r in item.blocks]
        if isinstance(item, _np.ndarray):
            a = _np.atleast_2d(item)
            if a.shape[0] == 1 and a.shape[1] > 1:
                a = a.T  # a bare 1-D array is a column for vertcat
            return [[float(a[i, j]) for j in range(a.shape[1])]
                    for i in range(a.shape[0])]
        if isinstance(item, (list, tuple)):
            return [list(item)]          # an explicit row
        return [[item]]                  # a scalar or Chebfun -> 1x1 row

    @classmethod
    def vertcat(cls, *items, domain=None) -> "ChebMatrix":
        """Vertical block concatenation ``[a; b; ...]`` (MATLAB vertcat).

        Each item may be a Chebfun, a scalar, a numeric column vector, a row
        (list of blocks), or another ChebMatrix; the resulting rows are
        stacked.  Returns a ChebMatrix.

        Provenance
        ----------
        MATLAB source : @chebfun/vertcat.m, @chebmatrix/vertcat.m
        Chebfun commit: 7574c77
        """
        rows: list[list] = []
        for it in items:
            rows.extend(cls._as_rows(it))
        return cls(rows, domain=domain)

    @classmethod
    def horzcat(cls, *items, domain=None) -> "ChebMatrix":
        """Horizontal block concatenation ``[a b ...]`` -- one block-row
        (MATLAB horzcat).

        Provenance
        ----------
        MATLAB source : @chebmatrix/horzcat.m
        Chebfun commit: 7574c77
        """
        return cls([list(items)], domain=domain)

    @property
    def size(self) -> tuple[int, int]:
        """(nrows, ncols) -- MATLAB size(A)."""
        return (self.nrows, self.ncols)

    def __len__(self) -> int:
        """max(nrows, ncols) -- MATLAB length(A)."""
        return max(self.nrows, self.ncols)

    @property
    def T(self) -> "ChebMatrix":
        """Transpose of the block layout (MATLAB A')."""
        blocks = [[self.blocks[i][j] for i in range(self.nrows)]
                  for j in range(self.ncols)]
        return ChebMatrix(blocks, domain=self.domain)

    def fliplr(self) -> "ChebMatrix":
        """Reverse block columns (MATLAB fliplr)."""
        return ChebMatrix([row[::-1] for row in self.blocks],
                          domain=self.domain)

    def flipud(self) -> "ChebMatrix":
        """Reverse block rows (MATLAB flipud)."""
        return ChebMatrix([list(r) for r in self.blocks[::-1]],
                          domain=self.domain)

    def deal(self):
        """All blocks in row-major order (MATLAB deal)."""
        return [blk for row in self.blocks for blk in row]

    def cellfun(self, fn) -> "ChebMatrix":
        """Apply fn to every block (MATLAB cellfun)."""
        return ChebMatrix(
            [[fn(blk) for blk in row] for row in self.blocks],
            domain=self.domain)

    def _zip(self, other, fn) -> "ChebMatrix":
        if isinstance(other, ChebMatrix):
            if (self.nrows, self.ncols) != (other.nrows,
                                            other.ncols):
                raise ValueError("chebmatrix: size mismatch")
            return ChebMatrix(
                [[fn(a, b) for a, b in zip(ra, rb)]
                 for ra, rb in zip(self.blocks, other.blocks)],
                domain=self.domain)
        return ChebMatrix(
            [[fn(a, other) for a in row] for row in self.blocks],
            domain=self.domain)

    def __add__(self, other) -> "ChebMatrix":
        return self._zip(other, lambda a, b: a + b)

    def __radd__(self, other) -> "ChebMatrix":
        return self._zip(other, lambda a, b: b + a)

    def __sub__(self, other) -> "ChebMatrix":
        return self._zip(other, lambda a, b: a - b)

    def __rsub__(self, other) -> "ChebMatrix":
        return self._zip(other, lambda a, b: b - a)

    def __neg__(self) -> "ChebMatrix":
        return self.cellfun(lambda a: -a)

    # ------------------------------------------------------------------
    # Block algebra (MATLAB @chebmatrix/mtimes, mpower, identity, iszero)
    # ------------------------------------------------------------------

    def __mul__(self, other):
        """Block composition ``A*B``, scalar scaling, or application.

        - ``A * c`` (scalar) scales every block.
        - ``A * B`` (ChebMatrix) composes block-wise:
          ``C[i][j] = sum_k A[i][k] * B[k][j]``.
        - ``A * u`` where ``u`` is a list/ChebMatrix of Chebfuns and scalars
          applies the operator, returning a ChebMatrix of the results.

        Provenance
        ----------
        MATLAB source : @chebmatrix/mtimes.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, (int, float, complex)):
            return self.cellfun(lambda blk: other * blk)

        from chebfunjax.chebfun1d.chebfun import Chebfun
        if isinstance(other, Chebfun) and self.ncols == 1:
            other = ChebMatrix([[other]], domain=self.domain)
        elif isinstance(other, (list, tuple)):
            other = ChebMatrix([[it] for it in other], domain=self.domain)

        if not isinstance(other, ChebMatrix):
            return NotImplemented

        if self.ncols != other.nrows:
            raise ValueError(
                "CHEBFUN:CHEBMATRIX:mtimes:dims -- operand inner dimensions "
                f"must agree ({self.ncols} vs {other.nrows}).")

        out: list[list] = []
        for i in range(self.nrows):
            row: list = []
            for j in range(other.ncols):
                acc = self.blocks[i][0] * other.blocks[0][j]
                for k in range(1, self.ncols):
                    acc = acc + self.blocks[i][k] * other.blocks[k][j]
                row.append(acc)
            out.append(row)
        if all(isinstance(blk, (int, float, complex))
               or (hasattr(blk, "ndim")
                   and getattr(blk, "ndim", 1) == 0)
               for row in out for blk in row):
            # MATLAB: chebmatrix operations resulting only in doubles
            # return a normal matrix (tests/chebmatrix/
            # test_matrixOutput.m).
            return jnp.asarray([[complex(blk) if isinstance(
                blk, complex) else float(blk) for blk in row]
                for row in out])
        return ChebMatrix(out, domain=self.domain)

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.cellfun(lambda blk: other * blk)
        return NotImplemented

    def __matmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, k: int) -> "ChebMatrix":
        """Repeated block composition ``A^k``.

        Provenance
        ----------
        MATLAB source : @chebmatrix/mpower.m
        Chebfun commit: 7574c77
        """
        if not (isinstance(k, int) and k >= 0):
            raise ValueError(
                f"ChebMatrix power must be a non-negative integer, got {k!r}.")
        result = self.identity()
        for _ in range(k):
            result = result * self
        return result

    def is_fun_variable(self) -> list[bool]:
        """For each block-row, whether the corresponding variable is a
        function (``True``) or a scalar (``False``).

        Provenance
        ----------
        MATLAB source : @chebmatrix/isFunVariable.m
        Chebfun commit: 7574c77
        """
        col_sizes = self.block_sizes()[0]
        return [sz[1] == float("inf") for sz in col_sizes]

    @property
    def is_not_diff_or_int(self):
        """Per-block flags: True where the block involves no
        differentiation or integration (MATLAB ``isNotDiffOrInt``).
        Chebfun and scalar blocks are multiplication-like, hence True;
        operator/functional blocks carry the flag through their algebra.

        Provenance
        ----------
        MATLAB source : @chebmatrix/chebmatrix.m (isNotDiffOrInt)
        Chebfun commit: 7574c77
        """
        return [[bool(getattr(blk, "isnotdiffint", True))
                 for blk in row] for row in self.blocks]

    def change_tech(self, tech) -> "ChebMatrix":
        """Convert every chebfun block to the given tech
        ('trigtech' or 'chebtech2'); scalars pass through, and blocks
        already carrying the target tech are returned unchanged.

        Provenance
        ----------
        MATLAB source : @chebmatrix/changeTech.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import Chebfun
        from chebfunjax.chebfun1d.chebfun import chebfun as _cf
        name = getattr(tech, "__name__", str(tech)).lower()
        want_trig = "trig" in name

        def conv(blk):
            if not isinstance(blk, Chebfun):
                return blk
            cur = type(blk.funs[0].tech).__name__.lower()
            if ("trig" in cur) == want_trig:
                return blk
            dom = tuple(float(v) for v in blk.domain.breakpoints)
            return _cf(lambda x: blk(x), domain=dom, trig=want_trig)

        return self.cellfun(conv)

    def identity(self) -> "ChebMatrix":
        """Identity ChebMatrix matching this one's variable structure.

        Provenance
        ----------
        MATLAB source : @chebmatrix/identity.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun1d.chebfun import Chebfun
        from chebfunjax.operators.blocks import (
            I,
            _as_domain_obj,
            zero_functional,
            zeros_op,
        )
        if self.nrows != self.ncols:
            raise ValueError(
                "CHEBFUN:CHEBMATRIX:identity:notSquare -- ChebMatrix must "
                "be square.")
        isfun = self.is_fun_variable()
        d = self.domain
        n = self.nrows
        out: list[list] = []
        for i in range(n):
            row: list = []
            for j in range(n):
                if i == j:
                    row.append(I(d) if isfun[i] else 1.0)
                elif isfun[i]:
                    row.append(zeros_op(d) if isfun[j]
                               else Chebfun.from_function(
                                   lambda t: 0.0 * t,
                                   domain=_as_domain_obj(d)))
                else:
                    row.append(zero_functional(d) if isfun[j] else 0.0)
            out.append(row)
        return ChebMatrix(out, domain=d)

    def iszero(self):
        """Matrix of zero-block flags (1 where the block is a zero block).

        Provenance
        ----------
        MATLAB source : @chebmatrix/iszero.m
        Chebfun commit: 7574c77
        """
        return [[1 if getattr(blk, "iszero", False) else 0 for blk in row]
                for row in self.blocks]

    def block_sizes(self):
        """Per-block ``(rows, cols)`` sizes, with ``inf`` for function
        dimensions (MATLAB ``blockSizes``).

        Provenance
        ----------
        MATLAB source : @chebmatrix/blockSizes.m
        Chebfun commit: 7574c77
        """
        inf = float("inf")
        out = []
        for row in self.blocks:
            srow = []
            for blk in row:
                if isinstance(blk, OperatorBlock):
                    srow.append((inf, inf))
                elif isinstance(blk, FunctionalBlock):
                    srow.append((1.0, inf))
                elif isinstance(blk, (int, float, complex)):
                    srow.append((1.0, 1.0))
                else:
                    srow.append((inf, 1.0))
            out.append(srow)
        return out

    def times(self, other) -> "ChebMatrix":
        """Elementwise product (MATLAB times)."""
        return self._zip(other, lambda a, b: a * b)

    def norm(self) -> float:
        """Frobenius-style norm over scalar/chebfun blocks
        (MATLAB norm(A)): sqrt(sum of squared block L2 norms).

        Provenance
        ----------
        MATLAB source : @chebmatrix/norm.m
        Chebfun commit: 7574c77
        """
        import numpy as _np
        total = 0.0
        for row in self.blocks:
            for blk in row:
                if isinstance(blk, (int, float)):
                    total += float(blk) ** 2
                elif hasattr(blk, "norm"):
                    total += float(blk.norm()) ** 2
                else:
                    raise TypeError(
                        "norm: operator blocks have no norm")
        return float(_np.sqrt(total))

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.blocks[idx[0]][idx[1]]
        r = [blk for row in self.blocks for blk in row]
        return r[idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple):
            self.blocks[idx[0]][idx[1]] = value
        else:
            i, j = divmod(idx, self.ncols)
            self.blocks[i][j] = value

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def matrix(
        self,
        n: int,
        domain: _DomainT | None = None,
    ) -> tuple[jnp.ndarray, list[int]]:
        """Assemble the full discretization matrix at size ``n``.

        Each block in the grid is discretized using a
        :class:`~chebfunjax.operators.blocks.ChebColloc2Disc` with ``n``
        collocation points and the physical ``domain``.

        - An ``OperatorBlock`` contributes an ``n x n`` sub-matrix.
        - A ``FunctionalBlock`` contributes a ``1 x n`` row.
        - A scalar ``c`` (``int`` or ``float``) contributes ``c * eye(n)``
          for an operator position, or ``c * ones(n)`` for a functional
          position.  (The caller must know which is which from context.)

        Parameters
        ----------
        n : int
            Number of collocation points.
        domain : (float, float) or None
            Override the domain for this call.  Defaults to ``self.domain``.

        Returns
        -------
        A : jnp.ndarray
            The assembled matrix.  Its shape is ``(total_rows, total_cols)``
            where ``total_rows = n * n_op_rows + n_func_rows`` (in a
            1-column system: ``total_rows = n * n_op_rows + n_func_rows``).
        row_sizes : list[int]
            For each block-row, the number of actual matrix rows contributed
            (``n`` for an OperatorBlock, 1 for a FunctionalBlock / scalar).

        Notes
        -----
        For a standard single-unknown BVP, ``self.ncols == 1``.  The
        assembled matrix has one column block of width ``n``.
        """
        dom = domain if domain is not None else self.domain
        disc = ChebColloc2Disc(n, dom)
        nn = disc.n

        # Determine total number of matrix rows and columns
        # by inspecting each block-row's type.
        row_sizes: list[int] = []
        for block_row in self.blocks:
            block = block_row[0]  # inspect first block in row
            if isinstance(block, FunctionalBlock):
                row_sizes.append(1)
            elif isinstance(block, OperatorBlock):
                row_sizes.append(nn)
            elif isinstance(block, (int, float, complex)):
                # Scalar: assumed to be a scalar (functional-style row)
                row_sizes.append(1)
            else:
                # A Chebfun block maps a scalar to a function.
                row_sizes.append(nn)

        # Build full matrix
        rows: list[jnp.ndarray] = []
        for block_row, rsize in zip(self.blocks, row_sizes):
            col_parts: list[jnp.ndarray] = []
            for block in block_row:
                if isinstance(block, OperatorBlock):
                    part = block.matrix(disc)  # shape (nn, nn)
                elif isinstance(block, FunctionalBlock):
                    row_vec = block.matrix(disc)   # shape (nn,)
                    part = row_vec[None, :]         # shape (1, nn)
                elif isinstance(block, (int, float, complex)):
                    c = block if isinstance(block, complex) else float(block)
                    part = jnp.asarray([[c]]) * jnp.ones(
                        (rsize, 1), dtype=jnp.float64)
                else:
                    vals = jnp.ravel(jnp.asarray(block(disc.points()),
                                                 dtype=jnp.float64))
                    part = vals[:, None] if rsize == nn else vals[None, :]
                col_parts.append(part)

            # Concatenate column parts horizontally
            row_mat = jnp.concatenate(col_parts, axis=1)
            rows.append(row_mat)

        A = jnp.concatenate(rows, axis=0)
        return A, row_sizes

    def dense(self, n, domain: _DomainT | None = None) -> jnp.ndarray:
        """Discretize this ChebMatrix as an ordinary matrix.

        Parameters
        ----------
        n : int or sequence of int
            Collocation dimension; one entry per subinterval when the
            domain has breakpoints.
        domain : tuple of float or None
            Override the domain (used to introduce breakpoints).

        Returns
        -------
        jnp.ndarray
            The assembled dense matrix.

        Examples
        --------
        >>> from chebfunjax.operators.blocks import D, I
        >>> A = ChebMatrix([[I((0.0, 1.0)), D((0.0, 1.0))]])
        >>> A.dense(5).shape
        (5, 10)

        Provenance
        ----------
        MATLAB source : @chebmatrix/matrix.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        matrix
        """
        return self.matrix(n, domain=domain)[0]

    # ------------------------------------------------------------------
    # Convenience: solve BVP
    # ------------------------------------------------------------------

    def solve(
        self,
        rhs: jnp.ndarray,
        n: int,
        bc_values: list[float],
        bc_row_indices: list[int] | None = None,
        domain: _DomainT | None = None,
    ) -> jnp.ndarray:
        """Solve the linear system defined by this ChebMatrix.

        Assembles the matrix, replaces the last ``len(bc_values)`` rows with
        the boundary-condition rows (FunctionalBlock rows), and solves via
        ``jnp.linalg.solve``.

        This is a convenience method for standard BVP setups where:

        - The first block-row is an ``OperatorBlock`` (the differential
          operator).
        - The remaining block-rows are ``FunctionalBlock`` (boundary conditions).

        The system is assembled as::

            [  L_interior  ]       [  f_interior  ]
            [  bc1_row     ]  x =  [  bc_val_1    ]
            [  bc2_row     ]       [  bc_val_2    ]

        where the interior rows of ``L`` are the interior ``n - n_bc`` rows.

        Parameters
        ----------
        rhs : jnp.ndarray, shape (n,) or (n - n_bc,)
            Right-hand side values at collocation points (interior only).
            Must have length ``n`` (the caller provides the full RHS; the
            method truncates to the interior rows).
        n : int
            Number of collocation points.
        bc_values : list[float]
            Values for the boundary conditions (one per FunctionalBlock row).
        bc_row_indices : list[int] or None
            Which rows of the interior ``L`` to replace with BC rows.
            Default: replace the last ``len(bc_values)`` rows.
        domain : (float, float) or None
            Override domain.

        Returns
        -------
        u_vals : jnp.ndarray, shape (n,)
            Solution values at the collocation points.
        """
        n_bc = len(bc_values)
        dom = domain if domain is not None else self.domain

        # Assemble raw matrix (shape: total_rows x n)
        A_raw, row_sizes = self.matrix(n, domain=dom)

        # The operator block's rows come first (row_sizes[0] = n rows).
        # BC rows follow (each 1 row).
        # We need an n x n square system.
        # Strategy: take first (n - n_bc) interior rows of the operator, then
        # append the n_bc BC rows.
        n_op_rows = row_sizes[0]
        if n_op_rows != n:
            raise ValueError(
                f"ChebMatrix.solve: first block must be an OperatorBlock "
                f"contributing {n} rows, got {n_op_rows}."
            )

        # Interior rows from the operator block
        if bc_row_indices is None:
            bc_row_indices = list(range(n - n_bc, n))

        # Build square matrix: replace bc_row_indices with BC rows
        op_matrix = A_raw[:n, :]  # (n, n) — operator rows

        # Extract BC rows from A_raw (after the op rows)
        bc_rows_list = []
        bc_rhs_list = []
        offset = n  # skip the n operator rows
        for idx, bc_val in zip(range(n_bc), bc_values):
            bc_row = A_raw[offset + idx, :]   # shape (n,)
            bc_rows_list.append(bc_row)
            bc_rhs_list.append(bc_val)

        # Build the square system
        A_sq = op_matrix
        f_full = jnp.asarray(rhs, dtype=jnp.float64)
        if f_full.shape[0] == n - n_bc:
            # Pad with BC values
            f_full = jnp.concatenate(
                [f_full, jnp.zeros(n_bc, dtype=jnp.float64)]
            )

        for row_idx, (bc_row, bc_val) in enumerate(
            zip(bc_rows_list, bc_rhs_list)
        ):
            replace_at = bc_row_indices[row_idx]
            A_sq = A_sq.at[replace_at, :].set(bc_row)
            f_full = f_full.at[replace_at].set(float(bc_val))

        return jnp.linalg.solve(A_sq, f_full)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def _plot_impl(self, coeffs: bool):
        import matplotlib.pyplot as plt

        from chebfunjax.chebfun1d.chebfun import Chebfun
        m, n = self.size
        fig, axes = plt.subplots(m, n, squeeze=False)
        for i in range(m):
            for j in range(n):
                ax = axes[i][j]
                blk = self.blocks[i][j]
                if isinstance(blk, Chebfun):
                    a, b = (float(blk.domain.breakpoints[0]),
                            float(blk.domain.breakpoints[-1]))
                    xs = jnp.linspace(a, b, 200)
                    if coeffs:
                        c = jnp.abs(jnp.ravel(jnp.asarray(
                            blk.funs[0].tech.coeffs))) + 1e-300
                        ax.semilogy(jnp.arange(c.size), c, ".")
                    else:
                        ax.plot(xs, jnp.asarray(blk(xs)))
                elif isinstance(blk, (int, float, complex)):
                    ax.text(0.5, 0.5, str(blk), ha="center",
                            va="center")
                    ax.set_axis_off()
                else:
                    # Operator/functional block: show its 10-point
                    # discretization matrix (MATLAB draws blocks by
                    # their realizations).
                    M = jnp.atleast_2d(jnp.asarray(blk.matrix(10)))
                    ax.imshow(jnp.abs(M), aspect="auto")
        return fig

    def plot(self, *args, **kwargs):
        """Plot every block on a subplot grid (chebfun blocks as line
        plots, operator blocks by their discretization matrices,
        scalars as text).

        Provenance
        ----------
        MATLAB source : @chebmatrix/plot.m
        Chebfun commit: 7574c77
        """
        return self._plot_impl(coeffs=False)

    def plotcoeffs(self, *args, **kwargs):
        """Coefficient plots of the chebfun blocks (MATLAB
        ``plotcoeffs``).

        Provenance
        ----------
        MATLAB source : @chebmatrix/plotcoeffs.m
        Chebfun commit: 7574c77
        """
        return self._plot_impl(coeffs=True)

    def _log_plot_guard(self, name: str):
        from chebfunjax.chebfun1d.chebfun import Chebfun
        for row in self.blocks:
            for blk in row:
                if not isinstance(blk, (Chebfun, int, float, complex)):
                    raise ValueError(
                        f"{name} plot of infinite blocks is not "
                        "supported.")

    def loglog(self, *args, **kwargs):
        """MATLAB ``loglog`` of a chebmatrix; raises for operator
        (infinite) blocks exactly as MATLAB does.

        Provenance
        ----------
        MATLAB source : @chebmatrix/loglog.m
        Chebfun commit: 7574c77
        """
        self._log_plot_guard("loglog")
        return self._plot_impl(coeffs=False)

    def semilogx(self, *args, **kwargs):
        """MATLAB ``semilogx`` of a chebmatrix (see :meth:`loglog`).

        Provenance
        ----------
        MATLAB source : @chebmatrix/semilogx.m
        Chebfun commit: 7574c77
        """
        self._log_plot_guard("semilogx")
        return self._plot_impl(coeffs=False)

    def waterfall(self, t=None, **kwargs):
        """Waterfall plot of a chebmatrix of chebfun snapshots.

        Provenance
        ----------
        MATLAB source : @chebmatrix/waterfall.m
        Chebfun commit: 7574c77
        """
        import matplotlib.pyplot as plt
        import numpy as _onp  # uses-numpy: concrete plotting grids

        from chebfunjax.chebfun1d.chebfun import Chebfun
        funs = [blk for row in self.blocks for blk in row
                if isinstance(blk, Chebfun)]
        if not funs:
            raise ValueError("waterfall: no chebfun blocks.")
        a, b = (float(funs[0].domain.breakpoints[0]),
                float(funs[0].domain.breakpoints[-1]))
        xs = _onp.linspace(a, b, 200)
        ts = (_onp.arange(len(funs), dtype=float) if t is None
              else _onp.asarray(t, dtype=float))
        X, T = _onp.meshgrid(xs, ts)
        Z = _onp.vstack([_onp.asarray(f(jnp.asarray(xs)))
                         for f in funs])
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        surf_kw = {}
        lw = kwargs.get("LineWidth", kwargs.get("linewidth"))
        if lw is not None:
            surf_kw["linewidth"] = float(lw)
        fc = kwargs.get("FaceColor", kwargs.get("facecolor"))
        if fc is not None:
            surf_kw["color"] = fc
        fa = kwargs.get("FaceAlpha", kwargs.get("alpha"))
        if fa is not None:
            surf_kw["alpha"] = float(fa)
        ax.plot_surface(X, T, Z, **surf_kw)
        return ax

    def __repr__(self) -> str:
        block_types = []
        for row in self.blocks:
            row_types = [type(b).__name__ for b in row]
            block_types.append(row_types)
        return (
            f"ChebMatrix({self.nrows}x{self.ncols}, "
            f"domain={self.domain}, blocks={block_types})"
        )


# ===========================================================================
# Private helpers
# ===========================================================================


def _infer_domain(blocks: list[list[_Block]]) -> _DomainT:
    """Infer the physical domain as the merge of every block's breakpoints."""
    bps: set[float] = set()
    ends: tuple[float, float] | None = None
    for row in blocks:
        for block in row:
            dom = getattr(block, "domain", None)
            if dom is None:
                continue
            if hasattr(dom, "breakpoints"):
                dom = dom.breakpoints
            vals = tuple(float(v) for v in dom)
            bps.update(vals)
            if ends is None:
                ends = (vals[0], vals[-1])
    if ends is None:
        return _DEFAULT_DOMAIN
    return tuple(sorted(v for v in bps if ends[0] <= v <= ends[1]))
