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

        # Determine total number of matrix rows and columns
        # by inspecting each block-row's type.
        row_sizes: list[int] = []
        for block_row in self.blocks:
            block = block_row[0]  # inspect first block in row
            if isinstance(block, FunctionalBlock):
                row_sizes.append(1)
            elif isinstance(block, OperatorBlock):
                row_sizes.append(n)
            elif isinstance(block, (int, float)):
                # Scalar: assumed to be a scalar (functional-style row)
                row_sizes.append(1)
            else:
                raise TypeError(
                    f"ChebMatrix: unsupported block type {type(block).__name__}."
                )

        sum(row_sizes)
        self.ncols * n

        # Build full matrix
        rows: list[jnp.ndarray] = []
        for bi, (block_row, rsize) in enumerate(zip(self.blocks, row_sizes)):
            col_parts: list[jnp.ndarray] = []
            for bj, block in enumerate(block_row):
                if isinstance(block, OperatorBlock):
                    part = block.matrix(disc)  # shape (n, n)
                elif isinstance(block, FunctionalBlock):
                    row_vec = block.matrix(disc)   # shape (n,)
                    part = row_vec[None, :]         # shape (1, n)
                elif isinstance(block, (int, float)):
                    c = float(block)
                    if rsize == n:
                        part = c * jnp.eye(n, dtype=jnp.float64)
                    else:
                        part = c * jnp.ones((1, n), dtype=jnp.float64)
                else:
                    raise TypeError(
                        f"ChebMatrix: unsupported block type {type(block).__name__}."
                    )
                col_parts.append(part)

            # Concatenate column parts horizontally
            row_mat = jnp.concatenate(col_parts, axis=1)
            rows.append(row_mat)

        A = jnp.concatenate(rows, axis=0)
        return A, row_sizes

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
    """Try to infer the physical domain from the first block with a domain."""
    for row in blocks:
        for block in row:
            if hasattr(block, "domain"):
                return block.domain
    return _DEFAULT_DOMAIN
