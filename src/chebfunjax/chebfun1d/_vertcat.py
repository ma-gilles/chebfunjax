"""Source public vertical concatenation; pin7574c77 vertcat/num2cell.

Python one-dimensional numeric inputs represent columns. Empty operand
lists are rejected because the pinned dispatch requires a Chebfun operand.
"""

import warnings

import jax.numpy as jnp


def source_vertcat(operands):
    """Assemble scalar blocks or apply the native row concatenation branch.

    Provenance
    ----------
    MATLAB source : @chebfun/vertcat.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d._horzcat import _merged_domain
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.chebfun1d.linalg import Quasimatrix
    from chebfunjax.operators.chebmatrix import ChebMatrix

    operands = list(operands)
    functions = [
        f
        for item in operands
        for f in (
            item.cols
            if isinstance(item, Quasimatrix)
            else [item]
            if isinstance(item, Chebfun)
            else []
        )
    ]
    if not functions:
        raise ValueError("vertcat requires a Chebfun operand.")
    row = functions[0].is_transposed
    if any(f.is_transposed != row for f in functions):
        raise ValueError(
            "CHEBFUN:CHEBFUN:vertcat:transpose: Dimensions of matrices being concatenated are not consistent."
        )

    def columns(item):
        return item.cols if isinstance(item, Quasimatrix) else item.mat2cell()

    split = any(
        (
            sum(f.n_columns for f in item.cols)
            if isinstance(item, Quasimatrix)
            else item.n_columns
            if isinstance(item, Chebfun)
            else jnp.asarray(item).size
        )
        > 1
        for item in operands
    )
    if row and not split:
        transposed = [
            item.T if isinstance(item, (Chebfun, Quasimatrix)) else jnp.asarray(item).T
            for item in operands
        ]
        return Chebfun.horzcat(*transposed).ctranspose()
    blocks = []
    for item in operands:
        if isinstance(item, (Chebfun, Quasimatrix)):
            cols = columns(item)
            blocks.extend([[col] for col in cols] if row else [cols])
        else:
            values = jnp.asarray(item)
            if values.ndim > 2:
                raise ValueError("vertcat numeric operands must be matrices.")
            values = values.reshape((-1, 1)) if values.ndim < 2 else values
            blocks.extend(
                [[values[i, j] for j in range(values.shape[1])] for i in range(values.shape[0])]
            )
    if split:
        warnings.warn(
            "CHEBFUN:CHEBFUN:vertcat:join: Vertical concatenation of CHEBFUN objects now produces a CHEBMATRIX.",
            UserWarning,
            stacklevel=3,
        )
    return ChebMatrix(blocks, domain=_merged_domain(functions))
