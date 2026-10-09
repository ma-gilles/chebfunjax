"""Public horizontal concatenation adapters.

Provenance: @chebfun/horzcat.m, vertcat.m, constructor.m, num2cell.m;
@chebmatrix/vertcat.m, chebmatrix.m, mergeDomains.m; @domain/merge.m.
Chebfun commit7574c77680d7e82b79626300bf255498271a72df.
"""
import warnings

import jax.numpy as jnp

from chebfunjax.chebfun1d._vander import source_column_horzcat


def _numeric(value):
    """Python shape adapter: a one-dimensional numeric operand is a row."""
    value = jnp.asarray(value)
    if value.ndim > 2:
        raise ValueError('horzcat numeric operands must be matrices.')
    return value.reshape((1, -1)) if value.ndim < 2 else value


def _merged_domain(functions):
    """Literal mergeDomains/domain.merge predicates, including signed ends."""
    domains = [jnp.asarray(f.domain.breakpoints) for f in functions]
    first = domains[0]
    if all(d.shape == first.shape and bool(jnp.all(d == first))
           for d in domains):
        return tuple(float(x) for x in first)
    scales = jnp.asarray([jnp.max(jnp.abs(d)) for d in domains])
    scales = jnp.where(jnp.isinf(scales), 1., scales)
    tol = 100 * jnp.finfo(jnp.float64).eps * jnp.max(scales)
    if any(bool(jnp.any(d[jnp.array([0, -1])] -
                            first[jnp.array([0, -1])] > tol)) for d in domains):
        raise ValueError('CHEBFUN:DOMAIN:merge:incompat: Incompatible domains.')
    out = []
    while domains:
        point = jnp.min(jnp.asarray([jnp.min(d) for d in domains]))
        out.append(float(point))
        if bool(jnp.isposinf(point)):
            break
        domains = [d[~(jnp.isneginf(d) | (d < point + tol))] for d in domains]
        domains = [d for d in domains if d.size]
    return tuple(out)


def _row_horzcat(inputs, functions):
    """Transpose, scalar-block vertcat, then native block ctranspose.

    Native ctranspose executes conjugation of entries despite its comment.
    This narrow adapter does not change public ChebMatrix.T or vertcat.
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.operators.chebmatrix import ChebMatrix

    blocks = []
    split = False
    for item in inputs:
        if isinstance(item, Chebfun):
            columns = item.T.mat2cell() if item.n_columns > 1 else [item.T]
            split |= len(columns) > 1
            blocks.extend(col.ctranspose() for col in columns)
        else:
            values = _numeric(item)
            # After transpose this operand must have one block column,
            # matching the transposed Chebfun's column of scalar blocks.
            if values.shape[0] != 1:
                raise ValueError('CHEBFUN:CHEBMATRIX:vertcat:sizeMismatch: '
                                 'Incompatible column sizes.')
            split |= values.size > 1
            blocks.extend(jnp.conj(value) for value in values[0])
    if split:
        warnings.warn('CHEBFUN:CHEBFUN:vertcat:join: Vertical concatenation '
                      'of CHEBFUN objects now produces a CHEBMATRIX.',
                      UserWarning, stacklevel=3)
    return ChebMatrix([blocks], domain=_merged_domain(functions))


def source_horzcat(operands):
    """Native public branch order; list/1D numeric conventions are adapters."""
    from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
    from chebfunjax.chebfun1d.linalg import Quasimatrix

    operands = list(operands)
    if not operands:
        return Chebfun.empty()

    def empty(item):
        if isinstance(item, Chebfun):
            return item.isempty()
        if isinstance(item, Quasimatrix):
            return not item.cols
        return jnp.asarray(item).size == 0

    active = [item for item in operands if not empty(item)]
    if not active:
        return operands[0]
    if len(active) == 1:
        return active[0]
    # Native quasimatrix elements are themselves Chebfun objects.
    functions = [f for item in active for f in
                 (item.cols if isinstance(item, Quasimatrix) else
                  [item] if isinstance(item, Chebfun) else [])]
    if not functions:
        raise ValueError('horzcat requires a Chebfun operand to infer its domain.')
    first = functions[0]
    if any(f.is_transposed != first.is_transposed for f in functions):
        raise ValueError('CHEBFUN:CHEBFUN:horzcat:transpose: '
                         'Dimensions of matrices being concatenated are not consistent.')
    flat = [f for item in active for f in
            (item.cols if isinstance(item, Quasimatrix) else [item])]
    if first.is_transposed:
        return _row_horzcat(flat, functions)
    promoted = []
    for item in flat:
        if isinstance(item, Chebfun):
            promoted.append(item)
        else:
            promoted.append(chebfun(_numeric(item),
                                    domain=first.domain.breakpoints))
    return source_column_horzcat(promoted)
