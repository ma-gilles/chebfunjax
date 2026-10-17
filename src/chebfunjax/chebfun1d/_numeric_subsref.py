"""Builtin numeric-result indexing after native Chebfun records.

@chebfun/subsref.m (7574c77) delegates following records to the result's
SUBSREF. Shapes/errors below are bound by the retained R2025b API probe.
Index records are static for JIT; numeric result values remain traced.
"""
import math
import numbers

import jax
import jax.numpy as jnp


def _error(identifier, message):
    from .chebfun import FevalError
    raise FevalError(identifier, message)


def _shape(array):
    dims = list(array.shape)
    while len(dims) > 2 and dims[-1] == 1:
        dims.pop()
    if not dims:
        dims = [1, 1]
    elif len(dims) == 1:
        dims = [1, dims[0]]
    return array.reshape(dims)


def _vector(shape):
    return len(shape) == 2 and (shape[0] == 1 or shape[1] == 1)


def _index(value, limit):
    if isinstance(value, str):
        if value == ':':
            return jnp.arange(limit), (limit, 1), 'colon'
        array = jnp.asarray([ord(c) for c in value]).reshape(1, -1)
    else:
        # Explicit object/text arrays model nonnumeric indexing types. Do not
        # invoke conversion-only foreign objects to manufacture numeric data.
        from ._subsref import _numeric_query
        def numeric_leaves(item):
            if isinstance(item, (list, tuple)):
                return all(numeric_leaves(v) for v in item)
            return isinstance(item, (numbers.Number, bool))
        logical_array = (
            isinstance(value, (jax.Array, jax.core.Tracer)) or
            type(value).__module__ == 'numpy') and getattr(getattr(value, 'dtype', None), 'kind', None) == 'b'
        if not _numeric_query(value) and not (
                isinstance(value, bool) or logical_array or
                isinstance(value, (list, tuple)) and numeric_leaves(value)):
            _error('MATLAB:matrix:unableToUseTypeAsIndex',
                   'Unable to use this type as an index.')
        array = _shape(jnp.asarray(value))
    if jnp.iscomplexobj(array):
        _error('MATLAB:badsubscript', 'Array indices must be positive integers or logical values.')
    if jnp.issubdtype(array.dtype, jnp.bool_):
        flat = array.reshape(-1, order='F')
        positions = jnp.nonzero(flat, size=int(jnp.sum(flat)))[0]
        if bool(jnp.any(positions >= limit)):
            _error('MATLAB:badsubscript', 'Logical index contains a true value outside array bounds.')
        return positions, array.shape, 'logical'
    valid = (jnp.isfinite(array) & (array >= 1) &
             (array == jnp.floor(array)) & (array <= limit))
    if not bool(jnp.all(valid)):
        _error('MATLAB:badsubscript', 'Array indices must be positive integers within array bounds.')
    return array.astype(jnp.int64).reshape(-1, order='F')-1, array.shape, 'numeric'


def source_numeric_subsref(value, records):
    """Apply native builtin records to a numeric/logical result, not a Chebfun."""
    array = _shape(jnp.asarray(value))
    if not records:
        return array
    record = records[0]
    kind = record['type']
    # Builtin SUBSREF validates this record-chain grammar before evaluation.
    if len(records) > 1 and kind == '()' and records[1]['type'] != '.':
        _error('MATLAB:subsDotMustFollow',
               'A dot indexing expression must follow a parentheses indexing expression.')
    if kind == '.':
        _error('MATLAB:structRefFromNonStruct', 'Dot indexing is not supported for this type.')
    if kind == '{}':
        _error('MATLAB:cellRefFromNonCell', 'Brace indexing is not supported for this type.')
    if kind != '()':
        raise NotImplementedError('Unqualified builtin numeric record type.')
    subs = list(record['subs'])
    if not subs:
        return source_numeric_subsref(array, records[1:]) if len(records) > 1 else array
    with jax.ensure_compile_time_eval():
        if len(subs) == 1:
            positions, index_shape, mode = _index(subs[0], array.size)
            count = positions.size
            if mode == 'colon':
                out_shape = (count, 1)
            elif _vector(array.shape) and _vector(index_shape):
                out_shape = (1, count) if array.shape[0] == 1 else (count, 1)
            elif mode == 'logical':
                if index_shape == (1, 1) and not count:
                    out_shape = (0, 0)
                elif _vector(index_shape):
                    out_shape = (1, count) if index_shape[0] == 1 else (count, 1)
                else:
                    out_shape = (count, 1)
            else:
                out_shape = index_shape
            shape = None
        else:
            shape = list(array.shape)
            if len(subs) < len(shape):
                shape = shape[:len(subs)-1] + [math.prod(shape[len(subs)-1:])]
            shape += [1]*max(0, len(subs)-len(shape))
            positions = [_index(index, size)[0] for index, size in zip(subs, shape)]
            out_shape = None
    if shape is None:
        out = jnp.take(array.reshape(-1, order='F'), positions).reshape(out_shape, order='F')
    else:
        out = array.reshape(shape, order='F')[jnp.ix_(*positions)]
    out = _shape(out)
    if len(records) > 1:
        return source_numeric_subsref(out, records[1:])
    return out
