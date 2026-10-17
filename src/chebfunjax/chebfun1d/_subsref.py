"""Native Chebfun indexing records, @chebfun/subsref.m (7574c77).

Record numeric queries use MATLAB row/column shape (rank-one Python arrays
stand for source rows); direct Python evaluation/index contracts are unchanged.
Column selectors are static for JIT. Numeric query values remain traced.
"""

import numbers

import jax
import jax.numpy as jnp


def _colon(value):
    return isinstance(value, str) and value == ':'


def _error(identifier, message):
    from .chebfun import FevalError
    raise FevalError(identifier, message)


def _matlab_array(value):
    array = jnp.asarray(value)
    shape = list(array.shape)
    while len(shape) > 2 and shape[-1] == 1:
        shape.pop()
    if not shape:
        shape = [1, 1]
    elif len(shape) == 1:
        shape = [1, shape[0]]
    return array.reshape(shape)


def _max_source(values):
    # MATLAB MAX reduces the first nonsingleton dimension and omits NaNs.
    # The complex magnitude/phase rule is also used by source strictCheck.
    axis = next((i for i, size in enumerate(values.shape) if size != 1), 0)
    if not jnp.iscomplexobj(values):
        return jnp.nanmax(values, axis=axis, keepdims=True)
    valid = ~jnp.isnan(values)
    magnitude = jnp.where(valid, jnp.abs(values), -jnp.inf)
    tied = valid & (magnitude == jnp.max(magnitude, axis=axis, keepdims=True))
    phase = jnp.where(tied, jnp.angle(values), -jnp.inf)
    winner = jnp.argmax(phase, axis=axis, keepdims=True)
    return jnp.take_along_axis(values, winner, axis=axis)


def _if_true(value):
    return value.size != 0 and bool(jnp.all(value))


def _source_columns(selector, ncols):
    default = jnp.arange(1, ncols + 1, dtype=jnp.float64).reshape(1, -1)
    if _colon(selector):
        return default, None
    if isinstance(selector, str):
        key = selector.lower()
        if key in ('left', 'right', '-', '+'):
            return default, selector
        # Character codes are source index metadata, not numeric query data.
        values = _matlab_array([ord(c) for c in selector])
    else:
        values = _matlab_array(selector)
    if values.size == 0:
        return values, None
    maximum = _max_source(values)
    # MATLAB relational comparisons of complex values use the real part.
    maximum = jnp.real(maximum)
    if _if_true(maximum <= ncols):
        columns = values
        if columns.ndim > 2:
            raise NotImplementedError('Higher-rank source column selectors are unqualified.')
        if columns.shape[1] == 1:
            columns = jnp.swapaxes(columns, 0, 1)
        if columns.shape[0] > 1:
            _error('CHEBFUN:CHEBFUN:subsref:colidx',
                   'Column index must be a vector of integers.')
        return columns, None
    if _if_true(maximum > ncols):
        _error('CHEBFUN:CHEBFUN:subsref:badsubscript',
               'Index exceeds CHEBFUN dimensions.')
    # Literal source retains all columns when neither condition is true.
    return default, None


def _numeric_positions(indices, limit):
    real = jnp.real(indices)
    valid = jnp.isfinite(real) & (real >= 1) & (real == jnp.floor(real)) & (real <= limit)
    if jnp.iscomplexobj(indices):
        valid = valid & (jnp.imag(indices) == 0)
    if not bool(jnp.all(valid)):
        _error('MATLAB:badsubscript',
               'Array indices must be positive integers or logical values.')
    return real.astype(jnp.int64).reshape(-1, order='F') - 1


def _colon_positions(columns, ncols):
    flat = columns.reshape(-1, order='F')
    if flat.size == ncols and bool(jnp.all(flat == jnp.arange(1, ncols + 1))):
        return None
    # Native raw indexing rejects complex storage, even zero imaginary part;
    # computed numeric-x outCols have different arithmetic semantics.
    if jnp.iscomplexobj(flat):
        _error('MATLAB:badsubscript',
               'Array indices must be positive integers or logical values.')
    if jnp.issubdtype(flat.dtype, jnp.bool_):
        positions = jnp.nonzero(flat, size=int(jnp.sum(flat)))[0]
        if positions.size and bool(jnp.any(positions >= ncols)):
            _error('MATLAB:badsubscript', 'Logical index exceeds CHEBFUN dimensions.')
        return positions
    return _numeric_positions(flat, ncols)


def _numeric_index(obj, x, columns, side):
    query = _matlab_array(x)
    # Native SUBSREF evaluates the complete array before selecting output
    # columns. Empty FEVAL retains query shape and skips component packing.
    full = obj._feval(query, side=side)
    with jax.ensure_compile_time_eval():
        width = query.shape[1]
        base = (columns.reshape(1, -1) - 1.) * width + 1.
        offsets = jnp.arange(width, dtype=base.dtype).reshape(-1, 1)
        out_cols = (base + offsets).reshape(-1, order='F')
        positions = _numeric_positions(out_cols, full.shape[1])
    return jnp.take(full, positions, axis=1)


def _function_operand(x):
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3
    from chebfunjax.diskfun.diskfun import Diskfun
    from chebfunjax.spherefun.spherefun import Spherefun

    from .chebfun import Chebfun
    from .linalg import Quasimatrix
    return isinstance(x, (Chebfun, Quasimatrix, Chebfun2, Chebfun3, Diskfun, Spherefun))


def _numeric_query(x):
    # Native ISNUMERIC excludes logicals, cells and conversion-only objects.
    # Lists/tuples are the Python numeric-array adapter, not MATLAB cells.
    if isinstance(x, bool):
        return False
    if isinstance(x, numbers.Number):
        return True
    if isinstance(x, (list, tuple)):
        def numeric_leaves(value):
            if isinstance(value, (list, tuple)):
                return all(numeric_leaves(item) for item in value)
            return isinstance(value, numbers.Number)
        if not numeric_leaves(x):
            return False
        try:
            dtype = jnp.asarray(x).dtype
        except (TypeError, ValueError):
            return False
    elif isinstance(x, (jax.Array, jax.core.Tracer)):
        dtype = x.dtype
    elif type(x).__module__ == 'numpy' and type(x).__name__ == 'ndarray':
        # Inspect host-array metadata without importing NumPy or invoking an
        # arbitrary object's __array__/__jax_array__ conversion protocol.
        dtype = x.dtype
    else:
        return False
    return jnp.issubdtype(dtype, jnp.number)


def source_subsref(f, index):
    records = [index] if isinstance(index, dict) else list(index)
    if not records:
        return f
    record = records[0]
    kind, subs = record['type'], record['subs']

    def fail(suffix, message):
        _error('CHEBFUN:CHEBFUN:subsref:' + suffix, message)

    if kind == '{}':
        if hasattr(subs, 'shape') and hasattr(subs, 'ndim'):
            row = subs.ndim < 2 or subs.shape[0] == 1
            if getattr(getattr(subs, 'dtype', None), 'kind', None) == 'O':
                args = list(subs.reshape(-1, order='F'))
            else:
                args = list(jnp.asarray(subs).reshape(-1, order='F'))
        else:
            row, args = True, list(subs)
        if len(args) == 1:
            if not _colon(args[0]):
                fail('badDomain', 'Invalid domain syntax.')
            out = f
        elif row:
            ends = jnp.concatenate([jnp.asarray(v).reshape(-1) for v in args])
            out = f.restrict(ends).simplify()
        else:
            fail('dimensions', 'Index exceeds chebfun dimensions.')
    elif kind == '.':
        out = f.get(subs)
    elif kind == '()':
        args = list(subs)
        if not args or len(args) > 2:
            fail('dimensions', 'Index exceeds CHEBFUN dimensions.')
        transposed = f.is_transposed
        obj = f.T if transposed else f
        if transposed and len(args) > 1:
            args[0], args[1] = args[1], args[0]
        x = args[0]
        with jax.ensure_compile_time_eval():
            columns, side = _source_columns(args[1] if len(args) == 2 else ':', obj.n_columns)
        if _numeric_query(x):
            out = _numeric_index(obj, x, columns, side)
        elif _function_operand(x):
            # Source ignores columnIndex in the typed composition branch.
            out = obj(x, side=side)
        elif _colon(x):
            with jax.ensure_compile_time_eval():
                positions = _colon_positions(columns, obj.n_columns)
            out = obj if positions is None else obj[:, positions]
        else:
            fail('nonnumeric', 'Cannot evaluate chebfun for non-numeric type.')
        if transposed:
            out = out.T if hasattr(out, 'funs') else (
                jnp.swapaxes(out, 0, 1) if out.ndim >= 2 else out)
    else:
        fail('unexpectedType', 'Unexpected index.type of ' + kind)

    if len(records) > 1 and kind in ('()', '.'):
        if hasattr(out, 'subsref'):
            return out.subsref(records[1:])
        # GET's public Python tuple/vector conventions are preserved. Only
        # the native record boundary restores their source numeric shapes;
        # FUNS and unresolved local-property lists remain cells/objects.
        if kind == '.' and subs in ('domain', 'ends'):
            out = jnp.asarray(out).reshape(1, -1)
        elif kind == '.' and subs in ('lval', 'rval') and jnp.ndim(out) <= 1:
            out = jnp.asarray(out).reshape((-1, 1) if f.is_transposed else (1, -1))
        if isinstance(out, (numbers.Number, bool, jax.Array, jax.core.Tracer)):
            from ._numeric_subsref import source_numeric_subsref
            return source_numeric_subsref(out, records[1:])
        raise NotImplementedError('Recursive indexing of cell/foreign results is unsupported.')
    return out
