"""Native Chebfun indexing records, @chebfun/subsref.m (7574c77)."""

import jax.numpy as jnp


def _colon(value):
    return isinstance(value, str) and value == ':'


def source_subsref(f, index):
    from .chebfun import FevalError

    records = [index] if isinstance(index, dict) else list(index)
    if not records:
        return f
    record = records[0]
    kind, subs = record['type'], record['subs']

    def fail(suffix, message):
        raise FevalError('CHEBFUN:CHEBFUN:subsref:' + suffix, message)

    if kind == '{}':
        # A Python list denotes a native row cell. Explicit ndarray shape
        # preserves column-cell syntax, including the source dimensions error.
        if hasattr(subs, 'shape') and hasattr(subs, 'ndim'):
            row = subs.ndim < 2 or subs.shape[0] == 1
            if getattr(getattr(subs, 'dtype', None), 'kind', None) == 'O':
                # Object-cell reshape changes structure only; never perform
                # numerical NumPy dispatch on its contents.
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
        side, columns = None, slice(None)
        if len(args) == 2:
            col = args[1]
            if isinstance(col, str) and col.lower() in ('left', 'right', '-', '+'):
                side = col
            elif not _colon(col):
                values = jnp.asarray(col)
                if values.ndim > 2 or (values.ndim == 2 and min(values.shape) > 1):
                    fail('colidx', 'Column index must be a vector of integers.')
                if values.size and jnp.max(values) > obj.n_columns:
                    fail('badsubscript', 'Index exceeds CHEBFUN dimensions.')
                columns = values.astype(int).reshape(-1, order='F') - 1
        if _colon(x):
            out = obj[:, columns]
        elif isinstance(x, str):
            fail('nonnumeric', 'Cannot evaluate chebfun for non-numeric type.')
        else:
            # __call__ performs source typed composition before numeric FEVAL.
            out = obj(x, side=side) if isinstance(columns, slice) else obj(x, side=side, columns=columns)
        if transposed:
            out = out.T if hasattr(out, 'funs') else (
                jnp.swapaxes(out, 0, 1) if out.ndim >= 2 else out)
    else:
        fail('unexpectedType', 'Unexpected index.type of ' + kind)

    if len(records) > 1 and kind in ('()', '.'):
        if not hasattr(out, 'subsref'):
            raise NotImplementedError('Recursive indexing of numeric results is unsupported.')
        return out.subsref(records[1:])
    return out
