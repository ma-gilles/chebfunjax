"""Continuous Chebfun norms and MATLAB's optional second output.

Provenance
----------
MATLAB source : @chebfun/norm.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
The public return_location keyword represents requesting MATLAB's second
output. Array 1-norm column indices retain MATLAB's one-based convention.
Continuous extrema reuse the existing piecewise extremum implementation.
"""
from numbers import Real

import jax.numpy as jnp


def _selector(p):
    if p is None:
        return 'fro'
    if isinstance(p, str):
        if p == 'fro':
            return p
        if p in ('inf', '-inf'):
            return float(p)
    elif isinstance(p, Real):
        return float(p)
    else:
        value = jnp.asarray(p)
        if value.ndim == 0 and jnp.issubdtype(value.dtype, jnp.number) and not jnp.iscomplexobj(value):
            return float(value)
    raise ValueError("CHEBFUN:CHEBFUN:norm:unknownNorm: "
                     "The only matrix norms available are 1, 2, inf, and 'fro'.")


def continuous_norm(f, p=None, *, return_location=False):
    """Literal scalar/array branches of MATLAB @chebfun/norm.m.

    Provenance
    ----------
    MATLAB source : @chebfun/norm.m
    Chebfun commit: 7574c77
    """
    quasi = hasattr(f, 'cols')
    if not quasi and f.isempty():
        result = jnp.asarray(0.)
        return (result, None) if return_location else result
    p = _selector(p)
    if quasi:
        from .mtimes import _columns
        columns = _columns(f)
    else:
        oriented = f.T if f.is_transposed else f
        columns = [oriented] if oriented.n_columns == 1 else oriented.mat2cell()
    count = len(columns)
    loc = None
    if count == 1:
        column = columns[0]
        if p not in (jnp.inf, -jnp.inf) and return_location:
            description = ('1-norms' if p == 1 else "'fro'-norms" if p in (2, 'fro') else 'p-norms.')
            raise ValueError('CHEBFUN:CHEBFUN:norm:argout: Cannot return two outputs for '+description)
        if p == 1:
            value = column.abs().sum()
        elif p in (2, 'fro'):
            value = jnp.sqrt(jnp.abs(jnp.reshape(column.inner(column), ())))
        elif p == jnp.inf:
            if column.isreal():
                low, high = column.minandmax()
                candidates = jnp.asarray([low[1], high[1]])
                index = jnp.argmax(jnp.abs(candidates))
                value = jnp.abs(candidates[index])
                loc = jnp.asarray([low[0], high[0]])[index]
            else:
                loc, value = (column.conj()*column).max()
                value = jnp.sqrt(jnp.asarray(value, dtype=jnp.complex128))
        elif p == -jnp.inf:
            loc, value = (column.conj()*column).min()
            value = jnp.sqrt(jnp.asarray(value, dtype=jnp.complex128))
        elif p % 2 == 0:
            value = ((column.conj()*column)**(p/2)).sum()**(1/p)
        else:
            value = (column.abs()**p).sum()**(1/p)
    elif p == 1:
        integrals = jnp.stack([continuous_norm(c, 1) for c in columns])
        index = jnp.argmax(integrals)
        value, loc = integrals[index], index+1
    elif p in (2, 'fro'):
        if return_location:
            name = '2' if p == 2 else "'fro'"
            raise ValueError('CHEBFUN:CHEBFUN:norm:argout: Cannot return two outputs for '
                             +name+'-norms of array-valued CHEBFUNs.')
        gram = jnp.stack([jnp.stack([a.inner(b) for b in columns]) for a in columns])
        value = (jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh(gram)[-1], 0)) if p == 2
                 else jnp.sqrt(jnp.abs(jnp.trace(gram))))
    else:
        row = columns[0].abs()
        if p not in (jnp.inf, -jnp.inf):
            row = row**p
        for column in columns[1:]:
            piece = column.abs()
            row = row+(piece if p in (jnp.inf, -jnp.inf) else piece**p)
        loc, value = row.min() if p == -jnp.inf else row.max()
        if p not in (jnp.inf, -jnp.inf):
            value = value**(1/p)
    value = jnp.real(jnp.asarray(value))
    return (value, loc) if return_location else value
