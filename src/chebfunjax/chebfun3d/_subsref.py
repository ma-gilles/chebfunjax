"""Chebfun3 indexing, property panels, and required composition adapters.

Provenance
----------
MATLAB source: @chebfun3/{subsref,get,feval}.m, @chebfun2v/compose.m,
    @spherefunv/compose.m, isSubset.m; Chebfun commit 7574c77.
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers.
Numeric property indexing uses one-based, column-major source conventions.
Recursive indexing of arbitrary foreign objects/cells is not implemented.
"""
import math

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.chebfun1d.mtimes import _columns

from ._mtimes import _native_squeeze, _panel
from ._plane import scalar_plane_product


def is_colon(value):
    return (isinstance(value, str) and value == ':') or (
        isinstance(value, slice) and value == slice(None))


def _numeric(value):
    if is_colon(value):
        return False
    try:
        return jnp.asarray(value).dtype.kind in 'iufc'
    except (TypeError, ValueError):
        return False


def _matlab_shape(array):
    array = jnp.asarray(array)
    shape = list(array.shape)
    while len(shape) > 2 and shape[-1] == 1:
        shape.pop()
    while len(shape) < 2:
        shape.insert(0, 1)
    return array.reshape(shape)


def source_get(f, name):
    if name in ('cols', 'rows', 'tubes'):
        k = ('cols', 'rows', 'tubes').index(name)
        return _panel(getattr(f, name), tuple(f.domain[2*k:2*k+2]))
    if name == 'core':
        return _matlab_shape(f.core)
    if name == 'domain':
        return jnp.asarray(f.domain).reshape((1, -1))
    raise ValueError(f'CHEBFUN:CHEBFUN3:get:propName: {name} is not a valid CHEBFUN3 property.')


def _native_indices(value, length):
    if is_colon(value):
        return jnp.arange(length)
    a = jnp.asarray(value)
    if a.dtype.kind == 'b':
        if a.size > length:
            raise IndexError('Logical index exceeds property dimensions.')
        return jnp.where(a.reshape(-1, order='F'))[0]
    if a.dtype.kind not in 'iuf' or not bool(jnp.all(jnp.isfinite(a) & (a == jnp.floor(a)) & (a >= 1) & (a <= length))):
        raise IndexError('Property indices must be positive integers within dimensions.')
    return a.astype(jnp.int64).reshape(-1, order='F')-1


def _numeric_subsref(value, subs):
    a = _matlab_shape(value)
    if not subs:
        raise IndexError('Empty numeric property index list.')
    if len(subs) == 1:
        flat = a.reshape(-1, order='F')
        out = flat[_native_indices(subs[0], flat.size)]
        return out.reshape(()) if not is_colon(subs[0]) and jnp.ndim(subs[0]) == 0 else out
    shape = list(a.shape)
    if len(subs) < len(shape):
        shape = shape[:len(subs)-1]+[math.prod(shape[len(subs)-1:])]
    shape += [1]*max(0, len(subs)-len(shape))
    a = a.reshape(shape, order='F')
    return a[jnp.ix_(*[_native_indices(v, n) for v, n in zip(subs, shape)])]


def _following(value, records):
    for position, record in enumerate(records):
        kind, subs = record['type'], record['subs']
        if hasattr(value, 'subsref'):
            return value.subsref(records[position:])
        if isinstance(value, (Chebfun, Quasimatrix)):
            if kind == '()':
                if len(subs) == 1 and _numeric(subs[0]):
                    value = value(subs[0])
                elif len(subs) == 2 and is_colon(subs[0]):
                    columns = _columns(value)
                    ix = _native_indices(subs[1], len(columns))
                    value = Chebfun.horzcat(*[columns[int(i)] for i in ix])
                else:
                    raise NotImplementedError('Recursive factor indexing supports evaluation and (:,columns).')
            elif kind == '.' and isinstance(value, Chebfun):
                value = value.get(subs)
            else:
                raise NotImplementedError('Recursive factor indexing does not support this record.')
        elif kind == '()' and _numeric(value):
            value = _numeric_subsref(value, subs)
        else:
            raise NotImplementedError('Recursive indexing of this property object is not implemented.')
    return value


def source_subsref(f, index):
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
    from chebfunjax.spherefun.spherefunv import Spherefunv

    from .chebfun3 import Chebfun3
    from .chebfun3v import Chebfun3v
    records = [index] if isinstance(index, dict) else list(index)
    first = records[0]
    kind, subs = first['type'], first['subs']
    if kind == '()':
        if len(subs) == 3:
            if all(_numeric(v) or is_colon(v) for v in subs):
                return f.feval(*subs)
            if all(isinstance(v, (Chebfun, Quasimatrix)) for v in subs):
                return Chebfun.horzcat(*subs).compose(f)
            if all(isinstance(v, Chebfun2) for v in subs):
                return Chebfun2v([v.approx for v in subs]).compose(f)
            if all(isinstance(v, Chebfun3) for v in subs):
                return Chebfun3v(*subs).compose(f)
            raise ValueError('CHEBFUN:CHEBFUN3:subsref:inputs3: Unrecognized inputs.')
        # Native non-triple branch dispatches on the first input, even when
        # further inputs are present. Empty subs retains an indexing error.
        if isinstance(subs[0], (Chebfun, Quasimatrix, Chebfun2v, Chebfun3v, Spherefunv)):
            return subs[0].compose(f)
        raise ValueError('CHEBFUN:CHEBFUN3:subsref:inputs: Can only evaluate at triples '
                         '(X,Y,Z), a CHEBFUN with 3 columns, a CHEBFUN2V or a CHEBFUN3V.')
    if kind == '.':
        return _following(f.get(subs), records[1:])
    if kind == '{}':
        if len(subs) != 6:
            raise ValueError('CHEBFUN:CHEBFUN3:subsref:dimensions: Index exceeds chebfun3 dimensions.')
        return f.restrict(jnp.concatenate([jnp.asarray(v).reshape(-1, order='F') for v in subs]))
    raise ValueError('Unknown source indexing record type.')


def source_colon_feval(f, x, y, z):
    args = (x, y, z)
    free = tuple(is_colon(v) for v in args)
    if all(free):
        return f
    panels = [source_get(f, name) for name in ('cols', 'rows', 'tubes')]
    vals = [None if flag else jnp.asarray(panels[k](args[k])).reshape((-1, len(_columns(panels[k]))))
            for k, flag in enumerate(free)]
    if any(v is not None and v.shape[0] != 1 for v in vals):
        raise NotImplementedError('Colon slices currently require scalar fixed coordinates; vector-fixed source cases remain open.')
    rx, ry, rz = f.rank
    if free == (False, True, True):
        core = _native_squeeze(f.txm(f.core, vals[0], 1))
        if ry == 1 or rz == 1:
            core = core.T
        return scalar_plane_product(panels[2], core.T, panels[1])
    if free == (True, False, True):
        core = _native_squeeze(f.txm(f.core, vals[1], 2))
        if rx == 1:
            core = core.T
        return scalar_plane_product(panels[2], core.T, panels[0])
    if free == (True, True, False):
        core = _native_squeeze(f.txm(f.core, vals[2], 3))
        return scalar_plane_product(panels[1], core.T, panels[0])
    if free == (True, False, False):
        core = f.txm(f.txm(f.core, vals[1], 2), vals[2], 3)
        out = panels[0] @ core.reshape((core.shape[0], 1))
    elif free == (False, True, False):
        core = f.txm(f.txm(f.core, vals[0], 1), vals[2], 3)
        # The source uses conjugate transpose in this branch.
        out = panels[1] @ jnp.conj(core.reshape((1, core.shape[1]))).T
    else:
        core = _native_squeeze(f.txm(f.txm(f.core, vals[0], 1), vals[1], 2))
        out = panels[2] @ core
    if isinstance(out, Chebfun):
        out = out.simplify()
    return _columns(out)[0].extract_columns(0)


def _check_image(components, outer, tolerance, identifier):
    # minandmax2est samples each component at the native default33 grid.
    ranges = []
    for c in components:
        values = c.sample(33, 33)
        ranges.append(jnp.asarray([jnp.min(values), jnp.max(values)]))
    if len(ranges)*2 != len(outer.domain):
        raise ValueError('CHEBFUN:isSubset:size: Domains must have the same number of entries.')
    if any(bool(v[0] < outer.domain[2*k]-tolerance) or bool(outer.domain[2*k+1]+tolerance < v[1])
           for k, v in enumerate(ranges)):
        raise ValueError(identifier+': OP(F) is not defined, since image(F) is not contained in domain(OP).')


def compose_chebfun2v_three(inner, outer):
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    from chebfunjax.chebpref import ChebfunPref

    from ._power import _source_vscale
    cs = [Chebfun2(approx=c) for c in inner.components]
    real = all(not jnp.iscomplexobj(c.pivot_values)
               and _panel(c.approx.cols, c.domain[2:4]).isreal()
               and _panel(c.approx.rows, c.domain[:2]).isreal() for c in cs)
    if not real:
        raise ValueError('CHEBFUN:CHEBFUN2V:COMPOSE:Complex: The first CHEBFUN2V object must be real-valued.')
    tol = 100*ChebfunPref().cheb2Prefs.chebfun2eps*max(*(c.vscale() for c in cs), float(_source_vscale(outer)))*max(abs(v) for v in cs[0].domain)
    _check_image(cs, outer, tol, 'CHEBFUN:CHEBFUN2V:COMPOSE:DomainMismatch3')
    return Chebfun2.from_function(lambda x, y: outer(*(c(x, y) for c in cs)),
                                  domain=cs[0].domain, trig=all(c.isPeriodicTech() for c in cs))


def check_sphere_composition(inner, outer):
    from chebfunjax.chebpref import ChebfunPref

    from ._power import _source_vscale
    cs = inner.components
    tol = 100*ChebfunPref().cheb2Prefs.chebfun2eps*max(*(c.vscale() for c in cs), float(_source_vscale(outer)))*math.pi
    _check_image(cs, outer, tol, 'CHEBFUN:SPHEREFUNV:COMPOSE:DomainMismatch3')
