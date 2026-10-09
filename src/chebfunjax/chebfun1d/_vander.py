"""Native Vandermonde and private column-only concatenation.

Provenance
----------
MATLAB: @chebfun/vander.m, horzcat.m, hscale.m, issing.m,
@classicfun/horzcat.m, @chebtech/horzcat.m, @trigtech/horzcat.m.
Chebfun commit7574c77680d7e82b79626300bf255498271a72df.
Public legacy Chebfun.horzcat and row/ChebMatrix concatenation are separate.
"""
import math

import jax.numpy as jnp

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech, _trig_prolong_coeffs


def _column_matrix(values):
    return values[:, None] if values.ndim == 1 else values


def _source_tech_columns(techs):
    """Source prolongation, first-Tech metadata, then coefficient columns."""
    first = techs[0]
    size = max(t.coeffs.shape[0] for t in techs)
    if all(isinstance(t, (Chebtech1, Chebtech2)) for t in techs):
        columns = [_column_matrix(t.prolong(size).coeffs) for t in techs]
        return type(first)(coeffs=jnp.concatenate(columns, axis=1),
                           ishappy=first.ishappy)
    if all(isinstance(t, Trigtech) for t in techs):
        columns = [_column_matrix(_trig_prolong_coeffs(t.coeffs, size))
                   for t in techs]
        # Trigtech's existing Python representation has one aggregate flag;
        # retaining full complex coefficients preserves every column value.
        return type(first)(coeffs=jnp.concatenate(columns, axis=1),
                           is_real=all(t.is_real for t in techs),
                           ishappy=first.ishappy)
    raise ValueError('CHEBFUN:CHEBTECH:horzcat:typeMismatch: '
                     'Incompatible concatenation technologies.')


def source_column_horzcat(inputs):
    """Column branch of native horzcat; no overlap or assign-columns append.

    Returns an array-valued Chebfun when source storage collates, otherwise
    a Quasimatrix retaining separate scalar columns. This private entry does
    not claim public numeric/row/ChebMatrix concatenation support.
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.chebfun1d.linalg import Quasimatrix

    inputs = list(inputs)
    nonempty = [f for f in inputs if not f.isempty()]
    if not nonempty:
        return inputs[0]
    if len(nonempty) == 1:
        return nonempty[0]
    if any(f.is_transposed for f in nonempty):
        raise ValueError('CHEBFUN:CHEBFUN:horzcat:transpose: '
                         'Private source concatenation requires columns.')
    first = nonempty[0]
    ends = (first.domain.a, first.domain.b)
    if any((f.domain.a, f.domain.b) != ends for f in nonempty):
        raise ValueError('CHEBFUN:CHEBFUN:horzcat:domains: Inconsistent domains.')
    domains = [f.domain.breakpoints for f in nonempty]
    different = any(len(d) != len(domains[0]) for d in domains)
    if not different:
        scales = [max(abs(f.domain.a), abs(f.domain.b)) for f in nonempty]
        scales = [1. if math.isinf(v) else v for v in scales]
        tolerance = max(scales) * float(jnp.finfo(jnp.float64).eps)
        # Preserve the literal source any(d-domain1)>tol, including its
        # boolean comparison and infinite-endpoint subtraction behavior.
        domain0 = jnp.asarray(domains[0])
        different = any(bool(jnp.any(jnp.asarray(d)-domain0) > tolerance)
                        for d in domains)
    singular = any(isinstance(piece.tech, Singfun)
                   for f in nonempty for piece in f.funs)
    delta = any(bool(f.deltas) for f in nonempty)
    periodic = [all(isinstance(piece.tech, Trigtech) for piece in f.funs)
                for f in nonempty]
    if different or singular or delta or (any(periodic) and not all(periodic)):
        columns = [col for f in nonempty
                   for col in (f.mat2cell() if f.n_columns > 1 else [f])]
        return Quasimatrix(columns, first.domain)
    pieces = [piece.with_tech(_source_tech_columns(
                  [f.funs[k].tech for f in nonempty]))
              for k, piece in enumerate(first.funs)]
    values = jnp.concatenate([_column_matrix(f.point_values)
                              for f in nonempty], axis=1)
    return Chebfun(funs=pieces, domain=first.domain).set_point_values(values)


def source_vander(f, n):
    """Follow constant initialization, sequential times, horzcat, fliplr."""
    from chebfunjax.chebfun1d.chebfun import chebfun

    if f.is_transposed or f.n_columns > 1:
        raise ValueError('CHEBFUN:CHEBFUN:vander:row: '
                         'Input must be a scalar-valued column CHEBFUN.')
    # Python adapter for MATLAB cell(1,n) size: no truncation of positive
    # fractional counts and no n>=1 restriction. Integer negative counts,
    # like zero, initialize an empty cell then A{1} expands it.
    dimension = jnp.asarray(n)
    if dimension.size != 1:
        raise ValueError('Vandermonde column count must be a scalar.')
    value = float(dimension.reshape(()))
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError('Vandermonde column count must be a finite integer.')
    count = int(value)
    columns = [chebfun(1., domain=f.domain.breakpoints)]
    for _ in range(1, count):
        columns.append(f * columns[-1])
    return source_column_horzcat(columns).fliplr()
