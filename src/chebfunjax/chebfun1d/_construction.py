"""Public construction control flow from Chebfun7574c77.

Source @chebfun/chebfun.m vectorCheck/vec and
@chebfun/getValuesAtBreakpoints.m. Eager Python owns callback order;
numerical values and arithmetic use JAX. Public wiring is a separate draft.
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp


def _matrix(value, *, scalar_input=False, sample_count=None):
    """Explicit Python rank-one convention at the source shape boundary.

    Matching sample count means column; nonmatching means constant row.
    Scalar-input rank-one means row. Ambiguous constant length two needs an
    explicit two-dimensional row at the native two-point probe.
    """
    value = jnp.asarray(value)
    if value.ndim == 0:
        return value.reshape(1, 1)
    if value.ndim == 1:
        return value[None, :] if scalar_input or (sample_count is not None and value.size != sample_count) else value[:, None]
    return value


def _expand(op):
    def expanded(x):
        x = jnp.asarray(x)
        length = max(x.shape) if x.ndim else 1
        value = _matrix(op(x), scalar_input=x.ndim == 0, sample_count=length)
        result = jnp.tile(value, (length, 1))
        return result[:, 0] if result.shape[1] == 1 else result
    return expanded


def _transpose(op):
    def transposed(x):
        value = _matrix(op(x), scalar_input=jnp.ndim(x) == 0)
        result = value.T
        return result[:, 0] if result.shape[1] == 1 else result
    warnings.warn('CHEBFUN:CHEBFUN:vectorCheck:transpose: '
                  'Chebfun input should return a COLUMN array.\n'
                  'Attempting to transpose.', stacklevel=3)
    return transposed


def _vec(op, point):
    """Native scalar probe followed by scalar/array explicit loop wrapper."""
    sample = jnp.asarray(op(point))
    if any(size > 1 for size in sample.shape):
        def array_wrapper(x):
            x = jnp.asarray(x)
            flat = jnp.ravel(x, order='F')
            first = _matrix(op(flat[0]), scalar_input=True)
            columns = first.shape[1]
            rows = x.shape[0] if x.ndim else 1
            values = [jnp.asarray(op(flat[j])).reshape(columns) for j in range(rows)]
            result = jnp.stack(values)
            # Native zeros default to double; complex assignment promotes.
            return result.astype(jnp.result_type(result, jnp.float64))
        return array_wrapper

    def scalar_wrapper(x):
        x = jnp.asarray(x)
        flat = jnp.ravel(x, order='F')
        values = [jnp.asarray(op(point)).reshape(()) for point in flat]
        if not values:
            return jnp.zeros(x.shape, dtype=jnp.float64)
        result = jnp.stack(values).reshape(x.shape, order='F')
        return result.astype(jnp.result_type(result, jnp.float64))
    return scalar_wrapper


def vector_check(op, domain, vectorize=False):
    """Literal near-endpoint vectorCheck with one source retry boundary."""
    ends = jnp.asarray([domain[0], domain[-1]], dtype=jnp.float64)
    y = ends + jnp.asarray([1., -1.]) * jnp.diff(ends) / 200
    if vectorize:
        op = _vec(op, y[0])
    try:
        value = _matrix(op(y), sample_count=2)
        shape = value.shape
        if shape[0] == 2:
            if shape[1] == 2:
                scalar = _matrix(op(y[0]), scalar_input=True)
                if scalar.shape[0] > 1:
                    op = _transpose(op)
                elif scalar.shape[1] != shape[1]:
                    raise ValueError('CHEBFUN:CHEBFUN:vectorCheck:numColumns: '
                                     'Number of columns increases with length(x).')
        elif all(size == 1 for size in shape):
            op = _expand(op)
        elif any(size == 2 for size in shape):
            # Source says any(sv)==1 (not any(sv==1)); preserve that quirk.
            if any(shape) == 1:
                scalar = _matrix(op(y[0]), scalar_input=True)
                if scalar.shape == shape:
                    return _expand(op)
            if _matrix(op(y), sample_count=2).shape[0] > 1:
                op = _transpose(op)
            else:
                op = vector_check(op, domain, True)
        elif any(size == 1 for size in shape):
            op = _expand(op)
    except Exception:
        if vectorize:
            raise
        op = vector_check(op, domain, True)
    return op


def _default_double(values):
    values = jnp.asarray(values)
    return values.astype(jnp.complex128 if jnp.iscomplexobj(values) else jnp.float64)


def endpoint_limit(fun, right=False):
    """Native classicfun/onefun endpoint dispatch without physical mapping.

    @deltafun/{lval,rval} unwrap funPart; @classicfun unwrap onefun.
    @chebtech endpoints use signed/plain coefficient sums; generic onefun
    (including Singfun and Trigtech) evaluates at the exact reference end.
    @trigtech/get.m33–38 delegates to feval/horner; it does not use cached
    values even if those differ from the coefficient interpolant.
    """
    from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

    if hasattr(fun, 'funPart'):
        return endpoint_limit(fun.funPart, right)
    onefun = fun.onefun if hasattr(fun, 'onefun') else fun.tech
    if isinstance(onefun, (Chebtech1, Chebtech2)):
        coeffs = onefun.coeffs
        if not right:
            coeffs = coeffs.at[1::2].set(-coeffs[1::2])
        return jnp.sum(coeffs, axis=0)
    return onefun(jnp.asarray(1. if right else -1.))


def values_at_breakpoints(funs, ends, op=None):
    """Source callable assignment, then only NaN replacement by limits."""
    if len(funs) == 1 and funs[0].tech.isempty():
        return jnp.empty((0, 0))
    coeffs = funs[0].tech.coeffs
    columns = coeffs.shape[1] if coeffs.ndim == 2 else 1

    def limit(index):
        if index == 0:
            return jnp.atleast_1d(endpoint_limit(funs[0]))
        if index == len(funs):
            return jnp.atleast_1d(endpoint_limit(funs[-1], True))
        return (jnp.atleast_1d(endpoint_limit(funs[index-1], True))
                + jnp.atleast_1d(endpoint_limit(funs[index])))/2

    if op is None or isinstance(op, (list, tuple)) or not callable(op):
        return _default_double(jnp.stack([limit(index) for index in range(len(funs)+1)]))
    values = _matrix(op(jnp.asarray(ends)))
    # Source assignment scalar-expands, but does not generally broadcast
    # arbitrary singleton rows/columns into a differently shaped slice.
    shape = (len(funs)+1, columns)
    if values.size == 1:
        values = jnp.full(shape, values.reshape(()))
    elif values.shape != shape:
        raise ValueError('Breakpoint operator values do not match FUN dimensions.')
    values = _default_double(values)
    mask = jnp.isnan(values)
    for index in range(len(funs)+1):
        if bool(jnp.any(mask[index])):
            replacement = limit(index)
            # MATLAB assignment may promote the original real array.
            values = values.astype(jnp.result_type(values, replacement))
            values = values.at[index].set(jnp.where(mask[index], replacement, values[index]))
    return values


class _Omitted:
    """Distinguish an omitted public argument from an explicit value."""


OMITTED = _Omitted()


def prepare_preferences(domain=OMITTED, pref=None, *, operand_domain=None,
                        keywords=None):
    """Private source pref/domain merge before operator normalization.

    This is a draft context primitive, not a replacement for all parseInputs
    modifiers. Remaining modifier ownership is documented in PARSE_BOUNDARY.
    """
    from chebfunjax.chebpref import ChebfunPref

    options = dict(keywords or {})
    private = ChebfunPref(pref) if pref is not None else ChebfunPref()
    # Select actual Tech before reading the lazy default view.
    if options.get('tech') is not None:
        private.tech = options['tech']
    if options.get('chebkind') is not None:
        kind = str(options['chebkind']).lower()
        if kind in ('1', '1st', 'first'):
            private.tech = 'chebtech1'
        elif kind in ('2', '2nd', 'second'):
            private.tech = 'chebtech2'
        else:
            raise ValueError('CHEBFUN:CHEBFUN:parseInputs:badChebkind')
    fields = {'n': 'fixedLength', 'eps': 'chebfuneps', 'max_length': 'maxLength',
              'min_samples': 'minSamples', 'turbo': 'useTurbo',
              'extrapolate': 'extrapolate', 'sample_test': 'sampleTest',
              'refinement_function': 'refinementFunction'}
    for name, target in fields.items():
        if name in options and options[name] is not None:
            setattr(private, target, options[name])
    for name in ('splitting', 'blowup'):
        if name in options and options[name] is not None:
            setattr(private, name, options[name])
    if options.get('equi'):
        private.enableFunqui = True
    if options.get('trunc') is not None:
        private.splitting = True
    if options.get('resampling'):
        private.refinementFunction = 'resampling'
    if domain is OMITTED or domain is None or jnp.asarray(domain).size == 0:
        domain = operand_domain if operand_domain is not None else private.domain
    points = tuple(float(x) for x in domain)
    if options.get('doubleLength'):
        if private.splitting:
            raise ValueError('CHEBFUN:CHEBFUN:parseInputs:doubleLengthSplitting')
        if len(points) > 2:
            raise ValueError('CHEBFUN:CHEBFUN:parseInputs:doubleLengthBreakpoints')
    explicit_periodic = bool(options.get('trig') or options.get('periodic'))
    if explicit_periodic:
        private.tech = 'trigtech'
        private.splitting = False
        private.enableFunqui = False
        if len(points) > 2:
            raise ValueError('CHEBFUN:parseInputs:periodic')
    hscale = jnp.max(jnp.abs(jnp.asarray(points)))
    hscale = jnp.where(jnp.isinf(hscale), 1., hscale)
    return private, points, {'hscale': hscale, 'vscale': 0.}, explicit_periodic
