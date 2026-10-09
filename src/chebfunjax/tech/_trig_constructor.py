"""Source Trig constructor, refinement and happiness orchestration.

MATLAB @trigtech/{trigtech,populate,refine,happinessCheck,classicCheck,
plateauCheck,sampleTest,parseDataInputs,techPref}.m, Chebfun7574c77.
Copyright 2017 by The University of Oxford and The Chebfun Developers.

Python owns eager callback/shape control. All numerical operations are JAX.
Public Chebfun routing and compose integration are separate pending work.
"""
from __future__ import annotations

import copy
import math
import warnings

import jax
import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref, _factory_tech_prefs

EPS = 2.0**-52


def _empty(value):
    return value is None or jnp.asarray(value).size == 0


def resolve_pref(pref=None, *, n=None, maxpow2=None):
    """Native Tech defaults/overlay; object selection is a Python adapter."""
    defaults = dict(_factory_tech_prefs('trigtech'))
    if isinstance(pref, ChebfunPref):
        private = ChebfunPref(pref)
        private.tech = 'trigtech'
        supplied = dict(private.techPrefs)
        explicit = private._tech_overrides
    else:
        supplied = {} if pref is None else copy.deepcopy(dict(pref))
        explicit = supplied
    for key in supplied:
        if key not in defaults:
            warnings.warn("CHEBFUN:TRIGTECH:techPref:unknownPref: "
                          f"Unrecognized input preference '{key}'.", stacklevel=3)
    defaults.update(supplied)
    if maxpow2 is not None and 'maxLength' not in explicit:
        defaults['maxLength'] = 2**maxpow2
    if n is not None:
        defaults['fixedLength'] = n
    return defaults


def parse_data(data=None):
    """Default missing/empty source fields; retain other callback data."""
    result = {} if data is None else copy.deepcopy(dict(data))
    for key, default in [('vscale', 0.), ('hscale', 1.)]:
        if key not in result or _empty(result[key]):
            result[key] = default
    scale = jnp.asarray(result['vscale'])
    if scale.ndim == 2 and scale.shape[0] == 1:
        # Python represents MATLAB row vectors by one-dimensional arrays.
        result['vscale'] = scale[0]
    return result


def spacing(x):
    """MATLAB eps(x), including finite realmax and subnormal spacing.

    Binary64 source adapter; exact exponent/mantissa construction avoids
    nextafter(realmax, Inf)-realmax and subnormal floating arithmetic.
    MathWorks Precision and realmax (2009), documented eps edge table.
    """
    value = jnp.asarray(x)
    if jnp.iscomplexobj(value):
        raise TypeError('eps(hscale) requires real input')
    value = value.astype(jnp.float64)
    bits = jax.lax.bitcast_convert_type(value, jnp.uint64)
    exponent = (bits >> jnp.uint64(52)) & jnp.uint64(2047)
    normal = (exponent - jnp.uint64(52)) << jnp.uint64(52)
    subnormal = jnp.uint64(1) << jnp.maximum(exponent.astype(jnp.int64)-1, 0).astype(jnp.uint64)
    result = jnp.where(exponent > 52, normal, subnormal)
    result = jnp.where(exponent == 2047, jnp.uint64(0x7ff8000000000000), result)
    return jax.lax.bitcast_convert_type(result, jnp.float64)


def _columns(values):
    values = jnp.asarray(values)
    return values[:, None] if values.ndim == 1 else values


def _values(f):
    """Read source cache, or reconstruct by the existing pure JAX transform."""
    from chebfunjax.tech.trigtech import _trig_coeffs2vals_impl, _trig_project_values

    if f._values is not None:
        return f._values
    return _trig_project_values(_trig_coeffs2vals_impl(f.coeffs), f.real_columns)


def _prolong(f, count):
    """Source prolongation with pure JAX coefficient/value helpers."""
    from chebfunjax.tech.trigtech import (
        Trigtech,
        _trig_coeffs2vals_impl,
        _trig_project_values,
        _trig_prolong_coeffs,
    )

    if _empty(count) or count == f.n:
        return f
    numeric = float(jnp.asarray(count).reshape(()))
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise ValueError('fixedLength/cutoff must be a nonnegative integer')
    coeffs = _trig_prolong_coeffs(f.coeffs, int(numeric))
    mask = () if coeffs.shape[0] == 0 else f.real_columns
    values = _trig_project_values(_trig_coeffs2vals_impl(coeffs), mask)
    return Trigtech(coeffs=coeffs, real_columns=mask, ishappy=f.ishappy, _values=values)


def classic_check(f, values, data, pref):
    """Literal classicCheck and happinessRequirements; returns source cutoff."""
    from chebfunjax.tech.trigtech import _trig_coeffs2vals_impl, trigpts

    original_n = _values(f).shape[0]
    coeffs = _columns(f.coeffs)
    m = coeffs.shape[1]
    tolerance = jnp.asarray(pref if not isinstance(pref, dict) else pref['chebfuneps'])
    if tolerance.size == 1:
        tolerance = jnp.full((m,), tolerance.reshape(()))
    elif tolerance.ndim == 2 and tolerance.shape[0] == 1:
        tolerance = tolerance[0]
    if original_n < 2:
        return False, original_n
    scale = jnp.asarray(data['vscale'])
    if bool(jnp.max(scale) == 0):
        return True, 1
    if bool(jnp.any(jnp.isinf(scale))):
        return False, original_n
    scale = jnp.where(scale == 0, EPS, scale)
    if bool(jnp.any(jnp.isnan(coeffs))):
        raise ValueError('CHEBFUN:FUN:classicCheck:NaNeval: Function returned NaN when evaluated.')
    if _empty(values):
        values = _trig_coeffs2vals_impl(f.coeffs)
    values = _columns(values)
    reversed_abs = jnp.abs(coeffs[::-1])
    n = original_n
    if n % 2 == 0:
        folded = jnp.concatenate((reversed_abs[n-1:n],
            reversed_abs[n-2:n//2-1:-1]+reversed_abs[:n//2-1],
            reversed_abs[n//2-1:n//2]), axis=0)
    else:
        folded = jnp.concatenate((reversed_abs[n-1:(n+1)//2-1:-1]
            + reversed_abs[:(n+1)//2-1],
            reversed_abs[(n-1)//2:(n+1)//2]), axis=0)
    n = folded.shape[0]
    ac = jnp.abs(folded) / scale
    test_length = min(n, max(3, math.floor((n-1)/8+0.5)))
    tail_error = min(EPS*test_length, 1e-4)
    # f.points uses original stored values length, not folded coefficient count.
    points = trigpts(original_n)
    dy = jnp.diff(values, axis=0)
    dx = jnp.diff(points)[:, None] * jnp.ones((1, values.shape[1]))
    gradient = jnp.max(jnp.abs(dy/dx), axis=0)
    condition = spacing(data['hscale']) / scale * gradient
    condition = jnp.minimum(condition, 1e-4)
    tolerance = jnp.maximum(jnp.maximum(tolerance, condition), tail_error)
    if not bool(jnp.all(jnp.max(ac[:test_length], axis=0) < tolerance)):
        return False, n
    large = jnp.any(ac >= tolerance, axis=1)
    positions = [j for j in range(n) if bool(large[j])]
    if not positions:
        return True, 1
    tloc = positions[0]
    ac = ac[:tloc]
    running = jnp.full((m,), .25*EPS)
    for k in range(ac.shape[0]):
        row = jnp.where(ac[k] < running, running, ac[k])
        running = jnp.where(row >= running, row, running)
        ac = ac.at[k].set(row)
    bang = jnp.log(1e3 * (tolerance/ac))
    buck = n-jnp.arange(1, tloc+1)[:, None]
    ratio = bang/buck
    if tloc < 3:
        # Native max(empty) and min(empty) produce an empty cutoff. Preserve
        # that value; do not replace it with a fabricated positive count.
        return True, jnp.asarray([], dtype=jnp.float64)
    first_maximum = jnp.argmax(ratio[2:tloc], axis=0)+1
    tchop = int(jnp.min(first_maximum))
    cutoff = n-tchop-2
    return True, 2*cutoff-1


def sample_test(op, f, pref):
    """Native full-interpolant sampleTest through aggregate realness."""
    from chebfunjax.tech.trigtech import _trig_eval

    x = jnp.asarray([-.357998918959666, .036785641195074], dtype=jnp.float64)
    vfun = _columns(_trig_eval(f.coeffs, x, is_real=f.is_real))
    vop = _columns(jnp.asarray(op(x)))
    tolerance = jnp.sqrt(jnp.maximum(jnp.asarray(pref['chebfuneps']), EPS))
    tolerance = tolerance * jnp.max(jnp.max(jnp.abs(_columns(_values(f))), axis=0))
    error = jnp.abs(vop-vfun)
    return bool(jnp.all(jnp.max(error, axis=0) <= tolerance))


def happiness(f, op=None, values=None, data=None, pref=None):
    """Source ordered dispatcher, custom callback, then optional sampleTest."""
    from chebfunjax.tech.trigtech import _trig_coeffs2vals_impl, _trig_standard_check

    pref = resolve_pref() if pref is None else pref
    data = parse_data(data)
    data['vscale'] = jnp.maximum(jnp.asarray(data['vscale']),
                                jnp.max(jnp.abs(_columns(_values(f))), axis=0))
    check = pref['happinessCheck']
    name = check.lower() if isinstance(check, str) else None
    if name == 'standard':
        try:
            standard_values = _trig_coeffs2vals_impl(f.coeffs) if _empty(values) else values
            happy, cutoff = _trig_standard_check(f.coeffs, standard_values, pref['chebfuneps'], data['vscale'])
        except ValueError as error:
            if str(error) != 'Trigtech standardCheck: function returned NaN':
                raise
            raise ValueError('CHEBFUN:TRIGTECH:standardCheck:nanEval: '
                             'Function returned NaN when evaluated.') from error
    elif name in ('classic', 'plateau'):
        happy, cutoff = classic_check(f, values, data, pref)
    elif name in ('strict', 'loose'):
        raise ValueError(f'CHEBFUN:TRIGTECH:happinessCheck:{name}Check: '
                         f'{name.capitalize()} check not implemented for TRIGTECH.  Please use classic check.')
    else:
        happy, cutoff = check(f, values, copy.deepcopy(data), copy.deepcopy(pref))
    if happy and op is not None and callable(op) and pref['sampleTest']:
        happy = sample_test(op, f, pref)
        if not happy:
            cutoff = _values(f).shape[0]
    return happy, cutoff


def refine(op, values, pref):
    """Native nested/resampling schedules and before-evaluation cap predicate."""
    from chebfunjax.utils._trigpts import global_trigpts_nodes

    choice = pref['refinementFunction']
    if not isinstance(choice, str) or choice.lower() not in ('nested', 'resampling'):
        raise ValueError('CHEBFUN:TRIGTECH:refine: No user defined refinement options allowed')
    first = _empty(values)
    if first:
        n = 2**math.ceil(math.log2(pref['minSamples']-1))
    elif choice.lower() == 'nested':
        n = 2*values.shape[0]
    else:
        power = math.log2(values.shape[0])
        n = 3*2**(int(power)-1) if power == math.floor(power) and power > 5 else 2**(math.floor(power)+1)
    if n > pref['maxLength']:
        return values, True
    if first or choice.lower() == 'resampling':
        x = jnp.concatenate((global_trigpts_nodes(n), jnp.ones(1)))
        values = jnp.asarray(op(x))
        values = values.at[0].set((.5*(values[0]+values[-1])).astype(values.dtype))
        return values[:-1], False
    fresh = jnp.asarray(op(global_trigpts_nodes(n)[1::2]))
    result = jnp.empty((n,)+values.shape[1:], dtype=jnp.result_type(values, fresh))
    return result.at[::2].set(values).at[1::2].set(fresh), False


def construct(op, *, pref=None, data=None, n=None, maxpow2=None, coefficients=False):
    """Native constructor/populate; fixed/numeric bypass adaptive validation."""
    from chebfunjax.tech.trigtech import (
        Trigtech,
        _trig_coeffs2vals_impl,
        _trig_column_mask,
        _trig_probe_mask,
        _trig_project_values,
        _trig_vals2coeffs_impl,
        trigpts,
    )

    if not callable(op) and not coefficients and _empty(op):
        return Trigtech.empty()
    pref = resolve_pref(pref, n=n, maxpow2=maxpow2)
    data = parse_data(data)
    fixed = pref['fixedLength']
    fixed = not _empty(fixed) and not math.isnan(float(jnp.asarray(fixed).reshape(())))
    if callable(op) and fixed:
        count_value = float(jnp.asarray(pref['fixedLength']).reshape(()))
        if not math.isfinite(count_value) or not count_value.is_integer() or count_value < 0:
            raise ValueError('fixedLength must be a nonnegative integer')
        count = int(count_value)
        x = jnp.concatenate((trigpts(count), jnp.ones(1)))
        values = jnp.asarray(op(x))
        values = values.at[0].set((.5*(values[0]+values[-1])).astype(values.dtype))
        op = values[:-1]
    if not callable(op):
        supplied = jnp.atleast_1d(jnp.asarray(op))
        if supplied.shape[0] == 0:
            # Native vals2coeffs/coeffs2vals return their input unchanged at
            # n<=1. Fixed-grid zero sampling reaches populate (happy numeric
            # data), preserving its empty column shape and complex storage.
            # This differs from the initial null-operand shortcut above.
            result = Trigtech(coeffs=supplied, real_columns=(), ishappy=True,
                             _values=supplied)
        else:
            coeffs = supplied if coefficients else _trig_vals2coeffs_impl(supplied)
            values = _trig_coeffs2vals_impl(coeffs) if coefficients else supplied
            mask = _trig_column_mask(values, data['vscale'])
            result = Trigtech(coeffs=coeffs, real_columns=mask, ishappy=True,
                             _values=_trig_project_values(values, mask))
        if fixed:
            result = _prolong(result, pref['fixedLength'])
        return result
    probe = jnp.asarray(op(jnp.asarray([2*.376989633393435-1], dtype=jnp.float64)))
    if not bool(jnp.all(jnp.isfinite(probe))):
        raise ValueError('Cannot handle functions that evaluate to Inf or NaN.')
    mask = _trig_probe_mask(probe)
    values = None
    result = None
    happy = None
    while True:
        values, give_up = refine(op, values, pref)
        if give_up:
            break
        finite = jnp.where(jnp.isfinite(values), values, 0)
        data['vscale'] = jnp.maximum(jnp.asarray(data['vscale']), jnp.max(jnp.abs(finite), axis=0))
        coeffs = _trig_vals2coeffs_impl(values)
        result = Trigtech(coeffs=coeffs, real_columns=mask, ishappy=None, _values=values)
        happy, cutoff = happiness(result, op, values, data, pref)
        if happy:
            # Native prolong(f,[]) takes none of its comparison branches
            # and returns f unchanged; preserve that callback outcome.
            if not _empty(cutoff):
                result = _prolong(result, cutoff)
            break
    if result is None:
        raise ValueError('CHEBFUN:TRIGTECH:populate: no initial grid below maxLength; source ishappy is undefined')
    return Trigtech(coeffs=result.coeffs, real_columns=mask, ishappy=bool(happy),
                    _values=_trig_project_values(_values(result), mask))
