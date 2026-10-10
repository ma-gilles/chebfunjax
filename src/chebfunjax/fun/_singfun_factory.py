"""Singfun factory dispatch, Chebfun7574c77; coefficient arithmetic uses JAX.

This adapter stays on the reference interval and never rescales hscale.
The low-level Singfun initializer and direct from_function API remain separate.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import jax.numpy as jnp

OMITTED = object()


def _empty(value):
    if value is None:
        return True
    if callable(value) or isinstance(value, Mapping):
        return False
    if hasattr(value, 'size'):
        return value.size == 0
    return hasattr(value, '__len__') and len(value) == 0


def make(cls, op, exponents, sing_type, pref, data):
    """Adapt native dictionary calls while retaining legacy Python arguments."""
    from chebfunjax.chebpref import ChebfunPref

    if data is not OMITTED:
        if exponents is not OMITTED or sing_type is not OMITTED:
            raise TypeError('data cannot be combined with exponents or singType')
    elif isinstance(exponents, Mapping) or (
            exponents is not OMITTED and _empty(exponents)
            and isinstance(sing_type, (Mapping, ChebfunPref))):
        data = exponents
        if sing_type is not OMITTED:
            if pref is not OMITTED:
                raise TypeError('preferences supplied twice')
            pref = sing_type
    elif exponents is not OMITTED or sing_type is not OMITTED:
        data = {}
        if exponents is not OMITTED:
            data['exponents'] = exponents
        if sing_type is not OMITTED:
            data['singType'] = sing_type
    return construct(cls, op, data, pref)


def _smooth_part(op, data, pref):
    """Source constructSmoothPart -> smoothfun -> selected tech constructor."""
    from chebfunjax.chebfun1d._construction_context import fixed_length, selected_tech
    from chebfunjax.tech.chebtech import Chebtech2, _extrapolate_values
    from chebfunjax.tech.trigtech import Trigtech
    from chebfunjax.utils.interpolation import funqui
    from chebfunjax.utils.quadrature import chebpts

    if pref.enableFunqui:
        op = funqui(jnp.asarray(op))
    tech = selected_tech(pref)
    # Unknown data fields are ignored by native Chebtech.parseDataInputs;
    # Trigtech receives the complete data record through its public adapter.
    local = copy.deepcopy(data)
    for key, default in [('vscale', 0.), ('hscale', 1.)]:
        if key not in local or _empty(local[key]):
            local[key] = default
    n = fixed_length(pref)
    if tech is Trigtech:
        if callable(op):
            return tech.from_function(op, data=local, pref=pref.techPrefs)
        return tech.from_values(jnp.atleast_1d(jnp.asarray(op)), data=local, pref=pref.techPrefs)
    if callable(op):
        options = dict(n=n, tol=pref.chebfuneps,
                       turbo=pref.useTurbo, check=pref.happinessCheck,
                       sample_test=pref.sampleTest,
                       refinement_function=pref.refinementFunction,
                       max_length=pref.maxLength, min_samples=pref.minSamples,
                       vscale=local['vscale'], hscale=local['hscale'])
        if tech is Chebtech2:
            options['extrapolate'] = pref.extrapolate
        return tech.from_function(op, **options)
    values = jnp.atleast_1d(jnp.asarray(op))
    values = values.astype(jnp.complex128 if jnp.iscomplexobj(values) else jnp.float64)
    if not bool(jnp.all(jnp.isnan(values))) and not bool(jnp.all(jnp.isfinite(values))):
        values = _extrapolate_values(values, chebpts(values.shape[0], kind=2 if tech is Chebtech2 else 1),
                                     tech.barywts(values.shape[0]))[0]
    result = tech.from_values(values)
    return result.prolong(n) if n is not None else result


def construct(cls, op=OMITTED, data=OMITTED, pref=OMITTED):
    """Literal Singfun constructor branch order; Python1D numeric means column."""
    from chebfunjax.chebpref import ChebfunPref
    from chebfunjax.fun.singfun import _find_sing_exponents
    from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
    from chebfunjax.tech.trigtech import Trigtech

    if op is OMITTED:
        return cls.empty()
    one_input = data is OMITTED and pref is OMITTED
    if data is OMITTED or _empty(data):
        data = {}
    elif not isinstance(data, Mapping):
        raise TypeError('Singfun constructor data must be a dictionary')
    data = copy.deepcopy(dict(data))
    private = ChebfunPref(None if pref is OMITTED or _empty(pref) else pref)
    supplied = data.get('exponents')
    hints = data.get('singType')
    if _empty(hints):
        hints = (private.blowupPrefs.defaultSingType,)*2
    data['singType'] = hints
    smooth = isinstance(op, (Chebtech1, Chebtech2, Trigtech))
    if one_input:
        if isinstance(op, cls):
            return op
        if smooth:
            return cls(op, (0., 0.))
    if _empty(supplied):
        exponents = _find_sing_exponents(op, hints)
    else:
        exponents = jnp.asarray(supplied)
        if exponents.shape not in ((2,), (1, 2)):
            raise ValueError('CHEBFUN:SINGFUN:singfun:badExponents: '
                             'Exponents must have two entries in a row.')
        exponents = exponents.reshape(2)
        if bool(jnp.any(jnp.isnan(exponents))):
            detected = jnp.asarray(_find_sing_exponents(op, hints))
            exponents = jnp.where(jnp.isnan(exponents), detected, exponents)
    a, b = float(exponents[0]), float(exponents[1])
    data['exponents'] = (a, b)
    if isinstance(op, cls) or isinstance(op, (str, Mapping)):
        raise TypeError('CHEBFUN:SINGFUN:singfun:badOp: Expected numeric, callable or smooth tech')
    if callable(op):
        probe = jnp.asarray(op(jnp.asarray(0.)))
        multiple = (probe.ndim == 1 and probe.size > 1) or (probe.ndim >= 2 and probe.shape[1] > 1)
    else:
        op = jnp.asarray(op)
        multiple = op.ndim >= 2 and op.shape[1] > 1
    if multiple:
        raise ValueError('CHEBFUN:SINGFUN:singfun:arrayValued: SINGFUN does not support array-valued construction.')
    if a < 0 or b < 0:
        private.extrapolate = True
    if smooth:
        return cls(op, (a, b))
    if not callable(op):
        return cls(_smooth_part(op, data, private), (a, b))
    if a != 0 or b != 0:
        private.chebfuneps = max(private.chebfuneps, 1e-14)

    def smooth_op(x):
        value = jnp.asarray(op(x))
        # Native callbacks sample column points. Python Tech grids are 1D;
        # align only a scalar column's point axis before source division.
        if value.shape == x.shape + (1,):
            x = x[..., None]
        if a != 0 and b != 0:
            return value / ((1+x)**a * (1-x)**b)
        if a != 0:
            return value / (1+x)**a
        if b != 0:
            return value / (1-x)**b
        return value

    return cls(_smooth_part(smooth_op, data, private), (a, b))
