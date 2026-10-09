"""Analytical/source control flow for R2 core, Chebfun7574c77.

Exact predicates/callback traces; no fitted tolerance. Classic short Fourier
cases have analytically known retained counts. IEEE spacing checks compare bits.
"""
from fractions import Fraction

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech._trig_constructor import (
    classic_check,
    construct,
    happiness,
    parse_data,
    refine,
    resolve_pref,
    sample_test,
    spacing,
)
from chebfunjax.tech.trigtech import Trigtech, trigpts
from chebfunjax.utils._trigpts import global_trigpts_nodes

EPS = 2.**-52


@pytest.mark.parametrize('bits,expected', [
    (0, 1), (1, 1), (0x0010000000000000, 1),
    (0x3fe0000000000000, 0x3ca0000000000000),
    (0x3ff0000000000000, 0x3cb0000000000000),
    (0x4000000000000000, 0x3cc0000000000000),
    (0xbff0000000000000, 0x3cb0000000000000),
    (0x7fefffffffffffff, 0x7ca0000000000000),
])
def test_spacing_exact_matlab_ieee_cases(bits, expected):
    x = jax.lax.bitcast_convert_type(jnp.asarray(bits, dtype=jnp.uint64), jnp.float64)
    actual = jax.jit(spacing)(x)
    assert jax.lax.bitcast_convert_type(actual, jnp.uint64) == expected


@pytest.mark.parametrize('value', [jnp.inf, -jnp.inf, jnp.nan])
def test_spacing_nonfinite_source_nan(value):
    assert jnp.isnan(spacing(value))


def test_resolver_raw_equal_default_and_private_caller_copy():
    p = ChebfunPref({'techPrefs': {'maxLength': 65537}})
    before = dict(p._tech_overrides)
    resolved = resolve_pref(p, maxpow2=4)
    assert resolved['maxLength'] == 65537
    assert p._tech_overrides == before
    assert resolve_pref()['maxLength'] == 65536
    assert resolve_pref(maxpow2=4)['maxLength'] == 16
    assert resolve_pref({'fixedLength': 8}, n=5)['fixedLength'] == 5


def test_unknown_preference_warns_then_retains_field():
    with pytest.warns(UserWarning, match="techPref:unknownPref.*'extra'"):
        pref = resolve_pref({'extra': {'x': 7}})
    assert pref['extra'] == {'x': 7}


def test_parse_data_only_missing_empty_defaults_and_private_extras():
    data = {'vscale': [], 'hscale': None, 'extra': {'x': 7}}
    result = parse_data(data)
    assert result['vscale'] == 0 and result['hscale'] == 1
    result['extra']['x'] = 9
    assert data['extra']['x'] == 7
    assert jnp.array_equal(parse_data({'vscale': jnp.asarray([[2., 3.]])})['vscale'],
                           jnp.asarray([2., 3.]))


@pytest.mark.parametrize('n', [16, 17])
def test_classic_known_degree_one_cutoff_and_original_points(n, monkeypatch):
    import chebfunjax.tech.trigtech as module
    points = trigpts(n)
    values = 1+.5*jnp.cos(jnp.pi*points)
    c = jnp.zeros(n, dtype=jnp.complex128).at[n//2].set(1)
    c = c.at[n//2-1].set(.25).at[n//2+1].set(.25)
    f = Trigtech(coeffs=c, real_columns=(True,), _values=values)
    seen = []
    original = module.trigpts

    def record(count):
        seen.append(count)
        return original(count)

    monkeypatch.setattr(module, 'trigpts', record)
    happy, cutoff = classic_check(f, values, {'vscale': 2., 'hscale': 1.},
                                  {'chebfuneps': 1e-10})
    assert happy and cutoff == 3
    assert seen == [n]


@pytest.mark.parametrize('scale,expected', [(0., (True, 1)), (jnp.inf, (False, 4))])
def test_classic_scale_shortcuts_precede_nan(scale, expected):
    f = Trigtech(coeffs=jnp.full(4, jnp.nan), _values=jnp.full(4, jnp.nan))
    assert classic_check(f, f.values, {'vscale': scale, 'hscale': 1.},
                         {'chebfuneps': EPS}) == expected


def test_classic_nan_after_scale_check_and_constant_unhappy():
    f = Trigtech(coeffs=jnp.full(4, jnp.nan), _values=jnp.full(4, jnp.nan))
    with pytest.raises(ValueError, match='classicCheck:NaNeval'):
        classic_check(f, f.values, {'vscale': 1., 'hscale': 1.}, {'chebfuneps': EPS})
    constant = Trigtech.from_values(jnp.asarray([1.]))
    assert classic_check(constant, constant.values, {'vscale': 1., 'hscale': 1.},
                         {'chebfuneps': EPS}) == (False, 1)


@pytest.mark.parametrize('below', [False, True])
def test_classic_tail_comparison_is_strict(below):
    threshold = jnp.asarray(1e-4)
    coefficient = jnp.nextafter(threshold, 0.) if below else threshold
    c = jnp.zeros(17, dtype=jnp.complex128).at[:3].set(coefficient).at[8].set(1.)
    f = Trigtech(coeffs=c, is_real=False, _values=jnp.ones(17))
    happy, cutoff = classic_check(f, f.values, {'vscale': 1., 'hscale': 0.},
                                  {'chebfuneps': threshold})
    assert happy is below
    if not below:
        assert cutoff == 9


@pytest.mark.parametrize('name', ['strict', 'loose'])
def test_source_happiness_errors_after_probe_and_grid(name):
    sizes = []

    def op(x):
        sizes.append(x.shape[0])
        return jnp.cos(jnp.pi*x)

    with pytest.raises(ValueError, match='happinessCheck:'+name+'Check'):
        Trigtech.from_function(op, pref={'happinessCheck': name})
    assert sizes == [1, 17]


@pytest.mark.parametrize('refinement,expected', [
    ('nested', [16, 32, 64, 128]), ('resampling', [16, 32, 64, 96, 128])])
def test_refinement_actual_schedule_cap_and_endpoint_average(refinement, expected):
    seen = []
    pref = resolve_pref({'refinementFunction': refinement, 'maxLength': 128})
    values = None

    def op(x):
        seen.append(x)
        return x

    for count in expected:
        previous = values
        values, giveup = refine(op, values, pref)
        assert not giveup and values.shape == (count,)
        assert values[0] == 0
        if refinement == 'nested' and previous is not None:
            assert jnp.array_equal(values[::2], previous)
            assert jnp.array_equal(seen[-1], global_trigpts_nodes(count)[1::2])
        else:
            assert jnp.array_equal(seen[-1], jnp.concatenate((global_trigpts_nodes(count), jnp.ones(1))))
    count_calls = len(seen)
    returned, giveup = refine(op, values, pref)
    assert giveup and returned is values and len(seen) == count_calls


def test_exact_nonpower_cap_and_no_initial_fallback():
    calls = []

    def op(x):
        calls.append(x.shape[0])
        return jnp.ones_like(x)

    with pytest.raises(ValueError, match='no initial grid'):
        Trigtech.from_function(op, pref={'maxLength': 15})
    assert calls == [1]
    calls.clear()
    f = Trigtech.from_function(op, pref={'maxLength': 33, 'sampleTest': False,
        'happinessCheck': lambda f, v, d, p: (False, f.n)})
    assert f.n == 32 and not f.ishappy and calls == [1, 17, 16]


def test_fixed_and_numeric_skip_all_adaptive_option_validation():
    def forbidden(*args):
        raise AssertionError('adaptive callback reached')

    pref = {'fixedLength': 5, 'happinessCheck': forbidden,
            'refinementFunction': forbidden, 'sampleTest': True}
    calls = []

    def op(x):
        calls.append(x)
        return 1+x

    f = Trigtech.from_function(op, pref=pref)
    assert f.n == 5 and f.ishappy and len(calls) == 1
    numeric = Trigtech.from_values(jnp.asarray([1., 2., 3.]), pref=pref)
    assert numeric.n == 5 and numeric.ishappy
    coefficient = Trigtech.from_coeffs(jnp.asarray([1.]), pref=pref)
    assert coefficient.n == 5 and coefficient.ishappy


def test_probe_nonfinite_error_precedes_invalid_refiner():
    with pytest.raises(ValueError) as error:
        Trigtech.from_function(lambda x: jnp.full_like(x, jnp.nan),
                              pref={'refinementFunction': lambda *args: None})
    assert str(error.value) == 'Cannot handle functions that evaluate to Inf or NaN.'


def test_custom_callback_current_tech_raw_values_and_value_semantics():
    seen = []

    def checker(f, values, data, pref):
        assert isinstance(f, Trigtech) and f.ishappy is None
        assert f.n == 16 and jnp.array_equal(f.values, values)
        assert values.dtype == jnp.float64
        assert data['hscale'] == 7 and data['extra']['x'] == 3
        assert pref['gridType'] == 1 and pref['extrapolate'] is True
        seen.append(f.n)
        data['extra']['x'] = 9
        pref['maxLength'] = 0
        return True, []

    data = {'hscale': 7, 'extra': {'x': 3}}
    f = Trigtech.from_function(lambda x: jnp.cos(jnp.pi*x), data=data,
        pref={'happinessCheck': checker, 'sampleTest': False,
              'gridType': 1, 'extrapolate': True})
    assert f.n == 16 and f.ishappy and seen == [16]
    assert data['extra']['x'] == 3


def test_sample_test_uses_full_aggregate_real_evaluator_and_reverts_cutoff():
    f = Trigtech(coeffs=jnp.asarray([1+1e-3j]), real_columns=(True,),
                 _values=jnp.asarray([1.]))
    pref = resolve_pref({'happinessCheck': lambda f, v, d, p: (True, 1)})
    assert sample_test(lambda x: jnp.ones_like(x), f, pref)
    f = Trigtech(coeffs=jnp.ones(4), real_columns=(False,), _values=jnp.ones(4))
    happy, cutoff = happiness(f, lambda x: jnp.full_like(x, 100.), f.values,
                             {'vscale': 1., 'hscale': 1.}, pref)
    assert not happy and cutoff == 4


def test_constructor_tolerance_not_compose_floor_and_plateau_alias(monkeypatch):
    import chebfunjax.tech.trigtech as module
    seen = []

    def standard(c, v, tol, scale):
        seen.append(tol)
        return True, 1

    monkeypatch.setattr(module, '_trig_standard_check', standard)
    Trigtech.from_function(lambda x: jnp.ones_like(x),
                          pref={'chebfuneps': 1e-30, 'sampleTest': False})
    assert seen == [1e-30]
    f = Trigtech.from_values(jnp.ones(16))
    args = (f, None, f.values, {'vscale': 1., 'hscale': 1.})
    classic = happiness(*args, resolve_pref({'happinessCheck': 'classic'}))
    plateau = happiness(*args, resolve_pref({'happinessCheck': 'plateau'}))
    assert classic == plateau


def test_numeric_data_scale_controls_three_eps_classification():
    values = jnp.full(5, 1+5j*EPS)
    assert Trigtech.from_values(values).real_columns == (False,)
    f = Trigtech.from_values(values, data={'vscale': 2.})
    assert f.real_columns == (True,) and jnp.all(jnp.imag(f.values) == 0)
    assert construct(None).isempty()


@pytest.mark.parametrize('tol,scale,columns,expected', [
    ([[1e-8], [1e-4]], 1., 1, [1e-8]),
    ([1e-8, 1e-4], 1., 1, [1e-4]),
    ([1e-8, 1e-4], [[2.], [3.]], 2, [float(2*Fraction.from_float(1e-8)), float(3*Fraction.from_float(1e-8))]),
    ([[1e-8, 1e-4], [2e-8, 2e-4]], 1., 2, [1e-8, 2e-8]),
])
def test_standard_tolerance_size_second_dimension_and_linear_index(
        tol, scale, columns, expected, monkeypatch):
    import chebfunjax.tech.trigtech as module
    seen = []

    def chop(coeffs, tolerance):
        seen.append(tolerance)
        return 1

    monkeypatch.setattr(module, 'standard_chop', chop)
    coeffs = jnp.zeros((16, columns)).at[8].set(1.)
    values = jnp.ones((16, columns))
    module._trig_standard_check(coeffs, values, jnp.asarray(tol), jnp.asarray(scale))
    assert jnp.array_equal(jnp.asarray(seen), jnp.asarray(expected))


def test_standard_incompatible_matrix_tolerance_is_not_flattened(monkeypatch):
    import chebfunjax.tech.trigtech as module

    def forbidden(*args):
        raise AssertionError('chopper must not see incompatible source shapes')

    monkeypatch.setattr(module, 'standard_chop', forbidden)
    with pytest.raises(TypeError):
        module._trig_standard_check(jnp.ones((16, 2)), jnp.ones((16, 2)),
                                   jnp.ones((2, 3)), 1.)


def test_custom_happiness_receives_original_empty_values():
    # Native happinessCheck forwards its VALUES argument unchanged to a
    # custom checker; standard/classic perform their own fallback reads.
    empty_values = []
    seen = []

    def checker(f, values, data, pref):
        assert values is empty_values
        assert values == []
        assert f.n == 4
        seen.append(True)
        return True, 3

    f = Trigtech.from_values(jnp.ones(4))
    pref = resolve_pref({'happinessCheck': checker, 'sampleTest': False})
    assert happiness(f, None, empty_values, {}, pref) == (True, 3)
    assert seen == [True]
