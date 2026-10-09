"""Public constructor traces from Chebfun7574c77 (unqualified draft).

@chebfun/chebfun.m parseInputs/parseOp and outer constructor;
@chebfun/constructor.m getFun; @bndfun/bndfun.m; @mapping/mapping.m.
All numerical controls are independent analytic checks, not native slots.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech, trigpts

EPS = jnp.finfo(jnp.float64).eps


@pytest.fixture(autouse=True)
def preserve_preferences():
    saved = ChebfunPref()
    ChebfunPref.setDefaults('factory')
    yield
    ChebfunPref.setDefaults(saved)


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2, Trigtech])
def test_session_selected_tech_and_fixed_length(tech):
    ChebfunPref.setDefaults('tech', tech, 'fixedLength', 8)
    f = chebfun(lambda x: jnp.ones_like(x))
    assert isinstance(f.funs[0].tech, tech) and f.funs[0].n == 8
    assert jnp.max(jnp.abs(f(jnp.asarray([-.7, .1, .6])) - 1)) <= 10 * EPS


def test_pref_then_keyword_precedence_and_no_caller_mutation():
    ChebfunPref.setDefaults('tech', Chebtech1, 'fixedLength', 16)
    pref = ChebfunPref({'tech': Trigtech, 'fixedLength': 9, 'maxLength': 65537})
    before = dict(pref.techPrefs)
    f = chebfun(lambda x: jnp.ones_like(x), pref=pref, n=8)
    assert isinstance(f.funs[0].tech, Trigtech) and f.funs[0].n == 8
    assert dict(pref.techPrefs) == before
    assert ChebfunPref().tech is Chebtech1 and ChebfunPref().fixedLength == 16


@pytest.mark.parametrize('double', [False, True])
def test_fixed_trig_single_normalization_metadata_and_double_length(double):
    calls = []
    def op(x):
        calls.append(jnp.asarray(x))
        return jnp.ones_like(x)
    f = chebfun(op, trig=True, n=5, doubleLength=double)
    grids = [5, 9] if double else [5]
    assert len(calls) == 2 + len(grids)
    assert jnp.array_equal(calls[0], jnp.asarray([-.99, .99]))
    for call, n in zip(calls[1:-1], grids):
        assert jnp.array_equal(call, jnp.concatenate((trigpts(n), jnp.ones(1))))
    assert jnp.array_equal(calls[-1], jnp.asarray([-1., 1.]))
    assert f.funs[0].n == grids[-1]
    assert jnp.max(jnp.abs(f(jnp.asarray([-.3, .4])) - 1)) <= 10 * EPS


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2, Trigtech])
def test_global_hscale_rescaled_per_interval_and_operator_shared(monkeypatch, tech):
    observations = []
    original = tech.from_function
    def observed(cls, op, **kwargs):
        observations.append(kwargs)
        return original(op, **kwargs)
    monkeypatch.setattr(tech, 'from_function', classmethod(observed))
    calls = []
    def op(x):
        calls.append(jnp.asarray(x))
        return jnp.ones_like(x)
    f = chebfun(op, domain=(-2., 0., 8.), tech=tech, n=8)
    assert len(observations) == 2
    scale = [entry['data']['hscale'] if tech is Trigtech else entry['hscale']
             for entry in observations]
    assert scale == [4., 1.]
    assert jnp.array_equal(calls[0], jnp.asarray([-1.95, 7.95]))
    assert jnp.array_equal(calls[-1], jnp.asarray([-2., 0., 8.]))
    assert len(calls) == 4
    assert all(isinstance(piece.tech, tech) for piece in f.funs)


def test_fixed_trig_literal_physical_forward_map():
    calls = []
    a, b, n = -.7, 2.3, 8
    def op(x):
        calls.append(jnp.asarray(x))
        return jnp.ones_like(x)
    chebfun(op, domain=(a, b), trig=True, n=n)
    t = jnp.concatenate((trigpts(n), jnp.ones(1)))
    assert jnp.array_equal(calls[1], b * (t + 1) / 2 + a * (1 - t) / 2)


def test_existing_trig_rebuild_uses_selected_tech_not_operand_tech():
    old = chebfun(lambda x: jnp.ones_like(x), trig=True, n=8)
    rebuilt = chebfun(old, n=8)
    assert isinstance(rebuilt.funs[0].tech, Chebtech2)
    assert jnp.max(jnp.abs(rebuilt(jnp.asarray([-.6, .2])) - 1)) <= 10 * EPS


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2, Trigtech])
def test_coefficient_double_length_reuses_coefficients_with_prolongation(tech):
    # A central Fourier coefficient vs first Chebyshev coefficient is constant.
    c = jnp.asarray([0., 0., 1., 0., 0.]) if tech is Trigtech else jnp.asarray([1., 0., 0., 0., 0.])
    pref = ChebfunPref({'tech': tech})
    f = chebfun(c, pref=pref, coeffs=True, doubleLength=True)
    assert f.funs[0].n == 9
    assert jnp.max(jnp.abs(f(jnp.asarray([-.6, .2])) - 1)) <= 10 * EPS


def test_endpoint_error_propagates_after_construction():
    calls = []
    def op(x):
        calls.append(jnp.asarray(x))
        if x.shape == (2,) and bool(jnp.array_equal(x, jnp.asarray([-1., 1.]))):
            raise RuntimeError('metadata sentinel')
        return jnp.ones_like(x)
    with pytest.raises(RuntimeError, match='metadata sentinel'):
        chebfun(op, trig=True, n=8)
    assert len(calls) == 3


def test_metadata_retains_inf_and_replaces_nan_only():
    def op(x):
        if x.shape == (2,) and bool(jnp.array_equal(x, jnp.asarray([-1., 1.]))):
            return jnp.asarray([jnp.inf, jnp.nan])
        return jnp.ones_like(x)
    f = chebfun(op, trig=True, n=8)
    assert jnp.isposinf(f.point_values[0])
    assert jnp.abs(f.point_values[1] - 1) <= 10 * EPS


def test_explicit_periodic_overrides_session_splitting():
    ChebfunPref.setDefaults('splitting', True)
    f = chebfun(lambda x: jnp.ones_like(x), trig=True, n=8)
    assert len(f.funs) == 1 and isinstance(f.funs[0].tech, Trigtech)
    assert ChebfunPref().splitting is True


def test_double_length_splitting_error_precedes_periodic_override():
    ChebfunPref.setDefaults('splitting', True)
    calls = []
    def op(x):
        calls.append(x)
        return x
    with pytest.raises(ValueError, match='doubleLengthSplitting'):
        chebfun(op, trig=True, doubleLength=True)
    assert not calls


def test_unknown_keyword_rejected_before_callback():
    calls = []
    def op(x):
        calls.append(x)
        return x
    with pytest.raises(TypeError):
        chebfun(op, imaginary_option=True)
    assert not calls


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2, Trigtech])
def test_split_context_overrides_cap_extrapolation_without_mutating_pref(monkeypatch, tech):
    from chebfunjax.chebfun1d import _construction_context as module
    observed = []
    original = module.bounded_get_fun
    def record(op, interval, data, pref):
        observed.append((pref.maxLength, pref.extrapolate, data['hscale']))
        return original(op, interval, data, pref)
    monkeypatch.setattr(module, 'bounded_get_fun', record)
    pref = ChebfunPref({'tech': tech, 'fixedLength': 8, 'splitting': True,
                       'splitPrefs': {'splitLength': 32, 'splitMaxLength': 128},
                       'maxLength': 64, 'extrapolate': False})
    f = chebfun(lambda x: jnp.ones_like(x), domain=(-2., 0., 8.), pref=pref)
    assert observed == [(32, True, 8.), (32, True, 8.)]
    assert pref.maxLength == 64 and pref.extrapolate is False
    assert all(isinstance(piece.tech, tech) for piece in f.funs)
    assert jnp.max(jnp.abs(f(jnp.asarray([-1., 1., 5.])) - 1)) <= 10 * EPS


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2, Trigtech])
def test_context_merge_keeps_selected_tech_and_endpoint_metadata(tech):
    pref = ChebfunPref({'tech': tech, 'fixedLength': 8, 'maxLength': 32})
    f = chebfun(lambda x: jnp.ones_like(x), domain=(-1., 0., 1.), pref=pref)
    merged = f.merge(index=[0.], _construction_pref=pref)
    assert len(merged.funs) == 1 and isinstance(merged.funs[0].tech, tech)
    assert jnp.array_equal(merged.point_values, f.point_values[jnp.asarray([0, 2])])
    assert jnp.max(jnp.abs(merged(jnp.asarray([-.6, .3])) - 1)) <= 10 * EPS


def test_existing_complex_row_uses_native_conjugate_transpose():
    old = chebfun(lambda x: (1 + 2j) * jnp.ones_like(x), trig=True, n=8).T
    result = chebfun(old, trig=True, n=8)
    assert not result.is_transposed
    x = jnp.asarray([-.4, .3])
    # Native even-length conj reverses coefficients, shifting the constant
    # to frequency-1. Chebfun endpoint metadata is conjugated independently.
    # Rebuilding therefore adds 2*C times the cardinal at the first grid node.
    theta = jnp.pi * (x + 1)
    cardinal = (1 + 2 * sum(jnp.cos(k * theta) for k in range(1, 4))
                + jnp.cos(4 * theta)) / 8
    expected = (1 - 2j) * (jnp.exp(-1j * jnp.pi * x) + 2 * cardinal)
    assert jnp.max(jnp.abs(result(x) - expected)) <= 30 * EPS


def test_existing_complex_row_odd_length_retains_constant_conjugation():
    old = chebfun(lambda x: (1 + 2j) * jnp.ones_like(x), trig=True, n=9).T
    result = chebfun(old, trig=True, n=9)
    assert not result.is_transposed
    assert jnp.max(jnp.abs(result(jnp.asarray([-.4, .3])) - (1 - 2j))) <= 30 * EPS


def test_cells_normalize_once_on_full_domain_then_use_limit_metadata():
    calls = [[], []]
    def left(x):
        calls[0].append(jnp.asarray(x))
        return jnp.ones_like(x)
    def right(x):
        calls[1].append(jnp.asarray(x))
        return 3 * jnp.ones_like(x)
    f = chebfun([left, right], domain=(-2., 0., 8.), tech=Trigtech, n=8)
    assert [len(items) for items in calls] == [2, 2]
    assert all(jnp.array_equal(items[0], jnp.asarray([-1.95, 7.95])) for items in calls)
    assert jnp.max(jnp.abs(f.point_values - jnp.asarray([1., 2., 3.]))) <= 30 * EPS


@pytest.mark.parametrize('operand', [None, [], '', jnp.asarray([]),
                                      jnp.empty((0, 3)), jnp.empty((2, 0))])
def test_empty_operand_bypasses_invalid_parser_inputs(operand):
    result = chebfun(operand, domain=object(), impossible_keyword=True)
    assert result.isempty()


def test_fun_cell_extra_keyword_reports_native_fast_path_error():
    represented = chebfun(lambda x: jnp.ones_like(x), n=4)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:chebfun:nargin'):
        chebfun(represented.funs, impossible_keyword=True)


def test_zero_truncation_sets_splitting_but_does_not_truncate(monkeypatch):
    from chebfunjax.chebfun1d import _construction_context as module
    original = module.construct_bounded
    observed = []
    def record(context):
        observed.append(context.pref.splitting)
        return original(context)
    monkeypatch.setattr(module, 'construct_bounded', record)
    result = chebfun(lambda x: jnp.ones_like(x), n=8, trunc=0)
    assert observed == [True]
    assert result.funs[0].n == 8
    assert jnp.max(jnp.abs(result(jnp.asarray([-.4, .6])) - 1)) <= 10 * EPS


def test_public_trig_full_preference_and_data_visibility():
    seen = []
    def checker(f, values, data, pref):
        assert f.n == 16 and f.ishappy is None
        assert pref['gridType'] == 1 and pref['minSamples'] == 17
        assert pref['maxLength'] == 33 and pref['chebfuneps'] == 2.**-35
        assert pref['sampleTest'] is False and pref['extrapolate'] is True
        assert pref['refinementFunction'] == 'resampling'
        assert data['hscale'] == .5
        seen.append(values.shape[0])
        return True, []
    pref = ChebfunPref({'tech': Trigtech, 'happinessCheck': checker,
                       'gridType': 1, 'minSamples': 17, 'maxLength': 33,
                       'chebfuneps': 2.**-35, 'sampleTest': False,
                       'extrapolate': True, 'refinementFunction': 'resampling'})
    f = chebfun(lambda x: jnp.cos(jnp.pi*x/2), domain=(-2., 2.), pref=pref)
    assert seen == [16] and f.funs[0].ishappy
    x = jnp.asarray([-1.3, .2, 1.4])
    assert jnp.max(jnp.abs(f(x) - jnp.cos(jnp.pi*x/2))) <= 32 * EPS
    assert pref.maxLength == 33 and pref.happinessCheck is checker


@pytest.mark.parametrize('kind', ['callable', 'values', 'coefficients'])
def test_public_trig_fixed_and_numeric_bypass_adaptive_callbacks(kind):
    def forbidden(*args):
        raise AssertionError('adaptive callback must be bypassed')
    pref = ChebfunPref({'tech': Trigtech, 'fixedLength': 5,
                       'happinessCheck': forbidden, 'refinementFunction': forbidden})
    op = (lambda x: jnp.ones_like(x)) if kind == 'callable' else (
        jnp.ones(3) if kind == 'values' else jnp.ones(1))
    f = chebfun(op, pref=pref, coeffs=kind == 'coefficients')
    assert f.funs[0].n == 5 and f.funs[0].ishappy
    assert jnp.max(jnp.abs(f(jnp.asarray([-.3, .7])) - 1)) <= 10 * EPS


def test_fun_cell_domain_is_sorted_union_without_reordering_funs():
    from chebfunjax.chebfun1d.chebfun import _Piece
    right = _Piece(Chebtech2.from_coeffs(jnp.asarray([3.])), (2., 3.))
    left = _Piece(Chebtech2.from_coeffs(jnp.asarray([1.])), (-2., -1.))
    result = chebfun([right, left])
    assert result.domain.breakpoints == (-2., -1., 2., 3.)
    assert result.funs[0] is right and result.funs[1] is left
    # Native metadata follows FUN order even though the endpoint union is
    # sorted independently and has a different count for this unusual input.
    assert jnp.array_equal(result.point_values, jnp.asarray([3., 2., 1.]))


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2, Trigtech])
@pytest.mark.parametrize('constant', [False, True])
def test_zero_fixed_length_keeps_native_fun_domain_and_sampling(tech, constant):
    calls = []
    def op(x):
        calls.append(jnp.asarray(x))
        return 2. if constant else jnp.ones_like(x)
    result = chebfun(op, domain=(-3., 7.), tech=tech, n=0)
    assert result.isempty() and len(result.funs) == 1
    assert result.domain.breakpoints == (-3., 7.)
    assert result.funs[0].interval == (-3., 7.)
    assert isinstance(result.funs[0].tech, tech)
    assert result.funs[0].tech.coeffs.size == 0
    assert result.funs[0].ishappy and result.funs[0].vscale == 0
    assert result.point_values.shape == (0, 0)
    assert len(calls) == 2  # parseOp probe then the actual native fixed grid
    assert jnp.array_equal(calls[0], jnp.asarray([-2.95, 6.95]))
    expected = jnp.asarray([7.]) if tech is Trigtech else jnp.asarray([])
    assert jnp.array_equal(calls[1], expected)


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2, Trigtech])
def test_zero_fixed_length_numeric_preserves_source_empty_structure(tech):
    if tech is Trigtech:
        # Native prolong.m58-67 deletes all rows, then indexes the first.
        # Fixed callable n=0 already has empty coefficients and bypasses this.
        with pytest.raises(IndexError):
            chebfun(jnp.asarray([1., 2., 3.]), domain=(-3., 7.), tech=tech, n=0)
        return
    result = chebfun(jnp.asarray([1., 2., 3.]), domain=(-3., 7.), tech=tech, n=0)
    assert result.isempty() and len(result.funs) == 1
    assert result.domain.breakpoints == (-3., 7.)
    assert result.funs[0].interval == (-3., 7.)
    assert result.funs[0].tech.coeffs.size == 0
    assert result.funs[0].ishappy and result.funs[0].vscale == 0
    assert result.point_values.shape == (0, 0)


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2, Trigtech])
def test_zero_grid_preserves_explicit_complex_column_shape(tech):
    calls = []
    def op(x):
        calls.append(jnp.asarray(x))
        return jnp.broadcast_to(jnp.asarray([1+2j, 3+4j, 5-1j]), x.shape + (3,))
    result = chebfun(op, domain=(-3., 7.), tech=tech, n=0)
    actual = result.funs[0].tech
    assert result.isempty() and result.domain.breakpoints == (-3., 7.)
    assert actual.coeffs.shape == (0, 3) and actual.coeffs.dtype == jnp.complex128
    assert actual.values.shape == (0, 3) and actual.values.dtype == jnp.complex128
    assert actual.ishappy and actual.vscale == 0
    assert len(calls) == 2
    assert calls[1].shape == ((1,) if tech is Trigtech else (0,))
    assert result.point_values.shape == (0, 0)


def test_trig_empty_coefficient_population_retains_shape_and_storage():
    from chebfunjax.tech._trig_constructor import construct
    represented = construct(jnp.empty((0, 3), dtype=jnp.complex128),
                            coefficients=True, pref={'fixedLength': 0})
    assert represented.coeffs.shape == represented.values.shape == (0, 3)
    assert represented.coeffs.dtype == represented.values.dtype == jnp.complex128
    assert represented.real_columns == () and represented.ishappy
