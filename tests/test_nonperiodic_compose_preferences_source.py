"""Nonperiodic source columnCompose preferences and actual public outputs."""
import copy

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece, chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.fixture(autouse=True)
def isolated_session():
    saved = ChebfunPref._defaults
    ChebfunPref.setDefaults('factory')
    yield
    ChebfunPref._defaults = saved


@pytest.mark.parametrize('Tech,kind', [(Chebtech1, 1), (Chebtech2, 2)])
@pytest.mark.parametrize('binary', [False, True])
@pytest.mark.parametrize('route', ['flat', 'nested', 'materialized', 'session'])
def test_source_pref_routing(monkeypatch, Tech, kind, binary, route):
    f = chebfun(lambda x: .2*x, domain=(-2, 3), chebkind=kind)
    settings = dict(maxLength=31, minSamples=33, refinementFunction='resampling',
                    extrapolate=True, happinessCheck='strict', sampleTest=True)
    if route == 'flat':
        pref = dict(settings)
    elif route == 'nested':
        pref = {'techPrefs': dict(settings)}
    elif route == 'materialized':
        p = ChebfunPref(**settings)
        pref = {'techPrefs': dict(p.techPrefs)}
    else:
        ChebfunPref.setDefaults(**settings)
        pref = None
    before = copy.deepcopy(pref)
    seen = []

    def spy(self, op, g=None, **kw):
        seen.append(copy.deepcopy(kw.get('pref')))
        return self

    monkeypatch.setattr(Tech, 'compose', spy)
    out = f.compose(jnp.add, f, pref=pref) if binary else f.compose(lambda v: v, pref=pref)
    assert type(out.funs[0].tech) is Tech
    assert len(seen) == 1 and seen[0] is not None
    for key, value in settings.items():
        if key != 'sampleTest':
            assert seen[0][key] == value
    assert seen[0]['sampleTest'] is True
    assert pref == before


def test_source_split_and_piece_overrides(monkeypatch):
    t = Chebtech2.from_coeffs(jnp.array([.1, .2]))
    f = Chebfun([_Piece(t, (-1, 0)), _Piece(t, (0, 1))], Domain((-1, 0, 1)))
    f = f.set_point_values(jnp.array([5., 6., 7.])).T
    pref = ChebfunPref(maxLength=65537, splitting=True, splitPrefs={'splitLength': 31})
    seen = []

    def spy(self, op, g=None, **kw):
        seen.append(copy.deepcopy(kw.get('pref')))
        return self

    monkeypatch.setattr(Chebtech2, 'compose', spy)
    out = f.compose(jnp.square, pref=pref)
    assert len(seen) == 2 and all(v is not None for v in seen)
    assert all(v['maxLength'] == 31 and v['extrapolate'] is True for v in seen)
    assert pref.maxLength == 65537 and pref.extrapolate is False
    assert out.is_transposed and out.domain == f.domain
    np.testing.assert_array_equal(out.point_values, [25., 36., 49.])


@pytest.mark.parametrize('kind', [1, 2])
@pytest.mark.parametrize('turbo', [False, True])
def test_source_fixed_and_turbo_raw_operator(kind, turbo):
    # Source installs a closure refiner but leaves OP raw. Fixed construction
    # and contour integration evaluate that raw OP, even for nonidentity F.
    f = chebfun(lambda x: .25+.5*x, chebkind=kind)
    g = f.compose(jnp.exp, pref={'fixedLength': 18, 'useTurbo': turbo})
    assert g.funs[0].tech.n == 18
    x = jnp.array([-.8, -.2, .4, .9])
    np.testing.assert_allclose(g(x), jnp.exp(x), atol=2e-13, rtol=2e-13)
    assert float(jnp.max(jnp.abs(g(x)-jnp.exp(.25+.5*x)))) > .1


@pytest.mark.parametrize('kind', [1, 2])
def test_actual_binary_preserves_kind_and_default_unary_coefficients(kind):
    f = chebfun(lambda x: .1+.2*x, domain=(-2, 3), chebkind=kind)
    expected = f.funs[0].tech.compose(jnp.sin)
    unary = f.compose(jnp.sin)
    np.testing.assert_array_equal(unary.funs[0].tech.coeffs, expected.coeffs)
    g = f.compose(jnp.multiply, f, pref={'minSamples': 33, 'refinementFunction': 'resampling'})
    assert type(g.funs[0].tech) is type(f.funs[0].tech)
    x = jnp.array([-1.8, -.2, 1.2, 2.9])
    np.testing.assert_allclose(g(x), (.1+.2*x)**2, atol=2e-14, rtol=2e-14)


def test_actual_mixed_pieces_points_orientation_and_empty():
    f = Chebfun([_Piece(Chebtech1.from_coeffs(jnp.array([-.5, .5])), (-1, 0)),
                 _Piece(Chebtech2.from_coeffs(jnp.array([.5, .5])), (0, 1))],
                Domain((-1, 0, 1))).set_point_values(jnp.array([5., 6., 7.])).T
    g = f.compose(jnp.square, pref={'refinementFunction': 'resampling'})
    assert [type(p.tech) for p in g.funs] == [Chebtech1, Chebtech2]
    assert g.is_transposed and g.domain == f.domain
    np.testing.assert_array_equal(g.point_values, [25., 36., 49.])
    for p in g.funs:
        x = jnp.array([p.interval[0]+.2, p.interval[1]-.2])
        np.testing.assert_allclose(p(x), x*x, atol=2e-14, rtol=2e-14)
    empty = Chebfun.empty().T
    assert empty.compose(jnp.exp, pref={'minSamples': 33}) is empty


@pytest.mark.parametrize('Tech,kind', [(Chebtech1, 1), (Chebtech2, 2)])
def test_actual_constructor_source_overrides(monkeypatch, Tech, kind):
    f = chebfun(lambda x: .1+.2*x, chebkind=kind)
    seen = []
    sample_sizes = []

    def operation(values):
        sample_sizes.append(values.shape[0])
        return jnp.square(values)

    original = Tech._adaptive_construct

    def observe(cls, raw_op, *args, **kwargs):
        seen.append((raw_op, kwargs.copy()))
        return original(raw_op, *args, **kwargs)

    monkeypatch.setattr(Tech, '_adaptive_construct', classmethod(observe))
    g = f.compose(operation, pref={'minSamples': 33, 'maxLength': 31,
                                   'chebfuneps': np.finfo(float).eps/2,
                                   'sampleTest': True, 'happinessCheck': 'strict'})
    assert len(seen) == 1 and seen[0][0] is operation
    opts = seen[0][1]
    assert opts['min_samples'] == 33 and opts['max_length'] == 31
    assert opts['sample_test'] is False and opts['check'] == 'strict'
    assert opts['tol'] == np.finfo(float).eps
    assert sample_sizes == [2, 31]
    x = jnp.array([-.8, -.2, .4, .9])
    np.testing.assert_allclose(g(x), (.1+.2*x)**2, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize('kind', [1, 2])
def test_custom_refiner_receives_raw_operator(kind):
    from chebfunjax.utils.quadrature import chebpts
    seen = []

    def refine(raw_op, old, pref):
        seen.append(pref.copy())
        assert raw_op is jnp.square
        n = pref['minSamples'] if old is None else 2*len(old)-1
        if n > pref['maxLength']:
            return old, True
        x = chebpts(n, kind)
        return raw_op(.1+.2*x), False

    f = chebfun(lambda x: .1+.2*x, chebkind=kind)
    g = f.compose(jnp.square, pref={'refinementFunction': refine, 'maxLength': 65,
                                   'sampleTest': True, 'extrapolate': True})
    assert seen and seen[0]['maxLength'] == 65
    assert seen[0]['sampleTest'] is False and seen[0]['extrapolate'] is True
    x = jnp.array([-.8, -.2, .4, .9])
    np.testing.assert_allclose(g(x), (.1+.2*x)**2, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize('mixed', [False, True])
def test_splitting_fallback_receives_complete_modified_pref(monkeypatch, mixed):
    import importlib
    module = importlib.import_module('chebfunjax.chebfun1d.chebfun')
    f = chebfun(lambda x: .1+.2*x, domain=(-1, 0, 1), chebkind=1)
    if mixed:
        f = Chebfun([f.funs[0], f.funs[1].with_tech(
            Chebtech2.from_coeffs(f.funs[1].tech.coeffs))], f.domain)
    f = f.set_point_values(jnp.array([5., 6., 7.])).T
    seen = []

    def unhappy(self, op, g=None, **kw):
        return type(self)(coeffs=self.coeffs, ishappy=False)

    def constructor(op, domain, **kwargs):
        pref = kwargs['pref']
        seen.append(ChebfunPref(pref))
        return f.restrict(domain)

    monkeypatch.setattr(Chebtech1, 'compose', unhappy)
    monkeypatch.setattr(Chebtech2, 'compose', unhappy)
    monkeypatch.setattr(module, 'chebfun', constructor)
    pref = ChebfunPref(splitting=True, minSamples=33, refinementFunction='resampling',
                       splitPrefs={'splitLength': 31, 'splitMaxLength': 200})
    g = f.compose(jnp.square, pref=pref)
    assert len(seen) == 2
    # Native columnCompose selects the first FUN technology once, including
    # every later splitting fallback; mixed internal pieces do not reset it.
    for p in seen:
        assert p.tech is type(f.funs[0].tech) and p.maxLength == 31 and p.minSamples == 33
        assert p.refinementFunction == 'resampling' and p.extrapolate is True
        assert p.splitPrefs.splitMaxLength == 200
    assert pref.extrapolate is False and pref.tech != Chebtech1
    assert g.is_transposed and g.domain == f.domain
    np.testing.assert_array_equal(g.point_values, [25., 36., 49.])


@pytest.mark.parametrize('kind', [1, 2])
def test_fractional_fixed_length_rejected_by_existing_constructor_contract(kind):
    f = chebfun(lambda x: .1+.2*x, chebkind=kind)
    with pytest.raises(ValueError, match='nonnegative integer'):
        f.compose(jnp.exp, pref={'fixedLength': 3.5})
