"""Exact source branches and independent low-degree extrema controls.

Provenance: @separableApprox/{minandmax2,iszero,fevalm,cdr}.m and native
Chebfun parseOp/simplify; pin7574c77680d7e82b79626300bf255498271a72df.
Selected-tech controls exercise the integrated shared constructor preference
contract; no test is skipped or relaxed to hide that dependency.
"""

import math
from fractions import Fraction

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun2d import _extrema_source as source
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech import chebtech as tech_module
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech

EPS = 2.0**-52


def poly(*coeffs):
    return Chebtech2.from_coeffs(jnp.asarray(coeffs, dtype=jnp.float64))


def function(cols, rows, weights, domain=(-1., 1., -1., 1.)):
    return Chebfun2(approx=SeparableApprox(
        cols=cols, rows=rows, pivots=jnp.asarray(weights, dtype=jnp.float64),
        domain=domain))


@pytest.fixture(autouse=True)
def prefs():
    saved = ChebfunPref()
    ChebfunPref.setDefaults('factory')
    try:
        yield
    finally:
        ChebfunPref.setDefaults(saved)


def assert_close(actual, expected, scale=1.):
    # Low-degree products: no high-degree conditioning claim. The predeclared
    # 256eps*represented scale covers fixed FFT, affine maps and root evaluation.
    assert bool(jnp.all(jnp.abs(jnp.asarray(actual)-jnp.asarray(expected))
                        <= 256*EPS*scale))


def test_01_empty_numeric_outputs():
    values, locations = Chebfun2.empty().minandmax2()
    assert values.shape == locations.shape == (0,)


def test_02_zero_weights_physical_midpoint():
    f = function([poly(1)], [poly(1)], [0], (2., 6., -7., -1.))
    values, locations = f.minandmax2()
    assert bool(jnp.array_equal(values, jnp.zeros(2)))
    assert bool(jnp.array_equal(locations, jnp.asarray([[4., -4.], [4., -4.]])))


def test_03_zero_columns_nonzero_weights():
    f = function([poly(0), poly(0)], [poly(1), poly(2)], [1, -4])
    assert source._iszero(f.approx)
    assert bool(jnp.array_equal(f.minandmax2()[0], jnp.zeros(2)))


def test_04_zero_rows_nonzero_weights():
    f = function([poly(1), poly(2)], [poly(0), poly(0)], [1, -4])
    assert source._iszero(f.approx)
    assert bool(jnp.array_equal(f.minandmax2()[0], jnp.zeros(2)))


def test_05_cancellation_and_cdr_action():
    cancel = function([poly(1), poly(1)], [poly(1), poly(1)], [1, -1])
    assert not source._iszero(cancel.approx)  # Native slice condition, not rank reduction.
    f = function([poly(1, 1), poly(2)], [poly(2, -1), poly(0, 1)], [2, -3])
    x = jnp.asarray([-1., 0., 1.])
    y = jnp.asarray([-1., 0., 1.])
    actual = source._mesh_values(f.approx, x, y)
    expected = 2*(1+y[:, None])*(2-x[None, :])-6*x[None, :]
    assert bool(jnp.array_equal(actual, expected))
    assert bool(jnp.array_equal(actual, f(x[None, :], y[:, None])))
    inf_weight = function([poly(1)], [poly(1)], [jnp.inf])
    assert bool(jnp.array_equal(source._mesh_values(inf_weight.approx, x, y),
                               jnp.zeros((3, 3))))


def test_06_tiny_nonzero_is_not_zero():
    f = function([poly(2.0**-100)], [poly(1)], [1])
    assert not source._slice_iszero(f.approx.cols[0])
    assert not source._iszero(f.approx)


def test_07_trig_uses_stored_values():
    # Deliberately distinguish native redundant storage fields, rather than
    # infer zero from coefficients. No numerical function-equivalence claim.
    t = Trigtech(coeffs=jnp.ones(3, dtype=jnp.complex128),
                 _values=jnp.zeros(3), is_real=True)
    assert source._slice_iszero(t)
    t = Trigtech(coeffs=jnp.zeros(3, dtype=jnp.complex128),
                 _values=jnp.asarray([0., 2.0**-100, 0.]), is_real=True)
    assert not source._slice_iszero(t)


def test_08_positive_weight_physical_extrema():
    # r=x_ref, c=1+y_ref; physical x[2,6], y[-3,5].
    f = function([poly(1, 1)], [poly(0, 1)], [9], (2., 6., -3., 5.))
    values, locations = f.minandmax2()
    assert_close(values, [-18., 18.], 18.)
    assert_close(locations, [[2., 5.], [6., 5.]], 6.)


def test_09_negative_weight_first_four_product_ties():
    # Distributed sign makes scaled row extrema occur at x=+1 then -1.
    # vv=[+4,-4,-4,+4]; first minimum1 and maximum0 are required.
    f = function([poly(0, 1)], [poly(0, 1)], [-4])
    values, locations = f.minandmax2()
    assert_close(values, [-4., 4.], 4.)
    assert_close(locations, [[1., 1.], [1., -1.]])


def test_10_constant_native_first_location():
    f = function([poly(2)], [poly(3)], [4], (2., 6., -3., 5.))
    values, locations = f.minandmax2()
    assert_close(values, [24., 24.], 24.)
    # Native @chebtech/minandmax.m77 returns reference0 for length1.
    assert_close(locations, [[4., 1.], [4., 1.]], 6.)


def test_11_periodic_fixed4000_before_simplify(monkeypatch):
    seen = []
    simplify = Chebfun.simplify

    def observe(self, tol=None):
        seen.append((len(self), type(self.funs[0].tech), tol))
        return simplify(self, tol)

    monkeypatch.setattr(Chebfun, 'simplify', observe)
    t = Trigtech.from_values(jnp.asarray([1., 0., -1., 0.]))
    out = source._reconstruct([t], -1., 1.)
    assert seen == [(4000, Chebtech2, EPS)]
    assert isinstance(out.funs[0].tech, Chebtech2)
    x = jnp.asarray([-.75, -.125, .25, .875])
    assert_close(jnp.asarray(out(x)).reshape(-1), t(x), 1.)
    assert len(out) < 4000


def preference_boundary(monkeypatch, technology, *, turbo=True):
    ChebfunPref.setDefaults('tech', technology)
    settings = dict(chebfuneps=2.0**-24, fixedLength=17, minSamples=33,
                    maxLength=4097, refinementFunction='resampling',
                    sampleTest=False, useTurbo=turbo, extrapolate=True,
                    happinessCheck='strict')
    for key, value in settings.items():
        ChebfunPref.setDefaults(key, value)
    seen = []
    turbo_seen = {}
    original_turbo = tech_module._turbo_coeffs
    original_contour = tech_module._cheb_coeffs_turbo

    def observe_turbo(op, plain_coeffs, num):
        turbo_seen['plain_length'] = len(plain_coeffs)
        turbo_seen['num'] = num
        return original_turbo(op, plain_coeffs, num)

    def observe_contour(op, rho, num):
        turbo_seen['rho'] = rho
        def observe_op(points):
            values = op(points)
            turbo_seen['sample_count'] = points.size
            return values
        out = original_contour(observe_op, rho, num)
        turbo_seen['coefficients'] = out
        return out

    monkeypatch.setattr(tech_module, '_turbo_coeffs', observe_turbo)
    monkeypatch.setattr(tech_module, '_cheb_coeffs_turbo', observe_contour)
    build = technology.from_function.__func__

    def observe(cls, callback, *args, **kwargs):
        seen.append(dict(kwargs))
        return build(cls, callback, *args, **kwargs)

    monkeypatch.setattr(technology, 'from_function', classmethod(observe))
    result = source._reconstruct([poly(1, 2)], 2., 6.)
    assert isinstance(result.funs[0].tech, technology)
    assert len(seen) == 1
    actual = seen[0]
    assert actual['n'] == 4000  # Source argument overrides session17.
    assert actual['tol'] == settings['chebfuneps']
    assert actual['min_samples'] == settings['minSamples']
    assert actual['max_length'] == settings['maxLength']
    assert actual['refinement_function'] == settings['refinementFunction']
    assert actual['sample_test'] is False
    assert actual['turbo'] is turbo
    assert actual['check'] == 'strict'
    # C2 forwards extrapolate to Tech; C1 has no endpoint samples.
    if technology is Chebtech2:
        assert actual['extrapolate'] is True
    if turbo:
        # Native constructorTurbo uses a huge ellipse for this length2 plain
        # polynomial. Finite contour samples can lose absolute c0 accuracy:
        # the original unqualified256eps analytic assertion failed and remains
        # archived in simple15_v1. No native test bound is changed here.
        # These assertions qualify routing/stage semantics, not turbo accuracy.
        assert turbo_seen['plain_length'] == 2
        assert turbo_seen['num'] == 4000
        assert turbo_seen['sample_count'] == 4*4000
        expected_rho = math.exp(abs(math.log(EPS))/2)**(2/3)
        assert turbo_seen['rho'] == expected_rho
        assert len(result) == 2
        assert bool(jnp.array_equal(result.funs[0].tech.coeffs,
                                    jnp.real(turbo_seen['coefficients'][:2])))
        assert bool(jnp.array_equal(jnp.asarray(result(jnp.asarray([2., 6.]))).reshape(-1),
                                    jnp.asarray([-1., 3.])))
    else:
        assert turbo_seen == {}
        # Independent polynomial oracle retains exactly the original bound
        # under ordinary fixed4000 construction, without contour cancellation.
        assert_close(jnp.asarray(result(jnp.asarray([2., 4., 6.]))).reshape(-1),
                     [-1., 1., 3.], 3.)


def test_12_selected_c1_full_constructor_preferences(monkeypatch):
    preference_boundary(monkeypatch, Chebtech1)


def test_13_selected_c2_full_constructor_preferences(monkeypatch):
    preference_boundary(monkeypatch, Chebtech2)


def test_15_nan_is_not_exact_zero():
    assert not source._slice_iszero(poly(jnp.nan))
    t = Trigtech(coeffs=jnp.zeros(3, dtype=jnp.complex128),
                 _values=jnp.asarray([0., jnp.nan, 0.]), is_real=True)
    assert not source._slice_iszero(t)
    f = function([poly(jnp.nan)], [poly(1)], [1])
    assert not source._iszero(f.approx)


def test_16_large_offset_native_inverse_map():
    # These exactly representable endpoints have a rounded sum a+b. The old
    # centered expression would give -1.2 and0.8, despite exact endpoints.
    a, b = float(2**52 + 1), float(2**52 + 6)
    offsets = (0, 1, 2, 4, 5)
    x = jnp.asarray([a+k for k in offsets])
    actual = source._slice_values([poly(0, 1)], x, a, b)[:, 0]
    expected = jnp.asarray([float(Fraction(2*k, 5)-1) for k in offsets])
    assert actual[0] == -1. and actual[-1] == 1.
    assert bool(jnp.all(jnp.abs(actual-expected) <= 2*EPS))


@pytest.mark.parametrize('technology', [Chebtech1, Chebtech2])
def test_17_selected_technology_nonturbo_analytic_accuracy(monkeypatch, technology):
    preference_boundary(monkeypatch, technology, turbo=False)
