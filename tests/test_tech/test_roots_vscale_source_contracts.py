"""Public roots use the original Tech grid's scalar-column vscale.

Provenance
----------
MATLAB source : @chebtech/roots.m, roots_scalar and array-column dispatch;
                @chebtech/vscale.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech import chebtech as module
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(jnp.finfo(jnp.float64).eps)
TECHS = [Chebtech1, Chebtech2]


def _record_normalized(monkeypatch):
    observed = []
    original = module._roots_main

    def recording(c, *args, **kwargs):
        observed.append(jnp.asarray(c))
        return original(c, *args, **kwargs)

    monkeypatch.setattr(module, "_roots_main", recording)
    return observed, original


def _positive_node(Tech):
    # With TWO coefficients, these are the original technology's right nodes.
    return 1. if Tech is Chebtech2 else 1/jnp.sqrt(2.)


@pytest.mark.parametrize("Tech", TECHS)
@pytest.mark.parametrize("amplitude", [1., -3., 2j, 1+2j])
def test_own_grid_normalization_real_and_complex(Tech, amplitude, monkeypatch):
    observed, _ = _record_normalized(monkeypatch)
    c = amplitude*jnp.asarray([1., 1.])
    roots = Tech(coeffs=c).roots()
    scale = jnp.abs(amplitude)*(1+_positive_node(Tech))
    assert len(observed) == 1
    # A two-point transform plus absolute value/division uses only a few
    # float64 operations; 16eps is fixed before execution, independently of
    # any observed discrepancy. Root predicates retain their native bounds.
    assert float(jnp.max(jnp.abs(observed[0]-c/scale))) < 16*EPS
    assert roots.shape == (1,)
    assert float(jnp.abs(roots[0]+1)) < 10*EPS


@pytest.mark.parametrize("Tech", TECHS)
def test_each_array_column_retains_own_tech_and_scale(Tech, monkeypatch):
    observed, _ = _record_normalized(monkeypatch)
    original_vscale = Tech.vscale
    scale_shapes = []

    def record_vscale(self):
        scale_shapes.append(self.coeffs.shape)
        return original_vscale.fget(self)

    monkeypatch.setattr(Tech, "vscale", property(record_vscale))
    amplitudes = jnp.asarray([1e12, 1e-6, 3j])
    c = jnp.ones((2, 1))*amplitudes[None, :]
    roots = Tech(coeffs=c).roots()
    assert scale_shapes == [(2,), (2,), (2,)]
    assert len(observed) == 3
    for j, normalized in enumerate(observed):
        scale = jnp.abs(amplitudes[j])*(1+_positive_node(Tech))
        assert float(jnp.max(jnp.abs(normalized-c[:, j]/scale))) < 16*EPS
    assert roots.shape == (1, 3)
    assert float(jnp.max(jnp.abs(roots+1))) < 10*EPS


@pytest.mark.parametrize("Tech", TECHS)
def test_changed_coefficients_recompute_same_shape_scale(Tech, monkeypatch):
    observed, _ = _record_normalized(monkeypatch)
    for constant, slope in [(1., 1.), (2., -1.), (3., .5)]:
        c = jnp.asarray([constant, slope])
        roots = Tech(coeffs=c).roots(all_roots=True)
        scale = constant+abs(slope)*_positive_node(Tech)
        assert float(jnp.max(jnp.abs(observed[-1]-c/scale))) < 16*EPS
        assert roots.shape == (1,)
        assert float(jnp.abs(roots[0]+constant/slope)) < 10*EPS
    assert len(observed) == 3


@pytest.mark.parametrize("Tech", TECHS)
def test_constants_shortcut_precedes_scale_and_engine(Tech, monkeypatch):
    inputs = [Tech(coeffs=jnp.asarray([value])) for value in (0., 2., 3+4j)]
    array = Tech(coeffs=jnp.asarray([[0., 2., 3+4j]]))
    scales = []
    engines = []
    original_vscale = Tech.vscale
    original_engine = module._roots_colleague

    def record_vscale(self):
        scales.append(self.coeffs.shape)
        return original_vscale.fget(self)

    def record_engine(*args, **kwargs):
        engines.append(args[0].shape)
        return original_engine(*args, **kwargs)

    monkeypatch.setattr(Tech, "vscale", property(record_vscale))
    monkeypatch.setattr(module, "_roots_colleague", record_engine)
    assert bool(jnp.array_equal(inputs[0].roots(), jnp.asarray([0.])))
    assert inputs[1].roots().shape == (0,)
    assert inputs[2].roots(complex_roots=True).shape == (0,)
    actual = array.roots(complex_roots=True)
    assert bool(jnp.array_equal(actual, jnp.asarray([[0., jnp.nan, jnp.nan]]), equal_nan=True))
    assert not scales and not engines


@pytest.mark.parametrize("Tech", TECHS)
@pytest.mark.parametrize("amplitude", [1e-200, 1e200])
def test_finite_extreme_amplitudes_do_not_change_policy(Tech, amplitude, monkeypatch):
    observed, _ = _record_normalized(monkeypatch)
    c = amplitude*jnp.asarray([1., 1.])
    roots = Tech(coeffs=c).roots()
    scale = amplitude*(1+_positive_node(Tech))
    assert bool(jnp.all(jnp.isfinite(observed[0])))
    assert float(jnp.max(jnp.abs(observed[0]-c/scale))) < 16*EPS
    assert float(jnp.abs(roots[0]+1)) < 10*EPS


@pytest.mark.parametrize("Tech", TECHS)
def test_native_complex_pair_order_survives_normalization(Tech):
    # Native roots slot6: 1+25*x**2 has the ordered roots [+i,-i]/5.
    roots = Tech(coeffs=jnp.asarray([13.5, 0., 12.5])).roots(complex_roots=True)
    assert roots.shape == (2,)
    assert float(jnp.max(jnp.abs(roots-jnp.asarray([1j, -1j])/5))) < 10*EPS


def test_coefficient_only_and_resultant_compatibility(monkeypatch):
    from chebfunjax.chebfun2d.resultant import _chebT1rts

    observed, _ = _record_normalized(monkeypatch)
    c = jnp.asarray([1., 1.])
    direct = module._roots_colleague(c)
    resultant = jnp.asarray(_chebT1rts(jax.device_get(c)))
    assert len(observed) == 2
    assert all(bool(jnp.array_equal(value, c)) for value in observed)
    assert bool(jnp.array_equal(direct, resultant))
    assert bool(jnp.array_equal(direct, jnp.asarray([-1.])))
