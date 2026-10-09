"""Public routing/source clauses; mock controls do not claim optimizer accuracy.

Provenance: @separableApprox/minandmax2.m, pin7574c77680d7e82b79626300bf255498271a72df.
"""
import math
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d import _extrema_source as source
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech


@pytest.fixture(autouse=True)
def prefs():
    saved = ChebfunPref()
    ChebfunPref.setDefaults('factory')
    try:
        yield
    finally:
        ChebfunPref.setDefaults(saved)


def poly(*coeffs):
    return Chebtech2.from_coeffs(jnp.asarray(coeffs, dtype=jnp.float64))


def function(cols, rows, weights):
    return Chebfun2(approx=SeparableApprox(cols=cols, rows=rows,
        pivots=jnp.asarray(weights), domain=(-1., 1., -1., 1.)))


def test_empty_wrappers_numeric_outputs():
    f = Chebfun2.empty()
    for call in (f.minandmax2, f.min2, f.max2):
        vals, locs = call()
        assert vals.shape == locs.shape == (0,)
    # Empty handling precedes unrelated legacy tuning.
    assert f.minandmax2(ngrid=17)[0].size == 0


def test_literal_scaling_operation_order():
    class Capture:
        def __matmul__(self, matrix):
            return matrix
    rows, cols = source._scale_factors(Capture(), Capture(), jnp.asarray([3., -3.]))
    expected = 1 / math.sqrt(1 / 3.)
    assert expected != math.sqrt(3.)  # This fixture distinguishes both formulas.
    assert bool(jnp.array_equal(rows, jnp.diag(jnp.asarray([expected, -expected]))))
    assert bool(jnp.array_equal(cols, jnp.eye(2)*expected))


def test_explicit_legacy_arguments_preserved(monkeypatch):
    f = function([poly(1)], [poly(1)], [1.])
    seen = []
    def legacy(self, **kwargs):
        seen.append(kwargs)
        return 'legacy'
    monkeypatch.setattr(Chebfun2, '_legacy_minandmax2', legacy)
    assert f.minandmax2(ngrid=17, n_starts=3) == 'legacy'
    assert seen == [{'ngrid': 17, 'n_starts': 3}]


def test_complex_legacy_adapter_preserved(monkeypatch):
    f = function([poly(1)], [poly(1)], [1j])
    monkeypatch.setattr(Chebfun2, '_legacy_minandmax2', lambda *a, **k: 'legacy')
    assert f.minandmax2() == 'legacy'


def test_complex_zero_takes_native_zero_branch():
    f = function([poly(0)], [poly(1)], [1j])
    values, locations = f.minandmax2()
    assert bool(jnp.array_equal(values, jnp.zeros(2)))
    assert bool(jnp.array_equal(locations, jnp.zeros((2, 2))))


def test_selected_trig_reconstruction():
    ChebfunPref.setDefaults('tech', Trigtech)
    t = Trigtech.from_values(jnp.asarray([1., 0., -1., 0.]))
    out = source._reconstruct([t], -1., 1.)
    assert isinstance(out.funs[0].tech, Trigtech)
    assert len(out) < 4000
    x = jnp.asarray([-.75, -.125, .25, .875])
    assert bool(jnp.all(jnp.abs(jnp.asarray(out(x)).reshape(-1)-t(x)) <= 256*2.**-52))


def test_frontend_conversion_scaling_dispatch_order(monkeypatch):
    calls = []
    f = SimpleNamespace(isempty=lambda: False, approx=SimpleNamespace(
        domain=(2., 6., -3., 5.), rows='r', cols='c', pivots='d', rank=2))
    monkeypatch.setattr(source, '_iszero', lambda a: False)
    monkeypatch.setattr(source, '_isreal', lambda a: True)
    def reconstruct(slices, a, b):
        calls.append(('convert', slices, a, b))
        return slices+'4000'
    def scale(rows, cols, pivots):
        calls.append(('scale', rows, cols, pivots))
        return rows+'scaled', cols+'scaled'
    def higher(approx, rows, cols, **kwargs):
        calls.append(('higher', rows, cols, kwargs))
        return 'answer'
    monkeypatch.setattr(source, '_reconstruct', reconstruct)
    monkeypatch.setattr(source, '_scale_factors', scale)
    monkeypatch.setattr(source, 'source_higher_extrema', higher)
    assert source.source_extrema(f) == 'answer'
    assert calls == [('convert', 'r', 2., 6.), ('convert', 'c', -3., 5.),
        ('scale', 'r4000', 'c4000', 'd'),
        ('higher', 'r4000scaled', 'c4000scaled', {'active_solver': None})]


def test_rank_limit_after_conversion_and_scaling(monkeypatch):
    calls = []
    f = SimpleNamespace(isempty=lambda: False, approx=SimpleNamespace(
        domain=(-1., 1., -1., 1.), rows='r', cols='c', pivots='d', rank=4001))
    monkeypatch.setattr(source, '_iszero', lambda a: False)
    monkeypatch.setattr(source, '_isreal', lambda a: True)
    def reconstruct(slices, a, b):
        calls.append(slices)
        return slices
    def scale(r, c, d):
        calls.append('scale')
        return r, c
    monkeypatch.setattr(source, '_reconstruct', reconstruct)
    monkeypatch.setattr(source, '_scale_factors', scale)
    with pytest.raises(ValueError, match='Rank is too large'):
        source.source_extrema(f)
    assert calls == ['r', 'c', 'scale']


def test_resource_stop_escapes_frontend(monkeypatch):
    class ResourceStop(BaseException):
        pass
    f = function([poly(1)], [poly(0, 1)], [1.])
    def stop(*args):
        raise ResourceStop()
    monkeypatch.setattr(source, '_reconstruct', stop)
    with pytest.raises(ResourceStop):
        f.minandmax2()


def test_separate_requests_execute_both_extrema(monkeypatch):
    f = function([poly(1)], [poly(1)], [1.])
    calls = []
    def extrema(obj):
        calls.append(obj)
        return jnp.asarray([2., 3.]), jnp.asarray([[4., 5.], [6., 7.]])
    monkeypatch.setattr(source, 'source_extrema', extrema)
    mn, xmin = f.min2()
    mx, xmax = f.max2()
    assert len(calls) == 2
    assert mn == 2 and mx == 3
    assert bool(jnp.array_equal(xmin, jnp.asarray([4., 5.])))
    assert bool(jnp.array_equal(xmax, jnp.asarray([6., 7.])))


def test_public_rank_two_quadratic_fallback():
    # (x-1/4)^2 + (y+1/8)^2. Exact coefficients independent of construction.
    f = function([poly(1), poly(33/64, 1/4, 1/2)],
                 [poly(9/16, -1/2, 1/2), poly(1)], [1., 1.])
    values, locations = f.minandmax2()
    scale = 181/64  # exact maximum at (-1,1)
    value_bound = 256*2.**-52*scale
    assert bool(jnp.all(jnp.abs(values-jnp.asarray([0., scale])) <= value_bound))
    # Quadratic Hessian is 2I: value error bounds Euclidean location error.
    assert bool(jnp.linalg.norm(locations[0]-jnp.asarray([1/4, -1/8])) <= math.sqrt(value_bound))
    assert bool(jnp.all(jnp.abs(locations[1]-jnp.asarray([-1., 1.])) <= math.sqrt(value_bound)))
