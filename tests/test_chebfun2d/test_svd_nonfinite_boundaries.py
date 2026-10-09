"""Stage-aware Python capability boundaries, not MATLAB exception assertions."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d import _svd
from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech, _trig_inner_product_jax

BAD = [float('nan'), complex(float('inf'), 1)]


def tech(cls, value, columns=False):
    coeffs = jnp.asarray([[value, 1.0]] if columns else [value])
    if cls is Trigtech:
        return cls(coeffs=coeffs, real_columns=(False, False) if columns else (False,))
    return cls(coeffs=coeffs)


def one():
    return Chebtech2(coeffs=jnp.ones(1))


def approximation(column, weight):
    return SeparableApprox(cols=[column], rows=[one()], pivots=jnp.asarray([weight]),
                           domain=(-1., 1., -1., 1.))


@pytest.mark.parametrize('cls', [Chebtech2, Trigtech])
@pytest.mark.parametrize('bad', BAD)
def test_nonfinite_scalar_reaches_normalization_boundary(cls, bad):
    with pytest.raises(NotImplementedError, match='scalar QR normalization'):
        _svd._axis_qr(tech(cls, bad), (-1., 1.))


@pytest.mark.parametrize('bad', BAD)
def test_zero_weights_values_only_precede_factor_boundary(bad, monkeypatch):
    a = approximation(tech(Trigtech, bad), 0.0)
    calls = []
    monkeypatch.setattr(_svd, '_axis_qr', lambda *args: calls.append(args))
    result = _svd.source_svd(a)
    assert jnp.array_equal(result, jnp.zeros(1))
    assert calls == []
    with pytest.raises(NotImplementedError, match='full-zero factor construction'):
        _svd.source_svd(a, full=True)
    assert calls == []


@pytest.mark.parametrize('cls', [Chebtech2, Trigtech])
@pytest.mark.parametrize('bad', BAD)
def test_nonfinite_multicolumn_boundary(cls, bad):
    with pytest.raises(NotImplementedError, match='multicolumn QR input'):
        _svd._axis_qr(tech(cls, bad, columns=True), (-1., 1.))


@pytest.mark.parametrize('bad', [float('nan'), float('inf'),
                               complex(float('nan'), 1), complex(1, float('inf'))])
def test_nonfinite_core_is_not_sent_to_provider(bad, monkeypatch):
    calls = []
    monkeypatch.setattr(_svd, '_core_provider', lambda *args: calls.append(args))
    with pytest.raises(NotImplementedError, match='small-core SVD input'):
        _svd._core_svd(jnp.asarray([[bad]]), jnp.ones(1), jnp.ones((1, 1)))
    assert calls == []


def test_scalar_inner_preserves_false_equality_complex_result():
    f = tech(Trigtech, complex(float('nan'), 1))
    matrix, equal = _trig_inner_product_jax(f, f)
    assert not bool(equal)
    result = _svd._scalar_inner(f)
    assert jnp.iscomplexobj(result)
    assert jnp.array_equal(result, matrix.reshape(()), equal_nan=True)


def test_nonfinite_weights_reach_core_after_both_axis_qrs(monkeypatch):
    calls = []
    original = _svd._axis_qr

    def observe(panel, interval):
        calls.append(interval)
        return original(panel, interval)

    monkeypatch.setattr(_svd, '_axis_qr', observe)
    with pytest.raises(NotImplementedError, match='small-core SVD input'):
        _svd.source_svd(approximation(one(), float('nan')))
    assert calls == [(-1., 1.), (-1., 1.)]


def test_scalar_nan_cache_not_rejected_when_norm_regenerates_values():
    # Scalar Trig innerProduct prolongs, replacing the old cache. Its finite
    # norm reaches the core even though normalized Q retains a NaN cache.
    f = Trigtech(coeffs=jnp.ones(1), real_columns=(True,), _values=jnp.asarray([jnp.nan]))
    q, r = _svd._axis_qr(f, (-1., 1.))
    assert jnp.isfinite(r[0, 0])
    assert jnp.isnan(q.funs[0].tech.values[0])
    assert jnp.isfinite(_svd.source_svd(approximation(f, 1.0))[0])


def test_nonfinite_provider_output_is_not_silently_returned(monkeypatch):
    monkeypatch.setattr(_svd, '_core_provider', lambda core:
                        (jnp.ones((1, 1)), jnp.asarray([jnp.nan]), jnp.ones((1, 1))))
    with pytest.raises(NotImplementedError, match='small-core SVD output'):
        _svd._core_svd(jnp.ones((1, 1)), jnp.ones(1), jnp.ones((1, 1)))
