"""Fixed-zero callback/population semantics from Chebfun7574c77.

Source @chebtech1/chebtech1.m, @chebtech2/chebtech2.m,
@chebtech/populate.m, @chebtech/prolong.m and @chebtech/vscale.m.
Empty scalar coefficients use the existing Python column-vector adapter.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_zero_grid_calls_operator_before_numeric_population(tech):
    calls = []

    def op(x):
        calls.append(x)
        return jnp.sin(x)

    f = tech.from_function(op, n=0)
    assert len(calls) == 1 and calls[0].shape == (0,)
    assert f.coeffs.shape == (0,) and f.coeffs.dtype == jnp.float64
    assert f.ishappy is True and f.isempty() and f.vscale == 0.


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_empty_complex_samples_preserve_native_numeric_shape_dtype(tech):
    calls = []

    def op(x):
        calls.append(x)
        return jnp.empty((0, 3), dtype=jnp.complex128)

    f = tech.from_function(op, n=0)
    assert len(calls) == 1 and calls[0].shape == (0,)
    assert f.coeffs.shape == (0, 3) and f.coeffs.dtype == jnp.complex128
    assert f.ishappy is True and f.vscale == 0.


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('data,shape', [(7., (0,)), ([[1., 2.]], (0, 2)),
                                      ([[1., 2.], [3., 4.]], (0, 2))])
def test_nonempty_callback_result_is_populated_then_prolonged(tech, data, shape, monkeypatch):
    calls = []
    populations = []
    original = tech.from_values

    def populate(cls, values):
        populations.append(values)
        return original(values)

    monkeypatch.setattr(tech, 'from_values', classmethod(populate))

    def op(x):
        calls.append(x)
        return jnp.asarray(data)

    f = tech.from_function(op, n=0)
    assert len(calls) == 1 and calls[0].shape == (0,)
    assert len(populations) == 1
    assert jnp.array_equal(populations[0], jnp.atleast_1d(jnp.asarray(data)))
    assert f.coeffs.shape == shape and f.ishappy is True
    assert f.n == 0 and f.isempty() and f.vscale == 0.


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_zero_grid_callback_exception_propagates(tech):
    calls = []

    def op(x):
        calls.append(x)
        raise RuntimeError('empty-grid callback error')

    with pytest.raises(RuntimeError, match='empty-grid callback error'):
        tech.from_function(op, n=0)
    assert len(calls) == 1 and calls[0].shape == (0,)


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_empty_vscale_including_existing_null_adapter(tech):
    assert tech.empty().vscale == 0.
    for shape in [(0,), (0, 0), (0, 3), (3, 0)]:
        assert tech.from_coeffs(jnp.empty(shape)).vscale == 0.
