"""Pinned7574c77 @trigtech/prolong.m zero-output source error boundary."""
import jax.numpy as jnp
import pytest

from chebfunjax.tech._trig_constructor import _prolong, construct
from chebfunjax.tech.trigtech import Trigtech, _trig_prolong_coeffs


@pytest.mark.parametrize('count', [1, 2, 3])
@pytest.mark.parametrize('complex_storage', [False, True])
def test_nonempty_prolong_zero_reaches_native_index_error(count, complex_storage):
    coeffs = jnp.arange(1., count + 1.)
    if complex_storage:
        coeffs = coeffs + 2j * coeffs
    f = Trigtech(coeffs=coeffs, real_columns=(not complex_storage,),
                 ishappy=True)
    # n=2 first splits the Nyquist row and becomes n=3. For every positive
    # input here, the subsequent deletions leave zero rows, kup<kdown, and
    # source RHS coeffs(1,:) is an indexing error. No error text is guessed.
    with pytest.raises(IndexError):
        _trig_prolong_coeffs(coeffs, 0)
    with pytest.raises(IndexError):
        f.prolong(0)
    with pytest.raises(IndexError):
        _prolong(f, 0)
    # Actual numeric population must reach the same boundary without
    # invoking adaptive callbacks, even with a deliberately invalid checker.
    def forbidden(*args):
        raise AssertionError('adaptive callback was entered')
    with pytest.raises(IndexError):
        construct(coeffs, pref={'fixedLength': 0, 'happinessCheck': forbidden,
                                'refinementFunction': forbidden})


@pytest.mark.parametrize('complex_storage', [False, True])
def test_empty_prolong_zero_preserves_object_and_cache(complex_storage):
    dtype = jnp.complex128 if complex_storage else jnp.float64
    data = jnp.empty((0, 3), dtype=dtype)
    f = Trigtech(coeffs=data, real_columns=(), ishappy=True, _values=data)
    assert f.prolong(0) is f
    assert _prolong(f, 0) is f
    assert f.coeffs.shape == f.values.shape == (0, 3)
    assert f.coeffs.dtype == f.values.dtype == dtype
    assert f._values is data and f.real_columns == () and f.ishappy
