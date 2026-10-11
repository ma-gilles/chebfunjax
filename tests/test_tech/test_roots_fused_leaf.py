import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech import chebtech as module


def _raw_bytes(value):
    return np.asarray(jax.device_get(value)).tobytes()


@pytest.mark.parametrize('dtype', [jnp.float64, jnp.complex128])
@pytest.mark.parametrize('size', [3, 7, 24])
def test_fused_leaf_outputs_match_two_phase_bits(dtype, size):
    c = jnp.linspace(0.25, 1.0, size).astype(dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        c = c * (1.0 + 0.25j)
    matrix, matrix_finite = module._roots_matrix_and_finite_jax(c)
    expected = module._roots_eig_and_metadata_jax(matrix)
    actual = module._roots_matrix_eig_and_metadata_jax(c)
    assert bool(matrix_finite) == bool(actual[0])
    assert _raw_bytes(matrix) == _raw_bytes(module._roots_colleague_matrix_jax(c))
    for got, want in zip(actual[1:], expected):
        assert str(got.dtype) == str(want.dtype)
        assert _raw_bytes(got) == _raw_bytes(want)


def test_nonfinite_matrix_raises_before_eigensolver_branch(monkeypatch):
    original_eigvals = module._roots_eigvals_jax
    callbacks = []

    def fail_if_called(_):
        callbacks.append('trace-or-runtime')
        raise RuntimeError('eigensolver branch executed')

    def observed_eigvals(matrix):
        jax.debug.callback(fail_if_called, matrix)
        return original_eigvals(matrix)

    # Force this trace even if an earlier test compiled the same leaf shape,
    # then discard the instrumented executable before restoring the provider.
    module._roots_matrix_eig_and_metadata_jax.clear_cache()
    with monkeypatch.context() as patch:
        patch.setattr(module, '_roots_eigvals_jax', observed_eigvals)
        try:
            coeffs = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
            with pytest.raises(np.linalg.LinAlgError, match='Array must not contain infs or NaNs'):
                module._roots_default_eigenvalues(coeffs)
            assert callbacks == [], 'the eig branch callback must not execute'
        finally:
            module._roots_matrix_eig_and_metadata_jax.clear_cache()
