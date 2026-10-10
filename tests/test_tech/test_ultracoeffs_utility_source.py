"""Coefficient-array adapter for native ultracoeffs.m7574c77 scaling."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import eval_gegenbauer

from chebfunjax.utils.transforms import ultracoeffs


@pytest.mark.parametrize("lam", [.5, 1., .6])
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("matrix", [False, True])
def test_ultraspherical_values(lam, complex_input, matrix):
    k = np.arange(12, dtype=float)
    c = np.cos(k+.4)/(k+1)**2
    if complex_input:
        c = c+1j*np.sin(k+.2)/(k+1)**2
    if matrix:
        c = np.stack([c, -.3*c], axis=1)
    out = np.asarray(jax.jit(lambda values: ultracoeffs(values, lam))(jnp.asarray(c)))
    x = np.linspace(-.9, .9, 31)
    target = np.stack([eval_gegenbauer(i, lam, x) for i in range(12)], axis=1)
    truth = np.polynomial.chebyshev.chebvander(x, 11) @ c
    np.testing.assert_allclose(target @ out, truth, rtol=0., atol=500*np.finfo(float).eps)
    assert out.shape == c.shape
    assert np.iscomplexobj(out) == complex_input


@pytest.mark.parametrize("n", [0, 1, 2])
@pytest.mark.parametrize("matrix", [False, True])
def test_second_kind_short_complex_series(n, matrix):
    c = np.asarray([2.+3j, -.5+2j][:n], dtype=np.complex128)
    if matrix:
        c = np.stack([c, -c], axis=1)
    out = np.asarray(jax.jit(lambda values: ultracoeffs(values, 1.))(jnp.asarray(c)))
    assert out.shape == c.shape
    assert np.iscomplexobj(out)
    if n:
        x = np.linspace(-.8, .8, 7)
        target = np.stack([eval_gegenbauer(i, 1., x) for i in range(n)], axis=1)
        np.testing.assert_allclose(target @ out,
                                   np.polynomial.chebyshev.chebvander(x, n-1) @ c,
                                   rtol=0., atol=10*np.finfo(float).eps)
