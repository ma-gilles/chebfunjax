"""Original ultra2ultra.m four predicates and independent value controls.

Provenance: tests/misc/test_ultra2ultra.m, Chebfun7574c77.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import eval_gegenbauer

from chebfunjax import chebfun
from chebfunjax.utils.transforms import ultra2ultra


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_native_coefficients(complex_input, reverse):
    f = chebfun(lambda x: jnp.exp(1j*x), domain=[0., 3.]) if complex_input else chebfun(jnp.exp)
    lam1, lam2 = (.7, .6) if reverse else (.6, .7)
    c1, c2 = f.ultracoeffs(lam1), f.ultracoeffs(lam2)
    assert float(jnp.linalg.norm(ultra2ultra(c1, lam1, lam2)-c2)) < 100*np.finfo(float).eps


def _basis(n, lam, x):
    if lam == 0:
        return np.polynomial.chebyshev.chebvander(x, n-1)
    return np.stack([eval_gegenbauer(k, lam, x) for k in range(n)], axis=1)


@pytest.mark.parametrize("parameters", [(0., .6), (.6, 0.), (.6, .7)])
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("matrix", [False, True])
def test_values_and_jit(parameters, complex_input, matrix):
    a, b = parameters
    k = np.arange(12, dtype=float)
    c = np.sin(k+.2)/(k+1)**2
    if complex_input:
        c = c + 1j*np.cos(k+.3)/(k+1)**2
    if matrix:
        c = np.stack([c, -.7*c], axis=1)
    convert = jax.jit(lambda values: ultra2ultra(values, a, b))
    out = np.asarray(convert(jnp.asarray(c)))
    x = np.linspace(-.9, .9, 31)
    np.testing.assert_allclose(_basis(12, b, x) @ out, _basis(12, a, x) @ c,
                               rtol=0., atol=500*np.finfo(float).eps)
    assert out.shape == c.shape
    assert np.iscomplexobj(out) == complex_input
