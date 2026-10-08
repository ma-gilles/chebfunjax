"""All eight source Airy assertions, including the complex arguments.

Provenance
----------
MATLAB source: tests/chebfun/test_airy.m
Chebfun commit: 7574c77
The source100eps*vscale bound is retained. SciPy supplies independent values;
no fresh native MATLAB output capture is claimed.
"""
# uses-numpy: independent SciPy oracle arrays in numerical tests.
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import airy

from chebfunjax.chebfun1d.chebfun import chebfun


@pytest.mark.parametrize('imaginary', [0, 1])
@pytest.mark.parametrize('kind', [0, 1, 2, 3])
def test_original_source_clause(imaginary, kind):
    x = chebfun(lambda t: t, domain=(-1., 5.))
    f = ((1+imaginary*1j)*x).airy(kind, 0)
    xx = np.linspace(-1., 5., 100)
    expected = airy((1+imaginary*1j)*xx)[kind]
    assert float(jnp.max(jnp.abs(f(jnp.asarray(xx))-expected))) < 100*np.finfo(float).eps*f.vscale
