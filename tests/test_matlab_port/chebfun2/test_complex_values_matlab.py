"""Complex value/coefficient regressions for the MATLAB Chebfun2 constructor.

Provenance
----------
MATLAB source : @chebfun2/constructor.m (constructFromDouble)
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.utils.quadrature import chebpts


@pytest.mark.parametrize('trig', [False, True])
@pytest.mark.parametrize('chop', [False, True])
def test_complex_pivots_preserve_interpolant(trig, chop):
    if trig:
        x = -jnp.pi + 2 * jnp.pi * jnp.arange(24) / 24
        y = -jnp.pi + 2 * jnp.pi * jnp.arange(32) / 32
        domain = (-jnp.pi, jnp.pi, -jnp.pi, jnp.pi)

        def function(a, b):
            return jnp.exp(1j * a) + 0.5j * jnp.cos(2 * b) + 0.3 * jnp.exp(1j * (a - b))
    else:
        x, y = chebpts(24), chebpts(32)
        domain = (-1, 1, -1, 1)

        def function(a, b):
            return (1 + 2j) * a + (2 - 0.5j) * b + 0.3j * a * b

    xx, yy = jnp.meshgrid(x, y)
    f = Chebfun2.from_values(function(xx, yy), domain=domain, trig=trig, chop=chop)
    tolerance = 100 * np.finfo(float).eps * float(jnp.max(jnp.abs(function(xx, yy))))
    np.testing.assert_allclose(f(xx, yy), function(xx, yy), atol=tolerance, rtol=0)
    a, b = jnp.linspace(-0.9, 0.9, 31), jnp.linspace(0.8, -0.8, 31)
    np.testing.assert_allclose(f(a, b), function(a, b), atol=tolerance, rtol=0)
    np.testing.assert_allclose(jax.jit(lambda s, t: f(s, t))(a, b), function(a, b),
                               atol=tolerance, rtol=0)


@pytest.mark.parametrize('chop', [False, True])
@pytest.mark.parametrize('domain', [(-1, 1, -1, 1), (2, 6, -3, 2)])
def test_complex_coefficient_constructor(chop, domain):
    # constructor.m transforms complex coefficient data without narrowing.
    # Direct evaluation of T_2(y)T_3(x) independently checks orientation,
    # domain scaling, transforms and complex low-rank pivots.
    C = jnp.zeros((4, 5), dtype=jnp.complex128)
    C = C.at[0, 0].set(1 + 0.2j).at[0, 1].set(1 + 2j)
    C = C.at[1, 0].set(2 - 0.5j).at[1, 1].set(0.3j)
    C = C.at[2, 3].set(-0.2 + 0.6j)
    f = Chebfun2.from_coeffs(C, domain=domain, chop=chop)
    x, y = jnp.meshgrid(jnp.linspace(domain[0], domain[1], 23),
                        jnp.linspace(domain[2], domain[3], 19))
    tx = (2 * x - domain[0] - domain[1]) / (domain[1] - domain[0])
    ty = (2 * y - domain[2] - domain[3]) / (domain[3] - domain[2])
    expected = (1 + 0.2j + (1 + 2j) * tx + (2 - 0.5j) * ty
                + 0.3j * tx * ty + (-0.2 + 0.6j) * (2 * ty**2 - 1) * (4 * tx**3 - 3 * tx))
    tolerance = 100 * np.finfo(float).eps * float(jnp.max(jnp.abs(expected)))
    np.testing.assert_allclose(f(x, y), expected, atol=tolerance, rtol=0)
    np.testing.assert_allclose(jax.jit(lambda a, b: f(a, b))(x, y), expected,
                               atol=tolerance, rtol=0)
