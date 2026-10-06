"""Independent kernel identities and a unique manufactured periodic equation.

Provenance
----------
MATLAB source : @trigcolloc/fred.m, @trigcolloc/functionPoints.m,
    tests/chebop/test_intops.m
Chebfun commit: 7574c77
No source solution coefficients/outputs enter construction. Small matrix
identity checks allow128eps times interval length (trigonometric evaluation
and32-term products/sums); rtol0. The public off-grid diagnostic target1e-10
is fixed independently and does not replace the original source assertions.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop, _FourierProxy
from chebfunjax.operators.integral import fred


@pytest.mark.parametrize('domain', [(-1.0, 1.0), (2.0, 5.0)])
def test_periodic_fredholm_constant_and_rank_one(domain):
    a, b = domain
    length = b - a
    n = 32
    x = a + length * jnp.arange(n) / n
    phase = 2 * jnp.pi * (x - a) / length
    proxy = _FourierProxy(n, length, jnp.eye(n), grid=x)
    constant = fred(lambda xx, yy: jnp.ones_like(xx + yy), proxy)
    bound = 128 * np.finfo(float).eps * length
    np.testing.assert_allclose(constant.mat @ jnp.ones(n), length, atol=bound, rtol=0)
    np.testing.assert_allclose(constant.mat @ jnp.sin(phase), 0, atol=bound, rtol=0)

    def kernel(xx, yy):
        return ((1 + 0.25 * jnp.cos(2 * jnp.pi * (xx - a) / length))
                * (1 + 0.5 * jnp.sin(2 * jnp.pi * (yy - a) / length)))

    # Nonidentity incoming action proves integral composition acts on the
    # existing expression, not always the bare unknown.
    expression = _FourierProxy(n, length, 0.5 * jnp.eye(n), grid=x)
    rank_one = fred(kernel, expression)
    expected = length / 8 * (1 + 0.25 * jnp.cos(phase))
    np.testing.assert_allclose(rank_one.mat @ jnp.sin(phase), expected, atol=bound, rtol=0)
    np.testing.assert_array_equal(rank_one.grid, x)


@pytest.mark.parametrize('domain', [(-1.0, 1.0), (2.0, 5.0)])
def test_periodic_fredholm_fourier_orthogonality(domain):
    a, b = domain
    length = b - a
    n = 32
    x = a + length * jnp.arange(n) / n
    theta = 2 * jnp.pi * (x - a) / length
    proxy = _FourierProxy(n, length, jnp.eye(n), grid=x)
    action = fred(lambda xx, yy: jnp.cos(2 * jnp.pi * (xx - yy) / length), proxy)
    bound = 128 * np.finfo(float).eps * length
    for mode, eigenvalue in [(jnp.ones(n), 0), (jnp.sin(theta), length / 2),
                             (jnp.cos(theta), length / 2), (jnp.sin(3 * theta), 0)]:
        np.testing.assert_allclose(action.mat @ mode, eigenvalue * mode, atol=bound, rtol=0)


def test_unique_periodic_fredholm_equation_off_grid():
    # -u''+u is strictly positive; the cosine kernel adds a nonnegative
    # rank-two action. Thus this check has no source-case9 constant nullspace.
    def kernel(x, y):
        return jnp.cos(jnp.pi * (x - y))
    operator = Chebop(lambda x, u: -u.diff(2) + u + fred(kernel, u), (-1.0, 1.0))
    operator.bc = 'periodic'
    rhs = chebfun(lambda x: (jnp.pi**2 + 2) * jnp.cos(jnp.pi * x)
                 + 0.25 * (4 * jnp.pi**2 + 1) * jnp.sin(2 * jnp.pi * x),
                 domain=(-1.0, 1.0))
    solution = operator.solve(rhs)
    x = jnp.asarray([-0.913, -0.637, -0.219, 0.071, 0.347, 0.683, 0.929])
    expected = jnp.cos(jnp.pi * x) + 0.25 * jnp.sin(2 * jnp.pi * x)
    np.testing.assert_allclose(solution(x), expected, atol=1e-10, rtol=0)
