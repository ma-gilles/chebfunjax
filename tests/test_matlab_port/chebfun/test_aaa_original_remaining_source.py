"""Remaining deterministic aaa.m predicates14,17,18,21--23,25--32.

Native tests/chebfun/test_aaa.m7574c77. Callable cases retain real autoZ;
MATLAB's omitted linspace count is100. Full complex residuals and original
bounds retained. Source15/16 RNG and source16 string API remain unqualified.
"""

import jax.numpy as jnp
import pytest
from jax.scipy.special import gamma

from chebfunjax.utils.aaa import aaa
from chebfunjax.utils.quadrature import chebpts, chebpts_ab


def matlab_gamma(x):
    """Native real gamma pole convention, with JAX finite-value evaluation.

    Actual R2025b primitive proof: tests/test_utils/
    gamma_matlab_primitive_fixture.json. Nonpositive integer poles are+Inf;
    mapping raw JAX NaN there would miss original17's infinite-data clause.
    """
    x = jnp.asarray(x)
    if jnp.iscomplexobj(x):
        raise TypeError("Native real gamma callback requires real arguments")
    poles = jnp.isfinite(x) & (x <= 0) & (x == jnp.floor(x))
    return jnp.where(poles, jnp.inf, gamma(x))


SLOTS = [14, 17, 18, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32]


@pytest.mark.parametrize("slot", SLOTS)
def test_original_remaining(slot):
    if slot == 14:
        r, *_ = aaa(matlab_gamma)
        assert float(jnp.abs(r(1.5) - matlab_gamma(1.5))) < 1e-3
    elif slot == 17:
        z = jnp.linspace(-1.0, 1.0, 100)
        f = matlab_gamma(z)
        assert bool(jnp.isposinf(f[0]))
        r, *_ = aaa(f, z)
        assert float(jnp.abs(r(0.63) - matlab_gamma(0.63))) < 1e-3
    elif slot == 18:
        x = jnp.linspace(0.0, 20.0, 100)
        f = jnp.sin(x) / x
        assert bool(jnp.isnan(f[0]))
        r, *_ = aaa(f, x)
        assert float(jnp.abs(r(2.0) - jnp.sin(2.0) / 2)) < 1e-3
    elif slot in (21, 22):
        options = {"degree": 3} if slot == 21 else {"mmax": 4}
        r, *_ = aaa(jnp.exp, **options)
        xx = jnp.linspace(-1.0, 1.0, 100)
        err = jnp.max(jnp.abs(jnp.exp(xx) - r(xx)))
        ratio = float(err / 1.550669058714149e-7)
        assert ratio < 1.1 if slot == 21 else ratio > 1.1
    elif slot == 23:
        xx = jnp.linspace(-1.0, 1.0, 100)
        r, *_ = aaa(jnp.tanh, xx)
        err1 = jnp.max(jnp.abs(jnp.tanh(xx) - r(xx)))
        r, *_ = aaa(jnp.tanh, xx, mmax=40)
        err2 = jnp.max(jnp.abs(jnp.tanh(xx) - r(xx)))
        assert float(jnp.abs(err2 / err1 - 1)) < 1.01
    elif slot == 25:
        za = jnp.linspace(-3.0, -1.0, 1000)
        zb = jnp.linspace(1.0, 3.0, 1000)
        z = jnp.concatenate((za, zb))
        f = jnp.concatenate((jnp.sign(za), jnp.sign(zb)))
        r, *_ = aaa(f, z, mmax=13, lawson=0)
        err1 = jnp.max(jnp.abs(f - r(z)))
        r, *_ = aaa(f, z, mmax=13)
        err2 = jnp.max(jnp.abs(f - r(z)))
        assert float(jnp.abs(err2 / err1 - 1)) < 1.01
    elif slot in (26, 27):
        z = jnp.linspace(-1.0, 1.0, 1000)
        options = {"degree": 3} if slot == 26 else {"mmax": 4}
        _, pol, *_ = aaa(jnp.exp(z), z, **options)
        assert pol.size == 3
    elif slot == 28:
        x = jnp.array([1.0, 2.0, 3.0])
        f = jnp.array([1.0, 0.0, 0.0])
        r, *_ = aaa(f, x)
        assert float(jnp.linalg.norm(f - r(x))) == 0.0
    elif slot in (29, 30):
        x = chebpts(200)
        f = jnp.maximum(x, 0.0)
        options = {"damping": 0.2} if slot == 29 else {"damping": 0.5, "sign": True}
        r, *_ = aaa(f, x, degree=4, lawson=100, **options)
        err = jnp.max(jnp.abs(f - r(x)))
        assert float(jnp.abs(err - 0.006)) < 0.002
    elif slot == 31:
        # Public mapped nodes: midpoint arithmetic is mathematically
        # equivalent to native scaleNodes, not claimed bitwise identical.
        x = chebpts_ab(1000, 0.0, 10.0)
        f = 1 / (1 + jnp.exp(5 / (x - 2)))
        r, *_ = aaa(f, x, degree=12, lawson=100, damping=0.85, sign=True)
        err = jnp.max(jnp.abs(f - r(x)))
        assert float(jnp.abs(err - 0.000035)) < 0.0001
    else:

        def f(x):
            return jnp.maximum(x, 0.0)

        r, *_ = aaa(f, degree=8, damping=0.5, lawson=200)
        xx = jnp.linspace(-1.0, 1.0, 300)
        err = jnp.max(jnp.abs(f(xx) - r(xx)))
        assert float(jnp.abs(err - 0.0006)) < 0.001
