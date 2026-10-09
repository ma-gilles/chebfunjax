"""Independent literal clauses of MATLAB tests/diskfun/test_constructor.m.

Literal source inputs and predicates; qualified in bounded CPU groups.
All constructor calls use the source-shaped public diskfun factory.
The _cart helper is only the source reference-function wrapper used to
check values, never a constructor adapter. No max_rank cap substitutes
for exact fixed-rank construction.

Provenance
----------
MATLAB source : tests/diskfun/test_constructor.m, @diskfun/constructor.m,
    @diskfun/sample.m, @diskfun/length.m, @separableApprox/rank.m
Chebfun commit: 7574c77
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.diskfun.diskfun import diskfun
from chebfunjax.utils.quadrature import chebpts, trigpts

jax.config.update("jax_enable_x64", True)

# Source factory chebfunpref().cheb2Prefs.chebfun2eps is binary64 eps.
TOL = 2e3 * float(jnp.finfo(jnp.float64).eps)


def _cart(f):
    """Source redefine_function_handle -> diskfun.pol2cartf."""
    return lambda theta, r: f(r * jnp.cos(theta), r * jnp.sin(theta))


def _polar_construct(f, **kwargs):
    return diskfun(f, "polar", **kwargs)


def _cart_construct(f, **kwargs):
    return diskfun(f, **kwargs)


def _sample_error(f, g):
    # Source getPoints(m=7,n=6), meshgrid and infinity norm of flattening.
    theta, _ = trigpts(12, (-jnp.pi, jnp.pi))
    r = chebpts(7)[3:]
    tt, rr = jnp.meshgrid(theta, r)
    return float(jnp.max(jnp.abs(f(tt, rr) - g.fevalm(theta, r))))


def _rank(f):
    return f.numerical_rank()


def _gaussian(x, y):
    return jnp.exp(-10 * ((x - .1) ** 2 + y ** 2))


def _polar_gaussian(theta, r):
    return jnp.exp(-10 * ((r * jnp.cos(theta) - .1) ** 2
                          + (r * jnp.sin(theta)) ** 2))


def test_source_clause_01():
    def f(t, r):
        return (r * jnp.cos(t)) ** 2 + (r * jnp.sin(t)) ** 2
    g = _polar_construct(f)
    assert _sample_error(f, g) < TOL


def test_source_clause_02():
    def f(x, y):
        return jnp.exp(-jnp.cos(jnp.pi * (x + y)))
    g = _cart_construct(f)
    assert _sample_error(_cart(f), g) < TOL


def test_source_clause_03():
    def f(t, r):
        return jnp.cos(jnp.pi * r * jnp.cos(t)) + jnp.sin(5 * r * jnp.sin(t)) - 1
    g = _polar_construct(f)
    assert _sample_error(f, g) < TOL


def test_source_clause_04():
    # Source explicit 'cart' uses the same Cartesian coordinate policy.
    def f(x, y):
        return jnp.exp(y)
    g = diskfun(f, "cart")
    assert _sample_error(_cart(f), g) < TOL


def test_source_clause_05():
    def f(x, y):
        return 1 - jnp.exp(x)
    g = _cart_construct(f)
    assert _sample_error(_cart(f), g) < TOL


def test_source_clause_06():
    def f(x, y):
        return jnp.exp(-x) + jnp.exp(-y)
    g = _cart_construct(f)
    assert _sample_error(_cart(f), g) < TOL


def test_source_clause_07():
    def f(x, y):
        return jnp.cos(2 * x * y)
    g = _cart_construct(f)
    assert _sample_error(_cart(f), g) < TOL


def test_source_clause_08():
    def f(x, y):
        return jnp.sin(x * y)
    g = _cart_construct(f)
    assert _sample_error(_cart(f), g) < TOL


def test_source_clause_09():
    def f(x, y):
        return jnp.sin(11 * jnp.pi * x) - jnp.sin(3 * jnp.pi * x) + jnp.sin(y)
    g = _cart_construct(f)
    assert _sample_error(_cart(f), g) < TOL


def test_source_clause_10():
    def f(x, y):
        return 1 / jnp.cosh(jnp.cos(9 * y) + jnp.sin(7 * (x - 1)))
    g = _cart_construct(f)
    assert _sample_error(_cart(f), g) < 2e1 * TOL


def test_source_clause_11():
    def f(x, y):
        return 0 * x
    g = _cart_construct(f)
    assert float(g.norm("inf")) == 0.0


def test_source_clause_12():
    def f(x, y):
        return 0 * x
    h = _polar_construct(f)
    assert float(h.norm("inf")) == 0.0


def test_source_clause_13():
    f = _cart_construct(lambda x, y: jnp.cos(x))
    g = _cart_construct(lambda x, y: jnp.cos(x), vectorize=True)
    assert float((f - g).norm()) < TOL


def test_source_clause_14():
    f = _polar_construct(lambda t, r: r * jnp.sin(t))
    # Source r*sin(t) is non-vectorized matrix multiplication; vectorize
    # must scalarize it. jnp.matmul preserves that distinction on arrays.
    def nonvectorized(t, r):
        if jnp.ndim(t) == 0 and jnp.ndim(r) == 0:
            return r * jnp.sin(t)
        return jnp.matmul(r, jnp.sin(t))
    g = _polar_construct(nonvectorized, vectorize=True)
    assert float((f - g).norm()) < TOL


def test_source_clause_15():
    f = _cart_construct(lambda x, y: x * y)
    def nonvectorized(x, y):
        if jnp.ndim(x) == 0 and jnp.ndim(y) == 0:
            return x * y
        return jnp.matmul(x, y)
    g = _cart_construct(nonvectorized, vectorize=True)
    assert float((f - g).norm()) < TOL


def test_source_clause_16():
    f = _cart_construct(lambda x, y: 1)
    g = _cart_construct(lambda x, y: 1, vectorize=True)
    assert float((f - g).norm()) < TOL


def test_source_clause_17():
    f = _cart_construct(lambda x, y: 1 + x * jnp.sin((x - .5) * y))
    m, n = f.length()
    samples = f.sample(m + m % 2, n)
    g = diskfun(samples)
    assert float((f - g).norm()) < TOL


def test_source_clause_18():
    f = _cart_construct(lambda x, y: 1 + 0 * x)
    g = diskfun(jnp.ones((1, 1), dtype=jnp.float64))
    assert float((f - g).norm()) < TOL


def test_source_clause_19():
    f = _cart_construct(_gaussian)
    coefficients = f.coeffs2()
    g = diskfun(coefficients, "coeffs")
    assert float((f - g).norm()) < TOL


def test_source_clause_20():
    f = diskfun(_gaussian, 5)
    assert _rank(f) == 5


def test_source_clause_21():
    f = diskfun(_gaussian, 6)
    assert _rank(f) == 6


def test_source_clause_22():
    f = _cart_construct(_gaussian)
    g = diskfun(f, 7)
    assert _rank(g) == 7


def test_source_clause_23():
    f = diskfun(_polar_gaussian, 5, "polar")
    assert _rank(f) == 5


def test_source_clause_24():
    # Source f is the polar fixed-rank-five object from clause23.
    f = diskfun(_polar_gaussian, 5, "polar")
    g = diskfun(f, 0)
    # Rank-zero is a zero FUNCTION, not an empty-object substitute.
    assert _rank(g) == 0


def test_source_clause_25():
    f = diskfun(_polar_gaussian, 5, "polar")
    g = diskfun(f, 0)
    assert float(g.norm()) < TOL


def test_source_clause_26():
    # Source ff remains the two-variable polar Gaussian; no 'polar' flag
    # is present in this invalid-rank call, so retain Cartesian default.
    with pytest.raises(Exception) as caught:
        diskfun(_polar_gaussian, -1)
    assert getattr(caught.value, "identifier", None) == (
        "CHEBFUN:DISKFUN:constructor:parseInputs:domain3"
    )


def test_source_clause_27():
    f = _cart_construct(_gaussian)
    g = diskfun(_gaussian, "eps", 1e-5)
    assert _rank(g) < _rank(f)


def test_source_clause_28():
    f = _cart_construct(_gaussian)
    g = diskfun(_gaussian, "eps", 1e-5)
    mf, nf = f.length()
    mg, ng = g.length()
    assert mg < mf and ng < nf


def test_source_clause_29():
    f = diskfun(jnp.asarray(1, dtype=jnp.float64))
    assert float((f - 1).norm("inf")) == 0.0


def test_source_clause_30():
    f = _cart_construct(_gaussian)
    g = diskfun("exp(-10*((x-.1).^2 + y.^2))")
    assert float((f - g).norm("inf")) == 0.0


def test_source_clause_31():
    f = _polar_construct(lambda t, r: r * jnp.cos(t))
    # The source alphabetic variables are x=theta and y=radius.
    g = diskfun("y.*cos(x)", "polar")
    assert float((f - g).norm("inf")) == 0.0


def test_source_clause_32():
    with pytest.raises(Exception) as caught:
        diskfun("x.*y.*z")
    assert getattr(caught.value, "identifier", None) == (
        "CHEBFUN:DISKFUN:constructor:str2op:depvars"
    )


def test_source_clause_33():
    f = diskfun(jnp.zeros((5, 4), dtype=jnp.float64))
    n, m = f.length()
    assert m == 5 and n == 4


def test_source_clause_34():
    m, n = 24, 10
    f = diskfun(
        lambda x, y: jnp.exp(-10 * ((x - .5 / jnp.sqrt(2)) ** 2
                                  + (y - .5 / jnp.sqrt(2)) ** 2)),
        (m, n),
    )
    nf, mf = f.length()
    assert mf == m + 1 and nf == 2 * n
