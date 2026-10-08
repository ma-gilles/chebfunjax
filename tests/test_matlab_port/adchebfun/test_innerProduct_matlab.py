"""All15 clauses of original AD test_innerProduct.m.

Provenance: Chebfun7574c77680d7e82b79626300bf255498271a72df,
tests/adchebfun/test_innerProduct.m and @adchebfun/*TestingBinary.m.
Source1e-14 value/remainder and1e-2 slope bounds unchanged. Fixed degree7
polynomials replace rand(8,1); independent seeds are used, not a shared seed.
"""
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import chebfun


@pytest.fixture(scope='module')
def data():
    u = chebfun(lambda x: .55+.02*x+.01*x**7)
    v = chebfun(lambda x: .56-.015*x+.012*x**6)
    w = chebfun(lambda x: .54+.018*x*x)
    z = chebfun(lambda x: .57-.01*x**3)
    p = chebfun(lambda x: .055+.002*x+.001*x**7)
    q = chebfun(lambda x: .056-.003*x+.001*x**6)
    return u, v, w, z, p, q


def _case(data, index, h=0., value_seeded=False):
    u, v, w, z, p, q = data
    a, b = ADChebfun(u+h*p), ADChebfun(v+h*q)
    if index == 0 or value_seeded:
        a, b = a.seed(1, 2), b.seed(2, 2)
    if index == 0:
        return cj.innerProduct(a, b)
    return (lambda: cj.innerProduct(a, z), lambda: cj.innerProduct(w, b),
            lambda: cj.innerProduct(a, .7), lambda: cj.innerProduct(.6, b))[index-1]()


def _action(result, data, index, h):
    p, q = data[-2:]
    if index == 0:
        return jnp.asarray(result.jacobian*[h*p, h*q]).reshape(())
    return result.jacobian.apply(h*(p if index in (1, 3) else q))


@pytest.mark.parametrize('index', range(5))
def test_source_value_clause(data, index):
    u, v, w, z, _, _ = data
    # Plain source numeric factors are constant functions on the same domain.
    # The current plain .inner API requires Chebfun operands.
    pairs = ((u, v), (u, z), (w, v), (u, chebfun(.7)), (chebfun(.6), v))
    plain = pairs[index][0].inner(pairs[index][1])
    assert abs(float(_case(data, index, value_seeded=True).func-plain)) < 1e-14


@pytest.mark.parametrize('index', range(5))
def test_source_taylor_clause(data, index):
    result = _case(data, index)
    first, second = [], []
    for exponent in range(2, 5):  # Actual source default hMax=3 (doc says4).
        h = .2**exponent
        delta = _case(data, index, h).func-result.func
        first.append(jnp.abs(delta))
        second.append(jnp.abs(delta-_action(result, data, index, h)))
    order1 = jnp.diff(jnp.log(jnp.asarray(first)))/jnp.log(.2)
    if index == 0:
        order2 = jnp.diff(jnp.log(jnp.asarray(second)))/jnp.log(.2)
        assert float(jnp.max(jnp.abs(order1-1))) < 1e-2
        assert float(jnp.max(jnp.abs(order2-2))) < 1e-2
    else:
        # Literal source predicate: max(abs(order1))-1, not abs(order1-1).
        assert float(jnp.max(jnp.abs(order1)))-1 < 1e-2
        assert float(jnp.max(jnp.asarray(second))) < 1e-14


@pytest.mark.parametrize('index', range(5))
def test_source_linearity_clause(data, index):
    assert _case(data, index, value_seeded=True).is_linear == (index != 0)


@pytest.mark.parametrize('reverse', [False, True])
def test_source_ad_complex_bilinear_dispatch(reverse):
    f = chebfun(lambda x: (1+2j)*(1+x))
    g = chebfun(lambda x: (2-1j)*(1-x))
    ad = ADChebfun(g)
    result = cj.innerProduct(f, ad) if reverse else cj.innerProduct(ad, f)
    expected = (f*g).sum()
    assert abs(complex(result.func-expected)) < 1e-12
    assert abs(complex(f.innerProduct(ad).func-expected)) < 1e-12
    # Plain Chebfun output keeps its established Hermitian convention.
    assert abs(complex(cj.innerProduct(f, g)-f.inner(g))) < 1e-12
    assert abs(complex(f.inner(g)-expected)) > 1
