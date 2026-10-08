"""Deterministic inputs and actions for original AD binary source clauses.

Provenance: @adchebfun/valueTestingBinary.m and taylorTestingBinary.m,
Chebfun 7574c77680d7e82b79626300bf255498271a72df.
Fixed degree-seven polynomials replace rand(8,1); no RNG parity is claimed.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import chebfun


@pytest.fixture(scope="module")
def data():
    u = chebfun(lambda x: .55+.02*x+.01*x**7)
    v = chebfun(lambda x: .56-.015*x+.012*x**6)
    w = chebfun(lambda x: .54+.018*x*x)
    z = chebfun(lambda x: .57-.01*x**3)
    p = chebfun(lambda x: .055+.002*x+.001*x**7)
    q = chebfun(lambda x: .056-.003*x+.001*x**6)
    return u, v, w, z, p, q


def evaluate(operation, data, index, h=0., value_seeded=False):
    u, v, w, z, p, q = data
    a, b = ADChebfun(u+h*p), ADChebfun(v+h*q)
    if index == 0 or value_seeded:
        a, b = a.seed(1, 2), b.seed(2, 2)
    pairs = ((a, b), (a, z), (w, b), (a, .7), (.6, b))
    return operation(*pairs[index])


def value_error(operation, data, index):
    u, v, w, z, _, _ = data
    pairs = ((u, v), (u, z), (w, v), (u, .7), (.6, v))
    expected = operation(*pairs[index])
    actual = evaluate(operation, data, index, value_seeded=True).func
    # valueTestingBinary uses the default norm, unlike unary valueTesting.
    return float((actual-expected).norm())


def taylor_errors(operation, data, index):
    p, q = data[-2:]
    result = evaluate(operation, data, index)
    first, second = [], []
    for exponent in range(2, 5):  # Literal default hMax=3; source doc says4.
        h = .2**exponent
        delta = evaluate(operation, data, index, h).func-result.func
        if index == 0:
            action = (result.jacobian*[h*p, h*q]).blocks[0][0]
        else:
            action = result.jacobian.apply(h*(p if index in (1, 3) else q))
        first.append(delta.norm(jnp.inf))
        second.append((delta-action).norm(jnp.inf))
    order1 = jnp.diff(jnp.log(jnp.asarray(first)))/jnp.log(.2)
    order2 = jnp.diff(jnp.log(jnp.asarray(second)))/jnp.log(.2)
    return order1, order2, jnp.asarray(second)
