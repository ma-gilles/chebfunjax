"""All 12 times 3 testUnary clauses at unchanged source bounds.

Provenance: tests/adchebfun/test_deflationFun.m, @adchebfun/testUnary.m,
valueTesting.m, taylorTesting.m; Chebfun commit: 7574c77.
Uses previously captured native seed6179 primitive inputs, not a fresh
native execution of the deflation assertions.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.operators.deflation import deflation_fun

CASES = [(kind, n, p, shift) for kind in ("L2", "H1")
         for n, p, shift in [(1, 2, .5), (2, 2, .5), (3, 2, .5),
                              (3, 3, 2), (2, 1, 0), (2, 1, 4)]]


@pytest.fixture(scope="module")
def inputs():
    raw = json.loads(Path(__file__).with_name("erf_matlab_inputs.json").read_text())
    u, p = jnp.asarray(raw["raw_u"]), jnp.asarray(raw["raw_p"])
    r = chebfun(jnp.sin)
    return (.1*Chebfun.from_values(u)+.5, Chebfun.from_values(.1*u+.5),
            Chebfun.from_values(.01*p+.05), [r, r.exp(), r.sin()])


def operation(u, roots, case):
    kind, count, power, shift = case
    return deflation_fun(u.diff(2)+u.sin(), u, roots[:count], power, shift, kind)


@pytest.mark.parametrize("case", CASES)
def test_source_value(inputs, case):
    u, _, _, roots = inputs
    assert float((operation(u, roots, case)-operation(ADChebfun(u), roots, case).func).norm(jnp.inf)) < 50*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize("case", CASES)
def test_source_taylor(inputs, case):
    _, u, p, roots = inputs
    v = ADChebfun(u)
    base = operation(v, roots, case)
    first, second = [], []
    for exponent in range(2, 6):
        h = .2**exponent*p
        difference = operation(v+h, roots, case).func-base.func
        first.append(difference.norm(jnp.inf))
        second.append((difference-base.jacobian.apply(h)).norm(jnp.inf))
    for errors, order in ((first, 1), (second, 2)):
        orders = jnp.diff(jnp.log(jnp.asarray(errors)))/jnp.diff(jnp.log(.2**jnp.arange(2, 6)))
        assert float(jnp.max(jnp.abs(orders-order))) < 1e-2


@pytest.mark.parametrize("case", CASES)
def test_source_linearity(inputs, case):
    assert not operation(ADChebfun(inputs[0]), inputs[3], case).is_linear
