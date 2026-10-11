"""Original Chebfun cumsum clauses20–29; native source inputs matched bit for bit.

Provenance
----------
MATLAB source: tests/chebfun/test_cumsum.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.utils.quadrature import trigpts

EPS = np.finfo(float).eps
XR = jnp.pi * jnp.asarray(2 * np.random.RandomState(7681).rand(1000) - 1)
DOM = (-jnp.pi, jnp.pi)


def test_source_20_to_23():
    op = lambda x: jnp.cos(x + jnp.sqrt(3.0)) + jnp.sin(x - jnp.sqrt(2.1))  # noqa: E731 - native anonymous function
    f = cj.chebfun(op, domain=DOM, trig=True)
    integral = f.cumsum()
    exact = (
        jnp.sin(XR + jnp.sqrt(3.0))
        - jnp.cos(XR - jnp.sqrt(2.1))
        - (jnp.sin(-jnp.pi + jnp.sqrt(3.0)) - jnp.cos(-jnp.pi - jnp.sqrt(2.1)))
    )
    assert float(jnp.max(jnp.abs(integral(XR) - exact))) < 10 * integral.vscale * EPS, "source20"
    assert integral.isPeriodicTech(), "source21"
    f = cj.chebfun(lambda x: 1 + op(x), domain=DOM, trig=True)
    integral = f.cumsum()
    exact = (
        XR
        + jnp.sin(XR + jnp.sqrt(3.0))
        - jnp.cos(XR - jnp.sqrt(2.1))
        - (-jnp.pi + jnp.sin(-jnp.pi + jnp.sqrt(3.0)) - jnp.cos(-jnp.pi - jnp.sqrt(2.1)))
    )
    assert float(jnp.max(jnp.abs(integral(XR) - exact))) < 10 * integral.vscale * EPS, "source22"
    assert not integral.isPeriodicTech(), "source23"


def test_source_24_to_25():
    f = cj.chebfun(
        lambda x: jnp.stack([jnp.cos(x), -jnp.sin(x), jnp.cos(x + jnp.sqrt(3.0))], axis=-1),
        domain=DOM,
        trig=True,
    )
    integral = f.cumsum()
    exact = jnp.stack([jnp.sin(XR), jnp.cos(XR), jnp.sin(XR + jnp.sqrt(3.0))], axis=-1) - jnp.array(
        [jnp.sin(-jnp.pi), jnp.cos(-jnp.pi), jnp.sin(-jnp.pi + jnp.sqrt(3.0))]
    )
    assert float(jnp.max(jnp.abs(integral(XR) - exact))) < 10 * integral.vscale * EPS, "source24"
    assert f.isPeriodicTech(), "source25 checks input f3 in native source"


def test_source_26_to_27():
    f = cj.chebfun(jnp.ones((32, 1)), domain=DOM, trig=True)
    integral = f.cumsum()
    assert (
        float(jnp.max(jnp.abs(integral(XR) - (XR + jnp.pi)[:, None]))) < 10 * integral.vscale * EPS
    ), "source26"
    assert not integral.isPeriodicTech(), "source27"


def test_source_28_to_29():
    # Source comment says even length, but actually calls trigpts(33); preserve33.
    f = cj.chebfun(jnp.sin(trigpts(33, DOM)[0]), domain=DOM, trig=True)
    integral = f.cumsum()
    assert (
        float(jnp.max(jnp.abs(integral(XR) - (-jnp.cos(XR) - 1)))) < 10 * integral.vscale * EPS
    ), "source28"
    assert integral.isPeriodicTech(), "source29"
