"""Original cumsum clauses1–9; native source input grid matched bit for bit.

Provenance
----------
MATLAB source: tests/chebfun/test_cumsum.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = np.finfo(float).eps
XR = jnp.asarray(2 * np.random.RandomState(7681).rand(1000) - 1)
DOM = [-1, -0.5, 0.5, 1]


def test_source_1_to_6():
    assert cj.chebfun().cumsum().isempty(), "source1"
    f1 = cj.chebfun(jnp.cos, domain=DOM)
    integral = f1.cumsum()
    exact = jnp.sin(XR) - jnp.sin(-1.0)
    assert float(jnp.max(jnp.abs(integral(XR) - exact))) < 10 * integral.vscale * EPS, "source2"
    row = f1.T.cumsum()
    # Native vector infinity norm applies to either row or column orientation.
    err = np.asarray(row(XR)).reshape(1, -1) - np.asarray(exact).reshape(1, -1)
    assert np.linalg.norm(err.reshape(-1), np.inf) < 10 * row.vscale * EPS, "source3"
    f3 = cj.chebfun(lambda x: jnp.stack([jnp.cos(x), -jnp.sin(x), jnp.exp(x)], axis=-1), domain=DOM)
    integral = f3.cumsum()
    exact3 = jnp.stack([jnp.sin(XR), jnp.cos(XR), jnp.exp(XR)], axis=-1) - jnp.array(
        [jnp.sin(-1.0), jnp.cos(-1.0), jnp.exp(-1.0)]
    )
    assert float(jnp.max(jnp.abs(integral(XR) - exact3))) < 10 * integral.vscale * EPS, "source4"
    row = f3.T.cumsum()
    assert float(jnp.max(jnp.abs(row(XR) - exact3.T))) < 10 * row.vscale * EPS, "source5"
    f4 = cj.chebfun(lambda x: jnp.stack([jnp.cos(x), -jnp.sin(x)], axis=-1), domain=DOM)
    integral = f4.cumsum()
    # Exact source helper masks both interior branches, which have identical formulas.
    mask = ((-1 < XR) & (XR < -0.5)) | ((-0.5 < XR) & (XR < 1))
    exact4 = jnp.where(
        mask[:, None], jnp.stack([jnp.sin(XR), jnp.cos(XR)], axis=-1), 0
    ) - jnp.array([jnp.sin(-1.0), jnp.cos(-1.0)])
    assert float(jnp.max(jnp.abs(integral(XR) - exact4))) < 10 * integral.vscale * EPS, "source6"


def test_source_7_to_9():
    f1 = cj.chebfun(jnp.cos, domain=DOM)
    # The native test repeats this exact second-integral assertion in slots7/8.
    for slot in (7, 8):
        integral = f1.cumsum(2)
        exact = -jnp.cos(XR) - jnp.sin(-1.0) * XR - (-jnp.cos(-1.0) - jnp.sin(-1.0) * -1)
        assert float(jnp.max(jnp.abs(integral(XR) - exact))) < 10 * integral.vscale * EPS, (
            f"source{slot}"
        )
    f3 = cj.chebfun(lambda x: jnp.stack([jnp.cos(x), -jnp.sin(x), jnp.exp(x)], axis=-1), domain=DOM)
    integral = f3.cumsum(2)
    exact = (
        jnp.stack([-jnp.cos(XR), jnp.sin(XR), jnp.exp(XR)], axis=-1)
        - XR[:, None] * jnp.array([jnp.sin(-1.0), jnp.cos(-1.0), jnp.exp(-1.0)])
        - jnp.array(
            [-jnp.cos(-1.0) + jnp.sin(-1.0), jnp.sin(-1.0) + jnp.cos(-1.0), 2 * jnp.exp(-1.0)]
        )
    )
    assert float(jnp.max(jnp.abs(integral(XR) - exact))) < 10 * integral.vscale * EPS, "source9"
