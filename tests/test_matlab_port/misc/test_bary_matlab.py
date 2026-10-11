"""Literal native bary predicates; RNG adapters use source seed7681.
No claim of MATLAB/NumPy matched draws. Norms/bounds/source sizes preserved.

Provenance
----------
MATLAB source: bary.m and tests/misc/test_bary.m.
Chebfun commit: 7574c77.
"""

import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.interpolation import bary, bary_weights, cheb_bary_weights
from chebfunjax.utils.quadrature import chebpts

RNG = np.random.RandomState(7681)
XR = 2 * RNG.rand(1000) - 1
TOL = 1e-14
p = lambda x: x**4 - 2 * x**3 + 3 * x**2 - 2 * x + 1
p2 = lambda x: 2 * x**4 - 3 * x**3 + x**2 - 5 * x + 2


def q(x):
    if np.ndim(x) < 2:
        return jnp.stack((p(x), p2(x)), axis=-1)
    return jnp.concatenate((p(x), p2(x)), axis=1)


def run_clause(clause):
    if clause == 14:
        x = chebpts(181)
        f = 1 / (1 + 25 * x**2)
        xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        got = bary(jnp.asarray(xx + 1j * yy), f)
        return {"clause": 14, "shape": list(got.shape), "status": "pass"}
    kind = 1 if clause in (3, 4, 10, 11) else 2
    xk = jnp.asarray(chebpts(16, kind=kind))
    vk = jnp.asarray(bary_weights(xk) if kind == 1 else cheb_bary_weights(16))
    if clause in (2, 4, 9, 11):
        x = jnp.asarray(XR[0])
    elif clause in (5, 12):
        x = jnp.asarray(XR.reshape(1, -1))
    elif clause in (6, 13):
        x = jnp.asarray(XR.reshape(100, 10, order="F"))
    elif clause == 7:
        x = jnp.asarray(XR.reshape(10, 10, 10, order="F"))
    else:
        x = jnp.asarray(XR)
    fn = q if clause >= 8 else p
    expected = fn(x)
    got = bary(x, fn(xk), xk, vk)
    assert got.shape == expected.shape, (got.shape, expected.shape)
    err = float(jnp.max(jnp.abs(expected - got)))
    assert err < TOL, (clause, err, TOL)
    return {
        "clause": clause,
        "error": err,
        "bound": TOL,
        "shape": list(got.shape),
        "status": "pass",
    }


import pytest


@pytest.mark.parametrize("clause", range(1, 15))
def test_native_bary_clause(clause):
    run_clause(clause)


def test_exact_node():
    xk = chebpts(16)
    vk = cheb_bary_weights(16)
    got = bary(xk[3:4], p(xk), xk, vk)
    assert abs(float(got[0] - p(xk)[3])) < TOL
