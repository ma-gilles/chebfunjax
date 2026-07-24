"""Port of MATLAB Chebfun tests/trigtech/test_circconv.m (Opus 4.8[1m]).

circconv is periodic (circular) convolution of two trigtechs, realised as
coefficient multiplication scaled by the mode norm.

Provenance
----------
MATLAB source : tests/trigtech/test_circconv.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
# A single deterministic test point in (-1, 1) (stands in for 2*rand-1).
X0 = 0.371542

def _tt(f):
    return Trigtech.from_function(f)


def _scalar(a):
    return complex(jnp.asarray(a).ravel()[0])


class TestTrigtechCircconv:
    def test_empty(self):
        g = _tt(lambda x: jnp.sin(jnp.pi * x))
        empty = Trigtech.empty()
        assert empty.circconv(g).isempty()
        assert g.circconv(empty).isempty()

    def test_odd_functions_zero(self):
        f = _tt(lambda x: jnp.tanh(5 * jnp.cos(jnp.pi * x)))
        g = _tt(lambda x: jnp.ones_like(x))
        hfg = f.circconv(g)
        hgf = g.circconv(f)
        assert abs(_scalar(hfg(jnp.array([X0])))) < 1e2 * EPS
        assert abs(_scalar(hgf(jnp.array([X0])))) < 1e2 * EPS

    def test_self_convolution_at_zero(self):
        f = _tt(lambda x: jnp.tanh(jnp.cos(jnp.pi * x)))
        g = f.circconv(f)
        approx = _scalar(g(jnp.array([0.0])))
        expected = _scalar((f * f).sum())
        assert abs(approx - expected) < 1e2 * EPS * g.vscale

    def test_self_convolution_at_x(self):
        f = _tt(lambda x: jnp.tanh(jnp.cos(jnp.pi * x)))
        g = f.circconv(f)
        approx = _scalar(g(jnp.array([X0])))
        h = _tt(lambda x: jnp.tanh(jnp.cos(jnp.pi * (x - X0))))
        expected = _scalar((f * h).sum())
        assert abs(approx - expected) < 1e2 * EPS * g.vscale
