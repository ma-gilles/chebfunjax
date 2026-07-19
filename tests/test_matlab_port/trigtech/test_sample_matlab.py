"""Port of MATLAB Chebfun tests/trigtech/test_sample.m (Opus 4.8).

sample(f) returns the values and points on the native equispaced grid, and
sample(f, m) aliases the coefficients to length m first.  chebfunjax now
implements ``Trigtech.sample`` (a port of ``@trigtech/sample.m``), so all three
grid cases run: because aliasing to m points reproduces the trigonometric
interpolant, its values at the m-point grid equal ``feval(f, trigpts(m))``.

Provenance
----------
MATLAB source : tests/trigtech/test_sample.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech, trigpts

EPS = float(np.finfo(np.float64).eps)


def _n2(a):
    return float(jnp.linalg.norm(jnp.asarray(a)))


def _f():
    return Trigtech.from_function(lambda x: jnp.exp(jnp.sin(jnp.pi * x - 0.1)))


class TestTrigtechSample:
    def test_native_grid_sample(self):
        f = _f()
        v, p = f.sample()
        p_ex = trigpts(len(f))
        v_ex = f(p_ex)
        assert _n2(p - p_ex) < 100 * EPS
        assert _n2(v - v_ex) < 100 * EPS

    def test_shorter_grid(self):
        f = _f()
        m = round(len(f) / 2)
        v, p = f.sample(m)
        p_ex = trigpts(m)
        v_ex = f(p_ex)
        assert _n2(p - p_ex) < 100 * EPS
        assert _n2(v - v_ex) < 100 * EPS

    def test_longer_grid(self):
        f = _f()
        m = round(2 * len(f))
        v, p = f.sample(m)
        p_ex = trigpts(m)
        v_ex = f(p_ex)
        assert _n2(p - p_ex) < 100 * EPS
        assert _n2(v - v_ex) < 100 * EPS
