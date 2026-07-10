"""Port of MATLAB Chebfun tests/singfun/test_sum.m (Opus 4.8).

Self-validating: each definite integral is checked against the exact value
(from the MATLAB test, computed analytically / in Mathematica) at the SAME
tolerance MATLAB uses.  A Singfun ``s(x)*(1+x)^a*(1-x)^b`` is built with
``Singfun.from_function(f, (a, b))`` where ``f`` evaluates the full product.

Provenance
----------
MATLAB source : tests/singfun/test_sum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.fun.singfun import Singfun

EPS = float(np.finfo(np.float64).eps)

# The order of the exponents (as in the MATLAB test):
A = 0.64
B = -0.64
C = 1.28
D = -1.28


def _sf(f, exps):
    return Singfun.from_function(f, exps)


class TestSingfunSum:
    def test_frac_root_left(self):
        # fractional root at the left endpoint
        f = _sf(lambda x: (1 + x) ** A * jnp.exp(x), (A, 0.0))
        I = float(f.sum())
        I_exact = 2.7263886326217359442
        assert abs(I - I_exact) < 10 * EPS * abs(I_exact)

    def test_frac_pole_left_divergent(self):
        # fractional pole at the left endpoint -> divergent integral
        f = _sf(lambda x: (1 + x) ** D * jnp.sin(x), (D, 0.0))
        I = float(f.sum())
        assert np.isinf(I) and np.sign(I) == np.sign(np.sin(-1.0))

    def test_frac_root_right(self):
        # fractional root at the right endpoint
        f = _sf(lambda x: (1 - x) ** C * jnp.cos(x), (0.0, C))
        I = float(f.sum())
        I_exact = 1.7756234306626192717
        assert abs(I - I_exact) < EPS * abs(I_exact)

    def test_frac_pole_right(self):
        # fractional pole at the right endpoint (integrable, -1 < b < 0)
        f = _sf(lambda x: (1 - x) ** B * (x ** 5), (0.0, B))
        I = float(f.sum())
        I_exact = 1.2101935306745953520
        assert abs(I - I_exact) < 1e1 * EPS * abs(I_exact)

    def test_pole_and_root(self):
        # a combination of fractional pole and fractional root
        f = _sf(lambda x: (1 + x) ** B * jnp.sin(x) * (1 - x) ** C, (B, C))
        I = float(f.sum())
        I_exact = -3.8210796477539148513
        assert abs(I - I_exact) < 1e1 * EPS * abs(I_exact)

    def test_equal_pole_orders(self):
        # pole orders at the two ends are equal (odd integrand -> 0)
        f = _sf(lambda x: jnp.sin(x) * (1 - x ** 2) ** B, (B, B))
        I = float(f.sum())
        assert abs(I - 0.0) < 10 * EPS

    def test_trivial_no_singularity(self):
        # both exponents vanish
        f = _sf(lambda x: jnp.exp(x) * x ** 3 * jnp.sin(2 * x), (0.0, 0.0))
        I = float(f.sum())
        I_exact = 0.644107794617991224
        assert abs(I - I_exact) < 10 * EPS * abs(I_exact)
