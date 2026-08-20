"""Port of MATLAB Chebfun tests/chebop/test_vectorizeOp.m (Fable 5).

MATLAB rewrites the op string (* -> .*, ^ -> .^); Python chebfun
operators are already elementwise, so vectorizeOp is the identity and
the string-comparison passes reduce to the value-equality checks.

Provenance
----------
MATLAB source : tests/chebop/test_vectorizeOp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

D = (1.0, 3.0)


class TestChebopVectorizeop:
    def test_all_matlab_assertions(self):
        x = cj.chebfun(lambda t: t, domain=D)
        f = (4 * x).sin().exp()
        u = (5 * x ** 2).sin()
        v = (x / 2).exp()
        b = 4 * np.pi
        c = -5 * np.e

        cases = [
            lambda w: w.diff(2) + b * w ** 3,
            lambda w: w.diff(2) - c * w * w.diff(),
            lambda w: f * w.diff(2) + (w ** 2) / (5 + w),
        ]
        for fun in cases:
            vec = Chebop.vectorizeOp(fun)
            assert float((fun(u) - vec(u)).norm()) == 0.0

        sys_fun = lambda w, z: [w.diff(2) - z * w.diff(),
                                z.diff(2) + w ** 2]
        vec = Chebop.vectorizeOp(sys_fun)
        outs_a = sys_fun(u, v)
        outs_b = vec(u, v)
        for a_, b_ in zip(outs_a, outs_b):
            assert float((a_ - b_).norm()) == 0.0
