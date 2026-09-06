"""Port of MATLAB Chebfun tests/linop/test_feval_lr.m (Fable 5).

Provenance
----------
MATLAB source : tests/linop/test_feval_lr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax

import chebfunjax as cj
from chebfunjax.operators.blocks import D, eval_at

jax.config.update("jax_enable_x64", True)

TOL = 1e-12


class TestLinopFevalLr:
    def test_all_matlab_assertions(self):
        # MATLAB err(2), err(3): one-sided evaluation functionals.
        d = (-1.0, 0.0, 1.0)
        x = cj.chebfun(lambda t: t, domain=d)
        s = (x + math.pi / 4).cos() * x.sign() + 0.5

        Ll = eval_at(0.0, d, "left")
        Lr = eval_at(0.0, d, "right")

        cl = -math.cos(math.pi / 4) + 0.5
        cr = math.cos(math.pi / 4) + 0.5

        err = [abs(float(Ll * s) - cl), abs(float(Lr * s) - cr)]

        # MATLAB err(4), err(5): composition with the derivative.
        s = (x + math.pi / 4).cos() * abs(x) + 0.5
        Dop = D(d)
        err.append(abs(float((Ll * Dop) * s) - (-math.sqrt(2) / 2)))
        err.append(abs(float((Lr * Dop) * s) - (math.sqrt(2) / 2)))

        assert all(e < TOL for e in err), err

    def test_undirected_eval_at_breakpoint(self):
        d = (-1.0, 0.0, 1.0)
        x = cj.chebfun(lambda t: t, domain=d)
        s = (x + math.pi / 4).cos() * x.sign() + 0.5
        L = eval_at(0.0, d)
        cl = -math.cos(math.pi / 4) + 0.5
        cr = math.cos(math.pi / 4) + 0.5
        assert abs(float(L * s) - (cl + cr) / 2) < TOL
