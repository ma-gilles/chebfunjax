"""Port of MATLAB Chebfun tests/misc/test_quantumstates.m (Fable 5).

``quantumstates`` returns ``(eigenvalues, eigenfunctions)``; MATLAB's
``[efuns, evals]`` (quasimatrix, diagonal matrix) order is reversed.

Provenance
----------
MATLAB source : tests/misc/test_quantumstates.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun, quantumstates
from chebfunjax.tech.trigtech import Trigtech

jax.config.update("jax_enable_x64", True)


class TestMiscQuantumstates:
    def test_all_matlab_assertions(self):
        dom = (-3.0, 3.0)
        x = chebfun(lambda t: t, domain=dom)
        V = x ** 2
        evals, efuns = quantumstates(V)
        D = np.asarray(evals)
        assert np.linalg.norm(np.diff(D) - .2) < 1e-12               # pass(1)

        h = .24
        V = .1 * abs(x - 2)
        n = 6
        evals, efuns = quantumstates(V, n, h)
        op = lambda u: -h ** 2 * u.diff(2) + V * u  # noqa: E731
        err = 0.0
        for k in range(n):
            err += float((op(efuns[k]) - efuns[k] * float(evals[k])).norm(2)) ** 2
        assert np.sqrt(err) < 5e-8                                  # pass(2)

        V = chebfun("sin(pi*x/2)^2", trig=True)
        evals, efuns = quantumstates(V)
        e = np.asarray(evals)
        assert abs(e[2] - .68934055) < 1e-3                         # pass(3)
        assert isinstance(efuns[0].funs[0].tech, Trigtech)          # pass(4)
