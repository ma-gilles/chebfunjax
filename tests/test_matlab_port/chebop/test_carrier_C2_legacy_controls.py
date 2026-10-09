"""Port of MATLAB Chebfun tests/chebop/test_carrier_C2.m (Fable 5).

Carrier equation with a wiggly initial guess: Newton must stay in the
guess's basin and land on the multi-bump solution pinned by the
hi-quality reference values.

Provenance
----------
MATLAB source : tests/chebop/test_carrier_C2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from chebfunjax.chebfun1d.chebfun import chebfun  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402


class TestChebopCarrierC2:
    def test_all_matlab_assertions(self):
        tol = 1e-10
        dom = (-1.0, 1.0)

        N = Chebop(lambda x, u: 0.01 * u.diff(2)
                   + 2.0 * (1.0 - x ** 2) * u + u ** 2 - 1.0, dom)
        N.bc = lambda x, u: [u(-1.0), u(1.0)]
        x = chebfun(lambda s: s, domain=dom)
        N.init = 2.0 * (x ** 2 - 1.0) * (1.0 - 2.0 / (1.0 + 20.0 * x ** 2))

        u = N.solvebvp(0.0)[0]

        xx = np.arange(-1.0, 1.001, 0.25)
        hiquality_ans = np.array([
            0.0,
            -1.487429807540814,
            -1.785617248281071,
            1.572366197526305,
            -1.539652044363185,
            1.572366197526230,
            -1.785617248281089,
            -1.487429807540795,
            0.0,
        ])
        vals = np.asarray(u(jnp.asarray(xx)))
        assert np.linalg.norm(vals - hiquality_ans) < tol   # pass(1)
        ends = np.asarray(u(jnp.asarray([-1.0, 1.0])))
        assert np.linalg.norm(ends) < tol                   # pass(2)
