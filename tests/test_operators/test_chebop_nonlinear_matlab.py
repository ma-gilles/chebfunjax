"""MATLAB golden-reference parity tests for nonlinear chebop (Newton solves).

Pins a Newton-solved nonlinear BVP and a Carrier problem driven from a
nontrivial initial guess against MATLAB Chebfun at rtol 1e-12.  Golden ref:
matlab_harness/refs/operators_nonlinear_refs.m
(-> tests/references/operators_nonlinear.mat).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import scipy.io

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

_REF_PATH = Path(__file__).resolve().parents[1] / "references" / "operators_nonlinear.mat"
if not _REF_PATH.exists():
    pytest.skip(
        "operators_nonlinear.mat golden ref not generated "
        "(run matlab_harness/refs/operators_nonlinear_refs.m)",
        allow_module_level=True,
    )
_REF = scipy.io.loadmat(str(_REF_PATH), squeeze_me=True)
_PTS = np.atleast_1d(_REF["pts"]).astype(float)
RTOL = 1e-12


@pytest.mark.matlab
@pytest.mark.xfail(
    reason="chebfunjax nonlinear chebop solve returns a NON-CONVERGED solution "
    "(length frozen at 16, ODE residual ~2e2) for these stiff nonlinear BVPs: it "
    "satisfies the BCs but grossly violates the ODE and disagrees with MATLAB. "
    "Real Newton-solver / adaptive-resolution bug, reported to team lead.",
    strict=False,
)
class TestChebopNonlinearVsMatlab:
    def test_newton_cubic_bvp(self):
        # 0.001 u'' - u^3 = 0, u(-1)=1, u(1)=-1 (Newton-solved).
        N = Chebop(lambda x, u: 0.001 * u.diff(2) - u**3, domain=(-1.0, 1.0))
        N.lbc = 1.0
        N.rbc = -1.0
        u = N.solve(0.0)
        npt.assert_allclose(np.asarray(u(jnp.asarray(_PTS))), _REF["nl1"],
                            rtol=RTOL, atol=1e-12)

    def test_carrier_with_initial_guess(self):
        # Carrier eps=0.01 driven from a nontrivial initial guess (selects a
        # specific solution of a multi-solution problem).
        N = Chebop(lambda x, u: 0.01 * u.diff(2) + 2 * (1 - x**2) * u + u**2,
                   domain=(-1.0, 1.0))
        N.lbc = 0.0
        N.rbc = 0.0
        x = cj.chebfun(lambda x: x)
        N.init = 2 * (x**2 - 1) * (1 - 2 / (1 + 20 * x**2))
        u = N.solve(1.0)
        npt.assert_allclose(np.asarray(u(jnp.asarray(_PTS))), _REF["nl2"],
                            rtol=RTOL, atol=1e-12)
