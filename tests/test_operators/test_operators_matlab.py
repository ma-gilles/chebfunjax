"""MATLAB golden-reference parity tests for the operators module (chebop/linop).

Pins linear BVP solutions at rtol 1e-12 and a Dirichlet-Laplacian eigenvalue
spectrum at a documented looser tolerance (eigensolvers are less precise than
function evaluation — Gate 3).  Golden ref: matlab_harness/refs/operators_refs.m
(-> tests/references/operators.mat).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import scipy.io

from chebfunjax.operators.chebop import Chebop

_REF_PATH = Path(__file__).resolve().parents[1] / "references" / "operators.mat"
if not _REF_PATH.exists():
    pytest.skip(
        "operators.mat golden ref not generated (run matlab_harness/refs/operators_refs.m)",
        allow_module_level=True,
    )
_REF = scipy.io.loadmat(str(_REF_PATH), squeeze_me=True)
_PTS = np.atleast_1d(_REF["pts"]).astype(float)
_PTS01 = np.atleast_1d(_REF["pts01"]).astype(float)

RTOL = 1e-12
# Eigenvalues come from a collocation eigensolve; chebfunjax and MATLAB use
# different discretization sizes, so the spectra agree to solver precision, not
# machine precision (Gate-3 documented tolerance).
EIG_RTOL = 1e-8


@pytest.mark.matlab
class TestChebopVsMatlab:
    def test_bvp_poisson(self):
        L = Chebop(lambda x, u: u.diff(2), domain=(-1.0, 1.0))
        L.lbc = 0.0
        L.rbc = 0.0
        u = L.solve(1.0)
        npt.assert_allclose(np.asarray(u(jnp.asarray(_PTS))), _REF["u1"],
                            rtol=RTOL, atol=1e-12)

    def test_bvp_helmholtz_rhs_x(self):
        L = Chebop(lambda x, u: u.diff(2) - u, domain=(-1.0, 1.0))
        L.lbc = 0.0
        L.rbc = 0.0
        u = L.solve(lambda x: x)
        npt.assert_allclose(np.asarray(u(jnp.asarray(_PTS))), _REF["u2"],
                            rtol=RTOL, atol=1e-12)

    def test_bvp_advection_diffusion(self):
        L = Chebop(lambda x, u: 0.02 * u.diff(2) + u.diff(), domain=(0.0, 1.0))
        L.lbc = 0.0
        L.rbc = 0.0
        u = L.solve(1.0)
        # Stiff boundary-layer BVP: chebfunjax and MATLAB use different adaptive
        # collocation sizes, so the solution (steep layer near x=0) agrees to
        # ~5e-11, not machine precision (Gate-3 documented tolerance). The two
        # smooth BVPs above still match to 1e-12.
        npt.assert_allclose(np.asarray(u(jnp.asarray(_PTS01))), _REF["u3"],
                            rtol=1e-9, atol=1e-10)

    def test_eigs_dirichlet_laplacian(self):
        Le = Chebop(lambda x, u: -u.diff(2), domain=(0.0, np.pi))
        Le.bc = 0.0
        got = np.sort(np.real(np.asarray(Le.eigs(k=6))))
        want = np.sort(np.atleast_1d(_REF["eig"]).astype(float))
        npt.assert_allclose(got, want, rtol=EIG_RTOL, atol=EIG_RTOL)
