"""MATLAB golden-reference parity tests for the ballfun module.

chebfunjax Ballfun is constructed Cartesian (op(x,y,z)) and evaluated in
spherical coordinates f(r, lam, th) with x=r sin(th)cos(lam), y=r sin(th)sin(lam),
z=r cos(th).  __call__ does a meshgrid evaluation, so a single matched point is
read as [0, 0, 0].  Pins evaluation, sum3 (integral over the unit ball) and the
L2 norm at rtol 1e-12.  Golden ref: matlab_harness/refs/ballfun_refs.m
(-> tests/references/ballfun.mat).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import scipy.io

from chebfunjax.ballfun.ballfun import Ballfun

_REF_PATH = Path(__file__).resolve().parents[1] / "references" / "ballfun.mat"
if not _REF_PATH.exists():
    pytest.skip(
        "ballfun.mat golden ref not generated (run matlab_harness/refs/ballfun_refs.m)",
        allow_module_level=True,
    )
_REF = scipy.io.loadmat(str(_REF_PATH), squeeze_me=True)
_RP = np.atleast_1d(_REF["rp"]).astype(float)
_LAMP = np.atleast_1d(_REF["lamp"]).astype(float)
_THP = np.atleast_1d(_REF["thp"]).astype(float)

# Same Cartesian battery as matlab_harness/refs/ballfun_refs.m.
_FUNS = {
    1: lambda x, y, z: z,
    2: lambda x, y, z: x,
    3: lambda x, y, z: jnp.exp(x),
    4: lambda x, y, z: 1 + x * z,
}
RTOL = 1e-12


def _eval_points(f):
    """Evaluate a Ballfun at the matched (r, lam, th) points (grid -> [0,0,0])."""
    out = np.empty(_RP.shape[0], dtype=float)
    for k in range(_RP.shape[0]):
        val = f(jnp.array([_RP[k]]), jnp.array([_LAMP[k]]), jnp.array([_THP[k]]))
        out[k] = float(np.asarray(val)[0, 0, 0])
    return out


@pytest.mark.matlab
@pytest.mark.parametrize("i", [1, 2, 3, 4])
class TestBallfunVsMatlab:
    def test_eval(self, i):
        f = Ballfun.from_function(_FUNS[i])
        npt.assert_allclose(_eval_points(f), _REF[f"f{i}_eval"], rtol=RTOL, atol=1e-12)

    def test_sum(self, i):
        f = Ballfun.from_function(_FUNS[i])
        npt.assert_allclose(float(f.integral()), float(_REF[f"f{i}_sum"]),
                            rtol=RTOL, atol=1e-12)

    def test_norm(self, i):
        f = Ballfun.from_function(_FUNS[i])
        npt.assert_allclose(float(f.norm()), float(_REF[f"f{i}_norm"]), rtol=RTOL)
