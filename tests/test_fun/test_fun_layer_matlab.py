"""MATLAB golden-ref parity for the fun layer (Unbndfun/Singfun/Deltafun).

These Layer-3 classes are the least-exercised in the suite and are not
yet reachable through the public chebfun() factory (tracked), so the
tests exercise the classes directly. References are generated through
MATLAB's public chebfun API (unbounded domains, 'exps', dirac) by
matlab_harness/refs/fun_layer_refs.m.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import scipy.io

jax.config.update("jax_enable_x64", True)

_REF_PATH = Path(__file__).resolve().parents[1] / "references" / "fun_layer.mat"
if not _REF_PATH.exists():
    pytest.skip(
        "fun_layer.mat golden ref not generated "
        "(run matlab_harness/refs/fun_layer_refs.m)",
        allow_module_level=True,
    )
_REF = scipy.io.loadmat(str(_REF_PATH), squeeze_me=True)

RTOL = 1e-12


@pytest.mark.matlab
class TestUnbndfunVsMatlab:
    def test_halfline_eval(self):
        from chebfunjax.domain import Domain
        from chebfunjax.fun.unbndfun import Unbndfun

        f = Unbndfun.from_function(
            lambda x: jnp.exp(-x), domain=Domain((0.0, float(jnp.inf)))
        )
        pts = jnp.asarray(_REF["ub1_pts"], dtype=jnp.float64)
        npt.assert_allclose(np.asarray(f(pts)), _REF["ub1_eval"],
                            rtol=RTOL, atol=1e-13)

    def test_halfline_sum(self):
        from chebfunjax.domain import Domain
        from chebfunjax.fun.unbndfun import Unbndfun

        f = Unbndfun.from_function(
            lambda x: jnp.exp(-x), domain=Domain((0.0, float(jnp.inf)))
        )
        # Gate-3 documented: the doubly-infinite map quadrature converges
        # to ~1e-11 on the half-line (MATLAB uses the same map family and
        # its own sum here is 0.999999999986 vs exactly 1).
        npt.assert_allclose(float(f.sum()), float(_REF["ub1_sum"]),
                            rtol=1e-10)

    def test_halfline_diff(self):
        from chebfunjax.domain import Domain
        from chebfunjax.fun.unbndfun import Unbndfun

        f = Unbndfun.from_function(
            lambda x: jnp.exp(-x), domain=Domain((0.0, float(jnp.inf)))
        )
        pts = jnp.asarray(_REF["ub1_pts"], dtype=jnp.float64)
        npt.assert_allclose(np.asarray(f.diff()(pts)), _REF["ub1_dval"],
                            rtol=1e-10, atol=1e-11)

    def test_realline_eval_and_sum(self):
        from chebfunjax.domain import Domain
        from chebfunjax.fun.unbndfun import Unbndfun

        g = Unbndfun.from_function(
            lambda x: 1.0 / (1.0 + x**2),
            domain=Domain((-float(jnp.inf), float(jnp.inf))),
        )
        pts = jnp.asarray(_REF["ub2_pts"], dtype=jnp.float64)
        npt.assert_allclose(np.asarray(g(pts)), _REF["ub2_eval"],
                            rtol=RTOL, atol=1e-13)
        npt.assert_allclose(float(g.sum()), float(_REF["ub2_sum"]),
                            rtol=1e-10)


@pytest.mark.matlab
class TestSingfunVsMatlab:
    def test_inverse_sqrt_eval(self):
        from chebfunjax.fun.singfun import Singfun

        s = Singfun.from_function(
            lambda x: 1.0 / jnp.sqrt(1.0 - x**2), exponents=(-0.5, -0.5)
        )
        pts = jnp.asarray(_REF["sg1_pts"], dtype=jnp.float64)
        npt.assert_allclose(np.asarray(s(pts)), _REF["sg1_eval"], rtol=RTOL)

    def test_inverse_sqrt_sum_is_pi(self):
        from chebfunjax.fun.singfun import Singfun

        s = Singfun.from_function(
            lambda x: 1.0 / jnp.sqrt(1.0 - x**2), exponents=(-0.5, -0.5)
        )
        npt.assert_allclose(float(s.sum()), float(_REF["sg1_sum"]),
                            rtol=RTOL)

    def test_branch_point_eval_and_sum(self):
        from chebfunjax.fun.singfun import Singfun

        s = Singfun.from_function(
            lambda x: jnp.sqrt(1.0 + x) * jnp.exp(x), exponents=(0.5, 0.0)
        )
        pts = jnp.asarray(_REF["sg2_pts"], dtype=jnp.float64)
        npt.assert_allclose(np.asarray(s(pts)), _REF["sg2_eval"], rtol=RTOL)
        npt.assert_allclose(float(s.sum()), float(_REF["sg2_sum"]),
                            rtol=RTOL)


@pytest.mark.matlab
class TestDeltafunVsMatlab:
    def test_dirac_at_zero_sum(self):
        from chebfunjax.domain import Domain
        from chebfunjax.fun.bndfun import Bndfun
        from chebfunjax.fun.deltafun import Deltafun

        zero = Bndfun.from_function(
            lambda x: jnp.zeros_like(x), Domain((-1.0, 1.0))
        )
        d = Deltafun.from_fun_and_deltas(
            zero, jnp.array([0.0]), jnp.array([[1.0]])
        )
        npt.assert_allclose(float(d.sum()), float(_REF["dl1_sum"]),
                            rtol=RTOL)

    def test_dirac_shifted_sum(self):
        from chebfunjax.domain import Domain
        from chebfunjax.fun.bndfun import Bndfun
        from chebfunjax.fun.deltafun import Deltafun

        zero = Bndfun.from_function(
            lambda x: jnp.zeros_like(x), Domain((-1.0, 1.0))
        )
        d = Deltafun.from_fun_and_deltas(
            zero, jnp.array([0.3]), jnp.array([[1.0]])
        )
        npt.assert_allclose(float(d.sum()), float(_REF["dl2_sum"]),
                            rtol=RTOL)
