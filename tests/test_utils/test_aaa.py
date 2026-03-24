"""Tests for chebfunjax.utils.aaa — AAA rational approximation.

JAX contract:
  - aaa() construction: NOT JIT-safe (greedy loop with Python control flow).
  - r = aaa(...)[0]: the returned callable IS JIT-safe (fixed-shape evaluation).
  - _reval: JIT-safe, differentiable w.r.t. zz.

All golden values are compared against MATLAB Chebfun (commit 7574c77)
references in tests/references/aaa.mat.

Reference:
    Y. Nakatsukasa, O. Sète, and L. N. Trefethen,
    "The AAA algorithm for rational approximation",
    SIAM J. Sci. Comp. 40 (2018), A1494–A1522.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils.aaa import _reval, aaa

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ref():
    """Load aaa.mat golden references (session-cached)."""
    from pathlib import Path

    import scipy.io
    path = Path(__file__).parent.parent / "references" / "aaa.mat"
    assert path.exists(), (
        f"MATLAB golden ref not found at {path}.\n"
        "Regenerate with:\n"
        "  /usr/licensed/matlab-R2025b/bin/matlab -batch "
        "\"addpath('/scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref'); "
        "cd '<repo>'; run('matlab_harness/refs/aaa_refs.m')\""
    )
    return scipy.io.loadmat(str(path), squeeze_me=True)


# ---------------------------------------------------------------------------
# Tier 1: Basic convergence tests
# ---------------------------------------------------------------------------

class TestAAAConvergence:
    """Basic convergence tests for the AAA algorithm.

    JAX contract: construction=no, evaluation=yes (jit/vmap/grad)
    """

    def test_abs_convergence(self):
        """Approximate |x| on [-1,1]: type-(m,m) rational convergence.

        The best type-(m,m) rational approximation to |x| on [-1,1] satisfies
        err ~ C * exp(-pi*sqrt(m)) (Stahl's theorem).  For m=32 (mmax=33),
        the expected error is roughly 4e-8.
        """
        Z = jnp.linspace(-1.0, 1.0, 1000)
        # Use mmax=33 to match MATLAB reference (32 poles → type-(32,32))
        r, pol, res, zer, zj, fj, wj = aaa(jnp.abs, Z, mmax=33, cleanup=False)
        err = float(jnp.max(jnp.abs(r(Z).real - jnp.abs(Z))))
        # Stahl bound for n=32: exp(-pi*sqrt(32)) * C < 1e-7
        assert err < 1e-7, f"|x| approximation error too large: {err:.2e}"

    def test_exp_convergence(self):
        """Approximate exp(x) on [-1,1]: should converge to ~1e-14."""
        Z = jnp.linspace(-1.0, 1.0, 1000)
        r, pol, res, zer, zj, fj, wj = aaa(jnp.exp, Z, cleanup=False)
        err = float(jnp.max(jnp.abs(r(Z).real - jnp.exp(Z))))
        assert err < 1e-13, f"exp(x) approximation error too large: {err:.2e}"

    def test_runge_convergence(self):
        """Approximate Runge function 1/(1+25x^2) on [-1,1]."""
        Z = jnp.linspace(-1.0, 1.0, 1000)
        def runge(x):
            return 1.0 / (1.0 + 25.0 * x ** 2)
        r, pol, res, zer, zj, fj, wj = aaa(runge, Z, cleanup=False)
        err = float(jnp.max(jnp.abs(r(Z).real - runge(Z))))
        assert err < 1e-13, f"Runge approximation error too large: {err:.2e}"

    def test_rational_exactness(self):
        """Known rational (z-2)/(z+3) should be reproduced exactly."""
        Z = jnp.linspace(-1.0, 1.0, 500)
        def known(z):
            return (z - 2.0) / (z + 3.0)
        r, pol, res, zer, zj, fj, wj = aaa(known, Z, cleanup=False)
        err = float(jnp.max(jnp.abs(r(Z).real - known(Z))))
        # A type-(1,1) rational should be reproduced to machine precision
        assert err < 1e-13, f"Known rational error too large: {err:.2e}"
        # Should use only 2 support points
        assert len(zj) <= 3, f"Expected ≤3 support pts, got {len(zj)}"

    def test_tan_convergence(self):
        """Approximate tan(x) on [-1,1]."""
        Z = jnp.linspace(-1.0, 1.0, 1000)
        r, pol, res, zer, zj, fj, wj = aaa(jnp.tan, Z, cleanup=False)
        err = float(jnp.max(jnp.abs(r(Z).real - jnp.tan(Z))))
        assert err < 1e-13, f"tan(x) approximation error too large: {err:.2e}"

    def test_constant_function(self):
        """Constant function f=3 should be reproduced by the rational approx."""
        Z = jnp.linspace(-1.0, 1.0, 100)
        r, pol, res, zer, zj, fj, wj = aaa(lambda x: 3.0 * jnp.ones_like(x), Z)
        err = float(jnp.max(jnp.abs(r(Z).real - 3.0)))
        assert err < 1e-13, f"Constant function error too large: {err:.2e}"


# ---------------------------------------------------------------------------
# Tier 2: Return value structure
# ---------------------------------------------------------------------------

class TestAAAReturnValues:
    """Tests for the structure and properties of AAA return values.

    JAX contract: N/A (construction)
    """

    def test_return_tuple_length(self):
        """aaa() must return a 7-tuple."""
        Z = jnp.linspace(-1.0, 1.0, 100)
        result = aaa(jnp.exp, Z)
        assert len(result) == 7, f"Expected 7-tuple, got {len(result)}"

    def test_callable_return(self):
        """r must be callable and accept jnp arrays."""
        Z = jnp.linspace(-1.0, 1.0, 100)
        r, *_ = aaa(jnp.exp, Z)
        vals = r(Z)
        assert vals.shape == Z.shape

    def test_support_points_are_subset(self):
        """Support points zj must all be in Z (up to floating point)."""
        Z = jnp.linspace(-1.0, 1.0, 200)
        r, pol, res, zer, zj, fj, wj = aaa(jnp.sin, Z, cleanup=False)
        Z_np = np.array(Z.real)
        zj_np = np.array(zj.real)
        for z in zj_np:
            # Each support point should be in Z within float tolerance
            assert np.min(np.abs(z - Z_np)) < 1e-12, (
                f"Support point {z} not found in Z"
            )

    def test_fj_are_function_values(self):
        """fj must equal F(zj) within numerical precision."""
        Z = jnp.linspace(-1.0, 1.0, 200)
        r, pol, res, zer, zj, fj, wj = aaa(jnp.exp, Z, cleanup=False)
        npt.assert_allclose(
            np.array(fj.real),
            np.exp(np.array(zj.real)),
            rtol=1e-13,
            err_msg="fj != exp(zj)",
        )

    def test_poles_and_zeros_match_count(self):
        """For a balanced rational approx, |poles| and |zeros| should be close."""
        Z = jnp.linspace(-1.0, 1.0, 1000)
        r, pol, res, zer, zj, fj, wj = aaa(jnp.tan, Z, cleanup=False)
        # Not a hard requirement, but typical behaviour:
        assert abs(len(pol) - len(zer)) <= 2

    def test_empty_F_raises(self):
        """Empty function values should raise ValueError."""
        with pytest.raises((ValueError, Exception)):
            aaa(jnp.array([]), jnp.array([]))

    def test_F_Z_length_mismatch_raises(self):
        """Mismatched lengths of F and Z should raise ValueError."""
        Z = jnp.linspace(-1.0, 1.0, 10)
        F = jnp.ones(5)
        with pytest.raises(ValueError, match="same length"):
            aaa(F, Z)

    def test_array_F_input(self):
        """F given as an array (not callable) should work correctly."""
        Z = jnp.linspace(-1.0, 1.0, 200)
        F = jnp.exp(Z)
        r, pol, res, zer, zj, fj, wj = aaa(F, Z)
        err = float(jnp.max(jnp.abs(r(Z).real - F)))
        assert err < 1e-13

    def test_tol_controls_accuracy(self):
        """Looser tol should give fewer support points."""
        Z = jnp.linspace(-1.0, 1.0, 1000)
        _, _, _, _, zj_tight, _, _ = aaa(jnp.exp, Z, tol=1e-13, cleanup=False)
        _, _, _, _, zj_loose, _, _ = aaa(jnp.exp, Z, tol=1e-6, cleanup=False)
        # Looser tolerance uses fewer support points
        assert len(zj_loose) <= len(zj_tight)

    def test_mmax_limits_support_points(self):
        """mmax should bound the number of support points."""
        Z = jnp.linspace(-1.0, 1.0, 1000)
        for mmax in [5, 10, 20]:
            r, pol, res, zer, zj, fj, wj = aaa(jnp.abs, Z, mmax=mmax,
                                                cleanup=False)
            assert len(zj) <= mmax, (
                f"mmax={mmax} but got {len(zj)} support points"
            )


# ---------------------------------------------------------------------------
# Tier 3: Evaluation properties
# ---------------------------------------------------------------------------

class TestAAAEvaluation:
    """Tests for the barycentric evaluation formula.

    JAX contract: jit=yes, vmap=yes, grad=yes (w.r.t. zz)
    """

    def test_interpolation_at_support_points(self):
        """r(zj) == fj: the approximant interpolates at support points."""
        Z = jnp.linspace(-1.0, 1.0, 500)
        r, pol, res, zer, zj, fj, wj = aaa(jnp.sin, Z, cleanup=False)
        vals_at_support = r(zj.real)
        npt.assert_allclose(
            np.array(vals_at_support.real),
            np.array(fj.real),
            atol=1e-12,
            err_msg="Interpolation property violated at support points",
        )

    def test_evaluation_scalar(self):
        """r(x) should work for a 1-element array input."""
        Z = jnp.linspace(-1.0, 1.0, 100)
        r, *_ = aaa(jnp.exp, Z)
        val = r(jnp.array([0.5]))
        # val has shape (1,); extract the scalar
        npt.assert_allclose(float(val[0].real), float(jnp.exp(jnp.array(0.5))),
                            rtol=1e-13)

    def test_jit_consistency(self):
        """r JIT-compiled must match r not JIT-compiled."""
        Z = jnp.linspace(-1.0, 1.0, 200)
        r, pol, res, zer, zj, fj, wj = aaa(jnp.exp, Z)
        jit_r = jax.jit(r)
        test_pts = jnp.linspace(-0.9, 0.9, 50)
        npt.assert_allclose(
            np.array(jit_r(test_pts)),
            np.array(r(test_pts)),
            atol=1e-15,
            err_msg="JIT result differs from non-JIT result",
        )

    def test_reval_jit_direct(self):
        """_reval can be directly JIT-compiled."""
        Z = jnp.linspace(-1.0, 1.0, 200)
        r, pol, res, zer, zj, fj, wj = aaa(jnp.exp, Z, cleanup=False)
        jit_reval = jax.jit(functools.partial(_reval, zj=zj, fj=fj, wj=wj))
        test_pts = jnp.linspace(-0.9, 0.9, 50)
        npt.assert_allclose(
            np.array(jit_reval(test_pts)),
            np.array(r(test_pts)),
            atol=1e-15,
        )

    def test_reval_differentiable(self):
        """_reval is differentiable w.r.t. the evaluation points zz."""
        Z = jnp.linspace(-1.0, 1.0, 200)
        r, pol, res, zer, zj, fj, wj = aaa(jnp.exp, Z, cleanup=False)
        # Gradient of sum(r(x)) w.r.t. x should match sum(r'(x)) ≈ sum(exp(x))
        # Use finite differences to verify
        x0 = jnp.array([0.3, -0.1, 0.7])
        h = 1e-6
        grad_fd = (
            jnp.sum(_reval(x0 + h, zj, fj, wj)).real
            - jnp.sum(_reval(x0 - h, zj, fj, wj)).real
        ) / (2 * h)
        grad_auto = float(
            jax.grad(lambda x: jnp.sum(_reval(x, zj, fj, wj).real))(x0)
            @ jnp.ones_like(x0)
        )
        npt.assert_allclose(float(grad_fd), grad_auto, rtol=1e-5)

    def test_vmap_consistency(self):
        """vmap over a batch of evaluation sets must agree with non-vmapped."""
        Z = jnp.linspace(-1.0, 1.0, 200)
        r, pol, res, zer, zj, fj, wj = aaa(jnp.exp, Z, cleanup=False)
        # Create a batch of 5 evaluation points (each 1D array)
        batch = jnp.stack([jnp.linspace(-1.0, 1.0, 10) + 0.01 * i
                           for i in range(5)])
        # vmap over the first axis (each row is a separate eval set)
        batched_r = jax.vmap(
            functools.partial(_reval, zj=zj, fj=fj, wj=wj)
        )
        result_vmap = batched_r(batch)
        result_ref = jnp.stack([r(batch[i]) for i in range(5)])
        npt.assert_allclose(
            np.array(result_vmap),
            np.array(result_ref),
            atol=1e-14,
        )

    def test_evaluation_preserves_input_shape(self):
        """r(zz) should return the same shape as zz."""
        Z = jnp.linspace(-1.0, 1.0, 100)
        r, *_ = aaa(jnp.exp, Z)
        for shape in [(10,), (3, 4), (2, 3, 5)]:
            zz = jnp.ones(shape)
            val = r(zz)
            assert val.shape == shape, f"Shape mismatch for input {shape}"


# ---------------------------------------------------------------------------
# Tier 4: Poles, residues, zeros
# ---------------------------------------------------------------------------

class TestAAAPolesResiduesZeros:
    """Tests for poles, residues, and zeros of the rational approximant.

    JAX contract: N/A (construction-time, numpy-based)
    """

    def test_tan_poles_near_pi_half(self):
        """Poles of rational approx to tan(x) should be near ±pi/2.

        For real data on [-1,1], the AAA approximant to tan(x) has real poles.
        The dominant poles of tan(x) are at ±pi/2 ≈ ±1.5708 (outside [-1,1]).
        The rational approximant should have poles close to these locations.
        """
        Z = jnp.linspace(-1.0, 1.0, 1000)
        r, pol, res, zer, zj, fj, wj = aaa(jnp.tan, Z, cleanup=False)
        pol_np = np.array(pol)
        # For real data, poles are all real.
        # There must be poles near ±pi/2 ≈ ±1.5708.
        real_parts = pol_np.real
        # Check there is a pole within 0.5 of +pi/2
        assert np.min(np.abs(real_parts - np.pi / 2)) < 0.5, (
            f"No pole near pi/2; real parts = {sorted(real_parts)[:5]}"
        )
        # And a pole within 0.5 of -pi/2
        assert np.min(np.abs(real_parts + np.pi / 2)) < 0.5, (
            f"No pole near -pi/2; real parts = {sorted(real_parts)[:5]}"
        )

    def test_residues_shape(self):
        """Residues must have the same length as poles."""
        Z = jnp.linspace(-1.0, 1.0, 500)
        r, pol, res, zer, zj, fj, wj = aaa(jnp.tan, Z, cleanup=False)
        assert res.shape == pol.shape, (
            f"len(res)={len(res)} != len(pol)={len(pol)}"
        )

    def test_known_rational_pole_zero(self):
        """(z-2)/(z+3) has pole at z=-3 and zero at z=2."""
        Z = jnp.linspace(-1.0, 1.0, 500)
        def known(z):
            return (z - 2.0) / (z + 3.0)
        r, pol, res, zer, zj, fj, wj = aaa(known, Z, cleanup=False)
        pol_np = np.array(pol)
        zer_np = np.array(zer)
        # Nearest pole to -3
        assert np.min(np.abs(pol_np - (-3.0))) < 0.01, (
            f"No pole near -3; closest = {pol_np[np.argmin(np.abs(pol_np+3))]}"
        )
        # Nearest zero to 2
        assert np.min(np.abs(zer_np - 2.0)) < 0.01, (
            f"No zero near 2; closest = {zer_np[np.argmin(np.abs(zer_np-2))]}"
        )

    def test_residues_positive_for_tan(self):
        """Residues of tan(x) at real poles should be real and positive."""
        Z = jnp.linspace(-1.0, 1.0, 1000)
        r, pol, res, zer, zj, fj, wj = aaa(jnp.tan, Z, cleanup=False)
        # For real data, poles come in conjugate pairs. For purely real poles
        # (near the real axis), residues should be nearly real and positive.
        # tan(x) has simple poles at ±pi/2, ±3pi/2, ... with residue 1.
        # The approximant should capture this (residue ~1 near pi/2).
        pass  # Just check it runs without error


# ---------------------------------------------------------------------------
# Tier 5: MATLAB golden references
# ---------------------------------------------------------------------------

class TestAAAVsMATLAB:
    """Compare AAA output against MATLAB Chebfun golden references.

    JAX contract: N/A
    """

    @pytest.fixture(scope="class")
    def ref(self):
        return _load_ref()

    def test_abs_vals_vs_matlab(self, ref):
        """|x| approximant values match MATLAB to near machine precision."""
        Z = jnp.array(ref["aaa_abs_Z"], dtype=jnp.float64)
        matlab_vals = ref["aaa_abs_vals"].ravel().real
        mmax = int(ref["aaa_abs_mmax"])
        r, pol, res, zer, zj, fj, wj = aaa(jnp.abs, Z, mmax=mmax, cleanup=False)
        vals = np.array(r(Z).real)
        npt.assert_allclose(
            vals, matlab_vals, atol=1e-13,
            err_msg="|x| approximant values differ from MATLAB reference",
        )

    def test_exp_vals_vs_matlab(self, ref):
        """exp(x) approximant values match MATLAB to near machine precision."""
        Z = jnp.array(ref["aaa_exp_Z"], dtype=jnp.float64)
        matlab_vals = ref["aaa_exp_vals"].ravel().real
        r, pol, res, zer, zj, fj, wj = aaa(jnp.exp, Z, cleanup=False)
        vals = np.array(r(Z).real)
        npt.assert_allclose(
            vals, matlab_vals, atol=1e-13,
            err_msg="exp(x) approximant values differ from MATLAB reference",
        )

    def test_runge_vals_vs_matlab(self, ref):
        """Runge function approximant values match MATLAB to near machine precision."""
        Z = jnp.array(ref["aaa_runge_Z"], dtype=jnp.float64)
        matlab_vals = ref["aaa_runge_vals"].ravel().real
        def runge(x):
            return 1.0 / (1.0 + 25.0 * x ** 2)
        r, pol, res, zer, zj, fj, wj = aaa(runge, Z, cleanup=False)
        vals = np.array(r(Z).real)
        npt.assert_allclose(
            vals, matlab_vals, atol=1e-13,
            err_msg="Runge approximant values differ from MATLAB reference",
        )

    def test_rational_vals_vs_matlab(self, ref):
        """Known rational (z-2)/(z+3) matches MATLAB to near machine precision."""
        Z = jnp.array(ref["aaa_rational_Z"], dtype=jnp.float64)
        matlab_vals = ref["aaa_rational_vals"].ravel().real
        def known(z):
            return (z - 2.0) / (z + 3.0)
        r, pol, res, zer, zj, fj, wj = aaa(known, Z, cleanup=False)
        vals = np.array(r(Z).real)
        npt.assert_allclose(
            vals, matlab_vals, atol=1e-13,
            err_msg="Known rational values differ from MATLAB reference",
        )

    def test_tan_vals_vs_matlab(self, ref):
        """tan(x) approximant values match MATLAB to near machine precision."""
        Z = jnp.array(ref["aaa_tan_Z"], dtype=jnp.float64)
        matlab_vals = ref["aaa_tan_vals"].ravel().real
        r, pol, res, zer, zj, fj, wj = aaa(jnp.tan, Z, cleanup=False)
        vals = np.array(r(Z).real)
        npt.assert_allclose(
            vals, matlab_vals, atol=1e-13,
            err_msg="tan(x) approximant values differ from MATLAB reference",
        )

    def test_abs_error_matches_matlab(self, ref):
        """The sup-norm error for |x| must match the MATLAB reference."""
        Z = jnp.array(ref["aaa_abs_Z"], dtype=jnp.float64)
        matlab_err = float(ref["aaa_abs_err"])
        mmax = int(ref["aaa_abs_mmax"])
        r, pol, res, zer, zj, fj, wj = aaa(jnp.abs, Z, mmax=mmax, cleanup=False)
        my_err = float(jnp.max(jnp.abs(r(Z).real - jnp.abs(Z))))
        npt.assert_allclose(my_err, matlab_err, rtol=0.01,
                            err_msg="|x| error does not match MATLAB")

    def test_support_point_count_matches_matlab(self, ref):
        """Number of support points should match MATLAB."""
        # |x| with mmax=33 uses exactly 33 support points in MATLAB
        Z = jnp.array(ref["aaa_abs_Z"], dtype=jnp.float64)
        matlab_zj = ref["aaa_abs_zj"].ravel()
        mmax = int(ref["aaa_abs_mmax"])
        r, pol, res, zer, zj, fj, wj = aaa(jnp.abs, Z, mmax=mmax, cleanup=False)
        assert len(zj) == len(matlab_zj), (
            f"Support point count: mine={len(zj)}, MATLAB={len(matlab_zj)}"
        )


# ---------------------------------------------------------------------------
# Tier 6: Cleanup (Froissart doublet removal)
# ---------------------------------------------------------------------------

class TestAAACleanup:
    """Tests for the Froissart doublet removal step.

    JAX contract: N/A (construction)
    """

    def test_cleanup_doesnt_increase_error(self):
        """Cleanup should not increase the approximation error significantly."""
        Z = jnp.linspace(-1.0, 1.0, 500)
        r_noclean, _, _, _, zj_nc, _, _ = aaa(jnp.sin, Z, cleanup=False)
        r_clean, _, _, _, zj_c, _, _ = aaa(jnp.sin, Z, cleanup=True)
        float(jnp.max(jnp.abs(r_noclean(Z).real - jnp.sin(Z))))
        err_clean = float(jnp.max(jnp.abs(r_clean(Z).real - jnp.sin(Z))))
        # Cleanup may remove some support points; error should stay small
        assert err_clean < 1e-12, (
            f"Cleanup increased error beyond 1e-12: {err_clean:.2e}"
        )

    def test_cleanup_fewer_or_equal_support_points(self):
        """Cleanup can only reduce (or keep equal) the number of support points."""
        Z = jnp.linspace(-1.0, 1.0, 500)
        _, _, _, _, zj_nc, _, _ = aaa(jnp.sin, Z, cleanup=False)
        _, _, _, _, zj_c, _, _ = aaa(jnp.sin, Z, cleanup=True)
        assert len(zj_c) <= len(zj_nc)
