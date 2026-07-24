"""Distributional-calculus tests for :class:`chebfunjax.fun.deltafun.Deltafun`.

A Deltafun models ``f(x) + Σ_k Σ_j m_{j,k} δ^{(j)}(x − x_k)``.  These tests
pin the distributional identities that the uncovered ``cumsum`` / ``__mul__`` /
sign-and-scale operators must satisfy:

* ``∫`` of a delta is a Heaviside step of the same magnitude (``cumsum``);
* ``d/dx`` turns ``δ`` into ``δ'`` and ``cumsum`` turns it back (ladder);
* ``g(x)·δ(x−x0) = g(x0)·δ(x−x0)`` (localisation);
* multiplying two Deltafuns is rejected.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

D = Domain((-1.0, 1.0))


def _cos_with_delta(mag=2.0, loc=0.3):
    """cos(x) on [-1,1] plus a delta of magnitude ``mag`` at ``loc``."""
    fun = Bndfun.from_function(jnp.cos, D)
    return Deltafun.from_fun_and_deltas(fun, jnp.array([loc]), jnp.array([mag]))


class TestCumsum:
    def test_cumsum_no_deltas_matches_funpart(self):
        df = Deltafun.from_function(jnp.sin, D)
        cs = df.cumsum()
        # MATLAB @deltafun/cumsum: a delta-free Deltafun integrates to its
        # bare funPart antiderivative (a Bndfun, not a Deltafun/cell array).
        assert not isinstance(cs, (Deltafun, list))
        xs = jnp.linspace(-1, 1, 11)
        npt.assert_allclose(np.asarray(cs(xs)),
                            np.asarray(df.funPart.cumsum()(xs)), atol=1e-12)

    def test_cumsum_delta_becomes_heaviside_step(self):
        mag, loc = 2.0, 0.3
        df = _cos_with_delta(mag, loc)
        cs = df.cumsum()
        # MATLAB @deltafun/cumsum: an interior delta splits the domain at the
        # jump, returning a cell array (Python list) of funs -- one on
        # [-1, loc], one on [loc, 1] -- with an exact +mag step across the cut.
        assert isinstance(cs, list) and len(cs) == 2
        below, above = cs

        def smooth(x):  # ∫ cos from -1
            return np.sin(x) - np.sin(-1.0)

        # Below the delta: no step; above the delta: +mag.  These are exact
        # (no Heaviside ringing) because each piece is a smooth antiderivative.
        npt.assert_allclose(float(below(jnp.array(-0.5))), smooth(-0.5), atol=1e-10)
        npt.assert_allclose(float(above(jnp.array(0.8))), smooth(0.8) + mag,
                            atol=1e-10)
        step = (float(above(jnp.array(0.8))) - smooth(0.8)) \
            - (float(below(jnp.array(-0.5))) - smooth(-0.5))
        npt.assert_allclose(step, mag, atol=1e-10)

    def test_diff_then_cumsum_restores_delta(self):
        # d/dx: δ -> δ'; ∫: δ' -> δ.  The magnitude survives the round trip.
        df = _cos_with_delta(mag=2.0, loc=0.3)
        dprime = df.diff(1)
        # δ' lives on row 1; row 0 (plain delta) is now zero.
        assert dprime.delta_mags.shape[0] == 2
        npt.assert_allclose(float(jnp.sum(dprime.delta_mags[0])), 0.0, atol=1e-14)
        restored = dprime.cumsum()
        assert restored.n_deltas == 1
        npt.assert_allclose(float(restored.delta_mags[0, 0]), 2.0, atol=1e-12)


class TestSumAndSign:
    def test_sum_is_smooth_integral_plus_delta(self):
        df = _cos_with_delta(mag=2.0, loc=0.3)
        expected = (np.sin(1.0) - np.sin(-1.0)) + 2.0
        npt.assert_allclose(float(df.sum()), expected, atol=1e-12)

    def test_neg_flips_funpart_and_deltas(self):
        df = _cos_with_delta(mag=2.0)
        neg = -df
        npt.assert_allclose(float(neg.delta_mags[0, 0]), -2.0, atol=1e-14)
        npt.assert_allclose(float(neg.sum()), -float(df.sum()), atol=1e-12)

    def test_pos_is_identity(self):
        df = _cos_with_delta(mag=2.0)
        pos = +df
        npt.assert_allclose(float(pos.delta_mags[0, 0]), 2.0, atol=1e-14)
        npt.assert_allclose(float(pos.sum()), float(df.sum()), atol=1e-12)

    def test_radd_and_rsub_scalar(self):
        df = _cos_with_delta(mag=2.0)
        base = float(df.sum())
        npt.assert_allclose(float((5.0 + df).sum()), base + 5.0 * 2.0, atol=1e-12)
        # (5 - df).sum() = 5*|domain| - base ; |domain| = 2.
        npt.assert_allclose(float((5.0 - df).sum()), 5.0 * 2.0 - base, atol=1e-12)


class TestMultiplication:
    def test_scalar_scales_everything(self):
        df = _cos_with_delta(mag=2.0)
        scaled = df * 3.0
        npt.assert_allclose(float(scaled.delta_mags[0, 0]), 6.0, atol=1e-13)
        npt.assert_allclose(float(scaled.sum()), 3.0 * float(df.sum()), atol=1e-12)

    def test_truediv_scalar(self):
        df = _cos_with_delta(mag=2.0)
        halved = df / 2.0
        npt.assert_allclose(float(halved.delta_mags[0, 0]), 1.0, atol=1e-13)

    def test_mul_by_bndfun_localises_delta(self):
        # g(x)·δ(x−x0) = g(x0)·δ(x−x0).  Here g(x)=x+2, x0=0.3 -> factor 2.3.
        df = _cos_with_delta(mag=2.0, loc=0.3)
        g = Bndfun.from_function(lambda x: x + 2.0, D)
        prod = df * g
        npt.assert_allclose(float(prod.delta_mags[0, 0]), 2.0 * 2.3, atol=1e-11)
        # funPart is the ordinary product cos(x)*(x+2).
        xs = jnp.linspace(-0.9, 0.9, 7)
        npt.assert_allclose(np.asarray(prod.funPart(xs)),
                            np.asarray(df.funPart(xs)) * (np.asarray(xs) + 2.0),
                            atol=1e-11)

    def test_mul_bndfun_no_deltas(self):
        df = Deltafun.from_function(jnp.sin, D)
        g = Bndfun.from_function(jnp.cos, D)
        prod = df * g
        assert prod.n_deltas == 0
        xs = jnp.linspace(-0.9, 0.9, 7)
        npt.assert_allclose(np.asarray(prod(xs)),
                            np.sin(np.asarray(xs)) * np.cos(np.asarray(xs)),
                            atol=1e-11)

    def test_mul_two_deltafuns_same_loc_rejected(self):
        # MATLAB @deltafun/times.m: products of deltafuns are allowed
        # when the delta supports do not intersect (Leibniz expansion,
        # Fable 5); coincident deltas remain undefined and raise.
        df = _cos_with_delta(mag=2.0)
        with pytest.raises(ValueError):
            df * df

    def test_mul_incompatible_type_raises(self):
        df = _cos_with_delta(mag=2.0)
        with pytest.raises(TypeError):
            df * "not a number"


class TestRepr:
    def test_repr_format(self):
        df = Deltafun.from_function(jnp.sin, D)
        s = repr(df)
        assert s.startswith("Deltafun([-1, 1]")
        assert "n_deltas=0" in s
