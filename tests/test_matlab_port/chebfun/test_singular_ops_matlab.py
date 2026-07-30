"""Exponent-preserving abs/power/real on singular chebfuns.

MATLAB reference: the published outputs of the Chebfun example
approx/GammaFun.m (chebfun.org/examples/approx/GammaFun.html), which pin
@chebfun/abs, @chebfun/power and @chebfun/real behaviour on a chebfun
with simple poles ('exps' construction):

    sum(gam)     = NaN
    sum(|gam|)   = Inf
    sum(sqrt(|gam|)) = 14.043323986892393

Chebfun commit: 7574c77.
"""
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gamma as _scipy_gamma

import chebfunjax as cj
from chebfunjax.fun.singfun import Singfun


@pytest.fixture(scope="module")
def gam():
    return cj.chebfun(
        lambda x: jnp.asarray(_scipy_gamma(np.asarray(x))),
        domain=[-4, -3, -2, -1, 0, 4],
        exps=[-1, -1, -1, -1, -1, 0],
    )


def test_sum_gamma_nan(gam):
    # Divergent with sign changes: MATLAB prints NaN.
    assert np.isnan(float(gam.sum()))


def test_abs_preserves_exponents(gam):
    absgam = abs(gam)
    for p_old, p_new in zip(gam.funs, absgam.funs):
        if isinstance(p_old.tech, Singfun):
            assert isinstance(p_new.tech, Singfun)
            assert p_new.tech.exponents == p_old.tech.exponents


def test_sum_abs_gamma_inf(gam):
    # |gamma| has non-integrable poles: MATLAB prints Inf.
    assert np.isposinf(float(abs(gam).sum()))


def test_sum_sqrt_abs_gamma(gam):
    # Convergent integral, published to 15 digits on the example page.
    val = float((abs(gam) ** 0.5).real().sum())
    assert val == pytest.approx(14.043323986892393, rel=1e-13)


def test_power_halves_exponents(gam):
    sq = abs(gam) ** 0.5
    for p_old, p_new in zip(gam.funs, sq.funs):
        if isinstance(p_old.tech, Singfun):
            assert isinstance(p_new.tech, Singfun)
            a_old, b_old = p_old.tech.exponents
            a_new, b_new = p_new.tech.exponents
            assert a_new == pytest.approx(0.5 * a_old)
            assert b_new == pytest.approx(0.5 * b_old)


def test_repr_singular_endpoints(gam):
    # MATLAB displays +/-Inf endpoint values for poles; repr must not
    # crash and must show the blowup.
    r = repr(gam)
    assert "inf" in r.lower()
    assert "5 smooth pieces" in r


class TestBlowupSplittingDetection:
    """'blowup' + 'splitting' automatic pole detection.

    MATLAB reference: the first construction of the GammaFun example page,
    chebfun('gamma(x)', [-4 4], 'blowup', 'on', 'splitting', 'on'), whose
    published display shows 5 pieces broken exactly at the poles -3, -2,
    -1, ~0 with endpoint exponents [-1 -1] ([-1 0] on the last piece).
    """

    @pytest.fixture(scope="class")
    def gam_auto(self):
        return cj.chebfun(
            lambda x: jnp.asarray(_scipy_gamma(np.asarray(x))),
            domain=[-4, 4], blowup=True, splitting=True,
        )

    def test_five_pieces_at_the_poles(self, gam_auto):
        assert len(gam_auto.funs) == 5
        bps = [p.interval[1] for p in gam_auto.funs[:-1]]
        for found, true in zip(bps, (-3.0, -2.0, -1.0, 0.0)):
            assert abs(found - true) < 1e-10

    def test_detected_exponents(self, gam_auto):
        exps = [tuple(round(float(e)) for e in p.tech.exponents)
                for p in gam_auto.funs]
        assert exps == [(-1, -1)] * 4 + [(-1, 0)]

    def test_integrals_match_exps_construction(self, gam_auto):
        assert np.isnan(float(gam_auto.sum()))
        assert np.isposinf(float(abs(gam_auto).sum()))
        val = float((abs(gam_auto) ** 0.5).real().sum())
        assert val == pytest.approx(14.043323986892393, rel=1e-12)
