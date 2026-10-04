"""Source-bound tests for strict/classic JAX happiness checks.

MATLAB provenance: Chebfun commit 7574c77680d7e82b79626300bf255498271a72df,
tests/chebtech/test_happinessCheck.m groups 7–8 and @chebtech/{strictCheck,
classicCheck}.m. Additional direct cutoffs preserve executable source quirks.
"""
import jax.numpy as jnp

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2, _classic_check, _strict_check
from chebfunjax.utils.quadrature import chebpts

TECHS = (Chebtech1, Chebtech2)


def test_source_group7_strict_then_classic_scalar():
    tol = 2.0**-52
    for tech, kind in ((Chebtech1, 1), (Chebtech2, 2)):
        def operator(x):
            return jnp.sin(10.0 * (x - 0.1))

        def make(n):
            x = chebpts(n, kind=kind)
            values = operator(x)
            coeffs = tech.vals2coeffs(values)
            return x, values, coeffs

        x1, v1, c1 = make(39)
        _, v2, c2 = make(41)
        assert not _strict_check(c1, jnp.max(jnp.abs(v1)), tol)[0]
        assert _strict_check(c2, jnp.max(jnp.abs(v2)), tol)[0]
        assert _classic_check(c1, v1, x1, jnp.max(jnp.abs(v1)), 1.0, tol)[0]


def test_source_group8_strict_array_valued():
    tol = 2.0**-52
    for tech, kind in ((Chebtech1, 1), (Chebtech2, 2)):
        def operator(x):
            return jnp.stack((jnp.sin(10.0 * (x - 0.1)), jnp.exp(x)), axis=-1)

        def make(n):
            x = chebpts(n, kind=kind)
            values = operator(x)
            coeffs = tech.vals2coeffs(values)
            return values, coeffs

        values1, coeffs1 = make(39)
        values2, coeffs2 = make(41)
        scales1 = jnp.max(jnp.abs(values1), axis=0)
        scales2 = jnp.max(jnp.abs(values2), axis=0)
        assert not _strict_check(coeffs1, scales1, tol)[0]
        assert _strict_check(coeffs2, scales2, tol)[0]


def test_strict_empty_source_cutoff_keeps_negative_coefficients():
    # MATLAB strictCheck's find(max(coeffs,[],2)>0,'last') is empty for this
    # negative-only coefficient vector. prolong(f,[]) leaves f unchanged.
    coeffs = -jnp.ones(32, dtype=jnp.float64).at[1:].set(0.0)
    assert _strict_check(coeffs, 1.0, 1e-12) == (True, None)
    assert coeffs[:None].shape == coeffs.shape


def test_strict_matrix_uses_one_source_row_cutoff_for_mixed_signs():
    coeffs = jnp.zeros((32, 2), dtype=jnp.float64)
    coeffs = coeffs.at[0, 0].set(-2.0).at[1, 1].set(3.0)
    coeffs = coeffs.at[0, 1].set(-1.0)
    assert _strict_check(coeffs, jnp.array([2.0, 3.0]), 1e-12) == (True, 2)


def test_strict_complex_row_ties_use_larger_phase_before_gt_zero():
    coeffs = jnp.zeros((32, 2), dtype=jnp.complex128)
    # Equal magnitudes: MATLAB max selects -1 (phase pi) over +1 (phase 0);
    # its subsequent relational comparison uses the selected real part.
    coeffs = coeffs.at[0, :].set(jnp.array([1.0 + 0j, -1.0 + 0j]))
    assert _strict_check(coeffs, 1.0, 1e-12) == (True, None)


def test_strict_zero_scale_and_strict_threshold_inclusive_boundary():
    zeros = jnp.zeros(32, dtype=jnp.float64)
    assert _strict_check(zeros, 0.0, 1e-12) == (True, 1)

    tol = 1e-3
    at_threshold = jnp.zeros(32, dtype=jnp.float64).at[0].set(1.0)
    at_threshold = at_threshold.at[-1].set(tol)
    above_threshold = at_threshold.at[-1].set(jnp.nextafter(tol, jnp.inf))
    # MATLAB zeros coefficients when ac <= epslevel, but a value just above
    # remains in the tested tail; strict's comparison is intentionally exact.
    assert _strict_check(at_threshold, 1.0, tol) == (True, 1)
    assert _strict_check(above_threshold, 1.0, tol) == (False, 32)

    try:
        _strict_check(at_threshold.at[2].set(jnp.nan), 1.0, tol)
    except ValueError as exc:
        assert exc.identifier == "CHEBFUN:CHEBTECH:strictCheck:nanEval"
        assert str(exc) == "Function returned NaN when evaluated."
    else:
        raise AssertionError("source strictCheck rejects NaN coefficients")


def test_classic_requirements_half_away_tail_length_edge():
    # MATLAB round((n-1)/8) is half-away-from-zero: n=53 gives round(6.5)=7.
    # A 1.1e-14 coefficient at -7 is included by MATLAB's seven-row tail and
    # fails epslevel=1e-14; ties-to-even's six-row tail would miss it and pass.
    n = 53
    coeffs = jnp.zeros(n, dtype=jnp.float64).at[0].set(1.0)
    coeffs = coeffs.at[-7].set(1.1e-14)
    values = jnp.ones(n, dtype=jnp.float64)
    points = chebpts(n, kind=2)
    happy, cutoff = _classic_check(coeffs, values, points, 1.0, 1.0, 1e-14)
    assert not happy and cutoff == 0


def test_classic_nonfinite_scale_and_nan_coefficient_contract():
    coeffs = jnp.zeros(20, dtype=jnp.float64).at[0].set(1.0)
    values = jnp.ones(20, dtype=jnp.float64)
    points = chebpts(20, kind=2)
    assert _classic_check(coeffs, values, points, jnp.inf, 1.0, 1e-12) == (False, 20)
    try:
        _classic_check(coeffs.at[2].set(jnp.nan), values, points, 1.0, 1.0, 1e-12)
    except ValueError as exc:
        assert exc.identifier == "CHEBFUN:CHEBTECH:classicCheck:nanEval"
        assert str(exc) == "Function returned NaN when evaluated."
    else:
        raise AssertionError("source classicCheck rejects NaN coefficients")


def test_public_strict_matrix_dispatch_preserves_source_signed_cutoff():
    coeffs = jnp.zeros((32, 2)).at[0, 0].set(2.).at[10, 1].set(-3.)
    for tech in TECHS:
        values = tech.coeffs2vals(coeffs)
        happy, cutoff = tech.happiness_check(
            coeffs, values, tol=1e-12, check="strict", sample_test=False)
        assert happy and cutoff == 1


def test_public_strict_empty_cutoff_sample_test_keeps_full_series():
    coeffs = -jnp.ones(32).at[2:].set(0.)
    for tech in TECHS:
        function = tech(coeffs=coeffs)
        assert tech.happiness_check(
            coeffs, function.values, op=function, tol=1e-12, check="strict"
        ) == (True, None)


def test_strict_negative_exp_keeps_source_full_happy_grid():
    function = Chebtech2.from_function(
        lambda x: -jnp.exp(x), check="strict", tol=1e-12, sample_test=False)
    assert function.ishappy
    assert function.n == 17
    points = jnp.linspace(-1., 1., 101)
    assert jnp.max(jnp.abs(function(points) + jnp.exp(points))) < 1e-12
