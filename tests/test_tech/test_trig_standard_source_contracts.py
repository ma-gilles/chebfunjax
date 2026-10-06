"""Independent source-standard trig count and analytic sample contracts.

Provenance
----------
MATLAB source : @trigtech/standardCheck.m, @trigtech/sampleTest.m
Chebfun commit: 7574c77
Original six test_happinessCheck source assertions remain unchanged elsewhere.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import (
    Trigtech,
    _chop_cutoff_to_ncoeffs,
    _trig_cutoff_decision,
    _trig_standard_check,
    trigpts,
)


def test_source_cutoff_integer_contract_table():
    # Abstract count contract, not a claim even raw counts are reachable
    # through finite exactly-duplicated source coefficient envelopes.
    for n in (16, 17, 32, 33):
        for raw in (1, 2, 3, 4, 8, n):
            happy, keep = _trig_cutoff_decision(raw, n)
            assert happy == (raw < n)
            assert keep == 2*(raw//2)+1
            assert _chop_cutoff_to_ncoeffs(raw, n) == min(keep, n)


@pytest.mark.parametrize("factor,expected", [(.5, True), (2., False)])
def test_source_two_point_test_uses_local_scale(factor, expected):
    values = jnp.ones(33)
    tech = Trigtech.from_values(values)
    seen = []
    def op(x):
        seen.append(np.asarray(x))
        return jnp.ones_like(x)+factor*np.sqrt(np.finfo(float).eps)
    happy, cutoff = Trigtech.happiness_check(tech.coeffs, values, op=op, vscale=8.)
    assert happy is expected
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], [-.357998918959666, .036785641195074])
    assert cutoff == (1 if expected else 33)


@pytest.mark.parametrize("n", [32, 33])
def test_matrix_standard_check_agrees_with_individual_columns(n):
    x = trigpts(n)
    values = jnp.stack([jnp.exp(jnp.cos(jnp.pi*x)),
                        8*jnp.exp(jnp.sin(jnp.pi*x))], axis=1)
    tech = Trigtech.from_values(values)
    local = jnp.max(jnp.abs(values), axis=0)
    eps = np.finfo(float).eps
    together = _trig_standard_check(tech.coeffs, values, eps, 4*local)
    results = [_trig_standard_check(tech.coeffs[:, j], values[:, j], eps, 4*local[j])
               for j in range(2)]
    visited = []
    for happy, keep in results:
        visited.append(keep)
        if not happy:
            break
    assert together == (all(h for h, _ in results), max(visited))


def test_source_sample_rejects_nan_error():
    values = jnp.ones(33)
    tech = Trigtech.from_values(values)
    happy, cutoff = Trigtech.happiness_check(
        tech.coeffs, values, op=lambda x: jnp.full_like(x, jnp.nan))
    assert not happy
    assert cutoff == 33


@pytest.mark.parametrize("scale", [1., 8.])
def test_adaptive_analytic_polynomial_accuracy(scale):
    def op(x):
        return scale*(2+jnp.cos(3*jnp.pi*x)+.25*jnp.sin(5*jnp.pi*x))
    tech = Trigtech.from_function(op)
    xx = jnp.asarray([-.91, -.31, .07, .61, .93])
    assert tech.ishappy
    assert float(jnp.max(jnp.abs(tech(xx)-op(xx)))) < 50*np.finfo(float).eps*scale*3.25


@pytest.mark.parametrize("scale", [1., 1e-12])
@pytest.mark.parametrize("complex_mode", [False, True])
def test_tiny_amplitude_aliasing_refines_with_relative_accuracy(scale, complex_mode):
    def op(x):
        if complex_mode:
            return scale*jnp.exp(17j*jnp.pi*x)
        return scale*jnp.sin(17*jnp.pi*x)
    tech = Trigtech.from_function(op)
    x = jnp.asarray([-.91, -.31, .07, .61, .93])
    assert tech.ishappy
    assert float(jnp.max(jnp.abs(tech(x)-op(x)))) <= 50*np.finfo(float).eps*scale




def test_zero_column_has_source_zero_cutoff_despite_zero_scale_ratio():
    x = trigpts(33)
    values = jnp.stack([jnp.zeros_like(x), jnp.sin(3*jnp.pi*x)], axis=1)
    tech = Trigtech.from_values(values)
    happy, keep = _trig_standard_check(tech.coeffs, values, np.finfo(float).eps, 0.)
    assert happy
    assert keep == 7


def test_column_tolerance_uses_source_max_broadcast():
    x = trigpts(33)
    values = jnp.stack([jnp.exp(jnp.cos(jnp.pi*x)), jnp.exp(jnp.sin(jnp.pi*x))], axis=1)
    tech = Trigtech.from_values(values)
    eps = np.finfo(float).eps
    column_tol = jnp.asarray([[eps], [8*eps]])
    assert _trig_standard_check(tech.coeffs, values, column_tol, 0.) == (
        _trig_standard_check(tech.coeffs, values, 8*eps, 0.))


def test_scalar_nan_coefficients_raise_source_error():
    values = jnp.ones(33)
    coeffs = Trigtech.from_values(values).coeffs.at[0].set(jnp.nan)
    with pytest.raises(ValueError, match="returned NaN"):
        _trig_standard_check(coeffs, values, np.finfo(float).eps, 0.)


def test_single_row_mixed_nan_raises_source_error():
    # MATLAB any() reduces across a row; one invalid column suffices here.
    # This invalid input exercises error dispatch, not coefficient fitting.
    coeffs = jnp.asarray([[1., jnp.nan]], dtype=jnp.complex128)
    with pytest.raises(ValueError, match="returned NaN"):
        Trigtech.happiness_check(coeffs, jnp.ones((1, 2)))
