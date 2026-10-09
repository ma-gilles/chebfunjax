"""Source iteration semantics independently exercised; no residual rescue.

Provenance
----------
MATLAB source : @chebfun/inv.m, fInverseRegulaFalsi, fInverseIllinois
Chebfun commit: 7574c77
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def module():
    return importlib.import_module("chebfunjax.chebfun1d.inverse")


@pytest.mark.parametrize("illinois", [False, True])
@pytest.mark.parametrize("slope", [2.0, -2.0])
def test_affine_exact_endpoints_interior_and_jit(illinois, slope):
    solve = module()._false_position
    expected = jnp.array([-2.0, -0.5, 1.0])
    target = slope * expected + 3

    def f(x):
        return slope * x + 3

    actual = solve(f, target, -2.0, 1.0, illinois)
    compiled = jax.jit(lambda y: solve(f, y, -2.0, 1.0, illinois))(target)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(compiled, expected)


def test_illinois_subset_logical_mask_targets_prefix_not_selected_positions():
    # Source fb(side(I1)==-1): side(I1) is[-1,1], hence logical[true,false]
    # addresses fb(1), although I1 selected original positions2and4.
    mask = module()._illinois_source_prefix_mask
    actual = mask(jnp.array([False, True, False, True]), jnp.array([1, -1, -1, 1]), -1)
    np.testing.assert_array_equal(actual, [True, False, False, False])


@pytest.mark.parametrize("illinois", [False, True])
def test_source_stagnation_returns_without_hidden_brent(monkeypatch, illinois):
    mod = module()

    def forbidden(*args, **kwargs):
        raise AssertionError("Source false-position does not invoke Brent")

    monkeypatch.setattr(mod, "_brent", forbidden)

    # Adversarial source-clause check, not a claim of a continuous inverse:
    # initial c=.5; huge interior residual gives next c=0; next c remains0.
    # Source stops on repeated iterates despite residual-.5.
    def f(x):
        return jnp.where(x == 0.0, 0.0, jnp.where(x == 1.0, 1.0, 2.0**100))

    actual = mod._false_position(f, jnp.array([0.5]), 0.0, 1.0, illinois)
    np.testing.assert_array_equal(actual, [0.0])


@pytest.mark.parametrize("illinois", [False, True])
def test_open_source_iteration_cap_raises_not_partial(illinois):
    def f(x):
        return jnp.where(x == 0.0, 0.0, jnp.where(x == 1.0, 1.0, 2.0**100))

    with pytest.raises(Exception, match="safety cap"):
        module()._false_position(f, jnp.array([0.5]), 0.0, 1.0, illinois, max_iterations=1)


@pytest.mark.parametrize("illinois", [False, True])
def test_absolute_eps_stops_small_domain_without_scaled_threshold(illinois):
    # The first displacement2**-71 is below absoluteeps, but above eps*2**-70.
    # One permitted source step therefore completes; a scaled condition
    # would still be open and should exhaust this explicit one-step cap.
    b = 2.0**-70

    def f(x):
        return jnp.where(x == 0.0, 0.0, jnp.where(x == b, 1.0, 2.0**100))

    actual = module()._false_position(f, jnp.array([0.5]), 0.0, b, illinois, max_iterations=1)
    np.testing.assert_array_equal(actual, [0.0])


@pytest.mark.parametrize("illinois", [False, True])
def test_scalar_and_empty_shape_adapters(illinois):
    def f(x):
        return 2 * x + 1

    scalar = module()._false_position(f, jnp.asarray(0.5), -1.0, 1.0, illinois)
    empty = module()._false_position(f, jnp.empty((0,)), -1.0, 1.0, illinois)
    assert scalar.shape == ()
    assert float(scalar) == -0.25
    assert empty.shape == (0,)


@pytest.mark.parametrize("illinois", [False, True])
def test_compiled_solver_retains_changed_chebfun_coefficients(illinois):
    from chebfunjax import Chebfun

    first = Chebfun.from_coeffs(jnp.asarray([0.0, 1.0]))
    second = Chebfun.from_coeffs(jnp.asarray([1.0, 2.0]))
    targets = jnp.asarray([-0.5, 0.5])
    before = module()._false_position(first, targets, -1.0, 1.0, illinois)
    after = module()._false_position(second, targets, -1.0, 1.0, illinois)
    np.testing.assert_array_equal(before, targets)
    np.testing.assert_array_equal(after, (targets - 1) / 2)
