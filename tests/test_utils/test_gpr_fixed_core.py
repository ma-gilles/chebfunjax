from __future__ import annotations

from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import pytest

gprmod = import_module("chebfunjax.utils.gpr")

jax.config.update("jax_enable_x64", True)


def test_training_covariance_has_source_total_noise_and_no_duplicate_pair_noise():
    x = jnp.array([0.0, 0.0, 1.0])
    sigma, ell, sigma_y = 2.0, 0.7, 0.25
    base = gprmod._base_covariance(x, x, sigma, ell, False, 2.0)
    training = gprmod._training_covariance(x, sigma, ell, sigma_y, False, 2.0, 1.0)
    # gpr.m forms K=Kbase+sigmaY^2*I, then chol(K+sigmaY^2*I).
    actual_chol = gprmod._training_cholesky(x, sigma, ell, sigma_y, False, 2.0, 1.0)
    expected = base + 2.0 * sigma_y**2 * jnp.eye(3)
    assert jnp.array_equal(training, base + sigma_y**2 * jnp.eye(3))
    assert jnp.allclose(actual_chol @ actual_chol.T, expected, rtol=2e-15, atol=2e-15)
    assert training[0, 1] == base[0, 1]


def test_source_mean_and_predictive_covariance_use_distinct_noise_terms():
    x = jnp.array([0.0, 0.0, 1.0])
    query = jnp.array([0.0, 0.5])
    sigma, ell, sigma_y = 2.0, 0.7, 0.25
    mask = gprmod._coincidence_mask(query, x)
    mean_k = gprmod._mean_covariance(query, x, sigma, ell, sigma_y, False, 2.0)
    cross = gprmod._prediction_cross_covariance(query, x, sigma, ell, sigma_y, False, 2.0)
    self_cov = gprmod._prediction_self_covariance(query, x, sigma, ell, sigma_y, False, 2.0)
    base_cross = gprmod._base_covariance(query, x, sigma, ell, False, 2.0)
    assert jnp.array_equal(mean_k - base_cross, sigma_y * mask)
    assert jnp.array_equal(cross - base_cross, sigma_y**2 * mask)
    assert self_cov[0, 0] == sigma**2 + sigma_y**2
    assert self_cov[1, 1] == sigma**2


def test_noisy_duplicate_predictions_match_explicit_source_linear_algebra():
    # Independent formula control from gpr.m: K gets noise^2 I, Cholesky gets
    # a second noise^2 I; mean uses noise at exact hits; predictive K uses
    # noise^2 at exact hits and Kss adds it once on a diagonal hit.
    x = jnp.array([-1.0, -0.25, -0.25, 0.7])
    y = jnp.array([0.4, -0.1, -0.1, 0.5])
    q = jnp.array([-0.25, 0.1, 0.55])
    sigma, ell, noise = 1.3, 0.42, 0.2
    rxx = x[:, None] - x[None, :]
    base_xx = sigma**2 * jnp.exp(-rxx**2 / (2 * ell**2))
    L = jnp.linalg.cholesky(base_xx + 2 * noise**2 * jnp.eye(x.size))
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))
    rqx = q[:, None] - x[None, :]
    base_qx = sigma**2 * jnp.exp(-rqx**2 / (2 * ell**2))
    exact = (q[:, None] == x[None, :]).astype(jnp.float64)
    expected_mean = (base_qx + noise * exact) @ alpha
    expected_cross = base_qx + noise**2 * exact
    v = jnp.linalg.solve(L, expected_cross.T)
    rqq = q[:, None] - q[None, :]
    self_cov = sigma**2 * jnp.exp(-rqq**2 / (2 * ell**2))
    self_cov = self_cov + noise**2 * jnp.diag(jnp.any(exact != 0, axis=1))
    expected_variance = jnp.diag(self_cov - v.T @ v)

    result = gprmod.gpr(x, y, sigma=sigma, length_scale=ell, noise=noise)
    assert jnp.allclose(result["mean_fn"](q), expected_mean, rtol=2e-14, atol=2e-14)
    assert jnp.allclose(result["variance_fn"](q), jnp.maximum(expected_variance, 0.0),
                        rtol=2e-14, atol=2e-14)


def test_trig_covariance_matches_source_formula():
    xa = jnp.array([-0.3, 0.4])
    xb = jnp.array([0.2, -0.1, 0.7])
    sigma, ell, period = 1.7, 0.45, 2.0
    got = gprmod._base_covariance(xa, xb, sigma, ell, True, period)
    r = xa[:, None] - xb[None, :]
    expected = sigma**2 * jnp.exp(-2.0 / ell**2 * jnp.sin(jnp.pi / period * r) ** 2)
    assert jnp.allclose(got, expected, rtol=0.0, atol=0.0)


def test_fixed_l_public_mapping_is_jax_arraylike_and_keeps_dense_grid_shape():
    result = gprmod.gpr([0.0, 0.5, 1.0], [1.0, -0.2, 0.7], length_scale=0.3, sigma=1.2)
    assert result["x_grid"].shape == (200,)
    assert result["mean"].shape == (200,)
    assert result["variance"].shape == (200,)
    assert isinstance(result["mean"], jax.Array)
    assert jnp.all(result["variance"] >= 0.0)
    assert jnp.allclose(result["mean_fn"](jnp.array([0.5])), jnp.array([-0.2]), atol=2e-12)


def test_core_is_jittable_for_fixed_shapes():
    x = jnp.array([0.0, 0.5, 1.0])
    y = jnp.array([1.0, -0.2, 0.7])
    q = jnp.array([0.25, 0.75])

    def evaluate(x_, y_, q_):
        L = gprmod._training_cholesky(x_, 1.2, 0.3, 0.0, False, 1.0, 1.0)
        alpha = gprmod._solve_cholesky(L, y_)
        mean = gprmod._mean_at(q_, x_, alpha, 1.2, 0.3, 0.0, False, 1.0)
        cov = gprmod._predictive_covariance(q_, x_, L, 1.2, 0.3, 0.0, False, 1.0)
        return mean, jnp.diag(cov)

    expected = evaluate(x, y, q)
    actual = jax.jit(evaluate)(x, y, q)
    assert all(jnp.allclose(a, b, rtol=2e-14, atol=2e-14) for a, b in zip(actual, expected))


def test_eager_cholesky_adapter_checks_finite_indefinite_and_valid_matrices():
    # JAX's raw Cholesky returns NaNs for a finite indefinite matrix; the
    # former NumPy-backed adapter raised LinAlgError. Numeric core stays JITable.
    bad = jnp.array([[1.0, 2.0], [2.0, 1.0]])
    good = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    with pytest.raises(np.linalg.LinAlgError, match="Matrix is not positive definite"):
        gprmod._checked_cholesky(bad)
    factor = gprmod._checked_cholesky(good)
    assert jnp.allclose(factor @ factor.T, good, rtol=2e-15, atol=2e-15)


def test_python_convenience_accepts_external_random_generator_without_numpy_kernel():
    result = gprmod.gpr([0.0, 1.0], [1.0, -1.0], sigma=1.0, length_scale=0.25,
                        n_samples=2, rng=np.random.default_rng(3))
    assert result["samples"].shape == (200, 2)
    assert isinstance(result["samples"], jax.Array)
    assert jnp.all(jnp.isfinite(result["samples"]))
