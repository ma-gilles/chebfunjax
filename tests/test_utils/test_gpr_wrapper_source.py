import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech
from chebfunjax.utils.gpr import _gpr_source_samples, gpr, gpr_chebfun
from chebfunjax.utils.quadrature import chebpts_ab


def _tech(fun):
    assert isinstance(fun, Chebfun)
    return tuple(piece.tech for piece in fun.funs)


def test_mean_trig_flag_is_source_conditional_and_noise_off():
    x = np.array([-1.0, -0.25, 0.4, 1.0])
    y = np.sin(x)
    mean, variance, _ = gpr_chebfun(x, y, length_scale=0.7, trig=True)
    assert _tech(mean) and all(isinstance(t, Trigtech) for t in _tech(mean))
    assert _tech(variance) and all(isinstance(t, Chebtech2) for t in _tech(variance))

    mean_noisy, variance_noisy, _ = gpr_chebfun(
        x, y, length_scale=0.7, trig=True, noise=0.1
    )
    assert _tech(mean_noisy) and all(isinstance(t, Chebtech2) for t in _tech(mean_noisy))
    assert _tech(variance_noisy) and all(isinstance(t, Chebtech2) for t in _tech(variance_noisy))


def test_wrapper_sample_grids_follow_source_layout_and_n_limit():
    x = np.linspace(-1.0, 1.0, 40)
    y = np.sin(x)
    for trig in (False, True):
        options = {"length_scale": 0.35, "trig": trig, "noise": 0.0}
        core = gpr(x, y, **options)
        nodes, values = _gpr_source_samples(
            x, y, core, options, None, trig, 2, mean_fun=core["mean_fn"]
        )
        assert nodes.shape == (800,)
        assert values.shape == (800, 2)
        expected = chebpts_ab(800, *core["domain"])
        np.testing.assert_array_equal(np.asarray(nodes), np.asarray(expected))


def test_wrapper_samples_are_available_with_source_grid_shape_and_type():
    x = np.array([-0.8, -0.1, 0.5, 1.0])
    y = np.cos(x)
    mean, variance, samples = gpr_chebfun(
        x, y, length_scale=0.6, n_samples=2, rng=np.random.default_rng(4)
    )
    assert all(isinstance(t, Chebtech2) for t in _tech(mean))
    assert all(isinstance(t, Chebtech2) for t in _tech(variance))
    assert isinstance(samples, Chebfun)
    assert all(isinstance(t, Chebtech2) for t in _tech(samples))
    result = np.asarray(samples(jnp.linspace(-1, 1, 5)))
    assert result.shape == (5, 2)
    assert np.all(np.isfinite(result))


def test_variance_grid_reconstructs_raw_prediction_to_float64_roundoff():
    from chebfunjax.utils.gpr import (
        _predictive_variance,
        _training_cholesky,
        gpr,
    )
    from chebfunjax.utils.quadrature import chebpts_ab

    x = np.array([-1.0, -0.3, 0.2, 0.7, 1.0])
    y = np.array([0.4, -0.2, 0.6, 0.1, 0.5])
    options = {"length_scale": 0.55, "noise": 0.08, "domain": (-1.0, 1.0)}
    _, variance, _ = gpr_chebfun(x, y, **options)
    core = gpr(x, y, **options)
    n = x.size
    q = min(20 * n, 2000)
    nodes = chebpts_ab(q, *core["domain"])
    factor = _training_cholesky(
        jnp.asarray(x), core["sigma"], core["length_scale"], options["noise"],
        False, 2.0, float(np.max(np.abs(y))),
    )
    raw = _predictive_variance(
        nodes, jnp.asarray(x), factor, core["sigma"], core["length_scale"],
        options["noise"], False, 2.0,
    )
    observed = variance(nodes)
    # A Chebyshev transform/interpolation of q binary64 samples has a
    # roundoff allowance proportional to q*eps and signal scale. This bound
    # is set from operation count/precision, not the measured discrepancy.
    scale = max(1.0, float(jnp.max(jnp.abs(raw))))
    bound = 512.0 * q * np.finfo(np.float64).eps * scale
    assert float(jnp.max(jnp.abs(observed - raw))) <= bound


def test_sample_paths_follow_fitted_mean_and_external_draw_formula():
    from chebfunjax.utils.gpr import (
        _checked_cholesky,
        _predictive_covariance,
        _training_cholesky,
    )
    from chebfunjax.utils.quadrature import chebpts_ab

    x = np.array([-0.8, -0.2, 0.35, 0.9])
    y = np.array([0.2, -0.1, 0.7, 0.5])
    domain = (-1.0, 1.0)
    length_scale = 0.6
    seed = 23
    rng_expected = np.random.default_rng(seed)
    draw = jnp.asarray(rng_expected.standard_normal((80, 2)), dtype=jnp.float64)
    mean, _, samples = gpr_chebfun(
        x, y, domain=domain, length_scale=length_scale, n_samples=2,
        rng=np.random.default_rng(seed),
    )
    nodes = chebpts_ab(80, *domain)
    core = gpr(x, y, domain=domain, length_scale=length_scale)
    scale = float(np.max(np.abs(y)))
    factor = _training_cholesky(
        jnp.asarray(x), core["sigma"], length_scale, 0.0, False, 2.0, scale,
    )
    covariance = _predictive_covariance(
        nodes, jnp.asarray(x), factor, core["sigma"], length_scale, 0.0,
        False, 2.0,
    ) + 1e-12 * scale**2 * x.size * jnp.eye(80, dtype=jnp.float64)
    sample_factor = _checked_cholesky(covariance)
    expected = mean(nodes)[:, None] + sample_factor @ draw
    actual = samples(nodes)
    tolerance = 512.0 * 80 * np.finfo(np.float64).eps * max(
        1.0, float(jnp.max(jnp.abs(expected)))
    )
    assert float(jnp.max(jnp.abs(actual - expected))) <= tolerance
