"""Source separableApprox FEVAL maps, Chebfun7574c77."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech


def _factor_disk():
    return Diskfun(cols=[Chebtech2.from_coeffs(jnp.asarray([.1, .3, -.2]))],
                   rows=[Trigtech.from_coeffs(jnp.asarray([.5, 0., .5]), is_real=True)],
                   pivots=jnp.ones(1), idx_plus=(0,), idx_minus=())


@pytest.mark.parametrize("factor", ["column", "row"])
def test_actual_factor_queries_use_native_bounded_maps(factor, monkeypatch):
    f = _factor_disk()
    theta = jnp.asarray([-.51, .24, 2.38])
    radius = jnp.asarray([.1, .37, .9])
    seen = []
    cls = Chebtech2 if factor == "column" else Trigtech
    original = cls.__call__
    def observe(self, query):
        seen.append(query)
        return original(self, query)
    monkeypatch.setattr(cls, "__call__", observe)
    with jax.disable_jit():
        f(theta, radius)
    with np.load(Path(__file__).parent/"fixtures/disk_source_map_native.npz") as saved:
        expected = jnp.asarray(saved["independent_r_mapped" if factor == "column"
                                     else "independent_theta_mapped"][:3])
    assert len(seen) == 1
    assert jnp.array_equal(seen[0], expected)


@pytest.mark.parametrize("stage", [0, 22])
@pytest.mark.parametrize("compiled", [False, True])
def test_same_native_factors_and_polar_stage_words(stage, compiled):
    # NumPy loads the retained fixture; all numerical evaluation is JAX.
    with np.load(Path(__file__).parent/"fixtures/disk_source_map_native.npz") as saved:
        a = {key: jnp.asarray(saved[key]) for key in saved.files}
    f = Diskfun(cols=[Chebtech2.from_coeffs(a["C"])],
                rows=[Trigtech.from_coeffs(a["R"], is_real=True)],
                pivots=1/a["D"].reshape(1), idx_plus=(), idx_minus=(0,))
    with jax.disable_jit(not compiled):
        actual = f(a[f"stage_{stage}_theta"], a[f"stage_{stage}_r"])
    expected = a[f"stage_{stage}_inner"]
    assert actual.shape == expected.shape
    assert jnp.array_equal(actual.view(jnp.uint64), expected.view(jnp.uint64))


@pytest.mark.parametrize("compiled", [False, True])
def test_native_independent_origin_boundary_and_tiny_function(compiled):
    with np.load(Path(__file__).parent/"fixtures/disk_source_map_native.npz") as saved:
        a = {key: jnp.asarray(saved[key]) for key in saved.files}
    f = Diskfun(cols=[Chebtech2.from_coeffs(a["C"])],
                rows=[Trigtech.from_coeffs(a["R"], is_real=True)],
                pivots=1/a["D"].reshape(1), idx_plus=(), idx_minus=(0,))
    with jax.disable_jit(not compiled):
        actual = f(a["probe9_theta"], a["probe9_r"])
    # Source test_feval.m vector predicate uses norm2 and 1000*cheb2eps.
    # Strict function-word diagnostic differs at one probe by1ULP; native
    # map-word controls stay exact and source disk31/32 bounds stay unchanged.
    tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
    assert float(jnp.linalg.norm(actual-a["probe9_inner"])) < tol


@pytest.mark.parametrize("family", ["theta", "r"])
@pytest.mark.parametrize("case", ["stage22", "independent"])
@pytest.mark.parametrize("compiled", [False, True])
def test_source_map_native_words_and_nonfinite_masks(family, case, compiled):
    from chebfunjax.diskfun.diskfun import _disk_inverse_map
    with np.load(Path(__file__).parent/"fixtures/disk_source_map_native.npz") as saved:
        x = jnp.asarray(saved[case+"_"+family])
        expected = jnp.asarray(saved[case+"_"+family+"_mapped"])
    a, b = (-1., 1.) if family == "r" else (-jnp.pi, jnp.pi)
    with jax.disable_jit(not compiled):
        actual = jax.jit(lambda x: _disk_inverse_map(x, a, b))(x)
    assert jnp.array_equal(jnp.isnan(actual), jnp.isnan(expected))
    assert jnp.array_equal(jnp.isposinf(actual), jnp.isposinf(expected))
    assert jnp.array_equal(jnp.isneginf(actual), jnp.isneginf(expected))
    finite = jnp.isfinite(expected)
    assert jnp.array_equal(jnp.where(finite, actual, 0.).view(jnp.uint64),
                           jnp.where(finite, expected, 0.).view(jnp.uint64))


def test_source_map_linear_first_and_second_jvp():
    from chebfunjax.diskfun.diskfun import _disk_inverse_map
    x = jnp.linspace(-2., 2., 11)
    def slope(x):
        return jax.jvp(lambda x: _disk_inverse_map(x, -jnp.pi, jnp.pi),
                       (x,), (jnp.ones_like(x),))[1]
    first, second = jax.jit(lambda x: jax.jvp(slope, (x,), (jnp.ones_like(x),)))(x)
    assert float(jnp.max(jnp.abs(first-1/jnp.pi))) < 4*jnp.finfo(jnp.float64).eps
    assert jnp.array_equal(second, jnp.zeros_like(x))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex128])
def test_source_map_other_dtype_fallback(dtype):
    from chebfunjax.diskfun.diskfun import _disk_inverse_map
    from chebfunjax.domain import _linear_inverse_map
    x = jnp.asarray([-.7, .2, 1.3], dtype=dtype)
    if dtype == jnp.complex128:
        x = x + .3j
    actual = jax.jit(lambda x: _disk_inverse_map(x, -jnp.pi, jnp.pi))(x)
    expected = jax.jit(lambda x: _linear_inverse_map(x, -jnp.pi, jnp.pi))(x)
    assert actual.dtype == expected.dtype
    assert jnp.array_equal(actual, expected)
