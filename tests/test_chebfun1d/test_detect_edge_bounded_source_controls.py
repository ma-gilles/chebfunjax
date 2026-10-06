"""Independent bounded source detector source contracts.

Provenance
----------
MATLAB source : @fun/detectEdge.m
Chebfun commit: 7574c77
"""

import importlib
import math

import jax.numpy as jnp
import pytest


def _module():
    return importlib.import_module("chebfunjax.chebfun1d.chebfun")


@pytest.mark.parametrize("x", [-1.0, -0.375, 0.0, 2.0**-1022])
def test_matlab_eps_is_positive_and_preserves_subnormal_spacing(x):
    assert _module()._edge_eps(x) == math.ulp(x)
    assert _module()._edge_eps(x) > 0.0


def test_derivatives_keep_sample_rows_and_reduce_function_columns():
    def columns(x):
        return jnp.stack((x, 1000 * x), axis=-1)

    _, _, derivatives = _module()._edge_find_max_der(columns, 0.0, 1.0, 1, 9)
    assert derivatives.shape == (1,)
    assert float(derivatives[0]) == 1000.0


def test_nonfinite_samples_are_retained_and_all_nan_derivatives_stay_nan():
    module = _module()
    rows = module._edge_sample_rows(
        lambda x: jnp.array([jnp.inf, jnp.nan]), jnp.array([0.0, 1.0]))
    assert rows.shape == (2, 1)
    assert bool(jnp.isinf(rows[0, 0]))
    assert bool(jnp.isnan(rows[1, 0]))
    _, _, derivatives = module._edge_find_max_der(
        lambda x: jnp.full_like(x, jnp.nan), 0.0, 1.0, 4, 50)
    assert bool(jnp.all(jnp.isnan(derivatives)))


def test_findjump_initial_source_denominator_includes_half_factor():
    calls = []

    def linear(x):
        calls.append(jnp.asarray(x))
        return 0.75e-5 * x

    _module()._find_jump(linear, 0.0, 1.0, 1.0, 1.0)
    # Without the denominator's /2, initial derivative0.75e-5 is rejected
    # immediately. Source derivative1.5e-5 enters the refinement loop.
    assert calls[0].shape == (2,)
    assert len(calls) > 1
    assert calls[1].ndim == 0


@pytest.mark.parametrize("scale,expected", [(0.0, 0.0), ([1.0, 2.0], 2.0)])
def test_source_scale_reduction_is_forwarded_without_floor(monkeypatch, scale, expected):
    module = _module()
    requests = []

    def derivatives(f, a, b, num, grid):
        requests.append(grid)
        # First bracket is already small enough for source findJump once
        # derivative growth has selected order1 in the second call.
        values = jnp.zeros(num) if len(requests) == 1 else jnp.full(num, 100.0)
        return jnp.zeros(num), jnp.full(num, 1e-4), values

    forwarded = []

    def jump(f, a, b, vscale, hscale):
        forwarded.append((a, b, vscale, hscale))
        return 5e-5

    monkeypatch.setattr(module, "_edge_find_max_der", derivatives)
    monkeypatch.setattr(module, "_find_jump", jump)
    edge = module._detect_edge_matlab(lambda x: x, 0.0, 1.0,
                                      vscale=scale, hscale=1.0)
    assert requests == [50, 15]
    assert forwarded == [(0.0, 1e-4, expected, 1.0)]
    assert edge == 5e-5


@pytest.mark.parametrize("location,domain", [(0.125, (-1.0, 1.0)),
                                            (-0.375, (-1.0, 0.0))])
def test_array_jump_visible_only_in_second_column(location, domain):
    def columns(x):
        x = jnp.asarray(x)
        return jnp.stack((jnp.zeros_like(x), (x >= location).astype(jnp.float64)),
                         axis=-1)

    edge = _module()._detect_edge_matlab(columns, *domain, vscale=1.0, hscale=1.0)
    assert edge is not None
    # A binary64 threshold step has exact constant one-sided values; source
    # bisection should terminate at an adjacent float, without a broad fit tol.
    assert abs(edge - location) <= 2 * math.ulp(location)
