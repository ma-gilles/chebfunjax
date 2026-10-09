"""Literal pinned roots subdivision branches and independent polynomial controls.

Provenance
----------
MATLAB source : @chebtech/roots.m, roots_main and chebptsAB
Chebfun commit: 7574c77
"""
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech import chebtech as module
from chebfunjax.utils.quadrature import chebpts

EPS = float(jnp.finfo(jnp.float64).eps)
SPLIT = -0.004849834917525


def _save(name, data):
    root = os.environ.get("CHEBFUN_RUNTIME_REPORT")
    if root:
        (Path(root).parent / name).write_text(json.dumps(data, indent=2)+"\n")


@jax.jit
def _literal_matrices():
    nodes = chebpts(513, kind=2)
    out = []
    for a, b in ((-1., SPLIT), (SPLIT, 1.)):
        x = b * (nodes + 1) / 2 + a * (1 - nodes) / 2
        values = jnp.ones((513, 513)).at[:, 1].set(x)

        def body(k, values):
            return values.at[:, k].set(2 * x * values[:, k-1] - values[:, k-2])

        values = jax.lax.fori_loop(2, 513, body, values)
        values = jnp.concatenate((values[512:0:-1], values[:512]))
        values = jnp.real(jnp.fft.fft(values, axis=0) / 512)
        out.append(jnp.triu(jnp.concatenate((.5*values[:1], values[1:512],
                                             .5*values[512:513]))))
    return tuple(out)


def test_fixed_transform_build_and_concrete_cache():
    module._roots_cached_transforms.cache_clear()
    device = jnp.asarray(0.).device
    begin = time.perf_counter()
    matrices = module._roots_cached_transforms(device, jax.config.x64_enabled)
    jax.block_until_ready(matrices)
    first = time.perf_counter()-begin
    begin = time.perf_counter()
    second = module._roots_cached_transforms(device, jax.config.x64_enabled)
    jax.block_until_ready(second)
    warm = time.perf_counter()-begin
    reference = _literal_matrices()
    errors = [float(jnp.max(jnp.abs(a-b))) for a, b in zip(matrices, reference)]
    _save("matrix_build.json", {"cold_seconds": first, "cached_seconds": warm,
                                "literal_max_errors": errors,
                                "cache": str(module._roots_cached_transforms.cache_info())})
    assert matrices is second
    assert all(isinstance(a, jax.Array) and not isinstance(a, jax.core.Tracer) for a in matrices)
    assert all(a.shape == (513, 513) for a in matrices)
    # MATLAB has no bitwise matrix-builder predicate. The original exact
    # diagnostic is retained in failed helpers_v1/v2 and the separate capture.
    # Two valid compilation contexts differed by <= 1.11e-16 in that diagnostic.
    # This PREDECLARED bound instead follows finite float64 operation counts:
    # mapped T_k sensitivity and recurrence propagation grow as (k+1)^2;
    # gamma80 covers ten FFT stages with eight real operations per stage.
    # Conservative factors cover both builders/normalizations (MATRIX_BOUND_v1).
    # This is not a rigorous backend FFT proof; original root bounds stay exact.
    u = EPS / 2
    gamma80 = 80*u / (1-80*u)
    bound = 64*u*(jnp.arange(513)+1)**2 + 8*gamma80
    assert all(bool(jnp.all(jnp.abs(a-b) <= bound[None, :]))
               for a, b in zip(matrices, reference))
    assert all(float(jnp.max(jnp.abs(jnp.tril(a, -1)))) == 0 for a in matrices)
    # Independent exact first-degree Chebyshev affine identity.
    for matrix, a, b in zip(matrices, (-1., SPLIT), (SPLIT, 1.)):
        assert float(jnp.max(jnp.abs(matrix[:, 0] - jnp.eye(513)[:, 0]))) == 0
        assert abs(float(matrix[0, 1]) - (a+b)/2) < 10*EPS
        assert abs(float(matrix[1, 1]) - (b-a)/2) < 10*EPS


@pytest.mark.parametrize("n", [50, 51, 513, 514, 4000, 4001])
def test_source_weighted_mapping(n):
    nodes = chebpts(n, kind=2)
    for a, b in ((-1., SPLIT), (SPLIT, 1.)):
        expected = jax.jit(lambda x, a, b: b*(x+1)/2 + a*(1-x)/2)(nodes, a, b)
        actual = module._roots_mapped_points(n, a, b)
        assert bool(jnp.array_equal(actual, expected))
        assert float(actual[0]) == a
        assert float(actual[-1]) == b


@pytest.mark.parametrize("n", [513, 514, 4000, 4001])
def test_actual_subdivision_branch_and_low_degree_identity(n, monkeypatch):
    calls = []
    original_matrix = module._roots_apply_transforms
    original_clenshaw = module._clenshaw
    original_ndct = module.ndct

    def matrix(c, left, right):
        calls.append("matrix")
        return original_matrix(c, left, right)

    def clenshaw(c, x):
        calls.append("clenshaw")
        return original_clenshaw(c, x)

    def ndct(x, c):
        calls.append("ndct")
        return original_ndct(x, c)

    monkeypatch.setattr(module, "_roots_apply_transforms", matrix)
    monkeypatch.setattr(module, "_clenshaw", clenshaw)
    monkeypatch.setattr(module, "ndct", ndct)
    module._roots_sampled_subdivision.clear_cache()
    c = jnp.zeros(n).at[:3].set(jnp.asarray([.25, .5, -.125]))
    children = module._roots_subdivide(c)
    assert calls == ["matrix" if n <= 513 else "clenshaw" if n <= 4000 else "ndct"]
    x = jnp.linspace(-1., 1., 37)
    errors = []
    for child, a, b in zip(children, (-1., SPLIT), (SPLIT, 1.)):
        mapped = b*(x+1)/2+a*(1-x)/2
        expected = .25+.5*mapped-.125*(2*mapped*mapped-1)
        actual = jax.jit(original_clenshaw)(child, x)
        errors.append(float(jnp.max(jnp.abs(actual-expected))))
    _save(f"branch_{n}.json", {"calls": calls, "errors": errors})
    # Independent degree-two identity: 30 epsilon allows FFT/NDCT roundoff,
    # far tighter than the original roots test's length(f)*eps scale here.
    assert max(errors) < 30*EPS


@pytest.mark.parametrize("n", [50, 51])
def test_direct_threshold_is_coefficient_count(n, monkeypatch):
    calls = []
    original = module._roots_subdivide

    def observed(c):
        calls.append(len(c))
        return original(c)

    monkeypatch.setattr(module, "_roots_subdivide", observed)
    c = jnp.zeros(n).at[-1].set(1.)
    roots = module._roots_colleague(c)
    exact = jnp.sort(jnp.cos(jnp.pi*(jnp.arange(n-1)+.5)/(n-1)))
    assert roots.size == n-1
    assert float(jnp.max(jnp.abs(roots-exact))) < n*EPS
    assert (n in calls) == (n > 50)


def test_recurse_false_and_trimmed_count_precede_subdivision(monkeypatch):
    calls = []
    original = module._roots_subdivide

    def observed(c):
        calls.append(len(c))
        return original(c)

    monkeypatch.setattr(module, "_roots_subdivide", observed)
    module._roots_colleague(jnp.zeros(51).at[-1].set(1.), recurse=False)
    assert calls == []
    c = jnp.zeros(514).at[49].set(1.).at[-1].set(EPS)
    roots = module._roots_colleague(c)
    assert calls == []
    assert roots.size == 49


@pytest.mark.parametrize("n", [513, 514, 4001])
def test_complex_and_pure_imaginary_source_policy(n):
    real = jnp.zeros(n).at[:3].set(jnp.asarray([.25, .5, -.125]))
    complex_children = module._roots_subdivide(real*(1+2j))
    imaginary_children = module._roots_subdivide(real*1j)
    real_children = module._roots_subdivide(real)
    for complex_child, imaginary_child, real_child in zip(complex_children, imaginary_children, real_children):
        assert float(jnp.max(jnp.abs(complex_child-(1+2j)*real_child))) < 30*EPS
        # Public NDCT has a native pure-imaginary-to-real output quirk.
        expected = real_child if n > 4000 else 1j*real_child
        assert float(jnp.max(jnp.abs(imaginary_child-expected))) < 30*EPS
