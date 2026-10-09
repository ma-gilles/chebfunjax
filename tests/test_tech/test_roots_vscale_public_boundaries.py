"""Public own-grid vscale path at pinned native roots branch boundaries.

Provenance
----------
MATLAB source : @chebtech/roots.m, @chebtech/vscale.m
Chebfun commit: 7574c77
"""
import json
import os
from collections import Counter
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.tech import chebtech as module

EPS = float(jnp.finfo(jnp.float64).eps)
TECHS = [module.Chebtech1, module.Chebtech2]


def _observe_engine(monkeypatch):
    normalized, subdivisions = [], []
    original_main = module._roots_main
    original_subdivide = module._roots_subdivide

    def main(c, *args, **kwargs):
        normalized.append(jnp.asarray(c))
        return original_main(c, *args, **kwargs)

    def subdivide(c):
        subdivisions.append(len(c))
        return original_subdivide(c)

    monkeypatch.setattr(module, "_roots_main", main)
    monkeypatch.setattr(module, "_roots_subdivide", subdivide)
    return normalized, subdivisions


def _report(name, report):
    destination = os.environ.get("CHEBFUN_RUNTIME_REPORT")
    if destination:
        (Path(destination).parent / f"{name}.json").write_text(
            json.dumps(report, indent=2) + "\n")


@pytest.mark.parametrize("Tech", TECHS)
@pytest.mark.parametrize("n", [50, 51, 513, 514, 4000, 4001])
def test_public_shifted_chebyshev_source_boundaries(Tech, n, monkeypatch):
    # T_d - 1/4 has d distinct real roots and a nontrivial native vscale.
    c = jnp.zeros(n).at[0].set(-.25).at[-1].set(1.)
    tech = Tech(coeffs=c)
    scale = tech.vscale
    literal_normalized = c / scale
    normalized, subdivisions = _observe_engine(monkeypatch)
    roots = tech.roots()
    exact = jnp.sort(jnp.cos((2*jnp.pi*jnp.arange(n-1)
                             + jnp.arccos(.25))/(n-1)))
    error = (float(jnp.max(jnp.abs(roots-exact)))
             if roots.size == n-1 else None)
    _report(f"public_boundary_{Tech.__name__}_{n}", {
        "tech": Tech.__name__, "input_count": n, "scale": float(scale),
        "root_count": roots.size, "maximum_error": error, "bound": n*EPS,
        "subdivision_counts": dict(Counter(subdivisions)),
        "normalized_first": float(normalized[0][0]),
        "normalized_last": float(normalized[0][-1]),
    })
    assert normalized[0].shape == (n,)
    # Source scale division is the same literal JAX operation at the adapter.
    assert bool(jnp.array_equal(normalized[0], literal_normalized))
    assert bool(subdivisions) == (n > 50)
    if n > 50:
        assert subdivisions[0] == n
    assert roots.size == n-1
    # Retain the pinned oscillatory native source's length(f)*eps bound.
    assert error < n*EPS


@pytest.mark.parametrize("Tech", TECHS)
@pytest.mark.parametrize("amplitude", [1., -3., 1+2j, 1e-200, 1e200])
@pytest.mark.parametrize("tail,split", [(4*EPS, False), (6*EPS, True)])
def test_public_scale_preserves_relative_trim_boundary(
        Tech, amplitude, tail, split, monkeypatch):
    c = amplitude*jnp.zeros(514).at[49].set(1.).at[-1].set(tail)
    tech = Tech(coeffs=c)
    normalized, subdivisions = _observe_engine(monkeypatch)
    roots = tech.roots()
    expected = jnp.sort(jnp.cos(jnp.pi*(jnp.arange(49)+.5)/49))
    assert bool(jnp.all(jnp.isfinite(normalized[0])))
    assert (514 in subdivisions) == split
    assert roots.size == 49
    assert float(jnp.max(jnp.abs(roots-expected))) < 514*EPS
