"""Actual roots across pinned source subdivision coefficient-count boundaries.

Provenance
----------
MATLAB source : @chebtech/roots.m, roots_main
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


@pytest.mark.parametrize("n", [513, 514, 4000, 4001])
def test_actual_chebyshev_roots_across_source_regimes(n, monkeypatch):
    counts = []
    original = module._roots_subdivide

    def observed(c):
        counts.append(len(c))
        return original(c)

    monkeypatch.setattr(module, "_roots_subdivide", observed)
    c = jnp.zeros(n).at[-1].set(1.)
    roots = module._roots_colleague(c)
    exact = jnp.sort(jnp.cos(jnp.pi*(jnp.arange(n-1)+.5)/(n-1)))
    error = float(jnp.max(jnp.abs(roots-exact))) if roots.size == n-1 else None
    report = {"input_count": n, "root_count": roots.size, "maximum_error": error,
              "bound": n*EPS, "subdivision_counts": dict(Counter(counts))}
    out = os.environ.get("CHEBFUN_RUNTIME_REPORT")
    if out:
        (Path(out).parent / f"actual_roots_{n}.json").write_text(json.dumps(report, indent=2)+"\n")
    assert counts[0] == n
    assert roots.size == n-1
    # Same length(f)*eps bound as original oscillatory source roots test.
    assert error < n*EPS


@pytest.mark.parametrize("tail,expected_split", [(4*EPS, False), (6*EPS, True)])
def test_actual_trim_threshold_controls_branch(tail, expected_split, monkeypatch):
    counts = []
    original = module._roots_subdivide

    def observed(c):
        counts.append(len(c))
        return original(c)

    monkeypatch.setattr(module, "_roots_subdivide", observed)
    c = jnp.zeros(514).at[49].set(1.).at[-1].set(tail)
    roots = module._roots_colleague(c)
    assert (514 in counts) == expected_split
    assert roots.size == 49
    exact = jnp.sort(jnp.cos(jnp.pi*(jnp.arange(49)+.5)/49))
    assert float(jnp.max(jnp.abs(roots-exact))) < 514*EPS
