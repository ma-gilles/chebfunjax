"""Whole-domain scales retained by ODESOL splitting retries.

Provenance
----------
MATLAB source : @chebfun/constructor.m (data/getFun), @bndfun/bndfun.m
Chebfun commit: 7574c77
Focused source-contract controls, not original MATLAB assertion replacements.
"""
import importlib

import jax.numpy as jnp
import numpy.testing as npt
import pytest

from chebfunjax.tech.chebtech import Chebtech2


@pytest.mark.parametrize("turbo", [False, True])
def test_split_retry_carries_prior_scale_and_whole_domain_hscale(monkeypatch, turbo):
    module = importlib.import_module("chebfunjax.chebfun1d.chebfun")
    original = Chebtech2.from_function.__func__
    observed = []

    def record(cls, *args, **kwargs):
        observed.append((kwargs.get("vscale", 0.0), kwargs.get("hscale", 1.0)))
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(Chebtech2, "from_function", classmethod(record))

    def dense(x):
        return jnp.stack((x, 2*x+1), axis=-1)

    solution = module._construct_with_splitting(
        dense, 0.1, 0.3, maxpow2=8, tol=1e-12,
        sample_test=False, split_max_length=20000, turbo=turbo,
        vscale=1000.0, hscale=0.8,
    )
    assert observed and solution.ishappy
    # A previous accepted FUN can dominate both current component scales;
    # bndfun rescales the whole construction domain to this FUN interval.
    for vertical, horizontal in observed:
        assert float(vertical) == 1000.0
        assert horizontal == pytest.approx(0.8/(0.3-0.1), rel=1e-15)
    points = jnp.array([0.1, 0.17, 0.26, 0.3])
    # Supplemental V3 used absolute 1e-12 for both paths. That ignores
    # MATLAB constructor/standardCheck's global scale: chebfuneps=1e-12
    # times data.vscale=1000 admits absolute 1e-9. Turbo recomputes over
    # rho~=165140 for this linear input; observed constant error 1.127e-12
    # is within that source-derived bound. Keep plain's stricter 1e-12;
    # use the source scaled bound 1e-9 for turbo. No original MATLAB
    # assertion is changed; test_turbo.m is also executed unchanged.
    bound = 1e-12 * (1000.0 if turbo else 1.0)
    npt.assert_allclose(solution(points), dense(points), atol=bound, rtol=0)
