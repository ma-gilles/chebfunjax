"""Port of MATLAB Chebfun tests/cheb/test_gallery2.m (Fable 5).

Every MATLAB gallery2 name constructs without crashing and matches its
anonymous function, plus the no-arg random pick and the unknown-name
error.  Marked slow: the near-characteristic ``bump`` entry is high rank
and takes tens of seconds to resolve.

Provenance
----------
MATLAB source : tests/cheb/test_gallery2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.gallery2 import _REGISTRY, gallery2, list_gallery2

# The names exercised by the MATLAB test.
NAMES = ["airyreal", "airycomplex", "challenge", "bump", "peaks",
         "rosenbrock", "smokering", "waffle"]


def _sample_err(f, fa, dom, seed):
    a, b, c, d = dom
    rng = np.random.default_rng(seed)
    xs = rng.uniform(a, b, 6)
    ys = rng.uniform(c, d, 6)
    return max(
        abs(complex(f(jnp.asarray(x), jnp.asarray(y)))
            - complex(fa(jnp.asarray(x), jnp.asarray(y))))
        for x, y in zip(xs, ys))


@pytest.mark.slow
@pytest.mark.matlab
class TestChebGallery2:
    @pytest.mark.parametrize("name", NAMES)
    def test_entry_constructs_and_matches(self, name):
        f, fa = gallery2(name, return_handle=True)
        dom = _REGISTRY[name][1]
        # Self-imposed bound (no MATLAB-mirrored tolerance exists for
        # gallery reconstruction), made vscale-relative in 2026-08: the
        # global-tolerance slice chop (matching MATLAB's quasimatrix
        # simplify) makes absolute error scale with the entry's vscale.
        # airycomplex has vscale 7.1e4; MATLAB R2025b's own chebfun2
        # reconstructs it to 4.6e-8 ABSOLUTE (6.6e-13 relative), ours
        # measures ~1.9e-7 (2.7e-12 relative).  1e5*eps*vscale gives
        # ~8x margin there while keeping 1e-7 for O(1)-vscale entries.
        import numpy as _np
        rng = _np.random.default_rng(0)
        xa, xb, ya, yb = dom
        xs = rng.uniform(xa, xb, 64)
        ys = rng.uniform(ya, yb, 64)
        vscale = float(_np.max(_np.abs(_np.asarray(fa(xs, ys)))))
        tol = max(1e-7, 1e5 * 2.22e-16 * vscale)
        assert _sample_err(f, fa, dom, seed=abs(hash(name)) % 2 ** 31) < tol

    def test_no_arg_returns_random(self):
        f = gallery2()
        assert f is not None

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            gallery2("asdfasdfasdfasdf")

    def test_names_match_matlab(self):
        names = set(list_gallery2().keys())
        # MATLAB gallery2.m defines these 11 entries.
        assert names == {
            "airyreal", "airycomplex", "bump", "challenge", "peaks",
            "rosenbrock", "roundpeg", "smokering", "squarepeg",
            "tiltedpeg", "waffle"}
