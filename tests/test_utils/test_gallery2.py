"""Core tests for chebfunjax.utils.gallery2 (Chebfun2 gallery).

Fast entries only; the heavy near-characteristic entries (bump, pegs,
smokering) are exercised in the slow MATLAB-port sweep.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.utils.gallery2 import _REGISTRY, gallery2, list_gallery2

# Fast-to-construct entries with a well-defined ground truth.
FAST = ["challenge", "rosenbrock", "peaks", "squarepeg"]


def _err(f, fa, dom, seed):
    a, b, c, d = dom
    rng = np.random.default_rng(seed)
    xs = rng.uniform(a, b, 8)
    ys = rng.uniform(c, d, 8)
    return max(
        abs(complex(f(jnp.asarray(x), jnp.asarray(y)))
            - complex(fa(jnp.asarray(x), jnp.asarray(y))))
        for x, y in zip(xs, ys))


class TestGallery2:
    @pytest.mark.parametrize("name", FAST)
    def test_matches_anonymous_function(self, name):
        f, fa = gallery2(name, return_handle=True)
        assert isinstance(f, Chebfun2)
        dom = _REGISTRY[name][1]
        assert _err(f, fa, dom, seed=123) < 1e-10

    def test_challenge_is_rank_four(self):
        # The SIAM challenge surface separates as g(x) + h(y) + sin(10(x+y))
        # -> rank 2 + 2 = 4.
        f = gallery2("challenge")
        assert f.rank == 4

    def test_returns_chebfun2_by_default(self):
        f = gallery2("rosenbrock")
        assert isinstance(f, Chebfun2)

    def test_return_handle_gives_callable(self):
        f, fa = gallery2("peaks", return_handle=True)
        x, y = jnp.asarray(0.3), jnp.asarray(-0.4)
        assert abs(complex(f(x, y)) - complex(fa(x, y))) < 1e-8

    def test_list_gallery2_has_all_entries(self):
        names = list_gallery2()
        assert "challenge" in names
        assert len(names) == 11
        assert all(isinstance(v, str) for v in names.values())

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            gallery2("not_a_real_entry")

    def test_case_insensitive(self):
        # Use a cheap entry (rosenbrock) for the name-casing check.
        f1 = gallery2("RosenBrock")
        f2 = gallery2("rosenbrock")
        xs = jnp.asarray(0.2)
        assert abs(complex(f1(xs, xs)) - complex(f2(xs, xs))) < 1e-12
