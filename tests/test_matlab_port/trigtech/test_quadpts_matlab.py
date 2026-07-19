"""Port of MATLAB Chebfun tests/trigtech/test_quadpts.m (Opus 4.8).

Tests the trigonometric quadrature weights ``trigtech.quadwts(n)``.  chebfunjax
now implements ``Trigtech.quadwts`` (a port of ``@trigtech/quadwts.m``, the
periodic trapezoid-rule weights ``2/n``), so the weight assertions run.

Provenance
----------
MATLAB source : tests/trigtech/test_quadpts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech, trigpts

EPS = float(np.finfo(np.float64).eps)


def _dot(w, v):
    return float(jnp.dot(jnp.asarray(w), jnp.asarray(v)))


class TestTrigtechQuadpts:
    def test_weights_sum_to_two(self):
        w = Trigtech.quadwts(10)
        assert abs(float(jnp.sum(w)) - 2) < 2 * EPS

    def test_weights_annihilate_sin(self):
        w = Trigtech.quadwts(10)
        x = trigpts(10)
        assert abs(_dot(w, jnp.sin(jnp.pi * x))) < EPS

    def test_weights_annihilate_sin_cos(self):
        w = Trigtech.quadwts(10)
        x = trigpts(10)
        assert abs(_dot(w, jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * x))) < 2 * EPS

    def test_weights_integrate_sin_squared(self):
        w = Trigtech.quadwts(10)
        x = trigpts(10)
        assert abs(_dot(w, jnp.sin(2 * jnp.pi * x) ** 2) - 1) < 2 * EPS

    def test_weights_integrate_cos_squared(self):
        w = Trigtech.quadwts(10)
        x = trigpts(10)
        assert abs(_dot(w, jnp.cos(2 * jnp.pi * x) ** 2) - 1) < 2 * EPS

    def test_weights_empty(self):
        assert Trigtech.quadwts(0).shape[0] == 0

    def test_weights_n_one(self):
        assert float(Trigtech.quadwts(1)[0]) == 2.0

    def test_weights_n_two(self):
        assert bool(jnp.all(Trigtech.quadwts(2) == 1))
