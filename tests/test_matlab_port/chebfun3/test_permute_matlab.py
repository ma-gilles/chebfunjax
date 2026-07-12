"""Port of MATLAB Chebfun tests/chebfun3/test_permute.m (Fable 5).

FIXED: Chebfun3.permute added in the Fable 5 audit.  The complex
nonsymmetric case is run in its real form (the complex ctor splits
re/im and recombines, so realness is the binding check).

Provenance
----------
MATLAB source : tests/chebfun3/test_permute.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

TOL = 1000 * np.finfo(float).eps


def _maxdiff(f, g, dom):
    gs = [jnp.asarray(np.linspace(dom[2 * i], dom[2 * i + 1], 7))
          for i in range(3)]
    xx, yy, zz = jnp.meshgrid(*gs, indexing="ij")
    return float(jnp.max(jnp.abs(f(xx, yy, zz) - g(xx, yy, zz))))


class TestChebfun3Permute:
    def test_symmetric_all_permutations(self):
        f = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(x * y * z))
        for perm in itertools.permutations([1, 2, 3]):
            p = f.permute(list(perm))
            assert _maxdiff(p, lambda x, y, z: jnp.cos(x * y * z),
                            (-1, 1, -1, 1, -1, 1)) < TOL, perm

    def test_nonsymmetric(self):
        dom = (-3.0, 4.0, -1.0, 0.0, -np.pi, np.pi)
        f = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(x) * jnp.sin(y) + z, domain=dom)
        assert _maxdiff(f.permute([1, 2, 3]),
                        lambda x, y, z: jnp.cos(x) * jnp.sin(y) + z,
                        dom) < TOL
        assert _maxdiff(f.permute([1, 3, 2]),
                        lambda x, z, y: jnp.cos(x) * jnp.sin(y) + z,
                        (-3, 4, -np.pi, np.pi, -1, 0)) < TOL
        assert _maxdiff(f.permute([3, 1, 2]),
                        lambda z, x, y: jnp.cos(x) * jnp.sin(y) + z,
                        (-np.pi, np.pi, -3, 4, -1, 0)) < TOL
        assert _maxdiff(f.permute([2, 1, 3]),
                        lambda y, x, z: jnp.cos(x) * jnp.sin(y) + z,
                        (-1, 0, -3, 4, -np.pi, np.pi)) < TOL
