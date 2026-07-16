"""Port of MATLAB Chebfun tests/trigtech/test_mat2cell.m (Opus 4.8).

mat2cell splits an array-valued trigtech into a list of trigtechs.  Array-valued
trigtechs are now supported (FIXED, Fable 5, Big-Three array-valued epic).
MATLAB ``mat2cell(f, 1, [1 2])`` maps to ``f.mat2cell([1, 2])`` (the row
argument is implicit for a single tech).

Provenance
----------
MATLAB source : tests/trigtech/test_mat2cell.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)


class TestTrigtechMat2cell:
    def _split(self):
        f = Trigtech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x), jnp.exp(jnp.cos(jnp.pi * x))],
                axis=-1,
            )
        )
        g = Trigtech.from_function(lambda x: jnp.sin(jnp.pi * x))
        h = Trigtech.from_function(
            lambda x: jnp.stack(
                [jnp.cos(jnp.pi * x), jnp.exp(jnp.cos(jnp.pi * x))], axis=-1
            )
        )
        return f.mat2cell([1, 2]), g, h

    def test_split_first(self):
        # pass(1): sum(F{1} - g) < 10*vscale(g)*eps.  MATLAB sum() is the
        # definite integral of the difference tech (a per-column integral).
        # FIXED (Fable 5, Big-Three array-valued epic).
        F, g, h = self._split()
        assert abs(complex((F[0] - g).sum())) < 10 * g.vscale * EPS

    def test_split_rest(self):
        # pass(2): all( sum(F{2} - h) < 10*max(vscale(h)*eps) ).
        # FIXED (Fable 5, Big-Three array-valued epic).
        F, g, h = self._split()
        diff = np.asarray((F[1] - h).sum())
        assert bool(np.all(np.abs(diff) < 10 * h.vscale * EPS))
