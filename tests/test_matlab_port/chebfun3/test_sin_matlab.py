"""Port of MATLAB Chebfun tests/chebfun3/test_sin.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``Chebfun3.sin()`` and the
other composition operators exist.

MATLAB pass(4-6) pass a 'fiberDim' constructor flag and pass(8) builds a
'trig' Chebfun3; neither option is exposed by the chebfunjax
constructor, so those four assertions are not ported (the underlying
value they check -- that construction from a Chebfun3 handle reproduces
the function -- is covered by pass(3)).

Provenance
----------
MATLAB source : tests/chebfun3/test_sin.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3

EPS = float(np.finfo(np.float64).eps)
TOL = 1e2 * EPS


class TestChebfun3Sin:
    def test_construct_function_with_sine(self):
        # pass(1): f = sin(x+y+z) evaluates correctly.
        f = chebfun3(lambda x, y, z: jnp.sin(x + y + z))
        assert abs(float(f(0.1, 0.2, 0.3)) - np.sin(0.6)) < TOL

    def test_sine_of_a_chebfun3(self):
        # pass(2): sin(f) composes.
        f = chebfun3(lambda x, y, z: jnp.sin(x + y + z))
        f2 = f.sin()
        assert abs(float(f2(0.2, 0.2, 0.2)) - np.sin(np.sin(0.6))) < TOL

    def test_sine_via_reconstruction(self):
        # pass(3): building sin(f(x,y,z)) from the Chebfun3 handle gives
        # the same result as sin(f).
        f = chebfun3(lambda x, y, z: jnp.sin(x + y + z))
        f2 = chebfun3(lambda x, y, z: jnp.sin(f(x, y, z)))
        assert abs(float(f2(0.2, 0.2, 0.2)) - np.sin(np.sin(0.6))) < TOL

    def test_varying_domain(self):
        # pass(7): sin(2x + 3y + z) on [1,3]^3 at (1,2,3) is sin(11).
        d = (1.0, 3.0, 1.0, 3.0, 1.0, 3.0)
        f3 = chebfun3(lambda x, y, z: jnp.sin(2 * x + 3 * y + z), domain=d)
        assert abs(float(f3(1.0, 2.0, 3.0)) - np.sin(11.0)) < TOL

    def test_loose_tolerance_construction(self):
        # pass(9): a chebfun3 built to a loose tolerance still evaluates
        # to within that tolerance.
        ep = 1e-8
        f5 = chebfun3(lambda x, y, z: jnp.exp(x * y * z), tol=ep)
        assert abs(np.sin(float(f5(0.5, 0.5, 0.5)))
                   - np.sin(np.exp(0.125))) < 1e2 * ep
