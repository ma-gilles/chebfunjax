"""Port of MATLAB Chebfun tests/chebfun3/test_plotting.m (Fable 5).

MATLAB's 'noslider' flag selects the non-interactive variant, which is
the only headless behaviour; ``slice(f, .5, -.3, .9)`` maps to the
xslices/yslices/zslices keywords, ``scan(f, k, 'hold')`` to
``scan(dim=k, hold=True)``.  ``cos(1i*w)`` is written as ``cosh(w)``
(identical function, real arithmetic).  As in MATLAB, the assertions
only check that nothing crashes.

Provenance
----------
MATLAB source : tests/chebfun3/test_plotting.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

jax.config.update("jax_enable_x64", True)


def _does_not_crash(fn):
    import matplotlib.pyplot as plt
    try:
        fn()
        return True
    finally:
        plt.close("all")


class TestChebfun3Plotting:
    def test_all_matlab_assertions(self):
        f1 = Chebfun3.from_function(lambda x, y, z: x * y * z,
                                    domain=(-1, 2, -1, 2, -1, 2))
        f2 = Chebfun3.from_function(
            lambda x, y, z: jnp.exp(jnp.cosh(x * y * z)))

        assert _does_not_crash(lambda: f1.plot())                       # 1
        assert _does_not_crash(lambda: f1.slice())                      # 2
        assert _does_not_crash(
            lambda: f1.slice(xslices=0.5, yslices=-0.3, zslices=0.9))   # 3
        assert _does_not_crash(lambda: f1.isosurface())                 # 4
        assert _does_not_crash(lambda: f1.isosurface([0.5, -0.6]))      # 5
        assert _does_not_crash(lambda: f1.scan())                       # 6a
        assert _does_not_crash(lambda: f1.scan(1))                      # 6b
        assert _does_not_crash(lambda: f1.scan(1, hold=True))           # 7
        assert _does_not_crash(lambda: f1.scan(3))                      # 8
        assert _does_not_crash(lambda: f1.isosurface())                 # 9
        assert _does_not_crash(lambda: f1.slice())                      # 10
        assert _does_not_crash(lambda: f1.surf())                       # 11

        assert _does_not_crash(lambda: f2.plot())                       # 12
        assert _does_not_crash(lambda: f2.slice())                      # 13
        assert _does_not_crash(
            lambda: f2.slice(xslices=0.5, yslices=-0.3, zslices=0.9))   # 14
        assert _does_not_crash(lambda: f2.isosurface())                 # 15
        assert _does_not_crash(lambda: f2.isosurface([0.5, -0.6]))      # 16
        assert _does_not_crash(lambda: f2.scan())                       # 17
        assert _does_not_crash(lambda: f2.scan(hold=True))              # 18
        assert _does_not_crash(lambda: f2.scan(2, hold=True))           # 19
