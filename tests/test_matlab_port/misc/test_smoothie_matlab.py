"""Port of MATLAB Chebfun tests/misc/test_smoothie.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_smoothie.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

from chebfunjax.chebfun1d.randfuns import smoothie

jax.config.update("jax_enable_x64", True)


def _std(f):
    return float(abs(f.std()))


class TestMiscSmoothie:
    def test_all_matlab_assertions(self):
        s = _std(smoothie())
        assert 0.2 < s < 5                                          # pass(1)
        s = _std(smoothie("trig"))
        assert 0.2 < s < 5                                          # pass(2)
        s = _std(smoothie([-.1, .1]))
        assert 0.1 < s < 5                                          # pass(3)
        s = _std(smoothie([-3, 3]))
        assert 0.2 < s < 5                                          # pass(4)
        s = _std(smoothie("complex"))
        assert 0.2 < s < 5                                          # pass(5)
        s = _std(smoothie("trig", "complex", [-7, -6]))
        assert 0.2 < s < 5                                          # pass(6)
        f = smoothie(3)
        assert f.n_columns == 3                                     # pass(7)
