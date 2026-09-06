"""Port of MATLAB Chebfun tests/spinpref2/test_spinpref2.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinpref2/test_spinpref2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.operators.spinpref import spinpref2


class TestSpinpref2:
    def test_all_matlab_assertions(self):
        pref = spinpref2(Clim=(0, 10), dataplot="abs", dealias="off")
        assert tuple(pref.Clim) == (0, 10)                                   # pass(1)
        assert pref.dataplot.lower() == "abs"                                # pass(2)
        assert pref.dealias.lower() == "off"                                 # pass(3)
        pref = spinpref2(iterplot=10, M=100)
        assert pref.iterplot == 10 and pref.M == 100                         # pass(4)-(5)
        pref = spinpref2(Nplot=2, plot="movie", scheme="lawson4")
        assert pref.Nplot == 2 and pref.plot == "movie" and pref.scheme == "lawson4"  # pass(6)-(8)
        pref = spinpref2(view=(10, 20))
        assert tuple(pref.view) == (10, 20)                                  # pass(9)
