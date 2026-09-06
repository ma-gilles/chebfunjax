"""Port of MATLAB Chebfun tests/spinpref/test_spinpref.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinpref/test_spinpref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.operators.spinpref import spinpref


class TestSpinpref:
    def test_all_matlab_assertions(self):
        pref = spinpref(Ylim=(0, 1), dataplot="real", dealias="on")
        assert tuple(pref.Ylim) == (0, 1)                                    # pass(1)
        assert pref.dataplot.lower() == "real"                               # pass(2)
        assert pref.dealias.lower() == "on"                                  # pass(3)
        pref = spinpref(iterplot=10, M=100)
        assert pref.iterplot == 10                                           # pass(4)
        assert pref.M == 100                                                 # pass(5)
        pref = spinpref(Nplot=2, plot="movie", scheme="lawson4")
        assert pref.Nplot == 2                                               # pass(6)
        assert pref.plot.lower() == "movie"                                  # pass(7)
        assert pref.scheme.lower() == "lawson4"                              # pass(8)
