"""Port of MATLAB Chebfun tests/spinprefsphere/test_spinprefsphere.m
(Fable 5).

Provenance
----------
MATLAB source : tests/spinprefsphere/test_spinprefsphere.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.operators.spinpref import spinprefsphere


class TestSpinprefsphere:
    def test_all_matlab_assertions(self):
        pref = spinprefsphere(Clim=(0, 10), dataplot="abs", dealias="off")
        assert tuple(pref.Clim) == (0, 10)                                   # pass(1)
        assert pref.dataplot.lower() == "abs"                                # pass(2)
        assert pref.dealias.lower() == "off"                                 # pass(3)
        pref = spinprefsphere(iterplot=10)
        assert pref.iterplot == 10                                           # pass(4)
        pref = spinprefsphere(Nplot=2, plot="movie")
        assert pref.Nplot == 2 and pref.plot == "movie"                      # pass(5)-(6)
        pref = spinprefsphere(view=(10, 20))
        assert tuple(pref.view) == (10, 20)                                  # pass(7)
