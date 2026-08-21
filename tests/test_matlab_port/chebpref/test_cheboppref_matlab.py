"""Port of MATLAB Chebfun tests/chebpref/test_cheboppref.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebpref/test_cheboppref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebpref import ChebopPref


class TestChebprefCheboppref:
    def test_all_matlab_assertions(self):
        p = ChebopPref()
        assert ChebopPref(p) == p  # pass(1)

        p = ChebopPref({"damping": 0, "plotting": "on"})
        assert (not p.damping) and p.plotting == "on"  # pass(2)

        p = ChebopPref()
        p.plotting = "on"
        assert p.plotting == "on"  # pass(3)

        saved = ChebopPref._defaults
        try:
            ChebopPref.setDefaults("factory")
            assert ChebopPref() == \
                ChebopPref.getFactoryDefaults()  # pass(4)

            p = ChebopPref()
            p.damping = 0
            p.plotting = "on"
            ChebopPref.setDefaults(p)
            assert not ChebopPref().damping  # pass(5)
            assert ChebopPref().plotting == "on"

            ChebopPref.setDefaults("factory")
            ChebopPref.setDefaults({"damping": 0, "plotting": "on"})
            assert not ChebopPref().damping  # pass(6)
            assert ChebopPref().plotting == "on"

            ChebopPref.setDefaults("factory")
            ChebopPref.setDefaults("damping", 0, "plotting", "on")
            assert not ChebopPref().damping  # pass(7)
            assert ChebopPref().plotting == "on"
        finally:
            ChebopPref._defaults = saved
