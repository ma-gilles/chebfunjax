"""Port of MATLAB Chebfun tests/chebpref/test_chebfunpref.m (Fable 5).

MATLAB structs map to dicts; subsref/subsasgn passthrough maps to
attribute access.  The tech-object techPref completeness passes
(16-17) are MATLAB class mechanics with no Python counterpart; every
other pass is asserted.

Provenance
----------
MATLAB source : tests/chebpref/test_chebfunpref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebpref import ChebfunPref


class TestChebprefChebfunpref:
    def test_construction_and_merging(self):
        p = ChebfunPref()
        assert ChebfunPref(p) == p  # pass(1)

        p = ChebfunPref({"splitting": True, "testPref": "test"})
        assert p.splitting and p.techPrefs.testPref == "test"  # 2

        p = ChebfunPref({"testPref1": "test1",
                         "techPrefs": {"testPref2": "test2"}})
        assert p.techPrefs.testPref1 == "test1"  # pass(3)
        assert p.techPrefs.testPref2 == "test2"

        q = {"techPrefs": {"testPref": "test",
                           "subPrefs": {"testSubPref": "subTest"}}}
        p = ChebfunPref(q)
        assert p.techPrefs.testPref == "test"
        assert p.testPref == "test"  # pass(4) passthrough read
        assert p.techPrefs.subPrefs.testSubPref == "subTest"
        assert p.subPrefs.testSubPref == "subTest"  # pass(5)

        p = ChebfunPref()
        p.maxLength = 1337
        assert p.maxLength == 1337  # pass(6)
        p.techPrefs.testPref1 = "test1"
        assert p.techPrefs.testPref1 == "test1"  # pass(7)
        p.testPref2 = "test2"
        assert p.techPrefs.testPref2 == "test2"  # pass(8)

    def test_merge_tech_prefs(self):
        p = ChebfunPref()
        p.techPrefs.testPref = "test"
        q = {"testPref": "testq"}
        # pass(13)/(14): ChebfunPref inputs behave as their techPrefs.
        assert ChebfunPref.mergeTechPrefs(p, q) == \
            ChebfunPref.mergeTechPrefs(p.techPrefs, q)
        assert ChebfunPref.mergeTechPrefs(q, p) == \
            ChebfunPref.mergeTechPrefs(q, p.techPrefs)
        q2 = ChebfunPref()
        q2.techPrefs.testPref = "testq"
        assert ChebfunPref.mergeTechPrefs(p, q2) == \
            ChebfunPref.mergeTechPrefs(p.techPrefs,
                                       q2.techPrefs)  # pass(15)
        assert ChebfunPref.mergeTechPrefs(p, q2).testPref == "testq"

    def test_defaults_management(self):
        saved = ChebfunPref._defaults
        try:
            ChebfunPref.setDefaults("factory")
            assert ChebfunPref() == \
                ChebfunPref.getFactoryDefaults()  # pass(18)

            p = ChebfunPref()
            p.domain = (-2.0, 7.0)
            p.testPref = "testq"
            ChebfunPref.setDefaults(p)
            assert ChebfunPref().testPref == "testq"  # pass(19)
            assert ChebfunPref().domain == (-2.0, 7.0)

            ChebfunPref.setDefaults("factory")
            ChebfunPref.setDefaults(
                {"domain": (-2.0, 7.0), "testPref": "testq"})
            assert ChebfunPref().testPref == "testq"  # pass(20)
            assert ChebfunPref().domain == (-2.0, 7.0)

            ChebfunPref.setDefaults("factory")
            ChebfunPref.setDefaults("domain", (-2.0, 7.0),
                                    "testPref", "testq")
            assert ChebfunPref().testPref == "testq"  # pass(21)
            assert ChebfunPref().domain == (-2.0, 7.0)

            # pass(22)-(24): default value types.
            assert isinstance(ChebfunPref().chebfuneps, float)
            assert isinstance(
                ChebfunPref().blowupPrefs.defaultSingType, str)
            assert isinstance(ChebfunPref().refinementFunction, str)

            # pass(25): per-preference factory reset.
            factory = ChebfunPref.getFactoryDefaults()
            ChebfunPref.setDefaults("factory")
            ChebfunPref.setDefaults("domain", (-2.0, 7.0))
            res1 = ChebfunPref().domain
            ChebfunPref.setDefaults("domain", "factory")
            res2 = ChebfunPref().domain
            assert res1 == (-2.0, 7.0) and res2 == factory.domain
        finally:
            ChebfunPref._defaults = saved
