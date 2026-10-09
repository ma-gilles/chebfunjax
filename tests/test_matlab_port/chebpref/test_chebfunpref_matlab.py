"""All28 literal clauses of tests/chebpref/test_chebfunpref.m, pin7574c77.

One source-sequence body records each slot explicitly. MATLAB structs/cells
map to dicts/lists; native unset fixedLengthNaN maps to existing PythonNone.
Native disables unknown-pref warnings in this test; warning parity stays open.
"""
from chebfunjax.chebpref import ChebfunPref


def test_all28_native_preference_slots(record_property):
    passed = []

    def check(slot, predicate):
        assert predicate, f'native preference slot{slot}'
        passed.append(slot)

    saved = ChebfunPref._defaults
    try:
        ChebfunPref.setDefaults('factory')
        p = ChebfunPref()
        check(1, p == ChebfunPref(p))
        p = ChebfunPref({'splitting': True, 'testPref': 'test'})
        check(2, p.splitting and p.techPrefs.testPref == 'test')
        p = ChebfunPref({'testPref1': 'test1', 'techPrefs': {'testPref2': 'test2'}})
        check(3, p.techPrefs.testPref1 == 'test1' and p.techPrefs.testPref2 == 'test2')
        p = ChebfunPref({'techPrefs': {'testPref': 'test', 'subPrefs': {'testSubPref': 'subTest'}}})
        check(4, p.techPrefs.testPref == 'test' and p.testPref == 'test')
        check(5, p.techPrefs.subPrefs.testSubPref == 'subTest' and p.subPrefs.testSubPref == 'subTest')
        p = ChebfunPref()
        p.maxLength = 1337
        check(6, p.maxLength == 1337)
        p.techPrefs.testPref1 = 'test1'
        check(7, p.techPrefs.testPref1 == 'test1')
        p.testPref2 = 'test2'
        check(8, p.techPrefs.testPref2 == 'test2')
        p = ChebfunPref()
        p.domain = (-2, 7)
        p.chebfuneps = 1e-6
        p.maxLength = 1337
        p.bogusPref = True
        q = ChebfunPref()
        q.splitting = True
        q.chebfuneps = 1e-12
        r = ChebfunPref(p, q)
        check(9, r.domain == q.domain and r.splitting == q.splitting
              and r.chebfuneps == q.chebfuneps and r.maxLength == q.maxLength
              and r.bogusPref == p.bogusPref)
        q = {'splitting': True, 'chebfuneps': 1e-12}
        r = ChebfunPref(p, q)
        check(10, r.domain == p.domain and r.splitting == q['splitting']
              and r.chebfuneps == q['chebfuneps'] and r.maxLength == p.maxLength
              and r.bogusPref == p.bogusPref)
        p = {'testPref': 'test'}
        q = {}
        check(11, ChebfunPref.mergeTechPrefs(p, q) == p)
        q['testPref'] = 'testq'
        check(12, ChebfunPref.mergeTechPrefs(p, q).testPref == 'testq')
        p = ChebfunPref()
        p.techPrefs.testPref = 'test'
        q = {'testPref': 'testq'}
        check(13, ChebfunPref.mergeTechPrefs(p, q) == ChebfunPref.mergeTechPrefs(p.techPrefs, q))
        check(14, ChebfunPref.mergeTechPrefs(q, p) == ChebfunPref.mergeTechPrefs(q, p.techPrefs))
        q = ChebfunPref()
        q.techPrefs.testPref = 'testq'
        check(15, ChebfunPref.mergeTechPrefs(p, q) == ChebfunPref.mergeTechPrefs(p.techPrefs, q.techPrefs))
        p = ChebfunPref()
        common = {'chebfuneps': 2**-52, 'minSamples': 17, 'fixedLength': None,
                  'extrapolate': False, 'sampleTest': True,
                  'refinementFunction': 'nested', 'happinessCheck': 'standard'}
        p.tech = 'chebtech2'
        check(16, dict(p.techPrefs) == {**common, 'maxLength': 65537, 'useTurbo': False})
        p.tech = 'trigtech'
        check(17, dict(p.techPrefs) == {**common, 'maxLength': 65536, 'gridType': 2})
        ChebfunPref.setDefaults('factory')
        factory = ChebfunPref.getFactoryDefaults()
        check(18, ChebfunPref() == factory)
        ChebfunPref.setDefaults('factory')
        p = ChebfunPref()
        p.domain = (-2, 7)
        p.testPref = 'testq'
        ChebfunPref.setDefaults(p)
        check(19, ChebfunPref().testPref == 'testq' and ChebfunPref().domain == (-2, 7))
        ChebfunPref.setDefaults('factory')
        ChebfunPref.setDefaults({'domain': (-2, 7), 'testPref': 'testq'})
        check(20, ChebfunPref().testPref == 'testq' and ChebfunPref().domain == (-2, 7))
        ChebfunPref.setDefaults('factory')
        ChebfunPref.setDefaults('domain', (-2, 7), 'testPref', 'testq')
        check(21, ChebfunPref().testPref == 'testq' and ChebfunPref().domain == (-2, 7))
        check(22, isinstance(ChebfunPref().chebfuneps, (int, float)))
        check(23, isinstance(ChebfunPref().blowupPrefs.defaultSingType, str))
        check(24, isinstance(ChebfunPref().refinementFunction, str))
        ChebfunPref.setDefaults('factory')
        ChebfunPref.setDefaults('domain', (-2, 7))
        res1 = ChebfunPref().domain
        ChebfunPref.setDefaults('domain', 'factory')
        check(25, res1 == (-2, 7) and ChebfunPref().domain == factory.domain)
        ChebfunPref.setDefaults('factory')
        ChebfunPref.setDefaults(['cheb2Prefs', 'maxRank'], 5)
        res1 = ChebfunPref().cheb2Prefs.maxRank
        ChebfunPref.setDefaults(['cheb2Prefs', 'maxRank'], 'factory')
        check(26, res1 == 5 and ChebfunPref().cheb2Prefs.maxRank == factory.cheb2Prefs.maxRank)
        ChebfunPref.setDefaults('factory')
        ChebfunPref.setDefaults('chebfuneps', 1e-6)
        res1 = ChebfunPref().chebfuneps
        ChebfunPref.setDefaults('chebfuneps', 'factory')
        check(27, res1 == 1e-6 and ChebfunPref().chebfuneps == factory.chebfuneps)
        ChebfunPref.setDefaults('factory')
        ChebfunPref.setDefaults('bogusPref', 'abc')
        res1 = ChebfunPref().bogusPref
        ChebfunPref.setDefaults('bogusPref', 'factory')
        check(28, res1 == 'abc' and 'bogusPref' not in ChebfunPref().techPrefs)
        assert passed == list(range(1, 29))
        record_property('native_slots', ','.join(map(str, passed)))
    finally:
        ChebfunPref._defaults = saved
