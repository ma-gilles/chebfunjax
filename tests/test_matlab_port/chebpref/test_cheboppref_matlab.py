"""All ten predicates of native tests/chebpref/test_cheboppref.m.

Provenance: Chebfun 7574c77, University of Oxford/Chebfun Developers (2017).
The local comparison translates source isequalNaN over represented preference
state; it is not a general MATLAB isequaln implementation or public == policy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Number

from chebfunjax.chebpref import ChebfunPref, ChebopPref


def _isequal_nan(a, b):
    """Compare complete preference state without coercing array truth values.

    Compare raw top-level/technology state AND resolved technology values.
    The lazy view cache is implementation state, not a preference value.
    Array-valued/custom equality is deliberately unsupported by this adapter.
    """
    if isinstance(a, ChebfunPref) or isinstance(b, ChebfunPref):
        if type(a) is not type(b):
            return False
        return all(_isequal_nan(x, y) for x, y in (
            (a._top, b._top),
            (a._tech_overrides, b._tech_overrides),
            (dict(a.techPrefs), dict(b.techPrefs)),
        ))
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if not isinstance(a, Mapping) or not isinstance(b, Mapping):
            return False
        return a.keys() == b.keys() and all(_isequal_nan(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(_isequal_nan(x, y) for x, y in zip(a, b)))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    equal = a == b
    if type(equal) is not bool:
        raise TypeError("isequalNaN test adapter requires scalar Boolean equality")
    return equal


class TestChebprefCheboppref:
    def test_all_matlab_assertions(self):
        original = ChebopPref._defaults
        try:
            p = ChebopPref()
            assert _isequal_nan(p, ChebopPref(p))  # pass(1)

            p = ChebopPref({"damping": 0, "plotting": "on"})
            assert (not p.damping) and p.plotting == "on"  # pass(2)

            p = ChebopPref()
            p.plotting = "on"
            assert p.plotting == "on"  # pass(3)

            saved_prefs = ChebopPref()
            ChebopPref.setDefaults("factory")
            factory_prefs = ChebopPref.getFactoryDefaults()
            p = ChebopPref()
            assert _isequal_nan(p, factory_prefs)  # pass(4)

            ChebopPref.setDefaults("factory")
            p = ChebopPref()
            p.damping = 0
            p.plotting = "on"
            ChebopPref.setDefaults(p)
            assert (not ChebopPref().damping) and ChebopPref().plotting == "on"  # pass(5)

            ChebopPref.setDefaults("factory")
            ChebopPref.setDefaults({"damping": 0, "plotting": "on"})
            assert (not ChebopPref().damping) and ChebopPref().plotting == "on"  # pass(6)

            ChebopPref.setDefaults("factory")
            ChebopPref.setDefaults("damping", 0, "plotting", "on")
            assert (not ChebopPref().damping) and ChebopPref().plotting == "on"  # pass(7)

            ChebopPref.setDefaults(saved_prefs)
            # MATLAB isnumeric excludes logical; factory value is a double.
            assert isinstance(ChebopPref().bvpTol, Number) and not isinstance(ChebopPref().bvpTol, bool)  # pass(8)

            ChebopPref.setDefaults("factory")
            ChebopPref.setDefaults("discretization", "values")
            pref = ChebopPref()
            assert pref.discretization == "values"  # pass(9)
            pref.discretization = "coeffs"
            assert pref.discretization == "coeffs"  # pass(10)
        finally:
            ChebopPref._defaults = original
