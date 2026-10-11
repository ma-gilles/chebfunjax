"""Legacy alternative-backend seed must not inherit its own session selector.

Provenance
----------
MATLAB source: @chebop/determineDiscretization.m.
Chebfun commit: 7574c77
This is a Python adapter recursion regression, not native Newton coverage.
"""
from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators import chebop_altdisc
from chebfunjax.operators.chebop import Chebop


def test_legacy_alternative_seed_requests_values(monkeypatch):
    previous = ChebopPref._defaults
    ChebopPref.setDefaults('discretization', 'coeffs')
    try:
        op = Chebop(lambda x, u, v: [u**2-1, v**2-1])
        received = []
        seed = [chebfun(1.), chebfun(1.)]
        monkeypatch.setattr(chebop_altdisc, '_linear_system_by_ad', lambda *a: False)
        def capture(self, f, **kwargs):
            received.append(kwargs)
            return seed
        monkeypatch.setattr(Chebop, 'solve', capture)
        result = chebop_altdisc.solve_bvp_altdisc(op, discretization='ultraS', max_iter=0)
        assert received == [{'discretization': 'values'}]
        assert result[0] is seed[0]
        assert result[1] is seed[1]
    finally:
        ChebopPref._defaults = previous
