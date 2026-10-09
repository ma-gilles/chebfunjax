"""Literal four-clause Chebfun exact-initial-guess source test.

The previous sampled/fixed-grid controls remain separately in
``test_exactInitial_legacy_controls.py`` and do not qualify source predicates.

Provenance
----------
MATLAB source: tests/chebop/test_exactInitial.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
import math

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop


def test_original_exact_initial_1_4(record_property):
    dom = (0., 10.)
    op = Chebop(lambda x, u: u.diff(2) + u.sin(), dom)
    op.bc = lambda x, u: [u(0.)-2, u(10.)-2]
    x = chebfun(lambda x: x, domain=dom)
    op.init = 2*(2*math.pi*x/10).cos()
    u, _ = op.solvebvp(0., discretization='chebcolloc2')
    errors = [float(op(u).norm())]
    op.init = u
    # Source keeps this initial guess for all three subsequent solves.
    for backend in ['chebcolloc2', 'chebcolloc1', 'ultraS']:
        u, _ = op.solvebvp(0., discretization=backend)
        errors.append(float(op(u).norm()))
    for slot, error in enumerate(errors, 1):
        record_property(f'source_exactInitial_{slot}', error)
    assert all(error < 1e-7 for error in errors), errors
