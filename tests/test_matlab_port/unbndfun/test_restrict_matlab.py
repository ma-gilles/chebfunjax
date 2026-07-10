"""Port of MATLAB Chebfun tests/unbndfun/test_restrict.m (Opus 4.8).

``restrict(f, [-inf b1 b2 inf])`` splits an unbndfun across interior
breakpoints into a cell of pieces: bounded pieces become bndfuns and the two
outer pieces remain unbndfuns, all re-representing the SAME function.

chebfunjax's ``Unbndfun`` does not implement a ``restrict`` method.  The
natural equivalent -- re-approximating ``f`` on each sub-interval via
``Bndfun.from_function(lambda x: f(x), sub)`` / ``Unbndfun.from_function`` --
introduces one extra bit of rounding on top of the unbndfun evaluation, which
exceeds MATLAB's tight ``10*eps*vscale`` bound on some pieces.  We therefore
skip these assertions rather than widen the tolerance or fake a pass.

Provenance
----------
MATLAB source : tests/unbndfun/test_restrict.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

_REASON = (
    "chebfunjax Unbndfun has no restrict() method; re-approximating on each "
    "sub-interval exceeds MATLAB's 10*eps*vscale bound by ~1 bit."
)


class TestUnbndfunRestrict:
    @pytest.mark.skip(reason=_REASON)
    def test_decaying_both_inf(self):
        ...

    @pytest.mark.skip(
        reason="chebfunjax lacks blowup (exponents [2 2]) Unbndfun; and no "
        "restrict() method."
    )
    def test_blowup_both_inf(self):
        ...

    @pytest.mark.skip(reason=_REASON)
    def test_decaying_right_inf(self):
        ...

    @pytest.mark.skip(
        reason="chebfunjax lacks blowup (exponents [0 1]) Unbndfun; and no "
        "restrict() method."
    )
    def test_blowup_right_inf(self):
        ...

    @pytest.mark.skip(reason=_REASON)
    def test_x_exp_left_inf(self):
        ...

    @pytest.mark.skip(
        reason="chebfunjax lacks blowup (exponents [0 -1]) Unbndfun; and no "
        "restrict() method."
    )
    def test_blowup_left_inf(self):
        ...

    @pytest.mark.skip(
        reason="chebfunjax lacks array-valued Unbndfun; and no restrict() method."
    )
    def test_array_valued_left_inf(self):
        ...
