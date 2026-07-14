"""Port of MATLAB Chebfun tests/chebtech/test_happinessCheck.m (Opus 4.8).

MATLAB ``happinessCheck(g, f, values, [], pref)`` returns ``[ishappy, tail]``.
chebfunjax exposes the static ``Tech.happiness_check(coeffs, values, op=, tol=,
vscale=)`` which returns ``(ishappy, cutoff)`` (cutoff == MATLAB's tail).  The
default check corresponds to MATLAB's standard check.  Passing ``op`` enables
the sample test (MATLAB ``pref.sampleTest = 1``); omitting ``op`` disables it
(``pref.sampleTest = 0``).

Ported (passing): the scalar sin@33 case (tail==14), the array-valued
[sin cos exp]@33 case (tail near 15, happy), and the aliasing case
(cos(80*acos x)@33 => tail 15/17 without sampleTest, unhappy/33 with).
Array-valued happiness_check on (n, m) coeffs takes the max cutoff across the
per-column standard_chop cutoffs (FIXED, Fable 5, Big-Three array-valued epic).
xfail/skip: the ``happinessCheck='strict'/'classic'`` pref variants (no such
pref in chebfunjax), which pull in the strict/plateau array cases (pass 8, 9).

Provenance
----------
MATLAB source : tests/chebtech/test_happinessCheck.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.quadrature import chebpts

CASES = [(Chebtech1, 1), (Chebtech2, 2)]
# pass(n, 5): expected tail differs between the two techs (15 vs 17).
CASES5 = [(Chebtech1, 1, 15), (Chebtech2, 2, 17)]

_NO_PREF = (
    "chebfunjax happiness_check has no 'strict'/'classic' happinessCheck pref "
    "variants (only the standard check)"
)


class TestChebtechHappinessCheck:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_scalar_tail_14(self, Tech, kind):
        # pass(n, 1): sin at 33 points => tail == 14.
        x = chebpts(33, kind)
        g = Tech.from_values(jnp.sin(x))
        values = Tech.coeffs2vals(g.coeffs)
        ishappy, tail = Tech.happiness_check(g.coeffs, values, op=jnp.sin)
        assert tail == 14

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_scalar_ishappy(self, Tech, kind):
        # pass(n, 2): sin at 33 points => ishappy.
        x = chebpts(33, kind)
        g = Tech.from_values(jnp.sin(x))
        values = Tech.coeffs2vals(g.coeffs)
        ishappy, tail = Tech.happiness_check(g.coeffs, values, op=jnp.sin)
        assert ishappy

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_tail(self, Tech, kind):
        # pass(n, 3): array-valued [sin cos exp] at 33 pts => abs(tail-15) < 2.
        # FIXED (Fable 5, Big-Three array-valued epic): happiness_check on (n, m)
        # coeffs returns the max cutoff across per-column standard_chop cutoffs.
        x = chebpts(33, kind)

        def op(xx):
            return jnp.stack([jnp.sin(xx), jnp.cos(xx), jnp.exp(xx)], axis=-1)

        g = Tech.from_values(op(x))
        values = Tech.coeffs2vals(g.coeffs)
        ishappy, tail = Tech.happiness_check(g.coeffs, values, op=op)
        assert abs(tail - 15) < 2

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_ishappy(self, Tech, kind):
        # pass(n, 4): array-valued [sin cos exp] is happy (every column resolved).
        # FIXED (Fable 5, Big-Three array-valued epic).
        x = chebpts(33, kind)

        def op(xx):
            return jnp.stack([jnp.sin(xx), jnp.cos(xx), jnp.exp(xx)], axis=-1)

        g = Tech.from_values(op(x))
        values = Tech.coeffs2vals(g.coeffs)
        ishappy, tail = Tech.happiness_check(g.coeffs, values, op=op)
        assert ishappy

    @pytest.mark.parametrize("Tech,kind,tail_ex", CASES5)
    def test_aliasing_fools_check(self, Tech, kind, tail_ex):
        # pass(n, 5): sampleTest off; cos((2k+m)*acos x) at k+1=33 pts aliases,
        # so the check is (wrongly) happy: ishappy && tail == 15 (cb1) / 17 (cb2).
        k = 32
        m = k // 2

        def f(xx):
            return jnp.cos((2 * k + m) * jnp.arccos(xx))

        x = chebpts(k + 1, kind)
        g = Tech.from_values(f(x))
        values = Tech.coeffs2vals(g.coeffs)
        ishappy, tail = Tech.happiness_check(g.coeffs, values)  # no op => no sampleTest
        assert ishappy and tail == tail_ex

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_sampletest_fixes_aliasing(self, Tech, kind):
        # pass(n, 6): sampleTest on => unhappy && tail == 33.
        k = 32
        m = k // 2

        def f(xx):
            return jnp.cos((2 * k + m) * jnp.arccos(xx))

        x = chebpts(k + 1, kind)
        g = Tech.from_values(f(x))
        values = Tech.coeffs2vals(g.coeffs)
        ishappy, tail = Tech.happiness_check(g.coeffs, values, op=f)  # op => sampleTest
        assert (not ishappy) and tail == 33

    @pytest.mark.xfail(reason=_NO_PREF, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_strict_vs_classic(self, Tech, kind):
        # pass(n, 7): strictCheck vs classicCheck pref variants.
        raise NotImplementedError(_NO_PREF)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_strict_array(self, Tech, kind):
        # pass(n, 8): strictCheck with an array-valued input.
        # Array-valuedness is now supported, but the blocker here is the missing
        # happinessCheck='strict' pref (no strictCheck in chebfunjax).
        pytest.skip(_NO_PREF)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_plateau_array(self, Tech, kind):
        # pass(n, 9): plateauCheck with an array-valued input.
        # Array-valuedness is now supported, but the blocker here is the missing
        # happinessCheck=@plateauCheck pref (no plateauCheck in chebfunjax).
        pytest.skip(_NO_PREF)
