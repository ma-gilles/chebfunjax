"""Port of MATLAB Chebfun tests/treeVar/test_diffArguments.m (Fable 5).

Complicated arguments inside diff() must raise (see chebfun #2191).

Provenance
----------
MATLAB source : tests/treeVar/test_diffArguments.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import pytest

import chebfunjax as cj
from chebfunjax.operators.treevar import TreeVarError, to_first_order

jax.config.update("jax_enable_x64", True)

DOM = (-1.0, 4.0)


class TestTreevarDiffArguments:
    @pytest.mark.parametrize("op", [
        lambda u: (-u).diff() + u,
        lambda u: (2 * u).diff() + u,
        lambda u: (2 * u).diff(2) + u,
        lambda u: (u + 5).diff(2) + u,
    ])
    def test_diff_arguments_raise(self, op):
        with pytest.raises(TreeVarError) as exc:
            to_first_order(op, 0, DOM)
        assert exc.value.identifier == \
            "CHEBFUN:TREEVAR:diff:diffArguments"

    def test_chebfun_times_u(self):
        x = cj.chebfun(lambda t: t, domain=DOM)
        with pytest.raises(TreeVarError) as exc:
            to_first_order(lambda u: (x * u).diff() + u, 0, DOM)
        assert exc.value.identifier == \
            "CHEBFUN:TREEVAR:diff:diffArguments"
