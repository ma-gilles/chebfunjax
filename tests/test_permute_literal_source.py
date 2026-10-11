"""All four literal source PERMUTE predicates; default continuous norm.

Source pin7574c77 tests/chebfun/test_permute.m. No RNG or tolerance changes.
"""

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref


def test_original_four_predicates():
    f = cj.chebfun(jnp.sin, pref=ChebfunPref())
    assert float((f - f.permute([1, 2])).norm()) == 0
    assert float((f.T - f.permute([2, 1])).norm()) == 0
    for order in [[1, 1], [1, 3]]:
        with pytest.raises(Exception):
            f.permute(order)
