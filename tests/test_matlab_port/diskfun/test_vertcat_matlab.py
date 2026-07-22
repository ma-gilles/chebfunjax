"""Port of MATLAB Chebfun tests/diskfun/test_vertcat.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_vertcat.m
Chebfun commit: 7574c77

MATLAB's ``[f; g]`` bracket syntax maps to ``Diskfun.vertcat(f, g)``.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.diskfun.diskfunv import Diskfunv


def _df(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Diskfun.from_function(fn)


class TestDiskfunVertcat:
    def test_vertcat_two_makes_diskfunv(self):
        # pass(1): F = [f; f]; iszero(F(1) - f)
        f = _df(lambda t, r: jnp.cos(r * jnp.cos(t)))
        F = Diskfun.vertcat(f, f)
        assert isinstance(F, Diskfunv)
        assert (F.components[0] - f).iszero() or float((F.components[0] - f).norm()) < 1e-13

    def test_vertcat_three_errors(self):
        # pass(2): [f; f; f] must error (Diskfunv has only two components).
        f = _df(lambda t, r: jnp.cos(r * jnp.cos(t)))
        with pytest.raises((ValueError, TypeError)):
            Diskfun.vertcat(f, f, f)
