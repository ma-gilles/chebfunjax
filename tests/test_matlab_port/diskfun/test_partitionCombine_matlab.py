"""Port of MATLAB Chebfun tests/diskfun/test_partitionCombine.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_partitionCombine.m
Chebfun commit: 7574c77

Cartesian ``@(x,y)`` handles are written in polar coordinates
``(theta, r)`` with ``x = r cos(theta)``, ``y = r sin(theta)``.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from chebfunjax.diskfun.diskfun import Diskfun

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL = 1e3 * _EPS


def _df(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Diskfun.from_function(fn)


def _fe(t, r):
    # sin(pi*x*y): strictly even / pi-periodic in the theta-shift.
    return jnp.sin(jnp.pi * (r * jnp.cos(t)) * (r * jnp.sin(t)))


def _fo(t, r):
    # sin(pi*x): strictly odd / anti-periodic in the theta-shift.
    return jnp.sin(jnp.pi * r * jnp.cos(t))


class TestDiskfunPartitioncombine:
    def test_partition_empty(self):
        # pass(1): partition of empty gives two empty diskfuns.
        feven, fodd = Diskfun.empty().partition()
        assert feven.isempty() and fodd.isempty()

    def test_combine_two_empty(self):
        # pass(2): combine of two empties is empty.
        f = Diskfun.combine(Diskfun.empty(), Diskfun.empty())
        assert f.isempty()

    def test_partition_even(self):
        # pass(3)/(4): even diskfun partitions to (f, empty).
        f = _df(_fe)
        feven, fodd = f.partition()
        assert float((feven - f).norm()) < _TOL
        assert fodd.isempty()

    def test_partition_odd(self):
        # pass(5)/(6): odd diskfun partitions to (empty, f).
        f = _df(_fo)
        feven, fodd = f.partition()
        assert float((fodd - f).norm()) < _TOL
        assert feven.isempty()

    def test_partition_mixed(self):
        # pass(7)/(8): mixed function splits into its even and odd parts.
        f = _df(lambda t, r: _fe(t, r) + _fo(t, r))
        feven, fodd = f.partition()
        assert float((_df(_fe) - feven).norm()) < _TOL
        assert float((_df(_fo) - fodd).norm()) < _TOL

    def test_combine_even_empty(self):
        # pass(9): combine(even, empty) recovers the even diskfun.
        feven = _df(_fe)
        f = Diskfun.combine(feven, Diskfun.empty())
        assert float((feven - f).norm()) < _TOL

    def test_combine_empty_odd(self):
        # pass(10): combine(empty, odd) recovers the odd diskfun.
        fodd = _df(_fo)
        f = Diskfun.combine(Diskfun.empty(), fodd)
        assert float((fodd - f).norm()) < _TOL

    def test_combine_even_odd(self):
        # pass(11): combine(even, odd) recovers even + odd.
        fcombine = _df(lambda t, r: _fe(t, r) + _fo(t, r))
        f = Diskfun.combine(_df(_fe), _df(_fo))
        assert float((fcombine - f).norm()) < _TOL

    def test_combine_mixed_parity_errors(self):
        # pass(12): combining diskfuns of mixed parity errors.
        f = _df(lambda t, r: _fe(t, r) + _fo(t, r))
        with pytest.raises(ValueError, match="parity"):
            Diskfun.combine(f, f)
