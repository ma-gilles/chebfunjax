"""Port of MATLAB Chebfun tests/bndfun/test_changeMap.m (Opus 4.8).

``changeMap(f, newDomain)`` in MATLAB keeps the underlying onefun (Chebyshev
representation on [-1, 1]) and only swaps the affine map so it targets a new
interval.  chebfunjax does not expose a ``changeMap`` method, but the exact
equivalent is ``Bndfun.from_chebtech(f.onefun, Domain(newDomain))`` -- the
onefun is unchanged, only the domain (hence the map) differs.  This is what
MATLAB @classicfun/changeMap.m does internally.

Self-validating: values at corresponding points must match, since the point
x in dom1 and its image y in dom2 map to the SAME reference point in [-1, 1].

Provenance
----------
MATLAB source : tests/bndfun/test_changeMap.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

EPS = float(np.finfo(np.float64).eps)
DOM1 = (-3.0, 1.0)
DOM2 = (-2.0, 7.0)
A, B = DOM1
C, D = DOM2
# deterministic points in dom1 and their linear images in dom2
XR = np.linspace(A, B, 100)
YR = C * (B - XR) / (B - A) + D * (XR - A) / (B - A)
X = jnp.asarray(XR)
Y = jnp.asarray(YR)


def _change_map(f: Bndfun, new_dom) -> Bndfun:
    """Faithful equivalent of MATLAB bndfun/changeMap: keep onefun, swap map."""
    return Bndfun.from_chebtech(f.onefun, Domain(new_dom))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestBndfunChangeMap:
    def test_change_dom1_to_dom2(self):
        f = Bndfun.from_function(lambda x: 1.0 / (1 + x ** 2), Domain(DOM1))
        g = _change_map(f, DOM2)
        assert _ninf(g(Y) - f(X)) < 10 * f.vscale * EPS

    def test_change_dom2_to_dom1_roundtrip(self):
        f0 = Bndfun.from_function(lambda x: 1.0 / (1 + x ** 2), Domain(DOM1))
        g = _change_map(f0, DOM2)
        f = _change_map(g, DOM1)
        assert _ninf(f(X) - g(Y)) < 10 * f.vscale * EPS

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular (blowup) Bndfun: 1/((x-a)(x-b)) with "
        "pref.blowup cannot be constructed via Bndfun.from_function."
    )
    def test_change_singular_dom1_to_dom2(self):
        def op(x):
            return 1.0 / ((x - A) * (x - B))

        f = Bndfun.from_function(op, Domain(DOM1))
        g = _change_map(f, DOM2)
        assert np.all(
            np.abs(np.asarray(g(Y)) - np.asarray(f(X)))
            < 1e4 * np.abs(op(XR)) * f.vscale * EPS
        )

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular (blowup) Bndfun (see above)."
    )
    def test_change_singular_dom2_to_dom1(self):
        def op(x):
            return 1.0 / ((x - A) * (x - B))

        f0 = Bndfun.from_function(op, Domain(DOM1))
        g = _change_map(f0, DOM2)
        f = _change_map(g, DOM1)
        assert np.all(
            np.abs(np.asarray(f(X)) - np.asarray(g(Y)))
            < 1e4 * np.abs(op(XR)) * f.vscale * EPS
        )
