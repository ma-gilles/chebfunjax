"""Port of MATLAB Chebfun tests/singfun/test_make.m (Opus 4.8).

MATLAB's ``f.make(...)`` is a factory that rebuilds a singfun from the same
arguments; the test checks ``isequal(f, f.make(...))``.  chebfunjax has no
``make`` method (and no exponent auto-detection for the singType-only cases),
so every assertion is xfailed.

Provenance
----------
MATLAB source : tests/singfun/test_make.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.fun.singfun import Singfun

_REASON = "chebfunjax Singfun has no make() factory method"


def _sf(f, exps):
    return Singfun.from_function(f, exps)


class TestSingfunMake:
    def test_handle_only(self):
        # MATLAB auto-detects exponents here (a, b in (0,1)); chebfunjax needs them.
        a, b = 0.3, 0.4
        fh = lambda x: jnp.sin(x) / ((1 + x) ** a * (1 - x) ** b)
        f = _sf(fh, (-a, -b))
        assert f.make(fh) == f

    def test_handle_and_exponents(self):
        a, b = 0.3, 0.4
        fh = lambda x: jnp.sin(x) * (1 + x) ** a * (1 - x) ** b
        f = _sf(fh, (a, b))
        assert f.make(fh, (a, b)) == f

    def test_handle_and_singtype(self):
        # MATLAB uses singType 'pole' with auto-detected integer exponents.
        a, b = 3, 4
        fh = lambda x: jnp.exp(x) / ((1 + x) ** a * (1 - x) ** b)
        f = _sf(fh, (-a, -b))
        assert f.make(fh, (-a, -b)) == f

    def test_handle_exponents_pref(self):
        a, b = 0.3, 0.4
        fh = lambda x: jnp.exp(jnp.sin(x)) / ((1 + x) ** a * (1 - x) ** b)
        f = _sf(fh, (-a, -b))
        assert f.make(fh, (-a, -b)) == f

    def test_handle_singtype_pref(self):
        a, b = 0.3, 0.4
        fh = lambda x: jnp.sin(jnp.exp(jnp.cos(x))) * (1 + x) ** a * (1 - x) ** b
        f = _sf(fh, (a, b))
        assert f.make(fh, (a, b)) == f

    def test_all_arguments(self):
        a, b = 2, 3
        fh = lambda x: jnp.exp(jnp.sin(x ** 2)) / ((1 + x) ** a * (1 - x) ** b)
        f = _sf(fh, (-a, -b))
        assert f.make(fh, (-a, -b)) == f
