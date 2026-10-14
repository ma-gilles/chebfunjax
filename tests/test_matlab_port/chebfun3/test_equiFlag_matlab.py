"""Literal port of MATLAB tests/chebfun3/test_equiFlag.m predicates 1–4.

The tensor samples use the original 100^3 linspace/ndgrid construction and
retain the native `norm(f-g) < 100*cheb3eps` bound, including its small-domain
case. This file is intentionally separate from the compact captured output
controls in `test_equi_native_capture.py`.

Provenance
----------
MATLAB source: tests/chebfun3/test_equiFlag.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

_SOURCE_BOUND = 100.0 * float(ChebfunPref().cheb3Prefs.chebfun3eps)


def _source_predicate(domain, operator):
    axes = [jnp.linspace(domain[2 * k], domain[2 * k + 1], 100)
            for k in range(3)]
    xx, yy, zz = jnp.meshgrid(*axes, indexing="ij")
    samples = operator(xx, yy, zz)
    reference = chebfun3(operator, domain=domain)
    actual = chebfun3(samples, domain=domain, equi=True)
    assert float((reference - actual).norm()) < _SOURCE_BOUND


def test_native_equi_predicate_1_default_domain():
    _source_predicate((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
                      lambda x, y, z: jnp.cos(x + y + z))


def test_native_equi_predicate_2_different_domain():
    _source_predicate((-1.0, 2.0, -2.0, 1.0, -3.0, 0.0),
                      lambda x, y, z: jnp.cos(x + y + z))


def test_native_equi_predicate_3_nonsymmetric_operator():
    _source_predicate((-1.0, 2.0, -2.0, 1.0, -3.0, 0.0),
                      lambda x, y, z: jnp.cos(x + 2.0 * y + 3.0 * z))


def test_native_equi_predicate_4_small_domain():
    h = 1.0e-2
    domain = (1.0 - h, 1.0 + h,
              1.0 - 2.0 * h, 1.0 + 2.0 * h,
              1.0 - 3.0 * h, 1.0 + 3.0 * h)
    _source_predicate(domain,
                      lambda x, y, z: jnp.cos(x + 2.0 * y + 3.0 * z))
