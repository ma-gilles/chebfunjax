"""Port of MATLAB Chebfun tests/chebfun/test_innerProduct.m (Fable 5).

The singular-exponent reference case is skipped (needs 'exps' in the
chebfun factory); smooth and piecewise inner products ported with
closed-form references.

Provenance
----------
MATLAB source : tests/chebfun/test_innerProduct.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunInnerProduct:
    def test_singular_exponent_reference(self):
        pytest.skip("chebfun factory has no 'exps' (singular exponents) "
                    "argument; singfun-level innerProduct is covered in "
                    "the singfun ports")

    def test_smooth_inner_product(self):
        # <sin, cos> on [0, pi] = 0; <sin, sin> = pi/2
        f = cj.chebfun(jnp.sin, domain=(0.0, float(np.pi)))
        g = cj.chebfun(jnp.cos, domain=(0.0, float(np.pi)))
        assert abs(float(f.innerProduct(g))) < 100 * EPS
        assert abs(float(f.innerProduct(f)) - np.pi / 2) < 100 * EPS

    def test_complex_conjugate_linearity(self):
        # <f, g> = conj(<g, f>)
        f = cj.chebfun(lambda x: jnp.exp(1j * x))
        g = cj.chebfun(lambda x: x + 0j)
        ip1 = complex(f.innerProduct(g))
        ip2 = complex(g.innerProduct(f))
        assert abs(ip1 - np.conj(ip2)) < 100 * EPS
