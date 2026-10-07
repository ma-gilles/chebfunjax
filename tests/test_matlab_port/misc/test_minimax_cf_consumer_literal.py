"""Exact second clause of the pinned source minimax test.

Provenance
----------
MATLAB source : tests/misc/test_minimax.m, lines 18-22
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Python minimax returns coefficients in a result object; wrap them in the
existing coefficient-input Chebfun API. The source default norm is continuous L2.
"""
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.cfpade import cf
from chebfunjax.utils.minimax import minimax


def test_source_minimax_clause2_cf_consumer():
    x = chebfun(lambda t: t, domain=(-1.0, 1.0))
    f = (x.exp().sin()).exp()
    pcf = cf(f, 7)[0]
    result = minimax(f, 7, domain=(-1.0, 1.0))
    pbest = chebfun(jnp.asarray(result.coeffs), domain=result.domain, coeffs=True)
    assert float((pcf - pbest).norm(2)) < 0.0003
