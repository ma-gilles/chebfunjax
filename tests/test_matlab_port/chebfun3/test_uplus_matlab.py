"""All three original tests/chebfun3/test_uplus.m predicates.

Provenance
----------
MATLAB source : tests/chebfun3/test_uplus.m
Chebfun commit: 7574c77
Original domains, continuous norms and preference bounds retained.
"""

import math

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref

DOMAINS = [(-1., 1., -1., 1., -1., 1.),
           (-2., 2., -2., 2., -2., 2.),
           (-1., math.pi, 0., 2*math.pi, -math.pi, math.pi)]
TOL = 1e3 * ChebfunPref().cheb3Prefs.chebfun3eps


@pytest.mark.parametrize("domain", DOMAINS)
def test_original_uplus(domain):
    f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x*y*z), domain=domain)
    reference = f  # Literal native test assigns uplusF=f.
    assert (f-reference).norm() < max(map(abs, domain))*TOL
    # Supplemental controls exercise the actual unary API.
    assert f.uplus() is f
    assert +f is f


def test_empty_unary_api():
    empty = Chebfun3.empty()
    assert +empty is empty
    assert empty.uplus() is empty
    assert (-empty).isempty()
    assert empty.uminus().isempty()
