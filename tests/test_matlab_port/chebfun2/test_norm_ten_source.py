"""All ten executed assignments in native tests/chebfun2/test_norm.m7574c77.

Native pass7/8 are overwritten later. Keep both earlier Frobenius assignments
and later operator/nuclear assignments; exact functions/references/bounds.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebpref import ChebfunPref


@pytest.mark.parametrize('slot', range(1, 11))
def test_native_norm_assignment(slot):
    tol = 1000*ChebfunPref().cheb2Prefs.chebfun2eps
    if slot <= 3:
        f = Chebfun2.from_function(lambda x, y: x)
    elif slot <= 6:
        f = Chebfun2.from_function(lambda x, y: 1j*x)
    else:
        f = Chebfun2.from_function(lambda x, y: jnp.exp(x*y))
    if slot in (1, 4):
        value, exact = f.norm(), jnp.sqrt(4/3)
    elif slot in (2, 5):
        value, exact = f.norm('inf'), 1.
    elif slot in (3, 6):
        value, exact = f.norm(4), (4/5)**(1/4)
    elif slot == 7:
        value, exact = f.norm(), 2.236768845167052
    elif slot == 8:
        value, exact = f.norm('fro'), 2.236768845167052
    elif slot == 9:
        value, exact = f.norm(2), 2.119814813637055
    else:
        value, exact = f.norm('nuc'), 2.925303491814361
    assert jnp.abs(value-exact) < tol
