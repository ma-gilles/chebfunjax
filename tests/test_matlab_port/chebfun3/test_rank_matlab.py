"""All six native test_rank.m predicates, Chebfun7574c77.

Each native function runs in a fresh process. Maximums adapt MATLAB's
single-output rank and length to the existing tuple APIs. The existing
qualified JAX Airy primitive replaces MATLAB's builtin Ai, with source
real projections and arguments unchanged.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.utils.airy_general import airy_all


@pytest.mark.parametrize('C,rank_bound,length_bound', [(1, 30, 100), (10, 60, 300)], ids=['C1', 'C10'])
def test_native_rational_rank_length(C, rank_bound, length_bound):
    f = chebfun3(lambda x, y, z: 1/(1+C*(x**2-y**2+z**2)**2))
    r = max(f.rank)
    m = max(f.length())
    print({'rank': r, 'length': m}, flush=True)
    assert r <= rank_bound
    assert m <= length_bound


def test_native_airy_rank_length():
    f = chebfun3(lambda x, y, z: jnp.real(airy_all(5*(x+y**2+z**2))[0])
                 * jnp.real(airy_all(-5*(x**2+y**2+z**2))[0]))
    r = max(f.rank)
    m = max(f.length())
    print({'rank': r, 'length': m}, flush=True)
    assert r <= 60
    assert m <= 200
