"""Source plus clause4 uses spectral rank, independently of storage length.

Provenance
----------
MATLAB source : tests/spherefun/test_plus.m (construction,10 additions,pass4)
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
The complete eight-clause source aggregate remains separately preserved.
"""
import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun


def test_source_plus_rank_clause_four():
    def unit_sphere(lam, theta):
        x = jnp.cos(lam) * jnp.sin(theta)
        y = jnp.sin(lam) * jnp.sin(theta)
        z = jnp.cos(theta)
        return x * x + y * y + z * z

    f = Spherefun.from_function(unit_sphere)
    rank = f.numerical_rank()
    g = f
    for _ in range(10):
        g = g + f
    assert g.numerical_rank() - rank == 0
