"""All nine native HOSVD predicates with their original norm and tolerance.

Provenance
----------
MATLAB source : tests/chebfun3/test_hosvd.m
Chebfun commit: 7574c77
Python one-dimensional singular-value arrays adapt MATLAB column vectors;
no truncation, conditional omission, or tolerance substitution is used.
"""

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 10*ChebfunPref().cheb3Prefs.chebfun3eps


def test_native_hackbusch_modal_values():
    f = chebfun3(lambda x, y, z: x*z+x**2*y, (0., 1., 0., 1., 0., 1.))
    values, _ = f.hosvd()
    exact1 = jnp.asarray([jnp.sqrt(109/720+jnp.sqrt(46.)/45),
                          jnp.sqrt(109/720-jnp.sqrt(46.)/45)])
    exact2 = jnp.asarray([jnp.sqrt(109/720+jnp.sqrt(2899.)/360),
                          jnp.sqrt(109/720-jnp.sqrt(2899.)/360)])
    for actual, expected in zip(values, (exact1, exact2, exact2)):
        assert actual.shape == expected.shape
        assert jnp.linalg.norm(actual-expected) < TOL


def test_native_two_and_five_outputs():
    f = chebfun3(lambda x, y, z: x*z+x**2*y)
    _, g = f.hosvd()
    assert jnp.linalg.norm(g.core[0, :, :]*g.core[1, :, :], ord=2) < TOL
    assert jnp.linalg.norm(g.core[:, 0, :]*g.core[:, 1, :], ord=2) < TOL
    assert jnp.linalg.norm(g.core[:, :, 0]*g.core[:, :, 1], ord=2) < TOL
    _, core, _, _, _ = f.hosvd(return_factors=True)
    assert jnp.linalg.norm(core[0, :, :]*core[1, :, :], ord=2) < TOL
    assert jnp.linalg.norm(core[:, 0, :]*core[:, 1, :], ord=2) < TOL
    assert jnp.linalg.norm(core[:, :, 0]*core[:, :, 1], ord=2) < TOL
