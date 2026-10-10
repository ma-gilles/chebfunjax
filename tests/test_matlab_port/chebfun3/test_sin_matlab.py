"""All nine assertions from MATLAB Chebfun tests/chebfun3/test_sin.m.

Chebfun commit: 7574c77. Original functions, point evaluations, construction
routes, sequence and thresholds are retained. The first six assertions
share their original f; resource gates may split independent later blocks.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 1e2 * ChebfunPref().cheb3Prefs.chebfun3eps


def test_native_sin_first_six():
    f = chebfun3(lambda x, y, z: jnp.sin(x+y+z))
    assert abs(f(.1, .2, .3)-jnp.sin(.6)) < TOL
    f2 = f.sin()
    assert abs(f2(.2, .2, .2)-jnp.sin(jnp.sin(.6))) < TOL
    f2 = chebfun3(lambda x, y, z: jnp.sin(f(x, y, z)))
    assert abs(f2(.2, .2, .2)-jnp.sin(jnp.sin(.6))) < TOL
    f2 = chebfun3(lambda x, y, z: jnp.sin(f(x, y, z)), fiberDim=1)
    assert abs(f2(.2, .2, .2)-jnp.sin(jnp.sin(.6))) < TOL
    f2 = chebfun3(lambda x, y, z: jnp.sin(f(x, y, z)), fiberDim=2)
    assert abs(f2(.2, .2, .2)-jnp.sin(jnp.sin(.6))) < TOL
    f2 = chebfun3(lambda x, y, z: jnp.sin(f(x, y, z)), fiberDim=3)
    assert abs(f2(.2, .2, .2)-jnp.sin(jnp.sin(.6))) < TOL


def test_native_sin_varying_domain():
    domain = (1., 3., 1., 3., 1., 3.)
    x = chebfun3(lambda x, y, z: x, domain)
    y = chebfun3(lambda x, y, z: y, domain)
    z = chebfun3(lambda x, y, z: z, domain)
    f3 = (2*x+3*y+z).sin()
    assert abs(f3(1., 2., 3.)-jnp.sin(11.)) < TOL


def test_native_sin_trig_evaluation():
    domain = (-math.pi, math.pi, -math.pi, math.pi, -math.pi, math.pi)
    f4 = chebfun3(lambda x, y, z: jnp.sin(2*x+3*y), domain, trig=True)
    assert abs(jnp.sin(f4(1., 1., 1.))-jnp.sin(jnp.sin(5.))) < TOL


def test_native_sin_eps_constructor():
    ep = 1e-8
    tol2 = 1e2*ep
    f5 = chebfun3(lambda x, y, z: jnp.exp(x*y*z), eps=ep)
    assert abs(jnp.sin(f5(.5, .5, .5))-jnp.sin(jnp.exp(.125))) < tol2
