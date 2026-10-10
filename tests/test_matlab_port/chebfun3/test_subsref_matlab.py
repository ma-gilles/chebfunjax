"""Original eighteen test_subsref.m predicates, Chebfun commit 7574c77.

Native source sections execute in separate bounded processes. Within each
section, functions, domains, constructor order, continuous norms and bounds
are unchanged. Colon syntax uses slice(None); explicit dot records exercise
native get dispatch while raw Python factor storage remains a tech list.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun import spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

COLON = slice(None)
TOL = 10000*ChebfunPref().cheb3Prefs.chebfun3eps


def test_native_1_to_5():
    def ff(x, y, z):
        return jnp.sin(x*(y-.1)*(z+.2))
    dom = (-2., 2., -3., 4., -jnp.pi, jnp.pi)
    f = chebfun3(ff, dom)
    assert abs(f(jnp.pi/4, jnp.pi/6, jnp.pi/3)-ff(jnp.pi/4, jnp.pi/6, jnp.pi/3)) < TOL
    cross1 = Chebfun2.from_function(lambda x, y: jnp.sin(x*(y-.1)*(jnp.pi/6+.2)), domain=dom[:4])
    cross2 = Chebfun2.from_function(lambda x, y: jnp.sin(x*(y-.1)*(jnp.pi/4+.2)), domain=dom[:4])
    assert (f(COLON, COLON, jnp.pi/6)-cross1).norm() < TOL and (f(COLON, COLON, jnp.pi/4)-cross2).norm() < TOL
    cross1 = Chebfun2.from_function(lambda y, z: jnp.sin(jnp.pi/4*(y-.1)*(z+.2)), domain=dom[2:])
    cross2 = Chebfun2.from_function(lambda y, z: jnp.sin(jnp.pi/6*(y-.1)*(z+.2)), domain=dom[2:])
    assert (f(jnp.pi/4, COLON, COLON)-cross1).norm() < TOL
    assert (f(jnp.pi/6, COLON, COLON)-cross2).norm() < TOL
    assert (f(COLON, COLON, COLON)-f).norm() < TOL


def test_native_6_to_11():
    f = chebfun3(lambda x, y, z: x*y*z)
    c1 = chebfun(lambda t: 1+0*t)
    c2 = chebfun(lambda t: -.3+0*t)
    c3 = chebfun(lambda t: .5+0*t)
    assert (f(c1, c2, c3)-1*(-.3)*.5).norm() < TOL
    assert (f.feval(c1, c2, c3)-f(c1, c2, c3)).norm() < TOL
    f = chebfun3(lambda x, y, z: x)
    core, rows, tubes, cols = [f.subsref({'type': '.', 'subs': n}) for n in ('core', 'rows', 'tubes', 'cols')]
    assert ((core*rows*tubes) @ cols-chebfun(lambda x: x)).norm() < TOL
    assert jnp.linalg.norm(f.subsref({'type': '.', 'subs': 'domain'})-jnp.asarray([-1, 1, -1, 1, -1, 1])) < TOL
    f = chebfun3(lambda x, y, z: x+y+z)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN3:subsref:inputs:'):
        f(.5)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN3:subsref:inputs:'):
        f(.5, .5)


def test_native_12_to_17():
    dom = (-1., 1., -1., 1., -2., 2.)
    g = chebfun3(lambda x, y, z: x+y+z, dom)
    field = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y, lambda x, y: x+y)
    h = g(field)
    true = Chebfun2.from_function(lambda x, y: 2*x+2*y)
    assert (h-true).norm() < TOL
    g = chebfun3(lambda x, y, z: x+y+z, dom)
    f1 = Chebfun2.from_function(lambda x, y: x)
    f2 = Chebfun2.from_function(lambda x, y: y)
    f3 = f1+f2
    h = g(f1, f2, f3)
    true = Chebfun2.from_function(lambda x, y: 2*x+2*y)
    assert (h-true).norm() < TOL
    g = chebfun3(lambda x, y, z: x+y+z)
    field = Chebfun3v.from_functions(lambda x, y, z: x, lambda x, y, z: y, lambda x, y, z: z)
    h = g(field)
    assert (h-g).norm() < TOL
    g = chebfun3(lambda x, y, z: x+y+z)
    f1 = chebfun3(lambda x, y, z: x)
    f2 = chebfun3(lambda x, y, z: y)
    f3 = chebfun3(lambda x, y, z: z)
    h = g(f1, f2, f3)
    assert (h-g).norm() < TOL
    field = chebfun(lambda t: jnp.stack([t, t, t], axis=-1))
    g = chebfun3(lambda x, y, z: x+y+z)
    h = g(field)
    true = chebfun(lambda t: 3*t)
    assert (h-true).norm() < TOL
    f = chebfun(lambda t: t)
    g = chebfun3(lambda x, y, z: x+y+z)
    h = g(f, f, f)
    true = chebfun(lambda t: 3*t)
    assert (h-true).norm() < TOL


def test_native_18():
    # Cartesian constructors preserve the native operators; only the vector
    # constructor is spelled as its three scalar component constructions.
    f = Spherefunv(spherefun(lambda x, y, z: x), spherefun(lambda x, y, z: y), spherefun(lambda x, y, z: z))
    g = chebfun3(lambda x, y, z: x**2+y**2+z**2)
    h = g(f)
    true = spherefun(lambda x, y, z: 0*x+1)
    assert (h-true).norm() < TOL
