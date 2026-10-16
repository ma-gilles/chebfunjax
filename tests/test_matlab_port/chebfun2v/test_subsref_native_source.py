"""All19 original subsref predicates, Chebfun commit7574c77.

Python zero-based indexing adapts MATLAB component indexing. Public function
calls exercise composition dispatch; continuous norms preserve source bounds.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
from chebfunjax.diskfun.diskfun import diskfun
from chebfunjax.diskfun.diskfunv import Diskfunv
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

TOL = 1000 * jnp.finfo(jnp.float64).eps


def field2(*functions, domain=None):
    return Chebfun2v.from_functions(*functions, domain=domain)


def field3(*functions, domain=(-1., 1., -1., 1., -1., 1.)):
    return Chebfun3v([Chebfun3.from_function(f, domain=domain) for f in functions])


@pytest.mark.parametrize("clause", range(1, 20))
def test_original_subsref_predicate(clause):
    if clause == 1:
        f = Chebfun2.from_function(lambda x, y: jnp.sin(x*y))
        F = f.vertcat(f).vertcat(f)
        assert jnp.linalg.norm(F[0].approx.pivot_values - f.approx.pivot_values) < TOL
    elif clause in (2, 3, 4):
        f1 = Chebfun2.from_function(lambda x, y: x+1)
        f2 = Chebfun2.from_function(lambda x, y: 2*y)
        functions = [lambda x, y: x+y, lambda x, y: x-y]
        exact = [lambda x, y: x+2*y+1, lambda x, y: x-2*y+1]
        if clause == 4:
            functions.append(lambda x, y: x*y)
            exact.append(lambda x, y: 2*(x+1)*y)
        H = field2(*functions, domain=(0., 2., -2., 2.))(f1, f2)
        if clause == 3:
            assert jnp.linalg.norm(H(1., 1.)-jnp.array([4., 0.])) < TOL
        else:
            assert (H-field2(*exact)).norm() < TOL
    elif clause == 5:
        f1 = Chebfun3.from_function(lambda x, y, z: x)
        f2 = Chebfun3.from_function(lambda x, y, z: y+z)
        G = field2(lambda x, y: x, lambda x, y: y, lambda x, y: x+y,
                   domain=(-1., 1., -2., 2.))
        exact = field3(lambda x, y, z: x, lambda x, y, z: y+z,
                       lambda x, y, z: x+y+z)
        assert (G(f1, f2)-exact).norm() < TOL
    elif clause in (6, 7):
        F = field2(lambda x, y: x, lambda x, y: y)
        functions = [lambda x, y: x+y, lambda x, y: x-y]
        if clause == 7:
            functions.append(lambda x, y: x*y)
        G = field2(*functions)
        assert (G(F)-G).norm() < TOL
    elif clause in (8, 9):
        F = field3(lambda x, y, z: x, lambda x, y, z: y+z)
        functions = [lambda x, y: x+y, lambda x, y: x-y]
        exact = [lambda x, y, z: x+y+z, lambda x, y, z: x-y-z]
        if clause == 9:
            functions.append(lambda x, y: x*y)
            exact.append(lambda x, y, z: x*(y+z))
        G = field2(*functions, domain=(-1., 1., -2., 2.))
        assert (G(F)-field3(*exact)).norm() < TOL
    elif clause in (10, 11):
        f = Chebfun2.from_function((lambda x, y: x+1j*y) if clause == 10
                                  else (lambda x, y: x))
        G = field2(lambda x, y: x, lambda x, y: y, lambda x, y: x+y)
        exact = G if clause == 10 else field2(lambda x, y: x,
                    lambda x, y: 0*x, lambda x, y: x)
        assert (G(f)-exact).norm() < TOL
    elif clause in (12, 13):
        f = Chebfun3.from_function((lambda x, y, z: x+y+1j*z) if clause == 12
                                   else (lambda x, y, z: x+y))
        d = (-2., 2., -1., 1.) if clause == 12 else (-2., 2., -2., 2.)
        G = field2(lambda x, y: x, lambda x, y: y, lambda x, y: x+y, domain=d)
        exact = field3(lambda x, y, z: x+y,
                       (lambda x, y, z: z) if clause == 12 else (lambda x, y, z: 0*y),
                       (lambda x, y, z: x+y+z) if clause == 12 else (lambda x, y, z: x+y))
        assert (G(f)-exact).norm() < TOL
    elif clause in (14, 15, 16):
        G = field2(lambda x, y: x+y, lambda x, y: x)
        exact = chebfun(lambda t: jnp.stack([2*t, t], axis=-1))
        if clause == 14:
            H = G(chebfun(lambda t: jnp.stack([t, t], axis=-1)))
        elif clause == 15:
            f = chebfun(lambda t: t)
            H = G(f, f)
        else:
            H = G(chebfun(lambda t: t+1j*t))
        assert (H-exact).norm() < TOL
    elif clause == 17:
        f = Spherefun.from_function(lambda lam, th: jnp.cos(lam)*jnp.sin(th))
        G = field2(lambda x, y: x+y, lambda x, y: 2*x, lambda x, y: y*y)
        zero = Spherefun.from_function(lambda lam, th: 0*lam)
        assert (G(f)-Spherefunv(f, 2*f, zero)).norm() < TOL
    elif clause == 18:
        f = diskfun(lambda x, y: x+y)
        G = field2(lambda x, y: x, lambda x, y: y, domain=(-2., 2., -2., 2.))
        assert (G(f)-Diskfunv(f, diskfun(lambda x, y: 0*x))).norm() < TOL
    else:
        F = Diskfunv(diskfun(lambda x, y: x), diskfun(lambda x, y: y))
        G = field2(lambda x, y: x+y, lambda x, y: x-y)
        exact = Diskfunv(diskfun(lambda x, y: x+y), diskfun(lambda x, y: x-y))
        assert (G(F)-exact).norm() < TOL
