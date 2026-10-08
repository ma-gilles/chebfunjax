# uses-numpy: test-only seeded input generation and MATLAB array-norm predicates
"""All 49 predicates of tests/chebfun3/test_feval.m.

Seed 42 uses NumPy MT19937 uniforms in MATLAB column order. Native input
capture is unavailable, so random-input identity remains unqualified. The
pinned constructor restores RNG state after its internal seeded draws.

Provenance
----------
MATLAB source : tests/chebfun3/test_feval.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun3d.chebfun3 import chebfun3

from ._helpers import EPS

TOL = 1e3 * EPS
DOM = (-1, 2, -np.pi/2, np.pi, -3, 1)
SECTION_DOM = (-1, 1, -4, -2, 6, 8)


def array_norm(values):
    """MATLAB default norm: vector Euclidean or matrix spectral norm."""
    a = np.asarray(values)
    return float(np.linalg.norm(a, ord=2)) if a.ndim else float(abs(a))


@pytest.fixture(scope='module')
def inputs():
    rng = np.random.RandomState(42)
    def rand(shape):
        return jnp.asarray(rng.random_sample(np.prod(shape)).reshape(shape, order='F'))
    return {'pts': 2*rand((3, 1))-1,
            'vectors': tuple(rand((10, 1)) for _ in range(3)),
            'random_vectors': [tuple(rand((100, 1)) for _ in range(3)) for _ in range(2)],
            'random_tensors': [tuple(rand((10, 20, 30)) for _ in range(3)) for _ in range(2)]}


@pytest.mark.parametrize('clause', range(1, 7))
def test_source_coordinates(clause):
    axis = (clause-1)//2
    f = chebfun3(lambda x, y, z: (x, y, z)[axis], domain=DOM)
    point = (0, 0, 0) if clause % 2 else (np.pi/6, np.pi/12, -1)
    bound = TOL if clause in (3, 5) else TOL*f.vscale()
    assert abs(f(*point)-point[axis]) < bound


@pytest.mark.parametrize('clause', range(7, 13))
def test_source_smooth(clause, inputs):
    def ff(x, y, z):
        return jnp.cos(x)+jnp.sin(x*y)+jnp.sin(z*x)
    domain = (-1, 1)*3 if clause < 10 else (-np.pi/6, np.pi/2, -np.pi/12, np.sqrt(3), -3, 1)
    f = chebfun3(ff, domain=domain)
    if clause == 7:
        xyz = tuple(inputs['pts'][:, 0])
    elif clause == 8:
        xyz = inputs['vectors']
    elif clause in (9, 12):
        xyz = jnp.meshgrid(*(v.ravel() for v in inputs['vectors']), indexing='xy')
    else:
        xyz = (0.126986816293506, 0.632359246225410, 0.351283361405006)
    error = ff(*xyz)-f(*xyz)
    value = jnp.max(jnp.abs(error)) if clause in (9, 12) else array_norm(error)
    assert value < TOL*f.vscale()


@pytest.fixture(scope='module')
def sine_functions():
    def ff(x, y, z):
        return jnp.sin(jnp.pi*(x+y+z))
    return ff, chebfun3(ff), chebfun3(ff, trig=True)


@pytest.mark.parametrize('clause', range(13, 30))
def test_source_arrays(clause, sine_functions, inputs):
    ff, polynomial, periodic = sine_functions
    f = periodic if clause in (15, 17, 25, 27, 29) else polynomial
    q = jnp.linspace(-1, 1, 100)
    if clause in (13, 15):
        xyz = (q[:, None],)*3
    elif clause == 14:
        xyz = (q[None, :],)*3
    elif clause in (16, 17):
        xyz = inputs['random_vectors'][clause-16]
    elif 18 <= clause <= 23:
        a, b = jnp.meshgrid(q, q, indexing='xy' if clause < 21 else 'ij')
        c = jnp.full_like(a, -1)
        xyz = [(a, b, c), (c, a, b), (a, c, b)][(clause-18) % 3]
    elif clause in (24, 25, 26, 27):
        xyz = jnp.meshgrid(q, q, q, indexing='xy' if clause < 26 else 'ij')
    else:
        xyz = inputs['random_tensors'][clause-28]
    error = f(*xyz)-ff(*xyz)
    if clause >= 24:
        error = error.ravel()
    assert array_norm(error) < 100*TOL


@pytest.mark.parametrize('clause', range(30, 42))
def test_source_sections(clause):
    axis, kind = divmod(clause-30, 4)
    functions = [lambda x, y, z: jnp.sin(x+y+z)+(z if axis == 2 else x)+y,
                 lambda x, y, z: jnp.cos(y+z)*jnp.sin(z)*jnp.exp(x),
                 lambda x, y, z: jnp.cos(x+z)*jnp.sin(z)*jnp.exp(y),
                 lambda x, y, z: jnp.cos(x+y)*jnp.sin(y)*jnp.exp(z)]
    ff = functions[kind]
    f = chebfun3(ff, domain=SECTION_DOM)
    point = (0.5, -3, 7)[axis]
    args = [slice(None)]*3
    args[axis] = point
    free = [i for i in range(3) if i != axis]
    def exact(a, b):
        xyz = [point]*3
        xyz[free[0]], xyz[free[1]] = a, b
        return ff(*xyz)
    domain = tuple(v for i in free for v in SECTION_DOM[2*i:2*i+2])
    reference = Chebfun2.from_function(exact, domain=domain)
    assert (reference-f(*args)).norm() < 100*TOL


@pytest.mark.parametrize('clause', range(42, 46))
def test_source_lines_identity(clause):
    f = chebfun3(lambda x, y, z: jnp.sin(x+y+z))
    if clause == 45:
        assert (f(slice(None), slice(None), slice(None))-f).norm() < 100*TOL
    else:
        args = [0.5]*3
        args[44-clause] = slice(None)
        reference = chebfun(lambda z: jnp.sin(1+z))
        assert (reference-f(*args)).norm() < 100*TOL


def test_source_path_46():
    f = chebfun3(lambda x, y, z: x+y*z)
    curve = chebfun(lambda t: jnp.stack((jnp.cos(t), jnp.sin(t), t/(8*jnp.pi)), axis=-1),
                    domain=(0, 8*np.pi))
    x, y, z = (curve[:, i] for i in range(3))
    assert (x+y*z-f(x, y, z)).norm() < 100*TOL


@pytest.mark.parametrize('clause', [47, 48, 49])
def test_source_orientation(clause):
    if clause == 49:
        def ff(x, y, z):
            return x
        xx = jnp.array([[-0.4, 0.4], [-0.4, 0.4]])
        yy = jnp.array([[-0.2, -0.2], [0.2, 0.2]])
        xyz = (xx, 0*xx, yy)
    else:
        def ff(x, y, z):
            return jnp.sin(x+2*y+z)
        xyz = jnp.meshgrid(jnp.linspace(-1, 1, 2), jnp.linspace(-1, 1, 4), indexing='xy')
        xyz = (*xyz, jnp.zeros_like(xyz[0]))
    f = chebfun3(ff)
    values = f(*xyz)
    if clause == 47:
        assert xyz[0].shape == values.shape
    else:
        assert array_norm(values-ff(*xyz)) < (100*TOL if clause == 49 else TOL)
