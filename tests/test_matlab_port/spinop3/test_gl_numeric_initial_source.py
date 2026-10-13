"""GL numeric initial field and shared JAX stream integration controls.

Provenance
----------
MATLAB source: @spinop3/spinop3.m, GL parseInputs branch.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
These are supplemental controls, not a MATLAB normal-stream oracle.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.operators.spinop3 import Spinop3, spin3
from chebfunjax.tech.trigtech import Trigtech
from chebfunjax.utils import _randnfun


def test_gl_initial_field_and_short_solve(monkeypatch):
    key = jax.random.key(6178)
    monkeypatch.setattr(_randnfun, '_DEFAULT_KEY', key)
    following, draw = jax.random.split(key)
    expected = .1*jax.random.normal(draw, (1, 32**3), dtype=jnp.float64).T
    expected = expected.reshape((32, 32, 32), order='F')
    operator = Spinop3('GL')
    assert isinstance(operator.init, Chebfun3)
    assert operator.init.domain == operator.domain == (0., 50., 0., 50., 0., 50.)
    assert operator.tspan == (0., 100.)
    groups = (operator.init.cols, operator.init.rows, operator.init.tubes)
    assert all(isinstance(t, Trigtech) and t.coeffs.shape[0] == 32
               for group in groups for t in group)
    np.testing.assert_array_equal(jax.random.key_data(_randnfun._DEFAULT_KEY),
                                  jax.random.key_data(following))
    points = jnp.arange(32)*50/32
    grid = jnp.meshgrid(points, points, points, indexing='ij')
    error = float(jnp.max(jnp.abs(operator.init(*grid)-expected)))
    # Supplemental constructor convention from tests/chebfun3/test_constructor.m.
    assert error < 100*jnp.finfo(jnp.float64).eps
    operator.tspan = (0., .02)
    solution = spin3(operator, 8, .01, 'plot', 'off')
    query = jnp.asarray([1.3, 17.1, 42.7])
    values = np.asarray(solution(query, query+1, query+2))
    assert values.shape == (3,)
    assert np.all(np.isfinite(values))
    assert np.max(np.abs(values-np.asarray(operator.init(query, query+1, query+2)))) > 0


@pytest.mark.parametrize('system', [False, True])
@pytest.mark.parametrize('options, keywords, masked', [
    ((), {}, False), (('dealias', 'on'), {}, True), ((), {'dealias': True}, True),
])
def test_public_dealias_default_and_explicit_on(system, options, keywords, masked):
    # Native masks a central Fourier cube at output, not high-frequency strips.
    domain = (0., 2*jnp.pi, 0., 2*jnp.pi, 0., 2*jnp.pi)
    operator = Spinop3(domain, (0., .01))
    def low(x, y, z):
        return jnp.sin(x)+.2*jnp.cos(3*x)
    def initial(x, y, z):
        return low(x,y,z)+.1*jnp.cos(3*x)*jnp.cos(3*y)*jnp.cos(3*z)
    if system:
        operator.lin = '@(u,v) [0*lap(u);0*lap(v)]'
        operator.nonlin = [lambda u,v: 0*u, lambda u,v: 0*v]
        operator.init = [initial, lambda x,y,z: 2*initial(x,y,z)]
    else:
        operator.lin = '@(u) 0*lap(u)'
        operator.nonlin = lambda u: 0*u
        operator.init = initial
    solution = spin3(operator, 8, .01, *options, **keywords)
    components = solution.components if system else [solution]
    expected = (low if masked else initial)(.31,.27,.19)
    for index, component in enumerate(components):
        assert abs(complex(component(.31,.27,.19))-(index+1)*complex(expected)) < 1e-12
