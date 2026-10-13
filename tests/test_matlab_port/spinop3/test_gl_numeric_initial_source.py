"""GL numeric initial field and shared JAX stream integration controls.

Provenance
----------
MATLAB source: @spinop3/spinop3.m, GL parseInputs branch.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
These are supplemental controls, not a MATLAB normal-stream oracle.
"""
import importlib

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
@pytest.mark.parametrize('options, keywords, expected', [
    ((), {}, False), (('dealias', 'on'), {}, True), ((), {'dealias': True}, True),
])
def test_public_dealias_default_and_explicit_on(system, options, keywords, expected,
                                              monkeypatch):
    module = importlib.import_module('chebfunjax.operators.spinop3')
    operator = Spinop3((0., 1., 0., 1., 0., 1.), (0., .01))
    operator.init = lambda x, y, z: x*0
    operator._lin_scales = (1.,)
    operator._lin_ops = ('lap',)
    operator._nonlin_vals = lambda u: u*0
    operator._is_real = False
    observed = []
    if system:
        operator._n_vars = 2
        def solve_system(s, n, dt, scheme, dealias):
            observed.append(dealias)
        monkeypatch.setattr(module, '_spin3_system', solve_system)
    else:
        def solve_scalar(s, n, dt, *, dealias, scheme):
            observed.append(dealias)
            return None, .01, jnp.zeros((4, 4, 4))
        monkeypatch.setattr(module, '_core_spin3', solve_scalar)
    spin3(operator, 4, .01, *options, **keywords)
    assert observed == [expected]
