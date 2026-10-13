"""Source-contract controls for the Swift-Hohenberg Spinop3 preset.

Provenance
----------
MATLAB source: @spinop3/spinop3.m parseInputs, SH branch (Chebfun 7574c77).
The native branch builds .1*randn(32,32,32) and passes that numeric tensor
and [0,25]^3 to chebfun3(...,'trig'). The JAX normal stream is a documented
cross-language stream adapter, not a MATLAB randn identity claim.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.operators.spinop3 import Spinop3, func2str, spin3
from chebfunjax.tech.trigtech import Trigtech
from chebfunjax.utils import _randnfun


def test_native_sh_constructor_stream_grid_and_short_solve(monkeypatch):
    key = jax.random.key(6178)
    monkeypatch.setattr(_randnfun, "_DEFAULT_KEY", key)
    following, draw = jax.random.split(key)
    expected = 0.1 * jax.random.normal(draw, (1, 32**3), dtype=jnp.float64).T
    expected = expected.reshape((32, 32, 32), order="F")

    operator = Spinop3("SH")
    assert operator.domain == (0., 25., 0., 25., 0., 25.)
    assert operator.tspan == (0., 800.)
    assert operator._lin_scales == (-2., -1.)
    assert operator._lin_ops == ("lap", "biharm")
    assert func2str(operator.lin) == "@(u)-2*lap(u)-biharm(u)"
    assert func2str(operator.nonlin) == "@(u)-.9*u-u.^3"
    assert operator._is_real is True
    assert isinstance(operator.init, Chebfun3)
    assert operator.init.domain == operator.domain
    assert all(isinstance(tech, Trigtech) and tech.coeffs.shape[0] == 32
               for group in (operator.init.cols, operator.init.rows, operator.init.tubes)
               for tech in group)
    np.testing.assert_array_equal(jax.random.key_data(_randnfun._DEFAULT_KEY),
                                  jax.random.key_data(following))

    nodes = jnp.arange(32) * (25.0 / 32)
    grid = jnp.meshgrid(nodes, nodes, nodes, indexing="ij")
    initial_values = np.asarray(operator.init(*grid))
    assert not np.iscomplexobj(initial_values)
    error = float(jnp.max(jnp.abs(operator.init(*grid) - expected)))
    # Same supplemental numeric-constructor control as the qualified GL case.
    assert error < 100 * jnp.finfo(jnp.float64).eps

    operator.tspan = (0., 0.02)
    solution = spin3(operator, 8, 0.01, "plot", "off")
    query = jnp.asarray([1.3, 8.7, 22.1])
    values = np.asarray(solution(query, query + 1, query + 2))
    assert values.shape == (3,)
    assert np.isfinite(values).all()
    assert np.max(np.abs(np.imag(values))) < 100 * np.finfo(float).eps
