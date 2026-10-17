"""Focused source-output and timing-output checks for 2D and 3D spin."""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.spinop2 import Spinop2, spin2
from chebfunjax.operators.spinop3 import Spinop3, spin3


def _operator(dim):
    domain = tuple(
        value for _ in range(dim) for value in (0.0, 2.0 * float(jnp.pi)))
    if dim == 2:
        op = Spinop2(domain, (0.0, 0.045))
        op.lin = "@(u) lap(u)"
        op.nonlin = "@(u) 0*u"
        op.init = lambda x, y: jnp.cos(x) * jnp.cos(y)
    else:
        op = Spinop3(domain, (0.0, 0.045))
        op.lin = "@(u) lap(u)"
        op.nonlin = "@(u) 0*u"
        op.init = lambda x, y, z: jnp.cos(x) * jnp.cos(y) * jnp.cos(z)
    return op


@pytest.mark.parametrize("dim", [2, 3])
def test_one_output_returns_initial_but_timed_output_requires_checkpoint(dim):
    op = _operator(dim)
    grid = jnp.arange(8) * (2.0 * jnp.pi / 8)
    if dim == 2:
        x, y = jnp.meshgrid(grid, grid)
        expected = op.init(x, y)
        result = spin2(op, 8, 0.02, "dealias", "off")
        actual = result(x, y)
        solve = spin2
    else:
        x, y, z = jnp.meshgrid(grid, grid, grid)
        expected = op.init(x, y, z)
        result = spin3(op, 8, 0.02, "dealias", "off")
        actual = result(x, y, z)
        solve = spin3

    # Native one-output solvepde returns vOut{end}; with no endpoint output
    # saved, that is its initialized vOut{1}=vInit.
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected),
                               rtol=1e-12, atol=1e-12)
    # spincomp's comparison call requests solvepde's third output, whose
    # nargout>2 branch indexes the absent tOut(2) for this nonaligned tf.
    with pytest.raises(IndexError, match=r"tOut\(2\)"):
        solve(op, 8, 0.02, "dealias", "off", return_computing_time=True)
