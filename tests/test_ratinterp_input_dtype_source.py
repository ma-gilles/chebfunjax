"""Native real data/pad arithmetic and preserved complex rational inputs."""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.ratapprox import ratinterp


@pytest.mark.parametrize("nodes", ["type1", "type2", "equi", jnp.linspace(-1, 1, 7)])
@pytest.mark.parametrize("complex_input", [False, True])
def test_real_and_complex_input_dtype_and_analytic_fit(nodes, complex_input):
    amplitude = 1 + 0.25j if complex_input else 1.0
    def function(x):
        return amplitude * (1 + 0.3 * x) / (3 + 0.2 * x)
    _, p, q, *_ = ratinterp(function, 1, 1, NN=7, xi=nodes)
    assert jnp.iscomplexobj(p) == complex_input and jnp.iscomplexobj(q) == complex_input
    x = jnp.asarray([-0.9, -0.3, 0.1, 0.8])
    # Independent first-degree Chebyshev evaluation avoids the handle adapter.
    values = (p[0] + p[1] * x) / (q[0] + q[1] * x)
    np.testing.assert_allclose(values, function(x), rtol=0, atol=100 * np.finfo(float).eps)
