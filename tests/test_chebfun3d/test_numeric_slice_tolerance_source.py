"""Native first-slice GetTol2D geometry, including equispaced input.

Provenance
----------
MATLAB source: @chebfun3/chebfun3double.m completeACA3D lines219–223.
Chebfun commit: 7574c77. This supplemental dispatch/formula test stops before
factor construction; it does not replace original equi predicates.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d import _chebfun3double as numeric
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2, Trigtech])
@pytest.mark.parametrize('equi', [False, True])
def test_first_slice_uses_selected_tech_grid(monkeypatch, tech, equi):
    values = jnp.arange(1., 61.).reshape(4, 3, 5)
    domain = (-2., 1., -1., 2., .5, 3.)
    observed = {}
    original = numeric.tensor_tolerance

    class Captured(Exception):
        pass

    def capture(matrix, axes, dom, epsilon):
        result = original(matrix, axes, dom, epsilon)
        if matrix.ndim == 2:
            observed.update(matrix=np.asarray(matrix), axes=[np.asarray(x) for x in axes],
                            tolerance=float(result[1]), epsilon=float(epsilon))
            raise Captured
        return result

    monkeypatch.setattr(numeric, 'tensor_tolerance', capture)
    with pytest.raises(Captured):
        numeric.construct(values, domain, np.finfo(float).eps, tech, equi=equi)
    expected_axes = []
    for n, a, b in zip((4, 3), domain[::2], domain[1::2]):
        if tech is Trigtech:
            t = -1 + 2*np.arange(n)/n
        elif tech is Chebtech1:
            t = -np.cos(np.pi*(2*np.arange(n)+1)/(2*n))
        else:
            t = -np.cos(np.pi*np.arange(n)/(n-1))
        expected_axes.append((b-a)*t/2 + (a+b)/2)
    for actual, expected in zip(observed['axes'], expected_axes):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=4*np.finfo(float).eps)
    matrix = observed['matrix']
    gradients = [np.max(np.abs(np.diff(matrix[:, :-1], axis=0)/np.diff(expected_axes[0])[:, None])),
                 np.max(np.abs(np.diff(matrix[:-1, :], axis=1)/np.diff(expected_axes[1])[None, :]))]
    scale = max(max(np.asarray(gradients)*[3., 3.]), np.max(np.abs(matrix)))
    expected = max(matrix.shape)**(4/5)*observed['epsilon']*scale
    np.testing.assert_allclose(observed['tolerance'], expected,
                               rtol=8*np.finfo(float).eps, atol=0)
