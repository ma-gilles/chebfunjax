"""Native chebpts.m scaleNodes identity-domain contract in actual seed calls.

Chebfun7574c77, copyright The University of Oxford and The Chebfun Developers.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np  # uses-numpy: exact host-side bit comparison in tests.
import pytest

from chebfunjax.chebfun2d._extrema_fallback import source_seed
from chebfunjax.utils.quadrature import chebpts


class RecordedFactor:
    def __init__(self, n, bounds):
        self.n = n
        self.domain = SimpleNamespace(a=bounds[0], b=bounds[1])
        self.points = None

    def __len__(self):
        return self.n

    def __call__(self, points):
        self.points = points
        return jnp.stack((jnp.ones_like(points), points), axis=1)


@pytest.mark.parametrize("n", [2, 3, 17, 100])
@pytest.mark.parametrize(
    "domains",
    [
        ((-1.0, 1.0), (-1.0, 1.0)),
        ((-1.0, 1.0), (2.0, 5.0)),
        ((-2.0, 3.0), (-1.0, 1.0)),
        ((-2.0, 3.0), (0.0, 1.0)),
    ],
)
def test_seed_uses_native_scale_nodes(n, domains):
    rows = RecordedFactor(n, domains[0])
    cols = RecordedFactor(n, domains[1])
    source_seed(rows, cols, lambda point: jnp.sum(point))
    nodes = np.asarray(chebpts(n))
    for factor, bounds in zip((rows, cols), domains):
        # Native chebpts.m lines129–135: exact identity shortcut, otherwise
        # two endpoint-weighted terms. Scalar host ops keep operation order.
        a, b = bounds
        expected = (
            nodes
            if bounds == (-1.0, 1.0)
            else np.array([b * (float(t) + 1) / 2 + a * (1 - float(t)) / 2 for t in nodes])
        )
        assert np.asarray(factor.points).tobytes() == expected.tobytes()
