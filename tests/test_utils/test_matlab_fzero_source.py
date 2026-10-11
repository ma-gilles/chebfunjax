"""Default-double source helper controls; native capture ddvdelay1.

Provenance: MATLAB R2025b fzero.m; Chebfun commit: 7574c77 example use.
These are added adapter controls, not original Chebfun test assertions.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.utils._matlab_fzero import _fzero_default


def test_native_vanishing_delay_construction():
    expected = [
        0,
        1.618033988749895,
        2.0952939852239143,
        2.4959439998341906,
        2.847169959075709,
        3.1632959308069997,
        3.452906976461304,
        3.7216080303711316,
        3.973288709314904,
        4.21077469973445,
        4.436193196131759,
        4.651191852746926,
    ]
    counts = [
        (5, 8, 22),
        (4, 7, 19),
        (5, 6, 18),
        (4, 5, 15),
        (4, 5, 15),
        (4, 4, 13),
        (4, 4, 13),
        (5, 4, 14),
        (4, 3, 11),
        (4, 3, 11),
    ]
    domain = [0.0, float((1 + jnp.sqrt(jnp.asarray(5.0))) / 2)]
    for k in range(10):
        previous = domain[-1]
        root, info = _fzero_default(lambda t: t - 1 / t - previous, previous)
        domain.append(float(root))
        assert (info["iterations"], info["intervaliterations"], info["funcCount"]) == counts[k]
    assert domain == expected


@pytest.mark.parametrize("initial", [0.0, -1.0, [0.0, 3.0], [2.0, 3.0]])
def test_linear_scalar_and_bracket(initial):
    root, _ = _fzero_default(lambda x: x - 2.0, initial)
    assert float(root) == 2.0
