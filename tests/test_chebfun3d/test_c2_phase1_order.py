"""Native column ordering and first-maximum ties in C2 construction.

Provenance
----------
MATLAB source : @chebfun3/chebfun3f.m, phase1 and ACA
Chebfun commit: 7574c77
"""

import importlib

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d import _chebfun3f_source as source
from chebfunjax.chebfun3d.chebfun3 import Chebfun3


def test_c2_first_fibers_visit_y_before_z(monkeypatch):
    module = importlib.import_module('chebfunjax.chebfun3d.chebfun3')
    sampled = []
    captured = []

    class FirstCrossCaptured(Exception):
        pass

    def function(x, y, z):
        values = x+10*y+100*z
        if jnp.ndim(values) == 3:
            sampled.append(values)
        return values

    def capture(matrix, tolerance, max_iterations):
        captured.append(jnp.asarray(matrix))
        raise FirstCrossCaptured

    monkeypatch.setattr(module, '_initialize_random_indices',
                        lambda *args: jnp.asarray([1, 3]))
    monkeypatch.setattr(source, 'aca', capture)
    with pytest.raises(FirstCrossCaptured):
        Chebfun3.from_function(function, _r_init=(2, 2, 2), _sample_test=False)
    values = sampled[-1]
    # Explicit native fiber sequence: y varies first at each fixed z.
    expected = jnp.stack([values[:, 0, 0], values[:, 1, 0],
                          values[:, 0, 1], values[:, 1, 1]], axis=1)
    assert jnp.array_equal(captured[0], expected)


def test_c2_refines_until_rank_fits_and_restarts_with_exact_ranks(monkeypatch):
    module = importlib.import_module('chebfunjax.chebfun3d.chebfun3')
    initializations = []
    shapes = []

    class RetryCaptured(Exception):
        pass

    def initialize(rank, size, restart):
        initializations.append((rank, size))
        return jnp.arange(rank)

    def capture(matrix, tolerance, max_iterations):
        shapes.append(matrix.shape)
        if len(shapes) == 4:
            raise RetryCaptured
        rank = 5 if len(shapes) == 1 else 1
        indices = jnp.arange(rank)
        return None, None, None, indices, indices

    monkeypatch.setattr(module, '_initialize_random_indices', initialize)
    monkeypatch.setattr(source, 'aca', capture)
    with pytest.raises(RetryCaptured):
        Chebfun3.from_function(lambda x, y, z: x + y + z,
                              _n_init=(9, 9, 9), _r_init=(1, 3, 2),
                              _sample_test=False)
    # Native rank 5 requires n >= 5*2*sqrt(2): 9 -> 12 -> 17.
    assert [shape[0] for shape in shapes[:3]] == [9, 9, 9]
    assert shapes[3] == (17, 1)
    assert initializations == [(3, 9), (2, 9), (1, 9), (1, 9)]
