"""Whole-kernel regression against the previous per-column recurrence.

Provenance
----------
MATLAB source : @chebfun3/feval.m; @chebtech/clenshaw.m
Chebfun commit: 7574c77
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np  # uses-numpy: deterministic inputs and binary64 byte comparisons.
import pytest

from chebfunjax.chebfun3d._power import _evaluate_grid, _evaluate_vector, _txm
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.domain import _linear_inverse_map
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2, _clenshaw
from chebfunjax.tech.trigtech import Trigtech, _trig_eval


def _old_panel(f, axis, points):
    t = _linear_inverse_map(points, *f.domain[2*axis:2*axis+2])
    group = (f.cols, f.rows, f.tubes)[axis]
    return jnp.stack([_trig_eval(g.coeffs, t, g.is_real)
                      if isinstance(g, Trigtech) else _clenshaw(g.coeffs, t)
                      for g in group], axis=-1)


@eqx.filter_jit
def _old_vector(f, points):
    panels = [_old_panel(f, axis, point.reshape(-1))
              for axis, point in enumerate(points)]
    result = _txm(f.core, panels[0], 0)
    result = jnp.einsum('pjk,pj->pk', result, panels[1])
    return jnp.sum(result*panels[2], axis=1).reshape(points[0].shape)


@eqx.filter_jit
def _old_grid(f, vectors, axes, order):
    result = jnp.transpose(f.core, tuple(axes.index(i) for i in range(3)))
    for axis in order:
        result = _txm(result, _old_panel(f, axis, vectors[axis]), axes[axis])
    return result


@pytest.mark.parametrize('case', ['c1', 'c2', 'complex', 'constant',
                                 'even_length', 'heterogeneous', 'mixed_dtype', 'trig',
                                 'complex_core', 'float32'])
def test_batching_preserves_whole_kernel_bytes(case):
    rng = np.random.default_rng(6178)
    groups = []
    for rank in (2, 3, 4):
        group = []
        for column in range(rank):
            n = 1 if case == 'constant' else 16 if case == 'even_length' else 15
            if case == 'heterogeneous':
                n -= column
            coeffs = rng.normal(size=n)*(.4**np.arange(n))
            if case == 'complex' or (case == 'mixed_dtype' and column == 0):
                coeffs = coeffs+1j*rng.normal(size=n)*(.4**np.arange(n))
            if case == 'float32':
                coeffs = coeffs.astype(np.float32)
            tech = Chebtech1 if case == 'c1' else Chebtech2
            if case == 'trig' and column == 0:
                factor = Trigtech.from_values(jnp.asarray(rng.normal(size=9)))
            elif case == 'float32':
                # from_coeffs promotes to double; use explicit stored dtype.
                factor = Chebtech2(coeffs=jnp.asarray(coeffs))
                assert factor.coeffs.dtype == jnp.float32
            else:
                factor = tech.from_coeffs(jnp.asarray(coeffs))
            group.append(factor)
        groups.append(group)
    core = jnp.asarray(rng.normal(size=(2, 3, 4)))
    if case == 'complex_core':
        core = core+1j*jnp.asarray(rng.normal(size=(2, 3, 4)))
    f = Chebfun3(*groups, core,
                (-1., 1., -1., 1., -1., 1.))
    points = tuple(jnp.asarray(rng.uniform(-1, 1, size=7)) for _ in range(3))
    assert (np.asarray(_evaluate_vector(f, points)).tobytes()
            == np.asarray(_old_vector(f, points)).tobytes())
    vectors = tuple(jnp.linspace(-1, 1, n) for n in (3, 4, 5))
    for axes, order in [((0, 1, 2), (0, 1, 2)), ((1, 0, 2), (1, 0, 2))]:
        assert (np.asarray(_evaluate_grid(f, vectors, axes, order)).tobytes()
                == np.asarray(_old_grid(f, vectors, axes, order)).tobytes())
