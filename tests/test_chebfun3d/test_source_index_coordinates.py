"""Source cosine mapping retains indices beyond the nominal grid bounds.

Provenance
----------
MATLAB source: @chebfun3/chebfun3f.m, chebX/Y/Z, evalTensor and phase2
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d import _chebfun3f_source as source3f
from chebfunjax.chebfun3d.chebfun3 import _eval_tensor


def _native_coordinates(indices, n, a, b):
    indices = np.asarray(indices, dtype=np.float64)
    return a + (-np.cos(indices * np.pi / (n - 1)) + 1) * (b - a) / 2


def test_arbitrary_indices_follow_native_cosine_formula_inside_domain():
    n, a, b = 7, -2.0, 3.0
    indices = np.asarray([0, n - 1, n, n + 2])
    actual = np.asarray(source3f.cosine_coordinates(indices, n, a, b))
    expected = _native_coordinates(indices, n, a, b)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    assert np.all((actual >= a) & (actual <= b))


def test_source_phase1_evaluation_maps_raw_indices_without_clipping():
    n, domain = 7, (-2., 3., -1., 2., 4., 5.)
    grids = [source3f.cosine_points(n, *domain[2*k:2*k+2]) for k in range(3)]
    seen = []

    def op(x, y, z):
        seen.append((np.asarray(x), np.asarray(y), np.asarray(z)))
        return x + 10*y + 100*z

    indices = (np.asarray([n]), np.asarray([n + 1]), np.asarray([n + 2]))
    values = _eval_tensor(op, *grids, *indices,
                          source_sizes=(n, n, n), source_domain=domain)
    mapped = tuple(axis.reshape(-1)[0] for axis in seen[0])
    expected = tuple(_native_coordinates([index[0]], n, *domain[2*k:2*k+2])[0]
                     for k, index in enumerate(indices))
    np.testing.assert_allclose(mapped, expected, rtol=0, atol=2e-15)
    assert np.asarray(values).shape == (1, 1, 1)


def test_nested_phase2_indices_use_native_cosine_mapping_and_retain_old_rows(
    monkeypatch,
):
    n, domain = 5, (-1., 1., -2., 2., 3., 4.)
    old = jnp.arange(1., n + 1)[:, None]
    fibers = (old, jnp.ones((n, 1)), jnp.ones((n, 1)))
    pairs = ((jnp.asarray([n]), jnp.asarray([n + 1])),
             (jnp.asarray([0]), jnp.asarray([0])),
             (jnp.asarray([0]), jnp.asarray([0])))
    observed = []
    happiness_calls = 0

    def happiness(values, tech, relative_tol):
        nonlocal happiness_calls
        happiness_calls += 1
        return happiness_calls != 1

    def op(x, y, z):
        observed.append((np.asarray(y), np.asarray(z)))
        return x + y + z

    monkeypatch.setattr(source3f, "envelope_happiness", happiness)
    refined, _, _, *_ = source3f.refine_polynomial_fibers(
        op, fibers, pairs, (n, n, n), domain, 1e-12, object())

    expected_y = _native_coordinates([n], n, *domain[2:4])[0]
    expected_z = _native_coordinates([n + 1], n, *domain[4:6])[0]
    np.testing.assert_allclose(observed[0][0], expected_y, rtol=0, atol=2e-15)
    np.testing.assert_allclose(observed[0][1], expected_z, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(np.asarray(refined[::2, 0]), np.asarray(old[:, 0]))


def test_phase2_uses_iteration_grid_snapshot_after_earlier_axis_refines(monkeypatch):
    n, domain = 5, (-1., 1., -2., 2., 3., 4.)
    fibers = tuple(jnp.ones((n, 1)) for _ in range(3))
    pairs = tuple((jnp.asarray([n]), jnp.asarray([n + 1])) for _ in range(3))
    # Initially U/V unresolved. U remains unresolved after its first update,
    # which must not change the coordinate map used by V in that iteration.
    happiness = iter((False, False, True, False, True, True))
    observed = []
    monkeypatch.setattr(source3f, "envelope_happiness", lambda *args: next(happiness))

    def op(x, y, z):
        observed.append(tuple(np.asarray(v) for v in (x, y, z)))
        return x + y + z

    source3f.refine_polynomial_fibers(
        op, fibers, pairs, (n, n, n), domain, 1e-12, object())
    # Second callback refines V; its selected x is mapped on m_x=9,
    # captured by native ff, even though the next U grid is already 17.
    expected_x = _native_coordinates([2*n], 2*n - 1, *domain[:2])[0]
    np.testing.assert_allclose(observed[1][0], expected_x, rtol=0, atol=2e-15)
