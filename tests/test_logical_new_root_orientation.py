"""Supplemental fresh-row root insertion; not counted as native predicates."""

import jax.numpy as jnp

import chebfunjax as cj


def test_fresh_row_root_insertion_preserves_storage_orientation():
    f = cj.chebfun(lambda x: jnp.stack([x - 0.25, x + 0.5], axis=-1)).transpose()
    assert tuple(f.domain.breakpoints) == (-1.0, 1.0)
    out = f.addBreaksAtRoots()
    assert out.is_transposed
    assert out.point_values.shape == (4, 2)
    assert bool(
        jnp.allclose(
            jnp.asarray(out.domain.breakpoints),
            jnp.asarray([-1.0, -0.5, 0.25, 1.0]),
            atol=0,
            rtol=1e-15,
        )
    )
    probes = jnp.asarray([-0.7, 0.1, 0.8])
    assert bool(jnp.allclose(out(probes), f(probes), atol=1e-15, rtol=1e-15))
    assert out.point_values[1, 1] == 0 and out.point_values[2, 0] == 0
    g = cj.chebfun(lambda x: jnp.stack([jnp.ones_like(x), jnp.ones_like(x)], axis=-1)).transpose()
    h = f.logical_and(g)
    assert h.is_transposed and bool(jnp.all(h(probes) == 1))
    assert h.point_values[1, 1] == 0 and h.point_values[2, 0] == 0
