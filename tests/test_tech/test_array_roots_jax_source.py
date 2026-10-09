"""Analytical polynomial roots and native per-column assembly controls.

Provenance: @chebtech/roots.m, Chebfun7574c77. These six analytical controls
are distinct from the original MATLAB predicates. No backend-order claim.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('mode', ['real', 'all', 'no_zero_fun'])
def test_polynomial_columns_keep_grid_flags_and_nan_padding(monkeypatch, tech, mode):
    # T0,T1,T2 columns represent x, x^2+1/4, 1 and 0. All retain length3.
    coefficients = jnp.asarray([[0., .75, 1., 0.],
                                [1., 0., 0., 0.],
                                [0., .5, 0., 0.]])
    f = tech.from_coeffs(coefficients, ishappy=False)
    original = tech.roots
    seen = []
    def observed(self, *args, **kwargs):
        if self.coeffs.ndim == 1:
            seen.append((type(self), self.n, self.ishappy, self.coeffs))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(tech, 'roots', observed)
    roots = f.roots(all_roots=mode == 'all', recurse=False,
                    zero_fun=mode != 'no_zero_fun')
    assert isinstance(roots, jax.Array)
    assert len(seen) == 4
    for column, (kind, length, happy, actual) in enumerate(seen):
        assert kind is tech and length == 3 and happy is False
        assert jnp.array_equal(actual, coefficients[:, column])
    assert roots.shape == ((2, 4) if mode == 'all' else (1, 4))
    assert roots.dtype == (jnp.complex128 if mode == 'all' else jnp.float64)
    bound = 50 * jnp.finfo(jnp.float64).eps
    assert abs(roots[0, 0]) < bound
    assert jnp.all(jnp.isnan(roots[:, 2]))
    if mode == 'all':
        pair = roots[:, 1]
        # Independent residual and root separation, without sorting eigenvalues.
        assert jnp.max(jnp.abs(pair * pair + .25)) < bound
        assert abs(pair[0] - pair[1]) > .9
        assert jnp.isnan(roots[1, 0]) and jnp.isnan(roots[1, 3])
    else:
        assert jnp.isnan(roots[0, 1])
    if mode == 'no_zero_fun':
        assert jnp.isnan(roots[0, 3])
    else:
        assert roots[0, 3] == 0
