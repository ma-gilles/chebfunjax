"""Source cleanup branch contracts, aaatrig.m at Chebfun 7574c77."""
import importlib

import jax.numpy as jnp
import pytest

module = importlib.import_module('chebfunjax.utils.aaa')


@pytest.mark.parametrize('form', ['odd', 'even'])
@pytest.mark.parametrize('count', [1, 2])
def test_native_sequential_truncated_support_removal(monkeypatch, form, count):
    # Native fix(real((z-p)/pi)) truncates toward zero. Each deletion
    # changes the support array before the next pole is considered.
    supports = jnp.array([.1, .5, 1., 2.], dtype=jnp.complex128)
    values = 2+jnp.cos(supports)
    poles = jnp.array([.15, .16][:count], dtype=jnp.complex128)
    monkeypatch.setattr(module, '_prztrig_np',
                        lambda *args: (poles, jnp.zeros(count), jnp.array([])))
    sample = jnp.linspace(.01, 3., 40)
    with pytest.warns(UserWarning, match='Froissart'):
        z, f, weights = module._cleanup_trig(
            supports, values, jnp.ones(4), sample, 2+jnp.cos(sample), 1e-12, form)
    assert jnp.array_equal(z, supports[count:])
    assert jnp.array_equal(f, values[count:])
    assert jnp.all(jnp.isfinite(weights))
    assert abs(jnp.linalg.norm(weights)-1) < 1e-14


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_public_removes_small_rational_term(form):
    sample = jnp.linspace(.1, 2*jnp.pi-.1, 80)
    # The added term has known poles +/-i*acosh(2), residues
    # 1e-8/sin(p); cleanup_tol=1e-5 must detect these small residues.
    values = jnp.sin(sample)+1e-8/(2-jnp.cos(sample))
    with pytest.warns(UserWarning, match='Froissart'):
        r, *_ = module.aaatrig(values, sample, form=form, cleanup_tol=1e-5)
    assert jnp.max(jnp.abs(r(sample)-values)) < 2e-8
    dense = jnp.linspace(.1, 2*jnp.pi-.1, 311)
    assert jnp.max(jnp.abs(r(dense)-jnp.sin(dense))) < 1e-7
