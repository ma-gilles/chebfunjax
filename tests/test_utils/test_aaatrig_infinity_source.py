"""Infinity mapping/cancellation in prztrig.m, Chebfun commit 7574c77."""
import importlib

import jax.numpy as jnp
import pytest

module = importlib.import_module('chebfunjax.utils.aaa')


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_native_infinity_multiplicity_and_first_pair_cancellation(monkeypatch, form):
    # Inject the generalized eigenproblem outputs to qualify the source's
    # inverse transforms, infinity thresholds, deletion order and projection.
    # Actual unmocked public eigenproblems are checked separately below.
    if form == 'odd':
        poles = [0., 0., 1e11, 2., jnp.inf]
        zeros = [0., 1e11, 1e12, 3., jnp.inf]
        finite_pole, finite_zero = -1j*jnp.log(2.), -1j*jnp.log(3.)
    else:
        poles = [1j, 1j, -1j, 2., jnp.inf]
        zeros = [1j, -1j, -1j, 3., jnp.inf]
        finite_pole, finite_zero = 2*jnp.arctan(2.), 2*jnp.arctan(3.)
    answers = iter([jnp.array(poles, dtype=jnp.complex128),
                    jnp.array(zeros, dtype=jnp.complex128)])
    monkeypatch.setattr(module.spla, 'eig', lambda *args, **kwargs: next(answers))
    p, residues, z = module._prztrig_np(
        jnp.array([0., .5, 1., 2.]), jnp.array([1., 2., 3., 4.]),
        jnp.array([1., -1., 2., 1.]), form)
    assert len(p) == len(z) == len(residues) == 2
    assert jnp.real(p[0]) == 0 and jnp.isposinf(jnp.imag(p[0]))
    assert jnp.real(z[0]) == 0 and jnp.isneginf(jnp.imag(z[0]))
    assert abs(p[1]-finite_pole) < 1e-14
    assert abs(z[1]-finite_zero) < 1e-14


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_public_native_rational_infinity_mapping(form):
    sample = jnp.linspace(.1, 2*jnp.pi-.1, 200)
    r, poles, _, zeros, *_ = module.aaatrig(
        1/(2-jnp.cos(sample)), sample, form=form, cleanup=False)
    assert jnp.max(jnp.abs(r(sample)-1/(2-jnp.cos(sample)))) < 1e-11
    assert not jnp.any(jnp.isnan(poles))
    assert not jnp.any(jnp.isnan(zeros))
    assert jnp.any(jnp.isposinf(jnp.imag(zeros)))
    # The odd generalized pencil can encode the negative-infinite zero
    # as an exact infinite eigenvalue, which native prztrig filters before
    # its logarithmic inverse map. The even finite roots +/-i map to both.
    if form == "even":
        assert jnp.any(jnp.isneginf(jnp.imag(zeros)))
