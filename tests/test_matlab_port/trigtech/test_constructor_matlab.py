"""Literal nine clauses of tests/trigtech/test_constructor.m, pin7574c77.

Nested/resampling are actual preferences. Matrix infinity norm is row-sum max;
slot7 uses adaptive minSamples=maxLength8. Slots8/9 use source normest.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech, trigpts

EPS = jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('slot,refinement,array', [
    (1, 'nested', False), (2, 'nested', True),
    (3, 'resampling', False), (4, 'resampling', True),
])
def test_source_resolved_values(slot, refinement, array, record_property):
    record_property('native_slot', slot)

    def op(x):
        if array:
            return jnp.stack([jnp.exp(jnp.sin(jnp.pi*x)),
                              jnp.sin(jnp.cos(4*jnp.pi*x)), jnp.cos(jnp.pi*x)], axis=1)
        return jnp.tanh(jnp.sin(jnp.pi*x))

    g = Trigtech.from_function(op, pref={'refinementFunction': refinement},
                              data={'vscale': 0., 'hscale': 1.})
    error = jnp.abs(op(trigpts(g.n))-g.values)
    norm = jnp.max(jnp.sum(error, axis=1)) if array else jnp.max(error)
    assert norm < 10*jnp.max(g.vscale_columns()*EPS)


@pytest.mark.parametrize('slot,bad', [(5, jnp.nan), (6, jnp.inf)])
def test_source_probe_error_message(slot, bad, record_property):
    record_property('native_slot', slot)
    with pytest.raises(ValueError) as error:
        Trigtech.from_function(lambda x: jnp.sin(jnp.pi*x)+bad,
                               pref={'refinementFunction': 'resampling'})
    assert str(error.value) == 'Cannot handle functions that evaluate to Inf or NaN.'


def test_source_actual_adaptive_min_equals_max(record_property):
    record_property('native_slot', 7)
    Trigtech.from_function(lambda x: jnp.sin(jnp.pi*x),
                          pref={'minSamples': 8, 'maxLength': 8,
                                'refinementFunction': 'resampling'})


@pytest.mark.parametrize('slot,truth', [(8, True), (9, False)])
def test_source_logical_normest(slot, truth, record_property):
    record_property('native_slot', slot)
    op = (lambda x: x > -2) if truth else (lambda x: x < -2)
    f = Trigtech.from_function(op)
    g = Trigtech.from_values(jnp.asarray([1. if truth else 0.]))
    assert (f-g).normest() < EPS
