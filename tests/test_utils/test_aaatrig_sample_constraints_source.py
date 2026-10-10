"""Input infinity constraints in aaatrig.m, Chebfun7574c77.

These are analytic/source controls, not replacements for native predicates.
Negative-only even, repeated constraints, and no finite samples remain
unqualified source edge branches; no symmetric normalization is presumed.
"""
import importlib

import jax.numpy as jnp
import pytest

module = importlib.import_module('chebfunjax.utils.aaa')
PLUS = complex(0., float('inf'))
MINUS = complex(0., -float('inf'))


@pytest.mark.parametrize('mode', ['plus', 'minus', 'both'])
def test_complex_odd_constraint_blocks(mode):
    z = jnp.array([.2+.3j, 1.-.2j])
    f = jnp.array([2.+1j, -.5j])
    p = jnp.array([3.+2j]) if mode != 'minus' else jnp.array([])
    m = jnp.array([-1.+.5j]) if mode != 'plus' else jnp.array([])
    blocks = module._trig_constraint_blocks_source(z, f, p, m, 'odd')
    expected = []
    if len(p):
        expected.append(jnp.array([[(p[0]-f[0])*jnp.exp(-1j*z[0]/2),
                                    (p[0]-f[1])*jnp.exp(-1j*z[1]/2)]]))
    if len(m):
        expected.append(jnp.array([[(m[0]-f[0])*jnp.exp(1j*z[0]/2),
                                    (m[0]-f[1])*jnp.exp(1j*z[1]/2)]]))
    assert len(blocks) == len(expected)
    for actual, reference in zip(blocks, expected):
        assert jnp.max(jnp.abs(actual-reference)) < 1e-14


@pytest.mark.parametrize('paired', [False, True])
def test_even_constraint_is_one_common_row(paired):
    blocks = module._trig_constraint_blocks_source(
        jnp.array([.2+.3j, 1.-.2j]), jnp.array([2.+1j, -.5j]),
        jnp.array([3.+2j]), jnp.array([3.+2j]) if paired else jnp.array([]), 'even')
    assert len(blocks) == 1
    assert jnp.array_equal(blocks[0], jnp.array([[1.+1j, 3.+2.5j]]))


def test_limit_error_native_order_and_signs():
    # For z=[0,pi], f=[3,5], w=[1,2], limits are4.6-/+.8i.
    error = module._trig_constraint_errors_source(
        jnp.array([0., jnp.pi]), jnp.array([3., 5.]), jnp.array([1., 2.]),
        jnp.array([2.+1j]), jnp.array([-1.+2j]), 'odd')
    assert jnp.max(jnp.abs(error-jnp.array([-2.6+1.8j, -5.6+1.2j]))) < 1e-14


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_constraint_error_prevents_finite_only_stopping(form):
    # First support is value1. Finite residual2 is below tol*finiteScale=3,
    # but the supplied limit1000 leaves error999 and requires step2.
    out = module.aaatrig(jnp.array([1., 2., 3., 1000.]),
                        jnp.array([.1, .8, 1.7, PLUS]), form=form,
                        tol=1., mmax=2, lawson=0, cleanup=False)
    assert len(out[7]) == 2
    assert abs(out[7][0]-999.) < 1e-12


def test_even_distinct_paired_limits_rejected():
    with pytest.raises(ValueError, match='must take the same values'):
        module.aaatrig(jnp.array([1., 2., 3., 4.]),
                       jnp.array([.1, .8, PLUS, MINUS]), form='even')


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_nonfinite_data_removed_before_constraint_extraction(form):
    z = jnp.array([.1, .8, 1.7])
    f = jnp.array([2., 3., 5.])
    out = module.aaatrig(jnp.concatenate((f, jnp.array([jnp.nan]))),
                        jnp.concatenate((z, jnp.array([PLUS]))),
                        form=form, lawson=0, cleanup=False)
    reference = module.aaatrig(f, z, form=form, lawson=0, cleanup=False)
    points = jnp.linspace(.2, 1.5, 13)
    assert jnp.max(jnp.abs(out[0](points)-reference[0](points))) < 1e-12
    assert jnp.all(jnp.isfinite(out[4]))


def _analytic_samples(form):
    z = jnp.linspace(.1, 2*jnp.pi-.1, 31)
    if form == 'odd':
        return z, 1/(2-jnp.exp(1j*z)), .5, 0.
    return z, 1/(2-jnp.cos(z)), 0., 0.


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_lawson_warns_and_does_not_call_cleanup(monkeypatch, form):
    z, f, p, m = _analytic_samples(form)
    def forbidden(*args, **kwargs):
        pytest.fail('Native skips cleanup when Lawson is active.')
    monkeypatch.setattr(module, '_cleanup_trig', forbidden)
    with pytest.warns(UserWarning, match='not currently compatible with Lawson'):
        r, *_ = module.aaatrig(jnp.concatenate((f, jnp.array([p, m]))),
                              jnp.concatenate((z, jnp.array([PLUS, MINUS]))),
                              form=form, lawson=1)
    assert jnp.all(jnp.isfinite(r(z)))
    # Source warns that Lawson may lose the limit constraints: no such claim.


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_cleanup_retains_constraints_with_zero_finite_refit_rows(monkeypatch, form):
    z = jnp.array([.1, .5, 1., 2.], dtype=jnp.complex128)
    f = 1/(2-jnp.exp(1j*z)) if form == 'odd' else 1/(2-jnp.cos(z))
    p, m = (.5, 0.) if form == 'odd' else (0., 0.)
    monkeypatch.setattr(module, '_prztrig_np',
                        lambda *args: (jnp.array([.11]), jnp.array([0.]), jnp.array([])))
    # Delete support0; all samples then coincide with surviving supports,
    # so cleanup has only the appended constraint rows for its SVD.
    with pytest.warns(UserWarning, match='Froissart'):
        zj, fj, w = module._cleanup_trig(
            z, f, jnp.ones(4), z[1:], f[1:], 1e-12, form,
            finfP=jnp.array([p]), finfM=jnp.array([m]))
    assert jnp.array_equal(zj, z[1:])
    assert w.shape == (3,)
    assert abs(jnp.linalg.norm(w)-1) < 1e-14
    r = module._make_trig_callable(zj, fj, w, form)
    assert jnp.max(jnp.abs(r(jnp.array([PLUS, MINUS]))-jnp.array([p, m]))) < 1e-12
    assert jnp.max(jnp.abs(r(zj)-fj)) < 1e-12


@pytest.mark.parametrize('form,mode', [('odd', 'plus'), ('odd', 'minus'),
                                      ('odd', 'both'), ('even', 'plus'), ('even', 'both')])
def test_public_analytic_input_constraints(form, mode):
    z, f, p, m = _analytic_samples(form)
    infz = [PLUS] if mode == 'plus' else [MINUS] if mode == 'minus' else [PLUS, MINUS]
    inff = [p] if mode == 'plus' else [m] if mode == 'minus' else [p, m]
    out = module.aaatrig(jnp.concatenate((f, jnp.array(inff))),
                        jnp.concatenate((z, jnp.array(infz))),
                        form=form, lawson=0, cleanup=False)
    r = out[0]
    assert jnp.all(jnp.isfinite(out[4]))
    assert jnp.max(jnp.abs(r(jnp.array(infz))-jnp.array(inff))) < 1e-12
    x = jnp.linspace(.05, 2*jnp.pi-.05, 101)
    reference = 1/(2-jnp.exp(1j*x)) if form == 'odd' else 1/(2-jnp.cos(x))
    assert jnp.max(jnp.abs(r(x)-reference)) < 1e-12
    assert out[7][-1] <= 1e-13*jnp.max(jnp.abs(f))
