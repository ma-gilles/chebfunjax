"""Native test_aaatrig.m predicates9,18-23 and option controls,7574c77."""
import importlib

import jax.numpy as jnp
import pytest

from chebfunjax.utils.quadrature import chebpts_ab

module = importlib.import_module('chebfunjax.utils.aaa')


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_native_degree_alias_21_to_23(form):
    z = jnp.linspace(-1., 1., 1000)
    f = jnp.exp(z)
    r, poles, *_ = module.aaatrig(f, z, form=form, degree=3)
    r2, poles2, *_ = module.aaatrig(f, z, form=form, mmax=4)
    assert len(poles) <= 4
    assert len(poles2) <= 4
    assert jnp.linalg.norm(r(z)-r2(z)) < 1e-10


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_native_lawson_machine_precision_18(form):
    x = jnp.linspace(-1., 1., 100)
    r, *_ = module.aaatrig(jnp.tanh, x, form=form)
    err1 = jnp.linalg.norm(jnp.tanh(x)-r(x), ord=jnp.inf)
    r, *_ = module.aaatrig(jnp.tanh, x, form=form, mmax=40)
    err2 = jnp.linalg.norm(jnp.tanh(x)-r(x), ord=jnp.inf)
    assert abs(err2/err1-1) < 1.01


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_native_lawson_symmetry_19(form):
    z = jnp.exp(2j*jnp.pi*jnp.arange(1, 501)/500)
    f = jnp.log(2-jnp.sin(z)**4)
    r, *_ = module.aaatrig(f, z, form=form, mmax=16, lawson=0)
    err1 = jnp.linalg.norm(f-r(z), ord=jnp.inf)
    r, *_ = module.aaatrig(f, z, form=form, mmax=16)
    err2 = jnp.linalg.norm(f-r(z), ord=jnp.inf)
    assert abs(err2/err1-1) < 1.01


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_native_lawson_troublesome_poles_20(form):
    za, zb = chebpts_ab(1000, -3., -1.), chebpts_ab(1000, 1., 3.)
    z = jnp.concatenate((za, zb))
    f = jnp.concatenate((jnp.sign(za), jnp.sign(zb)))
    r, *_ = module.aaatrig(f, z, form=form, mmax=13, lawson=0)
    err1 = jnp.linalg.norm(f-r(z), ord=jnp.inf)
    r, *_ = module.aaatrig(f, z, form=form, mmax=13)
    err2 = jnp.linalg.norm(f-r(z), ord=jnp.inf)
    assert abs(err2/err1-1) < 1.01


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_native_two_samples_9(form):
    z, f = jnp.array([0., 1.]), jnp.array([1., 2.])
    r, _, _, _, _, _, weights, errvec = module.aaatrig(f, z, form=form)
    assert jnp.linalg.norm(f-r(z), ord=jnp.inf) < 1e4*jnp.finfo(jnp.float64).eps
    # Native antisymmetric weights give midpoint average; old equal weights
    # produced a pole here despite interpolating both support points.
    assert abs(r(jnp.array(.5))-1.5) < 1e-14
    assert weights[0]+weights[1] == 0
    assert errvec[1] == 0


def test_degree_mmax_consistency():
    with pytest.raises(ValueError, match='degmmaxmismatch'):
        module.aaatrig(jnp.exp, jnp.linspace(-1., 1., 10), degree=3, mmax=5)


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_explicit_finite_lawson_steps_and_no_cleanup(monkeypatch, form):
    calls = []
    actual = module._trig_lawson_step_source

    def observed(*args):
        calls.append(1)
        return actual(*args)

    def forbidden(*args, **kwargs):
        raise AssertionError('Native cleanup must be disabled when nlawson>0.')

    monkeypatch.setattr(module, '_trig_lawson_step_source', observed)
    monkeypatch.setattr(module, '_cleanup_trig', forbidden)
    z = jnp.linspace(-1., 1., 100)
    r, *_ = module.aaatrig(jnp.exp, z, form=form, degree=3, lawson=3)
    assert len(calls) == 3
    assert jnp.all(jnp.isfinite(r(z)))
