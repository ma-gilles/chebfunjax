"""Source preference and public Fourier composition controls."""
import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.tech.trigtech import Trigtech


def test_pref_grid_sampletest_and_cap(monkeypatch):
    f = Trigtech.from_function(lambda x: jnp.sin(9*jnp.pi*x))
    seen = []
    original = Trigtech.happiness_check
    def check(coeffs, values, op=None, tol=None, vscale=0.):
        seen.append((len(coeffs), op, float(jnp.max(tol))))
        return original(coeffs, values, op, tol, vscale)
    monkeypatch.setattr(Trigtech, 'happiness_check', staticmethod(check))
    f.compose(jnp.cos, pref={'minSamples':65,'chebfuneps':1e-30,'sampleTest':True})
    assert seen[0][0] == 64 and all(op is None for _,op,_ in seen)
    assert all(tol == float(jnp.finfo(float).eps) for _,_,tol in seen)
    with pytest.warns(UserWarning, match='compose:convfail'):
        result = f.compose(lambda y: jnp.sin(30*y), pref={'maxLength':32})
    assert result.n == 32 and not result.ishappy


def test_public_high_frequency_and_jit_ad():
    g = cj.chebfun(lambda x: jnp.sin(x), domain=[-jnp.pi,jnp.pi], trig=True)
    h = (80*g).cos()
    x = jnp.linspace(-3.1,3.1,151)
    assert jnp.max(jnp.abs(h(x)-jnp.cos(80*jnp.sin(x)))) < 3e-13
    # The 249-mode interpolant evaluates phases up to O(100). A 3e-13
    # absolute allowance covers O(mode_count*eps) phase/reduction rounding
    # at unit scale in both eager and JIT paths; no crossmode bit identity.
    analytic = jnp.cos(80*jnp.sin(x))
    jit_error = float(jnp.max(jnp.abs(jax.jit(lambda z: h(z))(x)-analytic)))
    assert jit_error < 3e-13
    old_nodes = -jnp.pi+2*jnp.pi*jnp.arange(64)/64
    old = Trigtech.from_values(jnp.cos((80*g)(old_nodes))).simplify()
    old_error = float(jnp.max(jnp.abs(old(x/jnp.pi)-analytic)))
    assert old_error > 1e-3
    print({'new_length':len(h),'old_fixed_grid':64,'old_error':old_error,'jit_error':jit_error})
    derivative = jax.jit(jax.vmap(jax.grad(lambda t: h(t))))(x)
    assert jnp.max(jnp.abs(derivative+80*jnp.cos(x)*jnp.sin(80*jnp.sin(x)))) < 3e-11


def test_resampling_and_power_source():
    f = Trigtech.from_function(lambda x: 2+jnp.sin(jnp.pi*x))
    g = f.compose(jnp.exp, pref={'refinementFunction':'resampling'})
    x = jnp.linspace(-.99,.99,90)
    assert jnp.max(jnp.abs(g(x)-jnp.exp(2+jnp.sin(jnp.pi*x)))) < 2e-13
    assert jnp.max(jnp.abs((f**3)(x)-(2+jnp.sin(jnp.pi*x))**3)) < 2e-13
    with pytest.raises(ValueError, match='compose:range'):
        f.compose(f)


def test_public_frechet_composition_propagation():
    from chebfunjax.autodiff.adchebfun import ADChebfun
    domain = [-jnp.pi,jnp.pi]
    f = cj.chebfun(lambda x: .7*jnp.sin(x), domain=domain, trig=True)
    perturbation = cj.chebfun(lambda x: jnp.cos(2*x), domain=domain, trig=True)
    result = ADChebfun(f).cos().tanh()
    actual = result.jacobian.apply(perturbation)
    x = jnp.linspace(-3.1,3.1,101)
    expected = -jnp.sin(.7*jnp.sin(x))*jnp.cos(2*x)/jnp.cosh(jnp.cos(.7*jnp.sin(x)))**2
    assert jnp.max(jnp.abs(actual(x)-expected)) < 1e-13


def test_fixedlength_and_source_errors():
    f = Trigtech.from_function(lambda x: jnp.cos(jnp.pi*x))
    result = f.compose(jnp.sin, pref={'fixedLength':30})
    assert result.n == 30 and result.ishappy
    with pytest.raises(ValueError, match='happinessCheck:strictCheck'):
        f.compose(jnp.sin, pref={'happinessCheck':'strict'})
    with pytest.raises(ValueError, match='TRIGTECH:refine'):
        f.compose(jnp.sin, pref={'refinementFunction':lambda x:x})


def test_public_binary_and_explicit_periodic_preferences():
    f = cj.chebfun(lambda x: jnp.sin(x), domain=[-jnp.pi,jnp.pi], trig=True)
    g = cj.chebfun(lambda x: jnp.cos(x), domain=[-jnp.pi,jnp.pi], trig=True)
    h = f.compose(lambda x,y: jnp.exp(x+y), g)
    assert isinstance(h.funs[0].tech, Trigtech)
    x = jnp.linspace(-3.1,3.1,91)
    assert jnp.max(jnp.abs(h(x)-jnp.exp(jnp.sin(x)+jnp.cos(x)))) < 4e-14
    with pytest.warns(UserWarning, match='compose:convfail'):
        capped = f.compose(lambda y:jnp.cos(80*y), pref={'maxLength':32})
    assert capped.funs[0].tech.n == 32 and not capped.funs[0].tech.ishappy


def test_single_column_binary_does_not_outer_broadcast():
    f = Trigtech.from_function(lambda x:jnp.sin(jnp.pi*x))
    g = Trigtech.from_function(lambda x:jnp.cos(jnp.pi*x)[:,None])
    h = f.compose(jnp.add,g)
    assert h.coeffs.ndim == 2 and h.coeffs.shape[1] == 1


@pytest.mark.parametrize("scale", [None, [], jnp.asarray([])])
def test_source_empty_data_vscale_defaults_zero(scale):
    # @trigtech/parseDataInputs explicitly defaults isempty(data.vscale).
    f = Trigtech.from_function(lambda x: jnp.cos(jnp.pi*x))
    actual = f.compose(jnp.sin, data={'vscale':scale, 'hscale':7})
    expected = f.compose(jnp.sin, data={'vscale':0, 'hscale':1})
    assert actual.n == expected.n
    assert jnp.array_equal(actual.coeffs,expected.coeffs)


def test_empty_third_operand_is_unary():
    f = Trigtech.from_function(lambda x: jnp.cos(jnp.pi*x))
    actual = f.compose(jnp.sin, Trigtech.empty())
    expected = f.compose(jnp.sin)
    assert actual.n == expected.n
    assert jnp.array_equal(actual.coeffs,expected.coeffs)


def test_nested_complex_values_and_data_row_scale():
    f = Trigtech.from_function(lambda x: jnp.sin(jnp.pi*x))
    g = f.compose(lambda y: jnp.exp(20j*y))
    x = jnp.linspace(-.97,.97,123)
    assert not g.is_real and g.ishappy
    assert jnp.max(jnp.abs(g(x)-jnp.exp(20j*jnp.sin(jnp.pi*x)))) < 1e-13
    two = Trigtech.from_function(lambda x:jnp.stack([jnp.sin(jnp.pi*x),jnp.cos(jnp.pi*x)],axis=-1))
    row = two.compose(jnp.sin,data={'vscale':jnp.asarray([[2.,3.]])})
    flat = two.compose(jnp.sin,data={'vscale':jnp.asarray([2.,3.])})
    assert jnp.array_equal(row.coeffs,flat.coeffs)
