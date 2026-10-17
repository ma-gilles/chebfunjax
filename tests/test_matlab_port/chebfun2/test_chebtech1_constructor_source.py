"""First-kind grid/factor controls. Provenance: constructor.m, pin 7574c77."""
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebfun2d.separable_approx import _tech_grid_refine
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1
from chebfunjax.utils.quadrature import chebpts


def test_first_kind_grid_refinement_and_factors():
    samples = []
    def op(x, y):
        samples.append((np.asarray(x).copy(), np.asarray(y).copy()))
        return jnp.cos(30*x) * jnp.cos(5*y)
    pref = ChebfunPref()
    pref.tech = Chebtech1
    f = chebfun2(op, pref=pref)
    grids = [(x,y) for x,y in samples if x.shape == (17,17)]
    assert grids
    np.testing.assert_array_equal(grids[0][0][0], np.asarray(chebpts(17, kind=1)))
    assert any(51 in x.shape or 51 in y.shape for x,y in samples)
    n, nesting = _tech_grid_refine(17, "cheb1")
    assert n == 51 and nesting == slice(1, None, 3)
    np.testing.assert_allclose(np.asarray(chebpts(n,kind=1))[nesting],
                               np.asarray(chebpts(17,kind=1)), rtol=0, atol=3e-16)
    assert all(isinstance(t, Chebtech1) for t in f.approx.cols + f.approx.rows)
    x=jnp.linspace(-1,1,37)
    y=jnp.linspace(1,-1,37)
    np.testing.assert_allclose(f(x,y), jnp.cos(30*x)*jnp.cos(5*y), rtol=0, atol=100*np.finfo(float).eps)


def test_first_kind_zero_factors():
    pref=ChebfunPref()
    pref.tech=Chebtech1
    f=chebfun2(lambda x,y: jnp.zeros_like(x+y), pref=pref)
    assert all(isinstance(t,Chebtech1) for t in f.approx.cols+f.approx.rows)
    assert float(f(.2,.3)) == 0


def test_default_second_kind_and_trig_flags():
    from chebfunjax.tech.chebtech import Chebtech2
    from chebfunjax.tech.trigtech import Trigtech
    f = chebfun2(lambda x, y: jnp.cos(x*y))
    assert all(isinstance(t, Chebtech2) for t in f.approx.cols+f.approx.rows)
    assert abs(float(f(0., 0.))-1) < 100*np.finfo(float).eps
    g = chebfun2(lambda x,y: jnp.sin(jnp.pi*x)*jnp.cos(jnp.pi*y), trig=True)
    assert all(isinstance(t, Trigtech) for t in g.approx.cols+g.approx.rows)
    assert abs(float(g(.3, -.7))-np.sin(np.pi*.3)*np.cos(np.pi*-.7)) < 10000*np.finfo(float).eps


def test_preference_sample_test_disabled():
    shapes=[]
    def op(x,y):
        shapes.append(x.shape)
        return x+y
    pref=ChebfunPref()
    pref.tech=Chebtech1
    pref.cheb2Prefs.sampleTest=False
    # constructor.m uses selected tech defaults rather than these overrides.
    pref.techPrefs.minSamples=65
    pref.techPrefs.maxLength=9
    f=chebfun2(op,pref=pref)
    assert (100,1) not in shapes
    assert (17,17) in shapes
    assert (65,65) not in shapes
    assert abs(float(f(.3,.2))-.5) < 100*np.finfo(float).eps
