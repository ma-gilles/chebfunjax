"""ratinterp.m174-182 default domain, Chebfun commit7574c77."""
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


def test_chebfun_input_inherits_domain():
    f = cj.chebfun(lambda x:(x+2)/(x+1),domain=(2.,5.))
    p,q,evaluate,_,_ = cj.ratinterp(f,1,1,tol=1e-12,outputs=5)
    assert p.domain.breakpoints == q.domain.breakpoints == (2.,5.)
    x = jnp.asarray([2.2,3.7,4.8])
    np.testing.assert_allclose(evaluate(x),(x+2)/(x+1),rtol=0,atol=100*jnp.finfo(float).eps)


def test_piecewise_chebfun_uses_outer_endpoints():
    f = cj.chebfun(lambda x:(x+2)/(x+1),domain=(2.,3.,5.))
    p,_,evaluate,_,_ = cj.ratinterp(f,1,1,tol=1e-12,outputs=5)
    assert p.domain.breakpoints == (2.,5.)
    x = jnp.asarray([2.2,3.7,4.8])
    np.testing.assert_allclose(evaluate(x),(x+2)/(x+1),rtol=0,atol=100*jnp.finfo(float).eps)


def test_explicit_domain_remains_authoritative():
    f = cj.chebfun(lambda x:(x+2)/(x+1),domain=(2.,5.))
    p,_,evaluate,_,_ = cj.ratinterp(f,1,1,domain=(3.,4.),tol=1e-12,outputs=5)
    assert p.domain.breakpoints == (3.,4.)
    x = jnp.asarray([3.2,3.8])
    np.testing.assert_allclose(evaluate(x),(x+2)/(x+1),rtol=0,atol=100*jnp.finfo(float).eps)


def test_plain_callable_retains_default_interval():
    p,_,evaluate,_,_ = cj.ratinterp(lambda x:2+x,1,0,outputs=5)
    assert p.domain.breakpoints == (-1.,1.)
    np.testing.assert_allclose(evaluate(jnp.asarray([-.8,.3])),[1.2,2.3],rtol=0,atol=100*jnp.finfo(float).eps)


def test_unrelated_callable_domain_attribute_is_ignored():
    class Callable:
        domain = (2.,5.)
        def __call__(self,x):
            return 2+x
    p,_,_,_,_ = cj.ratinterp(Callable(),1,0,outputs=5)
    assert p.domain.breakpoints == (-1.,1.)
