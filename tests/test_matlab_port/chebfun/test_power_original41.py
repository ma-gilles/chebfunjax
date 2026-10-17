"""Literal default test_power.m, pinned7574c77; captured seed6178 operands.

The existing authored/adapted files remain separate. All41 expressions,
including the local normest extrapolation formula and inherited tolerance
in40/41, retain source operands and bounds. Fixture sites are captured binary64
words, not a claim of general RNG parity.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj

EPS = jnp.finfo(jnp.float64).eps
DATA = json.loads((Path(__file__).parent/'fixtures/power_original41_sites_native.json').read_text())


def sites(name):
    words = jnp.asarray([int(x,16) for x in DATA[name]],dtype=jnp.uint64)
    x = words.view(jnp.float64)
    return x.reshape((100,100)).T if name == 'vector' else x


def infnorm(a):
    a = jnp.asarray(a)
    return jnp.max(jnp.abs(a)) if a.ndim <= 1 else jnp.max(jnp.sum(jnp.abs(a),axis=1))


def normest(f, domain=False):
    return infnorm(f(sites('normest_domain' if domain else 'normest')))


def scale(f):
    v = f.vscale
    return jnp.max(jnp.asarray(v() if callable(v) else v))


def array3(x):
    return jnp.stack((jnp.sin(x),jnp.cos(x),1j*jnp.exp(x)),axis=-1)


def expected3(x,p):
    return (jnp.stack((jnp.sin(x)**2,jnp.cos(x)**2,-jnp.exp(2*x)),axis=-1)
            if p == 2 else jnp.stack((jnp.sin(x)**3,jnp.cos(x)**3,-1j*jnp.exp(3*x)),axis=-1))


@pytest.mark.parametrize('clause',range(1,22))
def test_literal_regular(clause):
    if clause <= 4:
        f = cj.chebfun(jnp.sin)
        p = clause-1
        g = f**p
        h = 1 if p == 0 else f if p == 1 else cj.chebfun(lambda x:jnp.sin(x)**p)
        error,bound = normest(g-h),10*EPS
    elif clause <= 12:
        f = cj.chebfun(array3)
        fq = cj.quasimatrix(f)
        p = (clause-5)//2
        g = f**p
        if clause%2 == 0:
            gq = fq**p
            error = normest(gq-g)
            bound = 10*EPS if clause == 10 else EPS*scale(g)
        else:
            h = 1 if p == 0 else f if p == 1 else cj.chebfun(lambda x:expected3(x,p))
            assert min(g.size()) == 3
            error = normest(g-h)
            bound = EPS if p < 2 else 10*scale(h)*EPS
    elif clause <= 18:
        f = cj.chebfun(jnp.sin if clause <= 14 else array3)
        base = 1 if clause in (13,15,16) else 2j
        g = base**f
        if clause in (16,18):
            fq = cj.quasimatrix(f)
            h = (cj.chebfun(lambda x:2j**jnp.sin(x)) if clause == 16
                 else cj.chebfun(lambda x:jnp.stack((2j**jnp.sin(x),2j**jnp.cos(x),2j**(1j*jnp.exp(x))),axis=-1)))
            error,bound = normest(base**fq-g),10*EPS*scale(h)
        else:
            h = (1 if base == 1 else cj.chebfun(lambda x:2j**jnp.sin(x)) if clause == 14
                 else cj.chebfun(lambda x:jnp.stack((2j**jnp.sin(x),2j**jnp.cos(x),2j**(1j*jnp.exp(x))),axis=-1)))
            if clause >= 15:
                assert min(g.size()) == 3
            error,bound = normest(g-h),(100 if clause == 17 else 10)*EPS
    else:
        domain = (.1,2.)
        op = (lambda x:x) if clause == 19 else (lambda x:jnp.stack((x,jnp.exp(1j*x)),axis=-1))
        x = cj.chebfun(op,domain=domain)
        g = x**x
        h = (cj.chebfun(lambda t:t**t,domain=domain) if clause == 19 else
             cj.chebfun(lambda t:jnp.stack((t**t,jnp.exp(1j*t)**jnp.exp(1j*t)),axis=-1),domain=domain))
        if clause == 21:
            xq = cj.quasimatrix(x)
            error = normest(g-xq**xq,True)
        else:
            error = normest(g-h,True)
        bound = 10*EPS*scale(h)
    print('literal',clause,'error',error.tolist(),'bound',bound.tolist())
    assert error < bound


@pytest.mark.parametrize('clause',range(22,34))
def test_literal_finite(clause):
    dom = (-2.,7.)
    kw = {'domain':dom}
    x = sites('singular' if clause <= 24 or clause >= 31 else 'smooth')
    if clause == 22:
        pow,b = -1.5,2
        op = lambda x:(x-dom[0])**pow
        exact = lambda x:(x-dom[0])**(b*pow)
        kw['exps']=(pow,0)
        mult = 1e2
    elif clause == 23:
        pow,b = -.5,3
        op = lambda x:jnp.sin(100*x)*(x-dom[0])**pow
        exact = lambda x:jnp.sin(100*x)**b*(x-dom[0])**(b*pow)
        kw.update(exps=(pow,0),splitting=True)
        mult = 1e2
    elif clause == 24:
        pow,b = -1.5,.5
        op = lambda x:(jnp.sin(100*x)**2+1)*(x-dom[0])**pow
        exact = lambda x:jnp.sqrt(jnp.sin(100*x)**2+1)*(x-dom[0])**(pow/2)
        kw.update(exps=(pow,0),splitting=True)
        mult = 1e2
    elif clause <= 26:
        b = .6 if clause == 25 else -1.6
        op = lambda x:1/(1+25*x**2)
        exact = lambda x:1/((1+25*x**2)**b)
        kw['splitting']=True
        mult = 1e1 if clause == 25 else 1e3
    elif clause <= 28:
        b = .8 if clause == 27 else -.65
        op = lambda x:jnp.sin(10*x)
        exact = lambda x:jnp.sin(10*x).astype(jnp.complex128)**b
        kw['splitting']=True
        mult = 1e4 if clause == 27 else 2e4
    elif clause <= 30:
        b = 1.7 if clause == 29 else -.77
        op = lambda x:jnp.sin(3*x)+1j*jnp.cos(2*x)
        exact = lambda x:op(x)**b
        kw['splitting']=True
        mult = 1e2
    elif clause <= 32:
        b = .77
        op = (lambda x:(x-dom[0])**-1.2) if clause == 31 else (lambda x:jnp.sin(20*x)*(x-dom[0])**-1.2)
        exact = ((lambda x:(x-dom[0])**(-1.2*b)) if clause == 31
                 else (lambda x:jnp.sin(20*x).astype(jnp.complex128)**b*(x-dom[0])**(-1.2*b)))
        kw.update(exps=(-1.2,0),splitting=True)
        mult = 1e1 if clause == 31 else 1e3
    else:
        b = .69
        op = lambda x:(jnp.sin(20*x)+1j*jnp.cos(3*x))*(x-dom[0])**-1.2*(dom[1]-x)**-.49
        exact = lambda x:(jnp.sin(20*x)+1j*jnp.cos(3*x))**b*(x-dom[0])**(-1.2*b)*(dom[1]-x)**(-.49*b)
        kw.update(exps=(-1.2,-.49),splitting=True)
        mult = 1e3
    f = cj.chebfun(op,**kw)
    g = f**b
    expected = exact(x)
    error,bound = infnorm(g(x)-expected),mult*EPS*infnorm(expected)
    print('literal',clause,'error',error.tolist(),'bound',bound.tolist(),'pieces',len(g.funs))
    assert error < bound


@pytest.mark.parametrize('clause',range(34,39))
def test_literal_unbounded(clause):
    kw = {}
    if clause <= 35:
        kw['domain']=(-jnp.inf,jnp.inf)
        x=sites('whole')
        if clause == 34:
            op=lambda x:x**2*jnp.exp(-x**2)+2
            b,mult=.6,1e2
        else:
            op=lambda x:x**2*(1-jnp.exp(-x**2))+2
            b,mult=1.5,1e7
            kw['exps']=(2,2)
    else:
        kw['domain']=(1.,jnp.inf)
        x=sites('right')
        if clause == 36:
            op=lambda x:x*jnp.exp(-x)+3
            b,mult=2,1e1
        elif clause == 37:
            op=lambda x:.1+jnp.sin(10*x)/jnp.exp(x)
            b,mult=3,1e5
            kw['splitting']=True
        else:
            op=lambda x:x*(5+jnp.exp(-x**3))
            b,mult=-3,1e2
            kw['exps']=(0,1)
    f=cj.chebfun(op,**kw)
    g=f**b
    error,bound=infnorm(g(x)-op(x)**b),mult*EPS*scale(g)
    print('literal',clause,'error',error.tolist(),'bound',bound.tolist())
    assert error < bound


@pytest.fixture(scope='module')
def vector39():
    x=cj.chebfun('x',domain=(-1.,1.5))
    f=cj.Chebfun.horzcat(x,x+.5)
    powers=jnp.asarray([[-.1,.3]])
    g=f**powers
    tol=1e4*jnp.max(EPS*scale(g))
    return g,tol


def test_literal_39(vector39):
    g,tol=vector39
    xx=sites('vector')
    z=xx.astype(jnp.complex128)
    exact=jnp.concatenate((z**-.1,(z+.5)**.3),axis=1)
    error=infnorm(g(xx)-exact)
    print('literal39','error',error.tolist(),'bound',tol.tolist())
    assert error < tol


@pytest.mark.parametrize('clause',[40,41])
def test_literal_vector_powers(clause,vector39):
    _,tol=vector39
    x=cj.chebfun('x')
    if clause == 40:
        powers=jnp.arange(6,dtype=jnp.float64)[None,:]
        f=x**powers
        g=cj.chebfun(lambda t:jnp.asarray(t)[...,None]**powers[0])
    else:
        powers=jnp.asarray([[-2.,-.5,1.25,3.]])
        f=(1+x**2)**powers
        g=cj.chebfun(lambda t:(1+jnp.asarray(t)[...,None]**2)**powers[0])
    error=(f-g).norm(jnp.inf)
    print('literal',clause,'error',error.tolist(),'inherited_bound',tol.tolist())
    assert error < tol
