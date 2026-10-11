"""Literal bounded clauses from tests/misc/test_jacpts.m, native 7574c77.
Python vectors adapt MATLAB row/column vectors to one dimension.
Large symmetric clauses 23,26,27 are in test_jacpts_large_source.py;
clause 29 is in test_jacpts_asy_source.py. Clause 24 is disabled upstream.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.utils.quadrature import jacpts


@pytest.fixture(scope="module")
def rules():
    return {(n,m):jacpts(n,a,b,interval=(0,10) if m else None,bary=True)
            for n,a,b in [(42,-.1,.3),(251,-.7,1.3)] for m in [False,True]}

@pytest.mark.parametrize("clause",list(range(1,23))+[25,28,30,31])
def test_native_clause(clause,rules):
    tol=1e-12
    eps=jnp.finfo(jnp.float64).eps
    if clause<=20:
        n=42 if clause<=10 else 251
        c=clause if n==42 else clause-10
        x,w,v=rules[n,c>=7]
        if c in (1,2):
            assert x.shape==w.shape==v.shape==(n,)
        elif c==3:
            u,z=(.363593965943934,.670376374709129) if n==42 else (5.693053859952719,5.218632704956658)
            bound=tol if n==42 else 2*tol
            assert abs(w@x-u)<bound
            assert abs(w@(x*x)-z)<bound
        elif c in (4,5,6):
            refs=(.912883347814032,.046661910947553,.320696510075909) if n==42 else (-.893103435898983,1.962534523788093e-4,.042040484981163)
            assert abs((x,w,v)[c-4][36]-refs[c-4])<tol
        elif c==7:
            u,z=(81.519974175437831,585.9248143859594) if n==42 else (859.7954446702054,7881.458242810217)
            k=1 if n==42 else 20
            assert abs(w@x-u)<10*k*tol
            assert abs(w@(x*x)-z)<100*k*tol
        else:
            refs=(9.702339316456870,.279566831611687,-.248832970298009) if n==42 else (.562894092705871,.002830847963541,-.045147639085330)
            assert abs((x,w,v)[c-8][37]-refs[c-8])<tol
    elif clause in (21,22):
        x,w=jacpts(1,1,2)
        assert abs((x[0]-.2) if clause==21 else (w[0]-4/3))<tol
    elif clause==25:
        x,w=jacpts(114,-.5,0)
        assert jnp.sum(w)-2*jnp.sqrt(2.)<10*eps
    elif clause==28:
        x,w=jacpts(25,.3,.5,'GW')
        assert abs(jnp.sum(w)-1.652030414885670)<100*eps
    elif clause==30:
        x,w=jacpts(1,.14,1.1,[-2,3])
        assert jnp.linalg.norm(w-14.285331512144)<1e-12
    elif clause==31:
        x,w=jacpts(1,2,2,[-2,3])
        assert jnp.linalg.norm(w-625/6)<1e-13
