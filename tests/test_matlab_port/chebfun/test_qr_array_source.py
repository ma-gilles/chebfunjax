"""Native QR1--15 and observable array-return controls; source pin7574c77.

QR13/14 occur twice in MATLAB with identical inputs; executed once here.
The point-assignment setup uses restrict + stored point values because Python
has no continuous-index assignment syntax. No source bound is changed.
"""
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

from .test_qr_column_source import EPS, mark


def array_fun(op, domain, cls=Chebtech2):
    dom = Domain(domain)
    return Chebfun(funs=[_Piece(cls.from_function(
        lambda t, a=a, b=b: op(b*(t+1)/2+a*(1-t)/2)), (a,b))
        for a,b in zip(dom.breakpoints[:-1], dom.breakpoints[1:])], domain=dom)


def require_array(Q, n):
    assert isinstance(Q, Chebfun)
    assert not hasattr(Q, "cols")
    assert Q.n_columns == n
    assert Q.size() == (float("inf"), n)
    assert not Q.is_transposed


@pytest.mark.parametrize("breakpoint", [False, True])
def test_source01_to08(breakpoint):
    mark("qr.complex.construction", breakpoint=breakpoint)
    A = array_fun(lambda x: jnp.stack([x,1j*x,1+0*x,1+1j+0*x,(2-1j)*x],axis=-1), (0.,1.))
    Aq = Quasimatrix(A.mat2cell(),A.domain)
    if breakpoint:
        # Native A(.5,:)=A(.5,:) inserts a breakpoint without changing values.
        midpoint = A(.5)
        A = A.restrict((0.,.5,1.))
        A = A.set_point_values(A.point_values.at[1].set(midpoint))
        # Aq(.5,1) inserts it only into the first quasimatrix column.
        cols = list(Aq.cols)
        cols[0] = cols[0].restrict((0.,.5,1.))
        cols[0] = cols[0].set_point_values(
            cols[0].point_values.at[1].set(midpoint[0]))
        Aq = Quasimatrix(cols, Aq.domain)
    mark("qr.complex.rank")
    assert int(A.rank()) == 2
    mark("qr.complex.qr")
    Q,R = A.qr()
    require_array(Q,5)
    mark("qr.complex.cond")
    assert abs(float(Q.cond())-1) < 1e-13
    mark("qr.complex.reconstruction")
    assert float((A-Q@R).norm()) < 1e-13
    Q2,R2 = Aq.qr()
    require_array(Q2,5)
    assert float((Q-Q2).normest()) + float(jnp.linalg.norm(R-R2,ord=2)) < 1e-13


@pytest.mark.parametrize("cls,slot", [(Chebtech2,9),(Chebtech1,13)])
def test_source09_10_13_14(cls,slot):
    f = array_fun(lambda x:jnp.stack([jnp.sin(x),jnp.cos(x),jnp.exp(x)],axis=-1),(-1.,0.,1.),cls)
    mark("qr.piecewise.qr", slot=slot)
    Q,R = f.qr()
    require_array(Q,3)
    assert all(isinstance(p.tech,cls) for p in Q.funs)
    assert float((f-Q@R).norm()) < 10*float(f.vscale)*EPS
    if slot == 13:
        assert isinstance(f.funs[0].tech,Chebtech1)
        assert float(jnp.linalg.norm(Q.H@Q-jnp.eye(3),ord="fro")) < 10*float(Q.vscale)*EPS
    Q2,R2 = Quasimatrix(f.mat2cell(),f.domain).qr()
    require_array(Q2,3)
    assert float((Q-Q2).normest()) + float(jnp.linalg.norm(R-R2,ord=2)) < 1e-13


def test_source11_12():
    f = cj.chebfun(lambda x:1+0*x,domain=(-1.,0.,1.))
    Q,R = f.qr()
    require_array(Q,1)
    assert float((f-Q@R).norm()) < 10*float(f.vscale)*EPS
    assert abs(complex(jnp.squeeze(Q.H@Q))-1) < 10*float(Q.vscale)*EPS
    Q2,R2 = Quasimatrix([f],f.domain).qr()
    require_array(Q2,1)
    assert float((Q-Q2).normest()) + float(jnp.linalg.norm(R-R2,ord=2)) < 1e-13


def test_source15():
    Q,R = cj.chebfun(lambda x:0*x,domain=(0.,3.)).qr()
    require_array(Q,1)
    assert float(R[0,0]) == 0
    assert abs(complex(jnp.squeeze(Q.H@Q))-1) < 1e-13


def test_normest_representation():
    # Distinct maxima in columns make source outer-reset semantics observable.
    f = array_fun(lambda x:jnp.stack([3+0*x,1+0*x],axis=-1),(-1.,0.,1.))
    assert float(f.normest()) == 6
    assert float(Quasimatrix(f.mat2cell(),f.domain).normest()) == 2
    assert float(f.set_point_values(jnp.full_like(f.point_values,100)).normest()) == 6


def test_array_svd_product():
    f = array_fun(lambda x:jnp.stack([1+0*x,x,x*x],axis=-1),(-1.,1.))
    U,S,V = f.svd()
    require_array(U,3)
    assert isinstance(U@jnp.diag(S)@jnp.conj(V.T),Chebfun)
    assert float((f-U@jnp.diag(S)@jnp.conj(V.T)).norm()) < 1e-13
    assert float(jnp.linalg.norm(U.H@U-jnp.eye(3))) < 1e-13


def test_row_complex_svd():
    f = array_fun(lambda x:jnp.stack([1+1j*x,x+2j*x*x],axis=-1),(-1.,1.))
    U,S,V = f.H.svd()
    require_array(V,2)
    residual = f.H - U@jnp.diag(S)@V.H
    assert float(residual.norm()) < 1e-13


def test_product_retains_panel_arithmetic():
    f = array_fun(lambda x:jnp.stack([1+0*x,x,x*x],axis=-1),(-1.,0.,1.))
    Q,R = f.qr()
    product = Q@R
    require_array(product,3)
    for p,q in zip(product.funs,Q.funs):
        assert jnp.array_equal(p.tech.coeffs,(q.tech@R).coeffs)


def test_quasimatrix_consumers():
    f = array_fun(lambda x:jnp.stack([1+0*x,x],axis=-1),(-1.,1.))
    A = Quasimatrix(f.mat2cell(),f.domain)
    assert A.rank() == 2
    assert abs(float(A.cond())-float(f.cond())) < 1e-13
    basis = A.orth()
    assert isinstance(basis,Quasimatrix)  # Retained compatibility API.
    gram = basis.H@basis
    assert float(jnp.linalg.norm(gram-jnp.eye(2))) < 1e-13
    target = cj.chebfun(lambda x:1+2*x)
    assert float(jnp.max(jnp.abs(A.pinv(target)-jnp.array([1.,2.])))) < 1e-13
