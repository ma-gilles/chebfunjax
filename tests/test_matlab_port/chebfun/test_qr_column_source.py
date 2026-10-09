"""Literal column solve slots5/6 and QR slots9/13/15; pin7574c77.

New separate tests retain the previous port unchanged. Python columns adapt
native arrays; all native numerical inputs and acceptance bounds are retained.
"""
import json
import os
import time
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece, mldivide
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.polynomials import chebpoly, legpoly

EPS = float(jnp.finfo(jnp.float64).eps)


def mark(stage, **data):
    folder = os.environ.get("VANDERMONDE_STAGE_DIR")
    if folder:
        p = Path(folder) / "stages.jsonl"
        with p.open("a") as f:
            f.write(json.dumps({"stage": stage, "time": time.monotonic(), **data}) + "\n")


def polynomial(c):
    return Chebfun(funs=[_Piece(Chebtech2.from_coeffs(c), (-1., 1.))],
                   domain=Domain((-1., 1.)))


def test_mldivide_source05():
    mark("source05.construction.begin")
    T = [polynomial(chebpoly(k)).restrict((-1., -.5, 0., .5, 1.)) for k in range(4)]
    L = [polynomial(legpoly(k)).restrict((-1., 0., 1.)) for k in range(4)]
    mark("source05.solve.begin")
    X = mldivide(Quasimatrix(T, T[0].domain), Quasimatrix(L, L[0].domain))
    C = jnp.diag(jnp.array([1., 1., .75, .625])).at[0, 2].set(.25).at[1, 3].set(.375)
    error = float(jnp.linalg.norm(X-C, ord=jnp.inf))
    mark("source05.solve.end", error=error, bound=10*EPS)
    assert error < 10*EPS


def test_mldivide_source06():
    mark("source06.construction.begin")
    x = cj.chebfun("x")
    A = [(1-4*abs(x-(-1+j/4))).maximum(0) for j in range(7)]
    mark("source06.solve.begin", pieces=[len(a.funs) for a in A])
    u = mldivide(Quasimatrix(A, x.domain), x)
    expected = jnp.array([-.999851249504164, -.750297500991670,
                          -.498958746529156, -.253867512891710,
                          .014428798095994, .196152320507735,
                          .700961919873066])
    error = float(jnp.linalg.norm(u-expected, ord=jnp.inf))
    mark("source06.solve.end", error=error, bound=1e3*EPS)
    assert error < 1e3*EPS


@pytest.mark.parametrize("cls,slot", [(Chebtech2, 9), (Chebtech1, 13)])
def test_qr_source_piecewise(cls, slot):
    mark(f"source{slot}.construction.begin")
    funcs = [jnp.sin, jnp.cos, jnp.exp]
    dom = Domain((-1., 0., 1.))
    F = Chebfun(funs=[_Piece(cls.from_function(
        lambda t, a=a, b=b: jnp.stack([
            f(b*(t+1)/2+a*(1-t)/2) for f in funcs], axis=-1)), (a,b))
        for a,b in zip(dom.breakpoints[:-1], dom.breakpoints[1:])], domain=dom)
    cols = F.mat2cell()
    mark(f"source{slot}.qr.begin")
    Q,R = F.qr()
    mark(f"source{slot}.product.begin", Rshape=list(R.shape))
    assert isinstance(Q, Chebfun)
    product = Q @ R
    assert isinstance(product, Chebfun)
    residual = F - product
    error = float(residual.norm())
    bound = 10*float(F.vscale)*EPS
    mark(f"source{slot}.norm.end", error=error, bound=bound)
    assert error < bound
    mark(f"source{slot+1}.quasi_qr.begin")
    Q2,R2 = Quasimatrix(cols,dom).qr()
    assert isinstance(Q2, Chebfun)
    assert float((Q-Q2).normest()) + float(jnp.linalg.norm(R-R2,ord=2)) < 1e-13
    if slot == 13:
        assert all(isinstance(p.tech, Chebtech1) for c in Q.mat2cell() for p in c.funs)
        gram = jnp.stack([jnp.stack([a.inner(b) for b in Q.mat2cell()]) for a in Q.mat2cell()])
        assert float(jnp.linalg.norm(gram-jnp.eye(3), ord="fro")) < 10*float(Q.vscale)*EPS


def test_qr_source15_zero():
    f = cj.chebfun(lambda x: 0*x, domain=(0.,3.))
    Q,R = Quasimatrix([f],f.domain).qr()
    assert float(R[0,0]) == 0
    assert abs(float(Q.mat2cell()[0].inner(Q.mat2cell()[0]))-1) < 1e-13


@pytest.mark.parametrize("cls", [Chebtech1,Chebtech2])
def test_physical_domain_polynomial_qr(cls):
    # Independent source delegation control on a noncanonical physical domain.
    dom = Domain((2.,5.))
    cols = [Chebfun(funs=[_Piece(cls.from_coeffs(chebpoly(k)),(2.,5.))],domain=dom) for k in range(3)]
    mark("physical.qr.begin", technology=cls.__name__)
    Q,R = Quasimatrix(cols,dom).qr()
    assert all(type(c.funs[0].tech) is cls for c in Q.mat2cell())
    gram = jnp.stack([jnp.stack([a.inner(b) for b in Q.mat2cell()]) for a in Q.mat2cell()])
    assert float(jnp.max(jnp.abs(gram-jnp.eye(3)))) < 32*EPS
    reconstructed = Q @ R
    assert max(float((a-b).norm()) for a,b in zip(cols,reconstructed.mat2cell())) < 32*EPS
