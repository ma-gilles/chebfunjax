"""Original autoVectorize execution setups1–5 and native syntax clause6.

Python */** spell MATLAB's vectorized pointwise result. These execution tests
cannot establish anonymous-function rewriting or native ode113/coupled-Newton
parity. Second-order IVPs retain the existing SciPy adapter; coupled5 retains
its existing system solver. Previous endpoint checks are separate legacy controls.

Provenance
----------
MATLAB source: tests/chebop/test_autoVectorize.m, @chebop/mldivide.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.operators.chebop import Chebop

DOMAIN = (0., 1.)


def _coefficient_operator(scale):
    x = chebfun(lambda x: x, domain=DOMAIN)
    f = (4*x).sin().exp()
    if scale == 1.:
        return lambda x, u: u.diff(2)-f*(1-u**2)*u.diff()+u
    return lambda x, u: u.diff(2)-scale*f*(1-u**2)*u.diff()+u


def test_original_1_scalar_ivp():
    op = Chebop(lambda u: u.diff(2)-5*(1-u**2)*u.diff()+u, DOMAIN)
    op.lbc = lambda u: [u-2, u.diff()]
    op.solve(0.)


def test_original_2_scalar_ivp_assigned():
    # MATLAB chebop(d) maps to Python's domain keyword; op set afterwards.
    op = Chebop(domain=DOMAIN)
    op.op = _coefficient_operator(1.)
    op.lbc = lambda u: [u-2, u.diff()]
    op.solve(0.)


def test_original_3_scalar_bvp():
    op = Chebop(lambda u: u.diff(2)-5*(1-u**2)*u.diff()+u, DOMAIN)
    op.bc = lambda x, u: [u(0.)-2, u(1.)-3]
    op.solve(0.)


def test_original_4_scalar_bvp_assigned():
    op = Chebop(domain=DOMAIN)
    op.op = _coefficient_operator(.2)
    op.bc = lambda x, u: [u(0.)-2, u(1.)-3]
    op.solve(0.)


def test_original_5_coupled_bvp():
    op = Chebop(lambda x, u, v: [u.diff(2)-5*u**2*v.diff(),
                                v.diff(2)+u.diff()/(v+2)**2], DOMAIN)
    op.bc = lambda x, u, v: [u(0.)-2, u(1.)-3, v(0.)-1,
                            v(1.)*u.diff()(1.)-17.909]
    u0 = Chebfun.from_values(jnp.asarray([2., 1.5059, 1.2210, 2.0354, 3.]), DOMAIN)
    v0 = Chebfun.from_values(jnp.asarray([1., 2.]), DOMAIN)
    op.init = [u0, v0]
    op.solve(0.)


def test_original_6_vectorize_disabled():
    fun = Chebop.nativeAnonymous('@(u) diff(u,2)-u*diff(u)')
    op = Chebop(domain=DOMAIN)
    op.vectorize = False
    op.op = fun
    op.bc = Chebop.nativeAnonymous('@(x,u) [u(0)-2; u(1)-3]')
    with pytest.raises(ValueError) as caught:
        op.solve(0.)
    assert caught.value.identifier == 'CHEBFUN:ADCHEBFUN:mtimes:dims'
