"""All ten source predicates from diskfun/test_helmholtz.m, Chebfun7574c77.

Preserves disk L2 norms, source grids and tolerances. No xfail/sampling
substitute. Function arguments use Python theta/radius convention.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.diskfun.diskfun import Diskfun

TOL = 2e3*jnp.finfo(jnp.float64).eps


def test_source_1_2_poisson_equality():
    exact = Diskfun.from_function(lambda t,r: jnp.exp(-r*jnp.cos(t)-r**2*jnp.sin(2*t)))
    rhs = exact.laplacian()
    bc = lambda t: jnp.exp(-jnp.cos(t)-jnp.sin(2*t))
    u = Diskfun.poisson(rhs, bc, m=100)
    v = Diskfun.helmholtz(rhs, 0., bc, m=100)
    assert float((v-exact).norm()) < 2e4*TOL
    assert float((v-u).norm()) < 2e4*TOL


@pytest.mark.parametrize('k', [.05, .25, 1., float(jnp.pi), 7.])
def test_source_3_to_7_wavenumbers(k):
    def value(t,r):
        x,y = r*jnp.cos(t),r*jnp.sin(t)
        return jnp.cos(5*(x+y)-.2)+jnp.sin(3*x*y)
    exact = Diskfun.from_function(value)
    rhs = exact.laplacian()+k*k*exact
    actual = Diskfun.helmholtz(rhs,k,lambda t:exact(t,jnp.ones_like(t)),m=257,n=256)
    error = float((exact-actual).norm())
    print('source wavenumber',k,'error',error,'bound',float(5e4*TOL))
    assert error < 5e4*TOL


def test_source_8_coefficient_rhs():
    k=jnp.sqrt(2.)
    exact=Diskfun.from_function(lambda t,r:jnp.cos(r**5*jnp.sin(5*t))-r**2)
    rhs=(exact.laplacian()+k*k*exact).coeffs2()
    actual=Diskfun.helmholtz(rhs,k,lambda t:exact(t,jnp.ones_like(t)),m=100)
    assert float((actual-exact).norm()) < TOL


def test_source_9_callable_rhs():
    rhs=lambda t,r:3*jnp.cos(r*jnp.cos(t))+jnp.cos(r*jnp.sin(t))
    exact=Diskfun.from_function(rhs)/3
    actual=Diskfun.helmholtz(rhs,2.,lambda t:exact(t,jnp.ones_like(t)),m=100)
    assert float((actual-exact).norm()) < TOL


def test_source_10_laplacian_eigenfunction():
    lam=5.52007811028631**2
    k=jnp.sqrt(lam+1)
    rhs=Diskfun.harmonic(0,2)
    actual=Diskfun.helmholtz(rhs,k,lambda t:0*t,m=100)
    assert float((actual-rhs).norm()) < TOL


class TestHelmholtzComplexK:
    """Complex (imaginary) K: the screened-Poisson shift used by BDF
    timestepping (disk/HeatEqn example).  K = i*k gives real K^2 = -k^2;
    the float(K) truncation previously silently dropped the shift."""

    def test_bdf1_step_decays_harmonic(self):
        import jax.numpy as jnp
        import numpy as np

        from chebfunjax.diskfun.diskfun import Diskfun
        u0 = Diskfun.harmonic(4, 4)
        lam = 17.615966049804832  # 4th positive root of J_4 in [16,18]
        alpha = 1.0 / lam ** 2
        dt = 0.01
        K = np.sqrt(1 / (dt * alpha)) * 1j
        u1 = Diskfun.helmholtz(u0 * (K ** 2), K, lambda t: 0 * t, 40, 40)
        # One BDF1 step of du/dt = alpha*lap(u) on an eigenfunction:
        # u1 = u0 / (1 + dt*alpha*lam^2) = u0 / 1.01 (alpha = 1/lam^2).
        th = jnp.asarray(np.linspace(-3, 3, 7))
        r = jnp.asarray(np.full(7, 0.55))
        ratio = np.asarray(u1(th, r)) / np.asarray(u0(th, r))
        assert float(np.max(np.abs(ratio - 1.0 / 1.01))) < 1e-8
