"""Complex residues from pinned prztrig.m and original AAAtrig predicates.

Provenance
----------
MATLAB source: prztrig.m; tests/chebfun/test_aaatrig.m, predicates 16 and 17.
Chebfun commit: 7574c77.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.utils.aaa import _prztrig_np, aaatrig


def test_two_support_odd_closed_form():
    # q=exp(i*z): r(z)=(-3*q+5)/(-q+3). Its only finite
    # pole is -i*log(3), with residue -4i/3, independently of SVD.
    poles, residues, _ = _prztrig_np(
        jnp.array([0., jnp.pi]), jnp.array([1., 2.]),
        jnp.array([1., 2j]), 'odd')
    poles, residues = jnp.asarray(poles), jnp.asarray(residues)
    target = -1j*jnp.log(3.)
    distances = jnp.abs(jnp.exp(1j*poles)-3.)
    k = int(jnp.argmin(distances))
    assert abs(jnp.imag(poles[k])-jnp.imag(target)) < 1e-12
    assert abs(residues[k]+4j/3) < 1e-12


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_public_complex_poles_analytic_residues(form):
    z = jnp.linspace(0.1, 2*jnp.pi-0.1, 200)
    values = 1/(2-jnp.cos(z))
    r, poles, residues, *_ = aaatrig(values, z, form=form, cleanup=False)
    assert jnp.max(jnp.abs(r(z)-values)) < 1e-11
    for sign in [-1, 1]:
        target = sign*1j*jnp.arccosh(2.)
        distance = jnp.abs(jnp.exp(1j*poles)-jnp.exp(1j*target))
        k = int(jnp.argmin(distance))
        assert distance[k] < 1e-10
        assert abs(residues[k]-1/jnp.sin(target)) < 1e-10


def test_native_residue_predicate16():
    x = jnp.linspace(-1.337, 2., 537)
    _, poles, residues, *_ = aaatrig(
        (1/jnp.tan(x/2))*jnp.exp(1j*jnp.tan(x/2)), x)
    selected = jnp.abs(jnp.mod(jnp.real(poles), 2*jnp.pi)) < 1e-8
    assert int(jnp.sum(selected)) > 0
    assert jnp.all(jnp.abs(residues[selected]-2) < 1e-10)


def test_native_residue_predicate17():
    x = jnp.linspace(-1.337, 2., 537)
    _, poles, residues, *_ = aaatrig(
        jnp.tan(x/2)*jnp.log(5+jnp.sin(x)), x)
    selected = jnp.abs(poles-jnp.pi) < 1e-8
    assert int(jnp.sum(selected)) > 0
    assert jnp.all(jnp.abs(residues[selected]+2*jnp.log(5.)) < 1e-8)
