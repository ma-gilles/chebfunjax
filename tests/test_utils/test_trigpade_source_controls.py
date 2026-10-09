"""Independent trigpade API and coefficient controls; not native RNG substitutes."""
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.utils.trigrational import _laurent_approx


def test_nonperiodic_source_identifier():
    with pytest.raises(ValueError) as caught:
        cj.trigpade(cj.chebfun(lambda x:x),1,1)
    assert caught.value.identifier == "CHEBFUN:CHEBFUN:trigpade:trig"


def test_even_coefficient_count_source_rejection():
    f = cj.chebfun(jnp.asarray([1.,2.,3.,4.]),coeffs=True,trig=True)
    with pytest.raises(ValueError,match="c must have odd length"):
        cj.trigpade(f,1,1)


def test_zero_denominator_at_origin_source_identifier():
    # n=1,m=0, c1!=0,c0=0 gives matrix[c1,c0] and null vector[0,1].
    with pytest.raises(ValueError) as caught:
        _laurent_approx(jnp.asarray([0.,0.,1.]),0,1,1,1e-14)
    assert caught.value.identifier == "CHEBFUN:TRIGPADE:laurent_approx"


def test_n_zero_preserves_resolved_function_and_half_decomposition():
    f = cj.chebfun(lambda x:jnp.sin(jnp.pi*x),trig=True)
    p,q,r,s,t,u,v = cj.trigpade(f,1,0)
    assert p is f
    x = jnp.asarray([-1.,-.47,0.,.26,1.])
    assert float(jnp.max(jnp.abs(r(x)-jnp.sin(jnp.pi*x)))) < 1e-14
    assert float(jnp.max(jnp.abs(s(x)-f(x)/2))) < 1e-14
    assert float(jnp.max(jnp.abs(u(x)-f(x)/2))) < 1e-14


def test_asymmetric_complex_coefficients_keep_order():
    # Finite Laurent series with non-Hermitian coefficients; n0 identity.
    c = jnp.asarray([.1+2j,3.,.7-1j])
    f = cj.chebfun(c,coeffs=True,trig=True)
    p,q,r,*_ = cj.trigpade(f,1,0)
    x = jnp.asarray([-.91,-.37,.2,.86])
    expected = c[0]*jnp.exp(-1j*jnp.pi*x)+c[1]+c[2]*jnp.exp(1j*jnp.pi*x)
    assert float(jnp.max(jnp.abs(r(x)-expected))) < 1e-13


def test_real_input_does_not_discard_material_imaginary_part(monkeypatch):
    import importlib

    module = importlib.import_module("chebfunjax.utils.trigrational")
    monkeypatch.setattr(module,"_laurent_pade",lambda *args:
        tuple(jnp.asarray(v,dtype=jnp.complex128) for v in ([1j],[1.],[0.],[1.])))
    f = cj.chebfun(lambda x:jnp.ones_like(x),trig=True)
    with pytest.warns(RuntimeWarning,match="imaginary part not negligible"):
        p,q,r,*_ = cj.trigpade(f,1,1)
    assert abs(complex(r(jnp.asarray(.31)))-1j) < 1e-14


def test_n_zero_truncation_retains_literal_values_constructor():
    # The source m<N branch omits its 'coeffs' flag. Match that exact API
    # invocation, without silently correcting its interpretation.
    c = jnp.asarray([.2,.4,1.,.6,.8],dtype=jnp.complex128)
    f = cj.chebfun(c,coeffs=True,trig=True)
    p,*_ = cj.trigpade(f,1,0)
    expected = cj.chebfun(c[1:4],trig=True)
    assert jnp.array_equal(p.domain.breakpoints, expected.domain.breakpoints)
    assert float(jnp.max(jnp.abs(p.funs[0].tech.coeffs-expected.funs[0].tech.coeffs))) == 0
