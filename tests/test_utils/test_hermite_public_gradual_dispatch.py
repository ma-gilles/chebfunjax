# uses-numpy: host reference assertions and captured MATLAB fixture inspection.
"""Public ASY stage routing controls, Chebfun hermpts.m169-174, commit7574c77."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.quadrature import hermpts


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("method", ["default", "ASY"])
@pytest.mark.parametrize("kind", ["phys", "prob"])
def test_public_keeps_tiny_stage_weights(monkeypatch, method, kind, disabled):
    from fractions import Fraction

    import chebfunjax.utils.hermite_asy as module
    raw = np.asarray([1, 3, 0x3ff0000000000000],dtype=np.uint64).view(np.float64)
    monkeypatch.setattr(module, "_hermpts_asy", lambda n: (jnp.array([-1.,0.,1.]),jnp.asarray(raw),jnp.array([1.,-1.,1.])))
    with jax.disable_jit(disabled):
        x,w,v=hermpts(201,kind,method=method,bary=True)
    factor=float(jnp.sqrt(jnp.pi)/jnp.sum(jnp.asarray(raw)))
    expected=np.asarray([float(Fraction.from_float(float(a))*Fraction.from_float(factor)) for a in raw])
    if kind == "prob":
        scale=float(jnp.sqrt(2.))
        expected=np.asarray([float(Fraction.from_float(float(a))*Fraction.from_float(scale)) for a in expected])
    np.testing.assert_array_equal(np.asarray(w).view(np.uint64),expected.view(np.uint64))
    np.testing.assert_array_equal(np.asarray(v),[1.,-1.,1.])

@pytest.mark.parametrize("kind", ["phys", "prob"])
def test_public_odd_asy_source_fold_and_mass(kind):
    x,w,v=hermpts(201,kind,method="ASY",bary=True)
    x,w,v=map(np.asarray,(x,w,v))
    assert x.shape==w.shape==v.shape==(201,)
    # MATLAB hermpts.m folds -x including its approximate central node.
    # Actual R2025b n201 centers are nonzero in both conventions.
    np.testing.assert_array_equal(x[:100],-x[:100:-1])
    np.testing.assert_array_equal(v,v[::-1])
    target=np.sqrt(np.pi)*(np.sqrt(2.) if kind=="prob" else 1.)
    assert abs(np.sum(w)-target)<3000*np.finfo(float).eps
