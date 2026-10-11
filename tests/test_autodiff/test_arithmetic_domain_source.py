import jax.numpy as jnp
import pytest

from chebfunjax import chebfun
from chebfunjax.autodiff.adchebfun import ADChebfun


@pytest.mark.parametrize("operation", ["add", "sub", "rsub", "ad_add", "ad_sub"])
def test_source_update_domain_preserves_inputs(operation):
    base = ADChebfun(chebfun(lambda x: x, domain=(0.0, 1.0)))
    other = chebfun(lambda x: x * x, domain=(0.0, 0.5, 1.0))
    before = base.jacobian.domain
    if operation == "add":
        result = base + other
    elif operation == "sub":
        result = base - other
    elif operation == "rsub":
        result = base.__rsub__(other)
    elif operation == "ad_add":
        result = base + ADChebfun(other)
    else:
        result = base - ADChebfun(other)
    assert result.domain == (0.0, 0.5, 1.0)
    assert result.jacobian.domain == result.domain
    assert base.jacobian.domain == before
    assert base.domain == (0.0, 1.0)
    expected = (
        (lambda x: x + x * x)
        if "add" in operation
        else ((lambda x: x * x - x) if operation == "rsub" else (lambda x: x - x * x))
    )
    x = jnp.array([0.2, 0.8])
    assert jnp.max(jnp.abs(result.func(x) - expected(x))) < 1e-14
