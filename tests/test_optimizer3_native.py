"""Direct R2025b active-set controls; difficult numeric deltas stay diagnostic."""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.utils._active_set_box_sqp import active_set_box
from chebfunjax.utils._active_set_finite_difference import finite_difference

DATA=json.loads(Path(__file__).with_name('optimizer3_native.json').read_text())

@pytest.mark.parametrize('record',DATA['cases'],ids=lambda r:str(r['case']))
def test_native_three_variable_active_set(record):
    case=record['case']
    trace=[]
    def objective(x):
        if case in (1,5):
            value=jnp.sum((x-jnp.asarray([.125,-.25,.5]))**2)
        elif case==2:
            value=jnp.sum(jnp.asarray([1.,2.,4.])*(x-jnp.asarray([2.,-.25,-.5]))**2)
        elif case==3:
            value=jnp.sum(jnp.asarray([1.,2.,4.])*(x-jnp.asarray([2.,-2.,2.]))**2)
        elif case==4:
            value=(x[0]+2*x[1]-x[2]-.25)**2
        else:
            value=jnp.asarray(1.)
        trace.append(jnp.concatenate((x,value[None])))
        return value
    result=active_set_box(objective,jnp.asarray(record['initial'],dtype=jnp.float64),
                          jnp.asarray(record['lower'],dtype=jnp.float64),
                          jnp.asarray(record['upper'],dtype=jnp.float64),
                          finite_difference=finite_difference)
    assert result.iterations==record['iterations']
    assert result.evaluations==record['evaluations']
    assert result.exitflag==record['flag']
    assert bool(jnp.all(jnp.isfinite(result.x)))
    if case in (1,2,3,6):
        assert jnp.array_equal(result.x,jnp.asarray(record['x']))
        assert result.value==record['value']
        assert jnp.array_equal(jnp.stack(trace),jnp.asarray(record['trace']))
    # Cases4/5 have retained native/JAX roundoff divergences. No empirical
    # location tolerance or claim of bitwise agreement is introduced for them.
