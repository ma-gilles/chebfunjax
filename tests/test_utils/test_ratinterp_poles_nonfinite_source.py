"""Actual eight source captures; verified archive bound by fixture provenance.

Provenance: MATLAB ratinterp.m copied helpers, Chebfun7574c77, MATLABR2025b.
Captured outputs are expected values only; inputs are independent literals.
"""
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._ratbary import _source_type2_rational_handle


@pytest.mark.parametrize("name,a", [("genuine",2+.25j),("pure_imag",2j),
                                    ("explicit_real",complex(2,0)),("explicit_zero",complex(0,0))])
@pytest.mark.parametrize("query_index", [1,2])
def test_source_pole_and_nan_capture(name,a,query_index):
    path=Path(__file__).parents[1]/'fixtures'/'ratinterp_type2_poles_2025b'/f'{name}_query{query_index}.json'
    record=json.loads(path.read_text())  # Missing capture is an error, never a skip.
    r=_source_type2_rational_handle(jnp.array([a]),jnp.array([2.,1.]),(-1.,1.))
    x=jnp.array([-2.,-1.,-.5,0.,.5,1.]) if query_index==1 else jnp.array(float('nan'))
    if record['error_identifier']:
        # MATLAB scalar-assignment multi-hit failure maps to Python ValueError.
        assert record['error_identifier']=='MATLAB:matrix:singleSubscriptNumelMismatch'
        with pytest.raises(ValueError,match='multiple indices'):
            r(x)
        return  # Traced source exceptions remain explicitly unsupported.
    def decode(words):
        return np.array([struct.unpack('>d',bytes.fromhex(word))[0] for word in words])
    er=decode(record['value']['real_num2hex']).reshape(x.shape,order='F')
    ei=decode(record['value']['imag_num2hex']).reshape(x.shape,order='F')
    for evaluate in (r,jax.jit(r)):
        actual=np.asarray(evaluate(x))
        assert np.iscomplexobj(actual)==(not record['value']['isreal'])
        np.testing.assert_array_equal(actual.real,er)
        np.testing.assert_array_equal(actual.imag,ei)
