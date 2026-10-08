"""All 12 source clauses, retaining the intentional duplicate.

Provenance: tests/adchebfun/test_linearityDetection.m, Chebfun 7574c77.
NumPy MT19937 input adapter matches the first 16 native seed6179 values
bitwise. Later draws have not been captured natively for this source test.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np  # uses-numpy: MATLAB uniform random-input adapter only.
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun


@pytest.fixture(scope="module")
def inputs():
    stream = np.random.RandomState(6179)
    raw = stream.rand(42)
    native = json.loads(Path(__file__).with_name("erf_matlab_inputs.json").read_text())
    np.testing.assert_array_equal(raw[:16], native["raw_u"]+native["raw_p"])
    functions = [Chebfun.from_values(.1*raw[k:k+8]+.5) for k in (0, 8, 16, 24, 34)]
    return functions, raw[32], raw[33]


@pytest.mark.parametrize("clause", range(12))
def test_source_linearity(inputs, clause):
    (u1, u2, w1, w2, u3), s1, s2 = inputs
    v1, v2 = ADChebfun(u1).seed(1, 2), ADChebfun(u2).seed(2, 2)
    if clause >= 9:
        v1, v2, v3 = (ADChebfun(u).seed(k, 3) for k, u in enumerate((u1, u2, u3), 1))
        expressions = [lambda: v1*v3+v2, lambda: v1/(v3+2)+v2,
                       lambda: v1+v2**2+v3.csc()]
        expected = [(False, True, False), (False, True, False), (True, False, False)]
        assert expressions[clause-9]().linearity == expected[clause-9]
    else:
        expressions = [lambda: v1.sin()+w1, lambda: v1.sin()+v1,
                       lambda: v1.cos()+w2.exp(), lambda: (v2+2).log()+jnp.exp(s1),
                       lambda: v1.sin()-v2.exp(), lambda: v1.sin()-v2.diff(),
                       lambda: v1.sin()-v2.diff(), lambda: v2.tan()*w2,
                       lambda: w2*(v1+2).log2()/4]
        expected = [(False, True)]*3+[(True, False), (False, False),
                   (False, True), (False, True), (True, False), (False, True)]
        assert expressions[clause]().linearity == expected[clause]


@pytest.mark.parametrize("case", range(3))
def test_independent_three_variable_derivative(inputs, case):
    (u1, u2, _, _, u3), _, _ = inputs
    v1, v2, v3 = (ADChebfun(u).seed(k, 3) for k, u in enumerate((u1, u2, u3), 1))
    expressions = [lambda: v1*v3+v2, lambda: v1/(v3+2)+v2,
                   lambda: v1+v2**2+v3.csc()]
    multipliers = [(u3, 1., u1), (1/(u3+2), 1., -u1/(u3+2)**2),
                   (1., 2*u2, -u3.csc()*u3.cot())]
    h = u2*.03
    result = expressions[case]()
    for block, multiplier in zip(result.jacobian.blocks[0], multipliers[case]):
        assert float((block.apply(h)-h*multiplier).norm(jnp.inf)) < 1e-12
