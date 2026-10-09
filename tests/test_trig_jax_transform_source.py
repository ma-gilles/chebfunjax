"""Unchanged native predicates exercised directly on the compiled JAX cores."""
import pytest

from chebfunjax.tech.trigtech import Trigtech, _trig_coeffs2vals_impl, _trig_vals2coeffs_impl
from tests.test_matlab_port.trigtech.test_coeffs2vals_matlab import (
    TestTrigtechCoeffs2Vals as _Coeffs,
)
from tests.test_matlab_port.trigtech.test_vals2coeffs_matlab import (
    TestTrigtechVals2Coeffs as _Values,
)


@pytest.fixture(autouse=True)
def direct_jax_cores(monkeypatch):
    # Public concrete wrappers retain a NumPy path. These source assertions
    # deliberately reach the JAX cores used by Spherefun construction.
    monkeypatch.setattr(Trigtech, "coeffs2vals", staticmethod(_trig_coeffs2vals_impl))
    monkeypatch.setattr(Trigtech, "vals2coeffs", staticmethod(_trig_vals2coeffs_impl))


class TestJaxCoefficients(_Coeffs):
    pass


class TestJaxValues(_Values):
    pass
