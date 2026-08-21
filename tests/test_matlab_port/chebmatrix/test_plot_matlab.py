"""Port of MATLAB Chebfun tests/chebmatrix/test_plot.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebmatrix/test_plot.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

import chebfunjax as cj  # noqa: E402
from chebfunjax.operators.blocks import (  # noqa: E402
    D,
    I,
    eval_at,
    mult,
    sum_functional,
)
from chebfunjax.operators.chebmatrix import ChebMatrix  # noqa: E402

jax.config.update("jax_enable_x64", True)

DOM = (-2.0, 2.0)


def _A():
    x = cj.chebfun(lambda t: t, domain=DOM)
    return ChebMatrix([
        [I(DOM) + D(DOM), abs(x), mult(x ** 2)],
        [sum_functional(DOM), 0.0, eval_at(2.0, DOM)],
        [D(DOM), x ** 2, I(DOM)],
    ])


class TestChebmatrixPlot:
    def test_all_matlab_assertions(self):
        import matplotlib.pyplot as plt
        A = _A()
        assert A.plot() is not None  # pass(1)
        with pytest.raises(ValueError, match="loglog plot of infinite"):
            A.loglog()  # pass(2)
        with pytest.raises(ValueError,
                           match="semilogx plot of infinite"):
            A.semilogx()  # pass(3)
        plt.close("all")
