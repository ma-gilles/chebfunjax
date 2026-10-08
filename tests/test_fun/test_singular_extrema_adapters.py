"""Independent exact controls for classicfun/singfun extrema adapters.

Provenance
----------
MATLAB source : @classicfun/minandmax.m, @singfun/minandmax.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.singfun import Singfun
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.tech.chebtech import Chebtech2


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("side", [0, 1])
def test_bounded_endpoint_pole_values_and_positions(sign, side):
    exps = (-0.5, 0.0) if side == 0 else (0.0, -0.5)
    f = Bndfun(Singfun(float(sign), exps), Domain((2.0, 8.0)))
    mn, mx = f.minandmax()
    pole, finite = (mx, mn) if sign > 0 else (mn, mx)
    assert pole[0] == sign * jnp.inf
    assert pole[1] == (2.0, 8.0)[side]
    assert abs(finite[0] - sign / jnp.sqrt(2.0)) < 4 * jnp.finfo(float).eps
    assert finite[1] == (8.0, 2.0)[side]


def test_smooth_singfun_adapter_keeps_finite_extrema():
    smooth = Chebtech2.from_coeffs(jnp.asarray([3.0, 2.0]))
    f = Bndfun(Singfun(smooth, (0.0, 0.0)), Domain((2.0, 8.0)))
    assert f.minandmax() == ((1.0, 2.0), (5.0, 8.0))


def test_unbounded_singfun_adapter_preserves_piece_position_first():
    # Mapped f(y)=1/(1-y) on [2,inf] is (x+13)/30.
    f = Unbndfun(Singfun(1.0, (0.0, -1.0)), Domain((2.0, jnp.inf)), "right_inf")
    (xp, mn), (xq, mx) = f.minandmax()
    assert xp == 2.0 and mn == 0.5
    assert xq == jnp.inf and mx == jnp.inf
    assert f.min() == (0.5, 2.0)
