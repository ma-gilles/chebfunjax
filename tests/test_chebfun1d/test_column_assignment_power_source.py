"""Source dispatch/pointValues reproduction, pinned Chebfun7574c77.

The wrappers only record genuine public calls; no sentinel or mathematical
substitute is used. Captures precede assertions so baseline failures persist.
"""
import json
import os
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def test_pointvalues_and_vector_power_source_reproduction(monkeypatch):
    x = cj.chebfun("x").set_point_values(jnp.array([7.,8.]))
    f = (1+0*x).set_point_values(jnp.array([11.,12.]))
    g = cj.chebfun("x").set_point_values(jnp.array([21.,22.]))
    assigned = f.assign_columns(1,g)
    expected_assignment = jnp.array([[11.,21.],[12.,22.]])
    order, composed = [], []
    original_power = Chebfun.__pow__
    original_compose = Chebtech2.compose

    def observed_power(self,exponent):
        if jnp.ndim(exponent) == 0:
            order.append(float(exponent))
        return original_power(self,exponent)

    def observed_compose(self,*args,**kwargs):
        composed.append({"length":len(self),"shape":list(self.coeffs.shape)})
        return original_compose(self,*args,**kwargs)

    monkeypatch.setattr(Chebfun,"__pow__",observed_power)
    monkeypatch.setattr(Chebtech2,"compose",observed_compose)
    vector = x ** jnp.array([0,1,2])
    vector_order, vector_compose = list(order), list(composed)
    identity = x ** jnp.asarray(1)
    squared = x ** jnp.asarray(2)
    expected_vector = jnp.array([[1.,7.,49.],[1.,8.,64.]])
    checks = {
        "assignment_pointvalues":bool(jnp.array_equal(assigned.point_values,expected_assignment)),
        "descending_power_order":vector_order == [2.,1.,0.],
        "no_composition_for_zero_one_two":not vector_compose,
        "identity_is_original":identity is x,
        "identity_pointvalues":bool(jnp.array_equal(identity.point_values,x.point_values)),
        "square_pointvalues":bool(jnp.array_equal(squared.point_values,jnp.array([49.,64.]))),
        "vector_pointvalues":bool(jnp.array_equal(vector.point_values,expected_vector)),
    }
    data={"checks":checks,"assigned":assigned.point_values.tolist(),
          "expected_assignment":expected_assignment.tolist(),
          "vector_order":vector_order,"compose_calls":vector_compose,
          "vector_pointvalues":vector.point_values.tolist(),
          "expected_vector_pointvalues":expected_vector.tolist(),
          "identity_pointvalues":identity.point_values.tolist(),
          "square_pointvalues":squared.point_values.tolist()}
    folder=os.environ.get("VANDERMONDE_STAGE_DIR")
    if folder:
        (Path(folder)/"source_reproduction.json").write_text(json.dumps(data,indent=2)+"\n")
    assert all(checks.values()), checks


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("row", [False, True])
def test_assignment_stored_values_and_orientation(tech, row):
    f = Chebfun(funs=[_Piece(tech=tech.from_coeffs(jnp.array([[1., 2.], [.5, 1.]])),
                            interval=(-1., 1.))], domain=Domain((-1., 1.)))
    f = Chebfun._as_transposed(f.set_point_values(jnp.array([[11., 12.], [21., 22.]])), row)
    g = Chebfun(funs=[_Piece(tech=tech.from_coeffs(jnp.array([[3., 4.]])),
                            interval=(-1., 1.))], domain=f.domain)
    g = Chebfun._as_transposed(g.set_point_values(jnp.array([[31.+1j, 32.+2j],
                                                          [41.+3j, 42.+4j]])), row)
    repeated = f.assign_columns([1, 1], g)
    assert jnp.array_equal(repeated.point_values, jnp.array([[11., 32.+2j], [21., 42.+4j]]))
    grown = f.assign_columns([3, 2], g)
    assert jnp.array_equal(grown.point_values, jnp.array([[11., 12., 32.+2j, 31.+1j],
                                                       [21., 22., 42.+4j, 41.+3j]]))
    deleted = grown.assign_columns([0, 2, 3], None)
    assert jnp.array_equal(deleted.point_values, jnp.array([12., 22.]))
    for value in [repeated, grown, deleted]:
        assert value.is_transposed == row
        assert all(type(piece.tech) is tech for piece in value.funs)
    assert f.assign_columns(":", g) is g


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_scalar_power_values_and_source_zero_support(tech):
    f = Chebfun(funs=[_Piece(tech=tech.from_coeffs(jnp.array([2., .25])),
                            interval=interval) for interval in [(-2., 0.), (0., 3.)]],
                domain=Domain((-2., 0., 3.))).set_point_values(jnp.array([7., 8., 9.]))
    for scalar in [1, jnp.asarray(1), jnp.asarray(1.)]:
        assert f ** scalar is f
    expected = f * f
    for scalar in [2, jnp.asarray(2), jnp.asarray(2.)]:
        result = f ** scalar
        assert jnp.array_equal(result.point_values, expected.point_values)
        for actual, target in zip(result.funs, expected.funs):
            assert type(actual.tech) is tech
            assert jnp.array_equal(actual.tech.coeffs, target.tech.coeffs)
    zero = f ** jnp.asarray(0)
    assert tuple(zero.domain.breakpoints) == (-2., 3.)
    assert jnp.array_equal(zero.point_values, jnp.ones_like(zero.point_values))
    assert jnp.array_equal(zero(jnp.array([-2., -1., 0., 3.])), jnp.ones(4))
