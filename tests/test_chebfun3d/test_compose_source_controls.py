"""Controls for the source dispatch branches beyond the original predicates.

Provenance
----------
MATLAB source : @chebfun3/compose.m, @chebfun3/sample.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import pytest

from chebfunjax import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun3d.chebfun3 import chebfun3

TOL = 1000 * float(jnp.finfo(jnp.float64).eps)


@pytest.mark.parametrize("operand", [2, "function"])
def test_binary_composition(operand):
    f = chebfun3(lambda x, y, z: x)
    g = chebfun3(lambda x, y, z: y) if operand == "function" else operand
    h = f.compose(lambda a, b: a + b, g)
    exact = chebfun3(lambda x, y, z: x + (y if operand == "function" else 2))
    assert (h - exact).norm() < TOL


@pytest.mark.parametrize("op,args", [(lambda a, b: a + b, ()),
                                      (lambda a: a, (2,)), (3, ()),
                                      (lambda a, b: a + b, (2, 3))])
def test_wrong_arity(op, args):
    f = chebfun3(lambda x, y, z: x)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:COMPOSE:OP"):
        f.compose(op, *args)


def test_complex_chebfun_rejected():
    f = chebfun3(lambda x, y, z: x + 1j * z)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:COMPOSE:Complex"):
        f.compose(chebfun(lambda t: t ** 2))


def test_range_rejected():
    f = chebfun3(lambda x, y, z: 2 * x)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:COMPOSE:DomainMismatch"):
        f.compose(chebfun(lambda t: t ** 2))


def test_chebfun2_range_rejected():
    f = chebfun3(lambda x, y, z: 2 * x + 1j * z)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3V:COMPOSE:DomainMismatch2"):
        f.compose(Chebfun2.from_function(lambda x, y: x + y))


def test_four_columns_rejected():
    f = chebfun3(lambda x, y, z: x)
    g = chebfun(lambda t: jnp.stack((t, t ** 2, t ** 3, t ** 4), axis=-1))
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:COMPOSE:Columns"):
        f.compose(g)


def test_three_columns_and_piece_warning():
    f = chebfun3(lambda x, y, z: x)
    g = chebfun(lambda t: jnp.stack((t, t ** 2, t ** 3), axis=-1),
                domain=[-1, 0, 1])
    with pytest.warns(UserWarning, match="CHEBFUN:CHEBFUN3:compose:pieces"):
        h = f.compose(g)
    assert h.n_components == 3
    for i, component in enumerate(h.components):
        assert (component - f ** (i + 1)).norm() < TOL


def test_periodic_values_and_unary_preservation():
    f = chebfun3(lambda x, y, z: jnp.cos(jnp.pi * x), trig=True)
    h = f.compose(chebfun(lambda t: t ** 2))
    unary = f.sin()
    assert unary.isPeriodicTech()
    x = jnp.linspace(-1, 1, 37)
    assert jnp.max(jnp.abs(h(x, 0., 0.) - jnp.cos(jnp.pi * x) ** 2)) < TOL
    assert jnp.max(jnp.abs(unary(x, 0., 0.) - jnp.sin(jnp.cos(jnp.pi * x)))) < TOL


@pytest.mark.parametrize("tech", ["chebtech2", "chebtech1", "trigtech"])
def test_sample_factor_grids(tech):
    f = chebfun3(lambda x, y, z: jnp.cos(jnp.pi * x), tech=tech)
    _, nodes = f.cols[0].sample(25)
    expected = jnp.broadcast_to(jnp.cos(jnp.pi * nodes)[:, None, None], (25, 3, 2))
    assert jnp.max(jnp.abs(f.sample(25, 3, 2) - expected)) < TOL
    assert jnp.max(jnp.abs(f.minandmax3est() - jnp.array([
        jnp.min(jnp.cos(jnp.pi * nodes)), jnp.max(jnp.cos(jnp.pi * nodes))]))) < TOL


def test_periodic_real_sine_isreal():
    f = chebfun3(lambda x, y, z: jnp.sin(jnp.pi * x), trig=True)
    assert f.isreal()
    h = f.compose(chebfun(lambda t: t ** 2))
    x = jnp.linspace(-1, 1, 31)
    assert jnp.max(jnp.abs(h(x, 0., 0.) - jnp.sin(jnp.pi * x) ** 2)) < TOL


@pytest.mark.parametrize("periodic", [False, True])
def test_complex_samples_and_factors_match_public_evaluation(periodic):
    if periodic:
        f = chebfun3(lambda x, y, z: jnp.exp(1j * jnp.pi * x)
                    * jnp.cos(jnp.pi * y) + .2 * jnp.sin(jnp.pi * z), trig=True)
    else:
        f = chebfun3(lambda x, y, z: x * y + 1j * (y + z),
                    (-2, 1, 0, 2, -3, 0))
    counts = (7, 10, 13)
    nodes = []
    for i, (group, count) in enumerate(zip((f.cols, f.rows, f.tubes), counts)):
        _, t = group[0].sample(count)
        a, b = f.domain[2 * i:2 * i + 2]
        nodes.append((b - a) * t / 2 + (b + a) / 2)
    x, y, z = jnp.meshgrid(*nodes, indexing="ij")
    expected = f(x, y, z)
    values = f.sample(*counts)
    core, columns, rows, tubes = f.sample(*counts, return_factors=True)
    contracted = jnp.einsum("ijk,ai,bj,ck->abc", core, columns, rows, tubes)
    assert values.shape == counts
    assert jnp.max(jnp.abs(values - expected)) < TOL
    assert jnp.max(jnp.abs(contracted - expected)) < TOL


def test_sample_defaults_and_partial_count_source_behavior():
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3

    # T60 needs 61 points in x, while source defaults require 51 in y/z.
    coefficients = jnp.zeros((61, 1, 1)).at[60, 0, 0].set(1)
    f = Chebfun3.from_coeffs(coefficients)
    assert f.sample().shape == (61, 51, 51)
    assert f.sample(5).shape == (61, 51, 51)
    core, columns, rows, tubes = f.sample(return_factors=True)
    assert columns.shape[0] == 61
    assert rows.shape[0] == tubes.shape[0] == 51
    assert core.shape == f.core.shape


def test_empty_sampling():
    f = chebfun3()
    assert f.sample().size == 0
    assert f.sample(3, 4, 5).size == 0
    assert f.sample(return_factors=True).size == 0
    assert jnp.array_equal(f.minandmax3est(), jnp.zeros(2))


@pytest.mark.parametrize("kind", ["ufunc", "jit", "frompyfunc", "opaque"])
def test_supported_jax_callables(kind):
    import jax

    class OpaqueUnary:
        @property
        def __signature__(self):
            raise ValueError("This callable does not expose a Python signature")

        def __call__(self, x):
            return jnp.sin(x)

    op = {"ufunc": jnp.sin, "jit": jax.jit(jnp.sin),
          "frompyfunc": jnp.frompyfunc(jnp.sin, 1, 1), "opaque": OpaqueUnary()}[kind]
    f = chebfun3(lambda x, y, z: x)
    h = f.compose(op)
    x = jnp.linspace(-1, 1, 31)
    assert jnp.max(jnp.abs(h(x, 0., 0.) - jnp.sin(x))) < TOL


def test_binary_ufunc_arity():
    f = chebfun3(lambda x, y, z: x)
    h = f.compose(jnp.add, 2)
    x = jnp.linspace(-1, 1, 17)
    assert jnp.max(jnp.abs(h(x, 0., 0.) - x - 2)) < TOL
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:COMPOSE:OP"):
        f.compose(jnp.add)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:COMPOSE:OP"):
        f.compose(jnp.sin, 2)


@pytest.mark.parametrize("vector", [False, True])
def test_direct_complex_vector_field_rejected(vector):
    from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
    from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

    field = Chebfun3v.from_functions(lambda x, y, z: x + 1j * y,
                                     lambda x, y, z: z)
    outer = (Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
             if vector else Chebfun2.from_function(lambda x, y: x + y))
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3V:COMPOSE:Complex"):
        field.compose(outer)
