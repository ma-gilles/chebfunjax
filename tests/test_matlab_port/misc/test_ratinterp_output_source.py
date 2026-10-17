"""Source output-count and symmetry controls for ratinterp (7574c77).

These are authored source controls, separate from the 23 original clauses.
Low-degree native comparisons retain test_ratinterp.m's 1e-10 bound; no
high-degree or general clustered-pole accuracy follows from these controls.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.utils.ratapprox import (
    _check_symmetries,
    _ratinterp_source_sort_order,
    ratinterp,
)


@pytest.mark.parametrize('count', range(1, 8))
def test_native_output_prefix(count):
    result = ratinterp(lambda x: (x + 1) / (x - 3), 1, 1,
                       tol=0, outputs=count)
    if count == 1:
        assert callable(result)
        assert abs(result(jnp.asarray(0.2)) - 1.2 / -2.8) < 1e-10
    else:
        assert len(result) == count
        p, q = result[:2]
        assert abs(p(0.2) / q(0.2) - 1.2 / -2.8) < 1e-10
        if count >= 6:
            assert abs(result[5][0] - 3) < 1e-10
        if count == 7:
            assert abs(result[6][0] - 4) < 1e-10


@pytest.mark.parametrize('grid', ['type0', 'type1', 'type2', 'equi'])
def test_source_polynomials_for_each_basis(grid):
    f = lambda x: (x*x + 1j*x + 2) / (x - (3 + 1j))
    p, q, _, mu, nu, poles, residues = ratinterp(
        f, 2, 1, xi=grid, tol=0, domain=(-2., 3.), outputs=7)
    assert (mu, nu) == (2, 1)
    x = jnp.asarray([-1.5, 0.25, 2.5])
    assert jnp.max(jnp.abs(p(x) / q(x) - f(x))) < 1e-10
    assert abs(poles[0] - (3 + 1j)) < 1e-10
    assert abs(residues[0] - (3 + 1j)**2 - 1j*(3 + 1j) - 2) < 1e-10


def test_six_vs_seven_nearby_poles_native_capture():
    # Fresh native source capture: six retains separate roots; seven groups
    # them, then ratinterp copies residue(power1)=0 onto power2 as well.
    f = lambda x: 1 / jnp.polyval(jnp.array([1., -2.0001, 1.0001]), x)
    six = ratinterp(f, 0, 2, tol=0, domain=(-2., 3.), outputs=6)
    seven = ratinterp(f, 0, 2, tol=0, domain=(-2., 3.), outputs=7)
    assert jnp.max(jnp.abs(jnp.sort(jnp.real(six[5])) - jnp.array([1., 1.0001]))) < 1e-10
    assert seven[5][0] == seven[5][1]
    assert jnp.max(jnp.abs(seven[5] - 1.00005)) < 1e-10
    assert jnp.array_equal(seven[6], jnp.zeros(2))


@pytest.mark.parametrize('values,order', [
    ([-2., 1., -1., 2.], [0, 2, 1, 3]),
    ([complex(-2, 0), complex(1, 0), complex(-1, 0), complex(2, 0)], [1, 2, 3, 0]),
    ([1j, -1j, 1+0j, -1+0j], [1, 2, 0, 3]),
    ([complex(-1, -0.0), complex(-1, 0.0), -1j, 1j], [0, 2, 3, 1]),
])
def test_native_sort_storage_and_signed_zero(values, order):
    assert jnp.array_equal(_ratinterp_source_sort_order(jnp.asarray(values)), jnp.asarray(order))


@pytest.mark.parametrize('kind,expected', [('even', (True, False)), ('odd', (False, True)), ('neither', (False, False))])
def test_arbitrary_real_node_symmetry(kind, expected):
    x = jnp.array([0.8, -0.3, -0.8, 0.3])
    f = x*x if kind == 'even' else x if kind == 'odd' else x*x+x
    assert _check_symmetries(f, x, 'ARBITRARY', 1e-14) == expected


def test_arbitrary_complex_sort_does_not_invent_symmetry():
    # Magnitude/phase ordering does not pair these nodes with their negatives;
    # the literal source first rejects the node symmetry, even for f=x**2.
    x = jnp.array([1+0j, -1+0j, 1j, -1j])
    assert _check_symmetries(x*x, x, 'ARBITRARY', 1e-14) == (False, False)


def test_unsymmetric_nodes_do_not_set_data_symmetry():
    x = jnp.array([-0.8, -0.3, 0.3, 0.7])
    assert _check_symmetries(jnp.ones(4), x, 'ARBITRARY', 1e-14) == (False, False)


@pytest.mark.parametrize('count', [0, 8, -1, 1.5, True])
def test_invalid_output_count(count):
    with pytest.raises(ValueError):
        ratinterp(lambda x: x, 1, 0, outputs=count)


@pytest.mark.parametrize('count', [6, 7])
@pytest.mark.parametrize('degree', [2, 3])
def test_matched_native_low_degree_complex_capture(degree, count):
    # Native R2025b, pinned Chebfun, same formulas/domains/degrees/tol=0.
    # These retained values are test references only, never fit inputs.
    if degree == 2:
        numerator = jnp.array([1+2j, -.5j, 2, 1])
        denominator = jnp.array([1, -.8-.3j, -.4+.6j])
        domain, m = (-2., 3.), 3
        expected_poles = jnp.array([-.3931358854162046+.45259631976437364j,
                                   1.1931358854162015-.15259631976437305j])
        expected_residues = jnp.array([.3055705705933914-.7717562327396891j,
                                      3.0344294294066176+2.151756232739688j])
    else:
        numerator = jnp.array([1+1j, 2])
        denominator = jnp.array([1, -4-1j, 3+2j, -1j])
        domain, m = (0., 4*jnp.pi), 1
        expected_poles = jnp.array([.07494687593591254+.3245679436242931j,
                                   .99999999999953-1.876753110892483e-13j,
                                   2.925053124063739+.6754320563762561j])
        expected_residues = jnp.array([.5771869249227635+.2712742617236015j,
                                      -1.500000000000384-.5000000000013954j,
                                      .9228130750776201+.22872573827779374j])
    f = lambda x: jnp.polyval(numerator, x)/jnp.polyval(denominator, x)
    result = ratinterp(f, m, degree, tol=0, domain=domain, outputs=count)
    assert jnp.max(jnp.abs(result[5] - expected_poles)) < 1e-10
    if count == 7:
        assert jnp.max(jnp.abs(result[6] - expected_residues)) < 1e-10
