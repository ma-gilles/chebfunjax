"""Source controls for Chebfun3 coefficient-input constructor ordering.

Provenance
----------
MATLAB source : @chebfun3/constructor.m, @chebfun3/coeffs2vals.m,
                @chebfun3/chebfun3double.m
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1
from chebfunjax.tech.trigtech import Trigtech

jax.config.update("jax_enable_x64", True)


def _points():
    return (jnp.array([-0.71, 0.12, 0.83]),
            jnp.array([-0.62, 0.35]),
            jnp.array([-0.91, -0.1, 0.77]))


def _assert_same_values(actual, expected, points):
    x, y, z = points
    for xv in x:
        for yv in y:
            for zv in z:
                a = jnp.asarray(actual(float(xv), float(yv), float(zv)))
                b = jnp.asarray(expected(float(xv), float(yv), float(zv)))
                assert jnp.array_equal(a, b)


def test_constructor2_coefficient_predicate_shapes_and_bound():
    # constructor2.m passes (1)-(2) use rand(3,3,3) and rand(4,4,4),
    # with norm(error) < 10*(100*pref.chebfun3eps). These deterministic
    # Python RNG tensors preserve the source shapes and bound, not MATLAB's
    # random input values or stream.
    tol = 100.0 * float(ChebfunPref().cheb3Prefs.chebfun3eps)
    rng = np.random.RandomState(0)
    for n in (3, 4):
        coeff = jnp.asarray(rng.rand(n, n, n))
        f = chebfun3(coeff, coeffs=True)
        got = f.chebcoeffs3()
        assert got.shape == coeff.shape
        assert float(jnp.linalg.norm((got - coeff).ravel())) < 10.0 * tol


def test_complex_coefficients_convert_before_selected_numeric_dispatch():
    domain = (-1., 1., -1., 1., -1., 1.)
    coeff = (jnp.arange(24, dtype=jnp.float64).reshape(3, 4, 2) / 17.0 +
             1j * jnp.flip(jnp.arange(24, dtype=jnp.float64).reshape(3, 4, 2)) / 31.0)
    values = Chebfun3.coeffs2vals(coeff)
    points = _points()
    cases = (
        (None, {}),
        (Trigtech, {"tech": Trigtech}),
        (Chebtech1, {"pref": ChebfunPref(tech=Chebtech1)}),
        (Chebtech1, {"tech": Chebtech1}),
    )
    for selected, opts in cases:
        via_coeffs = chebfun3(coeff, domain=domain, coeffs=True, **opts)
        via_values = chebfun3(values, domain=domain, **opts)
        assert via_coeffs.isreal() is False
        if selected is not None:
            assert all(isinstance(t, selected) for t in
                       (*via_coeffs.cols, *via_coeffs.rows, *via_coeffs.tubes))
        _assert_same_values(via_coeffs, via_values, points)
    source_tol = float(ChebfunPref().cheb3Prefs.chebfun3eps)
    direct_trig = Chebfun3.from_coeffs(coeff, domain=domain, tol=source_tol,
                                       tech=Trigtech)
    expected_trig = chebfun3(values, domain=domain, tol=source_tol,
                             tech=Trigtech)
    direct_c1 = Chebfun3.from_coeffs(coeff, domain=domain, tol=source_tol,
                                     tech=Chebtech1)
    expected_c1 = chebfun3(values, domain=domain, tol=source_tol,
                           pref=ChebfunPref(tech=Chebtech1))
    for direct, expected, selected in (
            (direct_trig, expected_trig, Trigtech),
            (direct_c1, expected_c1, Chebtech1)):
        assert direct.isreal() is False
        assert all(isinstance(t, selected) for t in
                   (*direct.cols, *direct.rows, *direct.tubes))
        _assert_same_values(direct, expected, points)

def test_complex_scalar_coefficients_preserve_complex_value():
    coeff = jnp.asarray([[[2.0 + 3.0j]]])
    f = chebfun3(coeff, coeffs=True)
    direct = Chebfun3.from_coeffs(coeff)
    scale_tol = 100.0 * float(ChebfunPref().cheb3Prefs.chebfun3eps)
    for value in (f, direct):
        assert value.isreal() is False
        assert abs(complex(value(0.17, -0.21, 0.38)) - (2.0 + 3.0j)) < scale_tol


def test_fixed_length_sampling_precedes_coefficient_conversion():
    domain = (-1., 1., -1., 1., -1., 1.)
    lengths = (3, 4, 2)
    from chebfunjax.utils.quadrature import chebpts3
    grid = chebpts3(*lengths, domain=domain)
    sampled = grid[0] + 2.0 * grid[1] - 0.5 * grid[2]
    converted = Chebfun3.coeffs2vals(sampled)
    actual = chebfun3(lambda x, y, z: x + 2.0*y - 0.5*z, domain=domain,
                      n=lengths, coeffs=True)
    expected = chebfun3(converted, domain=domain)
    _assert_same_values(actual, expected, _points())
