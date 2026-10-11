"""Public root recursion preference from MATLAB @chebfun/roots.m.

Provenance
----------
MATLAB source : @chebfun/roots.m parseInputs, complex/ZetaZeros.m caller.
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.tech.chebtech import Chebtech2


@pytest.mark.parametrize("complex_roots", [False, True])
@pytest.mark.parametrize("array_valued", [False, True])
def test_norecursion_is_forwarded(complex_roots, array_valued, monkeypatch):
    coefficients = jnp.asarray([-0.2, 0.0, 1.0])
    if array_valued:
        coefficients = jnp.stack([coefficients, coefficients], axis=1)
    f = cj.Chebfun.from_coeffs(coefficients, domain=[5.0, 50.0])
    calls = []
    original = Chebtech2.roots

    def observed(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Chebtech2, "roots", observed)
    roots = f.roots(complex_roots=complex_roots, norecursion=True)
    assert roots.size > 0
    assert calls and all(call.get("recurse") is False for call in calls)
    if complex_roots:
        assert calls[0].get("complex_roots") is True
        assert calls[0].get("all_roots") is False
        # Native complex flag becomes all+prune before per-column dispatch.
        assert all(call.get("all_roots") is True and call.get("prune") is True
                   for call in calls[1:])


def test_complex_norecursion_retains_source_pruning():
    coefficients = jnp.zeros(65, dtype=jnp.complex128)
    coefficients = coefficients.at[0].set(-0.2j).at[-1].set(1.0)
    f = cj.Chebfun.from_coeffs(coefficients, domain=[5.0, 50.0])
    direct = np.asarray(f.funs[0].tech.roots(complex_roots=True, recurse=False))
    expected = 22.5 * direct + 27.5
    actual = np.asarray(f.roots(complex_roots=True, norecursion=True))
    expected = np.asarray(sorted(expected, key=lambda root: (abs(root), np.angle(root))))
    np.testing.assert_array_equal(actual, expected)
    assert len(actual) == 64


@pytest.mark.parametrize("array_valued", [False, True])
def test_complex_roots_sort_after_physical_mapping(array_valued, monkeypatch):
    # Exact controlled provider values distinguish physical-domain sort from
    # reference-domain sort; MATLAB columnRoots sorts after FUN transplantation.
    f = cj.Chebfun.from_coeffs(jnp.asarray([1.0, 1.0]), domain=[5.0, 50.0])
    supplied = jnp.asarray([1j, -1j, -1.0, 1.0])
    expected = np.asarray([5.0, 27.5-22.5j, 27.5+22.5j, 50.0])
    if array_valued:
        supplied = jnp.stack([supplied, supplied[::-1]], axis=1)
        expected = np.stack([expected, expected], axis=1)
    monkeypatch.setattr(Chebtech2, "roots", lambda self, **kwargs: supplied)
    np.testing.assert_array_equal(f.roots(all_roots=True), expected)


def test_complex_root_equal_magnitude_phase_order(monkeypatch):
    f = cj.Chebfun.from_coeffs(jnp.asarray([1.0, 1.0]))
    supplied = jnp.asarray([2.0, 1j, -2.0, -1j])
    monkeypatch.setattr(Chebtech2, "roots", lambda self, **kwargs: supplied)
    np.testing.assert_array_equal(f.roots(all_roots=True), [-1j, 1j, 2.0, -2.0])


def test_real_all_roots_keep_signed_sort(monkeypatch):
    f = cj.Chebfun.from_coeffs(jnp.asarray([1.0, 1.0]))
    monkeypatch.setattr(Chebtech2, "roots", lambda self, **kwargs: jnp.asarray([1.0, -1.0, 0.0]))
    np.testing.assert_array_equal(f.roots(all_roots=True), [-1.0, 0.0, 1.0])


def test_complex_root_array_nan_padding_and_all_nan_rows(monkeypatch):
    f = cj.Chebfun.from_coeffs(jnp.asarray([1.0, 1.0]))
    supplied = jnp.asarray([[jnp.nan, 2.0], [1j, jnp.nan], [-1j, 1.0],
                            [jnp.nan, 3.0], [jnp.nan, jnp.nan]])
    monkeypatch.setattr(Chebtech2, "roots", lambda self, **kwargs: supplied)
    expected = np.asarray([[-1j, 1.0], [1j, 2.0], [np.nan, 3.0]])
    np.testing.assert_array_equal(f.roots(all_roots=True), expected)
