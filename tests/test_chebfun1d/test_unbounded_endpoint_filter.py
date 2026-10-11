"""Native @unbndfun/roots endpoint filtering, isolated source controls."""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.singfun import Singfun
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def _near_endpoint_decay_tech():
    # p(s)=s(1-s)(s-r), r=1-1e-8. The rational function p(mapped_x)
    # decays at +infinity; the nearby finite root lies entirely in a tail
    # whose sampled magnitude is below the source 10*eps*vscale cutoff.
    r = 1.0 - 1e-8
    coeffs = jnp.asarray([(1.0+r)/2.0, -(0.75+r), (1.0+r)/2.0, -0.25])
    return Chebtech2(coeffs=coeffs)


def test_unbounded_endpoint_decay_filter_removes_tail_roots_and_overrides_user_filter():
    piece = Unbndfun(onefun=_near_endpoint_decay_tech(),
                     domain=Domain((-jnp.inf, jnp.inf)), mapping_type="both_inf")

    def forbidden_user_filter(roots, tech):
        raise AssertionError("Unbndfun must replace the user filter")

    roots = np.asarray(piece.roots(filter=forbidden_user_filter))
    np.testing.assert_allclose(roots, [0.0], atol=2e-14, rtol=0)


def test_filter_runs_after_root_engine_and_before_top_level_prune(monkeypatch):
    import chebfunjax.tech.chebtech as module

    tech = Chebtech2(coeffs=jnp.asarray([1., 0., 1., 0.]))
    events = []

    def fake_main(*args, **kwargs):
        events.append("roots_main")
        return np.asarray([0.5, -0.5])

    def fake_prune(roots, rho):
        events.append(("top_prune", np.asarray(roots).tolist()))
        return np.asarray(roots)

    def keep_left(roots, root_tech):
        events.append(("filter", np.asarray(roots).tolist(), root_tech is tech))
        return roots[:1]

    monkeypatch.setattr(module, "_roots_main", fake_main)
    monkeypatch.setattr(module, "_prune_spurious_roots", fake_prune)
    got = np.asarray(tech.roots(all_roots=True, prune=True, recurse=False,
                                filter=keep_left))
    assert events == ["roots_main", ("filter", [-0.5, 0.5], True),
                      ("top_prune", [-0.5])]
    np.testing.assert_array_equal(got, [-0.5])


def test_singular_endpoint_is_inserted_after_smooth_unbounded_filter(monkeypatch):
    import chebfunjax.fun.unbndfun as module

    smooth = Chebtech2(coeffs=jnp.asarray([0.5, 1.0, 0.5]))  # s(s+1)
    piece = Unbndfun(onefun=Singfun(smooth, (1.0, 0.0)),
                     domain=Domain((-jnp.inf, jnp.inf)), mapping_type="both_inf")
    original = module._unbounded_endpoint_filter
    observations = []

    def observe(roots, tech):
        observations.append(np.asarray(roots).tolist())
        return original(roots, tech)

    monkeypatch.setattr(module, "_unbounded_endpoint_filter", observe)
    got = np.asarray(piece.roots())
    assert observations == [[-1.0, 0.0]]
    assert got.shape == (2,)
    assert np.isneginf(got[0])
    assert got[1] == 0.0


def test_array_roots_forward_filter_per_column_and_keep_nan_padding():
    tech = Chebtech2(coeffs=jnp.asarray([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]))
    seen = []

    def identity_filter(roots, scalar_tech):
        seen.append((np.asarray(roots).tolist(), scalar_tech.coeffs.ndim))
        return roots

    roots = np.asarray(tech.roots(filter=identity_filter))
    assert len(seen) == 2
    assert all(ndim == 1 for _, ndim in seen)
    assert roots.shape == (2, 2)
    np.testing.assert_allclose(roots[:, 0], [0.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(roots[:, 1], [-1/np.sqrt(2), 1/np.sqrt(2)],
                               atol=2e-14, rtol=0)


def test_constant_zero_returns_before_filter_callback():
    tech = Chebtech2(coeffs=jnp.asarray([0.0]))
    calls = []
    roots = np.asarray(tech.roots(filter=lambda r, f: calls.append(True)))
    np.testing.assert_array_equal(roots, [0.0])
    assert calls == []


@pytest.mark.parametrize("tech_type", [Chebtech1, Chebtech2])
def test_identity_filter_preserves_unfiltered_roots(tech_type):
    tech = tech_type(coeffs=jnp.asarray([0.0, 0.0, 1.0]))
    plain = np.asarray(tech.roots())
    filtered = np.asarray(tech.roots(filter=lambda roots, _tech: roots))
    np.testing.assert_array_equal(filtered, plain)


def test_complex_endpoint_filter_uses_matlab_real_part_relations():
    roots = jnp.asarray([-0.5 + 0.25j, 0.25 + 0.5j, 0.5 - 0.25j])

    class NonzeroTech:
        vscale = 1.0

        def __call__(self, x):
            return jnp.ones_like(x, dtype=jnp.complex128)

    # Exercise the callback directly: endpoint values are nonzero, so this
    # case verifies complex values survive unchanged and stay in source order.
    from chebfunjax.fun.unbndfun import _unbounded_endpoint_filter
    out = _unbounded_endpoint_filter(roots, NonzeroTech())
    np.testing.assert_array_equal(out, roots)


def test_matlab_complex_min_tie_selection_matches_native_capture():
    from chebfunjax.fun.unbndfun import _matlab_min_complex_one

    theta = np.pi / 8
    values = jnp.asarray([np.exp(-1j * theta), np.exp(1j * theta), 1j, -1j])
    got = np.asarray(_matlab_min_complex_one(values**2))
    expected = np.asarray([np.exp(-1j * np.pi / 4), 1 + 0j, 1 + 0j, 1 + 0j])
    np.testing.assert_allclose(got, expected, atol=2e-16, rtol=0)


def test_complex_unbounded_maps_match_direct_native_table():
    from chebfunjax.fun.unbndfun import (
        _forward_both,
        _forward_left,
        _forward_right,
    )

    values = jnp.asarray([0.2 + 0.3j, 0.2 - 0.3j])
    np.testing.assert_allclose(
        _forward_both(values),
        [0.7789417136717699 + 1.5175933387053451j,
         0.7789417136717699 - 1.5175933387053451j],
        atol=3e-15, rtol=0,
    )
    np.testing.assert_allclose(
        _forward_left(values, 3.0),
        [-5.529411764705882 + 5.882352941176471j,
         -5.529411764705882 - 5.882352941176471j],
        atol=3e-15, rtol=0,
    )
    np.testing.assert_allclose(
        _forward_right(values, 2.0),
        [19.876712328767123 + 12.328767123287671j,
         19.876712328767123 - 12.328767123287671j],
        atol=3e-15, rtol=0,
    )


def test_complex_root_callback_receives_matlab_magnitude_phase_order(monkeypatch):
    import chebfunjax.tech.chebtech as module

    values = jnp.asarray([1 + 2j, 1 - 2j, -1 + 2j, -1 - 2j,
                          1j, -1j, 0.2 + 0.3j, 0.2 - 0.3j])
    expected = np.asarray([0.2 - 0.3j, 0.2 + 0.3j, -1j, 1j,
                           -1 - 2j, 1 - 2j, 1 + 2j, -1 + 2j])
    tech = Chebtech2(coeffs=jnp.asarray([1.0, 0.0, 1.0]))
    monkeypatch.setattr(module, "_roots_main", lambda *a, **k: values)
    seen = []

    def identity(roots, _tech):
        seen.extend(np.asarray(roots).tolist())
        return roots

    result = np.asarray(tech.roots(all_roots=True, filter=identity))
    np.testing.assert_array_equal(np.asarray(seen), expected)
    np.testing.assert_array_equal(result, expected)
