"""Tests for chebfunjax.pref — global preferences management."""

from __future__ import annotations

import threading

import pytest

from chebfunjax.pref import ChebPreferences, pref


class TestDefaults:
    """Default values match MATLAB Chebfun factory defaults."""

    def test_eps_is_machine_epsilon(self):
        assert pref.eps == 2.220446049250313e-16

    def test_max_length(self):
        assert pref.max_length == 65537

    def test_tech(self):
        assert pref.tech == "chebtech2"

    def test_domain(self):
        assert pref.domain == (-1.0, 1.0)

    def test_chop_tol_default_none(self):
        assert pref.chop_tol is None


class TestGetSet:
    """Basic get/set behaviour."""

    def teardown_method(self):
        pref.reset()

    def test_set_and_get_eps(self):
        pref.eps = 1e-10
        assert pref.eps == 1e-10

    def test_set_and_get_max_length(self):
        pref.max_length = 1000
        assert pref.max_length == 1000

    def test_set_and_get_domain(self):
        pref.domain = (0.0, 2.0)
        assert pref.domain == (0.0, 2.0)

    def test_set_invalid_name_raises(self):
        with pytest.raises(AttributeError, match="not a valid chebfunjax preference"):
            pref.bogus_pref = 42  # type: ignore[attr-defined]

    def test_get_invalid_name_raises(self):
        with pytest.raises(AttributeError, match="not a valid chebfunjax preference"):
            _ = pref.bogus_pref  # type: ignore[attr-defined]


class TestReset:
    """reset() restores factory defaults."""

    def teardown_method(self):
        pref.reset()

    def test_reset_all(self):
        pref.eps = 1e-5
        pref.max_length = 100
        pref.reset()
        assert pref.eps == 2.220446049250313e-16
        assert pref.max_length == 65537

    def test_reset_single(self):
        pref.eps = 1e-5
        pref.max_length = 100
        pref.reset("eps")
        assert pref.eps == 2.220446049250313e-16
        assert pref.max_length == 100  # unchanged

    def test_reset_invalid_name_raises(self):
        with pytest.raises(AttributeError, match="not a valid chebfunjax preference"):
            pref.reset("nonexistent")


class TestContextManager:
    """context() provides temporary overrides and restores on exit."""

    def teardown_method(self):
        pref.reset()

    def test_context_overrides(self):
        original_eps = pref.eps
        with pref.context(eps=1e-10):
            assert pref.eps == 1e-10
        assert pref.eps == original_eps

    def test_context_multiple_overrides(self):
        with pref.context(eps=1e-10, max_length=1000):
            assert pref.eps == 1e-10
            assert pref.max_length == 1000
        assert pref.eps == 2.220446049250313e-16
        assert pref.max_length == 65537

    def test_context_invalid_key_raises(self):
        with pytest.raises(AttributeError, match="not a valid chebfunjax preference"):
            with pref.context(bad_key=1):
                pass  # pragma: no cover

    def test_context_restores_on_exception(self):
        original_eps = pref.eps
        with pytest.raises(RuntimeError):
            with pref.context(eps=1e-10):
                assert pref.eps == 1e-10
                raise RuntimeError("boom")
        assert pref.eps == original_eps

    def test_nested_context(self):
        with pref.context(eps=1e-8):
            assert pref.eps == 1e-8
            with pref.context(eps=1e-12):
                assert pref.eps == 1e-12
            assert pref.eps == 1e-8
        assert pref.eps == 2.220446049250313e-16

    def test_context_yields_pref(self):
        with pref.context(eps=1e-10) as p:
            assert p is pref
            assert p.eps == 1e-10

    def test_mutation_inside_context_does_not_leak(self):
        with pref.context(eps=1e-8):
            pref.max_length = 999
            assert pref.max_length == 999
        # After context exit, max_length should be back to default
        assert pref.max_length == 65537


class TestRepr:
    """repr() is informative."""

    def test_repr_contains_key_info(self):
        r = repr(pref)
        assert "ChebPreferences(" in r
        assert "eps=" in r
        assert "max_length=" in r
        assert "tech=" in r
        assert "domain=" in r

    def test_to_dict(self):
        d = pref.to_dict()
        assert isinstance(d, dict)
        assert d["eps"] == 2.220446049250313e-16
        assert d["max_length"] == 65537


class TestThreadSafety:
    """context() overrides are isolated between threads."""

    def teardown_method(self):
        pref.reset()

    def test_thread_isolation(self):
        """Overrides in one thread do not affect another thread."""
        barrier = threading.Barrier(2)
        results: dict[str, float] = {}

        def thread_a():
            with pref.context(eps=1e-3):
                barrier.wait(timeout=5)
                results["a"] = pref.eps
                barrier.wait(timeout=5)

        def thread_b():
            barrier.wait(timeout=5)
            results["b"] = pref.eps
            barrier.wait(timeout=5)

        t_a = threading.Thread(target=thread_a)
        t_b = threading.Thread(target=thread_b)
        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

        assert results["a"] == 1e-3
        assert results["b"] == 2.220446049250313e-16


class TestSingleton:
    """The module-level ``pref`` is the intended access point."""

    def test_module_level_pref_is_chebpreferences(self):
        assert isinstance(pref, ChebPreferences)

    def test_separate_instances_share_state(self):
        """Two ChebPreferences instances see the same global state."""
        other = ChebPreferences()
        pref.max_length = 9999
        assert other.max_length == 9999
        pref.reset()
