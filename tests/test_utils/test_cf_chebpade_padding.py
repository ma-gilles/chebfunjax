"""Source padding/degree contracts, not a MATLAB RNG bitstream assertion.

Provenance
----------
MATLAB source : @chebfun/chebpade.m (lines130–185), @chebfun/cf.m
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils import _cf_kernel as kernel
from chebfunjax.utils import _cf_padding as padding
from chebfunjax.utils import _randnfun as random_engine


@pytest.mark.parametrize("disabled", [False, True])
def test_exact_source_padding_and_no_draw_when_sufficient(monkeypatch, disabled):
    calls = []

    def draw(count):
        calls.append(count)
        return jnp.asarray([1., -2.])

    monkeypatch.setattr(padding, "_normal_draw", draw)
    with jax.disable_jit(disabled):
        c = jnp.asarray([1., .25])
        padded = padding._epsilon_pad(c, 4)
        assert jnp.array_equal(padded, jnp.asarray([1., .25, 2.**-52, -2.**-51]))
        assert jnp.array_equal(padding._epsilon_pad(c, 2), c)
        assert jnp.array_equal(padding._epsilon_pad(c, 1), c)
    assert calls == [2]


@pytest.mark.parametrize("disabled", [False, True])
def test_clenshaw_lord_one_pole_independent_formula(monkeypatch, disabled):
    # c=[1,eps,-2eps], beta=[1,-eps/2]. Terms quadratic in eps round
    # away here: source normalization gives p0=1 and q=[1,-eps].
    counts = []

    def draw(count):
        counts.append(count)
        return jnp.asarray([1., -2.])

    monkeypatch.setattr(padding, "_normal_draw", draw)
    with jax.disable_jit(disabled):
        p, q = kernel._chebpade_clenshaw_lord_jax(jnp.asarray([1.]), 0, 1)
    assert counts == [2]
    assert jnp.array_equal(p, jnp.asarray([1.]))
    assert jnp.array_equal(q, jnp.asarray([1., -2.**-52]))


@pytest.mark.parametrize("disabled", [False, True])
def test_zero_denominator_degree_uses_literal_beta_one(monkeypatch, disabled):
    monkeypatch.setattr(padding, "_normal_draw", lambda count: jnp.asarray([1., -2.]))
    with jax.disable_jit(disabled):
        p, q = kernel._chebpade_clenshaw_lord_jax(jnp.asarray([1., .25]), 3, 0)
    assert jnp.array_equal(p, jnp.asarray([1., .25, 2.**-52, -2.**-51]))
    assert jnp.array_equal(q, jnp.asarray([1.]))


def test_repeated_block_reduced_degrees_and_full_coefficients(monkeypatch):
    # Exact source index contract on seven singular-value membership flags.
    # Mocked getBlock tests branch wiring, NOT actual spectral reachability.
    k, ell, flag = kernel._scan_block_source([False, False, False, True, True, True, False], 4)
    assert (k, ell, flag) == (1, 1, True)
    monkeypatch.setattr(kernel, "_get_block_source", lambda *a: (0., None, k, ell, flag))
    observed = []

    def pade(c, m, n):
        observed.append((c, m, n))
        return jnp.asarray([1., 2.]), jnp.asarray([1., .5])

    monkeypatch.setattr(kernel, "_chebpade_clenshaw_lord_jax", pade)
    truncated = jnp.asarray([1., .3, .2, .1, .05, .02])
    full = jnp.asarray([1., .3, .2, .1, .05, .02, .01])
    p, q, error = kernel.cf_rational_small_jax(
        truncated, 2, 4, source_full_coeffs=full, source_vscale=2.,
    )
    assert len(observed) == 1
    assert observed[0][1:] == (1, 3)
    assert jnp.array_equal(observed[0][0], full)
    assert error == 2.**-52
    assert jnp.array_equal(p, jnp.asarray([1., 2.]))
    assert jnp.array_equal(q, jnp.asarray([1., .5]))


def test_stream_advances_only_for_missing_coefficients(monkeypatch):
    start = jax.random.key(6178)
    monkeypatch.setattr(random_engine, "_DEFAULT_KEY", start)
    c = jnp.asarray([1.])
    padding._epsilon_pad(c, 1)
    assert jnp.array_equal(jax.random.key_data(random_engine._DEFAULT_KEY), jax.random.key_data(start))
    state1, key1 = jax.random.split(start)
    state2, key2 = jax.random.split(state1)
    a = padding._epsilon_pad(c, 3)
    b = padding._epsilon_pad(c, 3)
    assert jnp.array_equal(a[1:], 2.**-52 * jax.random.normal(key1, (1, 2), dtype=jnp.float64)[0])
    assert jnp.array_equal(b[1:], 2.**-52 * jax.random.normal(key2, (1, 2), dtype=jnp.float64)[0])
    assert jnp.array_equal(jax.random.key_data(random_engine._DEFAULT_KEY), jax.random.key_data(state2))


def test_negative_reduced_degrees_still_rejected_without_rng(monkeypatch):
    def forbidden(count):
        raise AssertionError("invalid reduced degree consumed randomness")

    monkeypatch.setattr(padding, "_normal_draw", forbidden)
    with pytest.raises(kernel.UnsupportedCFBranch):
        kernel._chebpade_clenshaw_lord_jax(jnp.asarray([1.]), -1, 1)


@pytest.mark.parametrize("disabled", [False, True])
def test_default_randnfun_padding_interleave(monkeypatch, disabled):
    start = jax.random.key(9041)
    monkeypatch.setattr(random_engine, "_DEFAULT_KEY", start)
    state1, key1 = jax.random.split(start)
    state2, key2 = jax.random.split(state1)
    state3, key3 = jax.random.split(state2)
    with jax.disable_jit(disabled):
        first = random_engine.randnfun(float("inf"))
        padded = padding._epsilon_pad(jnp.asarray([1.]), 3)
        last = random_engine.randnfun(float("inf"))
    assert jnp.array_equal(jnp.asarray(first(0.)).reshape(()), jax.random.normal(key1, (1, 1), dtype=jnp.float64)[0, 0])
    assert jnp.array_equal(padded[1:], 2.**-52 * jax.random.normal(key2, (1, 2), dtype=jnp.float64)[0])
    assert jnp.array_equal(jnp.asarray(last(0.)).reshape(()), jax.random.normal(key3, (1, 1), dtype=jnp.float64)[0, 0])
    assert jnp.array_equal(jax.random.key_data(random_engine._DEFAULT_KEY), jax.random.key_data(state3))


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("explicit", ["seed", "key"])
def test_explicit_randomness_and_sufficient_padding_do_not_advance(monkeypatch, disabled, explicit):
    start = jax.random.key(319)
    monkeypatch.setattr(random_engine, "_DEFAULT_KEY", start)
    options = {"seed": 904} if explicit == "seed" else {"key": jax.random.key(904)}
    with jax.disable_jit(disabled):
        first = random_engine.randnfun(float("inf"), **options)
        c = jnp.asarray([1., .25])
        assert jnp.array_equal(padding._epsilon_pad(c, 2), c)
        assert jnp.array_equal(padding._epsilon_pad(c, 1), c)
        repeated = random_engine.randnfun(float("inf"), **options)
    assert jnp.array_equal(first(0.), repeated(0.))
    assert jnp.array_equal(jax.random.key_data(random_engine._DEFAULT_KEY), jax.random.key_data(start))
