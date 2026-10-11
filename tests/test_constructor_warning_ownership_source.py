"""Warning-owner source controls; all fits bounded at17 points."""

import os
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_lowlevel_silent(tech):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f = tech.from_function(lambda x: jnp.cos(80 * x), max_length=17, sample_test=False)
    assert not f.ishappy
    assert not caught


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("domain", [[-1, 1], [-1, 0, 1]])
def test_public_once(kind, domain):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f = cj.chebfun(
            lambda x: jnp.cos(80 * x),
            domain=domain,
            chebkind=kind,
            max_length=17,
            sample_test=False,
        )
    assert any(not p.ishappy for p in f.funs)
    assert len(caught) == 1
    assert "Function not resolved using 17 pts." in str(caught[0].message)


def test_callback_warning_preserved():
    def op(x):
        warnings.warn("callback sentinel", RuntimeWarning)
        return jnp.cos(80 * x)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cj.chebfun(op, max_length=17, sample_test=False)
    assert any(str(w.message) == "callback sentinel" for w in caught)
    assert sum("Function not resolved using" in str(w.message) for w in caught) == 1


def test_source_fit_receipt():
    records = {}
    caught_all = []
    for tech in [Chebtech1, Chebtech2]:
        calls = []

        def op(x):
            calls.append(np.asarray(x).copy())
            return jnp.cos(80 * x)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            f = tech.from_function(op, max_length=17, sample_test=False)
        records[tech.__name__ + "_coeffs"] = np.asarray(f.coeffs)
        for index, x in enumerate(calls):
            records[tech.__name__ + "_nodes_" + str(index)] = x
        caught_all.extend(caught)
    directory = os.environ.get("CHEBFUNJAX_CAPTURE_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "fit_receipt.npz", **records)
    assert not caught_all


@pytest.mark.parametrize("kind", [1, 2])
def test_split_limit_owner(kind):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f = cj.chebfun(  # noqa: F841 -- retain qualified construction binding
            lambda x: jnp.cos(80 * x),
            chebkind=kind,
            splitting=True,
            split_length=17,
            split_max_length=16,
            sample_test=False,
        )
    assert len(caught) == 1
    assert "Function not resolved using" in str(caught[0].message)
    assert "Have you tried" not in str(caught[0].message)


def test_legacy_public_owner_without_refit(monkeypatch):
    import importlib

    module = importlib.import_module("chebfunjax.chebfun1d.chebfun")
    tech = Chebtech2(coeffs=jnp.ones(17), ishappy=False)
    saved = cj.Chebfun(
        funs=[module._Piece(tech=tech, interval=(-1.0, 1.0))], domain=module.Domain((-1.0, 1.0))
    )
    calls = []

    def legacy(op, **kwargs):
        calls.append(kwargs)
        return saved

    monkeypatch.setattr(module, "_chebfun_build", legacy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = cj.chebfun(lambda x: jnp.cos(80 * x), exps=(0.0, 0.0), max_length=17)
    assert len(calls) == 1
    assert result is saved
    assert len(caught) == 1
    assert "Function not resolved using 17 pts." in str(caught[0].message)
