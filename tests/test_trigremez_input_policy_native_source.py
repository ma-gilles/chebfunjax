"""Native source guards/precedence/options; Chebfun7574c77680d7e82b79626300bf255498271a72df."""

import json
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.utils.trigrational import TrigRemezError

FIXTURES = Path(__file__).parent / "fixtures/trigremez_input_policy_source_2025b"


def _case(case):
    m = n = 1
    if case == 1:
        f = cj.chebfun(lambda x: 1j * x)
    elif case == 2:
        f = cj.chebfun(lambda x: jnp.column_stack((x, x * x)))
    elif case == 3:
        x = cj.chebfun("x")
        f = Quasimatrix([x, abs(x)], x.domain)
    elif case == 4:
        f = cj.chebfun(lambda x: jnp.sqrt(x + 1), exps=(0.5, 0))
    elif case == 5:
        f = cj.chebfun(lambda x: jnp.exp(-x * x), domain=(-jnp.inf, jnp.inf))
    elif case == 6:
        f = cj.dirac(cj.chebfun("x"))
    elif case == 7:
        f = cj.chebfun(lambda x: jnp.column_stack((1j * x, x * x)))
    elif case == 8:
        f = 1j * cj.chebfun(lambda x: jnp.sqrt(x + 1), exps=(0.5, 0))
    elif case in [9, 10, 11, 12]:
        f = cj.chebfun("x")
        if case == 9:
            m = -1
        if case == 10:
            m = 1.2
        if case == 11:
            n = -1
        if case == 12:
            n = 1.2
    else:
        f = cj.chebfun(lambda x: jnp.column_stack((1j * x, jnp.abs(x))), splitting=True)
    return f, m, n


@pytest.mark.parametrize("case", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14])
def test_native_guard_identifier_and_precedence(case):
    metadata = json.loads((FIXTURES / "guards.json").read_text())["cases"][str(case)]
    f, m, n = _case(case)
    with pytest.raises(TrigRemezError) as caught:
        cj.trigremez(f, m, n)
    assert caught.value.identifier == metadata["error_id"]
    assert str(caught.value) == metadata["error_id"] + ": " + metadata["error_message"]


def test_native_empty_early_return_before_norm_and_invalid_degrees(monkeypatch):
    f = cj.Chebfun.empty()

    def forbidden(*args, **kwargs):
        pytest.fail("native empty input returns before norm")

    monkeypatch.setattr(cj.Chebfun, "norm", forbidden)
    assert cj.trigremez(f, -1, 1.2) is f


@pytest.mark.parametrize("case,options", [(15, {"max_iter": 1}), (16, {"tol": 1e-3})])
def test_native_existing_option_adapters(case, options):
    f = abs(cj.chebfun("x"))
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        p, q, evaluator, error, status = cj.trigremez(f, 2, 2, **options)
    with np.load(FIXTURES / f"case{case}.npz") as native:
        x = jnp.asarray(native["sample_x"])
        np.testing.assert_allclose(evaluator(x), native["r_values"], rtol=0, atol=1e-12)
        assert status["iter"] == int(native["status_iter"])
    metadata = json.loads((FIXTURES / "guards.json").read_text())["cases"][str(case)]
    if metadata["warning_id"]:
        assert len(emitted) == 1
        assert (
            str(emitted[0].message) == metadata["warning_id"] + ": " + metadata["warning_message"]
        )
    else:
        assert not emitted
