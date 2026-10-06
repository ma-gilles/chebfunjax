"""Bounded source Gamma callback controls; outputs are assertion-only.

Provenance
----------
MATLAB source : built-in gamma, actual R2025b primitive capture
Chebfun commit: 7574c77
Finite off-pole controls preserve the existing SciPy callback exactly.
"""
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from scipy.special import gamma

from tests.test_matlab_port.chebfun.test_singular_ops_matlab import _matlab_gamma_callback

FIXTURE = json.loads(Path(__file__).with_name("gamma_matlab_primitive_fixture.json").read_text())
POLES = [r for r in FIXTURE["inputs"] if r["label"] in ("-4", "-3", "-2", "-1", "-0", "+0")]


@pytest.mark.parametrize("item", POLES, ids=[item["label"] for item in POLES])
def test_actual_matlab_gamma_pole_payload(item):
    x = struct.unpack(">d", bytes.fromhex(item["input_hex"]))[0]
    expected = next(r for r in FIXTURE["expected"] if r["label"] == item["label"])
    assert struct.pack(">d", float(_matlab_gamma_callback(x))).hex() == expected["output_hex"]


def test_off_pole_backend_identity():
    x = np.array([.25, -.25, -1.5, -3.5, np.nextafter(-3., -np.inf), np.nextafter(-3., np.inf)])
    actual = np.asarray(_matlab_gamma_callback(x))
    assert np.array_equal(actual.view(np.uint64), gamma(x).view(np.uint64))


def test_subnormal_nonpole_backend_identity():
    x = np.array([-np.nextafter(0., 1.), np.nextafter(0., 1.)])
    actual = np.asarray(_matlab_gamma_callback(x))
    assert np.array_equal(actual.view(np.uint64), gamma(x).view(np.uint64))
