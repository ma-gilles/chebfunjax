"""Independent native expectations only for the finite tiny quotient class."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).with_name("data")


@lru_cache(maxsize=1)
def _native_components():
    expected = {}
    for filename in [
        "tangent_quotient_native_20261018.json",
        "tan_infinite_phase_native_20261018.json",
    ]:
        for row in json.loads((DATA / filename).read_text())["rows"]:
            if row["name"].startswith("provider_"):
                continue
            for x, y, r, i in zip(
                row["input"]["real"],
                row["input"]["imag"],
                row["output"]["real"],
                row["output"]["imag"],
            ):
                x, y = int(x, 16), int(y, 16)
                mx, my = x & 0x7FFFFFFFFFFFFFFF, y & 0x7FFFFFFFFFFFFFFF
                if mx >= 0x7FF0000000000000 or my >= 0x7FF0000000000000:
                    continue
                for part, magnitude, word in [(0, mx, r), (1, my, i)]:
                    if 0 < magnitude < 0x3C80000000000000:
                        key = row["method"], x, y, part
                        word = int(word, 16)
                        assert expected.get(key, word) == word
                        expected[key] = word
    return expected


def expected_tiny_native_components(name, value, original):
    expected = np.array(original, copy=True)
    if value.dtype != np.complex128:
        return expected
    x, y = value.real.copy().view(np.uint64), value.imag.copy().view(np.uint64)
    mx, my = x & np.uint64(0x7FFFFFFFFFFFFFFF), y & np.uint64(0x7FFFFFFFFFFFFFFF)
    finite = (mx < 0x7FF0000000000000) & (my < 0x7FF0000000000000)
    native = _native_components()
    for part, magnitude in enumerate([mx, my]):
        selected = finite & (magnitude > 0) & (magnitude < 0x3C80000000000000)
        component = (expected.real if part == 0 else expected.imag).copy().view(np.uint64)
        for index in zip(*np.nonzero(selected)):
            key = name, int(x[index]), int(y[index]), part
            assert key in native, key
            component[index] = np.uint64(native[key])
        if part == 0:
            expected.real = component.view(np.float64)
        else:
            expected.imag = component.view(np.float64)
    return expected
