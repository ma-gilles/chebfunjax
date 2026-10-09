"""Literal MATLAB Carrier US clauses; previous controls kept separately.

Provenance
----------
MATLAB source: tests/chebop/test_carrier_US.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
from tests.test_matlab_port.chebop._carrier_source import run_original_carrier


def test_original_carrier_us(record_property):
    run_original_carrier('ultraS', record_property)
