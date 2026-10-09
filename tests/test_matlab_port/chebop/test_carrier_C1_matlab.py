"""Literal MATLAB Carrier C1 clauses; previous controls kept separately.

Provenance
----------
MATLAB source: tests/chebop/test_carrier_C1.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
from tests.test_matlab_port.chebop._carrier_source import run_original_carrier


def test_original_carrier_c1(record_property):
    run_original_carrier('chebcolloc1', record_property)
