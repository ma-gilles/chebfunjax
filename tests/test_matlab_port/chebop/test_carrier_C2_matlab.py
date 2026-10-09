"""Literal MATLAB Carrier C2 clauses; previous controls kept separately.

Provenance
----------
MATLAB source: tests/chebop/test_carrier_C2.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
from tests.test_matlab_port.chebop._carrier_source import run_original_carrier


def test_original_carrier_c2(record_property):
    run_original_carrier('chebcolloc2', record_property)
