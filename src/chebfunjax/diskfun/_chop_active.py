"""Reuse disk standardChop compilation after the source FFT is complete.

Provenance
----------
Chebfun standardChop.m, pin7574c77680d7e82b79626300bf255498271a72df;
the existing utils.misc._standard_chop_kernel translation supplies the
arithmetic and source indices. Only the physical buffer length is separated
from its active length. This helper receives at least17 active magnitudes and
zero padding. It never pads Fourier or Chebyshev coefficients before an FFT.

Inactive magnitudes are masked to minus infinity before the maximum scan. Source n still
bounds lookahead windows, fallback cutoffs and indexed endpoints. Threshold
counts exclude padding explicitly, including when the threshold underflows.
The minimization excludes indices beyond the source window. Thus capacity
controls compilation reuse while n retains the source stopping semantics.
"""

# Keep the existing private import path while sharing the source arithmetic.
from chebfunjax.utils.misc import _standard_chop_kernel as _standard_chop_active

__all__ = ["_standard_chop_active"]
