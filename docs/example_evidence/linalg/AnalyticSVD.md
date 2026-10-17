# Python execution evidence for AnalyticSVD

[Reference example page](../../examples/linalg/AnalyticSVD.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `a5985bba0a4967cc6fead20f2fc804620b788f2e73e57d0e3a4b49343d106af2`.
Cached native HTML SHA-256: `ffdb3f4c4b5fee4c7ef0a8a076b5c84ca45ef0fc80d4b5393556c90e76a15094`.

## Recorded note 1

<!-- Exact original note SHA-256: 4db515822fcdddb8f466fb99a5276ee016b397b119bc37041cf4ea0598a74ff0 -->

**CPU verification:** The complete computation uses checked-in matrices captured
from MATLAB R2025b's literal `rng(10); A=randn(4,4); B=randn(4,4)` source
commands. All 32 input values match the native capture bit-for-bit. The run
completed 86 finite, resolved constructions and generated all five 608×271
figures; the timing above is its actual output. Historical LAPACK sign choices,
coefficient rounding and visual parity remain unverified. Visible differences
include singular-vector signs, axes limits and placement, and typography.
The coefficient plot was rendered again from the six saved coefficient arrays,
verified bit-for-bit, to keep all labels visible. The other four figures come
directly from the completed run; the timing remains that original full run.

<!-- End preserved historical qualification note. -->
