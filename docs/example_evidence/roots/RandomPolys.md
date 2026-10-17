# Python execution evidence for RandomPolys

[Reference example page](../../examples/roots/RandomPolys.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `f713e8ccaa5cce95911d48f7fff8931f82fd64be99de2677022e76271e8cf3e1`.
Cached native HTML SHA-256: `a37e6fa778af47cbfdd31704174f751bde89a4f390dbe52a63dc8f3ff77dd3ca`.

## Recorded note 1

<!-- Exact original note SHA-256: b1f83d8523d5ca18926bb0ae68e4c9c6c8022a3c74df536b1af3dc694ad7e31d -->

> Execution qualification: The full degree-30 and ten degree-1000 computations use exact captured MATLAB R2025b `rng('default')` inputs through public chebfunjax transforms, constructors and roots. All ten printed fractions and the mean agree with the cached native output. This is a fixed input replay, not general RNG implementation parity. Figures were repaired from the saved degree-30 state; exact pixel parity and coefficient/root bit parity remain unverified. The final script is qualified against source baseline `b08fbe16`, including the endpoint-mapping correction; its saved numerical arrays and figures are unchanged from the earlier replay.

<!-- End preserved historical qualification note. -->
