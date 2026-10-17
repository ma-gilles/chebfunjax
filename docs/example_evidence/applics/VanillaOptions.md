# Python execution evidence for VanillaOptions

[Reference example page](../../examples/applics/VanillaOptions.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `c61d34222f2924bde72bfec3d0563c1c663fec35a6af115faee60b97bee3c8e1`.
Cached native HTML SHA-256: `4c96ef8285128934b2d9c150e7b2ddd0640279659788860c1d489aefde76e718`.

## Recorded note 1

<!-- Exact original note SHA-256: 88847e5b1a9b35ce6de6da100b473be0f07550e15c6f20abb66c2f8ef3c1a598 -->

> **Python execution qualification.** All source price, hedge, surface and six boundary computations run through public Chebfun operations on the original domains. Pinned MATLAB also returns an out-of-domain, nonzero-residual boundary at rate 0.0405; the source iteration is preserved. Exact curve coordinates, native rendering and pixel parity remain unverified.

<!-- End preserved historical qualification note. -->
