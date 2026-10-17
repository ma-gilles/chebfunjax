# Python execution evidence for SphereHeatConduction

[Reference example page](../../examples/sphere/SphereHeatConduction.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `f1eb16dfe367c811482ed0d9eaf383ac72ede7d114d3ad84f9d044435c93a600`.
Cached native HTML SHA-256: `da0e7af8b8c87fe579616e3d9576f7a247dbd8f89cd187aa1a38ec66443a69e4`.

## Recorded note 1

<!-- Exact original note SHA-256: 6a782e0b358463b6e04b4655adb21e7b6fcd364fa33c35f41de646624965c96c -->

**CPU verification:** Both 100-step trajectories below completed using the public Helmholtz solver. The ten reference-sized figures now use the source camera, interpolated surface colors, website layout, visible time titles and blue mean contours. For the five-bump initial condition, all ten `rng(10)` uniforms and the resulting five Gaussian center coordinates match MATLAB R2025b bit-for-bit. This verifies the seed-10 initialization only; full-trajectory numerical and pixel-level visual parity remain unverified.

<!-- End preserved historical qualification note. -->
