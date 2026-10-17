# Python execution evidence for LaplaceBall

[Reference example page](../../examples/sphere/LaplaceBall.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `ac72cf09e622c29c65fbe78f86530a60b0c3e8fee3655e719f50e2106992582b`.
Cached native HTML SHA-256: `bd153eab45ee2514b8386083b22dc5e14df7e255fb4a5e8f557488a3d6c83a9e`.

## Recorded note 1

<!-- Exact original note SHA-256: a691249f0754ee8c74a03e9b699e3e445f49632975e347d6efe6124b923e848e -->

> Execution qualification: The full public boundary construction, Poisson solve on the source-derived grid (65 in each direction), and inner-sphere extraction were executed using one captured MATLAB R2025b `rng(1)` input vector. This fixture does not establish general RNG parity. The boundary length is 65×65 versus the captured native 65×63; printed Oxford, origin, and inner-mean values differ in their final digits. The Python Cartesian boundary point is evaluated through its equivalent longitude/polar coordinates. The full current script, including rendering, has been rerun with retained numerical captures; colorbars are present. [Current execution evidence](../../laplaceball_current_cpu_20261017.json). Ball slices now use shared triangle depth ordering and Cartesian vertex lighting; the explicit 64-color Parula palette is supported by the retained reference colorbar and MATLAB source. Native camera, exact slice color limits, historical Phong shading, and exact pixel parity remain unverified.

<!-- End preserved historical qualification note. -->
