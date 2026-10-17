# Python execution evidence for MinimaxSqrt

[Reference example page](../../examples/approx/MinimaxSqrt.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `3051afb7eaf8d28c1325d12b9f07a0d86a2c8738f3aa5737a5a4d7d65f428ab1`.
Cached native HTML SHA-256: `daa3d58469b448b5faa3de5ca6407801a85e2720c0e5d7aeeb242f377cbaca38`.

## Recorded note 1

<!-- Exact original note SHA-256: 5fdc4376eafa347cfa796d31ce892253d3c8270783635859b5a7800b87f95713 -->

> **Execution qualification:** Seven public source constructors and all 116 approximation calls completed on frozen commit `288e3ddf`, using a fresh serial worker for each fit. The plots retain actual errors and all 11 nonconvergence warnings; no cumulative minimum or fallback approximation replaces them. The 6/6 and 7/7 matched-input native errors differ slightly, and no full native error table or pixel parity is established. This qualifies the recorded source calls, not one completed monolithic script process or the newer current-head roots implementation. Native matrix-conditioning warnings are not emitted by this backend.

<!-- End preserved historical qualification note. -->
