# Python execution evidence for Needle

[Reference example page](../../examples/opt/Needle.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `ca4b6e7ac5a6f3e8f4eb1fccc960a8dbc348ee5ec33243319214aaf374bd7459`.
Cached native HTML SHA-256: `fbe17770e7582c4d4acbbf1db194c8d3b4484a9c89281b73477133c333c8885b`.

## Recorded note 1

<!-- Exact original note SHA-256: 8f6b3e3360f33f2d63622e6cb37d63b8c7e5f18e2693ad4dcdf93ef9bc6d3785 -->

**Qualification.** Source grid coordinates match the captured MATLAB R2025b coordinates bit for bit. Python fminsearch used 128 iterations / 266 evaluations, versus native 136 / 289; final coordinates differ by at most 1.472753891e-9. The palette and layout approximate the reference; historical rendering and pixel parity are unverified. Printed timers include observers on a shared host and are not comparative benchmarks.

<!-- End preserved historical qualification note. -->
