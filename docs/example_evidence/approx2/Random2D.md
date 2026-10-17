# Python execution evidence for Random2D

[Reference example page](../../examples/approx2/Random2D.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `143531bfac56311d6ad57d3829910043d8d7d1f5e70a446cf4cca89370bc36fb`.
Cached native HTML SHA-256: `ca5d389dabdc122afab0f84837db753d6dea6d52f87dd8a9e8664a09ba099128`.

## Recorded note 1

<!-- Exact original note SHA-256: 9168a0fc518fdda1ab2bbe7339cc863c65c3587d6e94f8ae535ae75f015d4282 -->

**Python rendering note.** The 3D panel uses the source-derived 200-point surface grid, Parula colors, and measured `camlight left` material/light settings. Its viewport now follows the captured MATLAB normalized axes position, uses orthographic projection, and sets the sampled x/y domain limits explicitly. Automatic camera aperture and data-aspect framing for this random surface remain unqualified; no camera constants from a different test mesh are used. Python uses seeded JAX PRNG keys rather than MATLAB `rng(0)`: MATLAB seeds once and advances one stream, while this translation resets separate seeds for each new random function. Interpolated vertex colors are represented by per-quad face colors. These differences mean the panel is not pixel-equivalent to the MATLAB page.

<!-- End preserved historical qualification note. -->
