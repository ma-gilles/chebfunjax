# Python execution evidence for EigLandscapes

[Reference example page](../../examples/linalg/EigLandscapes.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `172fc63140c02fb99e36dc7340a93f5fdfbc7495563b893b34187998091e44d3`.
Cached native HTML SHA-256: `6796fedeeb441be2c1324409f34a6f8ad4e754753fb30557da1f0cd2a12438d4`.

## Recorded note 1

<!-- Exact original note SHA-256: 9d7ce185f970ec516fa154cb6f9631f5c5ce319914c2766b0b45146ba831e35a -->

> **Python execution qualification.** The adaptive Hermitian constructions and full 512×512 real-symmetric constructions execute the source eigenvalue and gap computations. NumPy RandomState draws differ from MATLAB rng(1)/rng(8), and JAX eigvalsh differs from the native eig/sort backend. The plotted minima and crossing location belong to these Python draws. Matplotlib sampling, palette, and lighting differ from native rendering; exact pixel and RNG parity are unverified.

<!-- End preserved historical qualification note. -->
