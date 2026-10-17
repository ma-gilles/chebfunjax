# Python execution evidence for ZetaZeros

[Reference example page](../../examples/complex/ZetaZeros.md)

These qualification notes were moved out of the reference example prose.
They describe their recorded executions and do not establish current full-page parity.

Original page commit: `7098a7a7a94e1d8d998e8ba4076bc5a07f673c78`.
Original Markdown SHA-256: `ddea2649775bbbcd5905fd6fd573edd8437deeef85312d2bcc27af0da4cae025`.
Cached native HTML SHA-256: `0e38bcd7994c92569a8ba5f52fd8c55ed707d9e2ef9f31f40a118123d67bd12f`.

## Recorded note 1

<!-- Exact original note SHA-256: ddf7a944bd50ef6342183995b0ba1f66a75c9481656b57639b6e30038d2c8256 -->

## Execution qualification

## Recorded note 2

<!-- Exact original note SHA-256: 72244caa7384a4f36afe20e4ff8efc27a6332490c39907fabe64a36ddd3343ff -->

The full 100000-term construction, eight complex roots with recursion disabled, and public Chebfun composition were executed with JAX 0.11.0 on CPU. The computed function has length 75. The eight printed computed rows differ from cached native rows by at most 1.000000082740371e-10; the full computed roots differ from the source ten-decimal comparison constants by at most 2.3169648471547077e-9. JAX reduction and complex-power rounding can differ from MATLAB. Exact coefficient, root-bit and pixel parity remain unverified. Both figures were visually reviewed for readable content. The 7.322813-second source timer is a shared-host observation, not a benchmark.

## Recorded note 3

<!-- Exact original note SHA-256: 89801b9209515d2e9bd3c2edf1701ae2b529395a33d37d980350c634090d8be6 -->

The actual public composition emitted this warning:

<!-- End preserved historical qualification note. -->
