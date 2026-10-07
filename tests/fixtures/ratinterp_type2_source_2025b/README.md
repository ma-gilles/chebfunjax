# TYPE2 rational evaluator source captures

Expected outputs only. Captured with MATLAB R2025b from standalone copies of
Chebfun7574c77680d7e82b79626300bf255498271a72df ratinterp.m local
constructRatApproxCheb2/ratbary, preserving their arithmetic. This is helper-derived
source evidence, not an untouched test_ratinterp.m aggregate or a full public
fitting oracle. The three JSON files are byte-identical to the original capture.

Source observation archive SHA256:
b0fd61e4b4e6d1b9e8612f92aac8f958417ad92f782d2d9bd8d5b19199df94c5

Both real and imaginary binary64 words are stored in MATLAB column-major order.
Tests reconstruct expected outputs only. Query and polynomial inputs are literal
expressions in test_ratinterp_type2_source_capture.py; captured query/value/degree
arrays are not supplied as fitted constructor inputs. Bounds belong to new
independent controls; no original MATLAB assertions are weakened.
