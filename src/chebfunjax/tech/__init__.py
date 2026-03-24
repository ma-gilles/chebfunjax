"""Tech layer — smooth function representations on [-1, 1]."""

from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import (
    Trigtech,
    trig_coeffs2vals,
    trig_vals2coeffs,
    trigpts,
)

__all__ = [
    "Chebtech2",
    "Trigtech",
    "trigpts",
    "trig_vals2coeffs",
    "trig_coeffs2vals",
]
