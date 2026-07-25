"""Port of MATLAB Chebfun tests/misc/test_scribble.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_scribble.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

from chebfunjax.utils.scribble import scribble


class TestScribble:
    def test_supported_characters_no_warning(self):
        # alphanumerics + . , ? ! ' - + and space are supported
        c = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!'- +"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s = scribble(c)
        assert s is not None

    def test_full_matlab_character_set(self):
        c = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:?!'\"`_ )([]{}"
             "-+*/^=<>\\|%#~@$&")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scribble(c)
