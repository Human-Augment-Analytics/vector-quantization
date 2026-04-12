"""Tests for the pure-Python SAQ placeholder.

The pure-Python saq.py module has been removed (Task 19 cleanup).
These tests are kept as historical reference but are skipped unconditionally.
The SAQ C++ wheel is tested via tests/test_saq_index.py (SaqIndex wrapper).
"""

import pytest

pytest.importorskip(
    "haag_vq.methods.saq",
    reason="Pure-Python saq.py has been removed; use SaqIndex (C++ wheel) instead.",
)
