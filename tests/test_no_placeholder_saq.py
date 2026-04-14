# tests/test_no_placeholder_saq.py
"""Guard: the pure-Python SAQ scaffold must stay removed.

It was deleted in commit 1f59a0c (April 2026) and replaced by SaqIndex (the
C++ engine wrapper). It subsequently got re-materialised on a shared clone via
a stray ``git stash pop``, producing ~4 hours of misleading benchmark numbers
before detection. This test locks in the invariant: importing the scaffold
must fail with ImportError.
"""

from __future__ import annotations

import pytest


def test_pure_python_saq_module_not_importable():
    with pytest.raises(ImportError):
        from haag_vq.methods.saq import SAQ  # noqa: F401
