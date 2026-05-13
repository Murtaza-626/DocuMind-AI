"""Force UTF-8 as the default text encoding on Windows.

Import this module BEFORE any library that reads package metadata
(trl, transformers, datasets, etc.).  Without this fix, libraries
that read METADATA / PKG-INFO files containing non-ASCII characters
(e.g. author names with accents) will crash with:

    'charmap' codec can't decode byte 0x81 in position 932

The fix patches FOUR separate entry points because:
  - builtins.open  -> normal open() calls in user code
  - io.open        -> pathlib.Path.open() delegates here (C-level)
  - pathlib.Path.read_text -> importlib.metadata reads METADATA via this
  - pathlib.Path.open      -> extra safety net
"""

import builtins
import io
import pathlib
import sys

if sys.platform == "win32" and not getattr(builtins, "_utf8_patched", False):

    # ── 1) Patch builtins.open ───────────────────────────────────────
    _original_builtin_open = builtins.open

    def _utf8_builtin_open(file, mode="r", buffering=-1, encoding=None,
                           errors=None, newline=None, closefd=True,
                           opener=None):
        if encoding is None and isinstance(mode, str) and "b" not in mode:
            encoding = "utf-8"
            if errors is None:
                errors = "replace"
        return _original_builtin_open(
            file, mode=mode, buffering=buffering, encoding=encoding,
            errors=errors, newline=newline, closefd=closefd, opener=opener,
        )

    builtins.open = _utf8_builtin_open

    # ── 2) Patch io.open (used by pathlib internally) ────────────────
    _original_io_open = io.open

    def _utf8_io_open(file, mode="r", buffering=-1, encoding=None,
                      errors=None, newline=None, closefd=True, opener=None):
        if encoding is None and isinstance(mode, str) and "b" not in mode:
            encoding = "utf-8"
            if errors is None:
                errors = "replace"
        return _original_io_open(
            file, mode=mode, buffering=buffering, encoding=encoding,
            errors=errors, newline=newline, closefd=closefd, opener=opener,
        )

    io.open = _utf8_io_open

    # ── 3) Patch pathlib.Path.read_text ──────────────────────────────
    _original_read_text = pathlib.Path.read_text

    def _utf8_read_text(self, encoding=None, errors=None):
        if encoding is None:
            encoding = "utf-8"
            if errors is None:
                errors = "replace"
        return _original_read_text(self, encoding=encoding, errors=errors)

    pathlib.Path.read_text = _utf8_read_text

    # ── 4) Patch pathlib.Path.open ───────────────────────────────────
    _original_path_open = pathlib.Path.open

    def _utf8_path_open(self, mode="r", buffering=-1, encoding=None,
                        errors=None, newline=None):
        if encoding is None and isinstance(mode, str) and "b" not in mode:
            encoding = "utf-8"
            if errors is None:
                errors = "replace"
        return _original_path_open(
            self, mode=mode, buffering=buffering, encoding=encoding,
            errors=errors, newline=newline,
        )

    pathlib.Path.open = _utf8_path_open

    # ── Guard flag ───────────────────────────────────────────────────
    builtins._utf8_patched = True  # type: ignore[attr-defined]
