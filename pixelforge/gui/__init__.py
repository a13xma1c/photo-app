"""Desktop front-end for PixelForge (Qt shell around a Pillow-rendered glass UI)."""

from __future__ import annotations

__all__ = ["main"]


def main(argv=None) -> int:  # lazy import keeps `pixelforge.core` usable without Qt
    from .app import main as _main

    return _main(argv)
