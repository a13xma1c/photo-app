"""Entry point for the desktop application (``python -m pixelforge``)."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    try:
        from .qt import run
    except ImportError as exc:  # PySide6 missing
        sys.stderr.write(f"PixelForge needs PySide6 for its interface ({exc}).\nInstall it with:  pip install PySide6-Essentials\n")
        return 1
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
