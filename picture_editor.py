#!/usr/bin/env python3
"""Launcher kept for backwards compatibility.

Run:  python3 picture_editor.py [picture]
(equivalent to ``python3 -m pixelforge``)
"""

import sys

from pixelforge.gui import main

if __name__ == "__main__":
    sys.exit(main())
