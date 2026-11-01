"""Entry point for the packaged application (PyInstaller runs this file)."""

import multiprocessing
import sys

from pixelforge.gui import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
