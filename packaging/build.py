"""Build a standalone PixelForge with PyInstaller.

    uv sync --group build
    uv run python packaging/build.py            # → dist/PixelForge(.app)
    uv run python packaging/build.py --onefile  # single executable (slower start-up)

The script wraps ``pyinstaller packaging/pixelforge.spec`` with a clean
``build/`` directory and prints where the result went.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    spec = ROOT / "packaging" / "pixelforge.spec"
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    command = [sys.executable, "-m", "PyInstaller", str(spec), "--noconfirm", "--clean", "--distpath", str(ROOT / "dist")]
    if "--onefile" in argv:
        # onefile needs the launcher directly (a spec already describes the layout)
        command = [
            sys.executable,
            "-m",
            "PyInstaller",
            str(ROOT / "packaging" / "launcher.py"),
            "--name",
            "PixelForge",
            "--onefile",
            "--windowed",
            "--noconfirm",
            "--clean",
            "--add-data",
            f"{ROOT / 'pixelforge' / 'gui' / 'fonts'}{';' if sys.platform == 'win32' else ':'}pixelforge/gui/fonts",
            "--add-data",
            f"{ROOT / 'pixelforge' / 'gui' / 'locales'}{';' if sys.platform == 'win32' else ':'}pixelforge/gui/locales",
            "--distpath",
            str(ROOT / "dist"),
        ]
    print(" ".join(command))
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode == 0:
        target = ROOT / "dist" / ("PixelForge.app" if sys.platform == "darwin" else "PixelForge")
        print(f"\nDone: {target}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
