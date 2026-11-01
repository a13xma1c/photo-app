# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: a self-contained PixelForge for the current platform.

    uv sync --group build
    uv run pyinstaller packaging/pixelforge.spec --noconfirm

Produces ``dist/PixelForge/`` (Windows / Linux) or ``dist/PixelForge.app``
(macOS).  Only the Qt modules PixelForge uses are bundled, which keeps the
result around a quarter of a full PySide6 install.
"""

import sys
from pathlib import Path


ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - provided by PyInstaller
APP_NAME = "PixelForge"

datas = [
    (str(ROOT / "pixelforge" / "gui" / "fonts"), "pixelforge/gui/fonts"),
    (str(ROOT / "pixelforge" / "gui" / "locales"), "pixelforge/gui/locales"),
]

excludes = [
    # Qt modules the app never imports - dropping them saves ~150 MB
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebChannel",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtNetwork",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtSerialPort",
    "PySide6.QtSensors",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtStateMachine",
    "PySide6.QtSvgWidgets",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtUiTools",
    "PySide6.QtXml",
    "PySide6.QtConcurrent",
    "PySide6.QtDBus",
    "PySide6.QtPrintSupport",
    "PySide6.QtTextToSpeech",
    "PySide6.QtWebSockets",
    "PySide6.QtHttpServer",
    "PySide6.QtSpatialAudio",
    "PySide6.QtGraphs",
    "PySide6.QtGraphsWidgets",
    "tkinter",
    "matplotlib",
    "scipy",
    "pytest",
    "mypy",
    "ruff",
]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=["PIL.ImageCms", "PIL.ImageFilter", "PIL.ImageChops", "PIL.ImageOps", "PIL.ImageDraw", "PIL.ImageFont"],
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

icon = None
if sys.platform == "darwin" and (ROOT / "packaging" / "icon.icns").is_file():
    icon = str(ROOT / "packaging" / "icon.icns")
elif sys.platform == "win32" and (ROOT / "packaging" / "icon.ico").is_file():
    icon = str(ROOT / "packaging" / "icon.ico")

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=icon,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(  # noqa: F821
        coll,
        name=f"{APP_NAME}.app",
        icon=icon,
        bundle_identifier="dev.pixelforge.app",
        info_plist={
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "CFBundleShortVersionString": "2.1.0",
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "Picture",
                    "CFBundleTypeRole": "Editor",
                    "LSHandlerRank": "Alternate",
                    "LSItemContentTypes": ["public.png", "public.jpeg", "public.tiff", "org.webmproject.webp", "com.microsoft.bmp"],
                },
                {
                    "CFBundleTypeName": "PixelForge Project",
                    "CFBundleTypeRole": "Editor",
                    "LSHandlerRank": "Owner",
                    "CFBundleTypeExtensions": ["pfp"],
                },
            ],
        },
    )
