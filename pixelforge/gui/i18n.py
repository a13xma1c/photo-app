"""Minimal string localisation.

``tr("Open…")`` returns the translation for the active language or the English
source text.  Translations live in ``locales/<code>.json`` as ``{"English": "Übersetzung"}``
maps; the language comes from ``set_language`` (settings / ``PIXELFORGE_LANG``).
"""

from __future__ import annotations

import json
import locale
import os
from pathlib import Path

LOCALES_DIR = Path(__file__).with_name("locales")
LANGUAGES: dict[str, str] = {"en": "English", "de": "Deutsch", "fr": "Français", "es": "Español"}

_current = "en"
_table: dict[str, str] = {}


def available_languages() -> dict[str, str]:
    codes = {"en": "English"}
    for path in sorted(LOCALES_DIR.glob("*.json")):
        codes[path.stem] = LANGUAGES.get(path.stem, path.stem)
    return codes


def set_language(code: str | None) -> str:
    """Activate a language ("en" or a code with a locale file); returns the code actually used."""
    global _current, _table
    code = code or os.environ.get("PIXELFORGE_LANG") or _system_language()
    code = code.split("-")[0].split("_")[0].lower()
    path = LOCALES_DIR / f"{code}.json"
    if code == "en" or not path.exists():
        _current, _table = "en", {}
        return "en"
    try:
        data = json.loads(path.read_text("utf-8"))
        _table = {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        _table = {}
    _current = code
    return code


def _system_language() -> str:
    try:
        lang = locale.getlocale()[0] or os.environ.get("LANG", "") or "en"
    except (ValueError, TypeError):  # pragma: no cover - odd locale settings
        lang = "en"
    return lang or "en"


def current_language() -> str:
    return _current


def tr(text: str) -> str:
    """Translate a source string (identity when no translation exists)."""
    return _table.get(text, text)
