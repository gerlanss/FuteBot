"""
Persistencia simples de preferencias do usuario.

Hoje o projeto opera majoritariamente em modo single-user, entao as
preferencias tambem servem como filtro runtime do scanner/alertas.
"""

from __future__ import annotations

import json
from pathlib import Path


PREFS_PATH = Path(__file__).resolve().parent.parent / "data" / "user_prefs.json"


def load_preferences() -> dict:
    if not PREFS_PATH.exists():
        return {}
    try:
        return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_preferences(data: dict):
    PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_preferences(chat_id: int | str | None) -> dict:
    if chat_id is None:
        return {}
    return load_preferences().get(str(chat_id), {})


def get_runtime_preferences(default_chat_id: int | str | None = None) -> dict:
    prefs = load_preferences()
    if default_chat_id is not None and str(default_chat_id) in prefs:
        return prefs[str(default_chat_id)]
    if prefs:
        return next(iter(prefs.values()))
    return {}
