"""The cast controls a reference-channel node exposes, so the render gate can force each one."""

from __future__ import annotations

from typing import Any

from ..characters.sheet import SheetOptions
from ..graph.descriptor import Option, ParamField, Widget


def cast_param_fields() -> tuple[ParamField, ...]:
    """Advanced: they only matter with two or more characters wired, and auto is the default."""
    return (
        ParamField(
            "cast_mode", "Cast placement (2+ characters)", Widget.SELECT, "auto",
            options=(
                Option("auto", "Auto"),
                Option("refs", "References per character"),
                Option("face-each", "One face reference each"),
                Option("sheet", "Character sheet"),
            ),
            advanced=True,
        ),
        ParamField(
            "cast_sheet_captions", "Name captions on the sheet", Widget.BOOLEAN, True,
            advanced=True,
        ),
        ParamField(
            "cast_sheet_split", "Two half-sheets for more than 3 characters", Widget.BOOLEAN, False,
            advanced=True,
        ),
        ParamField(
            "cast_sheet_faces_only", "Faces only on the sheet", Widget.BOOLEAN, False,
            advanced=True,
        ),
    )


def cast_mode(params: dict[str, Any]) -> str:
    return str(params.get("cast_mode") or "auto")


def sheet_options(params: dict[str, Any]) -> SheetOptions:
    return SheetOptions(
        captions=bool(params.get("cast_sheet_captions", True)),
        split=bool(params.get("cast_sheet_split", False)),
        faces_only=bool(params.get("cast_sheet_faces_only", False)),
    )


def character_files(inputs: dict[str, Any]) -> list[str]:
    """Every wired character's filename, in wire order; an unsaved one is an error, not a skip."""
    names: list[str] = []
    wires: list[Any] = list(inputs.get("character") or [])
    for wired in wires:
        if wired is None:
            continue
        name = str(getattr(wired, "file", "") or "")
        if not name:
            raise ValueError(
                "That character has not been saved yet. Wire it through Write .char first."
            )
        names.append(name)
    return names
