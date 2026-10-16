"""A cast composited into one reference image, for when the slots cannot carry each character."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import run_data_dir

#: Part of every cache key, so a layout change never serves a sheet drawn the old way.
SHEET_LAYOUT_VERSION = "1"

PAD = 16
CAPTION = 48
#: Mid-grey: a dark or white ground would read as part of the scene to the model.
BACKGROUND = (128, 128, 128)
FOREGROUND = (16, 16, 16)


@dataclass(frozen=True)
class SheetOptions:
    """What the render gate varies; the defaults are the untested starting point."""

    #: Off binds names by position only, for a model that copies the caption text into the image.
    captions: bool = True
    cell: int = 512
    #: Above three characters, two half-sheets in two slots, so each face keeps more pixels.
    split: bool = False
    #: Face cells only: half the height, so each face survives a 1 MP resize at a larger size.
    faces_only: bool = False


@dataclass(frozen=True)
class SheetMember:
    name: str
    face: Path
    body: Path | None = None


def rows_for(count: int) -> list[int]:
    """Characters per row: one row up to three, then as square as the count allows."""
    if count <= 3:
        return [count]
    if count == 4:
        return [2, 2]
    return [3, count - 3]


def sheet_size(count: int, options: SheetOptions) -> tuple[int, int]:
    rows = rows_for(count)
    cell_height = options.cell * (1 if options.faces_only else 2) + (
        CAPTION if options.captions else 0
    )
    return PAD + max(rows) * (options.cell + PAD), PAD + len(rows) * (cell_height + PAD)


def groups(count: int, options: SheetOptions) -> list[range]:
    """Which cast positions each sheet holds, in order."""
    if options.split and count > 3:
        half = math.ceil(count / 2)
        return [range(0, half), range(half, count)]
    return [range(0, count)]


def cast_sheets(members: Sequence[SheetMember], options: SheetOptions) -> list[Path]:
    """One sheet per group, cached by content so a repeated cast draws nothing."""
    return [
        _cached(members[group.start : group.stop], options)
        for group in groups(len(members), options)
    ]


def _cached(members: Sequence[SheetMember], options: SheetOptions) -> Path:
    key = _key(members, options)
    target = run_data_dir() / "characters" / "sheets" / f"{key}.png"
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(".part")
    draw_sheet(members, options).save(staging, format="PNG")
    staging.replace(target)
    return target


def _key(members: Sequence[SheetMember], options: SheetOptions) -> str:
    digest = hashlib.sha256()
    digest.update(SHEET_LAYOUT_VERSION.encode())
    digest.update(json.dumps(asdict(options), sort_keys=True).encode())
    for member in members:
        digest.update(member.name.encode())
        for path in (member.face, member.body):
            digest.update(hashlib.sha256(path.read_bytes()).digest() if path else b"-")
    return digest.hexdigest()


def draw_sheet(members: Sequence[SheetMember], options: SheetOptions) -> Any:
    from PIL import Image, ImageDraw

    width, height = sheet_size(len(members), options)
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    cell = options.cell
    cell_height = cell * (1 if options.faces_only else 2) + (CAPTION if options.captions else 0)
    font = _font(28)
    index = 0
    for row, count in enumerate(rows_for(len(members))):
        # A short last row is centred, so reading order stays left to right, top to bottom.
        offset = (width - (PAD + count * (cell + PAD))) // 2
        for column in range(count):
            member = members[index]
            index += 1
            x = offset + PAD + column * (cell + PAD)
            y = PAD + row * (cell_height + PAD)
            with Image.open(member.face) as face:
                sheet.paste(_cover(face.convert("RGB"), cell), (x, y))
            body_bottom = y + cell
            if not options.faces_only:
                if member.body is not None:
                    with Image.open(member.body) as body:
                        sheet.paste(_contain(body.convert("RGB"), cell), (x, body_bottom))
                body_bottom += cell
            if options.captions:
                text_width = draw.textlength(member.name, font=font)
                draw.text(
                    (x + (cell - text_width) / 2, body_bottom + 8),
                    member.name,
                    fill=FOREGROUND,
                    font=font,
                )
    return sheet


def _cover(image: Any, size: int) -> Any:
    """Centre-cropped to a square, so a face fills its cell."""
    from PIL import Image

    side = min(image.width, image.height)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    square = image.crop((left, top, left + side, top + side))
    return square.resize((size, size), Image.Resampling.LANCZOS)


def _contain(image: Any, size: int) -> Any:
    """Fitted whole on the sheet's ground, because cropping a body shot loses the build."""
    from PIL import Image

    scale = size / max(image.width, image.height)
    fitted = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (size, size), BACKGROUND)
    canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return canvas


def _font(size: int) -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Pillow below 10.1 takes no size; the bitmap default is small but always present.
        return ImageFont.load_default()
