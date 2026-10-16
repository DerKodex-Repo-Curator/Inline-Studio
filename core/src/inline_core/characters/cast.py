"""Several characters in one render: who gets which reference slots, and the prompt naming them."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..takes import AssetRef
from . import charfile as cf
from .apply import AppliedCharacter, char_apply
from .sheet import SheetMember, SheetOptions, cast_sheets, groups

MAX_CAST = 5
#: Below two references a character is one view of a face, which the gate has not shown holds.
MIN_REFS_PER_CHARACTER = 2
MODES = ("auto", "refs", "face-each", "sheet")

#: What the joined descriptions may take of a 512-token encoder, leaving the rest to the prompt.
DESCRIPTION_TOKENS = 240
#: A rough English average; no tokenizer is loaded here, and the cap only has to be in range.
CHARS_PER_TOKEN = 4
TEXT_ENCODER_TOKENS = 512


class CastError(ValueError):
    """A cast that cannot be applied as asked; the message says what to change."""


@dataclass(frozen=True)
class CastPlan:
    mode: str
    #: References per character, in cast order. Empty for a sheet.
    shares: tuple[int, ...]
    slots: int


def plan_cast(
    ref_counts: Sequence[int],
    budget: int,
    mode: str = "auto",
    sheet: SheetOptions | None = None,
) -> CastPlan:
    """How a cast fits ``budget`` reference slots. Mirrored in ``src/shared/castPlan.ts``."""
    count = len(ref_counts)
    if count == 0:
        raise CastError("No characters are wired.")
    _check_size(count)
    if mode not in MODES:
        raise CastError(f"Unknown cast mode {mode!r}; use one of {', '.join(MODES)}.")
    if budget <= 0:
        raise CastError(
            "The wired images already fill this model's reference slots, so the characters have "
            "none left. Unwire one to make room."
        )
    if mode == "auto":
        # One person never goes on a sheet: it would spend a slot to show what one reference shows.
        enough = count == 1 or count * MIN_REFS_PER_CHARACTER <= budget
        mode = "refs" if enough else "sheet"
    if mode == "sheet":
        slots = len(groups(count, sheet or SheetOptions()))
        if slots > budget:
            raise CastError(
                f"The character sheets need {slots} reference slots and {budget} are free. "
                "Unwire an image, or turn off split sheets."
            )
        return CastPlan("sheet", (), slots)
    if count > budget:
        raise CastError(
            f"{count} characters need at least {count} reference slots and {budget} are free. "
            "Use a character sheet, or unwire an image."
        )
    shares = (
        tuple(min(1, n) for n in ref_counts) if mode == "face-each" else _divide(ref_counts, budget)
    )
    return CastPlan(mode, shares, sum(shares))


def _divide(counts: Sequence[int], budget: int) -> tuple[int, ...]:
    """Round-robin, so the split is even and any remainder goes to the earliest characters."""
    shares = [0] * len(counts)
    spare = budget
    while spare > 0:
        moved = False
        for index, available in enumerate(counts):
            if spare and shares[index] < available:
                shares[index] += 1
                spare -= 1
                moved = True
        if not moved:
            break
    return tuple(shares)


@dataclass
class AppliedCast:
    mode: str
    members: list[AppliedCharacter]
    refs: list[AssetRef]
    #: Which cast positions each sheet holds; empty unless the mode is a sheet.
    sheets: list[range] = field(default_factory=lambda: [])

    def prompt_prefix(
        self, first_position: int, style: str = "ordinal", role_lines: bool = False
    ) -> str:
        """Every character named against the positions its references landed on, in cast order."""
        if self.mode != "sheet":
            out = ""
            position = first_position
            for member in self.members:
                out += member.prompt_prefix(position, style, role_lines=role_lines)
                position += len(member.refs)
            return out
        out = ""
        for offset, group in enumerate(self.sheets):
            names = [self.members[i].name for i in group]
            out += (
                f"{_label(style, first_position + offset)} is a character sheet. Left to right, "
                f"top to bottom, it shows {_join(names)}, {len(names)} different people. "
            )
        for member in self.members:
            detail = " ".join(member.description.split())
            if detail:
                out += f"{member.name}: {_sentence(detail)} "
        return out


def apply_cast(
    names: Sequence[str],
    arch: str,
    *,
    budget: int,
    mode: str = "auto",
    keep_roles: tuple[str, ...] | None = None,
    sheet: SheetOptions | None = None,
) -> AppliedCast:
    """A cast of two or more as references; one character keeps going through ``char_apply``."""
    chosen = [name for name in names if name]
    _check_size(len(chosen))
    options = sheet or SheetOptions()
    full: list[AppliedCharacter] = []
    for name in chosen:
        # Reference, never adapter: two identity adapters stacked on one model blend the faces.
        applied = char_apply(name, arch, prefer="reference", keep_roles=keep_roles)
        if applied is None or not applied.refs:
            raise CastError(
                f"{name} has no {arch} references to send, so it cannot share a render. Compile "
                "its references for this model and write it again."
            )
        full.append(applied)
    plan = plan_cast([len(a.refs) for a in full], budget, mode, options)
    share = _description_share(len(full))
    for applied in full:
        applied.description = cap_description(applied.description, share)

    if plan.mode == "sheet":
        sheets = groups(len(full), options)
        paths = cast_sheets([_sheet_member(a) for a in full], options)
        return AppliedCast(
            "sheet", full, [AssetRef(ref="path", path=str(p)) for p in paths], sheets
        )

    members: list[AppliedCharacter] = []
    for name, applied, count in zip(chosen, full, plan.shares, strict=True):
        roles = keep_roles
        if plan.mode == "face-each" and cf.ROLE_FACE in applied.roles:
            roles = (cf.ROLE_FACE,)
        fitted = char_apply(name, arch, prefer="reference", limit=count, keep_roles=roles)
        assert fitted is not None
        fitted.description = applied.description
        members.append(fitted)
    return AppliedCast(plan.mode, members, [ref for m in members for ref in m.refs])


@dataclass(frozen=True)
class AppliedAdapters:
    """A cast on a model with no reference channel: one adapter per character."""

    members: list[AppliedCharacter]

    def prompt_prefix(self) -> str:
        out = ""
        for member in self.members:
            detail = " ".join(member.description.split())
            out += f"{member.name}: {_sentence(detail)} " if detail else f"{member.name}. "
        return out


def apply_adapters(names: Sequence[str], arch: str) -> AppliedAdapters:
    """Every character's trained adapter, or an error naming whoever has none."""
    chosen = [name for name in names if name]
    _check_size(len(chosen))
    applied = [char_apply(name, arch, prefer="lora") for name in chosen]
    missing = [name for name, a in zip(chosen, applied, strict=True) if a is None or a.lora is None]
    if missing:
        raise CastError(
            f"{_join(missing)} {'has' if len(missing) == 1 else 'have'} no {arch} adapter, and "
            "this model takes a character only by adapter. Train one, or unwire them."
        )
    share = _description_share(len(chosen))
    members = [a for a in applied if a is not None]
    for member in members:
        member.description = cap_description(member.description, share)
    return AppliedAdapters(members)


def cap_description(text: str, tokens: int) -> str:
    """Cut at the last sentence that fits, so a long description cannot crowd out the prompt."""
    detail = " ".join(text.split())
    limit = tokens * CHARS_PER_TOKEN
    if len(detail) <= limit:
        return detail
    head = detail[:limit]
    ends = [m.end() for m in re.finditer(r"[.!?](\s|$)", head)]
    if ends:
        return head[: ends[-1]].strip()
    return head.rsplit(" ", 1)[0].rstrip(",;:") + "."


def overflow_notice(prefix: str, prompt: str, tokens: int = TEXT_ENCODER_TOKENS) -> str:
    """A warning when the cast text plus the prompt likely runs past what the encoder reads."""
    estimate = (len(prefix) + len(prompt)) // CHARS_PER_TOKEN
    if estimate <= tokens:
        return ""
    return (
        "The cast's descriptions and your prompt together are longer than this model reads, so "
        "the end of your prompt may be cut. Shorten the prompt, or a character's description."
    )


def _check_size(count: int) -> None:
    if count > MAX_CAST:
        raise CastError(
            f"At most {MAX_CAST} characters per generation, and {count} are wired. Unwire one."
        )


def _description_share(count: int) -> int:
    return DESCRIPTION_TOKENS // max(1, count)


def _sheet_member(applied: AppliedCharacter) -> SheetMember:
    faces = [
        Path(r.path or "")
        for r, role in zip(applied.refs, applied.roles, strict=False)
        if role == cf.ROLE_FACE
    ]
    bodies = [
        Path(r.path or "")
        for r, role in zip(applied.refs, applied.roles, strict=False)
        if role == cf.ROLE_BODY
    ]
    face = faces[0] if faces else Path(applied.refs[0].path or "")
    body = bodies[0] if bodies else (faces[1] if len(faces) > 1 else None)
    return SheetMember(applied.name, face, body)


def _label(style: str, position: int) -> str:
    if style == "token":
        return f"<Picture {position}>"
    if style == "at-image":
        return f"@Image{position}"
    return f"Image {position}"


def _join(names: Sequence[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _sentence(text: str) -> str:
    return text if text[-1] in ".!?" else f"{text}."
