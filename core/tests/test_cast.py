"""Several characters in one render: slot planning, the naming prompt, sheets and scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

_FIXTURE = Path(__file__).parent / "fixtures" / "cast_plan.json"


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("INLINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("INLINE_EXTRA_MODELS_DIRS", raising=False)
    # models_dirs() always appends the relative ./models, so the checkout's real one leaks in.
    monkeypatch.chdir(tmp_path)


def _image(path: Path, colour: tuple[int, int, int] = (180, 150, 140)) -> Path:
    Image.new("RGB", (512, 512), colour).save(path)
    return path


def _character(tmp_path: Path, name: str, refs: int = 3, description: str = "") -> str:
    from inline_core.characters import encode, library

    images = [_image(tmp_path / f"{name}-{i}.png", (40 * i, 90, 150)) for i in range(refs)]
    library.save(encode.char_encode(images, name=name, description=description))
    return f"{name}.char"


# --- plan ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", json.loads(_FIXTURE.read_text())["cases"])
def test_plan_matches_the_shared_fixture(case: dict[str, Any]) -> None:
    """The browser mirrors this rule to show what a run will send, so both read one fixture."""
    from inline_core.characters.cast import CastError, plan_cast
    from inline_core.characters.sheet import SheetOptions

    options = SheetOptions(split=bool(case.get("split", False)))
    if "error" in case:
        with pytest.raises(CastError, match=f"(?i){case['error']}"):
            plan_cast(case["counts"], case["budget"], case["mode"], options)
        return
    plan = plan_cast(case["counts"], case["budget"], case["mode"], options)
    assert plan.mode == case["expect"]["mode"]
    assert list(plan.shares) == case["expect"]["shares"]
    assert plan.slots == case["expect"]["slots"]


# --- applying ------------------------------------------------------------------------------------


def test_a_cast_numbers_each_character_after_the_one_before(tmp_path: Path) -> None:
    """Positions advance by what each character actually sent, so no two names claim one image."""
    from inline_core.characters import cast, encode

    names = [_character(tmp_path, "Aron"), _character(tmp_path, "Sia")]
    applied = cast.apply_cast(names, encode.FLUX2_KLEIN_ARCH, budget=4, mode="refs")

    assert applied.mode == "refs"
    assert [len(m.refs) for m in applied.members] == [2, 2]
    prefix = applied.prompt_prefix(2)
    assert "Images 2 and 3 show Aron" in prefix
    assert "Images 4 and 5 show Sia" in prefix


def test_one_face_each_sends_one_reference_per_character(tmp_path: Path) -> None:
    from inline_core.characters import cast, encode

    names = [_character(tmp_path, n) for n in ("Aron", "Sia", "Mei")]
    applied = cast.apply_cast(names, encode.FLUX2_KLEIN_ARCH, budget=4, mode="face-each")

    assert [len(m.refs) for m in applied.members] == [1, 1, 1]
    assert "Image 3 shows Mei" in applied.prompt_prefix(1)


def test_a_sheet_is_one_image_named_in_reading_order(tmp_path: Path) -> None:
    from inline_core.characters import cast, encode

    names = [_character(tmp_path, n, description=f"{n} wears red") for n in ("Aron", "Sia", "Mei")]
    applied = cast.apply_cast(names, encode.FLUX2_KLEIN_ARCH, budget=4)

    assert applied.mode == "sheet"
    assert len(applied.refs) == 1
    prefix = applied.prompt_prefix(1)
    assert prefix.startswith("Image 1 is a character sheet.")
    assert "it shows Aron, Sia and Mei, 3 different people" in prefix
    assert "Sia: Sia wears red." in prefix
    assert "<Picture 1> is a character sheet." in applied.prompt_prefix(1, style="token")


def test_a_character_without_references_cannot_join_a_cast(tmp_path: Path) -> None:
    from inline_core.characters import cast

    names = [_character(tmp_path, "Aron"), _character(tmp_path, "Sia")]
    with pytest.raises(cast.CastError, match="no krea2 references"):
        cast.apply_cast(names, "krea2", budget=4)


def test_more_than_five_is_refused_before_any_work(tmp_path: Path) -> None:
    from inline_core.characters import cast, encode

    with pytest.raises(cast.CastError, match="At most 5"):
        cast.apply_cast([f"c{i}.char" for i in range(6)], encode.FLUX2_KLEIN_ARCH, budget=9)


def test_long_descriptions_are_cut_at_a_sentence() -> None:
    from inline_core.characters.cast import CHARS_PER_TOKEN, cap_description

    text = "Short first. " + "word " * 400
    capped = cap_description(text, 10)
    assert capped == "Short first."
    assert len(cap_description("x " * 400, 10)) <= 10 * CHARS_PER_TOKEN + 1
    assert cap_description("fits.", 10) == "fits."


def test_an_overlong_prompt_is_flagged_not_cut() -> None:
    from inline_core.characters.cast import overflow_notice

    assert overflow_notice("short", "prompt") == ""
    # Says what to shorten rather than quoting an encoder's token budget at an artist.
    notice = overflow_notice("x" * 1500, "y" * 1000)
    assert "Shorten the prompt" in notice


# --- adapters (Krea 2) ---------------------------------------------------------------------------


def test_adapters_name_everyone_who_has_none(tmp_path: Path) -> None:
    """Silently skipping one would render a cast with a stranger in it."""
    from inline_core.characters import cast

    names = [_character(tmp_path, "Aron"), _character(tmp_path, "Sia")]
    with pytest.raises(cast.CastError, match="Aron.char and Sia.char have no krea2 adapter"):
        cast.apply_adapters(names, "krea2")


def test_adapters_stack_one_per_character(tmp_path: Path) -> None:
    from inline_core.characters import cast, encode, library
    from inline_core.characters import charfile as cf

    names = [_character(tmp_path, "Aron", description="tall"), _character(tmp_path, "Sia")]
    for name in names:
        path = library.resolve(name)
        assert path is not None
        doc = cf.read(path)
        encode.set_lora_payload(
            doc.manifest, doc.members, f"{name}-adapter".encode(), arch="krea2",
            base="krea2_turbo_bf16.safetensors", rank=16, steps=600, resolution=512,
        )
        cf.write(path, doc)

    stacked = cast.apply_adapters(names, "krea2")
    assert len({str(m.lora) for m in stacked.members}) == 2
    assert stacked.prompt_prefix() == "Aron: tall. Sia. "


# --- sheet ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("count", "rows"), [(2, [2]), (3, [3]), (4, [2, 2]), (5, [3, 2])])
def test_rows_stay_near_square(count: int, rows: list[int]) -> None:
    from inline_core.characters.sheet import rows_for

    assert rows_for(count) == rows


def test_sheet_size_follows_the_options() -> None:
    from inline_core.characters.sheet import CAPTION, PAD, SheetOptions, sheet_size

    full = sheet_size(5, SheetOptions())
    assert full == (PAD + 3 * (512 + PAD), PAD + 2 * (1024 + CAPTION + PAD))
    faces = sheet_size(5, SheetOptions(faces_only=True, captions=False))
    assert faces == (full[0], PAD + 2 * (512 + PAD))


def test_split_sheets_divide_the_cast_in_order() -> None:
    from inline_core.characters.sheet import SheetOptions, groups

    assert groups(5, SheetOptions(split=True)) == [range(0, 3), range(3, 5)]
    assert groups(3, SheetOptions(split=True)) == [range(0, 3)]
    assert groups(5, SheetOptions()) == [range(0, 5)]


def test_a_sheet_is_cached_and_the_layout_version_is_in_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inline_core.characters import sheet

    members = [
        sheet.SheetMember("Aron", _image(tmp_path / "a.png")),
        sheet.SheetMember("Sia", _image(tmp_path / "s.png", (20, 30, 40))),
    ]
    first = sheet.cast_sheets(members, sheet.SheetOptions())
    assert first == sheet.cast_sheets(members, sheet.SheetOptions())
    with Image.open(first[0]) as drawn:
        assert drawn.size == sheet.sheet_size(2, sheet.SheetOptions())

    monkeypatch.setattr(sheet, "SHEET_LAYOUT_VERSION", "test-next")
    assert sheet.cast_sheets(members, sheet.SheetOptions()) != first


def test_each_face_lands_in_its_own_cell(tmp_path: Path) -> None:
    from inline_core.characters import sheet

    colours = [(250, 0, 0), (0, 250, 0), (0, 0, 250)]
    members = [
        sheet.SheetMember(f"m{i}", _image(tmp_path / f"{i}.png", c)) for i, c in enumerate(colours)
    ]
    drawn = sheet.draw_sheet(members, sheet.SheetOptions(captions=False))
    centre = sheet.PAD + 256
    for index, colour in enumerate(colours):
        x = sheet.PAD + index * (512 + sheet.PAD) + 256
        assert drawn.getpixel((x, centre)) == colour


# --- scoring -------------------------------------------------------------------------------------


def test_faces_are_assigned_one_to_one() -> None:
    from inline_core.characters.scoring import assign_faces

    assert assign_faces([[90, 20], [85, 30]]) == [0, 1]
    # Greedy would give both to face 0; the best total gives each its own.
    assert assign_faces([[90, 80], [85, 10]]) == [1, 0]
    assert assign_faces([[50], [60]]) == [None, 0]


def test_a_swap_is_flagged_against_the_asked_for_order() -> None:
    """Aron asked for the left, Sia the right; the left face is Sia's, so the two are swapped."""
    from inline_core.characters.scoring import score_faces

    aron, sia = [1.0, 0.0], [0.0, 1.0]
    result = score_faces([sia, aron], [[aron], [sia]])
    assert result["faces"] == [1, 0]
    assert result["swapped"] == [0, 1]
    assert result["scores"] == [100.0, 100.0]

    in_place = score_faces([aron, sia], [[aron], [sia]])
    assert in_place["swapped"] == []


def test_a_missing_face_is_none_not_zero() -> None:
    from inline_core.characters.scoring import score_faces

    result = score_faces([[1.0, 0.0]], [[[1.0, 0.0]], [[0.0, 1.0]]])
    assert result["scores"][1] is None
    assert result["swapped"] == []


def test_cast_summary_averages_only_frames_with_a_face() -> None:
    from inline_core.studio.characters import _cast_summary  # pyright: ignore[reportPrivateUsage]

    frames = [
        {"scores": [80.0, None], "faces": [0, None], "unmatched": 0, "swapped": []},
        {"scores": [60.0, 70.0], "faces": [0, 1], "unmatched": 1, "swapped": [1]},
    ]
    out = _cast_summary(["a.char", "b.char"], ["A", "B"], frames)
    assert out is not None
    assert [p["score"] for p in out["perCharacter"]] == [70.0, 70.0]
    assert out["perCharacter"][1]["missingFrames"] == 1
    assert out["swapped"] == ["b.char"]
    assert out["unmatched"] == 1


# --- ports and runners ---------------------------------------------------------------------------


def test_a_character_feeds_a_cast_port() -> None:
    from inline_core.graph.schema import PortKind, is_list_kind, port_satisfies

    assert is_list_kind(PortKind.CHARACTER_LIST)
    assert port_satisfies(PortKind.CHARACTER, PortKind.CHARACTER_LIST)
    assert not is_list_kind(PortKind.CHARACTER)


def test_the_cast_ports_keep_every_wire_in_order() -> None:
    from inline_core.studio.graph_build import _edges_for  # pyright: ignore[reportPrivateUsage]

    connectors = [
        {"toItemId": "m", "fromItemId": f"c{i}", "data": {"targetHandle": "character"}}
        for i in range(3)
    ]
    by_id = {"m": {"id": "m", "data": {"core": {"type": "black-forest-labs/flux-2"}}}}
    edges = _edges_for("m", connectors, by_id, lambda _t, port: port == "character", {})
    assert [e["from"] for e in edges["character"]] == ["c0", "c1", "c2"]


def test_flux2_variants_each_carry_a_reference_limit() -> None:
    from inline_core.models.flux2 import variants

    assert all(v.max_references == 4 for v in variants.VARIANTS)


def test_both_reference_nodes_publish_the_cast_control() -> None:
    """The cloud reads `cast_mode` off the published descriptor to decide whether it can cast."""
    pytest.importorskip("torch")
    from inline_core.models.flux2.runner import FLUX2
    from inline_core.models.minimaxh3.runner import DESCRIPTORS

    for descriptor in (FLUX2, DESCRIPTORS["minimax/h3-reference-to-video"]):
        keys = {p.key for p in descriptor.params}
        assert "cast_mode" in keys, f"{descriptor.type} stopped publishing cast_mode"
