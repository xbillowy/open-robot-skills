"""Strict VLM letter parser shared (mirrored) by every letter-choosing skill.

The parser block is copied verbatim into each skill script because skill
scripts are copied into graphs as standalone files.  These tests check that
the copies are identical and exercise the block on real recorded replies.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MIRRORS = [
    "skills/perceiving-objects/scripts/perceive_dino_vlm.py",
    "skills/perceiving-next-item/scripts/perceive_dino_vlm.py",
    "skills/perceiving-object-parts/scripts/perceive_subpart.py",
    "skills/perceiving-objects-oneshot/scripts/perceive_simple.py",
    "skills/transporting-objects/scripts/perceive_placement_zone.py",
]
BEGIN = "# -- BEGIN strict letter parser"
END = "# -- END strict letter parser"
FIXTURE = ROOT / "tests/fixtures/vlm_letter_replies_20261014.json"


def _block(rel: str) -> str:
    src = (ROOT / rel).read_text()
    assert src.count(BEGIN) == 1 and src.count(END) == 1, rel
    return src[src.index(BEGIN):src.index(END) + len(END)]


def _load(rel: str):
    ns: dict = {"re": re, "json": json}
    exec(compile(_block(rel), rel, "exec"), ns)  # noqa: S102 - trusted repo source
    return ns["_parse_letter"]


parse = _load(MIRRORS[0])
CASES = json.loads(FIXTURE.read_text())["cases"]


def test_mirrors_are_identical():
    canonical = _block(MIRRORS[0])
    for rel in MIRRORS[1:]:
        assert _block(rel) == canonical, f"{rel} drifted from {MIRRORS[0]}"


def test_no_legacy_letter_parsers_remain():
    for rel in MIRRORS:
        src = (ROOT / rel).read_text()
        assert src.count("def _parse_letter(") == 1, rel
        assert "def _parse_pick(" not in src, rel


@pytest.mark.parametrize("text,n,want", [
    ("A", 2, 0), ("b", 2, 1), ("**B**", 2, 1), ("B.", 2, 1), (" (A) ", 2, 0),
    ("a", 2, 0),                       # a lone lowercase letter is a reply
    ("C", 2, None),                    # out of range
    ("H", 8, 7),
    ("", 2, None), ("model", 2, None),
    ("NONE", 8, None), ("None.", 8, None), ("Neither", 2, None),
    ('{"answer": "C"}', 8, 2), ('```json\n{"box": "none"}\n```', 8, None),
    ('{"choice": null}', 8, None),
    ("Answer: B", 2, 1), ("The final answer is **A**.", 2, 0),
    ("SELECTED: A", 2, 0), ("**Answer:** B", 2, 1), ("**Answer**: A", 2, 0),
    # "answer"/"choice" followed by a parenthetical is not an answer statement
    ("**A** is the correct answer (Image B shows a carton).\n\n**A**", 2, 0),
    ("Based on the images, **B** is the correct choice (A is a bottle).", 2, 1),
    ("It is a bottle of ketchup, so B", 2, None),  # no explicit form
    ("This is a carton of milk.", 2, None),   # article only -> unparsed
    ("A is a can and B is a bottle.", 2, None),  # no stated choice
    ("**A** is a can, **B** is the bbq sauce bottle.", 2, None),
    ("Neither box is the target. **A** is a bowl.", 8, None),
    ("Neither is a clear match; however, **B** is the closer match.", 2, 1),
])
def test_synthetic_forms(text, n, want):
    assert parse(text, n) == want


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['category']}-{c['n']}")
def test_recorded_replies(case):
    assert parse(case["text"], case["n"]) == case["expected"]


def test_recorded_article_a_never_read_as_label():
    article = [c for c in CASES if c["category"] == "article_a"]
    assert sum(c["count"] for c in article) >= 139
    for c in article:
        got = parse(c["text"], c["n"])
        assert c["shipped"] == 0  # the old parser read the article "a" as A
        if got == 0:
            # only when an uppercase/bold A is explicitly stated
            assert re.search(r"\*\*A\*\*|\bA\b", c["text"]), c["text"]
        bold = set(re.findall(r"\*\*([A-H])\*\*", c["text"]))
        declines = re.search(r"\b(neither|none of)\b", c["text"], re.I)
        if len(bold) == 1 and not declines and min(bold) in "ABCDEFGH"[:c["n"]]:
            assert got == "ABCDEFGH".index(bold.pop()), c["text"]


def test_recorded_pure_declines_return_none():
    declines = [c for c in CASES
                if re.match(r"\W*neither\b", c["text"], re.I)
                and not re.search(r"\b(closer|closest|designated|selected|answer|choose|choice)\b",
                                  c["text"], re.I)]
    assert declines
    for c in declines:
        assert parse(c["text"], c["n"]) is None, c["text"]
