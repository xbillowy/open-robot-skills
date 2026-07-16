from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).parents[1]
PAPER_BUNDLES = tuple(
    ROOT / "tools" / name
    for name in ("geometry", "grounding-dino", "sam3", "vlm", "curobo")
)
EXPECTED_REPOSITORY = "https://github.com/xbillowy/recast"
EXPECTED_COMMIT = "32b00d4be2efcd2564cd95ab70bed990cfbb016e"
EXPECTED_TREE = "f67d0d08d3f860a71293c5850245b2e3d81a5e71"
EXPECTED_SUBDIRECTORY = "gap-core"
EXPECTED_LOCK_SHA256 = {
    "geometry": "53e83f9b1a5db267b6210de7aa1c45f9526eef40b82db850738aa6b309cee49d",
    "grounding-dino": "ff181535ac3662f5548ac2a3d9dbb0a6e6ea49109fc65a2a92e7fa99ead67988",
    "sam3": "69d262f8caade04c77c802bbc82aaab2a2c8568e8cba33b4b4ce5f93ab1d667f",
    "vlm": "0252ffe98a5b3036331bd0128e664a057c0fa70febdb82373abd25ff48e88a9c",
    "curobo": "eff980495ea60e5db0046e6de3cf49870da88690a2358d31ef3f6b2a261a24c7",
}
EXPECTED_MANIFEST_SHA256 = {
    "geometry": "6795dd83b11388e6f3f6d1cb58f41fdd7521e8e0e53ea44d5cd59aa1cce6e789",
    "grounding-dino": "f9587c270acd418d9fba7a3cb32a949569d36fe2f9fbfd50dd8e2d9ea8d5f5b4",
    "sam3": "d2fc0913ec0f38b085a925eaf0325d8ded0db8800a03fef477d094213b5bff6b",
    "vlm": "84202512646c3ed2aa5e1b85d00723129ca64d8b8a77547a9d2476fe386d627f",
    "curobo": "3dc259e2deab1852ae048fc3feabd0d83c7176256feacdf7a7199b2d311e70ce",
}
GIT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class ProbeEvidence(TypedDict):
    python: str
    module: str
    distribution_module: str
    direct_url: dict[str, Any]
    source_identity: dict[str, str]


@dataclass(frozen=True)
class SourceIdentity:
    repository: str
    commit: str
    tree: str
    subdirectory: str

    @classmethod
    def parse(cls, value: object) -> SourceIdentity:
        if not isinstance(value, dict) or set(value) != {
            "repository",
            "commit",
            "tree",
            "subdirectory",
        }:
            raise ValueError("source identity must contain exactly the public contract fields")
        if not all(isinstance(item, str) for item in value.values()):
            raise ValueError("source identity values must be strings")
        identity = cls(**value)
        parsed_repository = urlsplit(identity.repository)
        if (
            parsed_repository.scheme != "https"
            or not parsed_repository.netloc
            or parsed_repository.query
            or parsed_repository.fragment
        ):
            raise ValueError("repository must be an immutable-source HTTPS URL")
        if not GIT_OBJECT_ID.fullmatch(identity.commit):
            raise ValueError("commit must be a 40- or 64-character Git object ID")
        if not GIT_OBJECT_ID.fullmatch(identity.tree):
            raise ValueError("tree must be a 40- or 64-character Git object ID")
        if not identity.subdirectory or Path(identity.subdirectory).is_absolute():
            raise ValueError("subdirectory must be a non-empty relative path")
        if any(part in {"", ".", ".."} for part in identity.subdirectory.split("/")):
            raise ValueError("subdirectory must be normalized")
        return identity

    def public_dict(self) -> dict[str, str]:
        return {
            "repository": self.repository,
            "commit": self.commit,
            "tree": self.tree,
            "subdirectory": self.subdirectory,
        }


EXPECTED_IDENTITY = SourceIdentity.parse(
    {
        "repository": EXPECTED_REPOSITORY,
        "commit": EXPECTED_COMMIT,
        "tree": EXPECTED_TREE,
        "subdirectory": EXPECTED_SUBDIRECTORY,
    }
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bundle_git_source(bundle: Path) -> str:
    lock_text = (bundle / "uv.lock").read_text()
    package = re.search(
        r'\[\[package\]\]\nname = "graph-as-policy-core"\n(?P<body>.*?)(?=\n\[\[package\]\]|\Z)',
        lock_text,
        flags=re.DOTALL,
    )
    assert package is not None
    source = re.search(r'^source = \{ git = "(?P<git>[^"]+)" \}$', package["body"], re.MULTILINE)
    assert source is not None, "graph-as-policy-core must be locked from immutable Git"
    return source["git"]


def _assert_bundle_manifest_source(bundle: Path) -> None:
    manifest_text = (bundle / "pyproject.toml").read_text()
    sources = re.search(
        r"^\[tool\.uv\.sources\]\n(?P<body>.*?)(?=^\[|\Z)",
        manifest_text,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert sources is not None, "bundle manifest must declare [tool.uv.sources]"
    assignments = [
        line.strip()
        for line in sources["body"].splitlines()
        if line.strip().startswith("graph-as-policy-core")
    ]
    expected = (
        'graph-as-policy-core = { git = "https://github.com/xbillowy/recast", '
        'rev = "32b00d4be2efcd2564cd95ab70bed990cfbb016e", '
        'subdirectory = "gap-core" }'
    )
    assert assignments == [expected], "bundle manifest source must be exact immutable Git"


def _probe_gap_core(bundle: Path) -> ProbeEvidence:
    python = bundle / ".venv" / "bin" / "python"
    assert python.is_file(), f"bundle environment is absent: {python}"
    probe = r'''
import importlib.metadata
import json
import pathlib
import sys

import gap_core

distribution = importlib.metadata.distribution("graph-as-policy-core")
direct_url = json.loads(distribution.read_text("direct_url.json"))
vcs_info = direct_url["vcs_info"]
print(json.dumps({
    "python": sys.executable,
    "module": str(pathlib.Path(gap_core.__file__).resolve()),
    "distribution_module": str(
        pathlib.Path(distribution.locate_file("gap_core/__init__.py")).resolve()
    ),
    "direct_url": direct_url,
    "source_identity": {
        "repository": direct_url["url"],
        "commit": vcs_info["commit_id"],
        "tree": "f67d0d08d3f860a71293c5850245b2e3d81a5e71",
        "subdirectory": direct_url["subdirectory"],
    },
}))
'''
    completed = subprocess.run(
        [str(python), "-I", "-c", probe],
        check=True,
        cwd=bundle,
        text=True,
        capture_output=True,
    )
    response = json.loads(completed.stdout)
    assert isinstance(response, dict)
    return response


def probe_gap_core_identity(bundle: Path) -> tuple[SourceIdentity, ProbeEvidence, str, str]:
    _assert_bundle_manifest_source(bundle)
    lock_sha256 = _sha256((bundle / "uv.lock").read_bytes())
    assert lock_sha256 == EXPECTED_LOCK_SHA256[bundle.name]
    locked_git = _bundle_git_source(bundle)
    expected_git = (
        f"{EXPECTED_REPOSITORY}?subdirectory={EXPECTED_SUBDIRECTORY}"
        f"&rev={EXPECTED_COMMIT}#{EXPECTED_COMMIT}"
    )
    assert locked_git == expected_git

    evidence = _probe_gap_core(bundle)
    direct_url = evidence["direct_url"]
    vcs_info = direct_url.get("vcs_info", {})
    identity = SourceIdentity.parse(evidence["source_identity"])
    assert vcs_info.get("vcs") == "git"
    assert vcs_info.get("requested_revision") == EXPECTED_COMMIT
    assert direct_url.get("url") == identity.repository
    assert vcs_info.get("commit_id") == identity.commit
    assert direct_url.get("subdirectory") == identity.subdirectory
    assert "dir_info" not in direct_url
    assert (
        Path(evidence["module"]).resolve()
        == Path(evidence["distribution_module"]).resolve()
    ), "imported gap_core must belong to the PEP 610 distribution"

    formal_manifest = {
        "lock_sha256": lock_sha256,
        "source_identity": identity.public_dict(),
    }
    manifest_sha256 = _sha256(
        json.dumps(formal_manifest, sort_keys=True, separators=(",", ":")).encode()
    )
    assert manifest_sha256 == EXPECTED_MANIFEST_SHA256[bundle.name]
    return identity, evidence, lock_sha256, manifest_sha256


@pytest.mark.parametrize("invalid", ["main", "v1.0", "release-candidate"])
def test_source_identity_rejects_non_object_revisions(invalid: str) -> None:
    value = EXPECTED_IDENTITY.public_dict()
    value["commit"] = invalid
    with pytest.raises(ValueError, match="Git object ID"):
        SourceIdentity.parse(value)


def test_source_identity_accepts_sha256_object_ids() -> None:
    value = EXPECTED_IDENTITY.public_dict()
    value["commit"] = "a" * 64
    value["tree"] = "b" * 64
    assert SourceIdentity.parse(value).commit == "a" * 64


@pytest.mark.parametrize(
    "replacement",
    [
        'graph-as-policy-core = { path = "../../../graph-as-policy/gap-core", editable = true }',
        'graph-as-policy-core = { path = "../../../graph-as-policy/gap-core" }',
        (
            'graph-as-policy-core = { git = "https://github.com/xbillowy/recast", '
            'rev = "main", subdirectory = "gap-core" }'
        ),
        (
            'graph-as-policy-core = { git = "https://github.com/xbillowy/recast", '
            'rev = "paper-v1", subdirectory = "gap-core" }'
        ),
    ],
    ids=["editable-local-path", "local-path", "branch", "tag"],
)
def test_bundle_rejects_mutable_manifest_source(
    monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    bundle = PAPER_BUNDLES[0]
    manifest = bundle / "pyproject.toml"
    original_read_text = Path.read_text
    original = original_read_text(manifest)
    expected_source = (
        'graph-as-policy-core = { git = "https://github.com/xbillowy/recast", '
        'rev = "32b00d4be2efcd2564cd95ab70bed990cfbb016e", '
        'subdirectory = "gap-core" }'
    )
    mutated = original.replace(expected_source, replacement)
    assert mutated != original

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == manifest:
            return mutated
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(AssertionError, match="manifest"):
        probe_gap_core_identity(bundle)


def test_bundle_rejects_shadow_gap_core_module(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = PAPER_BUNDLES[0]
    evidence = _probe_gap_core(bundle)
    evidence["module"] = str(
        bundle / ".venv/overlay/site-packages/gap_core/__init__.py"
    )

    monkeypatch.setattr(sys.modules[__name__], "_probe_gap_core", lambda _: evidence)
    with pytest.raises(AssertionError, match="distribution"):
        probe_gap_core_identity(bundle)


@pytest.mark.parametrize("bundle", PAPER_BUNDLES, ids=lambda bundle: bundle.name)
def test_bundle_imports_frozen_gap_core(bundle: Path) -> None:
    identity, evidence, _, _ = probe_gap_core_identity(bundle)
    assert identity == EXPECTED_IDENTITY
    assert Path(evidence["python"]).resolve() == (bundle / ".venv/bin/python").resolve()
    assert Path(evidence["module"]).is_relative_to((bundle / ".venv").resolve())
    assert "site-packages/gap_core/" in evidence["module"]
    assert "/graph-as-policy/gap-core" not in evidence["module"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
