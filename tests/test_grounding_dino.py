"""grounding-dino bundle: sealed local weights, schemas, and GPU smoke."""

from __future__ import annotations

import hashlib
import importlib.util
import multiprocessing
import os
import shutil
import socket
import stat
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture(scope="module")
def dino_module(skills_registry):
    """Import the RPC bundle implementation without importing transformers."""
    module_name = "ors_grounding_dino_tools_under_test"
    module_path = skills_registry.get("grounding-dino").meta.bundle_dir / "tools.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dino_registry(dino_module):
    """Build the bundle-local registry used by its out-of-process server."""
    from gap_core.tools import ToolRegistry

    registry = ToolRegistry()
    registry.discover_pending()
    return registry


def _snapshot_fixture(
    tmp_path: Path,
    *,
    revision: str = "a" * 40,
    contents: dict[str, bytes] | None = None,
) -> tuple[Path, dict[str, tuple[int, str]], dict[str, str]]:
    contents = contents or {
        "config.json": b"sealed-config\n",
        "model.safetensors": b"sealed-weights\n",
        "tokenizer.json": b"sealed-tokenizer\n",
    }
    model_root = tmp_path / "models--IDEA-Research--grounding-dino-base"
    blobs = model_root / "blobs"
    snapshot = model_root / "snapshots" / revision
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)

    authority: dict[str, tuple[int, str]] = {}
    cached: dict[str, str] = {}
    for index, (filename, payload) in enumerate(contents.items()):
        digest = hashlib.sha256(payload).hexdigest()
        blob = blobs / f"blob-{index}-{digest}"
        blob.write_bytes(payload)
        link = snapshot / filename
        link.symlink_to(Path("../../blobs") / blob.name)
        authority[filename] = (len(payload), digest)
        cached[filename] = str(link)
    return snapshot, authority, cached


def _spawn_snapshot_validation(
    module_path: str,
    snapshot: str,
    revision: str,
    authority: dict[str, tuple[int, str]],
) -> tuple[int, str, str, int, int, bool]:
    """Spawn-safe worker used to prove validation owns no shared client/state."""
    module_name = f"ors_grounding_dino_spawn_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module._SNAPSHOT_REVISION = revision
    module._SNAPSHOT_FILES = authority
    resolved = module._validate_local_snapshot(
        Path(snapshot), revision=revision, files=authority
    )
    private_root: Path | None = None
    with module._materialized_local_snapshot(resolved) as materialized:
        private_root = materialized.root
        private_device = materialized.root_device
        private_inode = materialized.root_inode
    assert private_root is not None
    return (
        os.getpid(),
        str(resolved.root),
        str(private_root),
        private_device,
        private_inode,
        private_root.exists(),
    )


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_snapshot: Path,
) -> tuple[list[tuple[str, str, dict[str, Any]]], object, object]:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    processor = object()

    class FakeModel:
        def __init__(self) -> None:
            self.devices: list[str] = []
            self.eval_called = False

        def to(self, device: str):
            self.devices.append(device)
            return self

        def eval(self) -> None:
            self.eval_called = True

    model = FakeModel()

    class FakeProcessorFactory:
        @staticmethod
        def from_pretrained(source: str, **kwargs: Any):
            calls.append(("processor", source, kwargs))
            if source != str(expected_snapshot) or kwargs != {"local_files_only": True}:
                # Faithful signature from the four Task21 Wave2 incidents: the
                # online HF retry reuses the client that the first ConnectError closed.
                raise RuntimeError("Cannot send a request, as the client has been closed.")
            return processor

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(source: str, **kwargs: Any):
            calls.append(("model", source, kwargs))
            assert source == str(expected_snapshot)
            assert kwargs == {"local_files_only": True, "use_safetensors": True}
            return model

    fake_transformers = types.SimpleNamespace(
        AutoProcessor=FakeProcessorFactory,
        AutoModelForZeroShotObjectDetection=FakeModelFactory,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return calls, processor, model


def test_detect_registered(dino_registry):
    assert "grounding-dino.detect" in dino_registry
    desc = dino_registry.get("grounding-dino.detect")
    assert desc.tags == ("perception",)


def test_detect_schema(dino_registry):
    schema = dino_registry.get("grounding-dino.detect").schema
    assert set(schema.inputs) == {"image", "query", "box_threshold", "text_threshold"}
    assert schema.inputs["image"].required
    assert schema.inputs["query"].required
    assert schema.inputs["box_threshold"].default == pytest.approx(0.20)
    assert schema.inputs["text_threshold"].default == pytest.approx(0.20)
    assert set(schema.outputs) == {"detections"}


def test_runtime_load_uses_one_sealed_local_snapshot_and_never_online_identifier(
    dino_module, monkeypatch, tmp_path,
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    resolver_calls: list[None] = []

    def _resolve():
        resolver_calls.append(None)
        return types.SimpleNamespace(root=snapshot)

    monkeypatch.setattr(dino_module, "_resolve_local_snapshot", _resolve, raising=False)
    monkeypatch.setattr(
        dino_module,
        "_assert_snapshot_authority_unchanged",
        lambda authority: None,
        raising=False,
    )

    @contextmanager
    def _materialized(authority):
        yield types.SimpleNamespace(loader_root=snapshot)

    monkeypatch.setattr(
        dino_module, "_materialized_local_snapshot", _materialized
    )
    monkeypatch.setattr(
        dino_module,
        "_checked_materialized_authority",
        lambda authority: None,
    )
    calls, processor, model = _install_fake_transformers(
        monkeypatch, expected_snapshot=snapshot
    )
    dino_module._model = None
    dino_module._processor = None

    actual_model, actual_processor = dino_module._get_model()

    assert resolver_calls == [None]
    assert calls == [
        ("processor", str(snapshot), {"local_files_only": True}),
        (
            "model",
            str(snapshot),
            {"local_files_only": True, "use_safetensors": True},
        ),
    ]
    assert actual_processor is processor
    assert actual_model is model
    assert model.devices == [dino_module._DEVICE]
    assert model.eval_called is True


def test_resolver_accepts_exact_cached_snapshot_without_network(
    dino_module, monkeypatch, tmp_path,
):
    snapshot, authority, cached = _snapshot_fixture(tmp_path)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    monkeypatch.setattr(
        dino_module,
        "_try_to_load_from_cache",
        lambda model_name, filename, revision: cached[filename],
        raising=False,
    )

    def _network_forbidden(*args, **kwargs):
        raise AssertionError("snapshot resolution attempted network access")

    monkeypatch.setattr(socket, "create_connection", _network_forbidden)
    authority_object = dino_module._resolve_local_snapshot()
    assert authority_object.root == snapshot
    dino_module._assert_snapshot_authority_unchanged(authority_object)


@pytest.mark.parametrize("defect", ["missing", "ambiguous", "snapshot_symlink", "tampered"])
def test_resolver_fails_closed_with_one_sanitized_error(
    dino_module, monkeypatch, tmp_path, defect,
):
    snapshot, authority, cached = _snapshot_fixture(tmp_path / "primary")
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)

    if defect == "missing":
        cached["model.safetensors"] = None
    elif defect == "ambiguous":
        other_snapshot, _, other_cached = _snapshot_fixture(
            tmp_path / "other", revision=snapshot.name
        )
        assert other_snapshot != snapshot
        cached["tokenizer.json"] = other_cached["tokenizer.json"]
    elif defect == "snapshot_symlink":
        real_snapshot = snapshot.with_name("real-snapshot")
        snapshot.rename(real_snapshot)
        snapshot.symlink_to(real_snapshot, target_is_directory=True)
    elif defect == "tampered":
        Path(cached["config.json"]).resolve().write_bytes(b"tampered\n")

    monkeypatch.setattr(
        dino_module,
        "_try_to_load_from_cache",
        lambda model_name, filename, revision: cached[filename],
        raising=False,
    )

    with pytest.raises(RuntimeError) as exc_info:
        dino_module._resolve_local_snapshot()
    assert str(exc_info.value) == dino_module._LOCAL_SNAPSHOT_ERROR
    assert str(tmp_path) not in str(exc_info.value)
    assert dino_module._MODEL_NAME not in str(exc_info.value)


def test_resolver_rejects_file_symlink_outside_model_blob_store(
    dino_module, monkeypatch, tmp_path,
):
    snapshot, authority, cached = _snapshot_fixture(tmp_path)
    external = tmp_path / "external"
    external.write_bytes(b"sealed-config\n")
    config_link = Path(cached["config.json"])
    config_link.unlink()
    config_link.symlink_to(external)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    monkeypatch.setattr(
        dino_module,
        "_try_to_load_from_cache",
        lambda model_name, filename, revision: cached[filename],
        raising=False,
    )

    with pytest.raises(RuntimeError, match=f"^{dino_module._LOCAL_SNAPSHOT_ERROR}$"):
        dino_module._resolve_local_snapshot()


def test_unsealed_model_override_fails_before_cache_lookup(dino_module, monkeypatch):
    monkeypatch.setattr(dino_module, "_MODEL_NAME", "unsealed/model")

    def _unexpected_lookup(*args, **kwargs):
        raise AssertionError("an unsealed model must not reach cache lookup")

    monkeypatch.setattr(dino_module, "_try_to_load_from_cache", _unexpected_lookup)
    with pytest.raises(RuntimeError, match=f"^{dino_module._LOCAL_SNAPSHOT_ERROR}$"):
        dino_module._resolve_local_snapshot()


def test_readiness_and_runtime_use_the_same_resolver(
    dino_module, monkeypatch, tmp_path,
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls: list[Path] = []

    def _resolve():
        calls.append(snapshot)
        return types.SimpleNamespace(root=snapshot)

    monkeypatch.setattr(dino_module, "_resolve_local_snapshot", _resolve, raising=False)
    authority_checks: list[Path] = []
    monkeypatch.setattr(
        dino_module,
        "_assert_snapshot_authority_unchanged",
        lambda authority: authority_checks.append(authority.root),
        raising=False,
    )

    @contextmanager
    def _materialized(authority):
        yield types.SimpleNamespace(loader_root=snapshot)

    monkeypatch.setattr(
        dino_module, "_materialized_local_snapshot", _materialized
    )
    monkeypatch.setattr(
        dino_module,
        "_checked_materialized_authority",
        lambda authority: None,
    )
    _install_fake_transformers(monkeypatch, expected_snapshot=snapshot)
    dino_module._model = None
    dino_module._processor = None

    assert dino_module.weights_cached() is True
    dino_module._get_model()
    assert calls == [snapshot, snapshot]
    # One final readiness check plus before/after each of the two loaders.
    assert authority_checks == [snapshot] * 5


@pytest.mark.parametrize(
    ("mutation_stage", "defect", "expected_model_calls", "expected_checks"),
    [
        ("processor", "symlink_target", 0, 2),
        ("model", "inode_replace", 1, 4),
        ("model", "content_tamper", 1, 4),
    ],
)
def test_loader_discards_partial_cache_when_snapshot_authority_drifts(
    dino_module,
    monkeypatch,
    tmp_path,
    mutation_stage,
    defect,
    expected_model_calls,
    expected_checks,
):
    snapshot, authority, cached = _snapshot_fixture(tmp_path)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    monkeypatch.setattr(
        dino_module,
        "_try_to_load_from_cache",
        lambda model_name, filename, revision: cached[filename],
    )
    frozen_authority = dino_module._resolve_local_snapshot()
    assert frozen_authority.root == snapshot
    monkeypatch.setattr(
        dino_module, "_resolve_local_snapshot", lambda: frozen_authority
    )

    config_link = Path(cached["config.json"])
    original_blob = config_link.resolve()

    def _mutate() -> None:
        if defect == "symlink_target":
            replacement = original_blob.parent / "unsealed-config"
            replacement.write_bytes(b"unsealed-config")
            config_link.unlink()
            config_link.symlink_to(Path("../../blobs") / replacement.name)
        elif defect == "inode_replace":
            payload = original_blob.read_bytes()
            original_inode = original_blob.stat().st_ino
            replacement = original_blob.with_name(original_blob.name + ".replacement")
            replacement.write_bytes(payload)
            replacement.replace(original_blob)
            assert original_blob.stat().st_ino != original_inode
        elif defect == "content_tamper":
            payload = original_blob.read_bytes()
            original_blob.write_bytes(b"x" * len(payload))

    calls = {"checks": 0, "model": 0, "processor": 0}
    real_assert = dino_module._assert_snapshot_authority_unchanged

    def _counted_assert(candidate) -> None:
        calls["checks"] += 1
        real_assert(candidate)

    monkeypatch.setattr(
        dino_module, "_assert_snapshot_authority_unchanged", _counted_assert
    )

    class FakeProcessorFactory:
        @staticmethod
        def from_pretrained(source: str, **kwargs: Any):
            calls["processor"] += 1
            if mutation_stage == "processor":
                _mutate()
            return object()

    class PartialModel:
        def to(self, device: str):
            raise AssertionError("drifted model must be discarded before device load")

        def eval(self) -> None:
            raise AssertionError("drifted model must be discarded before eval")

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(source: str, **kwargs: Any):
            calls["model"] += 1
            if mutation_stage == "model":
                _mutate()
            return PartialModel()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoProcessor=FakeProcessorFactory,
            AutoModelForZeroShotObjectDetection=FakeModelFactory,
        ),
    )
    dino_module._model = None
    dino_module._processor = None

    with pytest.raises(RuntimeError, match=f"^{dino_module._LOCAL_SNAPSHOT_ERROR}$"):
        dino_module._get_model()

    assert calls == {
        "checks": expected_checks,
        "model": expected_model_calls,
        "processor": 1,
    }
    assert dino_module._model is None
    assert dino_module._processor is None


def test_runtime_loads_from_private_snapshot_when_public_blobs_transiently_swap(
    dino_module,
    monkeypatch,
    tmp_path,
):
    """Reproduce the reviewer's check/use race against the public HF cache."""
    snapshot, authority, cached = _snapshot_fixture(tmp_path)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    monkeypatch.setattr(
        dino_module,
        "_try_to_load_from_cache",
        lambda model_name, filename, revision: cached[filename],
    )

    public_blobs = snapshot.parent.parent / "blobs"
    parked_blobs = public_blobs.with_name("blobs.sealed")
    observed: list[tuple[str, bytes]] = []
    loader_sources: list[Path] = []

    def _read_while_public_blobs_are_substituted(source: str) -> None:
        public_blobs.rename(parked_blobs)
        public_blobs.mkdir()
        try:
            for blob in parked_blobs.iterdir():
                (public_blobs / blob.name).write_bytes(b"unsealed\n")
            loader_source = Path(source)
            loader_sources.append(loader_source)
            observed.append((source, (loader_source / "config.json").read_bytes()))
        finally:
            shutil.rmtree(public_blobs)
            parked_blobs.rename(public_blobs)

    processor = object()

    class FakeProcessorFactory:
        @staticmethod
        def from_pretrained(source: str, **kwargs: Any):
            assert kwargs == {"local_files_only": True}
            _read_while_public_blobs_are_substituted(source)
            return processor

    class FakeModel:
        moved_after_cleanup = False

        def to(self, device: str):
            assert loader_sources
            assert all(not source.exists() for source in loader_sources)
            self.moved_after_cleanup = True
            return self

        def eval(self) -> None:
            return None

    model = FakeModel()

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(source: str, **kwargs: Any):
            assert kwargs == {"local_files_only": True, "use_safetensors": True}
            _read_while_public_blobs_are_substituted(source)
            return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoProcessor=FakeProcessorFactory,
            AutoModelForZeroShotObjectDetection=FakeModelFactory,
        ),
    )
    dino_module._model = None
    dino_module._processor = None

    actual_model, actual_processor = dino_module._get_model()

    assert actual_model is model
    assert actual_processor is processor
    assert [payload for _, payload in observed] == [b"sealed-config\n"] * 2
    assert len({source for source, _ in observed}) == 1
    assert loader_sources[0] != snapshot
    assert all(not source.exists() for source in loader_sources)
    assert model.moved_after_cleanup is True


def test_materialized_snapshot_is_regular_read_only_single_link_and_removed(
    dino_module,
    monkeypatch,
    tmp_path,
):
    snapshot, authority, _ = _snapshot_fixture(tmp_path)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    source_authority = dino_module._validate_local_snapshot(
        snapshot,
        revision=snapshot.name,
        files=authority,
    )

    private_root: Path | None = None
    loader_root: Path | None = None
    with dino_module._materialized_local_snapshot(source_authority) as materialized:
        private_root = materialized.root
        loader_root = materialized.loader_root
        assert private_root != snapshot
        assert stat.S_IMODE(os.fstat(materialized.root_fd).st_mode) == 0o500
        assert {item.filename for item in materialized.files} == set(authority)
        for item in materialized.files:
            info = os.stat(
                item.filename,
                dir_fd=materialized.root_fd,
                follow_symlinks=False,
            )
            assert stat.S_ISREG(info.st_mode)
            assert stat.S_IMODE(info.st_mode) == 0o400
            assert info.st_nlink == 1
            assert hashlib.sha256(
                (loader_root / item.filename).read_bytes()
            ).hexdigest() == item.sha256
        dino_module._assert_materialized_snapshot_unchanged(materialized)

    assert private_root is not None and not private_root.exists()
    assert loader_root is not None and not loader_root.exists()


def test_cleanup_removes_pinned_root_and_replacement_after_rename_within_anchor(
    dino_module,
    monkeypatch,
    tmp_path,
):
    snapshot, authority, _ = _snapshot_fixture(tmp_path / "source")
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    source_authority = dino_module._validate_local_snapshot(
        snapshot,
        revision=snapshot.name,
        files=authority,
    )
    external_sentinel = tmp_path / "external-sentinel"
    external_sentinel.write_bytes(b"must-survive\n")

    anchor: Path | None = None
    moved_root: Path | None = None
    replacement_root: Path | None = None
    with dino_module._materialized_local_snapshot(source_authority) as materialized:
        anchor = materialized.anchor
        moved_root = anchor / "snapshot.moved"
        replacement_root = materialized.root
        os.chmod(anchor, 0o700)
        replacement_root.rename(moved_root)
        replacement_root.mkdir()
        (replacement_root / "replacement-file").write_bytes(b"replacement\n")
        (replacement_root / "external-link").symlink_to(external_sentinel)
        os.chmod(anchor, 0o500)

    assert anchor is not None and not anchor.exists()
    assert moved_root is not None and not moved_root.exists()
    assert replacement_root is not None and not replacement_root.exists()
    assert external_sentinel.read_bytes() == b"must-survive\n"


def test_cleanup_locates_pinned_root_moved_to_another_parent(
    dino_module,
    monkeypatch,
    tmp_path,
):
    snapshot, authority, _ = _snapshot_fixture(tmp_path / "source")
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    source_authority = dino_module._validate_local_snapshot(
        snapshot,
        revision=snapshot.name,
        files=authority,
    )
    external_parent = tmp_path / "external-parent"
    external_parent.mkdir()
    external_sentinel = external_parent / "sibling-sentinel"
    external_sentinel.write_bytes(b"must-survive\n")

    anchor: Path | None = None
    moved_root = external_parent / "snapshot.moved"
    with dino_module._materialized_local_snapshot(source_authority) as materialized:
        anchor = materialized.anchor
        os.chmod(anchor, 0o700)
        os.chmod(materialized.root, 0o700)
        materialized.root.rename(moved_root)
        materialized.root.mkdir()
        (materialized.root / "replacement-file").write_bytes(b"replacement\n")
        os.chmod(anchor, 0o500)

    assert anchor is not None and not anchor.exists()
    assert not moved_root.exists()
    assert external_sentinel.read_bytes() == b"must-survive\n"


def test_cleanup_is_fd_relative_and_never_follows_nested_symlink(
    dino_module,
    monkeypatch,
    tmp_path,
):
    snapshot, authority, _ = _snapshot_fixture(tmp_path / "source")
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    source_authority = dino_module._validate_local_snapshot(
        snapshot,
        revision=snapshot.name,
        files=authority,
    )
    external_directory = tmp_path / "external-directory"
    external_directory.mkdir()
    external_sentinel = external_directory / "sentinel"
    external_sentinel.write_bytes(b"must-survive\n")

    anchor: Path | None = None
    with dino_module._materialized_local_snapshot(source_authority) as materialized:
        anchor = materialized.anchor
        os.chmod(materialized.root, 0o700)
        nested = materialized.root / "nested"
        nested.mkdir()
        (nested / "file").write_bytes(b"private\n")
        (nested / "external-link").symlink_to(external_directory, target_is_directory=True)
        os.chmod(materialized.root, 0o500)

    assert anchor is not None and not anchor.exists()
    assert external_sentinel.read_bytes() == b"must-survive\n"


def test_cleanup_identity_failure_is_sanitized_and_preserves_external_sentinel(
    dino_module,
    monkeypatch,
    tmp_path,
):
    snapshot, authority, _ = _snapshot_fixture(tmp_path / "source")
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    source_authority = dino_module._validate_local_snapshot(
        snapshot,
        revision=snapshot.name,
        files=authority,
    )
    external_directory = tmp_path / "external-directory"
    external_directory.mkdir()
    external_sentinel = external_directory / "sentinel"
    external_sentinel.write_bytes(b"must-survive\n")
    moved_root = tmp_path / "moved-private-root"
    anchor: Path | None = None

    with pytest.raises(dino_module._SnapshotAuthorityChanged) as exc_info:
        with dino_module._materialized_local_snapshot(source_authority) as materialized:
            anchor = materialized.anchor
            os.chmod(anchor, 0o700)
            os.chmod(materialized.root, 0o700)
            materialized.root.rename(moved_root)
            dino_module._empty_trusted_directory(
                dino_module._trusted_directory_handle(
                    materialized.root_fd,
                    device=materialized.root_device,
                    inode=materialized.root_inode,
                )
            )
            moved_root.rmdir()
            moved_root.symlink_to(external_directory, target_is_directory=True)

    assert str(exc_info.value) == "private snapshot cleanup failed"
    assert str(tmp_path) not in str(exc_info.value)
    assert anchor is not None and not anchor.exists()
    assert moved_root.is_symlink()
    assert external_sentinel.read_bytes() == b"must-survive\n"


def test_anchor_stat_open_replacement_is_never_treated_as_trusted_cleanup_root(
    dino_module,
    monkeypatch,
    tmp_path,
):
    snapshot, authority, _ = _snapshot_fixture(tmp_path / "source")
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    source_authority = dino_module._validate_local_snapshot(
        snapshot,
        revision=snapshot.name,
        files=authority,
    )
    private_parent = tmp_path / "private-parent"
    private_parent.mkdir()
    real_mkdtemp = dino_module.tempfile.mkdtemp
    monkeypatch.setattr(
        dino_module.tempfile,
        "mkdtemp",
        lambda *, prefix: real_mkdtemp(prefix=prefix, dir=private_parent),
    )
    real_open = dino_module.os.open
    swapped = False
    replacement: Path | None = None
    original: Path | None = None

    def _swap_anchor_before_open(path, flags, *args, **kwargs):
        nonlocal swapped, replacement, original
        if (
            not swapped
            and kwargs.get("dir_fd") is not None
            and str(path).startswith(".gap-grounding-dino-snapshot-")
        ):
            swapped = True
            parent = Path(os.readlink(f"/proc/self/fd/{kwargs['dir_fd']}"))
            replacement = parent / str(path)
            original = replacement.with_name(replacement.name + ".original")
            replacement.rename(original)
            replacement.mkdir()
            (replacement / "external-sentinel").write_bytes(b"must-survive\n")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(dino_module.os, "open", _swap_anchor_before_open)

    with pytest.raises(dino_module._SnapshotAuthorityChanged):
        with dino_module._materialized_local_snapshot(source_authority):
            raise AssertionError("mismatched anchor must never be yielded")

    assert swapped is True
    assert original is not None and not original.exists()
    assert replacement is not None
    assert (replacement / "external-sentinel").read_bytes() == b"must-survive\n"


def test_root_stat_open_replacement_is_never_treated_as_trusted_cleanup_root(
    dino_module,
    monkeypatch,
    tmp_path,
):
    snapshot, authority, _ = _snapshot_fixture(tmp_path / "source")
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    source_authority = dino_module._validate_local_snapshot(
        snapshot,
        revision=snapshot.name,
        files=authority,
    )
    private_parent = tmp_path / "private-parent"
    private_parent.mkdir()
    real_mkdtemp = dino_module.tempfile.mkdtemp
    monkeypatch.setattr(
        dino_module.tempfile,
        "mkdtemp",
        lambda *, prefix: real_mkdtemp(prefix=prefix, dir=private_parent),
    )
    real_open = dino_module.os.open
    swapped = False
    replacement: Path | None = None
    original: Path | None = None

    def _swap_root_before_open(path, flags, *args, **kwargs):
        nonlocal swapped, replacement, original
        if not swapped and path == "snapshot" and kwargs.get("dir_fd") is not None:
            swapped = True
            anchor = Path(os.readlink(f"/proc/self/fd/{kwargs['dir_fd']}"))
            replacement = anchor / "snapshot"
            original = anchor / "snapshot.original"
            replacement.rename(original)
            replacement.mkdir()
            (replacement / "external-sentinel").write_bytes(b"must-survive\n")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(dino_module.os, "open", _swap_root_before_open)

    with pytest.raises(dino_module._SnapshotAuthorityChanged):
        with dino_module._materialized_local_snapshot(source_authority):
            raise AssertionError("mismatched root must never be yielded")

    assert swapped is True
    assert original is not None and not original.exists()
    assert replacement is not None
    assert (replacement / "external-sentinel").read_bytes() == b"must-survive\n"


def test_child_stat_open_replacement_is_closed_without_recursive_cleanup(
    dino_module,
    monkeypatch,
    tmp_path,
):
    snapshot, authority, _ = _snapshot_fixture(tmp_path / "source")
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    source_authority = dino_module._validate_local_snapshot(
        snapshot,
        revision=snapshot.name,
        files=authority,
    )
    private_parent = tmp_path / "private-parent"
    private_parent.mkdir()
    real_mkdtemp = dino_module.tempfile.mkdtemp
    monkeypatch.setattr(
        dino_module.tempfile,
        "mkdtemp",
        lambda *, prefix: real_mkdtemp(prefix=prefix, dir=private_parent),
    )
    real_open = dino_module.os.open
    swapped = False
    replacement: Path | None = None
    original: Path | None = None

    with pytest.raises(dino_module._SnapshotAuthorityChanged):
        with dino_module._materialized_local_snapshot(source_authority) as materialized:
            os.chmod(materialized.root, 0o700)
            child = materialized.root / "cleanup-child"
            child.mkdir()
            (child / "owned-file").write_bytes(b"private\n")
            os.chmod(materialized.root, 0o500)

            def _swap_child_before_open(path, flags, *args, **kwargs):
                nonlocal swapped, replacement, original
                if (
                    not swapped
                    and path == "cleanup-child"
                    and kwargs.get("dir_fd") is not None
                ):
                    swapped = True
                    root = Path(os.readlink(f"/proc/self/fd/{kwargs['dir_fd']}"))
                    replacement = root / "cleanup-child"
                    original = root / "cleanup-child.original"
                    replacement.rename(original)
                    replacement.mkdir()
                    (replacement / "external-sentinel").write_bytes(
                        b"must-survive\n"
                    )
                return real_open(path, flags, *args, **kwargs)

            monkeypatch.setattr(dino_module.os, "open", _swap_child_before_open)

    assert swapped is True
    assert original is not None and original.exists()
    assert replacement is not None
    assert (replacement / "external-sentinel").read_bytes() == b"must-survive\n"


def test_cleanup_unlinks_dangling_anchor_replacement_before_parent_fd_closes(
    dino_module,
    monkeypatch,
    tmp_path,
):
    snapshot, authority, _ = _snapshot_fixture(tmp_path / "source")
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    source_authority = dino_module._validate_local_snapshot(
        snapshot,
        revision=snapshot.name,
        files=authority,
    )
    private_parent = tmp_path / "private-parent"
    private_parent.mkdir()
    real_mkdtemp = dino_module.tempfile.mkdtemp
    monkeypatch.setattr(
        dino_module.tempfile,
        "mkdtemp",
        lambda *, prefix: real_mkdtemp(prefix=prefix, dir=private_parent),
    )

    anchor: Path | None = None
    moved_anchor: Path | None = None
    with dino_module._materialized_local_snapshot(source_authority) as materialized:
        anchor = materialized.anchor
        moved_anchor = anchor.with_name(anchor.name + ".moved")
        anchor.rename(moved_anchor)
        anchor.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    assert anchor is not None and not anchor.is_symlink()
    assert moved_anchor is not None and not moved_anchor.exists()
    assert not (tmp_path / "missing-target").exists()


def test_parent_open_failure_never_mutates_lexical_empty_dir_replacement(
    dino_module,
    monkeypatch,
    tmp_path,
):
    snapshot, authority, _ = _snapshot_fixture(tmp_path / "source")
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    source_authority = dino_module._validate_local_snapshot(
        snapshot,
        revision=snapshot.name,
        files=authority,
    )
    private_parent = tmp_path / "private-parent"
    private_parent.mkdir()
    external_sentinel = tmp_path / "external-sentinel"
    external_sentinel.write_bytes(b"must-survive\n")
    real_mkdtemp = dino_module.tempfile.mkdtemp
    monkeypatch.setattr(
        dino_module.tempfile,
        "mkdtemp",
        lambda *, prefix: real_mkdtemp(prefix=prefix, dir=private_parent),
    )
    real_open = dino_module.os.open
    original: Path | None = None
    replacement: Path | None = None

    def _fail_parent_open(path, flags, *args, **kwargs):
        nonlocal original, replacement
        if Path(path) == private_parent and kwargs.get("dir_fd") is None:
            [replacement] = list(private_parent.glob(".gap-grounding-dino-snapshot-*"))
            original = replacement.with_name(replacement.name + ".original")
            replacement.rename(original)
            replacement.mkdir()
            raise OSError("simulated parent open failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(dino_module.os, "open", _fail_parent_open)

    with pytest.raises(dino_module._SnapshotAuthorityChanged) as exc_info:
        with dino_module._materialized_local_snapshot(source_authority):
            raise AssertionError("parent-open failure must not yield materialization")

    assert str(exc_info.value) == "private snapshot materialization failed"
    assert original is not None and original.exists()
    assert replacement is not None and replacement.is_dir()
    assert external_sentinel.read_bytes() == b"must-survive\n"


@pytest.mark.parametrize("restore_entry", [False, True])
def test_private_snapshot_drift_discards_cache_before_second_loader(
    dino_module,
    monkeypatch,
    tmp_path,
    restore_entry,
):
    snapshot, authority, cached = _snapshot_fixture(tmp_path)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    monkeypatch.setattr(
        dino_module,
        "_try_to_load_from_cache",
        lambda model_name, filename, revision: cached[filename],
    )
    calls = {"processor": 0, "model": 0}

    class FakeProcessorFactory:
        @staticmethod
        def from_pretrained(source: str, **kwargs: Any):
            calls["processor"] += 1
            root = Path(source)
            config = root / "config.json"
            os.chmod(root, 0o700)
            if restore_entry:
                parked = root / "config.json.sealed"
                config.rename(parked)
                config.write_bytes(b"unsealed\n")
                assert config.read_bytes() == b"unsealed\n"
                config.unlink()
                parked.rename(config)
            else:
                config.chmod(0o600)
                config.write_bytes(b"unsealed-data\n")
            os.chmod(root, 0o500)
            return object()

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(source: str, **kwargs: Any):
            calls["model"] += 1
            raise AssertionError("private drift must stop before the model loader")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoProcessor=FakeProcessorFactory,
            AutoModelForZeroShotObjectDetection=FakeModelFactory,
        ),
    )
    dino_module._model = None
    dino_module._processor = None

    with pytest.raises(RuntimeError, match=f"^{dino_module._LOCAL_SNAPSHOT_ERROR}$"):
        dino_module._get_model()

    assert calls == {"processor": 1, "model": 0}
    assert dino_module._model is None
    assert dino_module._processor is None


def test_private_snapshot_copy_rejects_source_content_drift(
    dino_module,
    monkeypatch,
    tmp_path,
):
    contents = {
        "config.json": b"a" * (1024 * 1024 + 1),
        "model.safetensors": b"sealed-weights\n",
    }
    snapshot, authority, _ = _snapshot_fixture(tmp_path, contents=contents)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_REVISION", snapshot.name)
    monkeypatch.setattr(dino_module, "_SNAPSHOT_FILES", authority)
    source_authority = dino_module._validate_local_snapshot(
        snapshot,
        revision=snapshot.name,
        files=authority,
    )
    config_blob = (snapshot / "config.json").resolve()
    real_read = dino_module.os.read
    drifted = False

    def _read_then_drift(fd: int, size: int) -> bytes:
        nonlocal drifted
        payload = real_read(fd, size)
        if not drifted and payload:
            drifted = True
            config_blob.write_bytes(b"b" * len(contents["config.json"]))
        return payload

    monkeypatch.setattr(dino_module.os, "read", _read_then_drift)

    with pytest.raises(dino_module._SnapshotAuthorityChanged):
        with dino_module._materialized_local_snapshot(source_authority):
            raise AssertionError("drifted bytes must not be yielded")


def test_snapshot_validation_is_isolated_across_spawned_processes(
    dino_module, skills_registry, tmp_path,
):
    snapshot, authority, _ = _snapshot_fixture(tmp_path)
    module_path = str(skills_registry.get("grounding-dino").meta.bundle_dir / "tools.py")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=2) as pool:
        results = pool.starmap(
            _spawn_snapshot_validation,
            [
                (module_path, str(snapshot), snapshot.name, authority),
                (module_path, str(snapshot), snapshot.name, authority),
            ],
        )

    assert len({result[0] for result in results}) == 2
    assert {result[1] for result in results} == {str(snapshot)}
    assert len({result[2] for result in results}) == 2
    assert {result[5] for result in results} == {False}


@pytest.mark.gpu
def test_detect_gpu_smoke(dino_registry):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    img = np.full((160, 160, 3), 255, dtype=np.uint8)
    img[40:120, 40:120] = (200, 30, 30)
    out = dino_registry.invoke(
        "grounding-dino.detect", image=img, query="red square"
    )
    assert set(out) == {"detections"}
    for det in out["detections"]:
        assert set(det) == {"box", "label", "score"}
        assert 0.0 <= det["score"] <= 1.0
        box = det["box"]
        assert box["x2"] >= box["x1"] and box["y2"] >= box["y1"]
