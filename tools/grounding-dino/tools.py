"""grounding-dino tool bundle — zero-shot object detection.

Extracted from the original Grounding DINO gRPC servicer in the dev
tree. The transformers pipeline
(processor → model → post_process_grounded_object_detection) is verbatim;
the proto byte decode/encode is replaced by numpy arrays + gap.types dicts.

The model loads lazily on first call (module-level singleton); importing
this module never pulls torch/transformers. Knobs via env:

- ``GAP_DINO_DEVICE`` — torch device (default ``cuda``).
- ``GAP_DINO_MODEL``  — must match the version-sealed default model;
  unsealed model overrides fail closed.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import stat
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import PerceptionFailed, ToolError
from gap_core.tools import tool
from gap_core.types import BoundingBox2D

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "IDEA-Research/grounding-dino-base"
_MODEL_CACHE_DIR = "models--IDEA-Research--grounding-dino-base"
_DEFAULT_BOX_THRESHOLD = 0.20
_DEFAULT_TEXT_THRESHOLD = 0.20

# Runtime model authority.  ``prefetch`` may use the network to populate this
# exact revision, but readiness and execution only accept these local bytes.
_SNAPSHOT_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
_SNAPSHOT_FILES: dict[str, tuple[int, str]] = {
    "config.json": (
        1_737,
        "eda416dae6f49419ff831b1c190ec430a060b19aae688dbaf2425a075b650608",
    ),
    "model.safetensors": (
        933_400_872,
        "5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21",
    ),
    "preprocessor_config.json": (
        457,
        "8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60",
    ),
    "special_tokens_map.json": (
        125,
        "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    ),
    "tokenizer_config.json": (
        1_237,
        "d40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56",
    ),
    "tokenizer.json": (
        711_396,
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
    "vocab.txt": (
        231_508,
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    ),
}
_LOCAL_SNAPSHOT_ERROR = "sealed local Grounding DINO snapshot is unavailable or invalid"

_DEVICE = os.environ.get("GAP_DINO_DEVICE", "cuda")
_MODEL_NAME = os.environ.get("GAP_DINO_MODEL", _DEFAULT_MODEL_NAME)

_load_lock = threading.Lock()
_model: Any = None
_processor: Any = None


class Detection(TypedDict):
    box: BoundingBox2D    # [x1, y1, x2, y2] in pixels
    label: str            # matched text label
    score: float          # confidence [0, 1]


class DetectResult(TypedDict):
    detections: list[Detection]


@dataclass(frozen=True)
class SnapshotFileAuthority:
    """Immutable identity of one required snapshot symlink and cache blob."""

    filename: str
    lexical_path: Path
    link_device: int
    link_inode: int
    link_mode: int
    relative_target: str
    blob_path: Path
    blob_device: int
    blob_inode: int
    blob_mode: int
    blob_size: int
    blob_mtime_ns: int
    blob_ctime_ns: int
    blob_sha256: str


@dataclass(frozen=True)
class SnapshotAuthority:
    """Immutable, re-checkable authority for one local HF snapshot."""

    repo_id: str
    revision: str
    root: Path
    root_device: int
    root_inode: int
    root_mode: int
    files: tuple[SnapshotFileAuthority, ...]


@dataclass(frozen=True)
class MaterializedSnapshotFileAuthority:
    """Identity of one regular file in a process-private sealed snapshot."""

    filename: str
    device: int
    inode: int
    mode: int
    nlink: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


@dataclass(frozen=True)
class MaterializedSnapshotAuthority:
    """Open-FD authority for one process-private immutable snapshot view."""

    anchor: Path
    anchor_parent_fd: int
    anchor_parent_device: int
    anchor_parent_inode: int
    anchor_fd: int
    anchor_device: int
    anchor_inode: int
    anchor_mode: int
    root: Path
    loader_root: Path
    root_fd: int
    mutation_watch_fd: int
    mutation_watch_descriptor: int
    root_device: int
    root_inode: int
    root_mode: int
    root_nlink: int
    root_size: int
    root_mtime_ns: int
    root_ctime_ns: int
    files: tuple[MaterializedSnapshotFileAuthority, ...]


class _SnapshotAuthorityChanged(RuntimeError):
    """The cache no longer matches a previously resolved authority."""


@dataclass(frozen=True)
class _TrustedDirectoryHandle:
    """An open directory FD validated against a pre-recorded inode identity."""

    fd: int
    device: int
    inode: int


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_local_snapshot(
    snapshot: Path,
    *,
    revision: str,
    files: Mapping[str, tuple[int, str]],
    repo_id: str = _DEFAULT_MODEL_NAME,
) -> SnapshotAuthority:
    """Validate one standard HF snapshot without following its directory.

    A Hugging Face snapshot directory is real while its entries are symlinks
    into the model-local ``blobs`` store.  Keeping that topology explicit lets
    the runtime reject an alias to a different revision or blob namespace.
    Ancestors above the model cache may themselves be deployment symlinks.
    """
    snapshot = Path(snapshot)
    snapshot_info = snapshot.lstat()
    if not stat.S_ISDIR(snapshot_info.st_mode) or snapshot.is_symlink():
        raise ValueError("snapshot directory is not authoritative")
    if (
        snapshot.name != revision
        or snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != _MODEL_CACHE_DIR
    ):
        raise ValueError("snapshot revision does not match authority")

    model_root = snapshot.parent.parent
    blobs = model_root / "blobs"
    for directory in (model_root, snapshot.parent, blobs):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or directory.is_symlink():
            raise ValueError("model cache topology is not authoritative")
    resolved_blobs = blobs.resolve(strict=True)

    file_authorities: list[SnapshotFileAuthority] = []
    for filename, (expected_size, expected_sha256) in files.items():
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise ValueError("snapshot authority contains an invalid filename")
        entry = snapshot / filename
        link_info = entry.lstat()
        if not stat.S_ISLNK(link_info.st_mode):
            raise ValueError("snapshot entry is not a cache blob link")
        relative_target = os.readlink(entry)
        expected_relative = str(Path("../../blobs") / Path(relative_target).name)
        if relative_target != expected_relative:
            raise ValueError("snapshot entry target is not canonical")
        target = entry.resolve(strict=True)
        target_info = target.lstat()
        if not stat.S_ISREG(target_info.st_mode) or target.parent != resolved_blobs:
            raise ValueError("snapshot entry escapes the model blob store")
        if target_info.st_size != expected_size:
            raise ValueError("snapshot entry size does not match authority")
        if _sha256_file(target) != expected_sha256:
            raise ValueError("snapshot entry digest does not match authority")
        file_authorities.append(SnapshotFileAuthority(
            filename=filename,
            lexical_path=entry,
            link_device=link_info.st_dev,
            link_inode=link_info.st_ino,
            link_mode=link_info.st_mode,
            relative_target=relative_target,
            blob_path=target,
            blob_device=target_info.st_dev,
            blob_inode=target_info.st_ino,
            blob_mode=target_info.st_mode,
            blob_size=target_info.st_size,
            blob_mtime_ns=target_info.st_mtime_ns,
            blob_ctime_ns=target_info.st_ctime_ns,
            blob_sha256=expected_sha256,
        ))

    authority = SnapshotAuthority(
        repo_id=repo_id,
        revision=revision,
        root=snapshot,
        root_device=snapshot_info.st_dev,
        root_inode=snapshot_info.st_ino,
        root_mode=snapshot_info.st_mode,
        files=tuple(file_authorities),
    )
    _assert_snapshot_authority_unchanged(authority)
    return authority


def _assert_snapshot_authority_unchanged(authority: SnapshotAuthority) -> None:
    """Fail if any contract-relevant snapshot identity or byte has drifted."""
    try:
        if (
            authority.repo_id != _MODEL_NAME
            or authority.revision != _SNAPSHOT_REVISION
            or authority.repo_id != _DEFAULT_MODEL_NAME
        ):
            raise ValueError("snapshot selector drifted")

        root_info = authority.root.lstat()
        if (
            authority.root.is_symlink()
            or not stat.S_ISDIR(root_info.st_mode)
            or (
                root_info.st_dev,
                root_info.st_ino,
                root_info.st_mode,
            )
            != (
                authority.root_device,
                authority.root_inode,
                authority.root_mode,
            )
        ):
            raise ValueError("snapshot root identity drifted")
        if (
            authority.root.name != authority.revision
            or authority.root.parent.name != "snapshots"
            or authority.root.parent.parent.name != _MODEL_CACHE_DIR
        ):
            raise ValueError("snapshot root topology drifted")
        for directory in (
            authority.root.parent.parent,
            authority.root.parent,
            authority.root.parent.parent / "blobs",
        ):
            info = directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or directory.is_symlink():
                raise ValueError("model cache topology drifted")

        expected_names = set(_SNAPSHOT_FILES)
        if len(authority.files) != len(expected_names):
            raise ValueError("snapshot authority width drifted")
        seen_names: set[str] = set()
        for item in authority.files:
            if item.filename in seen_names or item.filename not in expected_names:
                raise ValueError("snapshot authority names drifted")
            seen_names.add(item.filename)
            if item.lexical_path != authority.root / item.filename:
                raise ValueError("snapshot lexical path drifted")
            expected_size, expected_sha256 = _SNAPSHOT_FILES[item.filename]
            if (
                item.blob_size != expected_size
                or item.blob_sha256 != expected_sha256
            ):
                raise ValueError("snapshot declared bytes drifted")

            link_info = item.lexical_path.lstat()
            if (
                not stat.S_ISLNK(link_info.st_mode)
                or (
                    link_info.st_dev,
                    link_info.st_ino,
                    link_info.st_mode,
                )
                != (item.link_device, item.link_inode, item.link_mode)
                or os.readlink(item.lexical_path) != item.relative_target
            ):
                raise ValueError("snapshot symlink identity drifted")
            if item.lexical_path.resolve(strict=True) != item.blob_path:
                raise ValueError("snapshot symlink target drifted")

            blob_info = item.blob_path.lstat()
            if (
                not stat.S_ISREG(blob_info.st_mode)
                or item.blob_path.parent
                != (authority.root.parent.parent / "blobs").resolve(strict=True)
                or (
                    blob_info.st_dev,
                    blob_info.st_ino,
                    blob_info.st_mode,
                    blob_info.st_size,
                    blob_info.st_mtime_ns,
                    blob_info.st_ctime_ns,
                )
                != (
                    item.blob_device,
                    item.blob_inode,
                    item.blob_mode,
                    item.blob_size,
                    item.blob_mtime_ns,
                    item.blob_ctime_ns,
                )
                or _sha256_file(item.blob_path) != item.blob_sha256
            ):
                raise ValueError("snapshot blob identity drifted")
        if seen_names != expected_names:
            raise ValueError("snapshot authority is incomplete")
    except _SnapshotAuthorityChanged:
        raise
    except Exception as exc:
        raise _SnapshotAuthorityChanged("snapshot authority changed") from exc


def _snapshot_authority_digest(authority: SnapshotAuthority) -> str:
    """Canonical digest for gates that bind the resolved cache topology."""
    payload = {
        "repo_id": authority.repo_id,
        "revision": authority.revision,
        "root": str(authority.root),
        "root_device": authority.root_device,
        "root_inode": authority.root_inode,
        "root_mode": authority.root_mode,
        "files": [
            {
                "filename": item.filename,
                "link_device": item.link_device,
                "link_inode": item.link_inode,
                "link_mode": item.link_mode,
                "relative_target": item.relative_target,
                "blob_path": str(item.blob_path),
                "blob_device": item.blob_device,
                "blob_inode": item.blob_inode,
                "blob_mode": item.blob_mode,
                "blob_size": item.blob_size,
                "blob_mtime_ns": item.blob_mtime_ns,
                "blob_ctime_ns": item.blob_ctime_ns,
                "blob_sha256": item.blob_sha256,
            }
            for item in authority.files
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _try_to_load_from_cache(model_name: str, filename: str, revision: str):
    """Filesystem-only Hugging Face lookup seam (kept lazy for bare imports)."""
    from huggingface_hub import try_to_load_from_cache

    return try_to_load_from_cache(model_name, filename, revision=revision)


def _resolve_local_snapshot() -> SnapshotAuthority:
    """Resolve and verify the single sealed runtime model snapshot.

    All errors intentionally collapse to one constant message: deployment
    paths and cache metadata must not leak through an RPC error response.
    """
    try:
        if _MODEL_NAME != _DEFAULT_MODEL_NAME:
            raise ValueError("configured model has no sealed authority")

        snapshot: Path | None = None
        for filename in _SNAPSHOT_FILES:
            cached = _try_to_load_from_cache(
                _MODEL_NAME, filename, _SNAPSHOT_REVISION
            )
            if not isinstance(cached, str):
                raise ValueError("required snapshot entry is absent")
            entry = Path(cached)
            if snapshot is None:
                snapshot = entry.parent
            elif entry.parent != snapshot:
                raise ValueError("required entries resolve to multiple snapshots")
        if snapshot is None:
            raise ValueError("snapshot authority is empty")
        return _validate_local_snapshot(
            snapshot,
            revision=_SNAPSHOT_REVISION,
            files=_SNAPSHOT_FILES,
            repo_id=_MODEL_NAME,
        )
    except Exception:
        raise RuntimeError(_LOCAL_SNAPSHOT_ERROR) from None


def weights_cached() -> bool:
    """Return whether the sealed runtime snapshot passes full local validation."""
    try:
        authority = _resolve_local_snapshot()
        _assert_snapshot_authority_unchanged(authority)
    except RuntimeError:
        return False
    return True


def _checked_authority(authority: SnapshotAuthority) -> None:
    try:
        _assert_snapshot_authority_unchanged(authority)
    except _SnapshotAuthorityChanged:
        raise RuntimeError(_LOCAL_SNAPSHOT_ERROR) from None


def _source_blob_matches_authority(
    item: SnapshotFileAuthority,
    info: os.stat_result,
) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        == (
            item.blob_device,
            item.blob_inode,
            item.blob_mode,
            item.blob_size,
            item.blob_mtime_ns,
            item.blob_ctime_ns,
        )
    )


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("private snapshot write made no progress")
        view = view[written:]


def _copy_authoritative_blob(
    item: SnapshotFileAuthority,
    root_fd: int,
) -> MaterializedSnapshotFileAuthority:
    source_fd = -1
    destination_fd = -1
    try:
        source_fd = os.open(
            item.blob_path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
        source_before = os.fstat(source_fd)
        if not _source_blob_matches_authority(item, source_before):
            raise _SnapshotAuthorityChanged("source blob identity changed")

        destination_fd = os.open(
            item.filename,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=root_fd,
        )
        digest = sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            _write_all(destination_fd, chunk)

        source_after = os.fstat(source_fd)
        if (
            not _source_blob_matches_authority(item, source_after)
            or (
                source_before.st_dev,
                source_before.st_ino,
                source_before.st_mode,
                source_before.st_size,
                source_before.st_mtime_ns,
                source_before.st_ctime_ns,
            )
            != (
                source_after.st_dev,
                source_after.st_ino,
                source_after.st_mode,
                source_after.st_size,
                source_after.st_mtime_ns,
                source_after.st_ctime_ns,
            )
            or copied != item.blob_size
            or digest.hexdigest() != item.blob_sha256
        ):
            raise _SnapshotAuthorityChanged("source blob changed during copy")

        os.fsync(destination_fd)
        os.fchmod(destination_fd, 0o400)
        destination_info = os.fstat(destination_fd)
        if (
            not stat.S_ISREG(destination_info.st_mode)
            or destination_info.st_nlink != 1
            or stat.S_IMODE(destination_info.st_mode) != 0o400
            or destination_info.st_size != item.blob_size
        ):
            raise _SnapshotAuthorityChanged("private snapshot file is not sealed")
        return MaterializedSnapshotFileAuthority(
            filename=item.filename,
            device=destination_info.st_dev,
            inode=destination_info.st_ino,
            mode=destination_info.st_mode,
            nlink=destination_info.st_nlink,
            size=destination_info.st_size,
            mtime_ns=destination_info.st_mtime_ns,
            ctime_ns=destination_info.st_ctime_ns,
            sha256=item.blob_sha256,
        )
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        if source_fd >= 0:
            os.close(source_fd)


def _same_inode(info: os.stat_result, *, device: int, inode: int) -> bool:
    return info.st_dev == device and info.st_ino == inode


def _trusted_directory_handle(
    fd: int,
    *,
    device: int,
    inode: int,
) -> _TrustedDirectoryHandle:
    info = os.fstat(fd)
    if (
        not stat.S_ISDIR(info.st_mode)
        or not _same_inode(info, device=device, inode=inode)
    ):
        raise _SnapshotAuthorityChanged("directory descriptor identity changed")
    return _TrustedDirectoryHandle(fd=fd, device=device, inode=inode)


def _adopt_directory_fd_or_close(
    fd: int,
    *,
    device: int,
    inode: int,
) -> _TrustedDirectoryHandle | None:
    try:
        return _trusted_directory_handle(fd, device=device, inode=inode)
    except Exception:
        os.close(fd)
        return None


def _empty_trusted_directory(handle: _TrustedDirectoryHandle) -> None:
    """Remove descendants relative to an open directory without following links."""
    _trusted_directory_handle(
        handle.fd,
        device=handle.device,
        inode=handle.inode,
    )
    os.fchmod(handle.fd, 0o700)
    for name in os.listdir(handle.fd):
        info = os.stat(name, dir_fd=handle.fd, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            os.unlink(name, dir_fd=handle.fd)
            continue

        candidate_fd = os.open(
            name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=handle.fd,
        )
        try:
            child = _trusted_directory_handle(
                candidate_fd,
                device=info.st_dev,
                inode=info.st_ino,
            )
        except BaseException:
            os.close(candidate_fd)
            raise
        try:
            _empty_trusted_directory(child)
            current = os.stat(name, dir_fd=handle.fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(current.st_mode)
                or not _same_inode(current, device=info.st_dev, inode=info.st_ino)
            ):
                raise _SnapshotAuthorityChanged("cleanup directory entry changed")
            os.rmdir(name, dir_fd=handle.fd)
        finally:
            os.close(child.fd)


def _find_directory_entry_by_inode(
    parent: _TrustedDirectoryHandle,
    *,
    device: int,
    inode: int,
) -> str | None:
    _trusted_directory_handle(
        parent.fd,
        device=parent.device,
        inode=parent.inode,
    )
    matches: list[str] = []
    for name in os.listdir(parent.fd):
        info = os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
        if _same_inode(info, device=device, inode=inode):
            if not stat.S_ISDIR(info.st_mode):
                raise _SnapshotAuthorityChanged("cleanup entry type changed")
            matches.append(name)
    if len(matches) > 1:
        raise _SnapshotAuthorityChanged("cleanup directory identity is ambiguous")
    return matches[0] if matches else None


def _open_current_parent_for_directory_fd(
    directory: _TrustedDirectoryHandle,
) -> tuple[_TrustedDirectoryHandle, str]:
    locator = os.readlink(f"/proc/self/fd/{directory.fd}")
    if locator.endswith(" (deleted)"):
        raise _SnapshotAuthorityChanged("cleanup directory has no locator")
    current = Path(locator)
    parent_fd = os.open(
        current.parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        parent_info = os.fstat(parent_fd)
        parent = _trusted_directory_handle(
            parent_fd,
            device=parent_info.st_dev,
            inode=parent_info.st_ino,
        )
        info = os.stat(current.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(info.st_mode)
            or not _same_inode(
                info,
                device=directory.device,
                inode=directory.inode,
            )
        ):
            raise _SnapshotAuthorityChanged("cleanup locator identity changed")
    except BaseException:
        os.close(parent_fd)
        raise
    return parent, current.name


def _remove_pinned_directory_entry(
    directory: _TrustedDirectoryHandle,
    *,
    preferred_parent: _TrustedDirectoryHandle | None,
) -> None:
    parent = preferred_parent
    close_parent = False
    name = None
    if parent is not None:
        name = _find_directory_entry_by_inode(
            parent,
            device=directory.device,
            inode=directory.inode,
        )
    if name is None:
        parent, name = _open_current_parent_for_directory_fd(directory)
        close_parent = True
    try:
        if os.listdir(directory.fd):
            raise _SnapshotAuthorityChanged("cleanup directory is not empty")
        current = os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current.st_mode)
            or not _same_inode(
                current,
                device=directory.device,
                inode=directory.inode,
            )
        ):
            raise _SnapshotAuthorityChanged("cleanup entry identity changed")
        os.rmdir(name, dir_fd=parent.fd)
    finally:
        if close_parent:
            os.close(parent.fd)


def _open_trusted_directory_by_inode(
    parent: _TrustedDirectoryHandle,
    *,
    device: int,
    inode: int,
) -> _TrustedDirectoryHandle | None:
    name = _find_directory_entry_by_inode(
        parent,
        device=device,
        inode=inode,
    )
    if name is None:
        return None
    candidate_fd = os.open(
        name,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent.fd,
    )
    try:
        return _trusted_directory_handle(
            candidate_fd,
            device=device,
            inode=inode,
        )
    except BaseException:
        os.close(candidate_fd)
        raise


def _settle_anchor_lexical_remnant(
    anchor: Path,
    parent: _TrustedDirectoryHandle,
) -> None:
    """Remove only a replacement symlink; never follow or mutate other remnants."""
    try:
        info = os.stat(
            anchor.name,
            dir_fd=parent.fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if not stat.S_ISLNK(info.st_mode):
        raise _SnapshotAuthorityChanged("private anchor remnant remains")
    os.unlink(anchor.name, dir_fd=parent.fd)
    try:
        os.stat(
            anchor.name,
            dir_fd=parent.fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    raise _SnapshotAuthorityChanged("private anchor remnant changed")


def _cleanup_private_snapshot_handles(
    *,
    anchor: Path,
    anchor_parent: _TrustedDirectoryHandle,
    anchor_handle: _TrustedDirectoryHandle | None,
    anchor_device: int,
    anchor_inode: int,
    root_handle: _TrustedDirectoryHandle | None,
    root_device: int,
    root_inode: int,
    root_was_trusted: bool,
) -> None:
    cleanup_failed = False
    first_error: Exception | None = None
    if anchor_handle is None:
        try:
            anchor_handle = _open_trusted_directory_by_inode(
                anchor_parent,
                device=anchor_device,
                inode=anchor_inode,
            )
        except Exception as exc:
            cleanup_failed = True
            first_error = exc
    if anchor_handle is None:
        cleanup_failed = True
        first_error = first_error or _SnapshotAuthorityChanged(
            "private anchor cannot be recovered"
        )

    root_cleanup_succeeded = root_device < 0
    if anchor_handle is not None and root_handle is None and root_device >= 0:
        try:
            root_handle = _open_trusted_directory_by_inode(
                anchor_handle,
                device=root_device,
                inode=root_inode,
            )
        except Exception as exc:
            cleanup_failed = True
            first_error = first_error or exc
        if root_handle is None:
            cleanup_failed = True
            first_error = first_error or _SnapshotAuthorityChanged(
                "private root cannot be recovered"
            )

    if anchor_handle is not None:
        try:
            os.fchmod(anchor_handle.fd, 0o700)
        except OSError as exc:
            cleanup_failed = True
            first_error = first_error or exc
    if root_handle is not None:
        try:
            _empty_trusted_directory(root_handle)
            _remove_pinned_directory_entry(
                root_handle,
                preferred_parent=anchor_handle,
            )
            root_cleanup_succeeded = True
        except Exception as exc:
            cleanup_failed = True
            first_error = first_error or exc
        finally:
            try:
                os.close(root_handle.fd)
            except OSError:
                cleanup_failed = True
                first_error = first_error or _SnapshotAuthorityChanged(
                    "private root descriptor close failed"
                )

    if anchor_handle is not None:
        try:
            if root_was_trusted and root_cleanup_succeeded:
                _empty_trusted_directory(anchor_handle)
            _remove_pinned_directory_entry(
                anchor_handle,
                preferred_parent=anchor_parent,
            )
        except Exception as exc:
            cleanup_failed = True
            first_error = first_error or exc
        finally:
            try:
                os.close(anchor_handle.fd)
            except OSError:
                cleanup_failed = True
                first_error = first_error or _SnapshotAuthorityChanged(
                    "private anchor descriptor close failed"
                )
    try:
        _settle_anchor_lexical_remnant(anchor, anchor_parent)
    except Exception as exc:
        cleanup_failed = True
        first_error = first_error or exc
    try:
        os.close(anchor_parent.fd)
    except OSError:
        cleanup_failed = True
        first_error = first_error or _SnapshotAuthorityChanged(
            "private anchor parent descriptor close failed"
        )

    if cleanup_failed:
        raise _SnapshotAuthorityChanged("private snapshot cleanup failed") from first_error


def _remove_private_snapshot(authority: MaterializedSnapshotAuthority) -> None:
    anchor_parent = _adopt_directory_fd_or_close(
        authority.anchor_parent_fd,
        device=authority.anchor_parent_device,
        inode=authority.anchor_parent_inode,
    )
    anchor_handle = _adopt_directory_fd_or_close(
        authority.anchor_fd,
        device=authority.anchor_device,
        inode=authority.anchor_inode,
    )
    root_handle = _adopt_directory_fd_or_close(
        authority.root_fd,
        device=authority.root_device,
        inode=authority.root_inode,
    )
    if anchor_parent is None:
        for handle in (root_handle, anchor_handle):
            if handle is not None:
                os.close(handle.fd)
        raise _SnapshotAuthorityChanged("private snapshot cleanup failed")
    _cleanup_private_snapshot_handles(
        anchor=authority.anchor,
        anchor_parent=anchor_parent,
        anchor_handle=anchor_handle,
        anchor_device=authority.anchor_device,
        anchor_inode=authority.anchor_inode,
        root_handle=root_handle,
        root_device=authority.root_device,
        root_inode=authority.root_inode,
        root_was_trusted=root_handle is not None,
    )


_INOTIFY_MUTATION_MASK = (
    0x00000002  # IN_MODIFY
    | 0x00000004  # IN_ATTRIB
    | 0x00000008  # IN_CLOSE_WRITE
    | 0x00000040  # IN_MOVED_FROM
    | 0x00000080  # IN_MOVED_TO
    | 0x00000100  # IN_CREATE
    | 0x00000200  # IN_DELETE
    | 0x00000400  # IN_DELETE_SELF
    | 0x00000800  # IN_MOVE_SELF
    | 0x00002000  # IN_UNMOUNT
    | 0x00004000  # IN_Q_OVERFLOW
    | 0x00008000  # IN_IGNORED
)


def _start_private_snapshot_mutation_watch(root: Path) -> tuple[int, int]:
    """Open a Linux inotify queue that records even restored mutations."""
    libc = ctypes.CDLL(None, use_errno=True)
    watch_fd = libc.inotify_init1(os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    if watch_fd < 0:
        raise OSError(ctypes.get_errno(), "private snapshot watch unavailable")
    descriptor = libc.inotify_add_watch(
        watch_fd,
        os.fsencode(root),
        _INOTIFY_MUTATION_MASK,
    )
    if descriptor < 0:
        error_number = ctypes.get_errno()
        os.close(watch_fd)
        raise OSError(error_number, "private snapshot watch unavailable")
    return watch_fd, descriptor


def _assert_no_private_snapshot_mutation(
    authority: MaterializedSnapshotAuthority,
) -> None:
    try:
        event = os.read(authority.mutation_watch_fd, 64 * 1024)
    except BlockingIOError:
        return
    if event:
        raise _SnapshotAuthorityChanged("private snapshot mutation observed")


def _create_materialized_local_snapshot(
    authority: SnapshotAuthority,
) -> MaterializedSnapshotAuthority:
    anchor = Path(tempfile.mkdtemp(prefix=".gap-grounding-dino-snapshot-"))
    anchor_parent_fd = -1
    anchor_fd = -1
    root_fd = -1
    mutation_watch_fd = -1
    anchor_device = -1
    anchor_inode = -1
    root_device = -1
    root_inode = -1
    anchor_parent_handle: _TrustedDirectoryHandle | None = None
    anchor_handle: _TrustedDirectoryHandle | None = None
    root_handle: _TrustedDirectoryHandle | None = None
    try:
        anchor_parent_fd = os.open(
            anchor.parent,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
        )
        anchor_parent_info = os.fstat(anchor_parent_fd)
        anchor_parent_handle = _trusted_directory_handle(
            anchor_parent_fd,
            device=anchor_parent_info.st_dev,
            inode=anchor_parent_info.st_ino,
        )
        anchor_info = os.stat(
            anchor.name,
            dir_fd=anchor_parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(anchor_info.st_mode)
            or stat.S_IMODE(anchor_info.st_mode) != 0o700
        ):
            raise _SnapshotAuthorityChanged("private snapshot anchor is not private")
        anchor_device = anchor_info.st_dev
        anchor_inode = anchor_info.st_ino
        anchor_fd = os.open(
            anchor.name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=anchor_parent_fd,
        )
        try:
            anchor_handle = _trusted_directory_handle(
                anchor_fd,
                device=anchor_device,
                inode=anchor_inode,
            )
        except BaseException:
            os.close(anchor_fd)
            anchor_fd = -1
            raise

        root_name = "snapshot"
        os.mkdir(root_name, mode=0o700, dir_fd=anchor_fd)
        root = anchor / root_name
        initial_root_info = os.stat(
            root_name,
            dir_fd=anchor_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(initial_root_info.st_mode):
            raise _SnapshotAuthorityChanged("private snapshot root type changed")
        root_device = initial_root_info.st_dev
        root_inode = initial_root_info.st_ino
        root_fd = os.open(
            root_name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=anchor_fd,
        )
        try:
            root_handle = _trusted_directory_handle(
                root_fd,
                device=root_device,
                inode=root_inode,
            )
        except BaseException:
            os.close(root_fd)
            root_fd = -1
            raise
        files = tuple(
            _copy_authoritative_blob(item, root_fd) for item in authority.files
        )
        os.fchmod(root_fd, 0o500)
        os.fchmod(anchor_fd, 0o500)
        sealed_root_info = os.fstat(root_fd)
        sealed_anchor_info = os.fstat(anchor_fd)
        mutation_watch_fd, mutation_watch_descriptor = (
            _start_private_snapshot_mutation_watch(Path(f"/proc/self/fd/{root_fd}"))
        )
        materialized = MaterializedSnapshotAuthority(
            anchor=anchor,
            anchor_parent_fd=anchor_parent_fd,
            anchor_parent_device=anchor_parent_handle.device,
            anchor_parent_inode=anchor_parent_handle.inode,
            anchor_fd=anchor_fd,
            anchor_device=sealed_anchor_info.st_dev,
            anchor_inode=sealed_anchor_info.st_ino,
            anchor_mode=sealed_anchor_info.st_mode,
            root=root,
            loader_root=Path(f"/proc/self/fd/{root_fd}"),
            root_fd=root_fd,
            mutation_watch_fd=mutation_watch_fd,
            mutation_watch_descriptor=mutation_watch_descriptor,
            root_device=sealed_root_info.st_dev,
            root_inode=sealed_root_info.st_ino,
            root_mode=sealed_root_info.st_mode,
            root_nlink=sealed_root_info.st_nlink,
            root_size=sealed_root_info.st_size,
            root_mtime_ns=sealed_root_info.st_mtime_ns,
            root_ctime_ns=sealed_root_info.st_ctime_ns,
            files=files,
        )
        _assert_materialized_snapshot_unchanged(materialized)
        return materialized
    except BaseException as exc:
        cleanup_failed = False
        if mutation_watch_fd >= 0:
            os.close(mutation_watch_fd)
        root_was_trusted = root_handle is not None
        if anchor_parent_handle is not None:
            try:
                _cleanup_private_snapshot_handles(
                    anchor=anchor,
                    anchor_parent=anchor_parent_handle,
                    anchor_handle=anchor_handle,
                    anchor_device=anchor_device,
                    anchor_inode=anchor_inode,
                    root_handle=root_handle,
                    root_device=root_device,
                    root_inode=root_inode,
                    root_was_trusted=root_was_trusted,
                )
            except _SnapshotAuthorityChanged:
                cleanup_failed = True
        else:
            for fd in (root_fd, anchor_fd, anchor_parent_fd):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        cleanup_failed = True
            # Without a trusted parent descriptor no lexical entry is safe to
            # mutate, even if it currently looks empty.
            cleanup_failed = True
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, _SnapshotAuthorityChanged) and not cleanup_failed:
            raise
        raise _SnapshotAuthorityChanged("private snapshot materialization failed") from exc


def _assert_materialized_snapshot_unchanged(
    authority: MaterializedSnapshotAuthority,
) -> None:
    """Revalidate the open private directory and every sealed regular file."""
    try:
        _assert_no_private_snapshot_mutation(authority)
        root_info = os.fstat(authority.root_fd)
        anchor_info = os.fstat(authority.anchor_fd)
        anchor_parent_info = os.fstat(authority.anchor_parent_fd)
        if (
            not stat.S_ISDIR(anchor_parent_info.st_mode)
            or (
                anchor_parent_info.st_dev,
                anchor_parent_info.st_ino,
            )
            != (
                authority.anchor_parent_device,
                authority.anchor_parent_inode,
            )
        ):
            raise ValueError("private snapshot anchor parent identity drifted")
        if (
            not stat.S_ISDIR(anchor_info.st_mode)
            or (
                anchor_info.st_dev,
                anchor_info.st_ino,
                anchor_info.st_mode,
            )
            != (
                authority.anchor_device,
                authority.anchor_inode,
                authority.anchor_mode,
            )
            or stat.S_IMODE(anchor_info.st_mode) != 0o500
        ):
            raise ValueError("private snapshot anchor identity drifted")
        if (
            not stat.S_ISDIR(root_info.st_mode)
            or (
                root_info.st_dev,
                root_info.st_ino,
                root_info.st_mode,
                root_info.st_nlink,
                root_info.st_size,
                root_info.st_mtime_ns,
                root_info.st_ctime_ns,
            )
            != (
                authority.root_device,
                authority.root_inode,
                authority.root_mode,
                authority.root_nlink,
                authority.root_size,
                authority.root_mtime_ns,
                authority.root_ctime_ns,
            )
            or stat.S_IMODE(root_info.st_mode) != 0o500
        ):
            raise ValueError("private snapshot root identity drifted")

        expected_names = {item.filename for item in authority.files}
        if set(os.listdir(authority.root_fd)) != expected_names:
            raise ValueError("private snapshot contents drifted")
        for item in authority.files:
            fd = os.open(
                item.filename,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=authority.root_fd,
            )
            try:
                before = os.fstat(fd)
                digest = sha256()
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                after = os.fstat(fd)
            finally:
                os.close(fd)
            expected_identity = (
                item.device,
                item.inode,
                item.mode,
                item.nlink,
                item.size,
                item.mtime_ns,
                item.ctime_ns,
            )
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o400
                or before.st_nlink != 1
                or (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_nlink,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                != expected_identity
                or (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_nlink,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                != expected_identity
                or digest.hexdigest() != item.sha256
            ):
                raise ValueError("private snapshot file identity drifted")
        _assert_no_private_snapshot_mutation(authority)
    except _SnapshotAuthorityChanged:
        raise
    except Exception as exc:
        raise _SnapshotAuthorityChanged("private snapshot changed") from exc


@contextmanager
def _materialized_local_snapshot(
    authority: SnapshotAuthority,
) -> Iterator[MaterializedSnapshotAuthority]:
    """Yield one FD-pinned private snapshot and always remove its pathname."""
    materialized: MaterializedSnapshotAuthority | None = None
    try:
        materialized = _create_materialized_local_snapshot(authority)
        yield materialized
    finally:
        if materialized is not None:
            cleanup_error = False
            cleanup_cause: Exception | None = None
            try:
                os.close(materialized.mutation_watch_fd)
            except OSError as exc:
                cleanup_error = True
                cleanup_cause = exc
            try:
                _remove_private_snapshot(materialized)
            except _SnapshotAuthorityChanged as exc:
                cleanup_error = True
                cleanup_cause = cleanup_cause or exc
            if cleanup_error:
                raise _SnapshotAuthorityChanged(
                    "private snapshot cleanup failed"
                ) from cleanup_cause


def _checked_materialized_authority(
    authority: MaterializedSnapshotAuthority,
) -> None:
    try:
        _assert_materialized_snapshot_unchanged(authority)
    except _SnapshotAuthorityChanged:
        raise RuntimeError(_LOCAL_SNAPSHOT_ERROR) from None


def _load_with_authority_checks(
    source_authority: SnapshotAuthority,
    materialized_authority: MaterializedSnapshotAuthority,
    loader,
):
    """Load only while public and FD-pinned private authorities stay stable."""
    _checked_authority(source_authority)
    _checked_materialized_authority(materialized_authority)
    try:
        loaded = loader()
    except BaseException:
        # Prefer a deterministic authority incident over a loader's secondary
        # parse error when cache drift caused that error.
        _checked_authority(source_authority)
        _checked_materialized_authority(materialized_authority)
        raise
    _checked_authority(source_authority)
    _checked_materialized_authority(materialized_authority)
    return loaded


def _discard_partial_model_cache() -> None:
    global _model, _processor
    _model = None
    _processor = None


def prefetch() -> None:
    """Snapshot-download the configured GDINO weights into the HF cache.

    Called by ``gap skills check --download``. Idempotent: re-running
    against an already-cached snapshot is a near-no-op (HF revision
    check + symlink refresh). Raises on network / auth / disk errors so
    ``gap skills check --download`` exits non-zero.

    Uses ``snapshot_download`` rather than ``AutoModel.from_pretrained``
    so we never load torch / instantiate the model at prefetch time —
    important for CI lanes and for the bare-engine venv.
    """
    from huggingface_hub import snapshot_download

    logger.info("[grounding-dino] prefetching weights for %s ...", _MODEL_NAME)
    if _MODEL_NAME != _DEFAULT_MODEL_NAME:
        raise RuntimeError(_LOCAL_SNAPSHOT_ERROR)
    snapshot_download(
        repo_id=_MODEL_NAME,
        repo_type="model",
        revision=_SNAPSHOT_REVISION,
    )
    _resolve_local_snapshot()
    logger.info("[grounding-dino] prefetch complete (cached at HF default)")


def _get_model() -> tuple[Any, Any]:
    """Load the Grounding DINO model + processor once (lazy singleton)."""
    global _model, _processor
    with _load_lock:
        if _model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            authority = _resolve_local_snapshot()
            logger.info(
                "Loading Grounding DINO model: %s on %s ...", _MODEL_NAME, _DEVICE
            )
            try:
                with _materialized_local_snapshot(authority) as materialized:
                    loader_root = str(materialized.loader_root)
                    processor = _load_with_authority_checks(
                        authority,
                        materialized,
                        lambda: AutoProcessor.from_pretrained(
                            loader_root, local_files_only=True
                        ),
                    )
                    model = _load_with_authority_checks(
                        authority,
                        materialized,
                        lambda: AutoModelForZeroShotObjectDetection.from_pretrained(
                            loader_root,
                            local_files_only=True,
                            use_safetensors=True,
                        ),
                    )
            except _SnapshotAuthorityChanged:
                _discard_partial_model_cache()
                raise RuntimeError(_LOCAL_SNAPSHOT_ERROR) from None
            except BaseException:
                _discard_partial_model_cache()
                raise
            # Both transformers objects must be fully usable after the private
            # pathname and directory FD have been removed.
            model = model.to(_DEVICE)
            model.eval()
            _processor = processor
            _model = model
            logger.info("Grounding DINO model loaded successfully on %s.", _DEVICE)
        return _model, _processor


@tool(
    name="grounding-dino.detect",
    summary="Zero-shot object detection from a text prompt; returns labeled boxes with confidence scores.",
    tags=("perception",),
)
def detect(
    image: np.ndarray,
    query: str,
    box_threshold: float = _DEFAULT_BOX_THRESHOLD,
    text_threshold: float = _DEFAULT_TEXT_THRESHOLD,
) -> DetectResult:
    """Detect objects matching ``query`` in an RGB uint8 [H, W, 3] image.

    Grounding DINO expects period-separated phrases (``"red cube. green
    cube."``); a missing trailing period is appended automatically. Returns
    an empty detections list when nothing clears the thresholds — select the
    best box downstream (e.g. closest to a pointing-model pixel) and feed it
    to ``sam3.segment_box`` for a pixel-accurate mask.
    """
    import torch
    from PIL import Image

    model, processor = _get_model()

    arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ToolError(
            "grounding-dino.detect",
            f"expected RGB uint8 [H, W, 3] image, got shape {arr.shape}",
        )
    pil_image = Image.fromarray(arr, "RGB")

    # GDINO requires period-terminated phrases
    text_prompt = query
    if not text_prompt.endswith("."):
        text_prompt = text_prompt + "."

    # Defaults if zero/unset (mirrors the servicer's proto-default handling)
    box_threshold = box_threshold if box_threshold > 0 else _DEFAULT_BOX_THRESHOLD
    text_threshold = text_threshold if text_threshold > 0 else _DEFAULT_TEXT_THRESHOLD

    try:
        inputs = processor(
            images=pil_image, text=text_prompt, return_tensors="pt"
        ).to(_DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]
    except Exception as e:
        raise PerceptionFailed(f"Grounding DINO detection failed: {e}") from e

    detections: list[Detection] = []
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]

    for box, score, label in zip(boxes, scores, labels, strict=False):
        detections.append({
            "box": {
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
            },
            "label": str(label),
            "score": float(score),
        })

    logger.info("grounding-dino.detect returning %d detections.", len(detections))
    return {"detections": detections}
