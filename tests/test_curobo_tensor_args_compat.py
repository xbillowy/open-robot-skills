"""Compatibility contract for robot/world trajectory validation tensor args."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.curobo import _curobo_impl as impl

_PRESERVED_FIRST_WAYPOINT = np.array(
    [
        1.3860141861838168e-12,
        -0.16103738899999961,
        3.5801507986500924e-12,
        -2.44459747,
        1.7538935830794371e-12,
        2.2267522,
        0.7853981633985379,
    ],
    dtype=np.float64,
)


class _ReachedMotionGenCache(RuntimeError):
    pass


class _RecordingCache:
    def __init__(self) -> None:
        self.tensor_args = None

    def get(self, **kwargs):
        self.tensor_args = kwargs["tensor_args"]
        raise _ReachedMotionGenCache


def _invoke_until_cache(monkeypatch: pytest.MonkeyPatch, cache: _RecordingCache) -> None:
    monkeypatch.setattr(impl, "_motion_gen_cache", cache)
    with pytest.raises(_ReachedMotionGenCache):
        impl.validate_joint_trajectory_robot_world(
            SimpleNamespace(mesh=[]),
            _PRESERVED_FIRST_WAYPOINT.reshape(1, 7),
        )


def test_v08_validation_constructs_device_cfg_before_motion_gen_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preserved v0.8 request must not reference the absent v0.7 type."""

    class FakeDeviceCfg:
        pass

    cache = _RecordingCache()
    monkeypatch.setattr(impl, "_V1_AVAILABLE", False)
    monkeypatch.setattr(impl, "_V2_AVAILABLE", True)
    monkeypatch.setattr(impl, "DeviceCfg", FakeDeviceCfg, raising=False)
    _invoke_until_cache(monkeypatch, cache)
    assert isinstance(cache.tensor_args, FakeDeviceCfg)


def test_v07_validation_preserves_tensor_device_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTensorDeviceType:
        pass

    cache = _RecordingCache()
    monkeypatch.setattr(impl, "_V1_AVAILABLE", True)
    monkeypatch.setattr(impl, "_V2_AVAILABLE", False)
    monkeypatch.setattr(impl, "TensorDeviceType", FakeTensorDeviceType, raising=False)
    _invoke_until_cache(monkeypatch, cache)
    assert isinstance(cache.tensor_args, FakeTensorDeviceType)


def test_validation_fails_closed_when_no_supported_curobo_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _RecordingCache()
    monkeypatch.setattr(impl, "_V1_AVAILABLE", False)
    monkeypatch.setattr(impl, "_V2_AVAILABLE", False)
    monkeypatch.setattr(impl, "_motion_gen_cache", cache)
    with pytest.raises(RuntimeError, match="requires curobo v0.7 or v0.8"):
        impl.validate_joint_trajectory_robot_world(
            SimpleNamespace(mesh=[]),
            _PRESERVED_FIRST_WAYPOINT.reshape(1, 7),
        )
    assert cache.tensor_args is None


def test_v08_request_reaches_explicit_unported_motion_gen_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.8 must fail with the existing support gate, never an undefined symbol."""

    class FakeDeviceCfg:
        pass

    monkeypatch.setattr(impl, "_V1_AVAILABLE", False)
    monkeypatch.setattr(impl, "_V2_AVAILABLE", True)
    monkeypatch.setattr(impl, "DeviceCfg", FakeDeviceCfg, raising=False)
    monkeypatch.setattr(impl, "_motion_gen_cache", impl._MotionGenCache())
    with pytest.raises(RuntimeError, match="requires curobo v0.7 .* not.*available"):
        impl.validate_joint_trajectory_robot_world(
            SimpleNamespace(mesh=[]),
            _PRESERVED_FIRST_WAYPOINT.reshape(1, 7),
        )


def test_preserved_v08_request_reaches_explicit_motion_gen_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay the exact sealed Task03 request when its artifact is available."""

    call_root_text = os.environ.get("ORS_PRESERVED_CUROBO_CALL_ROOT")
    if call_root_text is None:
        pytest.skip("set ORS_PRESERVED_CUROBO_CALL_ROOT for the real-artifact gate")
    call_root = Path(call_root_text)
    call_meta = json.loads(
        (call_root / "003_validate_joint_trajectory_robot/call.meta.json").read_text()
    )
    failure = json.loads(
        (call_root / "003_validate_joint_trajectory_robot/failure.json").read_text()
    )
    world_response = json.loads(
        (call_root / "001_build_world_config/response.json").read_text()
    )
    trajectory_response = json.loads(
        (call_root / "002_plan_to_pose/response.json").read_text()
    )
    joint_waypoints = np.asarray(
        [
            waypoint["positions"]
            for waypoint in trajectory_response["trajectory"]["waypoints"]
        ],
        dtype=np.float64,
    )

    assert call_meta["request_fingerprint"] == (
        "4f5f5144f183f61a6148ded96610c2559f8179fbd6f4d2afcb881fa3cb8628d8"
    )
    assert failure["component"] == "curobo.validate_joint_trajectory_robot"
    assert "TensorDeviceType" in failure["message"]
    assert joint_waypoints.shape == (61, 7)

    class FakeDeviceCfg:
        pass

    monkeypatch.setattr(impl, "_V1_AVAILABLE", False)
    monkeypatch.setattr(impl, "_V2_AVAILABLE", True)
    monkeypatch.setattr(impl, "DeviceCfg", FakeDeviceCfg, raising=False)
    monkeypatch.setattr(impl, "_motion_gen_cache", impl._MotionGenCache())
    with pytest.raises(RuntimeError, match="requires curobo v0.7 .* not.*available"):
        impl.validate_joint_trajectory_robot_world(
            SimpleNamespace(mesh=world_response["config"]["meshes"]),
            joint_waypoints,
        )
