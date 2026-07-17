"""curobo bundle: signature/schema + converter units + GPU smoke.

cuRobo itself never imports during collection or the CPU tests — the
converters and validation guards are exercised through the tools module
without touching ``_curobo_impl``.
"""

from __future__ import annotations

import importlib.util
import sys
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gap_core.tools import ToolRegistry
from gap_core.tools._registry import _PENDING_TOOLS

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TOOLS = {
    "curobo.plan_to_grasp_poses",
    "curobo.plan_with_grasped_object",
    "curobo.plan_linear",
    "curobo.plan_directed_linear",
    "curobo.plan_grasp_motion",
    "curobo.plan_to_pose",
    "curobo.solve_ik",
    "curobo.batch_grasp_feasibility",
    "curobo.validate_joint_trajectory_robot",
    "curobo.validate_joint_trajectory_grasped",
}


@pytest.fixture(scope="module")
def curobo_module():
    name = "gap_skills.tools.curobo.tools"
    sys.modules.pop(name, None)
    _PENDING_TOOLS[:] = [e for e in _PENDING_TOOLS if e["name"] not in EXPECTED_TOOLS]
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools/curobo/tools.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool_registry(curobo_module):
    del curobo_module
    registry = ToolRegistry()
    registry.discover_pending()
    return registry


@pytest.fixture(scope="module")
def curobo_impl_module():
    name = "gap_skills.tools.curobo._curobo_impl"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools/curobo/_curobo_impl.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_FRANKA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def _pose(x, y, z, w=0.0, qx=1.0, qy=0.0, qz=0.0):
    return {
        "position": {"x": x, "y": y, "z": z},
        "rotation": {"w": w, "x": qx, "y": qy, "z": qz},
    }


def _cube_mesh(name, center, size):
    center = np.asarray(center, dtype=np.float32)
    offsets = np.array(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int32,
    )
    return {
        "name": name,
        "vertices": center + offsets * float(size),
        "faces": faces,
        "pose": None,
    }


def curobo_module_mesh(mesh):
    return SimpleNamespace(**mesh)


def test_all_tools_registered(tool_registry):
    for name in EXPECTED_TOOLS:
        assert name in tool_registry
        assert "planning" in tool_registry.get(name).tags


def test_plan_to_grasp_poses_schema(tool_registry):
    schema = tool_registry.get("curobo.plan_to_grasp_poses").schema
    required = {"world_config", "start_joint_position", "grasp_poses"}
    assert required <= set(schema.inputs)
    for name in required:
        assert schema.inputs[name].required
    # Proto/servicer defaults carried into the signature.
    assert schema.inputs["robot_file"].default == "franka.yml"
    assert schema.inputs["max_attempts"].default == 8
    assert schema.inputs["num_ik_seeds"].default == 128
    assert schema.inputs["robot_collision_sphere_buffer"].default == pytest.approx(-0.01)
    assert schema.inputs["collision_activation_distance"].default == pytest.approx(0.001)
    assert schema.inputs["grasp_pose_is_fingertip"].default is True
    assert schema.inputs["use_world_collision"].default is True
    assert set(schema.outputs) == {"success", "trajectory", "goalset_index"}


def test_validate_schema(tool_registry):
    schema = tool_registry.get("curobo.validate_joint_trajectory_grasped").schema
    assert {"world_config", "trajectory", "object_name"} <= set(schema.inputs)
    assert schema.inputs["link_name"].default == "attached_object"
    assert schema.inputs["surface_sphere_radius"].default == pytest.approx(0.001)
    assert set(schema.outputs) == {
        "success",
        "failure_reason",
        "first_collision_waypoint",
        "collision_status_detail",
    }


def test_trajectory_converters_roundtrip(curobo_module):
    mod = curobo_module
    arr = np.arange(21, dtype=np.float64).reshape(3, 7)
    traj = mod._traj_out(arr)
    assert len(traj["waypoints"]) == 3
    back = mod._traj_in(traj)
    np.testing.assert_allclose(back, arr)
    assert mod._traj_out(None) is None
    assert mod._traj_in({"waypoints": []}).shape == (0, 0)


def test_world_converter_builds_mesh_namespace(curobo_module):
    mod = curobo_module
    wc = {
        "meshes": [
            {
                "name": "scene",
                "vertices": np.zeros((3, 3), dtype=np.float32),
                "faces": np.array([[0, 1, 2]], dtype=np.int32),
                "pose": _pose(1.0, 2.0, 3.0, w=1.0, qx=0.0),
            }
        ]
    }
    ns = mod._world_ns(wc)
    assert len(ns.mesh) == 1
    mesh = ns.mesh[0]
    assert mesh.name == "scene"
    assert mesh.pose == [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    assert len(mesh.vertices) == 3 and len(mesh.faces) == 1
    # Identity pose default when the mesh has no pose.
    ns2 = mod._world_ns({"meshes": [{"name": "m", "vertices": [], "faces": [], "pose": None}]})
    assert ns2.mesh[0].pose == [0, 0, 0, 1, 0, 0, 0]


def test_curobo_evidence_is_algorithmic_without_fake_model_fields(curobo_module):
    fields = typing.get_type_hints(curobo_module.AlgorithmServiceEvidence)
    assert fields["kind"]
    assert "weights_sha256" not in fields
    assert "resolved_revision" not in fields


def test_curobo_paper_capability_reports_pinned_metadata(curobo_module, monkeypatch):
    monkeypatch.setattr(
        curobo_module,
        "_impl",
        lambda: SimpleNamespace(
            paper_v08_contracts=lambda: {
                "batch_ik": False,
                "robot_collision_validation": False,
                "held_object_collision_validation": False,
                "trajectory_validation": False,
            }
        ),
    )
    capability = curobo_module.paper_service_capability()
    assert capability["pinned_version"] == "0.8.0"
    assert capability["uv_lock_sha256"] == (
        "sha256:eff980495ea60e5db0046e6de3cf49870da88690a2358d31ef3f6b2a261a24c7"
    )
    assert capability["source_commit"] == "4ea77366ca48ee453e7df139e39fa6532af49f3b"


def test_curobo_paper_capability_requires_all_runtime_contracts(curobo_module, monkeypatch):
    fake = SimpleNamespace(
        paper_v08_contracts=lambda: {
            "batch_ik": True,
            "robot_collision_validation": True,
            "held_object_collision_validation": True,
            "trajectory_validation": True,
        }
    )
    monkeypatch.setattr(curobo_module, "_impl", lambda: fake)

    capability = curobo_module.paper_service_capability()

    assert capability["available"] is True
    assert capability["missing_required_paths"] == ()


def test_curobo_paper_capability_preserves_failed_runtime_contract(curobo_module, monkeypatch):
    fake = SimpleNamespace(
        paper_v08_contracts=lambda: {
            "batch_ik": True,
            "robot_collision_validation": True,
            "held_object_collision_validation": False,
            "trajectory_validation": True,
        }
    )
    monkeypatch.setattr(curobo_module, "_impl", lambda: fake)

    capability = curobo_module.paper_service_capability()

    assert capability["available"] is False
    assert capability["missing_required_paths"] == ("held_object_collision_validation",)


def test_curobo_paper_capability_sanitizes_probe_exception(curobo_module, monkeypatch, caplog):
    def fail_probe():
        raise RuntimeError("secret-token-must-not-leak")

    monkeypatch.setattr(
        curobo_module,
        "_impl",
        lambda: SimpleNamespace(paper_v08_contracts=fail_probe),
    )

    capability = curobo_module.paper_service_capability()

    assert capability["available"] is False
    assert capability["probe_error_type"] == "RuntimeError"
    assert "secret-token-must-not-leak" not in repr(capability)
    assert "secret-token-must-not-leak" not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.parametrize("invalid", ["main", "a" * 39, "a" * 41, "f" * 63])
def test_curobo_rejects_non_object_source_commits(curobo_module, invalid):
    with pytest.raises(ValueError, match="Git object ID"):
        curobo_module._validate_git_object_id(invalid)


@pytest.mark.parametrize("invalid", ["f" * 64, "sha256:abc", "SHA256:" + "f" * 64])
def test_curobo_rejects_noncanonical_digests(curobo_module, invalid):
    with pytest.raises(ValueError, match="canonical SHA256"):
        curobo_module._validate_digest(invalid)


def test_empty_trajectory_rejected_without_curobo(tool_registry):
    """Argument guards fire before the lazy cuRobo import."""
    from gap_core.errors import ToolError

    with pytest.raises(ToolError):
        tool_registry.invoke(
            "curobo.validate_joint_trajectory_robot",
            world_config={"meshes": []},
            trajectory={"waypoints": []},
        )
    with pytest.raises(ToolError):
        tool_registry.invoke(
            "curobo.validate_joint_trajectory_grasped",
            world_config={"meshes": []},
            trajectory={"waypoints": [{"positions": _FRANKA_HOME}]},
            object_name="",
        )
    with pytest.raises(ToolError):
        tool_registry.invoke(
            "curobo.batch_grasp_feasibility",
            world_config={"meshes": []},
            start_state={"positions": _FRANKA_HOME},
            grasp_poses=[_pose(0.4, 0.0, 0.2)],
        )


def test_v08_robot_trajectory_validation_reports_first_invalid_waypoint(
    curobo_impl_module, monkeypatch
):
    torch = pytest.importorskip("torch")

    class FakeChecker:
        device_cfg = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
        kinematics = SimpleNamespace(get_dof=lambda: 7)

        def validate_trajectory(self, q):
            assert tuple(q.shape) == (1, 3, 7)
            return torch.tensor([[True, False, False]])

    monkeypatch.setattr(
        curobo_impl_module,
        "_make_v2_collision_checker",
        lambda *args, **kwargs: (FakeChecker(), SimpleNamespace()),
        raising=False,
    )

    ok, reason, index, meta = curobo_impl_module.validate_joint_trajectory_robot_world(
        SimpleNamespace(mesh=[SimpleNamespace(name="scene")]),
        np.zeros((3, 7)),
    )

    assert ok is False
    assert reason == "collision_or_joint_limit"
    assert index == 1
    assert meta["num_waypoints"] == 3


def test_v08_held_object_validation_always_detaches(curobo_impl_module, monkeypatch):
    torch = pytest.importorskip("torch")
    events = []

    class FakeChecker:
        device_cfg = SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.float32,
            to_device=lambda value: torch.as_tensor(value, dtype=torch.float32),
        )
        kinematics = SimpleNamespace(
            get_dof=lambda: 7, joint_names=[f"joint_{i}" for i in range(7)]
        )
        scene_model = object()

        def validate_trajectory(self, q):
            events.append("validate")
            raise RuntimeError("probe failure")

    obstacle = SimpleNamespace(name="held", pose=[0, 0, 0, 1, 0, 0, 0])
    scene_cfg = SimpleNamespace(get_obstacle=lambda name: obstacle if name == "held" else None)
    monkeypatch.setattr(
        curobo_impl_module,
        "_make_v2_collision_checker",
        lambda *args, **kwargs: (FakeChecker(), scene_cfg),
        raising=False,
    )
    monkeypatch.setattr(
        curobo_impl_module, "_world_to_v2_scene_cfg", lambda *args, **kwargs: scene_cfg
    )

    class FakeManager:
        def __init__(self, *args, **kwargs):
            pass

        def attach(self, *args, **kwargs):
            events.append("attach")

        def detach(self, *args, **kwargs):
            events.append("detach")

    monkeypatch.setattr(curobo_impl_module, "_V2AttachmentManager", FakeManager, raising=False)

    with pytest.raises(RuntimeError, match="probe failure"):
        curobo_impl_module.validate_joint_trajectory_grasped_object(
            SimpleNamespace(mesh=[SimpleNamespace(name="held")]),
            np.zeros((2, 7)),
            "held",
        )

    assert events == ["attach", "validate", "detach"]


def test_v08_plan_with_grasped_object_plans_without_target_then_validates_attachment(
    curobo_impl_module, monkeypatch
):
    held = SimpleNamespace(name="held")
    blocker = SimpleNamespace(name="blocker")
    world = SimpleNamespace(mesh=[held, blocker])
    trajectory = np.stack([_FRANKA_HOME, _FRANKA_HOME])
    calls = []

    def fake_plan(target_position, target_quat, start, **kwargs):
        calls.append(("plan", kwargs["world_config"], target_position, target_quat, start))
        return True, trajectory

    def fake_validate(bound_world, waypoints, object_name, **kwargs):
        calls.append(("validate", bound_world, waypoints, object_name, kwargs))
        return True, "", None, {}

    monkeypatch.setattr(curobo_impl_module, "_V2_AVAILABLE", True)
    monkeypatch.setattr(curobo_impl_module, "plan_to_pose", fake_plan)
    monkeypatch.setattr(
        curobo_impl_module,
        "validate_joint_trajectory_grasped_object",
        fake_validate,
    )

    success, actual = curobo_impl_module.plan_with_grasped_object(
        world,
        _FRANKA_HOME,
        (np.array([0.45, 0.0, 0.35]), np.array([0.0, 1.0, 0.0, 0.0])),
        "held",
        debug_out_dir=None,
    )

    assert success is True
    np.testing.assert_array_equal(actual, trajectory)
    assert [mesh.name for mesh in calls[0][1].mesh] == ["blocker"]
    assert calls[1][1] is world
    assert calls[1][3] == "held"


def test_v08_plan_with_grasped_object_rejects_unknown_attachment_link(
    curobo_impl_module, monkeypatch
):
    monkeypatch.setattr(curobo_impl_module, "_V2_AVAILABLE", True)
    monkeypatch.setattr(
        curobo_impl_module,
        "plan_to_pose",
        lambda *_args, **_kwargs: pytest.fail("unsupported link must fail before planning"),
    )

    with pytest.raises(ValueError, match="attached_object"):
        curobo_impl_module.plan_with_grasped_object(
            SimpleNamespace(mesh=[SimpleNamespace(name="held")]),
            _FRANKA_HOME,
            (np.array([0.45, 0.0, 0.35]), np.array([0.0, 1.0, 0.0, 0.0])),
            "held",
            link_name="custom_attachment",
            debug_out_dir=None,
        )


def test_v08_plan_with_grasped_object_stops_before_validation_when_plan_fails(
    curobo_impl_module, monkeypatch
):
    monkeypatch.setattr(curobo_impl_module, "_V2_AVAILABLE", True)
    monkeypatch.setattr(curobo_impl_module, "plan_to_pose", lambda *_a, **_k: (False, None))
    monkeypatch.setattr(
        curobo_impl_module,
        "validate_joint_trajectory_grasped_object",
        lambda *_a, **_k: pytest.fail("failed plan must not be validated"),
    )

    assert curobo_impl_module.plan_with_grasped_object(
        SimpleNamespace(mesh=[SimpleNamespace(name="held")]),
        _FRANKA_HOME,
        (np.array([0.45, 0.0, 0.35]), np.array([0.0, 1.0, 0.0, 0.0])),
        "held",
        debug_out_dir=None,
    ) == (False, None)
    assert curobo_impl_module._last_planning_debug["status"] == "planning_failed"


def test_v08_plan_with_grasped_object_rejects_failed_attachment_validation_without_world(
    curobo_impl_module, monkeypatch
):
    held = SimpleNamespace(name="held")
    blocker = SimpleNamespace(name="blocker")
    trajectory = np.stack([_FRANKA_HOME, _FRANKA_HOME])
    calls = []
    monkeypatch.setattr(curobo_impl_module, "_V2_AVAILABLE", True)

    def fake_plan(*_args, **kwargs):
        calls.append(("plan", kwargs["world_config"]))
        return True, trajectory

    def fake_validate(world, _trajectory, _object_name, **kwargs):
        calls.append(("validate", world, kwargs))
        return False, "held_collision", 1, {"probe": "blocked"}

    monkeypatch.setattr(curobo_impl_module, "plan_to_pose", fake_plan)
    monkeypatch.setattr(
        curobo_impl_module,
        "validate_joint_trajectory_grasped_object",
        fake_validate,
    )

    assert curobo_impl_module.plan_with_grasped_object(
        SimpleNamespace(mesh=[held, blocker]),
        _FRANKA_HOME,
        (np.array([0.45, 0.0, 0.35]), np.array([0.0, 1.0, 0.0, 0.0])),
        "held",
        use_world_collision=False,
        debug_out_dir=None,
    ) == (False, None)
    assert calls[0] == ("plan", None)
    assert [mesh.name for mesh in calls[1][1].mesh] == ["held"]
    assert calls[1][2]["link_name"] == "attached_object"
    assert curobo_impl_module._last_planning_debug == {
        "function": "plan_with_grasped_object",
        "status": "attachment_validation_failed",
        "backend": "curobo_v0.8",
        "failure_reason": "held_collision",
        "first_collision_waypoint": 1,
        "validation": {"probe": "blocked"},
    }


def test_v08_batch_grasp_feasibility_preserves_candidate_alignment(curobo_impl_module, monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    def fake_batch_ik(world, poses, **kwargs):
        calls.append(poses)
        if len(calls) == 1:
            return np.array([True, False]), np.array([[[1] * 7], [[2] * 7]], dtype=float)
        return np.array([True, True]), np.array([[[3] * 7], [[4] * 7]], dtype=float)

    class FakeChecker:
        device_cfg = SimpleNamespace(device=torch.device("cpu"))

        def validate_trajectory(self, q):
            return torch.ones(q.shape[:2], dtype=torch.bool)

    monkeypatch.setattr(curobo_impl_module, "_batch_v2_ik", fake_batch_ik, raising=False)
    monkeypatch.setattr(
        curobo_impl_module,
        "_make_v2_collision_checker",
        lambda *args, **kwargs: (FakeChecker(), SimpleNamespace()),
    )
    poses = [
        (np.array([0.4, 0.0, 0.3]), np.array([1.0, 0.0, 0.0, 0.0])),
        (np.array([0.5, 0.0, 0.3]), np.array([1.0, 0.0, 0.0, 0.0])),
    ]

    feasible, grasp_ok, approach_ok, corridor = curobo_impl_module.batch_grasp_feasibility(
        SimpleNamespace(mesh=[SimpleNamespace(name="scene")]),
        np.zeros(7),
        poses,
        grasp_pose_is_fingertip=False,
        num_corridor_samples=3,
    )

    assert feasible == [True, False]
    assert grasp_ok == [True, False]
    assert approach_ok == [True, True]
    assert corridor == [0.0, 1.0]


@pytest.mark.gpu
def test_v08_paper_contracts_check_concrete_pinned_apis(
    curobo_impl_module, curobo_module, monkeypatch
):
    contracts = curobo_impl_module.paper_v08_contracts()

    assert contracts == {
        "batch_ik": True,
        "robot_collision_validation": True,
        "held_object_collision_validation": True,
        "trajectory_validation": True,
    }
    monkeypatch.setattr(curobo_module, "_impl", lambda: curobo_impl_module)
    capability = curobo_module.paper_service_capability()
    assert capability["available"] is True
    assert capability["missing_required_paths"] == ()


@pytest.mark.gpu
def test_plan_to_pose_gpu_smoke(tool_registry, curobo_module, curobo_impl_module, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    monkeypatch.setattr(curobo_module, "_impl", lambda: curobo_impl_module)

    out = tool_registry.invoke(
        "curobo.plan_to_pose",
        target_pose=_pose(0.45, 0.0, 0.35),  # reachable, gripper down
        start_joint_position={"positions": _FRANKA_HOME},
    )
    assert isinstance(out["success"], bool)
    if out["success"]:
        traj = out["trajectory"]
        assert traj is not None and len(traj["waypoints"]) > 1
        assert len(traj["waypoints"][0]["positions"]) >= 7


@pytest.mark.gpu
def test_plan_to_grasp_poses_gpu_smoke(
    tool_registry, curobo_module, curobo_impl_module, monkeypatch
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    monkeypatch.setattr(curobo_module, "_impl", lambda: curobo_impl_module)

    # Tiny floor slab as the world; grasp straight down above it.
    verts = np.array(
        [[-0.5, -0.5, -0.02], [0.5, -0.5, -0.02], [0.5, 0.5, -0.02], [-0.5, 0.5, -0.02]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    world = {"meshes": [{"name": "floor", "vertices": verts, "faces": faces, "pose": None}]}

    out = tool_registry.invoke(
        "curobo.plan_to_grasp_poses",
        world_config=world,
        start_joint_position={"positions": _FRANKA_HOME},
        grasp_poses=[_pose(0.45, 0.0, 0.25), _pose(0.45, 0.05, 0.25)],
    )
    assert isinstance(out["success"], bool)
    if out["success"]:
        assert out["goalset_index"] in (0, 1)
        assert out["trajectory"] is not None


@pytest.mark.gpu
def test_plan_with_grasped_object_gpu_smoke(
    tool_registry, curobo_module, curobo_impl_module, monkeypatch
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    monkeypatch.setattr(curobo_module, "_impl", lambda: curobo_impl_module)

    out = tool_registry.invoke(
        "curobo.plan_with_grasped_object",
        world_config={"meshes": [_cube_mesh("held", [0.55, 0.0, 0.45], 0.03)]},
        start_joint_position={"positions": _FRANKA_HOME},
        target_pose=_pose(0.45, 0.0, 0.35),
        object_name="held",
        remove_obstacles_from_world=True,
        debug_out_dir=None,
    )

    assert out["success"] is True
    assert out["trajectory"] is not None
    assert len(out["trajectory"]["waypoints"]) > 1


@pytest.mark.gpu
def test_paper_v08_robot_and_held_collision_contracts(curobo_impl_module):
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    far_world = SimpleNamespace(
        mesh=[curobo_module_mesh(_cube_mesh("far", [10.0, 10.0, 10.0], 0.1))]
    )
    free = curobo_impl_module.validate_joint_trajectory_robot_world(
        far_world, np.stack([_FRANKA_HOME, _FRANKA_HOME])
    )
    assert free[0] is True

    checker, _ = curobo_impl_module._make_v2_collision_checker(
        far_world,
        robot_file="franka.yml",
        robot_collision_sphere_buffer=-0.01,
        collision_activation_distance=0.01,
    )
    q = torch.as_tensor(_FRANKA_HOME, device=checker.device_cfg.device, dtype=torch.float32)
    state = checker.get_kinematics(q.view(1, 1, -1))
    spheres = state.robot_spheres.reshape(-1, 4)
    center = spheres[spheres[:, 3] > 0][0, :3].detach().cpu().numpy()
    hit_world = SimpleNamespace(mesh=[curobo_module_mesh(_cube_mesh("blocker", center, 0.2))])
    hit = curobo_impl_module.validate_joint_trajectory_robot_world(
        hit_world, np.stack([_FRANKA_HOME, _FRANKA_HOME])
    )
    assert hit[:3] == (False, "collision_or_joint_limit", 0)

    out_of_bounds = _FRANKA_HOME.copy()
    out_of_bounds[0] = 100.0
    bound_hit = curobo_impl_module.validate_joint_trajectory_robot_world(
        far_world, np.stack([out_of_bounds, _FRANKA_HOME])
    )
    assert bound_hit[3]["failure_components"][0] == ["joint_bound"]

    self_collision = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    self_hit = curobo_impl_module.validate_joint_trajectory_robot_world(
        far_world, np.stack([self_collision, _FRANKA_HOME])
    )
    assert self_hit[3]["failure_components"][0] == ["self_collision"]

    held = curobo_module_mesh(_cube_mesh("held", [0.55, 0.0, 0.45], 0.03))
    held_free = curobo_impl_module.validate_joint_trajectory_grasped_object(
        SimpleNamespace(mesh=[held]),
        np.stack([_FRANKA_HOME, _FRANKA_HOME]),
        "held",
    )
    assert held_free[0] is True
    assert held_free[3]["attached_sphere_capacity"] == 32
    assert 0 < held_free[3]["attached_sphere_count"] <= 32
    blocker = curobo_module_mesh(_cube_mesh("held_blocker", [0.55, 0.0, 0.45], 0.08))
    robot_only = curobo_impl_module.validate_joint_trajectory_robot_world(
        SimpleNamespace(mesh=[blocker, far_world.mesh[0]]),
        np.stack([_FRANKA_HOME, _FRANKA_HOME]),
    )
    assert robot_only[0] is True
    held_hit = curobo_impl_module.validate_joint_trajectory_grasped_object(
        SimpleNamespace(mesh=[held, blocker, far_world.mesh[0]]),
        np.stack([_FRANKA_HOME, _FRANKA_HOME]),
        "held",
    )
    assert held_hit[:3] == (False, "collision_or_joint_limit", 0)


@pytest.mark.gpu
def test_paper_v08_batch_ik_contract(curobo_impl_module):
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    world = SimpleNamespace(mesh=[curobo_module_mesh(_cube_mesh("far", [10.0, 10.0, 10.0], 0.1))])
    poses = [
        (np.array([0.45, 0.0, 0.35]), np.array([0.0, 1.0, 0.0, 0.0])),
        (np.array([10.0, 10.0, 10.0]), np.array([0.0, 1.0, 0.0, 0.0])),
    ]

    feasible, grasp_ok, approach_ok, corridor = curobo_impl_module.batch_grasp_feasibility(
        world,
        _FRANKA_HOME,
        poses,
        grasp_pose_is_fingertip=False,
        num_corridor_samples=3,
        num_ik_seeds=16,
    )

    assert grasp_ok == [True, False]
    assert approach_ok == [True, False]
    assert feasible == [True, False]
    assert corridor == [0.0, 1.0]
