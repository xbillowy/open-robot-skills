"""CPU characterization of the version dispatch; no cuRobo/CUDA imports."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_validation(**overrides):
    path = Path(__file__).parents[1] / 'tools/curobo/_curobo_impl.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'validate_joint_trajectory_robot_world')
    env = dict(Any=object, np=np, _V2_AVAILABLE=True)
    env.update(overrides)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env['validate_joint_trajectory_robot_world']


def test_v08_dispatch_does_not_access_removed_v07_types():
    calls = []
    expected = (False, 'collision_or_joint_limit', 1, {'num_waypoints': 2})
    def validate(world, q, **kwargs):
        calls.append((world, q, kwargs))
        return expected
    run = load_validation(_validate_joint_trajectory_v2=validate)
    world = SimpleNamespace(mesh=[])
    q = np.zeros((2, 7))
    assert run(world, q, robot_file='franka.yml',
               ignore_obstacle_names=['target']) == expected
    assert calls[0][0] is world
    assert calls[0][1] is q
    assert calls[0][2]['robot_collision_sphere_buffer'] == -0.01
    assert calls[0][2]['collision_activation_distance'] == 0.01
    assert calls[0][2]['ignore_obstacle_names'] == ['target']


@pytest.mark.parametrize('mask,expected', [([True, True], (True, '', None)),
    ([True, False], (False, 'collision_or_joint_limit', 1)),
    ([False, False], (False, 'collision_or_joint_limit', 0))])
def test_v08_mask_and_configuration(monkeypatch, mask, expected):
    import torch
    calls = {}
    device = SimpleNamespace(to_device=lambda q: torch.tensor(q, dtype=torch.float32))
    raw = {'kinematics': {}}
    scene = SimpleNamespace(mesh=['obstacle'])
    def load_config(**kwargs):
        calls['config'] = kwargs
        return kwargs
    class Checker:
        def __init__(self, config):
            self.self_collision_cost = object()
            def update_spheres(n, batch_size=-1, horizon=-1):
                assert (batch_size, horizon) == (2, 1)
                calls['spheres'] = n
            self.collision_constraint = SimpleNamespace(
                update_num_spheres=update_spheres,
                forward=lambda state: torch.tensor([not v for v in mask]).reshape(2,1))
        def get_kinematics(self, q):
            calls['q'] = q
            return SimpleNamespace(robot_spheres=torch.zeros(2,1,3,4))
        def setup_batch_tensors(self, batch, horizon):
            assert (batch,horizon) == (2,1)
        def get_bound(self, q):
            calls['bounds_checked'] = True
            return torch.zeros(2,1)
        def get_self_collision(self, spheres):
            calls['self_checked'] = True
            return torch.zeros(2,1)
    monkeypatch.setitem(sys.modules, 'curobo.collision_checking', SimpleNamespace(
        RobotCollisionChecker=Checker,
        RobotCollisionCheckerCfg=SimpleNamespace(load_from_config=load_config)))
    monkeypatch.setitem(sys.modules, 'curobo._src.types.robot', SimpleNamespace(
        RobotCfg=SimpleNamespace(create=lambda cfg, dev: cfg)))
    monkeypatch.setitem(sys.modules, 'curobo._src.util_file', SimpleNamespace(
        get_robot_configs_path=lambda: '/configs', join_path=lambda a,b: a+'/'+b,
        load_yaml=lambda p: {'robot_cfg': raw}))
    def convert(world, dev, ignored):
        calls['ignored'] = ignored
        assert dev is device
        return scene
    path = Path(__file__).parents[1] / 'tools/curobo/_curobo_impl.py'
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                and n.name == '_validate_joint_trajectory_v2')
    env = dict(np=np, torch=torch, DeviceCfg=lambda: device, _v2_scene_cfg_excluding=convert)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    result = env['_validate_joint_trajectory_v2'](object(), np.zeros((2,7)),
        robot_file='franka.yml', tensor_args=None, use_cuda_graph=False,
        robot_collision_sphere_buffer=-0.01, collision_activation_distance=0.02,
        ignore_obstacle_names=['target'])
    assert result[:3] == expected
    assert calls['q'].shape == (2,1,7)
    assert raw['kinematics']['collision_sphere_buffer'] == -0.01
    assert calls['config']['scene_model'] is scene
    assert calls['config']['collision_activation_distance'] == 0.02
    assert calls['ignored'] == ['target']
    assert calls['bounds_checked'] and calls['self_checked']
    assert calls['spheres'] == 3
