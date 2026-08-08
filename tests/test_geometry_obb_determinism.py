"""Determinism regression for the sealed same-GPU target-cloud producer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEALED_CLOUD_SHA256 = (
    "b0579caf0fc8c815479161558304d58d251909f6e487269f2da18c8dd2d15926"
)


def _load_impl():
    path = ROOT / "tools/geometry/_impl.py"
    spec = importlib.util.spec_from_file_location("ors_geometry_impl_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rng_states_equal(left: tuple, right: tuple) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _portable_cloud() -> np.ndarray:
    """Return a non-degenerate cuboid cloud without touching global RNG."""
    rng = np.random.default_rng(9417)
    return rng.uniform(
        low=(-0.035, -0.025, 0.005),
        high=(0.035, 0.025, 0.075),
        size=(3392, 3),
    ).astype(np.float32)


def _input_cloud() -> np.ndarray:
    sealed_fixture = os.environ.get("ORS_OBB_SEALED_FIXTURE")
    if not sealed_fixture:
        return _portable_cloud()

    path = Path(sealed_fixture)
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == SEALED_CLOUD_SHA256
    with np.load(path) as archive:
        points = archive["positions"]
    assert points.shape == (3392, 3)
    assert points.dtype == np.float32
    return points


class SealedObbDeterminismTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.points = _input_cloud()
        assert cls.points.shape == (3392, 3)
        assert cls.points.dtype == np.float32
        cls.impl = _load_impl()

    def test_output_is_independent_of_global_rng_prestate(self) -> None:
        original = np.random.get_state()
        try:
            np.random.seed(101)
            first = self.impl.compute_obb(self.points)
            np.random.seed(202)
            second = self.impl.compute_obb(self.points)
        finally:
            np.random.set_state(original)

        self.assertEqual(_canonical(first), _canonical(second))

    def test_call_preserves_global_rng_state(self) -> None:
        original = np.random.get_state()
        try:
            np.random.seed(303)
            before = np.random.get_state()
            self.impl.compute_obb(self.points)
            after = np.random.get_state()
        finally:
            np.random.set_state(original)

        self.assertTrue(_rng_states_equal(before, after))


if __name__ == "__main__":
    unittest.main()
