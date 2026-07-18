"""Contract tests for the paper-replication manipulation skill cells."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml
from gap.skills import load_skills

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CONTRACTS = {
    "perceiving-objects": {
        "required_inputs": {
            "query": "str",
            "semantic_role": "str",
            "preset_json": "str",
        },
        "produces_outputs": {
            "target_obb": "OrientedBoundingBox",
            "target_mask": "Mask",
            "target_observation": "Observation",
            "target_lineage_json": "str",
            "destination_obb": "OrientedBoundingBox",
            "destination_lineage_json": "str",
        },
    },
    "grasping-with-planner": {
        "required_inputs": {
            "target_obb": "OrientedBoundingBox",
            "target_mask": "Mask",
            "target_observation": "Observation",
            "target_lineage_json": "str",
            "target_name": "str",
            "preset_json": "str",
        },
        "produces_outputs": {
            "held_grasp_json": "str",
            "world_config": "WorldConfig",
        },
    },
    "transporting-objects": {
        "required_inputs": {
            "held_grasp_json": "str",
            "target_lineage_json": "str",
            "destination_obb": "OrientedBoundingBox",
            "destination_lineage_json": "str",
            "world_config": "WorldConfig",
            "target_name": "str",
            "preset_json": "str",
        },
        "produces_outputs": {"terminal_result_json": "str"},
    },
}

EXPECTED_EXITS = {
    "perceiving-objects": {"done", "rejected", "error"},
    "grasping-with-planner": {"done", "rejected", "error"},
    "transporting-objects": {"done", "rejected", "error"},
}


def _frontmatter(skill: str) -> dict:
    text = (ROOT / "skills" / skill / "SKILL.md").read_text()
    _, raw, _ = text.split("---", 2)
    return yaml.safe_load(raw)


def _example(skill: str, name: str = "canonical_subgraph.json") -> dict:
    path = ROOT / "skills" / skill / "examples" / name
    return json.loads(path.read_text())


def _refs(value: object) -> list[str]:
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            return [value["$ref"]]
        return [ref for item in value.values() for ref in _refs(item)]
    if isinstance(value, list):
        return [ref for item in value for ref in _refs(item)]
    return []


def test_frontmatter_declares_exact_paper_cell_contracts() -> None:
    for skill, expected in EXPECTED_CONTRACTS.items():
        gap = _frontmatter(skill)["gap"]
        assert gap["required_inputs"] == expected["required_inputs"]
        assert gap["produces_outputs"] == expected["produces_outputs"]
        assert set(gap["exit_conditions"]) == EXPECTED_EXITS[skill]


def test_canonical_examples_have_closed_dynamic_refs() -> None:
    examples = [(skill, _example(skill)) for skill in EXPECTED_CONTRACTS] + [
        (
            "perceiving-objects",
            _example("perceiving-objects", "canonical_destination_subgraph.json"),
        )
    ]
    for skill, example in examples:
        assert example["inputs"] == EXPECTED_CONTRACTS[skill]["required_inputs"]
        declared_inputs = set(example["inputs"])
        node_names = set(example["nodes"])
        predecessors: dict[str, set[str]] = {name: set() for name in example["nodes"]}
        for source, destination in example["edges"]:
            if source in predecessors and destination in predecessors:
                predecessors[destination].add(source)
        for source, conditional in example.get("conditional_edges", {}).items():
            for destination in conditional["mapping"].values():
                if source in predecessors and destination in predecessors:
                    predecessors[destination].add(source)

        changed = True
        while changed:
            changed = False
            for node, direct in predecessors.items():
                expanded = direct | {
                    ancestor for parent in direct for ancestor in predecessors[parent]
                }
                if expanded != direct:
                    predecessors[node] = expanded
                    changed = True

        for node_name, node in example["nodes"].items():
            for ref in _refs(node.get("inputs", {})):
                producer, _, _ = ref.partition(".")
                if producer == "in":
                    assert ref.split(".", 1)[1] in declared_inputs
                else:
                    assert producer in predecessors[node_name]
        for ref in _refs(example["outputs"]):
            assert ref.split(".", 1)[0] in node_names

    # The four paper cells form a closed cross-subgraph DAG. Task constants
    # are runner inputs; every other grasp/transport input has one producer.
    external = {"query", "semantic_role", "preset_json", "target_name"}
    producers: dict[str, str] = {}
    for role in ("target", "destination"):
        for output in EXPECTED_CONTRACTS["perceiving-objects"]["produces_outputs"]:
            if output.startswith(f"{role}_"):
                assert output not in producers
                producers[output] = f"{role}_perception"
    for output in EXPECTED_CONTRACTS["grasping-with-planner"]["produces_outputs"]:
        assert output not in producers
        producers[output] = "grasp"

    for skill in ("grasping-with-planner", "transporting-objects"):
        for name in EXPECTED_CONTRACTS[skill]["required_inputs"]:
            assert name in external or name in producers


def test_canonical_examples_use_only_admitted_scripts_and_role_outputs() -> None:
    perception = _example("perceiving-objects")
    assert [node["script"] for node in perception["nodes"].values() if "script" in node] == [
        "scripts/perceive_disambiguate_segment.py"
    ]
    assert perception["outputs"] == {
        "target_obb": {"$ref": "perceive.target_obb"},
        "target_mask": {"$ref": "perceive.mask"},
        "target_observation": {"$ref": "perceive.observation"},
        "target_lineage_json": {"$ref": "perceive.lineage_json"},
    }
    destination = _example("perceiving-objects", "canonical_destination_subgraph.json")
    assert destination["inputs"] == EXPECTED_CONTRACTS["perceiving-objects"]["required_inputs"]
    assert destination["outputs"] == {
        "destination_obb": {"$ref": "perceive.destination_obb"},
        "destination_lineage_json": {"$ref": "perceive.lineage_json"},
    }

    grasp = _example("grasping-with-planner")
    assert [node["script"] for node in grasp["nodes"].values() if "script" in node] == [
        "scripts/build_world.py",
        "scripts/plan_validate_grasp.py",
        "scripts/execute_verify_grasp.py",
    ]
    assert grasp["outputs"] == {
        "held_grasp_json": {"$ref": "execute.held_grasp_json"},
        "world_config": {"$ref": "build_world.config"},
    }

    transport = _example("transporting-objects")
    assert [node["script"] for node in transport["nodes"].values() if "script" in node] == [
        "scripts/plan_validate_transport.py"
    ]
    assert transport["outputs"] == {
        "terminal_result_json": {"$ref": "transport.terminal_result_json"}
    }


def test_examples_use_registered_scripts_and_exact_exits() -> None:
    registry = load_skills(ROOT)
    examples = [
        ("perceiving-objects", _example("perceiving-objects")),
        (
            "perceiving-objects",
            _example("perceiving-objects", "canonical_destination_subgraph.json"),
        ),
        ("grasping-with-planner", _example("grasping-with-planner")),
        ("transporting-objects", _example("transporting-objects")),
    ]
    for skill, example in examples:
        registered = {
            script_path.path.relative_to(ROOT / "skills" / skill).as_posix()
            for script_path in registry.get(skill).canonical_scripts.values()
        }
        used = {node["script"] for node in example["nodes"].values() if "script" in node}
        assert used <= registered
        assert example["exit"] == {
            "router_field": None,
            "success_values": ["done", "rejected"],
        }
        assert example["on_error"] == "error"


def test_paper_script_literal_tool_calls_are_allowed() -> None:
    """Cover direct ``ctx.tool(<literal>)`` calls in the admitted scripts.

    The paper scripts use direct literal tool names for their runtime calls;
    this deliberately small extractor does not attempt data-flow analysis for
    dynamically computed tool names.
    """
    examples = {
        "perceiving-objects": [_example("perceiving-objects")],
        "grasping-with-planner": [_example("grasping-with-planner")],
        "transporting-objects": [_example("transporting-objects")],
    }
    violations: dict[str, list[str]] = {}
    for skill, skill_examples in examples.items():
        allowed = set(_frontmatter(skill)["gap"]["allowed_tools"])
        called: set[str] = set()
        for example in skill_examples:
            for node in example["nodes"].values():
                script = node.get("script")
                if script is None:
                    continue
                tree = ast.parse((ROOT / "skills" / skill / script).read_text())
                for call in (item for item in ast.walk(tree) if isinstance(item, ast.Call)):
                    if (
                        isinstance(call.func, ast.Attribute)
                        and call.func.attr == "tool"
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "ctx"
                        and call.args
                        and isinstance(call.args[0], ast.Constant)
                        and isinstance(call.args[0].value, str)
                    ):
                        called.add(call.args[0].value)
        if undeclared := sorted(called - allowed):
            violations[skill] = undeclared
    assert not violations, f"undeclared literal tools: {violations}"
