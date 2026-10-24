"""Raise the sub-graph's failure exit with the main script's reason (system-provided, PS skills)."""

from typing import TypedDict

from gap import NodeContext
from gap_core.errors import VerificationFailed


class Output(TypedDict):
    raised: bool


def run(ctx: NodeContext, reason: str = "") -> Output:
    raise VerificationFailed(f"skill reported failure: {reason}")
