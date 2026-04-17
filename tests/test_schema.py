from __future__ import annotations

from whole_body_tracking.robots.g1.schema import (
    ANCHOR_BODY_NAME,
    G1_BODY_NAMES,
    G1_JOINT_NAMES,
)


def test_g1_schema_is_consistent() -> None:
    assert len(G1_JOINT_NAMES) == 29
    assert len(G1_BODY_NAMES) == 30
    assert len(set(G1_JOINT_NAMES)) == len(G1_JOINT_NAMES)
    assert len(set(G1_BODY_NAMES)) == len(G1_BODY_NAMES)
    assert ANCHOR_BODY_NAME == "torso_link"
    assert ANCHOR_BODY_NAME in G1_BODY_NAMES
