"""Tests for deterministic semantic-subgraph narrowing."""

from app.runtime.narrowing import narrow_subgraph


def _edge_pairs(result):
    return {(e.source, e.target) for e in result.relationships}


def test_bridge_connector_added_for_disconnected_selection(registry):
    # Condition and Observation share no direct edge; Patient bridges them.
    result = narrow_subgraph(registry, ["Condition", "Observation"])
    assert result.resources == ["Condition", "Observation"]
    assert result.bridge_connectors == ["Patient"]
    assert ("Condition", "Patient") in _edge_pairs(result)
    assert ("Observation", "Patient") in _edge_pairs(result)


def test_hub_selection_does_not_explode(registry):
    # Selecting the Patient hub plus Condition must not pull in every root.
    result = narrow_subgraph(registry, ["Patient", "Condition"])
    assert result.bridge_connectors == []
    nodes = set(result.all_nodes())
    assert nodes == {"Patient", "Condition"}
    assert "Encounter" not in nodes
    assert "Observation" not in nodes


def test_directly_connected_pair_needs_no_bridge(registry):
    result = narrow_subgraph(registry, ["Condition", "Encounter"])
    assert result.bridge_connectors == []
    assert ("Condition", "Encounter") in _edge_pairs(result)


def test_paths_restricted_to_node_set(registry):
    result = narrow_subgraph(registry, ["Condition", "Observation"])
    assert set(result.paths) == {"Condition", "Observation", "Patient"}


def test_unknown_resources_filtered(registry):
    result = narrow_subgraph(registry, ["Condition", "Bogus"])
    assert result.resources == ["Condition"]
