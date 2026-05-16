import os
import json
import pytest

from app.services.simulation_manager import SimulationManager


def _seed_sim(tmp_sim_dir, sim_id="sim_abc123"):
    sim_dir = os.path.join(str(tmp_sim_dir), sim_id)
    os.makedirs(sim_dir, exist_ok=True)
    with open(os.path.join(sim_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump({"simulation_id": sim_id, "status": "created"}, f)
    with open(os.path.join(sim_dir, "simulation_config.json"), "w", encoding="utf-8") as f:
        json.dump({"simulation_requirement": "test"}, f)
    return sim_dir


def test_delete_simulation_happy_path(tmp_sim_dir):
    sim_dir = _seed_sim(tmp_sim_dir, "sim_abc123")
    manager = SimulationManager()
    manager._simulations["sim_abc123"] = object()  # populate cache

    deleted = manager.delete_simulation("sim_abc123")

    assert deleted is True
    assert not os.path.exists(sim_dir)
    assert "sim_abc123" not in manager._simulations


def test_delete_simulation_not_found(tmp_sim_dir):
    manager = SimulationManager()
    assert manager.delete_simulation("sim_does_not_exist") is False


def test_delete_simulation_idempotent(tmp_sim_dir):
    _seed_sim(tmp_sim_dir, "sim_xyz")
    manager = SimulationManager()

    assert manager.delete_simulation("sim_xyz") is True
    assert manager.delete_simulation("sim_xyz") is False  # second call: not found


def test_delete_simulation_evicts_cache_even_if_dir_missing(tmp_sim_dir):
    manager = SimulationManager()
    manager._simulations["sim_ghost"] = object()
    # No directory on disk — manager should still evict the stale cache entry.
    assert manager.delete_simulation("sim_ghost") is False
    assert "sim_ghost" not in manager._simulations
