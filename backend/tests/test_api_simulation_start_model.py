import os
import json
import pytest
from unittest.mock import patch


def _seed_ready_sim(tmp_path, sim_id="sim_pick_model"):
    """Seed a simulation that's ready to start (has config, no run state).

    Uses status="ready" — the only SimulationStatus value that makes the
    start_simulation handler skip _check_simulation_prepared and proceed
    straight to the runner.  "config_generated" is a *field name*, not a
    valid enum member, so it would raise ValueError inside _load_simulation_state.
    """
    sim_dir = os.path.join(str(tmp_path), sim_id)
    os.makedirs(sim_dir, exist_ok=True)
    with open(os.path.join(sim_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump({"simulation_id": sim_id, "status": "ready",
                   "config_generated": True}, f)
    with open(os.path.join(sim_dir, "simulation_config.json"), "w", encoding="utf-8") as f:
        json.dump({"simulation_requirement": "test"}, f)
    return sim_dir


def test_start_rejects_unknown_model(flask_client, tmp_path):
    _seed_ready_sim(tmp_path)
    resp = flask_client.post("/api/simulation/start", json={
        "simulation_id": "sim_pick_model",
        "selected_model": "openai/gpt-5.4",  # not in allow-list
    })
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False
    assert "model" in body["error"].lower() or "模型" in body["error"]


def test_start_persists_known_model_into_config(flask_client, tmp_path):
    sim_dir = _seed_ready_sim(tmp_path)

    # Stub the runner so the test doesn't actually spawn a subprocess.
    with patch("app.api.simulation.SimulationRunner.start_simulation") as mock_start:
        mock_start.return_value.to_dict.return_value = {"runner_status": "running"}
        resp = flask_client.post("/api/simulation/start", json={
            "simulation_id": "sim_pick_model",
            "selected_model": "gemini/gemini-3.1-flash-lite",
        })

    assert resp.status_code == 200
    with open(os.path.join(sim_dir, "simulation_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["selected_model"] == "gemini/gemini-3.1-flash-lite"


def test_start_without_selected_model_leaves_config_unchanged(flask_client, tmp_path):
    sim_dir = _seed_ready_sim(tmp_path)
    with patch("app.api.simulation.SimulationRunner.start_simulation") as mock_start:
        mock_start.return_value.to_dict.return_value = {"runner_status": "running"}
        resp = flask_client.post("/api/simulation/start", json={
            "simulation_id": "sim_pick_model",
        })
    assert resp.status_code == 200
    with open(os.path.join(sim_dir, "simulation_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert "selected_model" not in cfg
