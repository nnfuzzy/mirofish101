import os
import json
import pytest


def _seed_sim(tmp_path, sim_id, runner_status=None):
    sim_dir = os.path.join(str(tmp_path), sim_id)
    os.makedirs(sim_dir, exist_ok=True)
    with open(os.path.join(sim_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump({"simulation_id": sim_id, "status": "created"}, f)
    if runner_status:
        with open(os.path.join(sim_dir, "run_state.json"), "w", encoding="utf-8") as f:
            json.dump({"simulation_id": sim_id, "runner_status": runner_status}, f)
    return sim_dir


def test_delete_returns_200_and_removes_dir(flask_client, tmp_path):
    sim_dir = _seed_sim(tmp_path, "sim_del_ok")
    resp = flask_client.delete("/api/simulation/sim_del_ok")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["data"]["simulation_id"] == "sim_del_ok"
    assert not os.path.exists(sim_dir)


def test_delete_returns_404_when_missing(flask_client):
    resp = flask_client.delete("/api/simulation/sim_nope")
    assert resp.status_code == 404
    body = resp.get_json()
    assert body["success"] is False
    assert "not_found" in body["error"] or "未找到" in body["error"]


@pytest.mark.parametrize("status", ["starting", "running", "paused", "stopping"])
def test_delete_returns_409_when_in_flight(flask_client, tmp_path, status):
    sim_dir = _seed_sim(tmp_path, f"sim_{status}", runner_status=status)
    resp = flask_client.delete(f"/api/simulation/sim_{status}")
    assert resp.status_code == 409, f"Expected 409 for status={status}, got {resp.status_code}"
    body = resp.get_json()
    assert body["success"] is False
    assert body["data"]["runner_status"] == status
    assert os.path.exists(sim_dir)  # dir untouched
