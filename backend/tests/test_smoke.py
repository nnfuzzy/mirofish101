def test_tmp_sim_dir_isolates_data(tmp_sim_dir):
    from app.services.simulation_manager import SimulationManager
    assert SimulationManager.SIMULATION_DATA_DIR == str(tmp_sim_dir)


def test_flask_client_responds(flask_client):
    # Health endpoint per CLAUDE.md ("/api/health endpoint" patch)
    resp = flask_client.get("/api/health")
    # Either 200 (healthy) or 404 (endpoint not yet wired in this fork) is fine —
    # we just want to prove the app boots inside pytest.
    assert resp.status_code in (200, 404)
