def test_tmp_sim_dir_isolates_data(tmp_sim_dir):
    from app.services.simulation_manager import SimulationManager
    assert SimulationManager.SIMULATION_DATA_DIR == str(tmp_sim_dir)


def test_flask_client_responds(flask_client):
    # Upstream ships the health route at /health (not /api/health); nginx and
    # the K8s probe both point at it, so assert it for real.
    resp = flask_client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"
