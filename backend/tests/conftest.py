import os
import sys
import pytest

# Make `app` importable when running pytest from source/backend/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def tmp_sim_dir(tmp_path, monkeypatch):
    """Point SimulationManager.SIMULATION_DATA_DIR at a tmp dir for the test."""
    from app.services.simulation_manager import SimulationManager
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def flask_client(monkeypatch, tmp_path):
    """Flask test client with an isolated SimulationManager data dir."""
    from app.services.simulation_manager import SimulationManager
    from app.services.simulation_runner import SimulationRunner
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(SimulationRunner, "_run_states", {})

    from app import create_app  # existing factory
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
