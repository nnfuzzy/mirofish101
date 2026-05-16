import os
import json
import pytest
from unittest.mock import patch, MagicMock


def _seed_sim_with_model(base_dir, sim_id="sim_env", model_id="gemini/gemini-3.1-flash-lite"):
    sim_dir = os.path.join(str(base_dir), sim_id)
    os.makedirs(sim_dir, exist_ok=True)
    with open(os.path.join(sim_dir, "simulation_config.json"), "w", encoding="utf-8") as f:
        json.dump({"selected_model": model_id, "simulation_requirement": "x"}, f)
    return sim_dir


def test_subprocess_env_carries_selected_model(tmp_path, monkeypatch):
    from app.services.simulation_runner import SimulationRunner

    # Patch RUN_STATE_DIR so the runner reads/writes under tmp_path, not the real dir.
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    # Clear class-level state cache to avoid spurious RUNNING/STARTING guard failures.
    monkeypatch.setattr(SimulationRunner, "_run_states", {})

    _seed_sim_with_model(tmp_path, "sim_env", "gemini/gemini-3.1-flash-lite")

    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})
        proc = MagicMock()
        proc.pid = 12345
        return proc

    with patch("app.services.simulation_runner.subprocess.Popen", side_effect=fake_popen), \
         patch.object(SimulationRunner, "_monitor_simulation"), \
         patch("threading.Thread") as mock_thread:
        mock_thread.return_value.start = MagicMock()
        try:
            SimulationRunner.start_simulation(
                simulation_id="sim_env",
                platform="parallel",
                max_rounds=10,
            )
        except Exception:
            # We only care that Popen got called with our env injection.
            pass

    assert "env" in captured, "Popen was never reached — a pre-Popen guard fired"
    assert captured["env"].get("LITELLM_MODEL") == "gemini/gemini-3.1-flash-lite"
    assert captured["env"].get("LITELLM_REPORT_MODEL") == "gemini/gemini-3.1-flash-lite"
