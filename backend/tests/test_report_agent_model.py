import os
import json
from unittest.mock import patch, MagicMock


def _seed_sim_for_report(tmp_path, sim_id, model_id=None):
    sim_dir = os.path.join(str(tmp_path), sim_id)
    os.makedirs(sim_dir, exist_ok=True)
    cfg = {"simulation_requirement": "x"}
    if model_id:
        cfg["selected_model"] = model_id
    with open(os.path.join(sim_dir, "simulation_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)


def test_report_agent_uses_selected_model_when_set(tmp_sim_dir):
    _seed_sim_for_report(tmp_sim_dir, "sim_rep", "gemini/gemini-3.1-pro-preview")

    with patch("app.services.report_agent.ReportLLMClient") as MockClient, \
         patch("app.services.report_agent.ZepToolsService"):
        from app.services.report_agent import ReportAgent
        ReportAgent(
            simulation_id="sim_rep",
            graph_id="g1",
            simulation_requirement="test requirement",
        )

    args, kwargs = MockClient.call_args
    assert kwargs.get("model") == "gemini/gemini-3.1-pro-preview"


def test_report_agent_omits_model_when_unset(tmp_sim_dir):
    _seed_sim_for_report(tmp_sim_dir, "sim_rep_default")  # no selected_model

    with patch("app.services.report_agent.ReportLLMClient") as MockClient, \
         patch("app.services.report_agent.ZepToolsService"):
        from app.services.report_agent import ReportAgent
        ReportAgent(
            simulation_id="sim_rep_default",
            graph_id="g1",
            simulation_requirement="test requirement",
        )

    args, kwargs = MockClient.call_args
    # When unset, we don't pass model — let env default kick in.
    assert kwargs.get("model") is None or "model" not in kwargs
