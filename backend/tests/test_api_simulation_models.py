def test_models_endpoint_returns_allow_list(flask_client):
    resp = flask_client.get("/api/simulation/models")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    ids = [m["id"] for m in body["data"]["models"]]
    assert "gemini/gemini-3-flash-preview" in ids
    assert "gemini/gemini-3.1-pro-preview" in ids
    assert "gemini/gemini-3.1-flash-lite" in ids
    assert body["data"]["default_model_id"] == "gemini/gemini-3-flash-preview"
