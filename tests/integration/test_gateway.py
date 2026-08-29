from fastapi.testclient import TestClient

from gateway.api import app


def test_health_and_model_catalog_endpoints():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert isinstance(response.json()["data"], list)


def test_chat_contract_rejects_streaming():
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}],
                                                          "stream": True})
    assert response.status_code == 400
