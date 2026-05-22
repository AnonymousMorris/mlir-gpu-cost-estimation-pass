from fastapi.testclient import TestClient

from cost_ui.app import app
from tests.test_expression import SAMPLE_COST_MLIR


def test_parse_endpoint(monkeypatch):
    monkeypatch.setattr("cost_ui.mlir_parser._parse_with_mlir_binding", lambda text: text)
    client = TestClient(app)

    response = client.post("/api/parse", json={"mlir": SAMPLE_COST_MLIR})

    assert response.status_code == 200
    payload = response.json()
    assert payload["variables"] == ["dpas_cost"]
    assert payload["equationText"] == "4096.0*dpas_cost + 3505.0"
    assert payload["javascriptParameters"] == ["v0"]
    assert payload["javascriptExpression"] == "4096.0*v0 + 3505.0"
