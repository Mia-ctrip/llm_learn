from fastapi.testclient import TestClient

from app.api import analyses, lineages
from app.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_openapi_exposes_tracking_endpoints() -> None:
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert "/check-ins" in paths
    assert "/check-ins/{check_in_id}/diary" in paths
    assert "/check-ins/{check_in_id}/analysis-summary" in paths
    assert "/check-ins/{check_in_id}/complete" in paths
    assert "/lineages" in paths
    assert "/lineages/by-check-in/{check_in_id}" in paths
    assert "/lineages/{lineage_id}" in paths
    assert "/trends/summary" in paths


def test_static_by_photo_routes_precede_dynamic_id_routes() -> None:
    analysis_paths = [route.path for route in analyses.router.routes]
    lineage_paths = [route.path for route in lineages.router.routes]

    assert analysis_paths.index("/analyses/by-photo/{photo_id}") < analysis_paths.index(
        "/analyses/{analysis_id}"
    )
    assert lineage_paths.index("/lineages/by-photo/{photo_id}") < lineage_paths.index(
        "/lineages/{lineage_id}"
    )
    assert lineage_paths.index("/lineages/by-check-in/{check_in_id}") < lineage_paths.index(
        "/lineages/{lineage_id}"
    )
