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

    assert "/api/v1/auth/register" in paths
    assert "/api/v1/auth/login" in paths
    assert "/api/v1/auth/refresh" in paths
    assert "/api/v1/me/consents" in paths
    assert "/api/v1/check-ins" in paths
    assert "/api/v1/check-ins/{check_in_id}/diary" in paths
    assert "/api/v1/check-ins/{check_in_id}/analysis-summary" in paths
    assert "/api/v1/check-ins/{check_in_id}/complete" in paths
    assert "/api/v1/lineages" in paths
    assert "/api/v1/lineages/by-check-in/{check_in_id}" in paths
    assert "/api/v1/lineages/{lineage_id}" in paths
    assert "/api/v1/trends/summary" in paths
    assert "/check-ins" not in paths


def test_business_endpoint_requires_bearer_token() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/check-ins")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


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
