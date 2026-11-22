import pytest

from app import app


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


VALID_INPUT = {
    "Age": 16,
    "ParentalEducation": 2,
    "StudyTimeWeekly": 10,
    "Absences": 5,
    "Tutoring": 0,
    "ParentalSupport": 2,
    "Extracurricular": 1,
    "Sports": 0,
    "Music": 1,
    "Volunteering": 0,
}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] in {"ok", "model_missing"}


def test_prediction_returns_uncertainty_and_explanation(client):
    if client.get("/api/model").status_code == 503:
        pytest.skip("Model artifacts are not present")
    response = client.post("/api/predict", json=VALID_INPUT)
    assert response.status_code == 200
    assert 0 <= response.json["predicted_gpa"] <= 4
    assert "uncertainty" in response.json
    assert "explanation" in response.json


def test_prediction_validation(client):
    response = client.post("/api/predict", json={"Age": 14})
    assert response.status_code == 400
    assert "fields" in response.json


def test_optional_account_registration_login_and_logout(client):
    registered = client.post("/api/auth/register", json={"username": "studentdemo", "password": "securepass123"})
    assert registered.status_code in {200, 409}

    if registered.status_code == 200:
        status = client.get("/api/auth/status")
        assert status.json["authenticated"] is True

        logout = client.post("/api/auth/logout")
        assert logout.status_code == 200
        assert logout.json["logged_out"] is True


def test_history_save_export_and_clear(client):
    if client.get("/api/model").status_code == 503:
        pytest.skip("Model artifacts are not present")
    client.post("/api/predict?save=1", json=VALID_INPUT)
    history = client.get("/api/history")
    assert history.status_code == 200
    assert isinstance(history.json, list)

    exported = client.get("/api/history/export")
    assert exported.status_code == 200
    assert exported.mimetype == "text/csv"

    cleared = client.delete("/api/history")
    assert cleared.status_code == 200
    assert cleared.json["cleared"] is True
