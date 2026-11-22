from app import app


def test_health():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.is_json


def test_index():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Goal planner" in response.data


def test_prediction_validation():
    client = app.test_client()
    response = client.post("/api/predict", json={"Age": 99})
    assert response.status_code == 400
    assert response.is_json
    assert "fields" in response.get_json()


def test_history_and_privacy():
    client = app.test_client()
    response = client.get("/api/history")
    assert response.status_code == 200
    deleted = client.post("/api/history/delete")
    assert deleted.status_code == 200


def test_scenarios():
    client = app.test_client()
    payload = {
        "Age": 16,
        "ParentalEducation": 2,
        "StudyTimeWeekly": 8,
        "Absences": 6,
        "Tutoring": 0,
        "ParentalSupport": 2,
        "Extracurricular": 1,
        "Sports": 0,
        "Music": 1,
        "Volunteering": 0,
    }
    response = client.post("/api/scenarios", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "current" in data
    assert "combined" in data


def test_goals_and_checkins():
    client = app.test_client()
    created = client.post("/api/goals", json={
        "target_gpa": 3.5,
        "horizon_days": 30,
        "weekly_study_hours": 15,
        "weekly_absence_limit": 1,
        "tutoring": 1,
    })
    assert created.status_code == 200
    goal_id = created.get_json()["goal_id"]
    checkin = client.post(f"/api/goals/{goal_id}/checkin", json={
        "study_hours": 12,
        "absences": 1,
        "actual_gpa": 3.2,
        "note": "Improving",
    })
    assert checkin.status_code == 200
    goals = client.get("/api/goals")
    assert goals.status_code == 200
    assert goals.get_json()[0]["checkins"]


def test_model_fairness_and_comparison():
    client = app.test_client()
    comparison = client.get("/api/model/comparison")
    fairness = client.get("/api/model/fairness")
    assert comparison.status_code == 200
    assert fairness.status_code == 200


def test_profile_save_and_list():
    client = app.test_client()
    payload = {
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
    prediction = client.post("/api/predict", json=payload)
    assert prediction.status_code == 200
    saved = client.post("/api/profile/save", json={
        "name": "Test student",
        "student": payload,
        "result": prediction.get_json(),
    })
    assert saved.status_code == 200
    profiles = client.get("/api/profile")
    assert profiles.status_code == 200
    assert any(profile["name"] == "Test student" for profile in profiles.get_json())


def test_30_day_plan():
    client = app.test_client()
    response = client.post("/api/plan", json={
        "Age": 16,
        "ParentalEducation": 2,
        "StudyTimeWeekly": 8,
        "Absences": 6,
        "Tutoring": 0,
        "ParentalSupport": 2,
        "Extracurricular": 1,
        "Sports": 0,
        "Music": 1,
        "Volunteering": 0,
    })
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["priorities"]) >= 1
    assert len(data["weeks"]) == 4
