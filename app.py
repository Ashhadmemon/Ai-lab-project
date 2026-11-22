import csv
import io
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session

from gpa_final_model import (FEATURES, RULES, GPAPredictor, InputValidationError,
                              ModelNotFoundError, validate_input)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "outputs" / "models"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or (
    "development-only-change-me" if not os.environ.get("VERCEL") else secrets.token_hex(32)
)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.environ.get("VERCEL")),
)

DB_PATH = Path("/tmp/app_history.db") if os.environ.get("VERCEL") else BASE_DIR / "app_history.db"


def db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                predicted_gpa REAL NOT NULL,
                grade_band TEXT NOT NULL,
                raw_result TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS student_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner TEXT NOT NULL,
                name TEXT NOT NULL,
                payload TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS study_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner TEXT NOT NULL,
                target_gpa REAL NOT NULL,
                horizon_days INTEGER NOT NULL,
                weekly_study_hours REAL NOT NULL,
                weekly_absence_limit INTEGER NOT NULL,
                tutoring INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS goal_checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL,
                owner TEXT NOT NULL,
                study_hours REAL NOT NULL,
                absences INTEGER NOT NULL,
                actual_gpa REAL,
                note TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(goal_id) REFERENCES study_goals(id)
            )
        """)


init_db()


def ensure_owner():
    if "owner" not in session:
        session["owner"] = "guest-" + secrets.token_hex(12)
    return session["owner"]

predictor = None
model_error = None
try:
    predictor = GPAPredictor.load(MODEL_DIR)
except ModelNotFoundError as exc:
    model_error = str(exc)


@app.get("/")
def index():
    meta = predictor.metadata if predictor else {}
    return render_template("index.html", model_error=model_error, rules=RULES,
                           features=FEATURES, meta=meta)


@app.post("/api/predict")
def api_predict():
    if predictor is None:
        return jsonify({"error": model_error}), 503
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Send a JSON body with the 10 student fields."}), 400
    try:
        result = predictor.predict(payload)
        mae = float(predictor.metadata.get("holdout_metrics_regression", {}).get("test_mae", 0))
        gpa = result["predicted_gpa"]
        result["uncertainty"] = {
            "mae": round(mae, 2),
            "lower": round(max(0, gpa - mae), 2),
            "upper": round(min(4, gpa + mae), 2),
        }
        result["coefficients"] = predictor.metadata.get("linear_coefficients", {})

        with db() as connection:
            connection.execute(
                """INSERT INTO predictions
                   (owner, created_at, payload, predicted_gpa, grade_band, raw_result)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ensure_owner(), datetime.now(timezone.utc).isoformat(),
                 json.dumps(payload), gpa, result["grade_band"], json.dumps(result)),
            )
        return jsonify(result)
    except InputValidationError as exc:
        return jsonify({"error": "Some fields need fixing.", "fields": exc.errors}), 400


@app.get("/api/history")
def api_history():
    with db() as connection:
        rows = connection.execute(
            "SELECT id, created_at, payload, predicted_gpa, grade_band FROM predictions "
            "WHERE owner = ? ORDER BY id DESC LIMIT 50",
            (ensure_owner(),),
        ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.post("/api/history/delete")
def delete_history():
    with db() as connection:
        connection.execute("DELETE FROM predictions WHERE owner = ?", (ensure_owner(),))
    return jsonify({"status": "deleted"})


@app.get("/api/history/export")
def export_history():
    with db() as connection:
        rows = connection.execute(
            "SELECT created_at, payload, predicted_gpa, grade_band FROM predictions "
            "WHERE owner = ? ORDER BY id DESC",
            (ensure_owner(),),
        ).fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["created_at", "predicted_gpa", "grade_band", "payload"])
    for row in rows:
        writer.writerow([row["created_at"], row["predicted_gpa"], row["grade_band"], row["payload"]])
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=gpa-history.csv"},
    )


@app.post("/api/profile/save")
def save_profile():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    student = payload.get("student")
    result = payload.get("result")
    if not name or not isinstance(student, dict) or not isinstance(result, dict):
        return jsonify({"error": "Provide a profile name, student inputs, and result."}), 400
    with db() as connection:
        connection.execute(
            """INSERT INTO student_profiles
               (owner, name, payload, result, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (ensure_owner(), name, json.dumps(student),
             json.dumps(result), datetime.now(timezone.utc).isoformat()),
        )
    return jsonify({"status": "saved", "name": name})


@app.get("/api/profile")
def list_profiles():
    with db() as connection:
        rows = connection.execute(
            "SELECT id, name, payload, result, created_at FROM student_profiles "
            "WHERE owner = ? ORDER BY id DESC LIMIT 20",
            (ensure_owner(),),
        ).fetchall()
    profiles = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item["payload"])
        item["result"] = json.loads(item["result"])
        profiles.append(item)
    return jsonify(profiles)


@app.post("/api/profile/delete")
def delete_profiles():
    with db() as connection:
        connection.execute("DELETE FROM student_profiles WHERE owner = ?", (ensure_owner(),))
    return jsonify({"status": "deleted"})


@app.post("/api/plan")
def create_plan():
    student = request.get_json(silent=True) or {}
    if not isinstance(student, dict):
        return jsonify({"error": "Send student details as a JSON object."}), 400
    try:
        clean = validate_input(student)
        result = predictor.predict(clean)
    except (InputValidationError, AttributeError) as exc:
        if isinstance(exc, InputValidationError):
            return jsonify({"error": "Some fields need fixing.", "fields": exc.errors}), 400
        return jsonify({"error": "The model is unavailable."}), 503

    study = float(clean["StudyTimeWeekly"])
    absences = int(clean["Absences"])
    tutoring = int(clean["Tutoring"])
    action_points = []

    if study < 20:
        target_study = min(20, round(study + 2, 1))
        action_points.append({
            "priority": "High",
            "title": "Build a consistent study block",
            "detail": f"Raise weekly study time from {study:g} to {target_study:g} hours.",
        })
    if absences > 0:
        action_points.append({
            "priority": "High",
            "title": "Reduce avoidable absences",
            "detail": f"Use a plan to lower absences from {absences} toward the next lower target.",
        })
    if tutoring == 0:
        action_points.append({
            "priority": "Medium",
            "title": "Schedule focused tutoring",
            "detail": "Choose one difficult subject and arrange one short weekly session.",
        })
    if not action_points:
        action_points.append({
            "priority": "Maintain",
            "title": "Protect your current habits",
            "detail": "Keep study time, attendance, and activities consistent for 30 days.",
        })

    weeks = [
        {"week": "Week 1", "focus": "Baseline", "tasks": ["Complete a study-time audit", "Choose two priority subjects", "Set one fixed study slot per day"]},
        {"week": "Week 2", "focus": "Build consistency", "tasks": ["Complete at least 5 planned study blocks", "Review missed work within 24 hours", "Track absences and reasons"]},
        {"week": "Week 3", "focus": "Increase practice", "tasks": ["Increase study time by 30 minutes on two days", "Complete a timed practice set", "Ask for help on one difficult topic"]},
        {"week": "Week 4", "focus": "Review and set targets", "tasks": ["Review progress against the baseline", "Remove one low-value habit", "Set the next 30-day goal"]},
    ]
    return jsonify({
        "predicted_gpa": result["predicted_gpa"],
        "priorities": action_points,
        "weeks": weeks,
    })


@app.get("/api/model")
def api_model():
    if predictor is None:
        return jsonify({"error": model_error}), 503
    metadata = dict(predictor.metadata)
    return jsonify(metadata)


@app.get("/api/model/comparison")
def model_comparison():
    path = BASE_DIR / "outputs_notebook" / "model_comparison.csv"
    if not path.exists():
        return jsonify({"available": False, "results": []})
    import pandas as pd
    frame = pd.read_csv(path)
    return jsonify({"available": True, "results": frame.fillna("").to_dict(orient="records")})


@app.get("/api/model/fairness")
def model_fairness():
    path = BASE_DIR / "data" / "processed" / "students_clean.csv"
    if not path.exists():
        return jsonify({"available": False, "slices": []})
    import pandas as pd
    frame = pd.read_csv(path)
    slices = []
    for column in ("Age", "ParentalEducation", "ParentalSupport"):
        if column not in frame.columns or "GPA" not in frame.columns:
            continue
        summary = frame.groupby(column)["GPA"].agg(["count", "mean", "std"]).reset_index()
        slices.append({
            "feature": column,
            "groups": [{"value": str(row[column]), "count": int(row["count"]),
                        "mean_gpa": round(float(row["mean"]), 3),
                        "std_gpa": round(float(row["std"] if pd.notna(row["std"]) else 0), 3)}
                       for row in summary.to_dict(orient="records")]
        })
    return jsonify({"available": True, "note": "Descriptive slice audit; it does not measure model error or prove fairness.", "slices": slices})


@app.post("/api/scenarios")
def api_scenarios():
    if predictor is None:
        return jsonify({"error": model_error}), 503
    payload = request.get_json(silent=True) or {}
    try:
        clean = validate_input(payload)
        base = predictor.predict(clean)
    except InputValidationError as exc:
        return jsonify({"error": "Some fields need fixing.", "fields": exc.errors}), 400

    scenario_specs = {
        "study-plus-five": {"StudyTimeWeekly": min(20, float(clean["StudyTimeWeekly"]) + 5)},
        "absence-minus-five": {"Absences": max(0, int(clean["Absences"])-5)},
        "combined": {
            "StudyTimeWeekly": min(20, float(clean["StudyTimeWeekly"]) + 5),
            "Absences": max(0, int(clean["Absences"])-5),
            "Tutoring": 1,
        },
    }
    results = {"current": base}
    for name, changes in scenario_specs.items():
        scenario = dict(clean)
        scenario.update(changes)
        results[name] = predictor.predict(scenario)
        results[name]["gain"] = round(results[name]["predicted_gpa"] - base["predicted_gpa"], 2)
    return jsonify(results)


@app.post("/api/goals")
def create_goal():
    payload = request.get_json(silent=True) or {}
    try:
        target = float(payload.get("target_gpa"))
        horizon = int(payload.get("horizon_days", 30))
        study = float(payload.get("weekly_study_hours", 10))
        absence_limit = int(payload.get("weekly_absence_limit", 0))
        tutoring = int(payload.get("tutoring", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Provide numeric target, horizon, study, absence, and tutoring values."}), 400
    if not 0 <= target <= 4 or horizon not in (30, 60, 90) or not 0 <= study <= 20 or absence_limit < 0:
        return jsonify({"error": "Use a GPA from 0 to 4, a 30/60/90-day horizon, and valid habit targets."}), 400
    with db() as connection:
        cursor = connection.execute("""
            INSERT INTO study_goals
            (owner, target_gpa, horizon_days, weekly_study_hours,
             weekly_absence_limit, tutoring, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ensure_owner(), target, horizon, study, absence_limit,
              1 if tutoring else 0, datetime.now(timezone.utc).isoformat()))
    return jsonify({"status": "created", "goal_id": cursor.lastrowid})


@app.get("/api/goals")
def list_goals():
    with db() as connection:
        rows = connection.execute("""
            SELECT * FROM study_goals WHERE owner = ? ORDER BY id DESC LIMIT 20
        """, (ensure_owner(),)).fetchall()
    goals = [dict(row) for row in rows]
    for goal in goals:
        with db() as connection:
            checkins = connection.execute("""
                SELECT * FROM goal_checkins WHERE goal_id = ? ORDER BY id ASC
            """, (goal["id"],)).fetchall()
        goal["checkins"] = [dict(row) for row in checkins]
    return jsonify(goals)


@app.post("/api/goals/<int:goal_id>/checkin")
def create_checkin(goal_id):
    payload = request.get_json(silent=True) or {}
    try:
        study = float(payload.get("study_hours"))
        absences = int(payload.get("absences"))
        actual_gpa = float(payload["actual_gpa"]) if payload.get("actual_gpa") not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify({"error": "Study hours and absences are required and must be numeric."}), 400
    if not 0 <= study <= 20 or absences < 0 or (actual_gpa is not None and not 0 <= actual_gpa <= 4):
        return jsonify({"error": "Check-in values are outside the supported ranges."}), 400
    with db() as connection:
        goal = connection.execute("SELECT id FROM study_goals WHERE id = ? AND owner = ?",
                                  (goal_id, ensure_owner())).fetchone()
        if goal is None:
            return jsonify({"error": "Goal not found."}), 404
        connection.execute("""
            INSERT INTO goal_checkins
            (goal_id, owner, study_hours, absences, actual_gpa, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (goal_id, ensure_owner(), study, absences, actual_gpa,
              str(payload.get("note", "")).strip(), datetime.now(timezone.utc).isoformat()))
    return jsonify({"status": "saved"})


@app.post("/api/goals/<int:goal_id>/complete")
def complete_goal(goal_id):
    with db() as connection:
        cursor = connection.execute("""
            UPDATE study_goals SET status = 'completed', completed_at = ?
            WHERE id = ? AND owner = ?
        """, (datetime.now(timezone.utc).isoformat(), goal_id, ensure_owner()))
    if cursor.rowcount == 0:
        return jsonify({"error": "Goal not found."}), 404
    return jsonify({"status": "completed"})


@app.post("/api/privacy/delete-all")
def privacy_delete_all():
    owner = ensure_owner()
    with db() as connection:
        connection.execute("DELETE FROM goal_checkins WHERE owner = ?", (owner,))
        connection.execute("DELETE FROM study_goals WHERE owner = ?", (owner,))
        connection.execute("DELETE FROM student_profiles WHERE owner = ?", (owner,))
        connection.execute("DELETE FROM predictions WHERE owner = ?", (owner,))
    return jsonify({"status": "deleted", "owner": owner})


@app.get("/health")
def health():
    return jsonify({"status": "ok" if predictor else "model_missing"})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
