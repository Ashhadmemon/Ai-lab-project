import csv
import io
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session, Response
from werkzeug.security import check_password_hash, generate_password_hash

from gpa_final_model import (FEATURES, RULES, GPAPredictor, InputValidationError,
                              ModelNotFoundError)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "outputs" / "models"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "development-only-change-this")
DATABASE = Path("/tmp/app_history.db") if os.environ.get("VERCEL") else BASE_DIR / "data" / "app_history.db"


def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                inputs TEXT NOT NULL,
                result TEXT NOT NULL
            )
        """)


def current_session_id():
    if "user_id" in session:
        return f"user:{session['user_id']}"
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


init_db()


@app.post("/api/auth/register")
def register():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip().lower()
    password = str(payload.get("password", ""))
    if len(username) < 3 or len(password) < 8:
        return jsonify({"error": "Use a username of at least 3 characters and a password of at least 8 characters."}), 400
    try:
        with get_db() as connection:
            cursor = connection.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, generate_password_hash(password), datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            session["user_id"] = cursor.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({"error": "That username is already registered."}), 409
    return jsonify({"username": username})


@app.post("/api/auth/login")
def login():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip().lower()
    password = str(payload.get("password", ""))
    with get_db() as connection:
        user = connection.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,)).fetchone()
    if user is None or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Username or password is incorrect."}), 401
    session["user_id"] = user["id"]
    return jsonify({"username": username})


@app.post("/api/auth/logout")
def logout():
    session.pop("user_id", None)
    return jsonify({"logged_out": True})


@app.get("/api/auth/status")
def auth_status():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"authenticated": False})
    with get_db() as connection:
        user = connection.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
    return jsonify({"authenticated": user is not None, "username": user["username"] if user else None})


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
        metadata = predictor.metadata
        holdout_mae = metadata.get("holdout_metrics_regression", {}).get("test_mae")
        if holdout_mae is not None:
            result["uncertainty"] = {
                "holdout_mae": round(float(holdout_mae), 3),
                "range_low": round(max(0.0, result["predicted_gpa"] - float(holdout_mae)), 2),
                "range_high": round(min(4.0, result["predicted_gpa"] + float(holdout_mae)), 2),
            }
        result["explanation"] = explain_prediction(payload)
        if request.args.get("save") == "1":
            with get_db() as connection:
                connection.execute(
                    "INSERT INTO prediction_history (session_id, created_at, inputs, result) VALUES (?, ?, ?, ?)",
                    (current_session_id(), datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     json.dumps(payload), json.dumps(result)),
                )
        return jsonify(result)
    except InputValidationError as exc:
        return jsonify({"error": "Some fields need fixing.", "fields": exc.errors}), 400


def explain_prediction(payload):
    clean = json.loads(json.dumps(payload))
    coefficients = predictor.metadata.get("linear_coefficients", {}) if predictor else {}
    ranked = sorted(
        ({"feature": name, "coefficient": round(float(value), 4)}
         for name, value in coefficients.items()),
        key=lambda item: abs(item["coefficient"]), reverse=True,
    )
    return ranked[:5]


@app.get("/api/history")
def prediction_history():
    with get_db() as connection:
        rows = connection.execute(
            "SELECT id, created_at, inputs, result FROM prediction_history WHERE session_id = ? ORDER BY id DESC LIMIT 50",
            (current_session_id(),),
        ).fetchall()
    return jsonify([{"id": row["id"], "created_at": row["created_at"],
                     "inputs": json.loads(row["inputs"]),
                     "result": json.loads(row["result"])} for row in rows])


@app.delete("/api/history")
def clear_history():
    with get_db() as connection:
        connection.execute("DELETE FROM prediction_history WHERE session_id = ?", (current_session_id(),))
    return jsonify({"cleared": True})


@app.get("/api/history/export")
def export_history():
    with get_db() as connection:
        rows = connection.execute(
            "SELECT created_at, inputs, result FROM prediction_history WHERE session_id = ? ORDER BY id DESC",
            (current_session_id(),),
        ).fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["created_at", "predicted_gpa", "grade_band", "inputs"])
    for row in rows:
        result = json.loads(row["result"])
        writer.writerow([row["created_at"], result.get("predicted_gpa"), result.get("grade_band"), row["inputs"]])
    return Response(output.getvalue(), mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=gpa-history.csv"
    })


@app.get("/api/model")
def api_model():
    if predictor is None:
        return jsonify({"error": model_error}), 503
    return jsonify(predictor.metadata)


@app.get("/health")
def health():
    return jsonify({"status": "ok" if predictor else "model_missing"})


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
