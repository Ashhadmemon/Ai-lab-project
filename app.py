import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from gpa_final_model import (FEATURES, RULES, GPAPredictor, InputValidationError,
                              ModelNotFoundError)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "outputs" / "models"

app = Flask(__name__)

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
        return jsonify(predictor.predict(payload))
    except InputValidationError as exc:
        return jsonify({"error": "Some fields need fixing.", "fields": exc.errors}), 400


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
