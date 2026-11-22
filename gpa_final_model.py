
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TEST_SIZE = 0.20
TARGET = "GPA"
FEATURES = ["Age", "ParentalEducation", "StudyTimeWeekly", "Absences", "Tutoring",
            "ParentalSupport", "Extracurricular", "Sports", "Music", "Volunteering"]

RULES = {
    "Age": (15, 18, True),
    "ParentalEducation": (0, 4, True),
    "StudyTimeWeekly": (0, 20, False),
    "Absences": (0, 30, True),
    "Tutoring": (0, 1, True),
    "ParentalSupport": (0, 4, True),
    "Extracurricular": (0, 1, True),
    "Sports": (0, 1, True),
    "Music": (0, 1, True),
    "Volunteering": (0, 1, True),
}

GPA_MIN, GPA_MAX = 0.0, 4.0
BAND_NAMES = ["A", "B", "C", "D", "F"]          # class codes 0..4
PIPELINE_FILE = "gpa_pipeline.joblib"
CLASSIFIER_FILE = "grade_classifier.joblib"
METADATA_FILE = "model_metadata.json"


class InputValidationError(ValueError):
    """Raised when a prediction request is missing fields or has bad values."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


class ModelNotFoundError(FileNotFoundError):
    """Raised when the saved model artifacts are missing or unreadable."""


def gpa_to_band_code(gpa) -> np.ndarray:
    """Grade band code from GPA: 0=A (>=3.5), 1=B (>=3.0), 2=C (>=2.5), 3=D (>=2.0), 4=F."""
    return np.select([gpa >= 3.5, gpa >= 3.0, gpa >= 2.5, gpa >= 2.0], [0, 1, 2, 3], default=4)


def build_regressor() -> Pipeline:
    """Winning GPA model: plain Linear Regression (defaults), wrapped as a Pipeline."""
    return Pipeline([("model", LinearRegression())])


def build_classifier() -> Pipeline:
    """Winning band model: scaled Logistic Regression, tuned C=10 (from the notebook)."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=2000))


def validate_input(raw: dict) -> dict:
    """Check one student's inputs. Returns clean numeric values or raises InputValidationError."""
    if not isinstance(raw, dict):
        raise InputValidationError({"input": "expected a JSON object / dict of fields"})

    errors: dict[str, str] = {}
    for key in raw:
        if key not in RULES:
            errors[key] = "unexpected field (only the 10 allowed features are accepted)"

    clean: dict[str, float] = {}
    for name, (lo, hi, must_be_int) in RULES.items():
        if name not in raw or raw[name] is None or raw[name] == "":
            errors[name] = "required"
            continue
        value = raw[name]
        if isinstance(value, bool):
            errors[name] = "must be a number"
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors[name] = "must be a number"
            continue
        if not math.isfinite(value):
            errors[name] = "must be a finite number"
            continue
        if must_be_int and value != int(value):
            errors[name] = "must be a whole number"
            continue
        if not lo <= value <= hi:
            errors[name] = f"must be between {lo} and {hi}"
            continue
        clean[name] = value

    if errors:
        raise InputValidationError(errors)
    return clean



class GPAPredictor:
    """Loads the saved pipelines once and turns a validated student record into a prediction."""

    def __init__(self, regressor: Pipeline, classifier: Pipeline | None, metadata: dict):
        self._regressor = regressor
        self._classifier = classifier
        self.metadata = metadata
        
        from advisor import LinearCoefficientAdvisor
        self._advisor = LinearCoefficientAdvisor(metadata.get("linear_coefficients", {}))

    @classmethod
    def load(cls, model_dir: str | Path = "outputs/models") -> "GPAPredictor":
        model_dir = Path(model_dir)
        try:
            regressor = joblib.load(model_dir / PIPELINE_FILE)
            classifier = joblib.load(model_dir / CLASSIFIER_FILE)
            metadata = json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, EOFError, ValueError) as exc:
            raise ModelNotFoundError(
                f"Could not load model files from '{model_dir}'. "
                f"Run: python gpa_final_model.py --data <students_clean.csv> --out {model_dir}"
            ) from exc
        return cls(regressor, classifier, metadata)

    def predict(self, raw: dict) -> dict:
        """Validate -> predict GPA -> band. Raises InputValidationError on bad input."""
        clean = validate_input(raw)
        row = pd.DataFrame([clean], columns=FEATURES)

        raw_gpa = float(self._regressor.predict(row)[0])
        gpa = min(max(raw_gpa, GPA_MIN), GPA_MAX)          
        band = BAND_NAMES[int(gpa_to_band_code(gpa))]

        result = {
            "predicted_gpa": round(gpa, 2),
            "grade_band": band,                              
            "was_clipped": raw_gpa != gpa,                   
            "raw_prediction": round(raw_gpa, 4),
            "disclaimer": ("Estimate from patterns in a high-school dataset. Not a guarantee "
                           "and not for decisions about real students."),
        }
        if self._classifier is not None:
            code = int(self._classifier.predict(row)[0])
            proba = self._classifier.predict_proba(row)[0]
            result["classifier_band"] = BAND_NAMES[code]
            result["classifier_probabilities"] = {
                BAND_NAMES[int(c)]: round(float(p), 3)
                for c, p in zip(self._classifier.classes_, proba)
            }

        suggestions = self._advisor.advise(clean, gpa, RULES)
        result["suggestions"] = [s.as_dict() for s in suggestions]
        return result



def train(data_path: str | Path, out_dir: str | Path) -> dict:
    """Check metrics on the 20% test split, refit on all rows, save artifacts + metadata."""
    random.seed(SEED)
    np.random.seed(SEED)

    try:
        df = pd.read_csv(data_path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise SystemExit(f"Cannot read data file '{data_path}': {exc}") from exc

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Data file is missing columns: {missing}")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise SystemExit("Data has missing values in the feature/target columns; clean it first.")

    X, y = df[FEATURES], df[TARGET]
 
    y_band = pd.Series(gpa_to_band_code(y.to_numpy()), index=df.index)


    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    reg_check = build_regressor().fit(X_dev, y_dev)
    pred = reg_check.predict(X_test)
    reg_metrics = {
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "test_r2": float(r2_score(y_test, pred)),
    }

    Xc_dev, Xc_test, yc_dev, yc_test = train_test_split(
        X, y_band, test_size=TEST_SIZE, random_state=SEED, stratify=y_band)
    clf_check = build_classifier().fit(Xc_dev, yc_dev)
    pc = clf_check.predict(Xc_test)
    clf_metrics = {
        "test_accuracy": float(accuracy_score(yc_test, pc)),
        "test_macro_f1": float(f1_score(yc_test, pc, average="macro")),
        "test_balanced_acc": float(balanced_accuracy_score(yc_test, pc)),
        "confusion_matrix": confusion_matrix(
            yc_test, pc, labels=sorted(set(yc_test) | set(pc))).tolist(),
        "confusion_matrix_labels": [BAND_NAMES[c] for c in sorted(set(yc_test) | set(pc))],
    }

    regressor = build_regressor().fit(X, y)
    classifier = build_classifier().fit(X, y_band)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(regressor, out_dir / PIPELINE_FILE)
    joblib.dump(classifier, out_dir / CLASSIFIER_FILE)

    coefs = dict(zip(FEATURES, (float(c) for c in regressor.named_steps["model"].coef_)))
    metadata = {
        "regression_model": "LinearRegression (defaults)",
        "classification_model": "LogisticRegression (C=10, StandardScaler)",
        "selected_by": "lowest 5x10 repeated-CV MAE (regression) / macro-F1 (classification)",
        "notebook_cv_mae": 0.1603,
        "notebook_cv_accuracy": 0.8244,
        "holdout_metrics_regression": reg_metrics,
        "holdout_metrics_classification": clf_metrics,
        "trained_on_rows": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "seed": SEED,
        "allowed_ranges": {k: [v[0], v[1]] for k, v in RULES.items()},
        "linear_coefficients": coefs,
        "intercept": float(regressor.named_steps["model"].intercept_),
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope_note": "High-school dataset; not a validated university CGPA predictor.",
    }
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Holdout check (80/20 split, seed 42)")
    print("  GPA   : MAE %.4f | RMSE %.4f | R2 %.4f" % tuple(reg_metrics.values()))
    print("  Band  : accuracy %.4f | macro-F1 %.4f | balanced acc %.4f" % (
        clf_metrics["test_accuracy"], clf_metrics["test_macro_f1"], clf_metrics["test_balanced_acc"]))
    print("  Confusion matrix %s:\n%s" % (clf_metrics["confusion_matrix_labels"],
                                           clf_metrics["confusion_matrix"]))
    print("Final models refit on all %d rows and saved to %s" % (len(df), out_dir.resolve()))
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and save the final GPA models.")
    parser.add_argument("--data", default="data/processed/students_clean.csv")
    parser.add_argument("--out", default="outputs/models")
    args = parser.parse_args(argv)
    train(args.data, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
