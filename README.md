# AI Based Student GPA Prediction and Improvement Advisor

A small Flask web app that estimates a high-school student's GPA and grade
band from ten background/habit fields, and suggests which of those habits 
would move the estimate up the most.

## Contributors
- Ashhad Pervaiz 24k-2544
- Waqar Nadeem 24k-2524

## Problem formulation (PEAS)

| | |
|---|---|
| **Performance measure** | Low MAE/RMSE and high R² on held-out GPA (regression); high accuracy/macro-F1 on held-out grade band (classification). See `outputs/models/model_metadata.json` after training. |
| **Environment** | A fixed historical dataset of high-school student records. |
| **Actuators** | The predicted GPA, grade band, and a ranked list of actionable habit changes (see `advisor.py`), shown through the web UI. |
| **Sensors** | The 10 fields a user submits through the form in `index.html` / `POST /api/predict`. |
| **Environment class** | Partially observable, deterministic (for this model), static, discrete, single-agent. |
| **Agent type** | A simple reflex agent: a learned linear mapping from percept (feature vector) to action (predicted GPA), with no state carried between requests. |

## What's in each file

- `gpa_final_model.py` — training (`train()`), input validation, and the
  `GPAPredictor` class used at request time. Run directly to (re)train.
- `advisor.py` — turns the trained model's coefficients into the "what
  would help" suggestions (`ImprovementAdvisor` base class,
  `LinearCoefficientAdvisor` implementation — see its docstring for the
  OOP design note).
- `app.py` — the Flask routes (`/`, `/api/predict`, `/api/model`, `/health`).
- `templates/index.html` — the form and result UI.

## Folder Hierarchy
project
│
├── data/
│   ├── raw/
│   │   └── Student_performance_data.csv
│   │
│   └── processed/
│       └── students_clean.csv
│
├── outputs/
│   └── models/
│       ├── gpa_pipeline.joblib
│       ├── grade_classifier.joblib
│       └── model_metadata.json
│
├── outputs_notebook/
│   ├── best_gpa_model.joblib
│   ├── classification_results.csv
│   ├── confusion_matrix.png
│   ├── cv_mae_by_model.png
│   ├── model_comparison.csv
│   ├── model_metadata.json
│   ├── permutation_importance.png
│   └── predicted_vs_actual_and_residuals.png
│
├── templates/
│   └── index.html
│
├── advisor.py
├── app.py
├── clean_data.py
├── gpa_final_model.py
├── gpa_model_comparison.ipynb
├── README.md
└── requirements.txt


## Setup

```bash
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt
```

## Train the model

```bash
python gpa_final_model.py --data data/processed/students_clean.csv --out outputs/models
```

This re-checks metrics on an untouched 20% test split (fixed seed 42),
prints them (including the confusion matrix for the band classifier), then
refits on all rows and writes `outputs/models/`.

## Run the app

```bash
python app.py
```

Then open `http://127.0.0.1:5000/`.

## Data source

https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset/data

## Limitations and ethical considerations

- Trained on a **high-school** dataset; it is **not** a validated predictor
  of university CGPA and must not be used for admissions, scholarships, or
  any real decision about a student (also shown as a disclaimer in the UI).
- The model finds correlations in past data, not causes. The improvement
  suggestions describe what has historically gone with a higher GPA, not
  a guarantee that changing one habit will raise any individual's grade.
- `Age`, `ParentalEducation`, and `ParentalSupport` are deliberately left
  out of the improvement suggestions in `advisor.py`: they correlate with
  GPA in the data, but a student cannot act on their own age or their
  parents' education or support level, so suggesting it would just be
  unhelpful (and a little tone-deaf).
- `GPA`, `GradeClass`, `StudentID`, `Gender`, and `Ethnicity` are never
  accepted as inputs (enforced in `validate_input`), so the deployed form
  cannot be used to probe the model by sensitive or identifying fields.



