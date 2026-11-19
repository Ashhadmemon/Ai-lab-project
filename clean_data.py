import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
TARGET, CLASS_TARGET, ID_COL = "GPA", "GradeClass", "StudentID"
NUMERIC = ["Age", "StudyTimeWeekly", "Absences", "GPA"]
CATEGORICAL = ["Gender", "Ethnicity", "ParentalEducation", "ParentalSupport",
               "Tutoring", "Extracurricular", "Sports", "Music", "Volunteering",
               "GradeClass"]
BINARY = ["Gender", "Tutoring", "Extracurricular", "Sports", "Music", "Volunteering"]
REQUIRED = [ID_COL, "Age", "Gender", "Ethnicity", "ParentalEducation",
            "StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport",
            "Extracurricular", "Sports", "Music", "Volunteering", TARGET, CLASS_TARGET]
RANGES = {
    "Age": (15, 18), "Ethnicity": (0, 3), "ParentalEducation": (0, 4),
    "StudyTimeWeekly": (0, 20), "Absences": (0, 30), "ParentalSupport": (0, 4),
    "GPA": (0, 4), "GradeClass": (0, 4),
}
INTEGER_COLS = ["Age", "Ethnicity", "ParentalEducation", "Absences",
                "ParentalSupport", "GradeClass"] + BINARY
GRADE_NAMES = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}


class DataValidationError(Exception):
    """Raised when the CSV is missing, empty or lacks required columns."""


class Log:
    """Print messages and keep them for cleaning_log.txt."""

    def __init__(self):
        self.lines = []

    def __call__(self, msg=""):
        print(msg)
        self.lines.append(str(msg))


def load_raw(path: Path) -> pd.DataFrame:
    """Read the CSV and check that all required columns exist."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise DataValidationError(f"CSV not found: {path}") from None
    except pd.errors.EmptyDataError:
        raise DataValidationError(f"CSV is empty: {path}") from None
    except UnicodeDecodeError:
        raise DataValidationError(f"Cannot decode {path}; save it as UTF-8") from None
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    return df



def audit(df: pd.DataFrame, res: Path, log: Log) -> None:
    """Write dataset summary, missing-value and duplicate reports (raw data)."""
    log("=== AUDIT OF RAW DATA ===")
    log(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    miss = pd.DataFrame({"missing": df.isna().sum(),
                         "percent": (df.isna().mean() * 100).round(2)})
    miss.to_csv(res / "missing_values.csv", index_label="column")
    log(f"Missing cells: {int(df.isna().sum().sum())}")

    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df[ID_COL].duplicated().sum())
    expected_ids = set(range(int(df[ID_COL].min()), int(df[ID_COL].max()) + 1))
    gaps = len(expected_ids - set(df[ID_COL]))
    text = (f"Duplicate rows: {dup_rows}\nDuplicate {ID_COL} values: {dup_ids}\n"
            f"{ID_COL} range: {df[ID_COL].min()} to {df[ID_COL].max()} "
            f"({gaps} IDs absent from the range)\n")
    (res / "duplicate_check.txt").write_text(text)
    log(text.strip())

    buf = [f"Shape: {df.shape}", "", "Data types:", df.dtypes.to_string(), "",
           "Numeric summary:", df.describe().T.round(3).to_string(), "",
           "Value counts for coded columns:"]
    for col in CATEGORICAL:
        buf += [f"\n{col}:", df[col].value_counts(dropna=False).sort_index().to_string()]
    (res / "dataset_summary.txt").write_text("\n".join(buf))


# ---------------------------------------------------------------- cleaning
def validation_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count impossible values per column (documented ranges and 0/1 flags)."""
    rows = []
    for col, (lo, hi) in RANGES.items():
        rows.append({"column": col, "rule": f"{lo} <= value <= {hi}",
                     "violations": int(((df[col] < lo) | (df[col] > hi)).sum())})
    for col in BINARY:
        rows.append({"column": col, "rule": "value in {0, 1}",
                     "violations": int((~df[col].isin([0, 1]) & df[col].notna()).sum())})
    return pd.DataFrame(rows)


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """IQR (1.5 x) outlier counts for the continuous columns."""
    rows = []
    for col in ["StudyTimeWeekly", "Absences", TARGET]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        rows.append({"column": col, "q1": round(q1, 3), "q3": round(q3, 3),
                     "lower_fence": round(lo, 3), "upper_fence": round(hi, 3),
                     "outliers": n})
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame, outliers: str, res: Path, log: Log) -> pd.DataFrame:
    """Clean the data step by step, logging every decision."""
    log("\n=== CLEANING ===")
    out = df.copy()
    start = len(out)

    # 1. Text/whitespace in non-numeric cells becomes NaN so it is handled below.
    for col in REQUIRED:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # 2. Duplicates
    before = len(out)
    out = out.drop_duplicates().drop_duplicates(subset=ID_COL, keep="first")
    log(f"Duplicates removed: {before - len(out)}")

    # 3. Missing values: rows without an ID or target are dropped; features are
    #    imputed (median for continuous/ordinal, mode for binary flags).
    before = len(out)
    out = out.dropna(subset=[ID_COL, TARGET, CLASS_TARGET])
    log(f"Rows dropped for missing ID/GPA/GradeClass: {before - len(out)}")
    for col in REQUIRED:
        n = int(out[col].isna().sum())
        if n:
            fill = out[col].mode().iloc[0] if col in BINARY else out[col].median()
            out[col] = out[col].fillna(fill)
            log(f"Imputed {n} missing values in {col} with {fill}")

    # 4. Impossible values (outside documented ranges, or flags not in {0,1})
    vr = validation_report(out)
    vr.to_csv(res / "validation_report.csv", index=False)
    bad = pd.Series(False, index=out.index)
    for col, (lo, hi) in RANGES.items():
        bad |= (out[col] < lo) | (out[col] > hi)
    for col in BINARY:
        bad |= ~out[col].isin([0, 1])
    log(f"Rows with impossible values removed: {int(bad.sum())}")
    out = out[~bad]

    # 5. Outliers (valid values far from the bulk). Default: keep, because the
    #    values are inside the documented range and so are real students.
    orp = outlier_report(out)
    orp.to_csv(res / "outlier_report.csv", index=False)
    log("IQR outliers found: " + ", ".join(
        f"{r.column}={r.outliers}" for r in orp.itertuples()))
    if outliers != "keep" and orp["outliers"].sum() > 0:
        flagged = pd.Series(False, index=out.index)
        for r in orp.itertuples():
            lo, hi = r.lower_fence, r.upper_fence
            if outliers == "clip" and r.column != TARGET:
                out[r.column] = out[r.column].clip(lo, hi)
            flagged |= (out[r.column] < lo) | (out[r.column] > hi)
        if outliers == "drop":
            log(f"Outlier rows dropped: {int(flagged.sum())}")
            out = out[~flagged]
        else:
            log("Outliers in features clipped to the IQR fences (target left as is)")
    else:
        log(f"Outlier policy: {outliers} (no rows changed)")

    # 6. Types and ordering
    for col in INTEGER_COLS + [ID_COL]:
        out[col] = out[col].round().astype(int)
    out = out.sort_values(ID_COL).reset_index(drop=True)

    # 7. GradeClass / GPA consistency (reported, rows kept)
    expected = pd.cut(out[TARGET], [-0.001, 2.0, 2.5, 3.0, 3.5, 4.001],
                      right=False, labels=[4, 3, 2, 1, 0]).astype(int)
    match = float((expected == out[CLASS_TARGET]).mean())
    log(f"GradeClass matches the documented GPA bands in {match:.1%} of rows "
        f"(mismatches kept, not altered)")
    log(f"GPA observed range: {out[TARGET].min():.3f} to {out[TARGET].max():.3f}")
    log(f"Rows: {start} -> {len(out)}")
    out.attrs["band_match"] = match
    return out


# ---------------------------------------------------------------- EDA
def group_summaries(df: pd.DataFrame, res: Path) -> pd.DataFrame:
    """Mean GPA (and count) grouped by key behaviour and background columns."""
    d = df.copy()
    d["StudyTimeBand"] = pd.cut(d["StudyTimeWeekly"], [-0.01, 5, 10, 15, 20.01],
                                labels=["0-5h", "5-10h", "10-15h", "15-20h"])
    d["AbsencesBand"] = pd.cut(d["Absences"], [-1, 5, 10, 20, 30],
                               labels=["0-5", "6-10", "11-20", "21-30"])
    parts = []
    for col in ["StudyTimeBand", "AbsencesBand", "Tutoring", "ParentalSupport",
                "ParentalEducation", "Extracurricular", "Sports", "Music",
                "Volunteering", "Age"]:
        g = d.groupby(col, observed=True)[TARGET].agg(["count", "mean", "median"]).round(3)
        g.insert(0, "group", col)
        parts.append(g.reset_index().rename(columns={col: "level"}))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(res / "group_summaries.csv", index=False)
    return out


def make_figures(df: pd.DataFrame, fig: Path) -> pd.Series:
    """Save all EDA figures; return correlations with GPA."""
    # Target distribution
    f, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df[TARGET], bins=30, color="#2F5597", edgecolor="white")
    ax[0].axvline(df[TARGET].mean(), color="red", ls="--", label=f"mean {df[TARGET].mean():.2f}")
    ax[0].set(title="GPA distribution", xlabel="GPA (points)", ylabel="Students")
    ax[0].legend()
    ax[1].boxplot(df[TARGET])
    ax[1].set(title="GPA box plot", ylabel="GPA (points)")
    f.tight_layout(); f.savefig(fig / "target_distribution.png", dpi=130); plt.close(f)

    # Numeric distributions
    f, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, col, unit in zip(ax, ["Age", "StudyTimeWeekly", "Absences"],
                            ["years", "hours per week", "days"]):
        a.hist(df[col], bins=20 if col != "Age" else 4, color="#2F5597", edgecolor="white")
        a.set(title=f"{col} distribution", xlabel=f"{col} ({unit})", ylabel="Students")
    f.tight_layout(); f.savefig(fig / "numeric_distributions.png", dpi=130); plt.close(f)

    # Categorical counts
    cols = [c for c in CATEGORICAL]
    f, ax = plt.subplots(2, 5, figsize=(16, 7))
    for a, col in zip(ax.ravel(), cols):
        vc = df[col].value_counts().sort_index()
        a.bar(vc.index.astype(str), vc.values, color="#2F5597")
        a.set(title=col, xlabel="Code", ylabel="Students")
    f.tight_layout(); f.savefig(fig / "categorical_counts.png", dpi=130); plt.close(f)

    # Correlation heatmap (excluding the ID)
    corr = df.drop(columns=[ID_COL]).corr()
    f, a = plt.subplots(figsize=(9, 8))
    im = a.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    a.set_xticks(range(len(corr))); a.set_xticklabels(corr.columns, rotation=60, ha="right")
    a.set_yticks(range(len(corr))); a.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            a.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    f.colorbar(im, label="Pearson correlation")
    a.set_title("Correlation heatmap")
    f.tight_layout(); f.savefig(fig / "correlation_heatmap.png", dpi=130); plt.close(f)

    # Mean GPA by key groups
    f, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, col in zip(ax, ["ParentalSupport", "Tutoring", "Absences"]):
        g = df.groupby(pd.cut(df[col], [-1, 5, 10, 20, 30]) if col == "Absences" else col,
                       observed=True)[TARGET].mean()
        a.bar(g.index.astype(str), g.values, color="#2F5597")
        a.set(title=f"Mean GPA by {col}", xlabel=col, ylabel="Mean GPA (points)")
        a.tick_params(axis="x", rotation=30)
    f.tight_layout(); f.savefig(fig / "gpa_by_groups.png", dpi=130); plt.close(f)

    return corr[TARGET].drop([TARGET, CLASS_TARGET]).sort_values(key=abs, ascending=False)


def write_findings(df: pd.DataFrame, corr: pd.Series, groups: pd.DataFrame,
                   orp: pd.DataFrame, res: Path) -> None:
    """Write eda_findings.md; every number comes from this run."""
    cls = df[CLASS_TARGET].value_counts().sort_index()
    share = (cls / len(df) * 100).round(1)
    top = corr.head(3)
    sib = df[["Sports", "Extracurricular"]].corr().iloc[0, 1]
    lines = [
        "# EDA findings (generated automatically)", "",
        f"- **Size:** {len(df)} students, {df.shape[1]} columns after cleaning.",
        f"- **GPA range:** {df[TARGET].min():.2f} to {df[TARGET].max():.2f}; mean "
        f"{df[TARGET].mean():.2f}, median {df[TARGET].median():.2f}, std {df[TARGET].std():.2f}.",
        "- **Class balance (GradeClass, 0=A ... 4=F):** "
        + ", ".join(f"{GRADE_NAMES[k]}: {v} ({share[k]}%)" for k, v in cls.items()) + ".",
        f"- **GradeClass vs GPA bands:** {df.attrs.get('band_match', float('nan')):.1%} of rows "
        "match the documented thresholds; the rest were kept unchanged.",
        "- **Strongest linear relationships with GPA:** "
        + "; ".join(f"{k} ({v:+.2f})" for k, v in top.items()) + ".",
        f"- **Outliers (IQR rule):** " + ", ".join(f"{r.column}: {r.outliers}" for r in orp.itertuples()) + ".",
        f"- **Redundancy check:** correlation between Sports and Extracurricular is {sib:.2f}.",
        "", "## Group means of GPA", "",
    ]
    for g, sub in groups.groupby("group", sort=False):
        lines.append(f"- **{g}:** " + ", ".join(
            f"{r.level} -> {r.mean:.2f} (n={r.count})" for r in sub.itertuples()))
    lines += ["", "## Cautions", "",
              "- Associations only; no causal conclusion should be drawn.",
              "- Confirm whether the dataset is real or synthetic and record the source and license.",
              "- The description says GPA is 2.0-4.0; compare with the observed range above."]
    (res / "eda_findings.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="EDA + cleaning for the student CSV")
    ap.add_argument("--csv", default="data/raw/Student_performance_data.csv")
    ap.add_argument("--out", default="data/processed/students_clean.csv")
    ap.add_argument("--results", default="outputs/results")
    ap.add_argument("--figures", default="outputs/figures")
    ap.add_argument("--outliers", choices=["keep", "clip", "drop"], default="keep",
                    help="how to handle IQR outliers (default: keep)")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    res, fig, out_path = Path(args.results), Path(args.figures), Path(args.out)
    log = Log()

    try:
        for d in (res, fig, out_path.parent):
            d.mkdir(parents=True, exist_ok=True)
        raw = load_raw(Path(args.csv))
        audit(raw, res, log)
        df = clean(raw, args.outliers, res, log)
        df.to_csv(out_path, index=False)
        groups = group_summaries(df, res)
        corr = make_figures(df, fig)
        corr.to_csv(res / "correlations_with_gpa.csv", header=["correlation"], index_label="feature")
        write_findings(df, corr, groups, outlier_report(df), res)
    except DataValidationError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    except PermissionError as err:
        print(f"ERROR: cannot write output ({err})", file=sys.stderr)
        return 1

    log(f"\nCleaned data saved to {out_path}")
    log(f"Tables and findings in {res}/, figures in {fig}/")
    (res / "cleaning_log.txt").write_text("\n".join(log.lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())