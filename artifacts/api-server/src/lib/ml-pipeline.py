#!/usr/bin/env python3
"""
AutoML Studio - ML Pipeline Service
A Flask-based microservice that handles dataset download, cleaning, model training, and selection.
Also provides prediction, explanation, and confidence scoring on trained models.
"""

import os
import sys
import json
import uuid
import time
import threading
import traceback
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

app = Flask(__name__)
CORS(app)

JOBS: Dict[str, Dict[str, Any]] = {}
RESULTS: Dict[str, Dict[str, Any]] = {}

WORK_DIR = Path("/tmp/automl-studio")
WORK_DIR.mkdir(exist_ok=True)
MODELS_DIR = WORK_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
DATASETS_DIR = WORK_DIR / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)


def update_job(job_id: str, **kwargs):
    JOBS[job_id].update(kwargs)
    JOBS[job_id]["updatedAt"] = datetime.utcnow().isoformat() + "Z"


# ─────────────────────────────────────────────
#  PREDICT / EXPLAIN / CONFIDENCE helpers
# ─────────────────────────────────────────────

def predict(artifact: Dict, feature_values: Dict[str, str]) -> Any:
    """Run model.predict() on user-supplied feature values."""
    model = artifact["model"]
    feature_names = artifact["feature_names"]

    row = []
    for name in feature_names:
        raw = feature_values.get(name, "0")
        try:
            row.append(float(raw))
        except (ValueError, TypeError):
            # Treat non-numeric as 0 (already encoded during training)
            row.append(0.0)

    X_input = np.array(row).reshape(1, -1)
    pred = model.predict(X_input)
    return pred[0]


def explain(artifact: Dict, top_n: int = 5) -> List[Dict[str, Any]]:
    """Return top N important features from the saved model."""
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    top_features = []

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        top_features = [
            {"feature": str(f), "importance": round(float(i), 6)}
            for f, i in pairs[:top_n]
        ]
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        if len(coefs.shape) > 1:
            coefs = np.abs(coefs).mean(axis=0)
        else:
            coefs = np.abs(coefs)
        pairs = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)
        top_features = [
            {"feature": str(f), "importance": round(float(i), 6)}
            for f, i in pairs[:top_n]
        ]

    return top_features


def compute_confidence(result: Dict) -> Dict[str, Any]:
    """
    Compute confidence level:
    - Classification: based on best accuracy score
      >= 0.90 → High, >= 0.75 → Medium, else → Low
    - Regression: based on RMSE
      < 30000 → High, < 70000 → Medium, else → Low
    """
    problem_type = result.get("problemType", "classification")
    best_score = result.get("bestScore", 0)

    if problem_type == "classification":
        if best_score >= 0.90:
            level = "High"
        elif best_score >= 0.75:
            level = "Medium"
        else:
            level = "Low"
        detail = f"Accuracy: {best_score:.4f}"
    else:
        rmse = best_score
        if rmse < 30000:
            level = "High"
        elif rmse < 70000:
            level = "Medium"
        else:
            level = "Low"
        detail = f"RMSE: {rmse:.2f}"

    return {"confidence": level, "confidenceDetail": detail}


# ─────────────────────────────────────────────
#  Pipeline runner
# ─────────────────────────────────────────────

def run_pipeline_async(job_id: str, dataset_name: str, target_column: Optional[str],
                       kaggle_username: Optional[str], kaggle_key: Optional[str]):
    try:
        update_job(job_id, status="running", stage="Downloading dataset...")

        env = os.environ.copy()
        if kaggle_username and kaggle_key:
            env["KAGGLE_USERNAME"] = kaggle_username
            env["KAGGLE_KEY"] = kaggle_key

        dataset_dir = DATASETS_DIR / job_id
        dataset_dir.mkdir(exist_ok=True)

        try:
            result = subprocess.run(
                ["kaggle", "competitions", "download", "-c", dataset_name.split("/")[-1],
                 "-p", str(dataset_dir)],
                capture_output=True, text=True, timeout=120, env=env
            )
            if result.returncode != 0:
                raise Exception("Not a competition dataset")
        except Exception:
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset_name,
                 "-p", str(dataset_dir), "--unzip"],
                capture_output=True, text=True, timeout=120, env=env
            )
            if result.returncode != 0:
                raise Exception(
                    f"Failed to download dataset '{dataset_name}'. "
                    f"Make sure KAGGLE_USERNAME and KAGGLE_KEY are set and the dataset name is correct. "
                    f"Error: {result.stderr}"
                )

        # Extract any zips that weren't unzipped
        for zf in dataset_dir.glob("*.zip"):
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(dataset_dir)
            zf.unlink()

        update_job(job_id, stage="Loading dataset...")

        csv_files = list(dataset_dir.rglob("*.csv"))
        if not csv_files:
            raise Exception("No CSV files found in the downloaded dataset.")

        csv_file = max(csv_files, key=lambda f: f.stat().st_size)
        df = pd.read_csv(csv_file)

        if df.empty:
            raise Exception("The dataset is empty.")
        if len(df) < 10:
            raise Exception("The dataset has fewer than 10 rows — too small to train models.")

        update_job(job_id, stage="Cleaning data...")

        initial_rows = len(df)
        initial_cols = len(df.columns)

        df_before_dedup = len(df)
        df = df.drop_duplicates()
        duplicates_removed = df_before_dedup - len(df)

        missing_count = int(df.isnull().sum().sum())

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown")

        # Auto-detect target column
        if not target_column:
            candidate_names = [
                "target", "label", "class", "outcome", "survived", "price",
                "y", "output", "result", "status", "category", "type",
                "quality", "score", "rating", "prediction", "diagnosis",
                "fraud", "default", "churn", "click", "conversion"
            ]
            found = False
            for name in candidate_names:
                for col in df.columns:
                    if col.lower() == name.lower():
                        target_column = col
                        found = True
                        break
                if found:
                    break
            if not found:
                target_column = df.columns[-1]

        if target_column not in df.columns:
            raise Exception(
                f"Target column '{target_column}' not found. "
                f"Available columns: {', '.join(df.columns.tolist())}"
            )

        update_job(job_id, stage="Detecting problem type...", targetColumn=target_column)

        target = df[target_column]
        n_unique = target.nunique()
        problem_type = "classification"
        if target.dtype in [np.float64, np.float32] and n_unique > 20:
            problem_type = "regression"
        elif target.dtype in [np.int64, np.int32] and n_unique > 20:
            problem_type = "regression"

        update_job(job_id, stage=f"Encoding features for {problem_type}...")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        for col in X.columns.tolist():
            if X[col].nunique() > 0.95 * len(X) and X[col].dtype == object:
                X = X.drop(columns=[col])
            elif X[col].isnull().all():
                X = X.drop(columns=[col])

        le = LabelEncoder()
        feature_cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in feature_cat_cols:
            X[col] = le.fit_transform(X[col].astype(str))

        if problem_type == "classification" and y.dtype == object:
            y = le.fit_transform(y.astype(str))

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        update_job(job_id, stage="Training models...")

        if problem_type == "classification":
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
            }
            metric = "accuracy"
        else:
            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "Ridge Regression": Ridge(random_state=42),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
            }
            metric = "RMSE"

        model_scores = []
        best_model_name = None
        best_score = None
        best_model_obj = None

        for model_name, model in models.items():
            try:
                update_job(job_id, stage=f"Training {model_name}...")
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                y_pred = model.predict(X_test)

                if problem_type == "classification":
                    score = float(accuracy_score(y_test, y_pred))
                    if best_score is None or score > best_score:
                        best_score = score
                        best_model_name = model_name
                        best_model_obj = model
                else:
                    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    score = rmse
                    if best_score is None or score < best_score:
                        best_score = score
                        best_model_name = model_name
                        best_model_obj = model

                model_scores.append({
                    "modelName": model_name,
                    "score": round(score, 6),
                    "trainingTime": round(training_time, 3),
                })
            except Exception:
                model_scores.append({"modelName": model_name, "score": 0.0, "trainingTime": 0.0})

        update_job(job_id, stage="Computing evaluation metrics...")

        # ── Additional metrics on best model ──
        best_precision = None
        best_recall = None
        best_f1 = None
        best_mae = None
        best_confusion = None
        class_imbalance_warning = None

        if best_model_obj is not None:
            y_best_pred = best_model_obj.predict(X_test)
            if problem_type == "classification":
                try:
                    best_precision = round(float(precision_score(y_test, y_best_pred, average="weighted", zero_division=0)), 6)
                    best_recall = round(float(recall_score(y_test, y_best_pred, average="weighted", zero_division=0)), 6)
                    best_f1 = round(float(f1_score(y_test, y_best_pred, average="weighted", zero_division=0)), 6)
                    cm = confusion_matrix(y_test, y_best_pred)
                    best_confusion = cm.tolist()
                except Exception:
                    pass
                # Detect class imbalance on full target column
                try:
                    value_counts = pd.Series(y).value_counts(normalize=True)
                    min_frac = float(value_counts.min())
                    if min_frac < 0.10:
                        class_imbalance_warning = (
                            "Imbalanced dataset detected. Accuracy may be misleading. "
                            f"Smallest class is only {min_frac*100:.1f}% of data. "
                            "F1 score is a better indicator of model quality."
                        )
                except Exception:
                    pass
            else:
                try:
                    best_mae = round(float(mean_absolute_error(y_test, y_best_pred)), 6)
                except Exception:
                    pass

        update_job(job_id, stage="Extracting feature importance...")

        feature_importance = []
        if best_model_obj is not None and hasattr(best_model_obj, "feature_importances_"):
            importances = best_model_obj.feature_importances_
            fi_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            feature_importance = [
                {"feature": str(f), "importance": round(float(i), 6)}
                for f, i in fi_pairs[:20]
            ]
        elif best_model_obj is not None and hasattr(best_model_obj, "coef_"):
            coefs = best_model_obj.coef_
            if len(coefs.shape) > 1:
                coefs = np.abs(coefs).mean(axis=0)
            else:
                coefs = np.abs(coefs)
            fi_pairs = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)
            feature_importance = [
                {"feature": str(f), "importance": round(float(i), 6)}
                for f, i in fi_pairs[:20]
            ]

        update_job(job_id, stage="Saving best model...")

        model_path = None
        if best_model_obj is not None:
            model_path = str(MODELS_DIR / f"{job_id}_best_model.joblib")
            # Save artifact dict with model + feature names for prediction
            joblib.dump({"model": best_model_obj, "feature_names": feature_names}, model_path)

        dataset_summary = {
            "rows": len(df),
            "columns": initial_cols,
            "columnNames": df.columns.tolist(),
            "missingValues": missing_count,
            "duplicatesRemoved": duplicates_removed,
            "categoricalColumns": categorical_cols,
            "numericColumns": numeric_cols,
        }

        RESULTS[job_id] = {
            "jobId": job_id,
            "datasetName": dataset_name,
            "targetColumn": target_column,
            "problemType": problem_type,
            "bestModel": best_model_name,
            "bestScore": round(best_score, 6) if best_score is not None else 0.0,
            "scoreMetric": metric,
            "precision": best_precision,
            "recall": best_recall,
            "f1Score": best_f1,
            "mae": best_mae,
            "confusionMatrix": best_confusion,
            "classImbalanceWarning": class_imbalance_warning,
            "modelScores": model_scores,
            "featureImportance": feature_importance,
            "featureNames": feature_names,
            "datasetSummary": dataset_summary,
            "modelPath": model_path,
        }

        update_job(job_id, status="completed", stage="Pipeline complete!")

    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        update_job(job_id, status="failed", stage="Failed", error=error_msg)


# ─────────────────────────────────────────────
#  Flask routes
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/pipeline/run", methods=["POST"])
def run_pipeline():
    data = request.get_json()
    if not data or not data.get("datasetName"):
        return jsonify({"error": "datasetName is required"}), 400

    dataset_name = data["datasetName"].strip()
    target_column = data.get("targetColumn") or None
    kaggle_username = data.get("kaggleUsername") or os.environ.get("KAGGLE_USERNAME")
    kaggle_key = data.get("kaggleKey") or os.environ.get("KAGGLE_KEY")

    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    JOBS[job_id] = {
        "jobId": job_id,
        "status": "pending",
        "stage": "Initializing...",
        "datasetName": dataset_name,
        "targetColumn": target_column,
        "createdAt": now,
        "updatedAt": now,
        "error": None,
    }

    t = threading.Thread(
        target=run_pipeline_async,
        args=(job_id, dataset_name, target_column, kaggle_username, kaggle_key),
        daemon=True
    )
    t.start()

    return jsonify(JOBS[job_id]), 202


@app.route("/pipeline/status/<job_id>", methods=["GET"])
def get_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/pipeline/result/<job_id>", methods=["GET"])
def get_result(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "completed":
        return jsonify({"error": f"Job is not complete. Status: {job['status']}"}), 400
    result = RESULTS.get(job_id)
    if not result:
        return jsonify({"error": "Result not found"}), 404
    return jsonify(result)


@app.route("/pipeline/predict/<job_id>", methods=["POST"])
def predict_endpoint(job_id):
    """Run prediction + explanation + confidence on a completed job's best model."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "completed":
        return jsonify({"error": f"Job is not complete. Status: {job['status']}"}), 400

    result = RESULTS.get(job_id)
    if not result or not result.get("modelPath"):
        return jsonify({"error": "No saved model found for this job"}), 400

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Request body must include 'features' object"}), 400

    feature_values = data["features"]
    if not isinstance(feature_values, dict):
        return jsonify({"error": "'features' must be an object mapping feature names to values"}), 400

    try:
        artifact = joblib.load(result["modelPath"])
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

    try:
        raw_pred = predict(artifact, feature_values)
        predicted_value = str(raw_pred)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    top_features = explain(artifact, top_n=5)

    # Generate human-readable explanation
    if top_features:
        top_names = [f["feature"] for f in top_features[:3]]
        if len(top_names) == 1:
            feature_str = top_names[0]
        elif len(top_names) == 2:
            feature_str = f"{top_names[0]} and {top_names[1]}"
        else:
            feature_str = f"{top_names[0]}, {top_names[1]}, and {top_names[2]}"
        explanation = f"This prediction is mainly influenced by {feature_str}."
    else:
        explanation = "Feature importance is not available for this model type."

    confidence_data = compute_confidence(result)

    return jsonify({
        "predictedValue": predicted_value,
        "topFeatures": top_features,
        "explanation": explanation,
        "confidence": confidence_data["confidence"],
        "confidenceDetail": confidence_data["confidenceDetail"],
        "problemType": result["problemType"],
    })


@app.route("/pipeline/jobs", methods=["GET"])
def list_jobs():
    jobs = sorted(JOBS.values(), key=lambda j: j["createdAt"], reverse=True)
    return jsonify(jobs)


@app.route("/kaggle/search", methods=["GET"])
def search_kaggle():
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        env = os.environ.copy()
        datasets = []

        result = subprocess.run(
            ["kaggle", "datasets", "list", "--search", query, "--csv", "--max-size", "5000000"],
            capture_output=True, text=True, timeout=30, env=env
        )

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                for line in lines[1:11]:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        datasets.append({
                            "ref": parts[0].strip(),
                            "title": parts[1].strip().strip('"'),
                            "size": parts[2].strip() if len(parts) > 2 else None,
                            "lastUpdated": parts[3].strip() if len(parts) > 3 else None,
                            "downloadCount": None,
                        })

        if not datasets:
            result = subprocess.run(
                ["kaggle", "competitions", "list", "--search", query, "--csv"],
                capture_output=True, text=True, timeout=30, env=env
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    for line in lines[1:6]:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            ref = parts[0].strip()
                            datasets.append({
                                "ref": ref,
                                "title": parts[1].strip().strip('"') if len(parts) > 1 else ref,
                                "size": None,
                                "lastUpdated": None,
                                "downloadCount": None,
                            })

        return jsonify(datasets)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Kaggle search timed out."}), 500
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("ML_SERVICE_PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
