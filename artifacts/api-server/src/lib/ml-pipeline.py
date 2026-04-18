#!/usr/bin/env python3
"""
AutoML Studio - ML Pipeline Service
A Flask-based microservice that handles dataset download, cleaning, model training, and selection.
Communicates with the Node.js API server via HTTP.
"""

import os
import sys
import json
import uuid
import time
import threading
import traceback
import zipfile
import glob
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR

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


def run_pipeline_async(job_id: str, dataset_name: str, target_column: Optional[str],
                       kaggle_username: Optional[str], kaggle_key: Optional[str]):
    try:
        update_job(job_id, status="running", stage="Downloading dataset...")

        # Set Kaggle credentials
        env = os.environ.copy()
        if kaggle_username and kaggle_key:
            env["KAGGLE_USERNAME"] = kaggle_username
            env["KAGGLE_KEY"] = kaggle_key

        # Download dataset
        dataset_dir = DATASETS_DIR / job_id
        dataset_dir.mkdir(exist_ok=True)

        # Determine if it looks like owner/dataset format or just a dataset name
        if "/" in dataset_name:
            kaggle_ref = dataset_name
        else:
            kaggle_ref = dataset_name

        try:
            # Try as a competition dataset first
            result = subprocess.run(
                ["kaggle", "competitions", "download", "-c", dataset_name.split("/")[-1],
                 "-p", str(dataset_dir)],
                capture_output=True, text=True, timeout=120, env=env
            )
            if result.returncode != 0:
                raise Exception("Not a competition dataset")
        except Exception:
            # Try as a regular dataset
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", kaggle_ref,
                 "-p", str(dataset_dir), "--unzip"],
                capture_output=True, text=True, timeout=120, env=env
            )
            if result.returncode != 0:
                raise Exception(f"Failed to download dataset '{dataset_name}'. "
                                f"Make sure KAGGLE_USERNAME and KAGGLE_KEY are set, and the dataset name is correct. "
                                f"Error: {result.stderr}")

        # Extract zip if present
        zip_files = list(dataset_dir.glob("*.zip"))
        for zf in zip_files:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(dataset_dir)
            zf.unlink()

        update_job(job_id, stage="Loading dataset...")

        # Find CSV file
        csv_files = list(dataset_dir.rglob("*.csv"))
        if not csv_files:
            raise Exception("No CSV files found in the downloaded dataset.")

        # Pick the largest CSV (most likely to be the main dataset)
        csv_file = max(csv_files, key=lambda f: f.stat().st_size)

        df = pd.read_csv(csv_file)

        if df.empty:
            raise Exception("The dataset is empty.")
        if len(df) < 10:
            raise Exception("The dataset has fewer than 10 rows — too small to train models.")

        update_job(job_id, stage="Cleaning data...")

        # Track initial stats
        initial_rows = len(df)
        initial_cols = len(df.columns)

        # Remove duplicates
        df_before_dedup = len(df)
        df = df.drop_duplicates()
        duplicates_removed = df_before_dedup - len(df)

        # Count missing values before cleaning
        missing_count = int(df.isnull().sum().sum())

        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown")

        # Auto-detect target column
        if not target_column:
            # Heuristic: look for common target column names
            candidate_names = ["target", "label", "class", "outcome", "survived", "price",
                               "y", "output", "result", "status", "category", "type",
                               "quality", "score", "rating", "prediction", "diagnosis",
                               "fraud", "default", "churn", "click", "conversion"]
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
                # Use the last column as target
                target_column = df.columns[-1]

        if target_column not in df.columns:
            raise Exception(f"Target column '{target_column}' not found in dataset. "
                            f"Available columns: {', '.join(df.columns.tolist())}")

        update_job(job_id, stage="Detecting problem type...", targetColumn=target_column)

        # Detect problem type
        target = df[target_column]
        n_unique = target.nunique()
        problem_type = "classification"

        if target.dtype in [np.float64, np.float32]:
            if n_unique > 20:
                problem_type = "regression"
        elif target.dtype in [np.int64, np.int32]:
            if n_unique > 20:
                problem_type = "regression"

        update_job(job_id, stage=f"Encoding features for {problem_type}...")

        # Prepare features
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Drop columns with too many unique values (IDs) or all-null
        for col in X.columns.tolist():
            if X[col].nunique() > 0.95 * len(X) and X[col].dtype == object:
                X = X.drop(columns=[col])
            elif X[col].isnull().all():
                X = X.drop(columns=[col])

        # Encode categorical columns
        feature_cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        le = LabelEncoder()
        for col in feature_cat_cols:
            X[col] = le.fit_transform(X[col].astype(str))

        # Encode target for classification if it's string
        if problem_type == "classification" and y.dtype == object:
            y = le.fit_transform(y.astype(str))

        # Convert to numeric, drop any remaining non-numeric
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(0)

        feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        update_job(job_id, stage="Training models...")

        # Define models
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

                if problem_type == "classification":
                    y_pred = model.predict(X_test)
                    score = float(accuracy_score(y_test, y_pred))
                    # Higher is better for accuracy
                    if best_score is None or score > best_score:
                        best_score = score
                        best_model_name = model_name
                        best_model_obj = model
                else:
                    y_pred = model.predict(X_test)
                    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    score = rmse
                    # Lower is better for RMSE
                    if best_score is None or score < best_score:
                        best_score = score
                        best_model_name = model_name
                        best_model_obj = model

                model_scores.append({
                    "modelName": model_name,
                    "score": round(score, 6),
                    "trainingTime": round(training_time, 3),
                })
            except Exception as e:
                model_scores.append({
                    "modelName": model_name,
                    "score": 0.0,
                    "trainingTime": 0.0,
                })

        update_job(job_id, stage="Extracting feature importance...")

        # Feature importance
        feature_importance = []
        if best_model_obj is not None and hasattr(best_model_obj, "feature_importances_"):
            importances = best_model_obj.feature_importances_
            fi_pairs = list(zip(feature_names, importances))
            fi_pairs.sort(key=lambda x: x[1], reverse=True)
            feature_importance = [
                {"feature": str(f), "importance": round(float(i), 6)}
                for f, i in fi_pairs[:20]  # top 20 features
            ]
        elif best_model_obj is not None and hasattr(best_model_obj, "coef_"):
            coefs = best_model_obj.coef_
            if len(coefs.shape) > 1:
                coefs = np.abs(coefs).mean(axis=0)
            else:
                coefs = np.abs(coefs)
            fi_pairs = list(zip(feature_names, coefs))
            fi_pairs.sort(key=lambda x: x[1], reverse=True)
            feature_importance = [
                {"feature": str(f), "importance": round(float(i), 6)}
                for f, i in fi_pairs[:20]
            ]

        update_job(job_id, stage="Saving best model...")

        # Save best model
        model_path = None
        if best_model_obj is not None:
            model_path = str(MODELS_DIR / f"{job_id}_best_model.joblib")
            joblib.dump(best_model_obj, model_path)

        # Dataset summary
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
            "modelScores": model_scores,
            "featureImportance": feature_importance,
            "datasetSummary": dataset_summary,
            "modelPath": model_path,
        }

        update_job(job_id, status="completed", stage="Pipeline complete!")

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        update_job(job_id, status="failed", stage="Failed", error=error_msg)


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
        kaggle_username = os.environ.get("KAGGLE_USERNAME")
        kaggle_key = os.environ.get("KAGGLE_KEY")

        env = os.environ.copy()
        if kaggle_username:
            env["KAGGLE_USERNAME"] = kaggle_username
        if kaggle_key:
            env["KAGGLE_KEY"] = kaggle_key

        # Try competitions search first
        datasets = []

        result = subprocess.run(
            ["kaggle", "datasets", "list", "--search", query, "--csv", "--max-size", "5000000"],
            capture_output=True, text=True, timeout=30, env=env
        )

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                for line in lines[1:11]:  # Skip header, take up to 10 results
                    parts = line.split(",")
                    if len(parts) >= 4:
                        datasets.append({
                            "ref": parts[0].strip(),
                            "title": parts[1].strip().strip('"'),
                            "size": parts[2].strip() if len(parts) > 2 else None,
                            "lastUpdated": parts[3].strip() if len(parts) > 3 else None,
                            "downloadCount": None,
                        })

        if not datasets:
            # Try competitions
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
        return jsonify({"error": "Kaggle search timed out. Check your internet connection."}), 500
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("ML_SERVICE_PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
