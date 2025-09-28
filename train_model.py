# train_model.py
import argparse
import json
import os
import platform
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# optional imports
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

import warnings
warnings.filterwarnings("ignore")


# ---- JSON converter to handle numpy types ----
def json_converter(o):
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def load_data(csv_path, n_mfcc=22):
    df = pd.read_csv(csv_path)
    if "status" not in df.columns:
        raise ValueError("CSV must contain 'status' column as target")

    y_raw = df["status"].values
    X = df.drop(columns=["status"]).select_dtypes(include=[np.number])

    # Clean feature names
    X.columns = [
        col.replace(":", "_").replace("(", "_").replace(")", "_").replace("-", "_")
        for col in X.columns
    ]

    if X.shape[1] != n_mfcc:
        print(f"Warning: expected {n_mfcc} features (n_mfcc). Found {X.shape[1]} numeric features.")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, le


def build_models(random_state=42):
    models = {}

    # Random Forest
    rf = RandomForestClassifier(random_state=random_state)
    rf_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", rf)])
    rf_params = {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 5, 10]}
    models["rf"] = (rf_pipeline, rf_params)

    # SVM
    svm = SVC(probability=True, random_state=random_state)
    svm_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", svm)])
    svm_params = {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", "auto"],
    }
    models["svm"] = (svm_pipeline, svm_params)

    # XGBoost
    if xgb is not None:
        xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)
        xgb_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", xgb_clf)])
        xgb_params = {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5, 7]}
        models["xgb"] = (xgb_pipeline, xgb_params)

    # LightGBM
    if lgb is not None:
        lgb_clf = lgb.LGBMClassifier(random_state=random_state)
        lgb_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", lgb_clf)])
        lgb_params = {"clf__n_estimators": [100, 200], "clf__max_depth": [-1, 5, 10]}
        models["lgb"] = (lgb_pipeline, lgb_params)

    return models


def evaluate_and_save(gs, X_test, y_test, model_name, output_dir):
    y_pred = gs.predict(X_test)

    try:
        y_prob = gs.predict_proba(X_test)
    except Exception:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Confusion matrix plot
    try:
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title(f"Confusion Matrix - {model_name}")
        cm_path = os.path.join(output_dir, f"cm_{model_name}.png")
        plt.savefig(cm_path)
        plt.close()
    except Exception:
        cm_path = None

    # ROC curve
    roc_auc, roc_path = None, None
    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                y_prob_pos = y_prob[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob_pos)
                RocCurveDisplay.from_predictions(y_test, y_prob_pos)
            elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
                roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
                RocCurveDisplay.from_predictions(y_test, y_prob[:, 0])
            else:
                roc_auc = roc_auc_score(y_test, y_prob)
                RocCurveDisplay.from_predictions(y_test, y_prob)

            plt.title(f"ROC Curve - {model_name}")
            roc_path = os.path.join(output_dir, f"roc_{model_name}.png")
            plt.savefig(roc_path)
            plt.close()
        except Exception:
            roc_auc, roc_path = None, None

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "roc_auc": roc_auc,
        "cm_path": cm_path,
        "roc_path": roc_path,
        "best_params": gs.best_params_,
    }


def run_training(csv_path, model_choice, test_size=0.2, cv=5, random_state=42, n_jobs=-1, explain=False, n_mfcc=22):
    print("Loading dataset...")
    X, y, label_encoder = load_data(csv_path, n_mfcc=n_mfcc)
    print(f"Loaded data: X.shape={X.shape}, y_distribution=\n{np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Train/test split: {X_train.shape} / {X_test.shape}")

    models = build_models(random_state=random_state)

    if model_choice != "all":
        if model_choice not in models:
            raise ValueError(f"Unknown model: {model_choice}")
        selected_models = {model_choice: models[model_choice]}
    else:
        selected_models = models

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "model_reports"
    os.makedirs(output_dir, exist_ok=True)

    all_reports = {}
    all_model_paths = {}

    unique_labels = np.unique(y)
    default_scoring = "f1" if len(unique_labels) == 2 else "f1_macro"

    for name, (pipeline, param_grid) in selected_models.items():
        print(f"\n--- Training candidate: {name} ---")

        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        safe_jobs = 1 if platform.system() == "Windows" else n_jobs

        gs = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_strategy,
            scoring=default_scoring,
            n_jobs=safe_jobs,
            verbose=2,
        )

        gs.fit(X_train, y_train)
        report = evaluate_and_save(gs, X_test, y_test, name, output_dir)
        all_reports[name] = report

        # Save models
        model_path = os.path.join(output_dir, f"{name}_model.pkl")
        joblib.dump(gs.best_estimator_, model_path)
        all_model_paths[name] = model_path
        print(f"Saved {name} best model to {model_path}")

        canonical = os.path.join(output_dir, "voice_model.pkl")
        joblib.dump(gs.best_estimator_, canonical)
        print(f"Saved canonical voice_model to {canonical}")

    # Metadata
    metadata = {
        "feature_names": list(X.columns),
        "label_classes": list(label_encoder.classes_),
        "n_mfcc": n_mfcc,
        "training_time": timestamp,
        "reports": all_reports,
    }

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=json_converter)
    print(f"Saved metadata to {meta_path}")

    return all_model_paths, meta_path


def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Train Parkinson’s detection models")
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--model", type=str, default="all", help="Model to train: rf | svm | xgb | lgb | all")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--explain", action="store_true", help="Use SHAP explainability")
    parser.add_argument("--n_mfcc", type=int, default=22, help="MFCC count expected")

    args = parser.parse_args()
    print("Running training with args:", args)

    run_training(
        csv_path=args.csv,
        model_choice=args.model,
        test_size=args.test_size,
        cv=args.cv,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        explain=args.explain,
        n_mfcc=args.n_mfcc,
    )


if __name__ == "__main__":
    parse_args_and_run()
