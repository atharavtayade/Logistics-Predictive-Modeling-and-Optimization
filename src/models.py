"""
Logistics Predictive Modeling and Optimization Pipeline
Module: Model Selection, Cross-Validation & Hyperparameter Optimization
Reference: LOG-ML-OPT-2026-T4
Author: Logistics Data Analyst Intern & Systems Engineer
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import norm

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes RMSE, MAE, R-squared, Median Absolute Error (MedAE), and MAPE.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-4))) * 100.0)

    return {
        "RMSE (hours)": round(rmse, 4),
        "MAE (hours)": round(mae, 4),
        "R2 Score": round(r2, 4),
        "MedAE (hours)": round(medae, 4),
        "MAPE (%)": round(mape, 2)
    }

def calculate_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Computes ROC-AUC, Brier score, F1-Score, Precision, and Recall for SLA Breach Risk.
    """
    y_pred = (y_prob >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "ROC-AUC": round(auc, 4),
        "Brier Score": round(brier, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4)
    }

def train_and_evaluate_cv(
    models_dict: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    sla_targets: np.ndarray,
    sla_actual_breach: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Performs 5-Fold Cross Validation across all model tiers.
    Evaluates both continuous transit hours forecasting and downstream SLA breach classification.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []
    trained_artifacts = {}

    for name, model_factory in models_dict.items():
        fold_reg_metrics = []
        oof_preds = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_factory()
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            oof_preds[val_idx] = val_preds

            m = calculate_regression_metrics(y_val, val_preds)
            fold_reg_metrics.append(m)

        # Aggregate Cross-Validation Regression Performance
        avg_rmse = np.mean([m["RMSE (hours)"] for m in fold_reg_metrics])
        avg_mae = np.mean([m["MAE (hours)"] for m in fold_reg_metrics])
        avg_r2 = np.mean([m["R2 Score"] for m in fold_reg_metrics])
        avg_medae = np.mean([m["MedAE (hours)"] for m in fold_reg_metrics])
        avg_mape = np.mean([m["MAPE (%)"] for m in fold_reg_metrics])

        # SLA Risk Derivation:
        # Residual variance sigma estimated from OOF errors
        oof_residuals = y - oof_preds
        sigma_est = np.std(oof_residuals)

        # Probabilistic SLA Breach Risk: P(Transit > SLA) = 1 - Phi((SLA - y_pred) / sigma)
        oof_z_scores = (sla_targets - oof_preds) / (sigma_est + 1e-6)
        oof_breach_probs = 1.0 - norm.cdf(oof_z_scores)

        clf_metrics = calculate_classification_metrics(sla_actual_breach, oof_breach_probs)

        # Refit model on full dataset for downstream production deployment
        full_model = model_factory()
        full_model.fit(X, y)

        trained_artifacts[name] = {
            "model": full_model,
            "oof_preds": oof_preds,
            "oof_residuals": oof_residuals,
            "sigma_est": sigma_est,
            "oof_breach_probs": oof_breach_probs
        }

        results.append({
            "Model Tier": name,
            "CV RMSE (hrs)": round(avg_rmse, 3),
            "CV MAE (hrs)": round(avg_mae, 3),
            "CV R2": round(avg_r2, 4),
            "CV MedAE (hrs)": round(avg_medae, 3),
            "CV MAPE (%)": round(avg_mape, 2),
            "SLA ROC-AUC": clf_metrics["ROC-AUC"],
            "SLA Brier Score": clf_metrics["Brier Score"],
            "SLA F1-Score": clf_metrics["F1-Score"]
        })

    summary_df = pd.DataFrame(results).sort_values(by="CV RMSE (hrs)")
    return summary_df, trained_artifacts
