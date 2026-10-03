"""
Logistics Predictive Modeling and Optimization Pipeline
Module: Feature Importance & Residual Diagnostics
Reference: LOG-ML-OPT-2026-T4
Author: Logistics Data Analyst Intern & Systems Engineer
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Any, Tuple
import shap

# Set high-grade publication aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#2c3e50"
plt.rcParams["axes.linewidth"] = 1.0

def compute_feature_importances(
    model: Any,
    feature_names: List[str],
    X_sample: np.ndarray = None,
    max_features: int = 15
) -> Tuple[pd.DataFrame, Any]:
    """
    Computes both SHAP values (via TreeExplainer) and Gini/Split feature importances.
    """
    importance_df = pd.DataFrame({"feature": feature_names})

    # Tree-based Gini / Split importance
    if hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
        # Normalize to percentage
        importance_df["gini_importance_pct"] = (raw_imp / np.sum(raw_imp)) * 100.0
    else:
        importance_df["gini_importance_pct"] = 0.0

    shap_values = None
    if X_sample is not None and len(X_sample) > 0:
        try:
            # TreeExplainer for LightGBM / XGBoost / RF
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_matrix = shap_values[0]
            else:
                shap_matrix = shap_values
            mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
            importance_df["mean_abs_shap"] = mean_abs_shap
            importance_df = importance_df.sort_values(by="mean_abs_shap", ascending=False)
        except Exception as e:
            print(f"SHAP explanation fallback: {e}")
            importance_df = importance_df.sort_values(by="gini_importance_pct", ascending=False)
    else:
        importance_df = importance_df.sort_values(by="gini_importance_pct", ascending=False)

    return importance_df.head(max_features), shap_values

def plot_feature_importance_and_shap(
    importance_df: pd.DataFrame,
    output_path: str = "figures/feature_importance.png"
):
    """
    Plots dual-bar chart showing Tree split importance and SHAP impact.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    y_pos = np.arange(len(importance_df))
    feat_labels = [f.replace("__", " - ").replace("_", " ").title() for f in importance_df["feature"]]

    if "mean_abs_shap" in importance_df.columns:
        bars = ax1.barh(y_pos, importance_df["mean_abs_shap"], color="#1f77b4", edgecolor="#0b3c61", alpha=0.88, height=0.65)
        ax1.set_xlabel("Mean |SHAP Value| (Impact on Transit Duration in Hours)", fontsize=11, fontweight="bold", color="#1a252f")
        ax1.set_title("Primary Transit Duration Variance Drivers (SHAP Game-Theoretic Attribution)", fontsize=13, fontweight="bold", pad=12)
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 0.05, bar.get_y() + bar.get_height()/2, f"{width:.2f}h", va='center', ha='left', fontsize=9, color="#2c3e50")
    else:
        bars = ax1.barh(y_pos, importance_df["gini_importance_pct"], color="#2ca02c", edgecolor="#145214", alpha=0.88, height=0.65)
        ax1.set_xlabel("Feature Split Importance (%)", fontsize=11, fontweight="bold", color="#1a252f")
        ax1.set_title("Gini Feature Importance Hierarchy", fontsize=13, fontweight="bold", pad=12)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feat_labels, fontsize=10)
    ax1.invert_yaxis()
    ax1.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "figures/residual_diagnostics.png"
) -> Dict[str, float]:
    """
    Generates a 4-panel publication-grade residual diagnostic grid:
    1. Predicted vs Actual Transit Hours (with 45-degree parity line)
    2. Residual Distribution vs Theoretical Gaussian (Skewness & Kurtosis)
    3. Residuals vs Predicted (Heteroscedasticity evaluation)
    4. Normal Q-Q Plot (Quantile-Quantile tail diagnostics)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    residuals = y_true - y_pred

    skewness = float(stats.skew(residuals))
    kurt = float(stats.kurtosis(residuals))

    # Breusch-Pagan / White Proxy Test for Heteroscedasticity:
    # Regress squared residuals e_i^2 on fitted values y_hat and y_hat^2
    res_sq = residuals ** 2
    poly_terms = np.column_stack([np.ones(len(y_pred)), y_pred, y_pred ** 2])
    beta_bp, _, _, _ = np.linalg.lstsq(poly_terms, res_sq, rcond=None)
    res_sq_pred = poly_terms @ beta_bp
    ss_total = np.sum((res_sq - np.mean(res_sq)) ** 2)
    ss_reg = np.sum((res_sq_pred - np.mean(res_sq)) ** 2)
    r2_bp = ss_reg / (ss_total + 1e-8)
    lm_stat = len(residuals) * r2_bp
    p_value_bp = float(1.0 - stats.chi2.cdf(lm_stat, df=2))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    # Panel 1: Observed vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.35, color="#2980b9", edgecolors="none", s=20)
    min_val, max_val = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal 1:1 Parity")
    ax1.set_xlabel("Actual Transit Duration (Hours)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predicted Transit Duration (Hours)", fontsize=11, fontweight="bold")
    ax1.set_title("Observed vs. Model Forecast Transit Time", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Residual Distribution
    ax2 = axes[0, 1]
    sns.histplot(residuals, kde=True, ax=ax2, color="#27ae60", bins=40, stat="density", edgecolor="white", alpha=0.6)
    # Overlay theoretical normal curve
    mu_res, std_res = np.mean(residuals), np.std(residuals)
    x_grid = np.linspace(np.min(residuals), np.max(residuals), 200)
    ax2.plot(x_grid, stats.norm.pdf(x_grid, mu_res, std_res), 'k--', lw=2, label=f"Normal Fit (μ={mu_res:.2f}, σ={std_res:.2f})")
    ax2.set_xlabel("Residual Error (Hours, e = Actual - Pred)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax2.set_title(f"Residual Distribution (Skew={skewness:.2f}, Kurtosis={kurt:.2f})", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Residuals vs Fitted (Heteroscedasticity Analysis)
    ax3 = axes[1, 0]
    ax3.scatter(y_pred, residuals, alpha=0.35, color="#e67e22", edgecolors="none", s=20)
    ax3.axhline(0, color="red", linestyle="--", lw=1.8)
    # Add rolling standard deviation band
    sorted_idx = np.argsort(y_pred)
    sorted_pred = y_pred[sorted_idx]
    sorted_res = residuals[sorted_idx]
    rolling_window = max(50, len(y_pred) // 30)
    rolling_std = pd.Series(sorted_res).rolling(window=rolling_window, min_periods=20).std().values
    ax3.plot(sorted_pred, 1.96 * rolling_std, color="#c0392b", linestyle=":", lw=1.8, label="±1.96σ Variance Envelope")
    ax3.plot(sorted_pred, -1.96 * rolling_std, color="#c0392b", linestyle=":", lw=1.8)
    ax3.set_xlabel("Predicted Transit Duration (Hours)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Residuals (Hours)", fontsize=11, fontweight="bold")
    ax3.set_title(f"Heteroscedasticity Diagnostic (BP LM Stat={lm_stat:.1f}, p={p_value_bp:.3e})", fontsize=12, fontweight="bold")
    ax3.legend(loc="lower left")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Normal Q-Q Plot
    ax4 = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.get_lines()[0].set_markerfacecolor('#8e44ad')
    ax4.get_lines()[0].set_markeredgecolor('none')
    ax4.get_lines()[0].set_alpha(0.5)
    ax4.get_lines()[0].set_markersize(5)
    ax4.get_lines()[1].set_color('#c0392b')
    ax4.get_lines()[1].set_linewidth(2)
    ax4.set_title("Normal Q-Q Plot (Heavy-Tail Diagnostics)", fontsize=12, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return {
        "Residual Mean": round(mu_res, 4),
        "Residual Std": round(std_res, 4),
        "Skewness": round(skewness, 4),
        "Kurtosis": round(kurt, 4),
        "BP_LM_Stat": round(lm_stat, 2),
        "BP_P_Value": p_value_bp,
        "Heteroscedastic": bool(p_value_bp < 0.05)
    }
