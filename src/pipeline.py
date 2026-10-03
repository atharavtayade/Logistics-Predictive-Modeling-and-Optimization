"""
Logistics Predictive Modeling and Optimization Pipeline
Module: Master End-to-End Orchestrator
Reference: LOG-ML-OPT-2026-T4
Author: Logistics Data Analyst Intern & Systems Engineer
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

from data_generator import generate_logistics_dataset
from features import engineer_features, get_feature_preprocessor, get_feature_names_from_preprocessor
from models import train_and_evaluate_cv
from diagnostics import compute_feature_importances, plot_feature_importance_and_shap, plot_residual_diagnostics
from optimizer import generate_cvrptw_scenario, solve_cvrptw, plot_cvrptw_routes

def run_master_pipeline():
    start_time = time.time()
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("================================================================================")
    print("LOG-ML-OPT-2026-T4: PREDICTIVE MODELING & OPTIMIZATION PIPELINE")
    print("================================================================================")

    # -------------------------------------------------------------------------
    # Step 1: Synthesize Multimodal Logistics Telemetry Dataset
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Synthesizing Multimodal Freight Dataset (Olist & DataCo Calibration)...")
    raw_df = generate_logistics_dataset(n_samples=15000, random_state=42)
    raw_csv_path = "data/logistics_multimodal_dataset.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"Generated {len(raw_df):,} shipment records across {raw_df['origin_hub'].nunique()} origin and destination hubs.")
    print(f"Overall SLA Breach Rate: {raw_df['sla_breach_risk'].mean():.2%}")
    print(f"Raw dataset saved to: {raw_csv_path}")

    # -------------------------------------------------------------------------
    # Step 2: Feature Engineering & Preprocessing Transformation
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Executing Feature Engineering and Preprocessing Pipeline...")
    feat_df = engineer_features(raw_df)

    numerical_features = [
        "radial_distance_km",
        "route_distance_km",
        "tortuosity_factor",
        "payload_weight_kg",
        "cargo_volume_m3",
        "volumetric_density_kg_m3",
        "departure_hour",
        "diurnal_congestion_coef",
        "weather_severity_idx",
        "elevation_gain_m",
        "toll_booths_count",
        "origin_dwell_hours",
        "destination_dwell_hours",
        "total_dwell_hours",
        "sin_departure",
        "cos_departure",
        "sin_day",
        "cos_day",
        "is_morning_rush",
        "is_evening_rush",
        "ton_kilometers",
        "density_relative_water",
        "topographic_resistance_idx",
        "weather_congestion_interaction",
        "toll_density_per_100km",
        "dwell_to_linehaul_ratio"
    ]

    categorical_features = [
        "carrier_tier",
        "origin_hub",
        "dest_hub"
    ]

    preprocessor = get_feature_preprocessor(categorical_features, numerical_features)
    X_processed = preprocessor.fit_transform(feat_df)
    feature_names = get_feature_names_from_preprocessor(preprocessor, numerical_features, categorical_features)

    y_continuous = feat_df["transit_duration_hours"].values
    sla_targets = feat_df["sla_target_hours"].values
    sla_breaches = feat_df["sla_breach_risk"].values

    print(f"Feature matrix generated: {X_processed.shape[0]:,} samples x {X_processed.shape[1]} engineered features.")

    # -------------------------------------------------------------------------
    # Step 3: Model Selection, 5-Fold Cross-Validation & Metric Evaluation
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Benchmarking 4 Model Tiers across 5-Fold Cross Validation...")

    model_factories = {
        "Tier 1: Ridge Regression (Baseline)": lambda: Ridge(alpha=15.0),
        "Tier 2: Random Forest Regressor": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=42
        ),
        "Tier 3A: LightGBM Regressor": lambda: lgb.LGBMRegressor(
            n_estimators=250, learning_rate=0.06, num_leaves=31, subsample=0.85,
            colsample_bytree=0.85, random_state=42, verbose=-1, n_jobs=-1
        ),
        "Tier 3B: XGBoost Regressor": lambda: xgb.XGBRegressor(
            n_estimators=250, learning_rate=0.06, max_depth=6, subsample=0.85,
            colsample_bytree=0.85, random_state=42, n_jobs=-1
        )
    }

    summary_df, trained_artifacts = train_and_evaluate_cv(
        model_factories,
        X_processed,
        y_continuous,
        sla_targets,
        sla_breaches,
        n_splits=5,
        random_state=42
    )

    print("\nModel Cross-Validation Leaderboard:")
    print(summary_df.to_string(index=False))

    # Plot Model Tier Comparison
    plt.figure(figsize=(10, 5), dpi=300)
    bars = plt.bar(
        [m.split(":")[0] + "\n" + m.split(":")[1].strip() for m in summary_df["Model Tier"]],
        summary_df["CV RMSE (hrs)"],
        color=["#95a5a6", "#3498db", "#2ecc71", "#1abc9c"],
        edgecolor="#2c3e50",
        width=0.55
    )
    plt.ylabel("Cross-Validation RMSE (Hours)", fontsize=11, fontweight="bold")
    plt.title("Model Tier Benchmark: Transit Duration Forecast Accuracy", fontsize=13, fontweight="bold", pad=12)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}h", ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/model_comparison.png")
    plt.close()

    # -------------------------------------------------------------------------
    # Step 4: Feature Importance & Residual Diagnostics
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Computing SHAP Attributions and Residual Diagnostic Suite...")
    best_tier_name = summary_df.iloc[0]["Model Tier"]
    best_model = trained_artifacts[best_tier_name]["model"]
    best_oof_preds = trained_artifacts[best_tier_name]["oof_preds"]

    # Sample 1000 rows for SHAP tree evaluation
    shap_sample_idx = np.random.RandomState(42).choice(len(X_processed), size=1000, replace=False)
    X_shap_sample = X_processed[shap_sample_idx]

    importance_df, _ = compute_feature_importances(
        best_model,
        feature_names,
        X_shap_sample,
        max_features=14
    )
    plot_feature_importance_and_shap(importance_df, "figures/feature_importance.png")
    print("Saved SHAP and feature importance visualization to: figures/feature_importance.png")

    residual_diag = plot_residual_diagnostics(y_continuous, best_oof_preds, "figures/residual_diagnostics.png")
    print(f"Residual Diagnostics: Mean={residual_diag['Residual Mean']}h, Std={residual_diag['Residual Std']}h, Skew={residual_diag['Skewness']}, BP LM Stat={residual_diag['BP_LM_Stat']}")
    print("Saved 4-panel residual diagnostic grid to: figures/residual_diagnostics.png")

    # -------------------------------------------------------------------------
    # Step 5: Prescriptive Optimization Architecture (Google OR-Tools CVRPTW)
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Executing Prescriptive CVRPTW Solver (Google OR-Tools Guided Local Search)...")
    cvrptw_scenario = generate_cvrptw_scenario(n_customers=26, depot_coord=(-23.5505, -46.6333), random_state=101)
    cvrptw_results = solve_cvrptw(cvrptw_scenario, time_limit_seconds=12)

    plot_cvrptw_routes(cvrptw_scenario, cvrptw_results, "figures/cvrptw_routes.png")
    print("Saved CVRPTW dispatch topology map to: figures/cvrptw_routes.png")

    print("\nOptimization Gains Summary:")
    print(f"  - Active Vehicles: {cvrptw_results['active_vehicles']} (Baseline: {cvrptw_results['baseline']['active_vehicles']}, Fleet Reduction: {cvrptw_results['savings']['fleet_reduction']})")
    print(f"  - Total Distance: {cvrptw_results['total_distance_km']} km (Baseline: {cvrptw_results['baseline']['total_distance_km']} km, Reduction: {cvrptw_results['savings']['distance_reduction_pct']}%)")
    print(f"  - Total Duration: {cvrptw_results['total_duration_hours']} hrs (Baseline: {cvrptw_results['baseline']['total_duration_hours']} hrs, Reduction: {cvrptw_results['savings']['time_reduction_pct']}%)")
    print(f"  - Fuel Conserved: {cvrptw_results['savings']['fuel_saved_liters']} L | CO2 Avoided: {cvrptw_results['savings']['co2_saved_kg']} kg")
    print(f"  - Weight Utilization: {cvrptw_results['avg_weight_utilization_pct']}% | Volume Utilization: {cvrptw_results['avg_volume_utilization_pct']}%")

    # Save structured execution results
    execution_summary = {
        "metadata": {
            "document_ref": "LOG-ML-OPT-2026-T4",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_duration_sec": round(time.time() - start_time, 2)
        },
        "dataset_statistics": {
            "total_records": len(raw_df),
            "num_features_engineered": int(X_processed.shape[1]),
            "sla_breach_rate": round(float(raw_df['sla_breach_risk'].mean()), 4),
            "mean_transit_hours": round(float(raw_df['transit_duration_hours'].mean()), 2),
            "std_transit_hours": round(float(raw_df['transit_duration_hours'].std()), 2)
        },
        "model_leaderboard": summary_df.to_dict(orient="records"),
        "top_features": importance_df.head(10).to_dict(orient="records"),
        "residual_diagnostics": residual_diag,
        "cvrptw_optimization": {
            "active_vehicles": cvrptw_results["active_vehicles"],
            "baseline_vehicles": cvrptw_results["baseline"]["active_vehicles"],
            "optimized_distance_km": cvrptw_results["total_distance_km"],
            "baseline_distance_km": cvrptw_results["baseline"]["total_distance_km"],
            "distance_reduction_pct": cvrptw_results["savings"]["distance_reduction_pct"],
            "time_reduction_pct": cvrptw_results["savings"]["time_reduction_pct"],
            "fuel_saved_liters": cvrptw_results["savings"]["fuel_saved_liters"],
            "co2_saved_kg": cvrptw_results["savings"]["co2_saved_kg"],
            "avg_weight_utilization_pct": cvrptw_results["avg_weight_utilization_pct"],
            "avg_volume_utilization_pct": cvrptw_results["avg_volume_utilization_pct"],
            "routes_summary": [
                {
                    "vehicle_id": r["vehicle_id"] + 1,
                    "stops": r["stops_count"],
                    "distance_km": r["route_distance_km"],
                    "duration_min": r["route_duration_min"],
                    "weight_kg": r["final_weight_kg"],
                    "volume_m3": r["final_volume_m3"],
                    "weight_util_pct": r["weight_utilization_pct"],
                    "volume_util_pct": r["volume_utilization_pct"]
                }
                for r in cvrptw_results["routes"]
            ]
        }
    }

    with open("data/pipeline_execution_summary.json", "w") as f:
        json.dump(execution_summary, f, indent=4)

    print("\n================================================================================")
    print(f"PIPELINE EXECUTION COMPLETED IN {time.time() - start_time:.2f} SECONDS")
    print("Structured metrics written to: data/pipeline_execution_summary.json")
    print("================================================================================")
    return execution_summary

if __name__ == "__main__":
    run_master_pipeline()
