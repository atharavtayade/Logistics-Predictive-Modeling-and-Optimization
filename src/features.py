"""
Logistics Predictive Modeling and Optimization Pipeline
Module: Feature Engineering & Preprocessing Architecture
Reference: LOG-ML-OPT-2026-T4
Author: Logistics Data Analyst Intern & Systems Engineer
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw shipment telemetry into supply-chain-specific features:
    cyclical temporal encodings, aerodynamic/volumetric burdens, and spatial interactions.
    """
    df = df.copy()

    # 1. Cyclical Temporal Encodings (Harmonic expansions for 24-hour day and 7-day week)
    df["sin_departure"] = np.sin(2 * np.pi * df["departure_hour"] / 24.0)
    df["cos_departure"] = np.cos(2 * np.pi * df["departure_hour"] / 24.0)
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # 2. Peak Traffic Window Flags
    df["is_morning_rush"] = ((df["departure_hour"] >= 7.0) & (df["departure_hour"] <= 9.5)).astype(int)
    df["is_evening_rush"] = ((df["departure_hour"] >= 16.5) & (df["departure_hour"] <= 19.5)).astype(int)

    # 3. Freight Physics & Kinematic Features
    # Ton-kilometer (momentum proxy)
    df["ton_kilometers"] = (df["payload_weight_kg"] / 1000.0) * df["route_distance_km"]
    # Volumetric density ratio relative to water standard (1000 kg/m3)
    df["density_relative_water"] = df["volumetric_density_kg_m3"] / 1000.0

    # 4. Route Topography & Resistance Interactions
    # Route complexity index combining curvature tortuosity and grade ascension
    df["topographic_resistance_idx"] = df["tortuosity_factor"] * (1.0 + df["elevation_gain_m"] / 1200.0)
    # Severe weather congestion interaction
    df["weather_congestion_interaction"] = df["weather_severity_idx"] * df["diurnal_congestion_coef"]
    # Toll booth delay density (booths per 100km)
    df["toll_density_per_100km"] = (df["toll_booths_count"] / (df["route_distance_km"] + 1e-5)) * 100.0

    # 5. Dwell-to-Linehaul Friction Ratio
    # Estimated nominal linehaul hours at 50 km/h baseline
    nominal_linehaul_hrs = df["route_distance_km"] / 50.0
    df["dwell_to_linehaul_ratio"] = df["total_dwell_hours"] / (nominal_linehaul_hrs + 1e-5)

    # 6. Origin-Destination Lane String
    df["corridor_lane"] = df["origin_hub"] + "__to__" + df["dest_hub"]

    return df

def get_feature_preprocessor(categorical_cols: List[str], numerical_cols: List[str]) -> ColumnTransformer:
    """
    Constructs a robust scikit-learn ColumnTransformer for baseline models (Ridge)
    and feature matrix generation.
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", RobustScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor

def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer, numerical_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """
    Extracts transformed feature names from fitted ColumnTransformer.
    """
    feature_names = list(numerical_cols)
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cats = cat_encoder.get_feature_names_out(categorical_cols)
        feature_names.extend(list(encoded_cats))
    except Exception:
        pass
    return feature_names
