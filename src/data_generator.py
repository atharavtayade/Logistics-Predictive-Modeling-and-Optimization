"""
Logistics Predictive Modeling and Optimization Pipeline
Module: Data Generation & Synthetic Simulation
Reference: LOG-ML-OPT-2026-T4
Author: Logistics Data Analyst Intern & Systems Engineer
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def haversine_distance_vectorized(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Computes the great-circle distance between two points on the Earth's surface
    using the Haversine formula in kilometers.
    """
    R = 6371.0  # Earth's radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def diurnal_congestion_factor(hour: np.ndarray) -> np.ndarray:
    """
    Models diurnal road traffic congestion coefficients.
    Peaks around 08:00 - 09:30 and 17:00 - 19:30.
    """
    # Base congestion = 1.0 (free flow)
    morning_peak = 0.45 * np.exp(-((hour - 8.5) ** 2) / (2 * 1.5 ** 2))
    evening_peak = 0.55 * np.exp(-((hour - 18.0) ** 2) / (2 * 2.0 ** 2))
    night_trough = -0.15 * np.exp(-((hour - 2.0) ** 2) / (2 * 2.0 ** 2))
    return 1.0 + morning_peak + evening_peak + night_trough

def generate_logistics_dataset(
    n_samples: int = 15000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Synthesizes multimodal freight logistics records calibrated against empirical
    distributions of the Brazilian E-Commerce (Olist) & DataCo Supply Chain datasets.
    """
    rng = np.random.RandomState(random_state)

    # 1. Geographic Hub Distribution (Simulating Brazilian Southeast / South Logistics Corridor)
    # São Paulo (-23.55, -46.63), Rio de Janeiro (-22.90, -43.17), Belo Horizonte (-19.92, -43.94),
    # Curitiba (-25.42, -49.27), Porto Alegre (-30.03, -51.23), Campinas (-22.90, -47.06)
    hub_coords = [
        (-23.5505, -46.6333),  # SP Central Hub
        (-22.9068, -43.1729),  # RJ Gateway
        (-19.9167, -43.9345),  # MG Hub
        (-25.4284, -49.2733),  # PR Hub
        (-30.0346, -51.2177),  # RS Hub
        (-22.9056, -47.0608),  # Viracopos Air / Road Freight
    ]
    hub_names = ["SP_METRO", "RJ_METRO", "BH_METRO", "CTB_METRO", "POA_METRO", "VCP_CARGO"]

    origin_idx = rng.choice(len(hub_coords), size=n_samples, p=[0.40, 0.20, 0.15, 0.10, 0.05, 0.10])
    dest_idx = rng.choice(len(hub_coords), size=n_samples, p=[0.25, 0.25, 0.15, 0.15, 0.10, 0.10])

    # Add Gaussian jitter to simulate dispersed fulfillment centers and regional delivery stops
    origin_lats = np.array([hub_coords[i][0] for i in origin_idx]) + rng.normal(0, 0.15, size=n_samples)
    origin_lons = np.array([hub_coords[i][1] for i in origin_idx]) + rng.normal(0, 0.15, size=n_samples)

    dest_lats = np.array([hub_coords[i][0] for i in dest_idx]) + rng.normal(0, 0.35, size=n_samples)
    dest_lons = np.array([hub_coords[i][1] for i in dest_idx]) + rng.normal(0, 0.35, size=n_samples)

    origin_hub = [hub_names[i] for i in origin_idx]
    dest_hub = [hub_names[i] for i in dest_idx]

    # Calculate Radial Haversine Distance (km)
    radial_distance_km = haversine_distance_vectorized(origin_lats, origin_lons, dest_lats, dest_lons)
    radial_distance_km = np.maximum(radial_distance_km, 8.5)  # Minimum intracity last-mile distance

    # Route Tortuosity Factor (Actual road network distance vs straight line)
    # Mountainous / coastal Brazilian corridors exhibit higher tortuosity (1.25 to 1.48)
    tortuosity = rng.uniform(1.22, 1.45, size=n_samples)
    route_distance_km = radial_distance_km * tortuosity

    # 2. Package & Payload Characteristics (Olist/DataCo Distributions)
    # Log-normal distribution for freight weight (kg) and dimensions (cm)
    payload_weight_kg = np.round(rng.lognormal(mean=1.5, sigma=1.0, size=n_samples), 2)
    payload_weight_kg = np.clip(payload_weight_kg, 0.3, 1200.0)  # From parcels to palletized LTL

    dim_length_cm = rng.uniform(15.0, 110.0, size=n_samples)
    dim_width_cm = rng.uniform(12.0, 85.0, size=n_samples)
    dim_height_cm = rng.uniform(10.0, 75.0, size=n_samples)
    cargo_volume_m3 = np.round((dim_length_cm * dim_width_cm * dim_height_cm) / 1e6, 4)
    volumetric_density_kg_m3 = np.round(payload_weight_kg / cargo_volume_m3, 2)

    # 3. Freight Mode & Carrier Service Tier
    carrier_tiers = ["STANDARD_PARCEL", "EXPRESS_COURIER", "LTL_FREIGHT", "HEAVY_DEDICATED"]
    tier_assignment = rng.choice(carrier_tiers, size=n_samples, p=[0.55, 0.25, 0.15, 0.05])

    carrier_speed_means = {
        "STANDARD_PARCEL": 48.0,  # km/h highway average including regional stops
        "EXPRESS_COURIER": 65.0,   # expedited direct routing
        "LTL_FREIGHT": 40.0,      # multi-stop consolidation
        "HEAVY_DEDICATED": 52.0   # long haul dedicated linehaul
    }
    nominal_speed_kmh = np.array([carrier_speed_means[t] for t in tier_assignment]) * rng.normal(1.0, 0.08, size=n_samples)

    # 4. Temporal & Diurnal Characteristics
    departure_hour = rng.uniform(0, 24, size=n_samples)
    day_of_week = rng.choice([0, 1, 2, 3, 4, 5, 6], size=n_samples, p=[0.18, 0.19, 0.19, 0.18, 0.16, 0.06, 0.04]) # Monday=0, Sun=6
    is_weekend = (day_of_week >= 5).astype(int)

    congestion_coef = diurnal_congestion_factor(departure_hour)
    # Weekday intercity bottleneck scaling
    congestion_coef = congestion_coef * np.where(is_weekend == 1, 0.85, 1.05)

    # 5. Weather Severity Index (WSI) and Environmental Stress
    # Beta distribution representing calm days with tail of convective storms / rain
    weather_severity = rng.beta(a=1.5, b=5.0, size=n_samples) # Scale 0 to 1
    elevation_gain_m = rng.exponential(scale=220.0, size=n_samples) + (radial_distance_km * 0.4)

    # 6. Dwell & Terminal Handling Delays (Hours)
    # Origin cross-dock dwell time + Destination staging/unloading dwell
    origin_dwell_hours = rng.gamma(shape=2.5, scale=1.2, size=n_samples) # mean ~ 3.0 hrs
    destination_dwell_hours = rng.gamma(shape=1.8, scale=0.8, size=n_samples) # mean ~ 1.4 hrs
    toll_booths_count = np.maximum(0, np.round(route_distance_km / 75.0) + rng.poisson(lam=1.0, size=n_samples)).astype(int)

    # 7. Physical Transit Duration Equation
    # Raw running transit hours = (Route Distance / (Effective Speed))
    effective_speed = nominal_speed_kmh / (congestion_coef * (1.0 + 0.35 * weather_severity))
    effective_speed = np.maximum(effective_speed, 18.0) # Lower bound under gridlock
    linehaul_duration_hours = route_distance_km / effective_speed

    # Additional delay components:
    # Topographic drag on heavy vehicles
    grade_penalty_hours = (elevation_gain_m / 1000.0) * (payload_weight_kg / 500.0) * 0.35
    toll_delay_hours = toll_booths_count * (8.0 / 60.0) # ~8 mins per plaza/queue

    # Total actual transit duration (continuous target)
    stochastic_noise = rng.lognormal(mean=0.0, sigma=0.12, size=n_samples)
    transit_duration_hours = (
        origin_dwell_hours
        + linehaul_duration_hours
        + grade_penalty_hours
        + toll_delay_hours
        + destination_dwell_hours
    ) * stochastic_noise
    transit_duration_hours = np.round(transit_duration_hours, 3)

    # 8. Service Level Agreement (SLA) Promised Window (Hours)
    # SLAs are promised based on nominal distance formula with standard logistics buffer
    # e.g., Standard parcel gives ~ 24h + 40km/h baseline buffer; Express courier promises tighter window
    sla_buffer_hours = {
        "STANDARD_PARCEL": 18.0,
        "EXPRESS_COURIER": 8.0,
        "LTL_FREIGHT": 24.0,
        "HEAVY_DEDICATED": 14.0
    }
    buffer_hours = np.array([sla_buffer_hours[t] for t in tier_assignment])
    sla_target_hours = np.round(buffer_hours + (route_distance_km / 50.0) * 1.35, 2)

    # Binary Classification Target: SLA Breach
    sla_breach_risk = (transit_duration_hours > sla_target_hours).astype(int)
    delay_severity_hours = np.maximum(0.0, transit_duration_hours - sla_target_hours)

    df = pd.DataFrame({
        "tracking_id": [f"TRK-2026-{i:07d}" for i in range(1, n_samples + 1)],
        "origin_hub": origin_hub,
        "dest_hub": dest_hub,
        "origin_lat": np.round(origin_lats, 5),
        "origin_lon": np.round(origin_lons, 5),
        "dest_lat": np.round(dest_lats, 5),
        "dest_lon": np.round(dest_lons, 5),
        "radial_distance_km": np.round(radial_distance_km, 2),
        "route_distance_km": np.round(route_distance_km, 2),
        "tortuosity_factor": np.round(tortuosity, 3),
        "payload_weight_kg": payload_weight_kg,
        "dim_length_cm": np.round(dim_length_cm, 1),
        "dim_width_cm": np.round(dim_width_cm, 1),
        "dim_height_cm": np.round(dim_height_cm, 1),
        "cargo_volume_m3": cargo_volume_m3,
        "volumetric_density_kg_m3": volumetric_density_kg_m3,
        "carrier_tier": tier_assignment,
        "departure_hour": np.round(departure_hour, 2),
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "diurnal_congestion_coef": np.round(congestion_coef, 3),
        "weather_severity_idx": np.round(weather_severity, 3),
        "elevation_gain_m": np.round(elevation_gain_m, 1),
        "toll_booths_count": toll_booths_count,
        "origin_dwell_hours": np.round(origin_dwell_hours, 2),
        "destination_dwell_hours": np.round(destination_dwell_hours, 2),
        "total_dwell_hours": np.round(origin_dwell_hours + destination_dwell_hours, 2),
        "sla_target_hours": sla_target_hours,
        "transit_duration_hours": transit_duration_hours,
        "sla_breach_risk": sla_breach_risk,
        "delay_severity_hours": np.round(delay_severity_hours, 2)
    })

    return df

if __name__ == "__main__":
    df = generate_logistics_dataset(n_samples=5000)
    print("Dataset synthesized successfully:")
    print(f"Shape: {df.shape}")
    print(f"SLA Breach Rate: {df['sla_breach_risk'].mean():.2%}")
    print(df[['route_distance_km', 'transit_duration_hours', 'sla_target_hours', 'sla_breach_risk']].describe())
