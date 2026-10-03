"""
Unit Tests: Feature Engineering & Kinematic Derivations
Reference: LOG-ML-OPT-2026-T4
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_generator import haversine_distance_vectorized, diurnal_congestion_factor, generate_logistics_dataset
from features import engineer_features

class TestFeatureEngineering(unittest.TestCase):
    
    def test_haversine_distance(self):
        # Known distance between SP (-23.5505, -46.6333) and RJ (-22.9068, -43.1729) is ~357-360 km
        sp_lat, sp_lon = np.array([-23.5505]), np.array([-46.6333])
        rj_lat, rj_lon = np.array([-22.9068]), np.array([-43.1729])
        
        dist = haversine_distance_vectorized(sp_lat, sp_lon, rj_lat, rj_lon)[0]
        self.assertGreater(dist, 340.0)
        self.assertLess(dist, 380.0)

    def test_diurnal_congestion_curve(self):
        # Test rush hour vs late night
        morning_rush = diurnal_congestion_factor(np.array([8.5]))[0]
        night_trough = diurnal_congestion_factor(np.array([2.0]))[0]
        
        self.assertGreater(morning_rush, 1.35)
        self.assertLess(night_trough, 1.0)
        self.assertGreater(morning_rush, night_trough)

    def test_feature_engineering_derivations(self):
        raw_df = generate_logistics_dataset(n_samples=20, random_state=42)
        feat_df = engineer_features(raw_df)
        
        # Check cyclical temporal bounds [-1, 1]
        self.assertTrue((feat_df["sin_departure"] >= -1.0).all() and (feat_df["sin_departure"] <= 1.0).all())
        self.assertTrue((feat_df["cos_departure"] >= -1.0).all() and (feat_df["cos_departure"] <= 1.0).all())
        
        # Check density and momentum features
        self.assertIn("volumetric_density_kg_m3", feat_df.columns)
        self.assertIn("ton_kilometers", feat_df.columns)
        self.assertIn("weather_congestion_interaction", feat_df.columns)
        self.assertTrue((feat_df["ton_kilometers"] > 0).all())

if __name__ == "__main__":
    unittest.main()
