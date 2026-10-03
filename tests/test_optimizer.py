"""
Unit Tests: CVRPTW Optimization & Dual Capacity Constraints
Reference: LOG-ML-OPT-2026-T4
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from optimizer import generate_cvrptw_scenario, solve_cvrptw

class TestCVRPTOptimizer(unittest.TestCase):
    
    def setUp(self):
        # 12 customer stops small scenario for quick test verification
        self.scenario = generate_cvrptw_scenario(n_customers=12, random_state=42)
        
    def test_scenario_dimensions(self):
        self.assertEqual(self.scenario["num_nodes"], 13) # 1 depot + 12 stops
        self.assertEqual(len(self.scenario["weight_demands"]), 13)
        self.assertEqual(len(self.scenario["volume_demands"]), 13)
        self.assertEqual(self.scenario["distance_matrix"].shape, (13, 13))

    def test_cvrptw_dual_capacity_feasibility(self):
        res = solve_cvrptw(self.scenario, time_limit_seconds=3)
        self.assertEqual(res["status"], "OPTIMAL_FOUND")
        self.assertGreater(res["active_vehicles"], 0)
        
        # Verify no vehicle exceeds weight or volume capacities
        max_weight = self.scenario["vehicle_weight_capacity"]
        max_vol_units = self.scenario["vehicle_volume_capacity"]
        
        for route in res["routes"]:
            self.assertLessEqual(route["final_weight_kg"], max_weight)
            # Volume in units of 0.01 m3
            self.assertLessEqual(int(route["final_volume_m3"] * 100), max_vol_units)
            self.assertGreater(route["stops_count"], 0)
            
        # Verify mileage reduction relative to unoptimized baseline
        self.assertGreater(res["savings"]["distance_reduction_pct"], 0.0)

if __name__ == "__main__":
    unittest.main()
