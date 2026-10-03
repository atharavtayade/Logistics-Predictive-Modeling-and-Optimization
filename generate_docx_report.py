"""
Script: generate_docx_report.py
Purpose: Wrapper to generate the executive DOCX report for Task 4.
Reference: LOG-ML-OPT-2026-T4
"""

import sys
import os
from generate_docx import build_executive_docx

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/Task_4_Predictive_Modeling_and_Optimization_Report.docx"
    build_executive_docx(out_path)
    build_executive_docx("Task_4_Predictive_Modeling_and_Optimization.docx")
    print("Report generation complete: Task_4_Predictive_Modeling_and_Optimization.docx")
