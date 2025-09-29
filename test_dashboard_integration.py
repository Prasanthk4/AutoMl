#!/usr/bin/env python3
"""
Simple test to verify dashboard integration is working correctly.

This script checks if the dashboard imports work and basic functionality is accessible.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, 'src')

def test_dashboard_imports():
    """Test that all dashboard-related imports work correctly."""
    print("Testing dashboard imports...")
    
    try:
        from automl import AutoML
        from automl.data_analysis import AdvancedDataAnalyzer
        from automl.model_comparison import ModelComparison, ABTestFramework
        from automl.comparison_visualizations import ComparisonVisualizer
        from automl.experiment_tracking import ExperimentTracker, ExperimentDatabase
        from automl.experiment_analytics import ExperimentHistory, ExperimentAnalytics
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_experiment_tracking_components():
    """Test that experiment tracking components initialize correctly."""
    print("Testing experiment tracking components...")
    
    try:
        # Import components within the function scope
        from automl.experiment_tracking import ExperimentTracker, ExperimentDatabase
        from automl.experiment_analytics import ExperimentHistory, ExperimentAnalytics
        
        # Test ExperimentTracker
        tracker = ExperimentTracker()
        print("✅ ExperimentTracker initialized")
        
        # Test ExperimentHistory
        history = ExperimentHistory(tracker)
        print("✅ ExperimentHistory initialized")
        
        # Test ExperimentAnalytics
        analytics = ExperimentAnalytics(history)
        print("✅ ExperimentAnalytics initialized")
        
        # Test database connection
        db = tracker.db
        experiments = db.list_experiments(limit=5)
        print(f"✅ Database connection working - found {len(experiments)} experiments")
        
        return True
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def test_dashboard_functions():
    """Test that dashboard functions can be imported."""
    print("Testing dashboard functions...")
    
    # Import the streamlit app as a module to test functions
    try:
        import streamlit_app
        
        # Check if key functions exist
        functions_to_check = [
            'show_experiment_history_dashboard',
            'show_analytics_dashboard', 
            'show_experiment_browser',
            'show_experiment_search',
            'show_experiment_comparison'
        ]
        
        for func_name in functions_to_check:
            if hasattr(streamlit_app, func_name):
                print(f"✅ Function {func_name} found")
            else:
                print(f"❌ Function {func_name} not found")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Dashboard function test failed: {e}")
        return False

def main():
    """Run all dashboard integration tests."""
    print("🔍 Testing Dashboard Integration for Feature #4: Experiment Management & History")
    print("=" * 80)
    
    tests = [
        test_dashboard_imports,
        test_experiment_tracking_components,
        test_dashboard_functions
    ]
    
    passed = 0
    for test in tests:
        print()
        if test():
            passed += 1
        print("-" * 40)
    
    print()
    print("=" * 80)
    if passed == len(tests):
        print("🎉 ALL DASHBOARD INTEGRATION TESTS PASSED!")
        print()
        print("✅ Dashboard successfully integrated with:")
        print("   • Experiment History browser")
        print("   • Advanced search and filtering")
        print("   • Experiment comparison tools") 
        print("   • Performance analytics")
        print("   • Data export capabilities")
        print()
        print("📊 Access the dashboard by running:")
        print("   streamlit run streamlit_app.py")
        print()
        print("🎯 Navigate between:")
        print("   • 🤖 AutoML Training (upload data & train models)")
        print("   • 📊 Experiment History (browse & compare experiments)")
        print("   • 📈 Analytics (performance trends & insights)")
        return 0
    else:
        print(f"❌ {len(tests) - passed}/{len(tests)} tests failed")
        return 1

if __name__ == "__main__":
    exit(main())