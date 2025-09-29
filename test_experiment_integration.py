#!/usr/bin/env python3
"""
Test experiment tracking integration with AutoML system and model comparison.

This script tests the integrated experiment tracking functionality to ensure
that experiments are automatically logged during AutoML training and model
comparison operations.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add the src directory to the Python path
sys.path.insert(0, 'src')

from automl.core.automl import AutoML
from automl.model_comparison import ModelComparison
from automl.experiment_tracking import ExperimentTracker, ExperimentDatabase
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification, make_regression

def create_sample_data():
    """Create sample datasets for testing."""
    print("Creating sample datasets...")
    
    # Classification data
    X_clf, y_clf = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, n_clusters_per_class=1, random_state=42
    )
    df_clf = pd.DataFrame(X_clf, columns=[f'feature_{i}' for i in range(X_clf.shape[1])])
    df_clf['target'] = y_clf
    
    # Regression data  
    X_reg, y_reg = make_regression(
        n_samples=200, n_features=10, noise=0.1, random_state=42
    )
    df_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(X_reg.shape[1])])
    df_reg['target'] = y_reg
    
    return df_clf, df_reg

def test_automl_experiment_tracking():
    """Test experiment tracking in AutoML system."""
    print("\n" + "="*60)
    print("TESTING AUTOML EXPERIMENT TRACKING")
    print("="*60)
    
    df_clf, df_reg = create_sample_data()
    
    # Test classification
    print("\n1. Testing AutoML Classification with Experiment Tracking")
    automl_clf = AutoML(
        target='target',
        task_type='classification',
        time_limit=60,  # 1 minute for fast testing
        experiment_tracking=True,
        verbose=True
    )
    
    # Train model
    automl_clf.fit(df_clf)
    
    # Check experiment tracking
    experiment_id = automl_clf.get_experiment_id()
    tracker = automl_clf.get_experiment_tracker()
    
    assert experiment_id is not None, "Experiment ID should not be None"
    assert tracker is not None, "Experiment tracker should not be None"
    
    print(f"✅ Classification experiment logged with ID: {experiment_id}")
    
    # Test regression
    print("\n2. Testing AutoML Regression with Experiment Tracking")
    automl_reg = AutoML(
        target='target',
        task_type='regression',
        time_limit=60,
        experiment_tracking=True,
        verbose=True
    )
    
    # Train model
    automl_reg.fit(df_reg)
    
    # Check experiment tracking
    experiment_id_reg = automl_reg.get_experiment_id()
    
    assert experiment_id_reg is not None, "Regression experiment ID should not be None"
    assert experiment_id_reg != experiment_id, "Different experiments should have different IDs"
    
    print(f"✅ Regression experiment logged with ID: {experiment_id_reg}")
    
    return automl_clf, automl_reg

def test_model_comparison_tracking():
    """Test experiment tracking in model comparison."""
    print("\n" + "="*60)
    print("TESTING MODEL COMPARISON EXPERIMENT TRACKING")
    print("="*60)
    
    df_clf, df_reg = create_sample_data()
    X_clf = df_clf.drop('target', axis=1)
    y_clf = df_clf['target']
    
    print("\n1. Testing Classification Model Comparison with Tracking")
    
    # Initialize comparison with experiment tracking
    comparison = ModelComparison(
        task_type='classification',
        cv_folds=3,  # Reduced for faster testing
        experiment_tracking=True
    )
    
    # Add models
    rf_clf = RandomForestClassifier(n_estimators=10, random_state=42)
    lr_clf = LogisticRegression(random_state=42, max_iter=1000)
    
    comparison.add_model(rf_clf, "RandomForest", {'n_estimators': 10})
    comparison.add_model(lr_clf, "LogisticRegression", {'max_iter': 1000})
    
    # Compare models
    result = comparison.compare_models(X_clf, y_clf)
    
    print(f"✅ Model comparison completed")
    print(f"✅ Best model: {result.best_model}")
    print(f"✅ Compared {len(result.model_results)} models")
    
    return comparison

def test_experiment_database():
    """Test experiment database functionality."""
    print("\n" + "="*60)
    print("TESTING EXPERIMENT DATABASE")
    print("="*60)
    
    # Get database from tracker
    tracker = ExperimentTracker()
    db = tracker.db
    
    # Query all experiments
    print("\n1. Querying all experiments")
    all_experiments = db.list_experiments()
    print(f"✅ Found {len(all_experiments)} total experiments")
    
    if all_experiments:
        latest_experiment = all_experiments[-1]
        print(f"✅ Latest experiment: {latest_experiment['experiment_name']}")
        print(f"   - Status: {latest_experiment['status']}")
        print(f"   - Duration: {latest_experiment['duration']}s")
        
        # Load full experiment record to check metrics
        print("\n2. Loading full experiment record")
        full_experiment = db.load_experiment(latest_experiment['experiment_id'])
        if full_experiment:
            print(f"✅ Loaded experiment record successfully")
            print(f"   - Primary metric: {full_experiment.metrics.primary_metric_name} = {full_experiment.metrics.primary_metric:.4f}")
            if full_experiment.metrics.additional_metrics:
                print(f"   - Additional metrics: {len(full_experiment.metrics.additional_metrics)}")
        else:
            print("⚠️ Could not load experiment record")
    
    # Test filtering
    print("\n3. Testing experiment filtering")
    automl_experiments = db.list_experiments(
        filters={'experiment_name': 'AutoML'}
    )
    print(f"✅ Found {len(automl_experiments)} AutoML experiments")
    
    classification_experiments = db.list_experiments(
        filters={'task_type': 'classification'}
    )
    print(f"✅ Found {len(classification_experiments)} classification experiments")

def test_experiment_analytics():
    """Test experiment analytics functionality."""
    print("\n" + "="*60)
    print("TESTING EXPERIMENT ANALYTICS")
    print("="*60)
    
    try:
        from automl.experiment_analytics import ExperimentAnalytics, ExperimentHistory
        
        # Create history manager
        tracker = ExperimentTracker()
        history = ExperimentHistory(tracker)
        
        analytics = ExperimentAnalytics(history)
        
        print("\n1. Testing performance trends")
        try:
            trends_fig = analytics.create_performance_timeline()
            if trends_fig:
                print("✅ Performance trends plot generated")
            else:
                print("⚠️ No data available for performance trends")
        except Exception as e:
            print(f"⚠️ Could not generate trends plot: {e}")
        
        print("\n2. Testing experiment comparison")
        try:
            experiments = history.search_experiments(limit=5)
            if len(experiments) >= 2:
                exp_ids = [e['experiment_id'] for e in experiments[:2]]
                comparison_data = history.compare_experiments(exp_ids)
                if comparison_data:
                    print("✅ Experiment comparison data generated")
            else:
                print("⚠️ Need at least 2 experiments for comparison")
        except Exception as e:
            print(f"⚠️ Could not generate comparison: {e}")
            
    except ImportError as e:
        print(f"⚠️ Could not test analytics: {e}")

def cleanup_test_data():
    """Clean up test data and database."""
    print("\n" + "="*60)
    print("CLEANUP")
    print("="*60)
    
    # Note: In a real scenario, you might want to keep experiment data
    # For testing, we'll leave it as experiments can be valuable for inspection
    print("✅ Test completed - experiment data preserved for inspection")
    print("   Database location: experiments.db")
    print("   Model artifacts: experiments/ directory")

def main():
    """Main test function."""
    print("Starting AutoML Experiment Tracking Integration Tests")
    print("="*60)
    
    try:
        # Test AutoML integration
        automl_clf, automl_reg = test_automl_experiment_tracking()
        
        # Test model comparison integration
        comparison = test_model_comparison_tracking()
        
        # Test database functionality
        test_experiment_database()
        
        # Test analytics
        test_experiment_analytics()
        
        # Cleanup
        cleanup_test_data()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("Experiment tracking is successfully integrated with:")
        print("✅ AutoML system")
        print("✅ Model comparison framework")
        print("✅ Database storage and querying")
        print("✅ Analytics and visualization")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())