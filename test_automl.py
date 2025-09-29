#!/usr/bin/env python
"""
Test the AutoML system with sample datasets.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from automl import AutoML
import pandas as pd

def test_classification():
    """Test AutoML on classification dataset."""
    print("=" * 60)
    print("TESTING CLASSIFICATION")
    print("=" * 60)
    
    # Load data
    data_path = 'data/iris_classification.xlsx'
    if not Path(data_path).exists():
        print(f"Dataset {data_path} not found. Please run create_sample_data.py first.")
        return
    
    print(f"Loading dataset: {data_path}")
    
    try:
        # Initialize AutoML for classification
        automl = AutoML(
            target='species',
            task_type='classification',
            time_limit=300,  # 5 minutes
            validation_split=0.2,
            verbose=True
        )
        
        print("\nStarting AutoML training for classification...")
        
        # Fit the model
        automl.fit(data_path)
        
        # Get results
        print("\nGetting results...")
        results = automl.get_results()
        
        print(f"\n🎉 AutoML Training Complete!")
        print(f"Best model: {results['best_model_name']}")
        print(f"Best score: {results['best_score']:.4f}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Features used: {results['feature_count']}")
        
        # Show feature importance
        print("\n📊 Top 5 Most Important Features:")
        feature_importance = results['feature_importance']
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
            
        # Test prediction on sample data
        print("\n🔮 Testing predictions...")
        sample_data = pd.read_excel(data_path).head(5)
        predictions = automl.predict(sample_data)
        
        print("Sample predictions:")
        for i, pred in enumerate(predictions[:5]):
            print(f"  Row {i+1}: {pred}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during classification test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_regression():
    """Test AutoML on regression dataset."""
    print("\n" + "=" * 60)
    print("TESTING REGRESSION")
    print("=" * 60)
    
    # Load data
    data_path = 'data/housing_regression.xlsx'
    if not Path(data_path).exists():
        print(f"Dataset {data_path} not found. Please run create_sample_data.py first.")
        return
    
    print(f"Loading dataset: {data_path}")
    
    try:
        # Initialize AutoML for regression
        automl = AutoML(
            target='price',
            task_type='regression',
            time_limit=300,  # 5 minutes
            validation_split=0.2,
            verbose=True
        )
        
        print("\nStarting AutoML training for regression...")
        
        # Fit the model
        automl.fit(data_path)
        
        # Get results
        print("\nGetting results...")
        results = automl.get_results()
        
        print(f"\n🎉 AutoML Training Complete!")
        print(f"Best model: {results['best_model_name']}")
        print(f"Best score (R²): {results['best_score']:.4f}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Features used: {results['feature_count']}")
        
        # Show feature importance
        print("\n📊 Top 5 Most Important Features:")
        feature_importance = results['feature_importance']
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
            
        # Test prediction on sample data
        print("\n🔮 Testing predictions...")
        sample_data = pd.read_excel(data_path).head(5)
        predictions = automl.predict(sample_data)
        
        print("Sample predictions (prices):")
        for i, pred in enumerate(predictions[:5]):
            print(f"  Row {i+1}: ${pred:,.0f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during regression test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing AutoML System")
    print("This will test both classification and regression tasks.")
    
    # Check if sample data exists
    if not Path('data/iris_classification.xlsx').exists() or not Path('data/housing_regression.xlsx').exists():
        print("\n📊 Sample datasets not found. Creating them now...")
        os.system('python create_sample_data.py')
    
    # Run tests
    classification_success = test_classification()
    regression_success = test_regression()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if classification_success:
        print("✅ Classification test: PASSED")
    else:
        print("❌ Classification test: FAILED")
    
    if regression_success:
        print("✅ Regression test: PASSED")
    else:
        print("❌ Regression test: FAILED")
        
    if classification_success and regression_success:
        print("\n🎉 All tests passed! The AutoML system is working correctly.")
        print("\nYou can now use your own Excel datasets with the AutoML system:")
        print("  from automl import AutoML")
        print("  automl = AutoML(target='your_target_column', task_type='classification')")
        print("  automl.fit('your_data.xlsx')")
        print("  results = automl.get_results()")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()