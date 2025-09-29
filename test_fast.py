#!/usr/bin/env python
"""
Fast test script for AutoML system with performance optimizations.
This should complete in under 2 minutes.
"""

import sys
import os
import time
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from automl import AutoML
import pandas as pd

def create_quick_test_data():
    """Create a small test dataset for fast training."""
    import numpy as np
    
    # Generate small synthetic dataset
    np.random.seed(42)
    n_samples = 200  # Small dataset for fast training
    
    # Features
    feature1 = np.random.randn(n_samples)
    feature2 = np.random.randn(n_samples)
    feature3 = np.random.choice(['A', 'B', 'C'], n_samples)
    feature4 = np.random.uniform(0, 100, n_samples)
    
    # Target (classification)
    target = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    
    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'target': target
    })
    
    return df

def test_fast_training():
    """Test fast AutoML training."""
    print("🚀 Testing Fast AutoML Training")
    print("=" * 50)
    
    # Create test data
    print("📊 Creating small test dataset...")
    data = create_quick_test_data()
    print(f"Dataset shape: {data.shape}")
    
    # Initialize AutoML with very fast settings
    print("⚙️  Initializing AutoML with performance optimizations...")
    
    # Custom config for even faster training
    fast_config = {
        'models': {
            'include_models': ['random_forest'],  # Only one model for speed
            'n_jobs': 1,  # Single core to avoid CPU overload
            'cv_folds': 2,  # Minimal CV folds
        },
        'tuning': {
            'n_trials': 3,  # Very few trials
            'timeout': 30,  # 30 seconds max
            'cv_folds': 2,
        },
        'evaluation': {
            'cv_folds': 2,
        }
    }
    
    start_time = time.time()
    
    automl = AutoML(
        target='target',
        task_type='classification',
        time_limit=120,  # 2 minutes max
        config=fast_config,
        verbose=True
    )
    
    print("🏃 Starting training (should complete in ~30-60 seconds)...")
    
    try:
        # Fit the model
        automl.fit(data)
        
        training_time = time.time() - start_time
        
        # Get results
        results = automl.get_results()
        
        print("\n✅ FAST TRAINING SUCCESSFUL!")
        print(f"⏱️  Total time: {training_time:.1f} seconds")
        print(f"🏆 Best model: {results['best_model_name']}")
        print(f"📊 Score: {results['best_score']:.3f}")
        
        # Test prediction
        predictions = automl.predict(data.head(5))
        print(f"🔮 Sample predictions: {predictions}")
        
        if training_time < 120:  # Under 2 minutes
            print(f"\n🎉 SUCCESS: Training completed in {training_time:.1f}s (target: <120s)")
            return True
        else:
            print(f"\n⚠️  WARNING: Training took {training_time:.1f}s (target: <120s)")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🏎️  FAST AutoML Performance Test")
    print("This test should complete in under 2 minutes")
    print("-" * 50)
    
    success = test_fast_training()
    
    if success:
        print("\n✅ Performance optimization successful!")
        print("You can now train models much faster.")
    else:
        print("\n❌ Performance test failed. Check the errors above.")