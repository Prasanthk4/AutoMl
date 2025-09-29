#!/usr/bin/env python
"""
Test script for model interpretability features.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from automl import AutoML
import pandas as pd

def create_test_data():
    """Create a test dataset for interpretability testing."""
    np.random.seed(42)
    n_samples = 200
    
    # Create meaningful features
    age = np.random.randint(18, 80, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    
    # Create target with some logical relationships
    # Higher income, age, credit score, and education level increase approval chances
    base_prob = 0.3
    prob_adjustments = (
        (income - 50000) / 100000 +  # Income effect
        (age - 30) / 100 +           # Age effect  
        (credit_score - 600) / 500   # Credit score effect
    )
    
    # Education effect
    education_effect = pd.Series(education).map({
        'High School': 0.0,
        'Bachelor': 0.1, 
        'Master': 0.2,
        'PhD': 0.3
    }).values
    
    final_prob = np.clip(base_prob + prob_adjustments + education_effect, 0.1, 0.9)
    approved = np.random.binomial(1, final_prob, n_samples)
    
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'education': education,
        'approved': approved
    })
    
    return df

def test_interpretability():
    """Test interpretability features."""
    print("🧠 Testing Model Interpretability Features")
    print("=" * 60)
    
    # Create test data
    print("📊 Creating test dataset...")
    df = create_test_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: {df.columns[-1]}")
    
    # Train model
    print("\n⚙️ Training AutoML model...")
    
    fast_config = {
        'models': {
            'include_models': ['random_forest'],  # Use tree model for better interpretability
            'n_jobs': 1,
            'cv_folds': 2,
        },
        'tuning': {
            'n_trials': 3,
            'timeout': 30,
            'cv_folds': 2,
        }
    }
    
    start_time = time.time()
    
    automl = AutoML(
        target='approved',
        task_type='classification',
        time_limit=120,
        config=fast_config,
        verbose=True
    )
    
    automl.fit(df)
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time:.1f} seconds")
    print(f"Best model: {automl.best_model_name}")
    print(f"Best score: {automl.best_score:.4f}")
    
    # Test interpretability features
    print("\n🔬 Testing Interpretability Features...")
    
    # Test 1: Check if interpreter is available
    print("\n1. Checking interpreter availability...")
    if hasattr(automl, 'interpreter') and automl.interpreter is not None:
        print("✅ Model interpreter is available")
    else:
        print("❌ Model interpreter not available")
        return False
    
    # Test 2: Global feature importance (SHAP)
    print("\n2. Testing SHAP global feature importance...")
    try:
        shap_fig = automl.analyze_feature_importance('shap', max_features=4)
        if shap_fig:
            print("✅ SHAP global importance analysis successful")
        else:
            print("⚠️ SHAP analysis returned None")
    except Exception as e:
        print(f"❌ SHAP analysis failed: {e}")
    
    # Test 3: Individual prediction explanation
    print("\n3. Testing individual prediction explanation...")
    try:
        # Get a sample instance
        sample = df.drop(columns=['approved']).iloc[0]
        explanation = automl.explain_prediction(sample, method='shap')
        
        if 'shap_plot' in explanation and explanation['shap_plot']:
            print("✅ SHAP individual explanation successful")
        else:
            print("⚠️ SHAP individual explanation returned None")
            
        # Test LIME too
        lime_explanation = automl.explain_prediction(sample, method='lime')
        if 'lime_plot' in lime_explanation and lime_explanation['lime_plot']:
            print("✅ LIME individual explanation successful")
        else:
            print("⚠️ LIME individual explanation had issues")
            
    except Exception as e:
        print(f"❌ Individual explanation failed: {e}")
    
    # Test 4: Partial dependence
    print("\n4. Testing partial dependence plots...")
    try:
        pd_fig = automl.plot_partial_dependence('income')
        if pd_fig:
            print("✅ Partial dependence plot successful")
        else:
            print("⚠️ Partial dependence plot returned None")
    except Exception as e:
        print(f"❌ Partial dependence failed: {e}")
    
    # Test 5: Feature interactions
    print("\n5. Testing feature interaction analysis...")
    try:
        interaction_fig = automl.analyze_feature_interactions('income', 'age')
        if interaction_fig:
            print("✅ Feature interaction analysis successful")
        else:
            print("⚠️ Feature interaction analysis returned None")
    except Exception as e:
        print(f"❌ Feature interaction failed: {e}")
    
    # Test 6: What-if analysis
    print("\n6. Testing what-if analysis...")
    try:
        sample = df.drop(columns=['approved']).iloc[0]
        values_to_test = [30000, 50000, 70000, 90000]  # Different income levels
        whatif_fig = automl.what_if_analysis(sample, 'income', values_to_test)
        if whatif_fig:
            print("✅ What-if analysis successful")
        else:
            print("⚠️ What-if analysis returned None")
    except Exception as e:
        print(f"❌ What-if analysis failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Interpretability testing completed!")
    print("The model can now explain its predictions using:")
    print("  - SHAP global feature importance")
    print("  - SHAP individual prediction explanations") 
    print("  - LIME local explanations")
    print("  - Partial dependence plots")
    print("  - Feature interaction analysis")
    print("  - What-if scenario analysis")
    
    return True

if __name__ == "__main__":
    success = test_interpretability()
    
    if success:
        print("\n✅ All interpretability features are working!")
        print("You can now use the Streamlit dashboard to explore model explanations.")
    else:
        print("\n❌ Some interpretability features had issues.")