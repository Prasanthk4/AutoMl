"""
Model registry and training system for AutoML.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score, train_test_split
import warnings

warnings.filterwarnings('ignore')

# Optional advanced models
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class ModelRegistry:
    """
    Registry of machine learning models for classification and regression.
    
    Supports:
    - Random Forest
    - Logistic/Linear Regression
    - SVM (optional)
    - XGBoost (if available)
    - LightGBM (if available) 
    - CatBoost (if available)
    """
    
    def __init__(self, task_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model registry.
        
        Args:
            task_type: 'classification' or 'regression'
            config: Configuration dictionary
        """
        self.task_type = task_type.lower()
        self.config = config or {}
        
        if self.task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be 'classification' or 'regression'")
            
        self.models = self._initialize_models()
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all available models."""
        models = {}
        random_state = self.config.get('random_state', 42)
        n_jobs = self.config.get('n_jobs', -1)
        
        if self.task_type == 'classification':
            # Random Forest
            models['random_forest'] = RandomForestClassifier(
                n_estimators=self.config.get('random_forest', {}).get('n_estimators', 100),
                max_depth=self.config.get('random_forest', {}).get('max_depth', None),
                min_samples_split=self.config.get('random_forest', {}).get('min_samples_split', 2),
                min_samples_leaf=self.config.get('random_forest', {}).get('min_samples_leaf', 1),
                random_state=random_state,
                n_jobs=n_jobs
            )
            
            # Logistic Regression
            models['logistic_regression'] = LogisticRegression(
                random_state=random_state,
                n_jobs=n_jobs,
                max_iter=1000
            )
            
            # XGBoost
            if HAS_XGBOOST:
                models['xgboost'] = xgb.XGBClassifier(
                    n_estimators=self.config.get('xgboost', {}).get('n_estimators', 100),
                    max_depth=self.config.get('xgboost', {}).get('max_depth', 6),
                    learning_rate=self.config.get('xgboost', {}).get('learning_rate', 0.1),
                    subsample=self.config.get('xgboost', {}).get('subsample', 1.0),
                    random_state=random_state,
                    n_jobs=n_jobs,
                    verbosity=0
                )
                
            # LightGBM
            if HAS_LIGHTGBM:
                models['lightgbm'] = lgb.LGBMClassifier(
                    n_estimators=self.config.get('lightgbm', {}).get('n_estimators', 100),
                    max_depth=self.config.get('lightgbm', {}).get('max_depth', -1),
                    learning_rate=self.config.get('lightgbm', {}).get('learning_rate', 0.1),
                    num_leaves=self.config.get('lightgbm', {}).get('num_leaves', 31),
                    random_state=random_state,
                    n_jobs=n_jobs,
                    verbosity=-1
                )
                
            # CatBoost
            if HAS_CATBOOST:
                models['catboost'] = CatBoostClassifier(
                    iterations=self.config.get('catboost', {}).get('iterations', 100),
                    depth=self.config.get('catboost', {}).get('depth', 6),
                    learning_rate=self.config.get('catboost', {}).get('learning_rate', 0.1),
                    random_state=random_state,
                    verbose=False
                )
                
        else:  # regression
            # Random Forest
            models['random_forest'] = RandomForestRegressor(
                n_estimators=self.config.get('random_forest', {}).get('n_estimators', 100),
                max_depth=self.config.get('random_forest', {}).get('max_depth', None),
                min_samples_split=self.config.get('random_forest', {}).get('min_samples_split', 2),
                min_samples_leaf=self.config.get('random_forest', {}).get('min_samples_leaf', 1),
                random_state=random_state,
                n_jobs=n_jobs
            )
            
            # Linear Regression
            models['linear_regression'] = LinearRegression(n_jobs=n_jobs)
            
            # XGBoost
            if HAS_XGBOOST:
                models['xgboost'] = xgb.XGBRegressor(
                    n_estimators=self.config.get('xgboost', {}).get('n_estimators', 100),
                    max_depth=self.config.get('xgboost', {}).get('max_depth', 6),
                    learning_rate=self.config.get('xgboost', {}).get('learning_rate', 0.1),
                    subsample=self.config.get('xgboost', {}).get('subsample', 1.0),
                    random_state=random_state,
                    n_jobs=n_jobs,
                    verbosity=0
                )
                
            # LightGBM
            if HAS_LIGHTGBM:
                models['lightgbm'] = lgb.LGBMRegressor(
                    n_estimators=self.config.get('lightgbm', {}).get('n_estimators', 100),
                    max_depth=self.config.get('lightgbm', {}).get('max_depth', -1),
                    learning_rate=self.config.get('lightgbm', {}).get('learning_rate', 0.1),
                    num_leaves=self.config.get('lightgbm', {}).get('num_leaves', 31),
                    random_state=random_state,
                    n_jobs=n_jobs,
                    verbosity=-1
                )
                
            # CatBoost
            if HAS_CATBOOST:
                models['catboost'] = CatBoostRegressor(
                    iterations=self.config.get('catboost', {}).get('iterations', 100),
                    depth=self.config.get('catboost', {}).get('depth', 6),
                    learning_rate=self.config.get('catboost', {}).get('learning_rate', 0.1),
                    random_state=random_state,
                    verbose=False
                )
                
        # Filter models based on configuration
        include_models = self.config.get('include_models', 'auto')
        exclude_models = self.config.get('exclude_models', [])
        
        if include_models != 'auto' and include_models != 'all':
            models = {k: v for k, v in models.items() if k in include_models}
            
        if exclude_models:
            models = {k: v for k, v in models.items() if k not in exclude_models}
            
        return models
        
    def get_models(self) -> Dict[str, Any]:
        """Get all available models."""
        return self.models.copy()
        
    def get_model(self, model_name: str) -> Any:
        """Get a specific model by name."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        return self.models[model_name]
        
    def train_model(
        self, 
        model_name: str,
        X: pd.DataFrame, 
        y: pd.Series,
        validation_split: float = 0.2,
        random_state: int = 42
    ) -> Any:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            X: Feature matrix
            y: Target vector
            validation_split: Fraction of data for validation
            random_state: Random seed
            
        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        
        # Split data for training if needed
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=random_state,
                stratify=y if self.task_type == 'classification' and y.nunique() > 1 else None
            )
        else:
            X_train, y_train = X, y
            
        # Train the model
        model.fit(X_train, y_train)
        
        return model
        
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())
        
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all models."""
        info = {}
        
        for name, model in self.models.items():
            info[name] = {
                'class': model.__class__.__name__,
                'module': model.__class__.__module__,
                'parameters': model.get_params()
            }
            
        return info