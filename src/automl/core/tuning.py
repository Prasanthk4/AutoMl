"""
Hyperparameter tuning for AutoML system.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
import warnings

warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """
    Simple hyperparameter tuner using RandomizedSearchCV.
    
    For now, uses scikit-learn's RandomizedSearchCV.
    Can be extended with Optuna later.
    """
    
    def __init__(self, task_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            task_type: 'classification' or 'regression'
            config: Configuration dictionary
        """
        self.task_type = task_type.lower()
        self.config = config or {}
        
        # Define parameter grids for different models
        self.param_grids = self._get_param_grids()
        
    def _get_param_grids(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter grids for hyperparameter tuning."""
        grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            },
            
            'linear_regression': {
                # Linear regression has no hyperparameters to tune
            },
            
            'xgboost': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
            },
            
            'lightgbm': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 4, 5, 6, 7, 8, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'num_leaves': [15, 31, 63, 127, 255],
                'min_child_samples': [5, 10, 20, 30, 50]
            },
            
            'catboost': {
                'iterations': [50, 100, 200, 300],
                'depth': [3, 4, 5, 6, 7, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7, 9]
            }
        }
        
        return grids
        
    def tune_model(
        self,
        model_name: str,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: Optional[float] = None
    ) -> Tuple[Any, Dict[str, Any], float]:
        """
        Tune hyperparameters for a model.
        
        Args:
            model_name: Name of the model
            model: The model instance to tune
            X: Feature matrix
            y: Target vector
            time_limit: Time limit for tuning (not used in this simple version)
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        if model_name not in self.param_grids:
            # Return original model if no param grid available
            return model, {}, self._score_model(model, X, y)
            
        param_grid = self.param_grids[model_name]
        
        # Skip if no parameters to tune
        if not param_grid:
            return model, {}, self._score_model(model, X, y)
            
        # Setup scoring
        scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        
        # Configure search
        n_iter = min(50, len(param_grid) * 5)  # Reasonable number of iterations
        cv_folds = self.config.get('cv_folds', 3)
        
        try:
            # Perform randomized search with timeout consideration
            # Reduce iterations if time is limited
            if time_limit and time_limit < 120:  # Less than 2 minutes
                n_iter = min(n_iter, 10)
                cv_folds = min(cv_folds, 3)
            
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring,
                random_state=42,
                n_jobs=1  # Use single job to avoid hanging
            )
            
            search.fit(X, y)
            
            return search.best_estimator_, search.best_params_, search.best_score_
            
        except Exception as e:
            # Fallback to original model if tuning fails
            print(f"Hyperparameter tuning failed for {model_name}: {e}")
            return model, {}, self._score_model(model, X, y)
            
    def _score_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> float:
        """Score a model using cross-validation."""
        scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        
        try:
            scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
            return scores.mean()
        except:
            # Fallback scoring
            model.fit(X, y)
            y_pred = model.predict(X)
            
            if self.task_type == 'classification':
                return accuracy_score(y, y_pred)
            else:
                return r2_score(y, y_pred)