"""
Model evaluation and metrics for AutoML system.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, log_loss,
    matthews_corrcoef
)
from sklearn.model_selection import cross_val_score, cross_validate
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation system.
    
    Provides:
    - Cross-validation
    - Multiple evaluation metrics
    - Feature importance extraction
    - Performance comparisons
    """
    
    def __init__(self, task_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model evaluator.
        
        Args:
            task_type: 'classification' or 'regression'
            config: Configuration dictionary
        """
        self.task_type = task_type.lower()
        self.config = config or {}
        self.last_metrics = {}
        
    def cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: The model to evaluate
            X: Feature matrix
            y: Target vector
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with CV scores
        """
        # Determine primary scoring metric
        if self.task_type == 'classification':
            primary_scoring = 'accuracy'
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            
            # Add ROC-AUC for binary classification
            if y.nunique() == 2:
                scoring.append('roc_auc')
        else:
            primary_scoring = 'r2'
            scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
            
        try:
            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y,
                cv=cv_folds,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1
            )
            
            # Process results
            results = {}
            for metric, scores in cv_results.items():
                if metric.startswith('test_'):
                    metric_name = metric[5:]  # Remove 'test_' prefix
                    
                    # Handle negative scores (sklearn convention)
                    if metric_name.startswith('neg_'):
                        metric_name = metric_name[4:]  # Remove 'neg_' prefix
                        scores = -scores
                        
                    results[f'{metric_name}_mean'] = scores.mean()
                    results[f'{metric_name}_std'] = scores.std()
                    
            # Add primary score for easy access
            primary_key = f'{primary_scoring}_mean'
            if primary_key in results:
                results['mean_score'] = results[primary_key]
            else:
                results['mean_score'] = results['accuracy_mean']
                
            return results
            
        except Exception as e:
            print(f"Cross-validation failed: {e}")
            # Fallback to simple scoring
            model.fit(X, y)
            y_pred = model.predict(X)
            
            if self.task_type == 'classification':
                score = accuracy_score(y, y_pred)
            else:
                score = r2_score(y, y_pred)
                
            return {'mean_score': score, 'accuracy_mean': score}
            
    def evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            y_test: Test target (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Training metrics
        y_train_pred = model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, y_train_pred, 'train')
        metrics.update(train_metrics)
        
        # Test metrics if available
        if X_test is not None and y_test is not None:
            y_test_pred = model.predict(X_test)
            test_metrics = self._calculate_metrics(y_test, y_test_pred, 'test')
            metrics.update(test_metrics)
            
        # Cross-validation metrics
        cv_metrics = self.cross_validate(model, X_train, y_train)
        cv_metrics = {f'cv_{k}': v for k, v in cv_metrics.items()}
        metrics.update(cv_metrics)
        
        self.last_metrics = metrics
        return metrics
        
    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        prefix: str = ''
    ) -> Dict[str, float]:
        """Calculate metrics for true vs predicted values."""
        metrics = {}
        
        if prefix:
            prefix = f'{prefix}_'
            
        if self.task_type == 'classification':
            # Basic classification metrics
            metrics[f'{prefix}accuracy'] = accuracy_score(y_true, y_pred)
            metrics[f'{prefix}precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics[f'{prefix}recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics[f'{prefix}f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            # Matthews correlation coefficient
            try:
                metrics[f'{prefix}matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
            except:
                metrics[f'{prefix}matthews_corrcoef'] = 0.0
                
            # ROC-AUC for binary classification
            if y_true.nunique() == 2:
                try:
                    if hasattr(self, 'model') and hasattr(self.model, 'predict_proba'):
                        y_proba = self.model.predict_proba(self.last_X)[:, 1]
                        metrics[f'{prefix}roc_auc'] = roc_auc_score(y_true, y_proba)
                except:
                    pass
                    
        else:
            # Regression metrics
            metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)
            metrics[f'{prefix}mse'] = mean_squared_error(y_true, y_pred)
            metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)
            metrics[f'{prefix}rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Mean Absolute Percentage Error
            try:
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                metrics[f'{prefix}mape'] = mape
            except:
                metrics[f'{prefix}mape'] = np.inf
                
        return metrics
        
    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        importance_df = pd.DataFrame({'feature': feature_names})
        
        try:
            # Try different ways to get feature importance
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance_df['importance'] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                if len(model.coef_.shape) == 1:
                    importance_df['importance'] = np.abs(model.coef_)
                else:
                    # Multi-class case - use mean of absolute coefficients
                    importance_df['importance'] = np.mean(np.abs(model.coef_), axis=0)
            else:
                # Default: equal importance
                importance_df['importance'] = 1.0 / len(feature_names)
                
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
            importance_df['importance'] = 1.0 / len(feature_names)
            
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df = importance_df.reset_index(drop=True)
        
        return importance_df
        
    def get_classification_report(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Get detailed classification report."""
        if self.task_type != 'classification':
            raise ValueError("Classification report only available for classification tasks")
            
        return classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
    def get_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Get confusion matrix for classification."""
        if self.task_type != 'classification':
            raise ValueError("Confusion matrix only available for classification tasks")
            
        return confusion_matrix(y_true, y_pred)
        
    def compare_models(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.
        
        Args:
            models: Dictionary of models to compare
            X: Feature matrix
            y: Target vector
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for name, model in models.items():
            cv_results = self.cross_validate(model, X, y)
            
            result = {
                'model': name,
                'cv_score_mean': cv_results.get('mean_score', 0),
                'cv_score_std': cv_results.get('accuracy_std', 0) if self.task_type == 'classification' else cv_results.get('r2_std', 0)
            }
            
            # Add specific metrics
            if self.task_type == 'classification':
                result.update({
                    'cv_precision_mean': cv_results.get('precision_macro_mean', 0),
                    'cv_recall_mean': cv_results.get('recall_macro_mean', 0),
                    'cv_f1_mean': cv_results.get('f1_macro_mean', 0)
                })
            else:
                result.update({
                    'cv_r2_mean': cv_results.get('r2_mean', 0),
                    'cv_mse_mean': cv_results.get('mean_squared_error_mean', 0),
                    'cv_mae_mean': cv_results.get('mean_absolute_error_mean', 0)
                })
                
            results.append(result)
            
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('cv_score_mean', ascending=False)
        comparison_df = comparison_df.reset_index(drop=True)
        
        return comparison_df
        
    def get_last_metrics(self) -> Dict[str, float]:
        """Get the last calculated metrics."""
        return self.last_metrics.copy()