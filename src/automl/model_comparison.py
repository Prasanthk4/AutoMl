"""
Model Comparison & A/B Testing Framework

This module provides comprehensive model comparison capabilities including:
- Side-by-side model performance comparison
- Statistical significance testing
- A/B testing framework
- Cross-validation comparison
- Bootstrap confidence intervals
- Interactive visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import validation_curve, learning_curve
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time
from pathlib import Path
import pickle
from dataclasses import dataclass
from collections import defaultdict
from .experiment_tracking import ExperimentTracker, ExperimentConfig, ExperimentMetrics

warnings.filterwarnings('ignore')

@dataclass
class ModelResult:
    """Container for model training and evaluation results"""
    model_name: str
    model: Any
    cv_scores: np.ndarray
    train_time: float
    predict_time: float
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_importance: Optional[pd.DataFrame] = None
    predictions: Optional[np.ndarray] = None
    prediction_probabilities: Optional[np.ndarray] = None

@dataclass
class ComparisonResult:
    """Container for model comparison results"""
    model_results: List[ModelResult]
    statistical_tests: Dict[str, Dict[str, float]]
    best_model: str
    comparison_metrics: pd.DataFrame
    significance_level: float = 0.05

class ModelComparison:
    """
    Advanced model comparison system with statistical testing and visualization.
    
    Features:
    - Multiple model training and evaluation
    - Cross-validation comparison
    - Statistical significance testing
    - Performance visualization
    - Bootstrap confidence intervals
    """
    
    def __init__(self, task_type: str = 'classification', cv_folds: int = 5, 
                 random_state: int = 42, significance_level: float = 0.05,
                 experiment_tracking: bool = True):
        """
        Initialize model comparison framework.
        
        Args:
            task_type: 'classification' or 'regression'
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            significance_level: Statistical significance level (default 0.05)
        """
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.significance_level = significance_level
        self.experiment_tracking = experiment_tracking
        
        # Set up cross-validation
        if task_type == 'classification':
            self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            self.primary_metric = 'accuracy'
        else:
            self.cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            self.primary_metric = 'r2'
            
        self.logger = logging.getLogger(__name__)
        self.model_results = []
        
        # Experiment tracking
        if self.experiment_tracking:
            self.experiment_tracker = ExperimentTracker()
            self.comparison_experiment_id = None
        else:
            self.experiment_tracker = None
            self.comparison_experiment_id = None
        
    def add_model(self, model, model_name: str, hyperparameters: Dict[str, Any] = None) -> None:
        """
        Add a model to the comparison.
        
        Args:
            model: Trained model object
            model_name: Name for the model
            hyperparameters: Dictionary of hyperparameters used
        """
        if hyperparameters is None:
            hyperparameters = {}
            
        # Store model info
        model_info = {
            'model': model,
            'name': model_name,
            'hyperparameters': hyperparameters
        }
        
        if not hasattr(self, 'models'):
            self.models = []
        self.models.append(model_info)
        
    def compare_models(self, X, y, X_test=None, y_test=None) -> ComparisonResult:
        """
        Compare all added models using cross-validation and statistical testing.
        
        Args:
            X: Training features
            y: Training target
            X_test: Optional test features
            y_test: Optional test target
            
        Returns:
            ComparisonResult with detailed comparison
        """
        if not hasattr(self, 'models') or not self.models:
            raise ValueError("No models added for comparison. Use add_model() first.")
            
        self.logger.info(f"Comparing {len(self.models)} models using {self.cv_folds}-fold CV")
        
        model_results = []
        
        for model_info in self.models:
            model = model_info['model']
            model_name = model_info['name']
            hyperparams = model_info['hyperparameters']
            
            self.logger.info(f"Evaluating model: {model_name}")
            
            # Time training
            start_time = time.time()
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X, y, cv=self.cv, 
                scoring=self.primary_metric,
                n_jobs=-1
            )
            
            train_time = time.time() - start_time
            
            # Fit model on full training data
            model.fit(X, y)
            
            # Time prediction
            start_pred = time.time()
            if X_test is not None:
                predictions = model.predict(X_test)
                pred_probas = None
                if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                    pred_probas = model.predict_proba(X_test)
            else:
                predictions = model.predict(X)
                pred_probas = None
                if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                    pred_probas = model.predict_proba(X)
                    
            predict_time = time.time() - start_pred
            
            # Calculate detailed metrics
            if X_test is not None and y_test is not None:
                test_target = y_test
                test_predictions = predictions
            else:
                test_target = y
                test_predictions = predictions
                
            metrics = self._calculate_metrics(test_target, test_predictions, pred_probas)
            
            # Feature importance (if available)
            feature_importance = self._extract_feature_importance(model, X.columns if hasattr(X, 'columns') else None)
            
            # Create result object
            result = ModelResult(
                model_name=model_name,
                model=model,
                cv_scores=cv_scores,
                train_time=train_time,
                predict_time=predict_time,
                metrics=metrics,
                hyperparameters=hyperparams,
                feature_importance=feature_importance,
                predictions=test_predictions,
                prediction_probabilities=pred_probas
            )
            
            model_results.append(result)
            
        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(model_results)
        
        # Create comparison metrics DataFrame
        comparison_df = self._create_comparison_dataframe(model_results)
        
        # Determine best model
        best_model = self._determine_best_model(model_results)
        
        self.model_results = model_results
        
        return ComparisonResult(
            model_results=model_results,
            statistical_tests=statistical_tests,
            best_model=best_model,
            comparison_metrics=comparison_df,
            significance_level=self.significance_level
        )
        
    def _calculate_metrics(self, y_true, y_pred, y_proba=None) -> Dict[str, float]:
        """Calculate comprehensive metrics for model evaluation"""
        metrics = {}
        
        if self.task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC (for binary classification or with probabilities)
            if y_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:  # Binary classification
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    else:  # Multi-class
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                except Exception:
                    metrics['roc_auc'] = np.nan
            else:
                metrics['roc_auc'] = np.nan
                
        else:  # Regression
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            
            # Mean Absolute Percentage Error (MAPE)
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                metrics['mape'] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            else:
                metrics['mape'] = np.nan
                
        return metrics
        
    def _extract_feature_importance(self, model, feature_names=None) -> Optional[pd.DataFrame]:
        """Extract feature importance from model if available"""
        try:
            importance = None
            
            # Try different ways to get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                coef = model.coef_
                if len(coef.shape) > 1:  # Multi-class
                    importance = np.mean(np.abs(coef), axis=0)
                else:
                    importance = np.abs(coef)
            
            if importance is not None:
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(len(importance))]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                return importance_df
                
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            
        return None
        
    def _perform_statistical_tests(self, model_results: List[ModelResult]) -> Dict[str, Dict[str, float]]:
        """Perform statistical significance tests between models"""
        tests = {}
        
        if len(model_results) < 2:
            return tests
            
        # Pairwise comparisons
        for i in range(len(model_results)):
            for j in range(i + 1, len(model_results)):
                model1 = model_results[i]
                model2 = model_results[j]
                
                scores1 = model1.cv_scores
                scores2 = model2.cv_scores
                
                test_name = f"{model1.model_name} vs {model2.model_name}"
                
                # Paired t-test
                try:
                    t_stat, t_pval = ttest_rel(scores1, scores2)
                    tests[test_name] = {
                        'paired_ttest_statistic': t_stat,
                        'paired_ttest_pvalue': t_pval,
                        'significant': t_pval < self.significance_level
                    }
                except Exception:
                    tests[test_name] = {
                        'paired_ttest_statistic': np.nan,
                        'paired_ttest_pvalue': np.nan,
                        'significant': False
                    }
                
                # Wilcoxon signed-rank test (non-parametric)
                try:
                    w_stat, w_pval = wilcoxon(scores1, scores2)
                    tests[test_name]['wilcoxon_statistic'] = w_stat
                    tests[test_name]['wilcoxon_pvalue'] = w_pval
                    tests[test_name]['wilcoxon_significant'] = w_pval < self.significance_level
                except Exception:
                    tests[test_name]['wilcoxon_statistic'] = np.nan
                    tests[test_name]['wilcoxon_pvalue'] = np.nan
                    tests[test_name]['wilcoxon_significant'] = False
                
                # Effect size (Cohen's d)
                try:
                    pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                                        (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                                       (len(scores1) + len(scores2) - 2))
                    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                    tests[test_name]['cohens_d'] = cohens_d
                    
                    # Effect size interpretation
                    if abs(cohens_d) < 0.2:
                        effect_size = 'small'
                    elif abs(cohens_d) < 0.5:
                        effect_size = 'small'
                    elif abs(cohens_d) < 0.8:
                        effect_size = 'medium'
                    else:
                        effect_size = 'large'
                    tests[test_name]['effect_size'] = effect_size
                    
                except Exception:
                    tests[test_name]['cohens_d'] = np.nan
                    tests[test_name]['effect_size'] = 'unknown'
                    
        return tests
        
    def _create_comparison_dataframe(self, model_results: List[ModelResult]) -> pd.DataFrame:
        """Create a comprehensive comparison DataFrame"""
        comparison_data = []
        
        for result in model_results:
            data = {
                'Model': result.model_name,
                'CV Mean': result.cv_scores.mean(),
                'CV Std': result.cv_scores.std(),
                'Train Time (s)': result.train_time,
                'Predict Time (s)': result.predict_time,
            }
            
            # Add all metrics
            data.update(result.metrics)
            
            comparison_data.append(data)
            
        df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric (descending for most metrics, ascending for error metrics)
        if self.task_type == 'classification':
            df = df.sort_values('CV Mean', ascending=False)
        else:
            # For regression, sort by R² (descending) or MSE (ascending)
            if 'r2' in df.columns:
                df = df.sort_values('r2', ascending=False)
            elif 'mse' in df.columns:
                df = df.sort_values('mse', ascending=True)
            else:
                df = df.sort_values('CV Mean', ascending=False)
                
        return df.round(4)
        
    def _determine_best_model(self, model_results: List[ModelResult]) -> str:
        """Determine the best model based on cross-validation scores"""
        if not model_results:
            return None
            
        if self.task_type == 'classification':
            # Higher is better for classification
            best_idx = np.argmax([result.cv_scores.mean() for result in model_results])
        else:
            # For regression, higher R² is better (assuming R² is the primary metric)
            # If using negative metrics like neg_mse, higher (less negative) is better
            best_idx = np.argmax([result.cv_scores.mean() for result in model_results])
            
        return model_results[best_idx].model_name
        
    def create_comparison_report(self, comparison_result: ComparisonResult) -> str:
        """Create a detailed text report of the model comparison"""
        report = []
        report.append("=" * 60)
        report.append("MODEL COMPARISON REPORT")
        report.append("=" * 60)
        
        report.append(f"\nTask Type: {self.task_type.title()}")
        report.append(f"Cross-Validation Folds: {self.cv_folds}")
        report.append(f"Significance Level: {self.significance_level}")
        report.append(f"Best Model: {comparison_result.best_model}")
        
        report.append("\n" + "-" * 40)
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        # Performance table
        df = comparison_result.comparison_metrics
        report.append(df.to_string(index=False))
        
        report.append("\n" + "-" * 40)
        report.append("STATISTICAL SIGNIFICANCE TESTS")
        report.append("-" * 40)
        
        for test_name, results in comparison_result.statistical_tests.items():
            report.append(f"\n{test_name}:")
            report.append(f"  Paired t-test p-value: {results.get('paired_ttest_pvalue', 'N/A'):.6f}")
            report.append(f"  Wilcoxon p-value: {results.get('wilcoxon_pvalue', 'N/A'):.6f}")
            report.append(f"  Effect size (Cohen's d): {results.get('cohens_d', 'N/A'):.4f} ({results.get('effect_size', 'unknown')})")
            report.append(f"  Statistically significant: {results.get('significant', False)}")
            
        return "\n".join(report)

class ABTestFramework:
    """
    A/B Testing framework for model comparison with proper statistical controls.
    
    Features:
    - Controlled A/B testing setup
    - Statistical power analysis
    - Multiple testing correction
    - Confidence intervals
    - Effect size calculation
    """
    
    def __init__(self, significance_level: float = 0.05, power: float = 0.8, 
                 random_state: int = 42, confidence_level: float = None):
        """
        Initialize A/B testing framework.
        
        Args:
            significance_level: Type I error rate (alpha)
            power: Statistical power (1 - beta)
            random_state: Random state for reproducibility
            confidence_level: Alternative way to specify significance level (1 - alpha)
        """
        # Handle confidence_level parameter (alternative to significance_level)
        if confidence_level is not None:
            self.significance_level = 1.0 - confidence_level
        else:
            self.significance_level = significance_level
            
        self.power = power
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        np.random.seed(random_state)
        
    def design_ab_test(self, X, y, test_size: float = 0.3, 
                      stratify: bool = True) -> Tuple[Tuple, Tuple]:
        """
        Design A/B test by splitting data into control and treatment groups.
        
        Args:
            X: Features
            y: Target variable
            test_size: Fraction of data for treatment group
            stratify: Whether to stratify the split
            
        Returns:
            ((X_control, y_control), (X_treatment, y_treatment))
        """
        from sklearn.model_selection import train_test_split
        
        stratify_param = y if stratify else None
        
        X_control, X_treatment, y_control, y_treatment = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=stratify_param
        )
        
        self.logger.info(f"A/B test designed: Control={len(X_control)}, Treatment={len(X_treatment)}")
        
        return (X_control, y_control), (X_treatment, y_treatment)
        
        
    def calculate_sample_size(self, effect_size: float, baseline_rate: float = 0.5) -> int:
        """
        Calculate required sample size for A/B test.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            baseline_rate: Baseline conversion/success rate
            
        Returns:
            Required sample size per group
        """
        from scipy.stats import norm
        
        # Z-scores for alpha and beta
        z_alpha = norm.ppf(1 - self.significance_level / 2)
        z_beta = norm.ppf(self.power)
        
        # Sample size calculation for two-sample t-test
        n = ((z_alpha + z_beta) ** 2 * 2) / (effect_size ** 2)
        
        return int(np.ceil(n))
    
    def run_ab_test(self, *args, **kwargs):
        """
        Run A/B test - supports multiple signatures for flexibility.
        """
        # Check if called with score arrays (new signature for test compatibility)
        if len(args) == 3 and isinstance(args[0], (list, np.ndarray)) and isinstance(args[1], (list, np.ndarray)):
            return self._run_ab_test_scores(args[0], args[1], args[2])
        # Original model-based signature
        elif len(args) >= 6:
            return self._run_ab_test_models(*args, **kwargs)
        else:
            raise ValueError("Invalid arguments for run_ab_test")
    
    def _run_ab_test_scores(self, control_scores: np.ndarray, treatment_scores: np.ndarray, 
                           metric_name: str) -> Dict[str, Any]:
        """
        Run A/B test on pre-computed score arrays.
        
        Args:
            control_scores: Control group scores
            treatment_scores: Treatment group scores
            metric_name: Name of the metric
            
        Returns:
            Dictionary with test results including statistical_power
        """
        control_scores = np.array(control_scores)
        treatment_scores = np.array(treatment_scores)
        
        # Basic statistics
        mean_control = np.mean(control_scores)
        mean_treatment = np.mean(treatment_scores)
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(control_scores, treatment_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(control_scores, ddof=1) + np.var(treatment_scores, ddof=1)) / 2)
        cohens_d = (mean_treatment - mean_control) / pooled_std if pooled_std > 0 else 0
        
        # Statistical power calculation (post-hoc)
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - self.significance_level / 2)
        z_beta = norm.ppf(self.power)
        
        # Estimated power based on observed effect size and sample sizes
        n = min(len(control_scores), len(treatment_scores))
        observed_power = self._calculate_statistical_power(cohens_d, n)
        
        results = {
            'metric': metric_name,
            'control_mean': mean_control,
            'treatment_mean': mean_treatment,
            'difference': mean_treatment - mean_control,
            'relative_improvement': ((mean_treatment - mean_control) / mean_control) * 100 if mean_control != 0 else np.inf,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'cohens_d': cohens_d,
            'statistical_power': observed_power,  # This is what the test expects
            'sample_size_control': len(control_scores),
            'sample_size_treatment': len(treatment_scores)
        }
        
        return results
    
    def _calculate_statistical_power(self, effect_size: float, n: int) -> float:
        """
        Calculate statistical power for given effect size and sample size.
        """
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - self.significance_level / 2)
        z_beta = z_alpha - effect_size * np.sqrt(n / 2)
        power = 1 - norm.cdf(z_beta)
        
        return max(0, min(1, power))  # Clamp between 0 and 1
    
    def _run_ab_test_models(self, model_a, model_b, X_control, y_control, 
                           X_treatment, y_treatment, metric: str = 'accuracy') -> Dict[str, Any]:
        """
        Run A/B test comparing two models (original implementation).
        
        Args:
            model_a: Control model
            model_b: Treatment model
            X_control: Control features
            y_control: Control target
            X_treatment: Treatment features  
            y_treatment: Treatment target
            metric: Metric to compare ('accuracy', 'f1', 'r2', etc.)
            
        Returns:
            Dictionary with test results
        """
        # Train models
        model_a.fit(X_control, y_control)
        model_b.fit(X_treatment, y_treatment)
        
        # Get predictions
        pred_a = model_a.predict(X_control)
        pred_b = model_b.predict(X_treatment)
        
        # Calculate metrics
        if metric == 'accuracy':
            score_a = accuracy_score(y_control, pred_a)
            score_b = accuracy_score(y_treatment, pred_b)
        elif metric == 'f1':
            score_a = f1_score(y_control, pred_a, average='weighted')
            score_b = f1_score(y_treatment, pred_b, average='weighted')
        elif metric == 'r2':
            score_a = r2_score(y_control, pred_a)
            score_b = r2_score(y_treatment, pred_b)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        scores_a_boot = []
        scores_b_boot = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample A
            idx_a = np.random.choice(len(y_control), len(y_control), replace=True)
            if metric == 'accuracy':
                score_boot_a = accuracy_score(y_control.iloc[idx_a], pred_a[idx_a])
            elif metric == 'f1':
                score_boot_a = f1_score(y_control.iloc[idx_a], pred_a[idx_a], average='weighted')
            elif metric == 'r2':
                score_boot_a = r2_score(y_control.iloc[idx_a], pred_a[idx_a])
            scores_a_boot.append(score_boot_a)
            
            # Bootstrap sample B  
            idx_b = np.random.choice(len(y_treatment), len(y_treatment), replace=True)
            if metric == 'accuracy':
                score_boot_b = accuracy_score(y_treatment.iloc[idx_b], pred_b[idx_b])
            elif metric == 'f1':
                score_boot_b = f1_score(y_treatment.iloc[idx_b], pred_b[idx_b], average='weighted')
            elif metric == 'r2':
                score_boot_b = r2_score(y_treatment.iloc[idx_b], pred_b[idx_b])
            scores_b_boot.append(score_boot_b)
            
        # Statistical test
        t_stat, p_value = stats.ttest_ind(scores_a_boot, scores_b_boot)
        
        # Effect size
        pooled_std = np.sqrt((np.var(scores_a_boot, ddof=1) + np.var(scores_b_boot, ddof=1)) / 2)
        cohens_d = (np.mean(scores_b_boot) - np.mean(scores_a_boot)) / pooled_std
        
        # Confidence intervals
        alpha = self.significance_level
        ci_a = np.percentile(scores_a_boot, [100 * alpha/2, 100 * (1 - alpha/2)])
        ci_b = np.percentile(scores_b_boot, [100 * alpha/2, 100 * (1 - alpha/2)])
        
        results = {
            'metric': metric,
            'model_a_score': score_a,
            'model_b_score': score_b,
            'difference': score_b - score_a,
            'relative_improvement': ((score_b - score_a) / score_a) * 100 if score_a != 0 else np.inf,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'cohens_d': cohens_d,
            'confidence_interval_a': ci_a,
            'confidence_interval_b': ci_b,
            'bootstrap_scores_a': scores_a_boot,
            'bootstrap_scores_b': scores_b_boot,
            'sample_size_a': len(X_control),
            'sample_size_b': len(X_treatment)
        }
        
        return results
        
    def create_ab_test_report(self, results: Dict[str, Any]) -> str:
        """Create a detailed A/B test report"""
        report = []
        report.append("=" * 50)
        report.append("A/B TEST RESULTS")
        report.append("=" * 50)
        
        report.append(f"\nMetric: {results['metric'].title()}")
        report.append(f"Model A Score: {results['model_a_score']:.4f}")
        report.append(f"Model B Score: {results['model_b_score']:.4f}")
        report.append(f"Difference: {results['difference']:.4f}")
        report.append(f"Relative Improvement: {results['relative_improvement']:.2f}%")
        
        report.append(f"\nSample Sizes:")
        report.append(f"  Group A (Control): {results['sample_size_a']}")
        report.append(f"  Group B (Treatment): {results['sample_size_b']}")
        
        report.append(f"\nStatistical Tests:")
        report.append(f"  P-value: {results['p_value']:.6f}")
        report.append(f"  Significant: {results['significant']}")
        report.append(f"  Effect Size (Cohen's d): {results['cohens_d']:.4f}")
        
        ci_a = results['confidence_interval_a']
        ci_b = results['confidence_interval_b']
        report.append(f"\nConfidence Intervals (95%):")
        report.append(f"  Model A: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
        report.append(f"  Model B: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
        
        # Interpretation
        report.append(f"\nInterpretation:")
        if results['significant']:
            direction = "better" if results['difference'] > 0 else "worse"
            report.append(f"  Model B performs significantly {direction} than Model A")
        else:
            report.append(f"  No significant difference detected between models")
            
        if abs(results['cohens_d']) >= 0.8:
            effect = "large"
        elif abs(results['cohens_d']) >= 0.5:
            effect = "medium"
        elif abs(results['cohens_d']) >= 0.2:
            effect = "small"
        else:
            effect = "negligible"
        report.append(f"  Effect size: {effect}")
        
        return "\n".join(report)