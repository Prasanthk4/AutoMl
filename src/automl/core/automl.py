"""
Core AutoML class that orchestrates the entire machine learning pipeline.
"""

import os
import time
import warnings
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
import plotly.graph_objects as go

from .data_ingestion import DataIngestion
from .preprocessing import Preprocessor
from .models import ModelRegistry
from .tuning import HyperparameterTuner
from .evaluation import ModelEvaluator
from ..utils.config import Config
from ..utils.logging import setup_logging
from ..interpretability import ModelInterpreter
from ..experiment_tracking import ExperimentTracker, ExperimentConfig, ExperimentMetrics

warnings.filterwarnings('ignore')


class AutoML:
    """
    Main AutoML class that automates the entire machine learning pipeline.
    
    This class provides an end-to-end automated machine learning solution that:
    - Automatically preprocesses data
    - Selects and trains multiple models
    - Performs hyperparameter tuning
    - Evaluates and compares models
    - Provides model interpretability
    
    Args:
        target (str): Name of the target variable
        task_type (str): Type of ML task ('classification' or 'regression')
        time_limit (int, optional): Maximum time for training in seconds. Defaults to 3600.
        validation_split (float, optional): Fraction of data for validation. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        config (dict, optional): Custom configuration dictionary. Defaults to None.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.
        
    Example:
        >>> automl = AutoML(
        ...     target='price',
        ...     task_type='regression',
        ...     time_limit=1800
        ... )
        >>> automl.fit('housing_data.csv')
        >>> predictions = automl.predict(test_data)
    """
    
    def __init__(
        self,
        target: str,
        task_type: str,
        time_limit: int = 300,  # reduced from 3600 (1 hour) to 5 minutes
        validation_split: float = 0.2,
        random_state: int = 42,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        experiment_tracking: bool = True
    ):
        self.target = target
        self.task_type = task_type.lower()
        self.time_limit = time_limit
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose
        self.experiment_tracking = experiment_tracking
        
        # Validate task type
        if self.task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be 'classification' or 'regression'")
            
        # Load configuration
        self.config = Config(config)
        
        # Setup logging
        self.logger = setup_logging(verbose=verbose)
        
        # Initialize components
        self.data_ingestion = DataIngestion(verbose=verbose)
        self.preprocessor = Preprocessor(config=self.config.preprocessing)
        self.model_registry = ModelRegistry(
            task_type=self.task_type,
            config=self.config.models
        )
        self.tuner = HyperparameterTuner(
            task_type=self.task_type,
            config=self.config.tuning
        )
        self.evaluator = ModelEvaluator(
            task_type=self.task_type,
            config=self.config.evaluation
        )
        
        # Experiment tracking
        if self.experiment_tracking:
            self.experiment_tracker = ExperimentTracker()
            self.current_experiment_id = None
        else:
            self.experiment_tracker = None
            self.current_experiment_id = None
        
        # Training state
        self.is_fitted = False
        self.training_time = None
        self.best_model = None
        self.best_score = None
        self.best_model_name = None
        self.feature_names = None
        self.training_history = []
        
        # Interpretability
        self.interpreter = None
        self.X_train = None
        self.y_train = None
        
    def fit(
        self, 
        data: Union[str, pd.DataFrame],
        test_data: Optional[Union[str, pd.DataFrame]] = None,
        feature_engineering: bool = False,
        **kwargs
    ) -> 'AutoML':
        """
        Fit the AutoML system on the provided data.
        
        Args:
            data: Training data as file path or DataFrame
            test_data: Optional test data for final evaluation
            feature_engineering: Whether to perform automated feature engineering
            **kwargs: Additional arguments passed to components
            
        Returns:
            self: Returns the fitted AutoML instance
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting AutoML training pipeline...")
            
            # Start experiment tracking
            if self.experiment_tracking and self.experiment_tracker:
                dataset_name = self._get_dataset_name(data)
                dataset_hash = self._calculate_dataset_hash(data) if hasattr(self, '_calculate_dataset_hash') else "unknown"
                
                config = ExperimentConfig(
                    experiment_name=f"AutoML_{self.task_type}_{dataset_name}",
                    description=f"AutoML training on {dataset_name} with {self.task_type} task",
                    tags=["automl", self.task_type],
                    dataset_name=dataset_name,
                    dataset_hash=dataset_hash,
                    target_column=self.target,
                    task_type=self.task_type
                )
                
                parameters = {
                    'time_limit': self.time_limit,
                    'validation_split': self.validation_split,
                    'random_state': self.random_state,
                    'feature_engineering': feature_engineering
                }
                
                self.current_experiment_id = self.experiment_tracker.start_experiment(config, parameters)
                if self.verbose:
                    self.logger.info(f"Started experiment: {self.current_experiment_id}")
            
            # Step 1: Data Ingestion
            self.logger.info("Step 1/5: Data ingestion and profiling...")
            train_df, profile = self.data_ingestion.load_and_profile(data)
            
            if test_data is not None:
                test_df, _ = self.data_ingestion.load_and_profile(test_data)
            else:
                test_df = None
                
            self.logger.info(f"Loaded data with shape: {train_df.shape}")
            self.logger.info(f"Target variable: {self.target}")
            
            # Step 2: Data Preprocessing
            self.logger.info("Step 2/5: Data preprocessing...")
            X, y = self._prepare_features_target(train_df)
            X_processed, y_processed = self.preprocessor.fit_transform(
                X, y, feature_engineering=feature_engineering
            )
            self.feature_names = X_processed.columns.tolist()
            
            # Store processed training data for interpretability
            self.X_train = X_processed
            self.y_train = y_processed
            
            # Process test data if provided
            if test_df is not None:
                X_test, y_test = self._prepare_features_target(test_df)
                X_test_processed, y_test_processed = self.preprocessor.transform(X_test, y_test)
            else:
                X_test_processed, y_test_processed = None, None
                
            self.logger.info(f"Preprocessed features shape: {X_processed.shape}")
            
            # Step 3: Model Selection and Training
            self.logger.info("Step 3/5: Model selection and training...")
            models = self.model_registry.get_models()
            
            trained_models = {}
            model_scores = {}
            
            remaining_time = self.time_limit - (time.time() - start_time)
            time_per_model = remaining_time / len(models)
            
            for model_name, model in models.items():
                model_start_time = time.time()
                
                self.logger.info(f"Training {model_name}...")
                
                try:
                    # Train base model
                    trained_model = self.model_registry.train_model(
                        model_name, X_processed, y_processed,
                        validation_split=self.validation_split,
                        random_state=self.random_state
                    )
                    
                    # Evaluate base model
                    cv_score = self.evaluator.cross_validate(
                        trained_model, X_processed, y_processed
                    )
                    
                    trained_models[model_name] = trained_model
                    model_scores[model_name] = cv_score['mean_score']
                    
                    self.logger.info(f"{model_name} CV score: {cv_score['mean_score']:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to train {model_name}: {str(e)}")
                    continue
                    
                # Check time limit
                if time.time() - start_time > self.time_limit * 0.6:  # Reserve 40% for tuning
                    self.logger.info("Time limit reached, stopping model training")
                    break
                    
            if not trained_models:
                raise RuntimeError("No models were successfully trained")
                
            # Step 4: Hyperparameter Tuning
            self.logger.info("Step 4/5: Hyperparameter tuning...")
            
            # Select top models for tuning (limit to 2 for faster training)
            top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            
            best_tuned_models = {}
            remaining_time = self.time_limit - (time.time() - start_time)
            
            # Warn if time is running low
            if remaining_time < 60:
                self.logger.warning(f"Low time remaining for tuning: {remaining_time:.0f} seconds")
            
            for model_name, _ in top_models:
                if remaining_time < 10:  # Skip tuning if very low time
                    # Fall back to base model
                    best_tuned_models[model_name] = {
                        'model': trained_models[model_name],
                        'score': model_scores[model_name],
                        'params': {}
                    }
                    continue
                    
                self.logger.info(f"Tuning {model_name}...")
                
                try:
                    # Only tune if we have sufficient time
                    if remaining_time >= 30:
                        tuned_model, best_params, tuning_score = self.tuner.tune_model(
                            model_name, 
                            trained_models[model_name],
                            X_processed, 
                            y_processed,
                            time_limit=min(remaining_time / len(top_models), remaining_time * 0.8)
                        )
                        
                        best_tuned_models[model_name] = {
                            'model': tuned_model,
                            'score': tuning_score,
                            'params': best_params
                        }
                        
                        self.logger.info(f"{model_name} tuned score: {tuning_score:.4f}")
                    else:
                        # Skip tuning, use base model
                        self.logger.info(f"Insufficient time for tuning {model_name}, using base model")
                        best_tuned_models[model_name] = {
                            'model': trained_models[model_name],
                            'score': model_scores[model_name],
                            'params': {}
                        }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to tune {model_name}: {str(e)}")
                    # Fall back to base model
                    best_tuned_models[model_name] = {
                        'model': trained_models[model_name],
                        'score': model_scores[model_name],
                        'params': {}
                    }
                    
                remaining_time = self.time_limit - (time.time() - start_time)
                
            # Step 5: Model Evaluation and Selection
            self.logger.info("Step 5/5: Final model evaluation and selection...")
            
            # If no models were tuned (e.g., due to time constraints), use trained models
            if not best_tuned_models:
                self.logger.warning("No models were tuned, using best trained model")
                best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k])
                self.best_model = trained_models[best_model_name]
                self.best_score = model_scores[best_model_name]
                self.best_model_name = best_model_name
            else:
                # Select best model from tuned models
                best_model_name = max(best_tuned_models.keys(), 
                                    key=lambda k: best_tuned_models[k]['score'])
                self.best_model = best_tuned_models[best_model_name]['model']
                self.best_score = best_tuned_models[best_model_name]['score']
                self.best_model_name = best_model_name
            
            
            # Final evaluation
            final_metrics = self.evaluator.evaluate_model(
                self.best_model, X_processed, y_processed,
                X_test_processed, y_test_processed
            )
            
            self.training_time = time.time() - start_time
            self.is_fitted = True
            
            # Initialize model interpreter
            try:
                self.interpreter = ModelInterpreter(
                    model=self.best_model,
                    X_train=self.X_train,
                    y_train=self.y_train,
                    feature_names=self.feature_names
                )
                self.logger.info("Model interpreter initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize model interpreter: {e}")
                self.interpreter = None
            
            # Store training history
            self.training_history.append({
                'timestamp': time.time(),
                'model_name': best_model_name,
                'score': self.best_score,
                'training_time': self.training_time,
                'metrics': final_metrics,
                'features_used': len(self.feature_names)
            })
            
            # End experiment tracking
            if self.experiment_tracking and self.experiment_tracker and self.current_experiment_id:
                # Log metrics
                primary_metric_name = 'accuracy' if self.task_type == 'classification' else 'r2_score'
                metrics_to_log = {
                    primary_metric_name: self.best_score,
                    'training_time': self.training_time,
                    'models_evaluated': len(trained_models),
                    'features_count': len(self.feature_names)
                }
                
                # Add detailed metrics
                if final_metrics:
                    for key, value in final_metrics.items():
                        if isinstance(value, (int, float)):
                            metrics_to_log[key] = value
                
                self.experiment_tracker.log_metrics(metrics_to_log, primary_metric_name)
                
                # Save best model
                if self.best_model:
                    try:
                        model_path = self.experiment_tracker.save_model(
                            self.best_model, 
                            f"best_model_{best_model_name}.pkl"
                        )
                        if self.verbose:
                            self.logger.info(f"Saved model to: {model_path}")
                    except Exception as e:
                        if self.verbose:
                            self.logger.warning(f"Could not save model: {e}")
                
                # End experiment
                status = "completed" if self.best_model else "failed"
                notes = f"Training completed. Best model: {best_model_name} with {primary_metric_name}: {self.best_score:.4f}"
                self.experiment_tracker.end_experiment(status=status, notes=notes)
                
                if self.verbose:
                    self.logger.info(f"Experiment {self.current_experiment_id} completed and saved.")
            
            self.logger.info("=" * 50)
            self.logger.info("AutoML Training Complete!")
            self.logger.info(f"Best model: {best_model_name}")
            self.logger.info(f"Best score: {self.best_score:.4f}")
            self.logger.info(f"Training time: {self.training_time:.2f} seconds")
            self.logger.info("=" * 50)
            
            return self
            
        except Exception as e:
            self.logger.error(f"AutoML training failed: {str(e)}")
            raise
            
    def predict(self, data: Union[str, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Input data as file path or DataFrame
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("AutoML must be fitted before making predictions")
            
        # Load data
        if isinstance(data, str):
            df, _ = self.data_ingestion.load_and_profile(data)
        else:
            df = data.copy()
            
        # Prepare features
        X = df.drop(columns=[self.target] if self.target in df.columns else [])
        
        # Transform features
        X_processed, _ = self.preprocessor.transform(X)
        
        # Make predictions
        predictions = self.best_model.predict(X_processed)
        
        return predictions
        
    def predict_proba(self, data: Union[str, pd.DataFrame]) -> np.ndarray:
        """
        Get prediction probabilities for classification tasks.
        
        Args:
            data: Input data as file path or DataFrame
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
            
        if not self.is_fitted:
            raise RuntimeError("AutoML must be fitted before making predictions")
            
        # Load data
        if isinstance(data, str):
            df, _ = self.data_ingestion.load_and_profile(data)
        else:
            df = data.copy()
            
        # Prepare features
        X = df.drop(columns=[self.target] if self.target in df.columns else [])
        
        # Transform features
        X_processed, _ = self.preprocessor.transform(X)
        
        # Get probabilities
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X_processed)
        else:
            raise AttributeError(f"Model {self.best_model_name} does not support predict_proba")
            
        return probabilities
        
    def get_best_model(self):
        """Get the best trained model."""
        if not self.is_fitted:
            raise RuntimeError("AutoML must be fitted before accessing the best model")
        return self.best_model
        
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics of the best model."""
        if not self.is_fitted:
            raise RuntimeError("AutoML must be fitted before accessing metrics")
            
        return self.evaluator.get_last_metrics()
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the best model."""
        if not self.is_fitted:
            raise RuntimeError("AutoML must be fitted before accessing feature importance")
            
        return self.evaluator.get_feature_importance(
            self.best_model, self.feature_names
        )
        
    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive results from the AutoML training."""
        if not self.is_fitted:
            raise RuntimeError("AutoML must be fitted before accessing results")
            
        return {
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'training_time': self.training_time,
            'feature_count': len(self.feature_names),
            'metrics': self.get_metrics(),
            'feature_importance': self.get_feature_importance(),
            'training_history': self.training_history
        }
        
    def save_model(self, path: str) -> None:
        """Save the trained AutoML system to disk."""
        if not self.is_fitted:
            raise RuntimeError("AutoML must be fitted before saving")
            
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save the entire AutoML object
        joblib.dump(self, save_path / 'automl_system.pkl')
        
        # Save individual components
        joblib.dump(self.best_model, save_path / 'best_model.pkl')
        joblib.dump(self.preprocessor, save_path / 'preprocessor.pkl')
        
        # Save metadata
        metadata = {
            'target': self.target,
            'task_type': self.task_type,
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'training_time': self.training_time,
            'feature_names': self.feature_names
        }
        
        with open(save_path / 'metadata.json', 'w') as f:
            import json
            json.dump(metadata, f, indent=2, default=str)
            
        self.logger.info(f"AutoML system saved to {save_path}")
        
    @classmethod
    def load_model(cls, path: str) -> 'AutoML':
        """Load a saved AutoML system from disk."""
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Path {load_path} does not exist")
            
        automl_path = load_path / 'automl_system.pkl'
        
        if not automl_path.exists():
            raise FileNotFoundError(f"AutoML system file not found at {automl_path}")
            
        automl = joblib.load(automl_path)
        return automl
        
    def get_interpreter(self) -> 'ModelInterpreter':
        """Get the model interpreter for explainability analysis."""
        if not self.is_fitted:
            raise RuntimeError("AutoML must be fitted before accessing interpreter")
        return self.interpreter
    
    def explain_prediction(self, instance: Union[pd.Series, pd.DataFrame], 
                         method: str = 'both') -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP and/or LIME.
        
        Args:
            instance: Single instance to explain
            method: Explanation method ('shap', 'lime', or 'both')
            
        Returns:
            Dictionary with explanation results
        """
        if not self.is_fitted or self.interpreter is None:
            raise RuntimeError("Model interpreter not available")
            
        if isinstance(instance, pd.DataFrame):
            if len(instance) != 1:
                raise ValueError("Only single instances can be explained")
            instance = instance.iloc[0]
        
        results = {}
        
        if method in ['shap', 'both']:
            try:
                shap_fig = self.interpreter.plot_shap_waterfall(instance)
                results['shap_plot'] = shap_fig
            except Exception as e:
                results['shap_error'] = str(e)
        
        if method in ['lime', 'both']:
            try:
                lime_data = self.interpreter.explain_instance_lime(instance)
                if lime_data:
                    lime_fig = self.interpreter.plot_lime_explanation(lime_data)
                    results['lime_plot'] = lime_fig
                    results['lime_data'] = lime_data
            except Exception as e:
                results['lime_error'] = str(e)
        
        return results
    
    def analyze_feature_importance(self, method: str = 'shap', 
                                 max_features: int = 20) -> go.Figure:
        """
        Analyze global feature importance using SHAP.
        
        Args:
            method: Analysis method ('shap' for now)
            max_features: Maximum number of features to display
            
        Returns:
            Plotly figure with feature importance
        """
        if not self.is_fitted or self.interpreter is None:
            raise RuntimeError("Model interpreter not available")
            
        if method == 'shap':
            return self.interpreter.plot_shap_summary(self.X_train, max_features)
        else:
            raise ValueError(f"Method '{method}' not supported")
    
    def plot_partial_dependence(self, feature: str) -> go.Figure:
        """
        Create partial dependence plot for a feature.
        
        Args:
            feature: Feature name
            
        Returns:
            Plotly figure with partial dependence plot
        """
        if not self.is_fitted or self.interpreter is None:
            raise RuntimeError("Model interpreter not available")
            
        return self.interpreter.plot_partial_dependence(feature, self.X_train)
    
    def analyze_feature_interactions(self, feature1: str, feature2: str) -> go.Figure:
        """
        Analyze interaction between two features.
        
        Args:
            feature1: First feature name
            feature2: Second feature name
            
        Returns:
            Plotly figure with interaction analysis
        """
        if not self.is_fitted or self.interpreter is None:
            raise RuntimeError("Model interpreter not available")
            
        return self.interpreter.analyze_feature_interactions([feature1, feature2], self.X_train)
    
    def what_if_analysis(self, base_instance: pd.Series, feature: str, 
                        values: List[float]) -> go.Figure:
        """
        Perform what-if analysis on a feature.
        
        Args:
            base_instance: Base instance to modify
            feature: Feature to analyze
            values: List of values to test
            
        Returns:
            Plotly figure with what-if analysis
        """
        if not self.is_fitted or self.interpreter is None:
            raise RuntimeError("Model interpreter not available")
            
        what_if_data = self.interpreter.what_if_analysis(base_instance, feature, values)
        if what_if_data:
            return self.interpreter.plot_what_if_analysis(what_if_data)
        return None
        
    def _prepare_features_target(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target from DataFrame."""
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in data")
            
        X = df.drop(columns=[self.target])
        y = df[self.target]
        
        return X, y
    
    def _get_dataset_name(self, data: Union[str, pd.DataFrame]) -> str:
        """Extract dataset name from input data."""
        if isinstance(data, str):
            return Path(data).stem
        else:
            return "uploaded_dataset"
    
    def get_experiment_id(self) -> Optional[str]:
        """Get the current experiment ID."""
        return self.current_experiment_id
    
    def get_experiment_tracker(self) -> Optional[ExperimentTracker]:
        """Get the experiment tracker instance."""
        return self.experiment_tracker
