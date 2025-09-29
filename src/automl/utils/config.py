"""
Configuration management for the AutoML system.
"""

from typing import Dict, Any, Optional
import logging


class Config:
    """
    Centralized configuration management for AutoML system.
    
    Provides default configurations and allows customization for:
    - Data preprocessing parameters
    - Model configurations
    - Hyperparameter tuning settings
    - Evaluation metrics
    """
    
    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with defaults and custom overrides.
        
        Args:
            custom_config: Dictionary of custom configuration overrides
        """
        self._config = self._get_default_config()
        
        if custom_config:
            self._merge_configs(self._config, custom_config)
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for all components."""
        return {
            'preprocessing': {
                'missing_value_strategy': 'auto',  # auto, drop, mean, median, mode, knn
                'categorical_encoding': 'auto',    # auto, onehot, label, target, binary
                'feature_scaling': 'auto',         # auto, standard, minmax, robust, none
                'outlier_handling': 'auto',        # auto, clip, remove, none
                'feature_selection': True,
                'feature_selection_method': 'auto',  # auto, univariate, recursive, lasso
                'max_categorical_cardinality': 50,
                'outlier_threshold': 1.5,  # IQR multiplier
                'correlation_threshold': 0.95,  # For removing highly correlated features
                'variance_threshold': 0.0,  # For removing low-variance features
            },
            
            'models': {
                'include_models': 'auto',  # auto, all, or list of model names
                'exclude_models': [],      # Models to exclude
                'max_training_time_per_model': 120,  # seconds - reduced from 600
                'cv_folds': 3,  # reduced from 5 for faster training
                'random_state': 42,
                'n_jobs': 2,  # limited to 2 cores instead of -1 (all cores)
                
                # Model-specific configurations - optimized for speed
                'random_forest': {
                    'n_estimators': 50,  # reduced from 100
                    'max_depth': 10,     # limited depth for faster training
                    'min_samples_split': 5,  # increased for faster training
                    'min_samples_leaf': 2,   # increased for faster training
                },
                
                'xgboost': {
                    'n_estimators': 30,  # reduced from 100
                    'max_depth': 4,      # reduced from 6
                    'learning_rate': 0.2, # increased for faster convergence
                    'subsample': 0.8,    # reduced for faster training
                },
                
                'lightgbm': {
                    'n_estimators': 30,  # reduced from 100
                    'max_depth': 4,      # limited depth
                    'learning_rate': 0.2, # increased for faster convergence
                    'num_leaves': 15,    # reduced from 31
                },
                
                'catboost': {
                    'iterations': 30,    # reduced from 100
                    'depth': 4,          # reduced from 6
                    'learning_rate': 0.2, # increased for faster convergence
                    'verbose': False,
                }
            },
            
            'tuning': {
                'method': 'optuna',  # optuna, grid, random
                'n_trials': 10,   # drastically reduced from 100
                'timeout': 180,   # reduced from 3600 (1 hour) to 3 minutes
                'cv_folds': 2,    # reduced from 3
                'scoring': 'auto',  # auto or specific metric name
                'direction': 'maximize',
                'sampler': 'tpe',  # tpe, random, cma-es
                'pruner': 'median',  # median, successive_halving, hyperband
                
                # Search spaces for different algorithms
                'search_spaces': {
                    'random_forest': {
                        'n_estimators': (50, 500),
                        'max_depth': (3, 20),
                        'min_samples_split': (2, 20),
                        'min_samples_leaf': (1, 10),
                        'max_features': ['auto', 'sqrt', 'log2'],
                    },
                    
                    'xgboost': {
                        'n_estimators': (50, 500),
                        'max_depth': (3, 12),
                        'learning_rate': (0.01, 0.3),
                        'subsample': (0.6, 1.0),
                        'colsample_bytree': (0.6, 1.0),
                        'reg_alpha': (0, 10),
                        'reg_lambda': (0, 10),
                    },
                    
                    'lightgbm': {
                        'n_estimators': (50, 500),
                        'max_depth': (3, 15),
                        'learning_rate': (0.01, 0.3),
                        'num_leaves': (10, 100),
                        'min_child_samples': (5, 100),
                        'subsample': (0.6, 1.0),
                        'colsample_bytree': (0.6, 1.0),
                    },
                    
                    'catboost': {
                        'iterations': (50, 500),
                        'depth': (3, 10),
                        'learning_rate': (0.01, 0.3),
                        'l2_leaf_reg': (1, 10),
                        'border_count': (32, 255),
                    }
                }
            },
            
            'evaluation': {
                'classification_metrics': [
                    'accuracy', 'precision', 'recall', 'f1', 
                    'roc_auc', 'log_loss', 'matthews_corrcoef'
                ],
                'regression_metrics': [
                    'r2', 'mean_squared_error', 'mean_absolute_error',
                    'root_mean_squared_error', 'mean_absolute_percentage_error'
                ],
                'primary_metric': 'auto',  # auto or specific metric name
                'cv_folds': 3,  # reduced from 5
                'test_size': 0.2,
                'shuffle': True,
                'stratify': True,  # For classification
                'generate_plots': True,
                'plot_types': [
                    'feature_importance', 'confusion_matrix', 
                    'roc_curve', 'precision_recall_curve',
                    'residuals', 'prediction_error'
                ],
            },
            
            'feature_engineering': {
                'enabled': False,
                'methods': [
                    'polynomial', 'interactions', 'clustering', 
                    'dimensionality_reduction', 'text_features'
                ],
                'polynomial_degree': 2,
                'interaction_only': False,
                'n_clusters': 'auto',
                'pca_components': 'auto',
                'text_max_features': 1000,
            },
            
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_logging': False,
                'log_file': 'automl.log',
            },
            
            'performance': {
                'memory_limit_gb': None,  # None for no limit
                'n_jobs': 2,     # limited to 2 cores for better system responsiveness
                'chunk_size': 5000,  # reduced for faster processing
                'early_stopping': True,
                'patience': 5,   # reduced patience for faster stopping
            }
        }
        
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
                
    @property
    def preprocessing(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._config['preprocessing']
        
    @property
    def models(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config['models']
        
    @property
    def tuning(self) -> Dict[str, Any]:
        """Get hyperparameter tuning configuration."""
        return self._config['tuning']
        
    @property
    def evaluation(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self._config['evaluation']
        
    @property
    def feature_engineering(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self._config['feature_engineering']
        
    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config['logging']
        
    @property
    def performance(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self._config['performance']
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'models.random_forest.n_estimators')."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default
            
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key path."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """Get the complete configuration as dictionary."""
        return self._config.copy()
        
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._merge_configs(self._config, updates)
        
    def validate(self) -> bool:
        """Validate configuration values."""
        try:
            # Validate preprocessing config
            preprocessing = self.preprocessing
            valid_missing_strategies = ['auto', 'drop', 'mean', 'median', 'mode', 'knn']
            if preprocessing['missing_value_strategy'] not in valid_missing_strategies:
                raise ValueError(f"Invalid missing_value_strategy: {preprocessing['missing_value_strategy']}")
                
            valid_encodings = ['auto', 'onehot', 'label', 'target', 'binary']
            if preprocessing['categorical_encoding'] not in valid_encodings:
                raise ValueError(f"Invalid categorical_encoding: {preprocessing['categorical_encoding']}")
                
            valid_scaling = ['auto', 'standard', 'minmax', 'robust', 'none']
            if preprocessing['feature_scaling'] not in valid_scaling:
                raise ValueError(f"Invalid feature_scaling: {preprocessing['feature_scaling']}")
                
            # Validate model config
            models = self.models
            if models['cv_folds'] < 2:
                raise ValueError("cv_folds must be at least 2")
                
            # Validate tuning config
            tuning = self.tuning
            valid_methods = ['optuna', 'grid', 'random']
            if tuning['method'] not in valid_methods:
                raise ValueError(f"Invalid tuning method: {tuning['method']}")
                
            if tuning['n_trials'] < 1:
                raise ValueError("n_trials must be at least 1")
                
            # Validate evaluation config
            evaluation = self.evaluation
            if evaluation['cv_folds'] < 2:
                raise ValueError("evaluation cv_folds must be at least 2")
                
            if not 0 < evaluation['test_size'] < 1:
                raise ValueError("test_size must be between 0 and 1")
                
            return True
            
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            return False