"""
AutoML System - An end-to-end automated machine learning pipeline.

This package provides a comprehensive AutoML solution that automates:
- Data preprocessing and feature engineering
- Model selection and hyperparameter tuning
- Model evaluation and interpretation
- Deployment and monitoring

Example:
    >>> from automl import AutoML
    >>> automl = AutoML(target='price', task_type='regression')
    >>> automl.fit('data.csv')
    >>> predictions = automl.predict(test_data)
"""

from .core.automl import AutoML
from .core.data_ingestion import DataIngestion
from .core.preprocessing import Preprocessor
from .core.models import ModelRegistry
from .core.tuning import HyperparameterTuner
from .core.evaluation import ModelEvaluator

__version__ = "0.1.0"
__author__ = "AutoML Developer"
__email__ = "dev@automl.local"

__all__ = [
    "AutoML",
    "DataIngestion", 
    "Preprocessor",
    "ModelRegistry",
    "HyperparameterTuner",
    "ModelEvaluator",
]