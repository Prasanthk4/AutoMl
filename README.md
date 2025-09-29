# 🤖 Advanced AutoML System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green.svg)]()

A comprehensive, production-ready AutoML system with advanced data analysis, model interpretability, experiment tracking, and interactive visualizations.

## 🌟 Features

### 🔍 **Feature #1: Advanced Data Analysis & Profiling**
- **Smart Data Ingestion**: Multi-format support (CSV, Excel, JSON, Parquet, Databases)
- **Comprehensive Data Profiling**: Quality scores, missing value analysis, outlier detection
- **Automated Insights**: Statistical analysis, correlation detection, distribution analysis
- **Smart Recommendations**: Data preprocessing suggestions based on analysis

### 🔬 **Feature #2: Model Interpretability & Explainability**
- **SHAP Integration**: Global and local feature importance analysis
- **LIME Explanations**: Local interpretable model-agnostic explanations
- **Partial Dependence Plots**: Feature impact visualization
- **What-if Analysis**: Interactive prediction exploration
- **Feature Interactions**: Multi-dimensional relationship analysis

### ⚖️ **Feature #3: Model Comparison & A/B Testing**
- **Advanced Model Comparison**: Statistical significance testing
- **A/B Testing Framework**: Proper statistical controls and power analysis
- **Performance Visualizations**: ROC curves, precision-recall, learning curves
- **Bootstrap Analysis**: Confidence intervals and effect size calculation
- **Cross-validation**: Robust model evaluation with multiple metrics

### 📊 **Feature #4: Experiment Management & History**
- **Complete Experiment Tracking**: Parameters, metrics, results, artifacts
- **Experiment Analytics**: Performance trends, comparative analysis
- **Advanced Search & Filtering**: Find experiments by any criteria
- **Experiment Comparison**: Side-by-side result analysis
- **Automated Logging**: Seamless integration with AutoML workflows

### 🎯 **Core AutoML Engine**
- **Multi-algorithm Support**: Random Forest, XGBoost, LightGBM, CatBoost, Linear Models
- **Intelligent Preprocessing**: Automatic encoding, scaling, feature engineering
- **Hyperparameter Optimization**: Bayesian optimization with time constraints
- **Performance Optimized**: Smart resource management and parallel processing

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Prasanthk4/AutoMl.git
cd AutoMl

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Launch Dashboard

```bash
# Start the interactive Streamlit dashboard
streamlit run streamlit_app.py

# Alternative launch method
python run_automl_dashboard.py
```

### Quick API Example

```python
from automl import AutoML
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Create and configure AutoML
automl = AutoML(
    task_type='classification',
    time_limit=300,  # 5 minutes
    random_state=42
)

# Train models
automl.fit(df, target_column='target')

# Get best model and predictions
best_model = automl.get_best_model()
predictions = automl.predict(df)

# Model interpretability
explanations = automl.explain_prediction(df.iloc[0])
feature_importance = automl.analyze_feature_importance()
```

## 📁 Project Structure

```
automl-system/
├── src/automl/                 # Core AutoML package
│   ├── core/                   # Core functionality
│   │   ├── automl.py          # Main AutoML class
│   │   ├── preprocessing.py    # Data preprocessing
│   │   ├── models.py          # Model registry
│   │   └── data_ingestion.py  # Data loading & profiling
│   ├── interpretability.py    # SHAP, LIME, explanations
│   ├── model_comparison.py     # A/B testing, comparisons
│   ├── experiment_tracking.py  # Experiment management
│   ├── experiment_analytics.py # Analytics & insights
│   ├── data_analysis.py       # Advanced data analysis
│   └── utils/                 # Utilities and helpers
├── streamlit_app.py           # Main Streamlit dashboard
├── run_automl_dashboard.py    # Dashboard launcher
├── test_*.py                  # Comprehensive test suite
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## 📋 Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- streamlit >= 1.28.0
- plotly >= 5.0.0
- shap >= 0.41.0
- lime >= 0.2.0
- optuna >= 3.0.0 (optional, for advanced hyperparameter tuning)
- xgboost >= 1.5.0 (optional)
- lightgbm >= 3.0.0 (optional)
- catboost >= 1.0.0 (optional)

## Examples

### Classification Example

```python
from automl import AutoML
import pandas as pd

# Load data
data = pd.read_csv('data/iris.csv')

# Initialize for classification
automl = AutoML(
    target='species',
    task_type='classification',
    validation_split=0.2,
    random_state=42
)

# Train models
automl.fit(data)

# Get results
results = automl.get_results()
print(f"Best model: {results['best_model_name']}")
print(f"Accuracy: {results['accuracy']:.3f}")

# Make predictions
new_predictions = automl.predict(test_data)
```

### Regression Example

```python
from automl import AutoML

# Initialize for regression
automl = AutoML(
    target='price',
    task_type='regression',
    time_limit=1800,  # 30 minutes
    n_trials=100
)

# Train with automatic feature engineering
automl.fit(
    'data/housing.csv',
    feature_engineering=True,
    generate_features=['polynomial', 'interactions']
)

# Get feature importance
feature_importance = automl.get_feature_importance()
print(feature_importance.head(10))
```

## Advanced Features

### Custom Configuration

```python
config = {
    'preprocessing': {
        'handle_missing': 'knn',
        'categorical_encoding': 'target',
        'scaling': 'robust'
    },
    'models': ['xgboost', 'lightgbm', 'catboost'],
    'tuning': {
        'method': 'bayesian',
        'n_trials': 200,
        'timeout': 3600
    }
}

automl = AutoML(config=config)
```

### Model Interpretability

```python
# Generate SHAP explanations
explanations = automl.explain_predictions(test_data)

# Create interpretability dashboard
automl.create_interpretation_dashboard()

# Get partial dependence plots
automl.plot_partial_dependence(['feature1', 'feature2'])
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Support for time series forecasting
- [ ] Neural network integration
- [ ] Automated feature engineering
- [ ] Multi-objective optimization
- [ ] Distributed training support
- [ ] AutoML for deep learning
- [ ] Model monitoring and drift detection

## Support

For questions and support:
- 📧 Email: dev@automl.local
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/automl-system/issues)
- 📖 Documentation: [Read the Docs](https://automl-system.readthedocs.io)