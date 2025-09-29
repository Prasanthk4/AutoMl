# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common Development Commands

### Installation and Setup
```bash
# Install package in development mode
pip install -e .

# Install with development dependencies
pip install -r requirements.txt

# Run tests with sample datasets
python test_automl.py

# Create sample datasets for testing
python create_sample_data.py
```

### Running the Application
```bash
# Launch Streamlit dashboard (recommended)
python run_automl_dashboard.py

# Alternative: Direct streamlit launch
streamlit run streamlit_app.py

# Run core AutoML tests
python -m pytest tests/ -v    # (if pytest tests exist)

# Quick performance test (completes in ~15-30 seconds)
python test_fast.py

# Test interpretability features
python test_interpretability.py

# Test advanced data analysis dashboard
python test_advanced_dashboard.py

# Test model comparison and A/B testing
python test_model_comparison.py
```

### API Development
```bash
# Start FastAPI server (mentioned in README but not implemented yet)
uvicorn automl.api.main:app --reload

# Run specific test file
python test_automl.py
```

### Code Quality
```bash
# Format code (if using black)
black src/ tests/

# Lint code (if using flake8)
flake8 src/ tests/

# Type checking (if using mypy)
mypy src/automl/
```

## High-Level Architecture

### Core Design Pattern
The AutoML system follows a **modular pipeline architecture** where each component is responsible for a specific stage of the ML workflow:

1. **Data Ingestion** (`data_ingestion.py`) → Load and profile datasets
2. **Preprocessing** (`preprocessing.py`) → Clean, encode, and prepare features  
3. **Model Registry** (`models.py`) → Manage available algorithms
4. **Hyperparameter Tuning** (`tuning.py`) → Optimize model parameters
5. **Evaluation** (`evaluation.py`) → Assess and compare models

### Key Architectural Principles

**AutoML Orchestration**: The main `AutoML` class in `src/automl/core/automl.py` acts as the central coordinator, managing the entire pipeline with time-bounded training and automatic fallbacks.

**Configuration-Driven Design**: All components use the centralized `Config` class (`utils/config.py`) which provides hierarchical defaults for:
- Preprocessing strategies (missing values, encoding, scaling)
- Model parameters for each algorithm
- Hyperparameter search spaces
- Evaluation metrics

**Graceful Degradation**: The system is designed to handle missing optional dependencies (XGBoost, LightGBM, CatBoost) and continues working with available models.

**Time-Aware Training**: Training is split with 60% for base model training and 40% reserved for hyperparameter tuning, with automatic time management.

### Component Interactions

**Data Flow**: Raw data → DataIngestion → Preprocessor → ModelRegistry → HyperparameterTuner → ModelEvaluator → Best Model

**State Management**: Each component maintains its fitted state and can transform new data using the same parameters learned during training.

**Error Handling**: Failed model training doesn't stop the pipeline - it continues with other available models and logs warnings.

### Module Dependencies

- **Core modules** depend only on scikit-learn and basic ML libraries
- **Optional modules** (XGBoost, LightGBM, CatBoost) are imported conditionally
- **UI components** are completely separate from core logic
- **Configuration** is injected at initialization, not imported directly

### Key Design Decisions

**Lazy Algorithm Loading**: Models are only instantiated when training begins, allowing the system to work even with missing optional dependencies.

**Feature Name Preservation**: The preprocessor maintains original feature names where possible and creates interpretable names for encoded features.

**Validation Strategy**: Uses cross-validation for model selection but reserves a separate test set for final evaluation when provided.

## Working with This Codebase

### Adding New Models
1. Add conditional import in `models.py`
2. Update `_initialize_models()` method
3. Add default configuration in `config.py`
4. Add hyperparameter search space in tuning configuration

### Testing Changes
- Use `test_automl.py` for end-to-end validation with both classification and regression
- Use `test_interpretability.py` for testing SHAP, LIME, and feature analysis
- Use `test_advanced_dashboard.py` for testing data analysis features
- Use `test_model_comparison.py` for testing model comparison and A/B testing
- Sample datasets are auto-generated in `data/` directory
- Dashboard can be tested with built-in sample data download

### Configuration Customization
The system supports deep configuration overrides through the Config class:
```python
custom_config = {
    'preprocessing': {'missing_value_strategy': 'knn'},
    'models': {'include_models': ['random_forest', 'xgboost']},
    'tuning': {'n_trials': 200}
}
automl = AutoML(config=custom_config)
```

### UI and Core Separation
The Streamlit dashboard (`streamlit_app.py`) and the core AutoML system are completely decoupled. The dashboard imports from `src/automl/` and provides a wrapper interface but doesn't modify core functionality.

### Entry Points
- **Programmatic**: Import from `src/automl` package
- **Interactive**: Run `streamlit_app.py` or `run_automl_dashboard.py`
- **Testing**: Use `test_automl.py` for validation
- **Fast Testing**: Use `test_fast.py` for quick 15-second validation
- **Interpretability Testing**: Use `test_interpretability.py` for explainability features
- **Data Analysis Testing**: Use `test_advanced_dashboard.py` for EDA features
- **Model Comparison Testing**: Use `test_model_comparison.py` for model comparison
- **CLI**: Mentioned in setup.py but not yet implemented

## Experiment Tracking & History (Feature #4)

The system now includes comprehensive experiment tracking and history management:

### Automatic Experiment Logging
- **Seamless Integration**: Experiments are automatically logged during AutoML training
- **Comprehensive Tracking**: Records parameters, metrics, artifacts, and model performance
- **Persistent Storage**: SQLite database for experiment history with full metadata
- **Model Artifacts**: Automatic saving of best models and associated files

### Key Features
- **Parameter Tracking**: All hyperparameters and configuration settings
- **Metrics Logging**: Performance metrics, training times, and cross-validation scores
- **Artifact Management**: Model files, plots, and data exports
- **Experiment Comparison**: Side-by-side analysis of different runs
- **Performance Trends**: Historical analysis and optimization tracking
- **Search & Filtering**: Advanced querying by task type, dataset, dates, etc.

### Usage Examples
```python
# AutoML with experiment tracking (enabled by default)
automl = AutoML(
    target='target',
    task_type='classification',
    experiment_tracking=True  # Default: True
)

# Training automatically logs experiment
automl.fit(data)

# Get experiment ID and tracker
exp_id = automl.get_experiment_id()
tracker = automl.get_experiment_tracker()

# Query experiment history
all_experiments = tracker.db.list_experiments()
classification_exps = tracker.db.list_experiments(
    filters={'task_type': 'classification'}
)

# Load specific experiment
experiment = tracker.db.load_experiment(exp_id)
print(f"Best score: {experiment.metrics.primary_metric}")
```

### Database Location
- **Database**: `experiments/experiments.db` (SQLite)
- **Model Files**: `experiments/{experiment_id}/` directories
- **Automatic Cleanup**: No automatic deletion - all experiments preserved for analysis

### Analytics & Visualization
```python
from automl.experiment_analytics import ExperimentHistory, ExperimentAnalytics

# Create analytics tools
history = ExperimentHistory(tracker)
analytics = ExperimentAnalytics(history)

# Performance trends over time
trends_fig = analytics.create_performance_timeline()

# Compare specific experiments
comparison = history.compare_experiments([exp_id1, exp_id2])

# Export experiment data
exp_data = history.export_experiments(export_format="csv")
```

### Integration with Model Comparison
- **Unified Tracking**: Both AutoML and model comparison support experiment logging
- **Consistent Storage**: Same database schema for all experiment types
- **Cross-Analysis**: Compare AutoML runs with manual model comparisons

## Performance Optimizations

The system has been optimized for development speed and system responsiveness:

### Default Performance Settings
- **CPU Usage**: Limited to 2 cores (instead of all cores) to prevent system overload
- **Training Time**: Default 5 minutes (reduced from 1 hour)
- **Hyperparameter Trials**: 10 trials (reduced from 100)
- **Cross-validation**: 3 folds (reduced from 5)
- **Model Parameters**: Optimized for faster training

### Training Modes (Streamlit Dashboard)
- **⚡ Fast Mode (1-2 min)**: Single model, minimal trials, optimized for speed
- **🎯 Balanced Mode (3-5 min)**: Multiple models, reasonable accuracy
- **🔬 Thorough Mode (5-10 min)**: All models, maximum trials for best performance

### Quick Performance Test
```bash
# Test system performance (should complete in ~15 seconds)
python test_fast.py
```

### Troubleshooting Performance Issues
If training is still slow:
1. Use Fast Mode in the dashboard
2. Reduce dataset size (< 1000 rows for development)
3. Check system resources: `htop` or `ps aux | grep python`
4. Kill hanging processes: `pkill -f "automl_venv/bin/python"`

## Model Interpretability Features

The system now includes comprehensive model interpretability and explainability tools:

### SHAP Integration
- **Global Feature Importance**: Understand which features matter most across all predictions
- **Individual Explanations**: See why the model made a specific prediction
- **Waterfall Charts**: Visualize how each feature contributes to a single prediction

### LIME Explanations
- **Local Interpretable Explanations**: Understand model behavior for individual instances
- **Feature Impact Analysis**: See positive/negative contributions of features
- **Model-Agnostic**: Works with any model type

### Advanced Analysis Tools
- **Partial Dependence Plots**: How individual features affect predictions
- **Feature Interaction Analysis**: Understand how features work together
- **What-If Analysis**: Explore prediction changes when modifying feature values

### Access via Streamlit Dashboard
After training a model, the interpretability features are available in a dedicated tab with:
- Interactive SHAP plots
- LIME explanations for sample predictions
- Feature analysis tools
- What-if scenario exploration

### Programmatic Access
```python
# After training an AutoML model
explanation = automl.explain_prediction(sample_instance)
shap_importance = automl.analyze_feature_importance('shap')
partial_dep = automl.plot_partial_dependence('feature_name')
whatif_analysis = automl.what_if_analysis(instance, 'feature', [val1, val2, val3])
```

## Advanced Data Analysis Dashboard (Feature #2)

The system now includes a comprehensive data analysis dashboard with automated exploratory data analysis (EDA), data quality assessment, and smart recommendations.

### Key Features

**Automated Data Quality Assessment**: Get an overall quality score (0-100) based on completeness, uniqueness, consistency, validity, and cardinality.

**Smart Recommendations**: Receive actionable insights about data issues:
- Missing data patterns
- Duplicate row detection
- High cardinality features
- Outlier analysis
- Dataset size recommendations
- Task type suggestions

**Interactive Analysis Tabs**:
- **Overview**: Dataset metrics, data preview, summary statistics
- **Data Quality**: Quality scoring with detailed breakdown and recommendations
- **Correlations**: Feature correlation matrix and target variable analysis
- **Distributions**: Statistical analysis with normality tests and shape analysis
- **Outliers**: Univariate and multivariate outlier detection with visualizations

**Performance Optimized**: Analyzes 3000+ rows in under 1 second with intelligent caching.

### Dashboard Access
The advanced analysis dashboard is the default view when you upload a dataset. It automatically:
1. Profiles your data comprehensively
2. Identifies quality issues
3. Provides smart recommendations
4. Creates interactive visualizations
5. Suggests preprocessing steps

### Programmatic Access
```python
# Use AdvancedDataAnalyzer directly
from automl.data_analysis import AdvancedDataAnalyzer

# Initialize analyzer
analyzer = AdvancedDataAnalyzer(df, target_column='your_target')

# Get data quality assessment
quality_metrics = analyzer.compute_data_quality_score()
print(f"Data quality: {quality_metrics['overall_quality_score']:.1f}/100")

# Generate recommendations
recommendations = analyzer.generate_smart_recommendations()
for rec in recommendations:
    print(f"{rec['type'].upper()}: {rec['title']} - {rec['action']}")

# Analyze correlations
correlations = analyzer.analyze_correlations()

# Detect outliers
outliers = analyzer.detect_outliers()

# Create visualizations
corr_heatmap = analyzer.create_correlation_heatmap()
dist_plots = analyzer.create_distribution_plots()
target_plot = analyzer.create_target_analysis_plot()
outlier_plot = analyzer.create_outlier_plot()
```

### Advanced Analysis Features

**Data Quality Scoring Algorithm**:
- Completeness (30%): Percentage of non-missing values
- Uniqueness (20%): Absence of duplicate rows
- Consistency (20%): Data type and format consistency
- Validity (20%): Outlier detection and range validation
- Cardinality (10%): Appropriate uniqueness levels for categorical features

**Correlation Analysis**:
- Pearson correlation for numeric features
- Mutual information for mixed feature types
- High correlation pair detection (>0.8)
- Target variable correlation ranking

**Distribution Analysis**:
- Skewness and kurtosis calculation
- Normality testing (Jarque-Bera test)
- Distribution shape classification
- Statistical summaries with confidence intervals

**Outlier Detection Methods**:
- IQR-based univariate outlier detection
- Isolation Forest for multivariate outliers
- Percentage and count reporting
- Visual identification in box plots

## Model Comparison & A/B Testing (Feature #3)

The system now includes comprehensive model comparison capabilities with statistical significance testing and A/B testing framework.

### Key Features

**Multi-Model Comparison**: Compare multiple machine learning models side-by-side with automated training, cross-validation, and statistical significance testing.

**Statistical Testing**: Rigorous statistical analysis including:
- Paired t-tests for comparing model performance
- Wilcoxon signed-rank tests (non-parametric alternative)
- Effect size calculation (Cohen's d)
- Multiple testing corrections
- Bootstrap confidence intervals

**A/B Testing Framework**: Controlled experiments for model comparison:
- Proper A/B test design with control and treatment groups
- Statistical power analysis
- Sample size calculations
- Bootstrap resampling for confidence intervals
- Effect size and practical significance assessment

**Interactive Visualizations**: Comprehensive visual analysis:
- Performance comparison charts
- ROC and Precision-Recall curves
- Statistical significance visualizations
- Performance radar charts
- Bootstrap distribution plots
- A/B test result dashboards

**Performance Optimized**: Efficient comparison of multiple models with parallel processing and intelligent caching.

### Dashboard Access
The model comparison features are available in the **⚔️ Model Comparison** tab of the dashboard. Features include:
1. **🏁 Quick Model Comparison**: Compare 3-6 models with automated training
2. **🧪 A/B Testing Framework**: Controlled experiments (coming soon)
3. **📊 Advanced Statistical Analysis**: Comprehensive testing (coming soon)

### Programmatic Access
```python
# Model Comparison
from automl.model_comparison import ModelComparison, ABTestFramework
from automl.comparison_visualizations import ComparisonVisualizer

# Initialize comparison framework
comparison = ModelComparison(
    task_type='classification',  # or 'regression'
    cv_folds=5,
    random_state=42
)

# Add models to compare
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

comparison.add_model(RandomForestClassifier(), "Random Forest")
comparison.add_model(LogisticRegression(), "Logistic Regression")

# Run comparison
results = comparison.compare_models(X_train, y_train, X_test, y_test)

# Access results
print(f"Best model: {results.best_model}")
print(results.comparison_metrics)  # Performance DataFrame
print(results.statistical_tests)   # Statistical significance tests

# Generate detailed report
report = comparison.create_comparison_report(results)
print(report)
```

### A/B Testing Usage
```python
# A/B Testing Framework
ab_test = ABTestFramework(significance_level=0.05, power=0.8)

# Design A/B test
(X_control, y_control), (X_treatment, y_treatment) = ab_test.design_ab_test(
    X, y, test_size=0.3, stratify=True
)

# Run A/B test
results = ab_test.run_ab_test(
    model_a, model_b,
    X_control, y_control,
    X_treatment, y_treatment,
    metric='accuracy'
)

# Analyze results
print(f"P-value: {results['p_value']:.6f}")
print(f"Effect size: {results['cohens_d']:.4f}")
print(f"Significant: {results['significant']}")
print(f"Relative improvement: {results['relative_improvement']:.2f}%")
```

### Visualization Examples
```python
# Create visualizations
viz = ComparisonVisualizer(dark_theme=True)

# Performance comparison
perf_fig = viz.create_performance_comparison_chart(results)
perf_fig.show()

# ROC curves (for classification)
roc_fig = viz.create_roc_curves(results.model_results, X_test, y_test)
roc_fig.show()

# Statistical significance
sig_fig = viz.create_statistical_significance_chart(results)
sig_fig.show()

# A/B test results
ab_fig = viz.create_ab_test_visualization(ab_results)
ab_fig.show()
```

### Model Comparison Features

**Supported Models**: The framework works with any scikit-learn compatible model:
- Random Forest, Gradient Boosting, XGBoost, LightGBM
- Logistic/Linear Regression, SVM, Neural Networks
- Custom models with fit/predict interface

**Metrics Calculated**:
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression**: R², MSE, RMSE, MAE, MAPE
- **Performance**: Training time, prediction time, memory usage
- **Stability**: Cross-validation mean and standard deviation

**Statistical Tests**:
- **Paired T-test**: Compares mean performance across CV folds
- **Wilcoxon Test**: Non-parametric alternative for non-normal distributions
- **Effect Size**: Cohen's d for practical significance
- **Confidence Intervals**: Bootstrap-based intervals for robust estimation

**Visualization Types**:
- **Performance Charts**: Bar charts with error bars and statistical annotations
- **ROC/PR Curves**: Comparative curves with AUC calculations
- **Radar Charts**: Multi-dimensional performance comparison
- **Box Plots**: Distribution analysis and outlier detection
- **Heatmaps**: Correlation and significance matrices
