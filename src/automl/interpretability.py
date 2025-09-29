"""
Model Interpretability and Explainability Module
Provides SHAP, LIME, and other interpretability tools for AutoML models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance
from sklearn.base import is_classifier, is_regressor
import logging

warnings.filterwarnings('ignore')

class ModelInterpreter:
    """
    Comprehensive model interpretability class using SHAP, LIME, and other techniques.
    
    Features:
    - SHAP (SHapley Additive exPlanations) integration
    - LIME (Local Interpretable Model-agnostic Explanations)
    - Partial Dependence Plots
    - Feature Importance Analysis
    - What-if Analysis
    - Interactive Visualizations
    """
    
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                 feature_names: List[str] = None):
        """
        Initialize the model interpreter.
        
        Args:
            model: Trained ML model
            X_train: Training features
            y_train: Training target
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names or list(X_train.columns)
        self.is_classifier = is_classifier(model)
        self.is_regressor = is_regressor(model)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.shap_values = None
        
        self.logger = logging.getLogger(__name__)
        
    def _initialize_shap_explainer(self, method: str = "auto") -> None:
        """Initialize SHAP explainer based on model type."""
        try:
            if method == "auto":
                # Auto-detect best explainer
                if hasattr(self.model, 'predict_proba'):
                    # Tree-based models
                    if hasattr(self.model, 'feature_importances_'):
                        self.shap_explainer = shap.TreeExplainer(self.model)
                    else:
                        # Use sampling for complex models
                        sample_size = min(100, len(self.X_train))
                        background = self.X_train.sample(n=sample_size, random_state=42)
                        self.shap_explainer = shap.KernelExplainer(
                            self.model.predict_proba, background
                        )
                else:
                    # Regression models
                    if hasattr(self.model, 'feature_importances_'):
                        self.shap_explainer = shap.TreeExplainer(self.model)
                    else:
                        sample_size = min(100, len(self.X_train))
                        background = self.X_train.sample(n=sample_size, random_state=42)
                        self.shap_explainer = shap.KernelExplainer(
                            self.model.predict, background
                        )
            elif method == "tree":
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif method == "kernel":
                sample_size = min(100, len(self.X_train))
                background = self.X_train.sample(n=sample_size, random_state=42)
                if self.is_classifier:
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba, background
                    )
                else:
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict, background
                    )
            elif method == "linear":
                self.shap_explainer = shap.LinearExplainer(self.model, self.X_train)
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _initialize_lime_explainer(self) -> None:
        """Initialize LIME explainer."""
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train.values,
                feature_names=self.feature_names,
                class_names=['negative', 'positive'] if self.is_classifier else None,
                mode='classification' if self.is_classifier else 'regression',
                discretize_continuous=True,
                random_state=42
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize LIME explainer: {e}")
            self.lime_explainer = None
    
    def compute_shap_values(self, X: pd.DataFrame, method: str = "auto") -> np.ndarray:
        """
        Compute SHAP values for given data.
        
        Args:
            X: Input data
            method: SHAP method ('auto', 'tree', 'kernel', 'linear')
            
        Returns:
            SHAP values array
        """
        if self.shap_explainer is None:
            self._initialize_shap_explainer(method)
        
        if self.shap_explainer is None:
            return None
            
        try:
            # Limit sample size for performance
            sample_size = min(50, len(X))  # Reduced for stability
            X_sample = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X
            
            # Ensure numeric data for SHAP
            X_numeric = X_sample.select_dtypes(include=[np.number])
            
            if len(X_numeric.columns) == 0:
                self.logger.warning("No numeric columns found for SHAP analysis")
                return None
            
            shap_values = self.shap_explainer.shap_values(X_numeric)
            
            # Handle different return formats
            if isinstance(shap_values, list):
                # Multi-class classification - use positive class
                shap_values = shap_values[-1] if len(shap_values) > 1 else shap_values[0]
            
            self.shap_values = shap_values
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Error computing SHAP values: {e}")
            return None
    
    def plot_shap_summary(self, X: pd.DataFrame, max_display: int = 20) -> go.Figure:
        """
        Create interactive SHAP summary plot.
        
        Args:
            X: Input data
            max_display: Maximum number of features to display
            
        Returns:
            Plotly figure
        """
        shap_values = self.compute_shap_values(X)
        if shap_values is None:
            return None
            
        try:
            # Ensure shap_values is 2D array
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Calculate feature importance (mean absolute SHAP values)
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Get numeric feature names (since SHAP only uses numeric features)
            X_sample = X.sample(n=min(50, len(X)), random_state=42) if len(X) > 50 else X
            numeric_features = X_sample.select_dtypes(include=[np.number]).columns.tolist()
            
            # Ensure feature_importance is 1D
            if feature_importance.ndim > 1:
                feature_importance = feature_importance.flatten()
            
            if len(numeric_features) != len(feature_importance):
                # Fallback to generic names if mismatch
                numeric_features = [f"Feature_{i}" for i in range(len(feature_importance))]
            
            # Safely get top features with bounds checking
            n_features = min(max_display, len(feature_importance))
            if n_features == 0:
                self.logger.warning("No features available for SHAP plot")
                return None
                
            top_indices = np.argsort(feature_importance)[-n_features:][::-1]
            
            # Ensure indices are within bounds
            top_indices = [i for i in top_indices if 0 <= i < len(numeric_features)]
            
            if not top_indices:
                self.logger.warning("No valid feature indices for SHAP plot")
                return None
                
            top_features = [numeric_features[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            # Create interactive plot
            fig = go.Figure(data=go.Bar(
                x=top_importance,
                y=top_features,
                orientation='h',
                marker_color='rgb(16, 185, 129)',
                name='SHAP Importance'
            ))
            
            fig.update_layout(
                title="SHAP Feature Importance Summary",
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="Features",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f9fafb'),
                height=max(400, len(top_features) * 25)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating SHAP summary plot: {e}")
            return None
    
    def plot_shap_waterfall(self, instance: pd.Series, max_display: int = 10) -> go.Figure:
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            instance: Single instance to explain
            max_display: Maximum number of features to display
            
        Returns:
            Plotly figure
        """
        if self.shap_explainer is None:
            self._initialize_shap_explainer()
            
        if self.shap_explainer is None:
            return None
            
        try:
            # Get SHAP values for single instance
            instance_df = pd.DataFrame([instance])
            shap_values = self.shap_explainer.shap_values(instance_df)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[-1] if len(shap_values) > 1 else shap_values[0]
            
            shap_values = shap_values[0]  # Single instance
            
            # Get base value (expected value)
            if hasattr(self.shap_explainer, 'expected_value'):
                base_value = self.shap_explainer.expected_value
                if isinstance(base_value, np.ndarray):
                    base_value = base_value[-1] if len(base_value) > 1 else base_value[0]
            else:
                base_value = self.y_train.mean()
            
            # Sort by absolute impact
            feature_impacts = [(self.feature_names[i], shap_values[i], instance.iloc[i]) 
                             for i in range(len(shap_values))]
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            feature_impacts = feature_impacts[:max_display]
            
            # Create waterfall data
            names = ['Base Value'] + [f[0] for f in feature_impacts] + ['Prediction']
            values = [base_value] + [f[1] for f in feature_impacts] + [base_value + sum(shap_values)]
            cumulative = [base_value]
            
            for impact in [f[1] for f in feature_impacts]:
                cumulative.append(cumulative[-1] + impact)
            
            colors = ['blue'] + ['green' if v > 0 else 'red' for _, v, _ in feature_impacts] + ['purple']
            
            fig = go.Figure()
            
            # Add bars for each feature impact
            for i, (name, value, feature_val) in enumerate(zip(names[1:-1], 
                                                              [f[1] for f in feature_impacts],
                                                              [f[2] for f in feature_impacts])):
                fig.add_trace(go.Bar(
                    x=[name],
                    y=[abs(value)],
                    base=[cumulative[i] if value > 0 else cumulative[i+1]],
                    marker_color='rgba(16, 185, 129, 0.8)' if value > 0 else 'rgba(239, 68, 68, 0.8)',
                    name=f'{name}: {feature_val:.3f}' if isinstance(feature_val, (int, float)) else f'{name}: {feature_val}',
                    showlegend=False,
                    hovertemplate=f"<b>{name}</b><br>SHAP Value: {value:.4f}<br>Feature Value: {feature_val}<extra></extra>"
                ))
            
            # Add base and prediction lines
            fig.add_hline(y=base_value, line_dash="dash", line_color="blue", 
                         annotation_text=f"Base: {base_value:.3f}")
            fig.add_hline(y=cumulative[-1], line_dash="dash", line_color="purple", 
                         annotation_text=f"Prediction: {cumulative[-1]:.3f}")
            
            fig.update_layout(
                title="SHAP Waterfall Plot - Individual Prediction Explanation",
                xaxis_title="Features",
                yaxis_title="SHAP Value Impact",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f9fafb'),
                height=500,
                xaxis={'categoryorder': 'total descending'}
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating SHAP waterfall plot: {e}")
            return None
    
    def explain_instance_lime(self, instance: pd.Series, num_features: int = 10) -> Dict:
        """
        Explain single instance using LIME.
        
        Args:
            instance: Single instance to explain
            num_features: Number of top features to explain
            
        Returns:
            Dictionary with explanation data
        """
        if self.lime_explainer is None:
            self._initialize_lime_explainer()
            
        if self.lime_explainer is None:
            return None
            
        try:
            if self.is_classifier:
                explanation = self.lime_explainer.explain_instance(
                    instance.values, 
                    self.model.predict_proba,
                    num_features=num_features
                )
            else:
                explanation = self.lime_explainer.explain_instance(
                    instance.values,
                    self.model.predict,
                    num_features=num_features
                )
            
            # Extract explanation data
            exp_data = explanation.as_list()
            features = [item[0] for item in exp_data]
            impacts = [item[1] for item in exp_data]
            
            return {
                'features': features,
                'impacts': impacts,
                'prediction': self.model.predict([instance.values])[0],
                'explanation': explanation
            }
            
        except Exception as e:
            self.logger.error(f"Error creating LIME explanation: {e}")
            return None
    
    def plot_lime_explanation(self, explanation_data: Dict) -> go.Figure:
        """
        Create interactive LIME explanation plot.
        
        Args:
            explanation_data: LIME explanation data
            
        Returns:
            Plotly figure
        """
        if not explanation_data:
            return None
            
        try:
            features = explanation_data['features']
            impacts = explanation_data['impacts']
            
            colors = ['rgba(16, 185, 129, 0.8)' if impact > 0 else 'rgba(239, 68, 68, 0.8)' 
                     for impact in impacts]
            
            fig = go.Figure(data=go.Bar(
                x=impacts,
                y=features,
                orientation='h',
                marker_color=colors,
                name='LIME Impact'
            ))
            
            fig.update_layout(
                title="LIME Local Explanation",
                xaxis_title="Feature Impact on Prediction",
                yaxis_title="Features",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f9fafb'),
                height=max(400, len(features) * 30)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating LIME plot: {e}")
            return None
    
    def plot_partial_dependence(self, feature: str, X: pd.DataFrame = None) -> go.Figure:
        """
        Create partial dependence plot for a feature.
        
        Args:
            feature: Feature name
            X: Input data (uses training data if None)
            
        Returns:
            Plotly figure
        """
        if X is None:
            X = self.X_train
            
        try:
            feature_idx = self.feature_names.index(feature)
            
            # Compute partial dependence
            pd_results = partial_dependence(
                self.model, X, [feature_idx], 
                grid_resolution=20, percentiles=(0.05, 0.95)
            )
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pd_results['grid_values'][0],
                y=pd_results['average'][0],
                mode='lines+markers',
                name='Partial Dependence',
                line=dict(color='rgb(16, 185, 129)', width=3),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=f"Partial Dependence Plot: {feature}",
                xaxis_title=feature,
                yaxis_title="Partial Dependence",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f9fafb'),
                height=400
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating partial dependence plot: {e}")
            return None
    
    def analyze_feature_interactions(self, features: List[str], X: pd.DataFrame = None) -> go.Figure:
        """
        Analyze interactions between two features using partial dependence.
        
        Args:
            features: List of two feature names
            X: Input data (uses training data if None)
            
        Returns:
            Plotly figure
        """
        if len(features) != 2:
            raise ValueError("Exactly two features required for interaction analysis")
            
        if X is None:
            X = self.X_train
            
        try:
            feature_indices = [self.feature_names.index(f) for f in features]
            
            # Compute 2D partial dependence
            pd_results = partial_dependence(
                self.model, X, feature_indices, 
                grid_resolution=20, percentiles=(0.05, 0.95)
            )
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                x=pd_results['grid_values'][0],
                y=pd_results['grid_values'][1],
                z=pd_results['average'],
                colorscale='Viridis',
                hovertemplate=f"{features[0]}: %{{x}}<br>{features[1]}: %{{y}}<br>Partial Dependence: %{{z}}<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"Feature Interaction: {features[0]} vs {features[1]}",
                xaxis_title=features[0],
                yaxis_title=features[1],
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f9fafb'),
                height=500
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating interaction plot: {e}")
            return None
    
    def what_if_analysis(self, base_instance: pd.Series, feature: str, 
                        values: List[float]) -> Dict:
        """
        Perform what-if analysis by changing feature values.
        
        Args:
            base_instance: Base instance to modify
            feature: Feature to modify
            values: List of values to try
            
        Returns:
            Dictionary with analysis results
        """
        try:
            results = []
            base_prediction = self.model.predict([base_instance.values])[0]
            
            for value in values:
                modified_instance = base_instance.copy()
                modified_instance[feature] = value
                
                prediction = self.model.predict([modified_instance.values])[0]
                change = prediction - base_prediction
                
                results.append({
                    'feature_value': value,
                    'prediction': prediction,
                    'change': change
                })
            
            return {
                'base_prediction': base_prediction,
                'feature': feature,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error in what-if analysis: {e}")
            return None
    
    def plot_what_if_analysis(self, what_if_data: Dict) -> go.Figure:
        """
        Plot what-if analysis results.
        
        Args:
            what_if_data: What-if analysis data
            
        Returns:
            Plotly figure
        """
        if not what_if_data:
            return None
            
        try:
            feature = what_if_data['feature']
            results = what_if_data['results']
            base_prediction = what_if_data['base_prediction']
            
            feature_values = [r['feature_value'] for r in results]
            predictions = [r['prediction'] for r in results]
            changes = [r['change'] for r in results]
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Predictions vs Feature Value', 'Change from Base Prediction'),
                vertical_spacing=0.1
            )
            
            # Predictions plot
            fig.add_trace(go.Scatter(
                x=feature_values,
                y=predictions,
                mode='lines+markers',
                name='Predictions',
                line=dict(color='rgb(16, 185, 129)', width=3),
                marker=dict(size=6)
            ), row=1, col=1)
            
            # Base prediction line
            fig.add_hline(y=base_prediction, line_dash="dash", line_color="blue", 
                         annotation_text=f"Base: {base_prediction:.3f}", row=1)
            
            # Changes plot
            colors = ['rgba(16, 185, 129, 0.8)' if c >= 0 else 'rgba(239, 68, 68, 0.8)' 
                     for c in changes]
            
            fig.add_trace(go.Bar(
                x=feature_values,
                y=changes,
                name='Change',
                marker_color=colors
            ), row=2, col=1)
            
            fig.update_xaxes(title_text=feature, row=2, col=1)
            fig.update_yaxes(title_text="Prediction", row=1, col=1)
            fig.update_yaxes(title_text="Change", row=2, col=1)
            
            fig.update_layout(
                title=f"What-If Analysis: {feature}",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f9fafb'),
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating what-if plot: {e}")
            return None
    
    def plot_feature_importance(self, method: str = "shap", max_display: int = 15) -> go.Figure:
        """
        Plot feature importance using different methods.
        
        Args:
            method: Method to use ('shap', 'permutation', 'builtin')
            max_display: Maximum number of features to display
            
        Returns:
            Plotly figure
        """
        try:
            if method == "shap":
                # Use SHAP feature importance if available
                if self.shap_values is not None:
                    shap_values = self.shap_values
                else:
                    # Compute SHAP values first
                    shap_values = self.compute_shap_values(self.X_train.head(50))
                    if shap_values is None:
                        return None
                
                # Ensure shap_values is 2D array
                if shap_values.ndim == 1:
                    shap_values = shap_values.reshape(1, -1)
                
                feature_importance = np.abs(shap_values).mean(axis=0)
                feature_names = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
                
                # Ensure feature_importance is 1D
                if feature_importance.ndim > 1:
                    feature_importance = feature_importance.flatten()
            
            elif method == "permutation":
                # Use permutation importance
                result = permutation_importance(
                    self.model, self.X_train.head(100), self.y_train.head(100),
                    n_repeats=10, random_state=42, n_jobs=1
                )
                feature_importance = result.importances_mean
                feature_names = self.feature_names
            
            elif method == "builtin":
                # Use model's built-in feature importance if available
                if hasattr(self.model, 'feature_importances_'):
                    feature_importance = self.model.feature_importances_
                    feature_names = self.feature_names
                elif hasattr(self.model, 'coef_'):
                    feature_importance = np.abs(self.model.coef_).flatten()
                    feature_names = self.feature_names
                else:
                    raise ValueError("Model does not have built-in feature importance")
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Handle length mismatch
            if len(feature_names) != len(feature_importance):
                min_len = min(len(feature_names), len(feature_importance))
                feature_names = feature_names[:min_len]
                feature_importance = feature_importance[:min_len]
            
            # Safety check for empty arrays
            if len(feature_importance) == 0 or len(feature_names) == 0:
                self.logger.warning("No features available for importance plot")
                return None
            
            # Get top features safely
            n_features = min(max_display, len(feature_importance))
            top_indices = np.argsort(feature_importance)[-n_features:][::-1]
            
            # Ensure indices are within bounds
            top_indices = [i for i in top_indices if 0 <= i < len(feature_names) and 0 <= i < len(feature_importance)]
            
            if not top_indices:
                self.logger.warning("No valid indices for importance plot")
                return None
            
            top_features = [feature_names[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            # Create plot
            fig = go.Figure(data=go.Bar(
                x=top_importance,
                y=top_features,
                orientation='h',
                marker_color='rgb(16, 185, 129)',
                name=f'{method.title()} Importance'
            ))
            
            fig.update_layout(
                title=f"Feature Importance ({method.title()})",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f9fafb'),
                height=max(400, len(top_features) * 25)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating feature importance plot: {e}")
            return None
