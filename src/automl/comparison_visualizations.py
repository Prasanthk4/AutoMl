"""
Visualization Module for Model Comparison & A/B Testing

This module provides comprehensive visualizations for model comparison including:
- Performance comparison charts
- ROC curves and precision-recall curves
- Learning curves
- Statistical significance visualizations
- A/B test result plots
- Bootstrap distribution plots
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import learning_curve, validation_curve
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from .model_comparison import ModelResult, ComparisonResult

warnings.filterwarnings('ignore')

class ComparisonVisualizer:
    """
    Comprehensive visualization system for model comparison and A/B testing.
    
    Features:
    - Performance comparison charts
    - ROC and PR curves
    - Learning curves
    - Statistical test visualizations
    - Bootstrap distribution plots
    """
    
    def __init__(self, dark_theme: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            dark_theme: Whether to use dark theme styling
        """
        self.dark_theme = dark_theme
        self.logger = logging.getLogger(__name__)
        
        # Color scheme
        if dark_theme:
            self.colors = {
                'background': 'rgba(0,0,0,0)',
                'paper': 'rgba(0,0,0,0)',
                'text': '#f9fafb',
                'grid': 'rgba(75, 85, 99, 0.3)',
                'primary': '#10b981',
                'secondary': '#3b82f6',
                'accent': '#f59e0b',
                'danger': '#ef4444'
            }
        else:
            self.colors = {
                'background': 'white',
                'paper': 'white',
                'text': 'black',
                'grid': 'rgba(0,0,0,0.1)',
                'primary': '#059669',
                'secondary': '#1d4ed8',
                'accent': '#d97706',
                'danger': '#dc2626'
            }
            
        # Color palette for multiple models
        self.model_colors = [
            '#10b981', '#3b82f6', '#f59e0b', '#ef4444', 
            '#8b5cf6', '#06b6d4', '#84cc16', '#f97316'
        ]
        
    def _apply_theme(self, fig: go.Figure, title: str = "", height: int = 500) -> go.Figure:
        """Apply consistent theming to plotly figures"""
        fig.update_layout(
            title=title,
            title_font_size=16,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['paper'],
            font=dict(color=self.colors['text']),
            height=height,
            showlegend=True,
            legend=dict(
                bgcolor='rgba(255,255,255,0.1)',
                bordercolor=self.colors['grid'],
                borderwidth=1
            )
        )
        
        fig.update_xaxes(
            gridcolor=self.colors['grid'],
            title_font_size=14,
            tickfont_size=12
        )
        fig.update_yaxes(
            gridcolor=self.colors['grid'],
            title_font_size=14,
            tickfont_size=12
        )
        
        return fig
        
    def create_performance_comparison_chart(self, comparison_result: ComparisonResult) -> go.Figure:
        """
        Create a comprehensive performance comparison chart.
        
        Args:
            comparison_result: Results from ModelComparison
            
        Returns:
            Plotly figure with performance comparison
        """
        df = comparison_result.comparison_metrics.copy()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cross-Validation Scores', 'Training Time', 'Prediction Time', 'Primary Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models = df['Model'].tolist()
        colors = self.model_colors[:len(models)]
        
        # Ensure we have enough colors, cycle if needed
        if len(colors) < 4:
            colors = (self.model_colors * 2)[:4]
        
        # 1. CV Scores with error bars
        cv_means = df['CV Mean'].tolist()
        cv_stds = df['CV Std'].tolist()
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=cv_means,
                error_y=dict(type='data', array=cv_stds, visible=True),
                name='CV Score',
                marker_color=colors[0],
                text=[f"{score:.4f}±{std:.4f}" for score, std in zip(cv_means, cv_stds)],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Training Time
        train_times = df['Train Time (s)'].tolist()
        fig.add_trace(
            go.Bar(
                x=models,
                y=train_times,
                name='Train Time',
                marker_color=colors[1 % len(colors)],
                text=[f"{time:.2f}s" for time in train_times],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # 3. Prediction Time
        pred_times = df['Predict Time (s)'].tolist()
        fig.add_trace(
            go.Bar(
                x=models,
                y=pred_times,
                name='Predict Time',
                marker_color=colors[2 % len(colors)],
                text=[f"{time:.4f}s" for time in pred_times],
                textposition='outside'
            ),
            row=2, col=1
        )
        
        # 4. Primary metrics comparison (radar-like bar chart)
        primary_metrics = ['accuracy', 'f1', 'precision', 'recall', 'r2', 'mse', 'mae']
        available_metrics = [col for col in df.columns if col.lower() in primary_metrics]
        
        if available_metrics:
            metric_data = df[available_metrics].iloc[0]  # Take first model as example
            fig.add_trace(
                go.Bar(
                    x=available_metrics,
                    y=metric_data.values,
                    name='Metrics',
                    marker_color=colors[3 % len(colors)],
                    text=[f"{val:.4f}" for val in metric_data.values],
                    textposition='outside'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig = self._apply_theme(fig, "Model Performance Comparison", 600)
        
        return fig
    
    def plot_model_performance_comparison(self, comparison_result: ComparisonResult) -> go.Figure:
        """
        Alias for create_performance_comparison_chart for compatibility.
        
        Args:
            comparison_result: Results from ModelComparison
            
        Returns:
            Plotly figure with performance comparison
        """
        return self.create_performance_comparison_chart(comparison_result)
        
    def create_statistical_significance_chart(self, comparison_result: ComparisonResult) -> go.Figure:
        """
        Create visualization of statistical significance tests.
        
        Args:
            comparison_result: Results from ModelComparison
            
        Returns:
            Plotly figure with statistical test results
        """
        tests = comparison_result.statistical_tests
        
        if not tests:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No statistical tests available<br>(Need at least 2 models)",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return self._apply_theme(fig, "Statistical Significance Tests")
        
        # Extract data for visualization
        comparisons = list(tests.keys())
        p_values = [tests[comp].get('paired_ttest_pvalue', np.nan) for comp in comparisons]
        effect_sizes = [tests[comp].get('cohens_d', np.nan) for comp in comparisons]
        significant = [tests[comp].get('significant', False) for comp in comparisons]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('P-values (Lower is more significant)', 'Effect Sizes (Cohen\'s d)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # P-values plot
        colors_p = [self.colors['primary'] if sig else self.colors['danger'] for sig in significant]
        fig.add_trace(
            go.Bar(
                x=comparisons,
                y=p_values,
                name='P-values',
                marker_color=colors_p,
                text=[f"{p:.6f}" for p in p_values],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Add significance threshold line
        fig.add_hline(
            y=comparison_result.significance_level,
            line_dash="dash",
            line_color=self.colors['accent'],
            annotation_text=f"α = {comparison_result.significance_level}",
            row=1, col=1
        )
        
        # Effect sizes plot
        colors_effect = [self.colors['primary'] if abs(es) >= 0.5 else self.colors['secondary'] for es in effect_sizes]
        fig.add_trace(
            go.Bar(
                x=comparisons,
                y=effect_sizes,
                name='Effect Size',
                marker_color=colors_effect,
                text=[f"{es:.4f}" for es in effect_sizes],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # Add effect size reference lines
        fig.add_hline(y=0.2, line_dash="dot", line_color=self.colors['grid'], annotation_text="Small", row=1, col=2)
        fig.add_hline(y=0.5, line_dash="dot", line_color=self.colors['grid'], annotation_text="Medium", row=1, col=2)
        fig.add_hline(y=0.8, line_dash="dot", line_color=self.colors['grid'], annotation_text="Large", row=1, col=2)
        
        fig = self._apply_theme(fig, "Statistical Significance Analysis", 500)
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig
        
    def create_roc_curves(self, model_results: List[ModelResult], X_test, y_test) -> go.Figure:
        """
        Create ROC curves for classification models.
        
        Args:
            model_results: List of ModelResult objects
            X_test: Test features
            y_test: Test target
            
        Returns:
            Plotly figure with ROC curves
        """
        fig = go.Figure()
        
        for i, result in enumerate(model_results):
            if result.prediction_probabilities is None:
                continue
                
            try:
                # Get probabilities for positive class
                if len(result.prediction_probabilities.shape) == 2:
                    y_scores = result.prediction_probabilities[:, 1]
                else:
                    y_scores = result.prediction_probabilities
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                
                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode='lines',
                        name=f"{result.model_name} (AUC = {roc_auc:.4f})",
                        line=dict(color=self.model_colors[i % len(self.model_colors)], width=2)
                    )
                )
                
            except Exception as e:
                self.logger.warning(f"Could not create ROC curve for {result.model_name}: {e}")
                continue
        
        # Add diagonal reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(dash='dash', color=self.colors['grid'], width=1)
            )
        )
        
        fig = self._apply_theme(fig, "ROC Curves Comparison")
        fig.update_xaxes(title_text="False Positive Rate")
        fig.update_yaxes(title_text="True Positive Rate")
        
        return fig
        
    def create_precision_recall_curves(self, model_results: List[ModelResult], X_test, y_test) -> go.Figure:
        """
        Create precision-recall curves for classification models.
        
        Args:
            model_results: List of ModelResult objects
            X_test: Test features
            y_test: Test target
            
        Returns:
            Plotly figure with PR curves
        """
        fig = go.Figure()
        
        for i, result in enumerate(model_results):
            if result.prediction_probabilities is None:
                continue
                
            try:
                # Get probabilities for positive class
                if len(result.prediction_probabilities.shape) == 2:
                    y_scores = result.prediction_probabilities[:, 1]
                else:
                    y_scores = result.prediction_probabilities
                
                # Calculate PR curve
                precision, recall, _ = precision_recall_curve(y_test, y_scores)
                pr_auc = auc(recall, precision)
                
                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=recall,
                        y=precision,
                        mode='lines',
                        name=f"{result.model_name} (AUC = {pr_auc:.4f})",
                        line=dict(color=self.model_colors[i % len(self.model_colors)], width=2)
                    )
                )
                
            except Exception as e:
                self.logger.warning(f"Could not create PR curve for {result.model_name}: {e}")
                continue
        
        # Add baseline (random classifier)
        baseline = sum(y_test) / len(y_test)
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color=self.colors['grid'],
            annotation_text=f"Random Baseline ({baseline:.3f})"
        )
        
        fig = self._apply_theme(fig, "Precision-Recall Curves")
        fig.update_xaxes(title_text="Recall")
        fig.update_yaxes(title_text="Precision")
        
        return fig
        
    def create_learning_curves(self, model_results: List[ModelResult], X, y, cv_folds: int = 5) -> go.Figure:
        """
        Create learning curves showing model performance vs training set size.
        
        Args:
            model_results: List of ModelResult objects
            X: Features
            y: Target
            cv_folds: Number of CV folds for learning curve
            
        Returns:
            Plotly figure with learning curves
        """
        fig = go.Figure()
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        for i, result in enumerate(model_results):
            try:
                # Calculate learning curve
                train_sizes_abs, train_scores, val_scores = learning_curve(
                    result.model, X, y, 
                    train_sizes=train_sizes,
                    cv=cv_folds,
                    n_jobs=-1,
                    random_state=42
                )
                
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                color = self.model_colors[i % len(self.model_colors)]
                
                # Training scores
                fig.add_trace(
                    go.Scatter(
                        x=train_sizes_abs,
                        y=train_mean,
                        mode='lines+markers',
                        name=f"{result.model_name} (Training)",
                        line=dict(color=color, dash='solid'),
                        error_y=dict(
                            type='data',
                            array=train_std,
                            visible=True,
                            color=color
                        )
                    )
                )
                
                # Validation scores
                fig.add_trace(
                    go.Scatter(
                        x=train_sizes_abs,
                        y=val_mean,
                        mode='lines+markers',
                        name=f"{result.model_name} (Validation)",
                        line=dict(color=color, dash='dash'),
                        error_y=dict(
                            type='data',
                            array=val_std,
                            visible=True,
                            color=color
                        )
                    )
                )
                
            except Exception as e:
                self.logger.warning(f"Could not create learning curve for {result.model_name}: {e}")
                continue
        
        fig = self._apply_theme(fig, "Learning Curves")
        fig.update_xaxes(title_text="Training Set Size")
        fig.update_yaxes(title_text="Score")
        
        return fig
        
    def create_performance_radar_chart(self, comparison_result: ComparisonResult) -> go.Figure:
        """
        Create radar chart comparing model performance across multiple metrics.
        
        Args:
            comparison_result: Results from ModelComparison
            
        Returns:
            Plotly figure with radar chart
        """
        df = comparison_result.comparison_metrics.copy()
        
        # Select metrics for radar chart (normalize to 0-1 scale)
        metrics_for_radar = []
        for col in df.columns:
            if col not in ['Model', 'Train Time (s)', 'Predict Time (s)']:
                if df[col].dtype in ['float64', 'int64'] and not df[col].isna().all():
                    metrics_for_radar.append(col)
        
        if not metrics_for_radar:
            fig = go.Figure()
            fig.add_annotation(
                text="No suitable metrics for radar chart",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return self._apply_theme(fig, "Performance Radar Chart")
        
        fig = go.Figure()
        
        for i, (_, row) in enumerate(df.iterrows()):
            model_name = row['Model']
            
            # Normalize metrics to 0-1 scale
            values = []
            for metric in metrics_for_radar:
                value = row[metric]
                if pd.isna(value):
                    values.append(0)
                else:
                    # Normalize based on metric type
                    if 'time' in metric.lower() or 'mse' in metric.lower() or 'mae' in metric.lower():
                        # Lower is better - invert
                        max_val = df[metric].max()
                        values.append(1 - (value / max_val) if max_val != 0 else 0)
                    else:
                        # Higher is better - normalize
                        max_val = df[metric].max()
                        values.append(value / max_val if max_val != 0 else 0)
            
            # Close the polygon
            theta = metrics_for_radar + [metrics_for_radar[0]]
            r = values + [values[0]]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=r,
                    theta=theta,
                    fill='toself',
                    name=model_name,
                    line_color=self.model_colors[i % len(self.model_colors)],
                    fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(self.model_colors[i % len(self.model_colors)])) + [0.1])}"
                )
            )
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor=self.colors['grid'],
                    tickfont=dict(color=self.colors['text'], size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(color=self.colors['text'], size=11),
                    gridcolor=self.colors['grid']
                )
            )
        )
        
        fig = self._apply_theme(fig, "Performance Radar Chart", 600)
        
        return fig
        
    def create_ab_test_visualization(self, ab_results: Dict[str, Any]) -> go.Figure:
        """
        Create visualization for A/B test results including bootstrap distributions.
        
        Args:
            ab_results: Results from ABTestFramework.run_ab_test()
            
        Returns:
            Plotly figure with A/B test results
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Performance Comparison',
                'Bootstrap Score Distributions',
                'Confidence Intervals',
                'Effect Size Visualization'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 1. Performance comparison bar chart
        models = ['Model A (Control)', 'Model B (Treatment)']
        scores = [ab_results['model_a_score'], ab_results['model_b_score']]
        colors = [self.colors['secondary'], self.colors['primary']]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=scores,
                name='Performance',
                marker_color=colors,
                text=[f"{score:.4f}" for score in scores],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Bootstrap distributions
        scores_a = ab_results['bootstrap_scores_a']
        scores_b = ab_results['bootstrap_scores_b']
        
        fig.add_trace(
            go.Histogram(
                x=scores_a,
                name='Model A Distribution',
                opacity=0.7,
                nbinsx=30,
                marker_color=self.colors['secondary']
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=scores_b,
                name='Model B Distribution',
                opacity=0.7,
                nbinsx=30,
                marker_color=self.colors['primary']
            ),
            row=1, col=2
        )
        
        # 3. Confidence intervals
        ci_a = ab_results['confidence_interval_a']
        ci_b = ab_results['confidence_interval_b']
        
        # Error bar plot for confidence intervals
        fig.add_trace(
            go.Scatter(
                x=models,
                y=scores,
                mode='markers',
                name='Score with CI',
                marker=dict(size=10, color=colors),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[ci_a[1] - ab_results['model_a_score'], ci_b[1] - ab_results['model_b_score']],
                    arrayminus=[ab_results['model_a_score'] - ci_a[0], ab_results['model_b_score'] - ci_b[0]],
                    thickness=3,
                    width=10
                )
            ),
            row=2, col=1
        )
        
        # 4. Effect size visualization
        cohens_d = ab_results['cohens_d']
        p_value = ab_results['p_value']
        
        # Create effect size bar with color coding
        if abs(cohens_d) >= 0.8:
            effect_color = self.colors['primary']
            effect_label = 'Large'
        elif abs(cohens_d) >= 0.5:
            effect_color = self.colors['accent']
            effect_label = 'Medium'
        elif abs(cohens_d) >= 0.2:
            effect_color = self.colors['secondary']
            effect_label = 'Small'
        else:
            effect_color = self.colors['grid']
            effect_label = 'Negligible'
        
        fig.add_trace(
            go.Bar(
                x=['Effect Size (Cohen\'s d)', f'P-value (×{1/ab_results["p_value"]:.0f})'],
                y=[abs(cohens_d), p_value * 10],  # Scale p-value for visibility
                name=f'Effect Size: {effect_label}',
                marker_color=[effect_color, self.colors['danger'] if ab_results['significant'] else self.colors['grid']],
                text=[f"{cohens_d:.4f}", f"{p_value:.6f}"],
                textposition='outside'
            ),
            row=2, col=2
        )
        
        # Add significance threshold line
        fig.add_hline(
            y=0.05 * 10,  # Scaled p-value threshold
            line_dash="dash",
            line_color=self.colors['accent'],
            row=2, col=2
        )
        
        fig = self._apply_theme(fig, f"A/B Test Results - {ab_results['metric'].title()}", 800)
        
        # Add overall result annotation
        result_text = "Statistically Significant" if ab_results['significant'] else "Not Statistically Significant"
        improvement = ab_results['relative_improvement']
        
        fig.add_annotation(
            text=f"<b>{result_text}</b><br>Improvement: {improvement:.2f}%<br>Effect Size: {effect_label}",
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            xanchor='center', yanchor='top',
            showarrow=False,
            font_size=14,
            bgcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(self.colors['primary'])) + [0.1])}",
            bordercolor=self.colors['primary'],
            borderwidth=2
        )
        
        return fig
        
    def create_model_comparison_summary(self, comparison_result: ComparisonResult) -> go.Figure:
        """
        Create a comprehensive summary dashboard for model comparison.
        
        Args:
            comparison_result: Results from ModelComparison
            
        Returns:
            Plotly figure with summary dashboard
        """
        # Create subplots with mixed types
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Performance Overview',
                'Statistical Significance',
                'Training Efficiency',
                'Cross-Validation Stability',
                'Model Rankings',
                'Key Statistics'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "table"}]
            ]
        )
        
        df = comparison_result.comparison_metrics.copy()
        models = df['Model'].tolist()
        colors = self.model_colors[:len(models)]
        
        # 1. Performance overview (bar chart with error bars)
        cv_means = df['CV Mean'].tolist()
        cv_stds = df['CV Std'].tolist()
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=cv_means,
                error_y=dict(type='data', array=cv_stds),
                name='CV Performance',
                marker_color=colors[0],
                text=[f"{mean:.4f}" for mean in cv_means],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Statistical significance summary (changed to bar chart)
        tests = comparison_result.statistical_tests
        if tests:
            significant_count = sum(1 for test in tests.values() if test.get('significant', False))
            total_tests = len(tests)
            
            # Use bar chart instead of pie chart for subplot compatibility
            fig.add_trace(
                go.Bar(
                    x=['Significant', 'Not Significant'],
                    y=[significant_count, total_tests - significant_count],
                    marker_color=[self.colors['primary'], self.colors['grid']],
                    name='Significance Tests',
                    text=[significant_count, total_tests - significant_count],
                    textposition='outside'
                ),
                row=1, col=2
            )
        
        # 3. Training efficiency (scatter plot)
        train_times = df['Train Time (s)'].tolist()
        pred_times = df['Predict Time (s)'].tolist()
        
        fig.add_trace(
            go.Scatter(
                x=train_times,
                y=pred_times,
                mode='markers+text',
                text=models,
                textposition='top center',
                marker=dict(
                    size=15,
                    color=cv_means,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="CV Score")
                ),
                name='Efficiency vs Performance'
            ),
            row=2, col=1
        )
        
        # 4. CV Stability (coefficient of variation)
        cv_coeff = (np.array(cv_stds) / np.array(cv_means)) * 100  # CV as percentage
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=cv_coeff,
                name='CV Coefficient (%)',
                marker_color=colors[1],
                text=[f"{coeff:.2f}%" for coeff in cv_coeff],
                textposition='outside'
            ),
            row=2, col=2
        )
        
        # 5. Model rankings
        rankings = list(range(1, len(models) + 1))
        fig.add_trace(
            go.Bar(
                x=models,
                y=rankings,
                name='Rank',
                marker_color=colors[2],
                text=[f"#{rank}" for rank in rankings],
                textposition='outside'
            ),
            row=3, col=1
        )
        
        # 6. Summary statistics table
        summary_data = []
        for _, row in df.iterrows():
            summary_data.append([
                row['Model'],
                f"{row['CV Mean']:.4f}",
                f"{row['CV Std']:.4f}",
                f"{row['Train Time (s)']:.2f}s",
                f"{row['Predict Time (s)']:.4f}s"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Model', 'CV Mean', 'CV Std', 'Train Time', 'Predict Time'],
                    fill_color=self.colors['primary'],
                    font=dict(color='white')
                ),
                cells=dict(
                    values=list(zip(*summary_data)),
                    fill_color='rgba(255,255,255,0.1)',
                    font=dict(color=self.colors['text'])
                )
            ),
            row=3, col=2
        )
        
        fig = self._apply_theme(fig, "Model Comparison Summary Dashboard", 900)
        
        return fig