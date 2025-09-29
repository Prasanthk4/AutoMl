"""
Experiment History Management & Analytics

This module provides comprehensive experiment analysis and management capabilities including:
- Experiment history querying and filtering
- Performance trend analysis
- Hyperparameter optimization tracking  
- Experiment comparison and ranking
- Data export and reporting
- Interactive visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import json

from .experiment_tracking import ExperimentTracker, ExperimentRecord, ExperimentConfig

class ExperimentHistory:
    """
    Comprehensive experiment history management system.
    
    Features:
    - Advanced querying and filtering
    - Trend analysis and performance tracking
    - Hyperparameter optimization insights
    - Data export capabilities
    """
    
    def __init__(self, tracker: ExperimentTracker):
        """
        Initialize experiment history manager.
        
        Args:
            tracker: ExperimentTracker instance
        """
        self.tracker = tracker
        self.logger = logging.getLogger(__name__)
    
    def search_experiments(self, query: str = None, filters: Dict[str, Any] = None,
                         date_range: Tuple[datetime, datetime] = None,
                         sort_by: str = "timestamp", ascending: bool = False,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Advanced experiment search with multiple criteria.
        
        Args:
            query: Text search in experiment names and descriptions
            filters: Dictionary of filters (task_type, dataset_name, status, etc.)
            date_range: Tuple of (start_date, end_date)
            sort_by: Field to sort by
            ascending: Sort order
            limit: Maximum results to return
            
        Returns:
            List of experiment summaries
        """
        # Get all experiments
        all_experiments = self.tracker.list_experiments(limit=1000)
        
        # Apply filters
        filtered_experiments = []
        
        for exp in all_experiments:
            # Text search
            if query:
                search_text = f"{exp.get('experiment_name', '')} {exp.get('description', '')}".lower()
                if query.lower() not in search_text:
                    continue
            
            # Date range filter
            if date_range:
                exp_date = datetime.fromisoformat(exp['timestamp'].replace('Z', '+00:00'))
                if not (date_range[0] <= exp_date <= date_range[1]):
                    continue
            
            # Apply additional filters
            if filters:
                skip_exp = False
                for key, value in filters.items():
                    if key in exp and exp[key] != value:
                        skip_exp = True
                        break
                if skip_exp:
                    continue
            
            filtered_experiments.append(exp)
        
        # Sort results
        if sort_by in ['timestamp']:
            filtered_experiments.sort(
                key=lambda x: x.get(sort_by, ''),
                reverse=not ascending
            )
        elif sort_by in ['duration']:
            filtered_experiments.sort(
                key=lambda x: x.get(sort_by, 0),
                reverse=not ascending
            )
        
        return filtered_experiments[:limit]
    
    def get_experiment_metrics_history(self, experiment_ids: List[str] = None,
                                     metric_name: str = None,
                                     date_range: Tuple[datetime, datetime] = None) -> pd.DataFrame:
        """
        Get metrics history across experiments.
        
        Args:
            experiment_ids: Specific experiments to include (None for all)
            metric_name: Specific metric to track
            date_range: Date range filter
            
        Returns:
            DataFrame with metrics history
        """
        if experiment_ids is None:
            experiments = self.search_experiments(date_range=date_range, limit=1000)
            experiment_ids = [exp['experiment_id'] for exp in experiments]
        
        metrics_data = []
        
        for exp_id in experiment_ids:
            experiment = self.tracker.get_experiment(exp_id)
            if not experiment:
                continue
            
            # Basic experiment info
            base_data = {
                'experiment_id': exp_id,
                'experiment_name': experiment.config.experiment_name,
                'timestamp': experiment.timestamp,
                'task_type': experiment.config.task_type,
                'dataset_name': experiment.config.dataset_name,
                'target_column': experiment.config.target_column,
                'duration': experiment.duration,
                'status': experiment.status
            }
            
            # Primary metric
            if hasattr(experiment.metrics, 'primary_metric'):
                metric_data = base_data.copy()
                metric_data['metric_name'] = experiment.metrics.primary_metric_name
                metric_data['metric_value'] = experiment.metrics.primary_metric
                metric_data['is_primary'] = True
                
                if metric_name is None or metric_name == experiment.metrics.primary_metric_name:
                    metrics_data.append(metric_data)
            
            # Additional metrics
            if experiment.metrics.additional_metrics:
                for name, value in experiment.metrics.additional_metrics.items():
                    if metric_name is None or metric_name == name:
                        metric_data = base_data.copy()
                        metric_data['metric_name'] = name
                        metric_data['metric_value'] = value
                        metric_data['is_primary'] = False
                        metrics_data.append(metric_data)
            
            # CV metrics
            for attr in ['cv_mean', 'cv_std', 'train_time', 'predict_time']:
                value = getattr(experiment.metrics, attr, None)
                if value is not None and (metric_name is None or metric_name == attr):
                    metric_data = base_data.copy()
                    metric_data['metric_name'] = attr
                    metric_data['metric_value'] = value
                    metric_data['is_primary'] = False
                    metrics_data.append(metric_data)
        
        return pd.DataFrame(metrics_data)
    
    def get_hyperparameter_analysis(self, parameter_name: str,
                                  metric_name: str = None,
                                  task_type: str = None,
                                  limit: int = 100) -> pd.DataFrame:
        """
        Analyze hyperparameter impact on model performance.
        
        Args:
            parameter_name: Hyperparameter to analyze
            metric_name: Metric to correlate with
            task_type: Filter by task type
            limit: Maximum experiments to analyze
            
        Returns:
            DataFrame with hyperparameter analysis
        """
        filters = {}
        if task_type:
            filters['task_type'] = task_type
        
        experiments = self.search_experiments(filters=filters, limit=limit)
        
        analysis_data = []
        
        for exp_summary in experiments:
            experiment = self.tracker.get_experiment(exp_summary['experiment_id'])
            if not experiment:
                continue
            
            # Check if parameter exists
            if parameter_name not in experiment.parameters:
                continue
            
            param_value = experiment.parameters[parameter_name]
            
            # Get metric value
            if metric_name:
                if metric_name == experiment.metrics.primary_metric_name:
                    metric_value = experiment.metrics.primary_metric
                elif metric_name in experiment.metrics.additional_metrics:
                    metric_value = experiment.metrics.additional_metrics[metric_name]
                elif hasattr(experiment.metrics, metric_name):
                    metric_value = getattr(experiment.metrics, metric_name)
                else:
                    continue
            else:
                metric_value = experiment.metrics.primary_metric
                metric_name = experiment.metrics.primary_metric_name
            
            analysis_data.append({
                'experiment_id': experiment.experiment_id,
                'experiment_name': experiment.config.experiment_name,
                'parameter_name': parameter_name,
                'parameter_value': param_value,
                'parameter_type': type(param_value).__name__,
                'metric_name': metric_name,
                'metric_value': metric_value,
                'timestamp': experiment.timestamp,
                'task_type': experiment.config.task_type,
                'dataset_name': experiment.config.dataset_name
            })
        
        return pd.DataFrame(analysis_data)
    
    def get_best_experiments(self, metric_name: str = None, 
                           task_type: str = None,
                           dataset_name: str = None,
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get top performing experiments.
        
        Args:
            metric_name: Metric to rank by (None for primary metric)
            task_type: Filter by task type
            dataset_name: Filter by dataset
            top_k: Number of top experiments to return
            
        Returns:
            List of top experiments with metrics
        """
        filters = {}
        if task_type:
            filters['task_type'] = task_type
        if dataset_name:
            filters['dataset_name'] = dataset_name
        
        experiments = self.search_experiments(filters=filters, limit=1000)
        
        experiment_scores = []
        
        for exp_summary in experiments:
            experiment = self.tracker.get_experiment(exp_summary['experiment_id'])
            if not experiment:
                continue
            
            # Get metric value
            if metric_name:
                if metric_name == experiment.metrics.primary_metric_name:
                    score = experiment.metrics.primary_metric
                elif metric_name in experiment.metrics.additional_metrics:
                    score = experiment.metrics.additional_metrics[metric_name]
                elif hasattr(experiment.metrics, metric_name):
                    score = getattr(experiment.metrics, metric_name)
                else:
                    continue
            else:
                score = experiment.metrics.primary_metric
                metric_name = experiment.metrics.primary_metric_name
            
            experiment_scores.append({
                'experiment_id': experiment.experiment_id,
                'experiment_name': experiment.config.experiment_name,
                'score': score,
                'metric_name': metric_name,
                'timestamp': experiment.timestamp,
                'task_type': experiment.config.task_type,
                'dataset_name': experiment.config.dataset_name,
                'duration': experiment.duration,
                'parameters': experiment.parameters
            })
        
        # Sort by score (higher is better for most metrics)
        # Note: For error metrics like MSE, lower is better, but we'll assume higher is better by default
        experiment_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return experiment_scores[:top_k]
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments side by side.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            Comparison results
        """
        if len(experiment_ids) < 2:
            raise ValueError("At least 2 experiments required for comparison")
        
        experiments = []
        for exp_id in experiment_ids:
            experiment = self.tracker.get_experiment(exp_id)
            if experiment:
                experiments.append(experiment)
        
        if len(experiments) < 2:
            raise ValueError("Could not load enough experiments for comparison")
        
        # Comparison data
        comparison = {
            'experiments': [],
            'common_parameters': {},
            'different_parameters': {},
            'metrics_comparison': {},
            'performance_ranking': []
        }
        
        # Collect experiment summaries
        all_parameters = {}
        all_metrics = {}
        
        for exp in experiments:
            exp_data = {
                'experiment_id': exp.experiment_id,
                'name': exp.config.experiment_name,
                'timestamp': exp.timestamp,
                'duration': exp.duration,
                'task_type': exp.config.task_type,
                'dataset_name': exp.config.dataset_name,
                'status': exp.status,
                'primary_metric': exp.metrics.primary_metric,
                'primary_metric_name': exp.metrics.primary_metric_name
            }
            comparison['experiments'].append(exp_data)
            
            # Collect parameters
            for param, value in exp.parameters.items():
                if param not in all_parameters:
                    all_parameters[param] = {}
                all_parameters[param][exp.experiment_id] = value
            
            # Collect metrics
            all_metrics[exp.experiment_id] = {
                exp.metrics.primary_metric_name: exp.metrics.primary_metric,
                **exp.metrics.additional_metrics
            }
            
            # Add CV metrics
            for attr in ['cv_mean', 'cv_std', 'train_time', 'predict_time']:
                value = getattr(exp.metrics, attr, None)
                if value is not None:
                    all_metrics[exp.experiment_id][attr] = value
        
        # Analyze parameters
        for param, values in all_parameters.items():
            unique_values = set(values.values())
            if len(unique_values) == 1:
                comparison['common_parameters'][param] = list(unique_values)[0]
            else:
                comparison['different_parameters'][param] = values
        
        # Metrics comparison
        comparison['metrics_comparison'] = all_metrics
        
        # Performance ranking
        comparison['performance_ranking'] = sorted(
            experiments,
            key=lambda x: x.metrics.primary_metric,
            reverse=True
        )
        
        return comparison
    
    def export_experiments(self, experiment_ids: List[str] = None,
                          export_format: str = "csv",
                          include_parameters: bool = True,
                          include_metrics: bool = True) -> Union[pd.DataFrame, str]:
        """
        Export experiment data to various formats.
        
        Args:
            experiment_ids: Specific experiments to export (None for all)
            export_format: Export format (csv, json, excel)
            include_parameters: Include parameter columns
            include_metrics: Include metric columns
            
        Returns:
            DataFrame or JSON string depending on format
        """
        if experiment_ids is None:
            experiments = self.search_experiments(limit=1000)
            experiment_ids = [exp['experiment_id'] for exp in experiments]
        
        export_data = []
        
        for exp_id in experiment_ids:
            experiment = self.tracker.get_experiment(exp_id)
            if not experiment:
                continue
            
            # Basic experiment info
            row = {
                'experiment_id': exp_id,
                'experiment_name': experiment.config.experiment_name,
                'description': experiment.config.description,
                'task_type': experiment.config.task_type,
                'dataset_name': experiment.config.dataset_name,
                'target_column': experiment.config.target_column,
                'timestamp': experiment.timestamp,
                'duration': experiment.duration,
                'status': experiment.status,
                'notes': experiment.notes,
                'tags': ', '.join(experiment.config.tags) if experiment.config.tags else ''
            }
            
            # Add parameters
            if include_parameters:
                for param, value in experiment.parameters.items():
                    row[f'param_{param}'] = value
            
            # Add metrics
            if include_metrics:
                row['primary_metric'] = experiment.metrics.primary_metric
                row['primary_metric_name'] = experiment.metrics.primary_metric_name
                
                # CV metrics
                for attr in ['cv_mean', 'cv_std', 'train_time', 'predict_time']:
                    value = getattr(experiment.metrics, attr, None)
                    if value is not None:
                        row[f'metric_{attr}'] = value
                
                # Additional metrics
                for metric, value in experiment.metrics.additional_metrics.items():
                    row[f'metric_{metric}'] = value
            
            export_data.append(row)
        
        df = pd.DataFrame(export_data)
        
        if export_format.lower() == 'csv':
            return df
        elif export_format.lower() == 'json':
            return df.to_json(orient='records', date_format='iso')
        elif export_format.lower() == 'excel':
            return df
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

class ExperimentAnalytics:
    """
    Advanced analytics and visualization for experiment tracking.
    
    Features:
    - Performance trend analysis
    - Hyperparameter optimization insights
    - Experiment comparison visualizations
    - Statistical analysis of results
    """
    
    def __init__(self, history: ExperimentHistory, dark_theme: bool = True):
        """
        Initialize experiment analytics.
        
        Args:
            history: ExperimentHistory instance
            dark_theme: Use dark theme for plots
        """
        self.history = history
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
    
    def _apply_theme(self, fig: go.Figure, title: str = "", height: int = 500) -> go.Figure:
        """Apply consistent theming to plotly figures"""
        fig.update_layout(
            title=title,
            title_font_size=16,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['paper'],
            font=dict(color=self.colors['text']),
            height=height,
            showlegend=True
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
    
    def create_performance_timeline(self, task_type: str = None, 
                                  metric_name: str = None,
                                  date_range: Tuple[datetime, datetime] = None) -> go.Figure:
        """
        Create performance timeline showing improvement over time.
        
        Args:
            task_type: Filter by task type
            metric_name: Specific metric to track
            date_range: Date range for timeline
            
        Returns:
            Plotly figure with performance timeline
        """
        # Get metrics history
        filters = {'task_type': task_type} if task_type else None
        experiments = self.history.search_experiments(filters=filters, date_range=date_range, limit=200)
        experiment_ids = [exp['experiment_id'] for exp in experiments]
        
        metrics_df = self.history.get_experiment_metrics_history(
            experiment_ids=experiment_ids,
            metric_name=metric_name,
            date_range=date_range
        )
        
        if metrics_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No experiments found for the specified criteria",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return self._apply_theme(fig, "Performance Timeline")
        
        # Group by metric name
        fig = go.Figure()
        
        for metric in metrics_df['metric_name'].unique():
            metric_data = metrics_df[metrics_df['metric_name'] == metric]
            metric_data = metric_data.sort_values('timestamp')
            
            fig.add_trace(go.Scatter(
                x=metric_data['timestamp'],
                y=metric_data['metric_value'],
                mode='lines+markers',
                name=metric,
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate=(
                    f"<b>{metric}</b><br>"
                    "Value: %{y:.4f}<br>"
                    "Date: %{x}<br>"
                    "Experiment: %{customdata}<br>"
                    "<extra></extra>"
                ),
                customdata=metric_data['experiment_name']
            ))
        
        fig = self._apply_theme(fig, "Performance Timeline", 500)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Metric Value")
        
        return fig
    
    def create_hyperparameter_analysis(self, parameter_name: str,
                                     metric_name: str = None,
                                     task_type: str = None) -> go.Figure:
        """
        Create hyperparameter impact analysis visualization.
        
        Args:
            parameter_name: Hyperparameter to analyze
            metric_name: Metric to correlate with
            task_type: Filter by task type
            
        Returns:
            Plotly figure with hyperparameter analysis
        """
        analysis_df = self.history.get_hyperparameter_analysis(
            parameter_name=parameter_name,
            metric_name=metric_name,
            task_type=task_type
        )
        
        if analysis_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No experiments found with parameter '{parameter_name}'",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return self._apply_theme(fig, f"Hyperparameter Analysis: {parameter_name}")
        
        # Determine plot type based on parameter type
        param_type = analysis_df['parameter_type'].iloc[0]
        
        fig = go.Figure()
        
        if param_type in ['int', 'float']:
            # Scatter plot for numeric parameters
            fig.add_trace(go.Scatter(
                x=analysis_df['parameter_value'],
                y=analysis_df['metric_value'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=analysis_df['metric_value'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Metric Value")
                ),
                hovertemplate=(
                    f"<b>{parameter_name}</b>: %{{x}}<br>"
                    f"<b>{analysis_df['metric_name'].iloc[0]}</b>: %{{y:.4f}}<br>"
                    "Experiment: %{customdata}<br>"
                    "<extra></extra>"
                ),
                customdata=analysis_df['experiment_name'],
                name="Experiments"
            ))
            
            # Add trend line if enough data points
            if len(analysis_df) >= 3:
                try:
                    z = np.polyfit(analysis_df['parameter_value'], analysis_df['metric_value'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(analysis_df['parameter_value'].min(), 
                                        analysis_df['parameter_value'].max(), 100)
                    fig.add_trace(go.Scatter(
                        x=x_trend,
                        y=p(x_trend),
                        mode='lines',
                        name='Trend',
                        line=dict(color=self.colors['accent'], dash='dash')
                    ))
                except:
                    pass
            
            fig.update_xaxes(title_text=parameter_name)
            
        else:
            # Box plot or bar chart for categorical parameters
            unique_values = analysis_df['parameter_value'].unique()
            
            if len(unique_values) <= 10:  # Box plot for few categories
                for value in unique_values:
                    value_data = analysis_df[analysis_df['parameter_value'] == value]
                    fig.add_trace(go.Box(
                        y=value_data['metric_value'],
                        name=str(value),
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8
                    ))
                    
                fig.update_xaxes(title_text=parameter_name)
            else:
                # Bar chart for many categories (show top/bottom performers)
                grouped = analysis_df.groupby('parameter_value')['metric_value'].agg(['mean', 'count']).reset_index()
                grouped = grouped.sort_values('mean', ascending=False)
                
                fig.add_trace(go.Bar(
                    x=grouped['parameter_value'],
                    y=grouped['mean'],
                    marker_color=self.colors['primary'],
                    hovertemplate=(
                        f"<b>{parameter_name}</b>: %{{x}}<br>"
                        "Mean Metric: %{y:.4f}<br>"
                        "Count: %{customdata}<br>"
                        "<extra></extra>"
                    ),
                    customdata=grouped['count']
                ))
                
                fig.update_xaxes(title_text=parameter_name)
        
        fig.update_yaxes(title_text=f"{analysis_df['metric_name'].iloc[0]} Value")
        
        return self._apply_theme(fig, f"Hyperparameter Analysis: {parameter_name}")
    
    def create_experiment_comparison(self, experiment_ids: List[str]) -> go.Figure:
        """
        Create side-by-side experiment comparison visualization.
        
        Args:
            experiment_ids: Experiments to compare
            
        Returns:
            Plotly figure with comparison
        """
        comparison = self.history.compare_experiments(experiment_ids)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Performance Comparison',
                'Training Time Comparison', 
                'Parameter Differences',
                'Metrics Overview'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "table"}]
            ]
        )
        
        experiments = comparison['experiments']
        
        # 1. Performance comparison
        fig.add_trace(
            go.Bar(
                x=[exp['name'] for exp in experiments],
                y=[exp['primary_metric'] for exp in experiments],
                name='Performance',
                marker_color=self.colors['primary'],
                text=[f"{exp['primary_metric']:.4f}" for exp in experiments],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Training time comparison
        fig.add_trace(
            go.Bar(
                x=[exp['name'] for exp in experiments],
                y=[exp['duration'] for exp in experiments],
                name='Duration',
                marker_color=self.colors['secondary'],
                text=[f"{exp['duration']:.1f}s" for exp in experiments],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # 3. Parameter differences (show only different parameters)
        param_names = list(comparison['different_parameters'].keys())[:5]  # Top 5
        
        if param_names:
            for i, param_name in enumerate(param_names):
                param_values = comparison['different_parameters'][param_name]
                y_values = [str(param_values.get(exp['experiment_id'], 'N/A')) for exp in experiments]
                
                fig.add_trace(
                    go.Bar(
                        x=[exp['name'] for exp in experiments],
                        y=[i+1] * len(experiments),  # Constant height for visualization
                        name=param_name,
                        text=y_values,
                        textposition='middle',
                        hovertemplate=f"<b>{param_name}</b>: %{{text}}<extra></extra>"
                    ),
                    row=2, col=1
                )
        
        # 4. Metrics table
        metrics_data = []
        all_metric_names = set()
        
        for exp_id, metrics in comparison['metrics_comparison'].items():
            all_metric_names.update(metrics.keys())
        
        # Create table data
        table_headers = ['Experiment'] + list(all_metric_names)
        table_cells = []
        
        for exp in experiments:
            row = [exp['name']]
            exp_metrics = comparison['metrics_comparison'][exp['experiment_id']]
            for metric_name in all_metric_names:
                value = exp_metrics.get(metric_name, 'N/A')
                if isinstance(value, (int, float)):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            table_cells.append(row)
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=table_headers,
                    fill_color=self.colors['primary'],
                    font=dict(color='white')
                ),
                cells=dict(
                    values=list(zip(*table_cells)),
                    fill_color='rgba(255,255,255,0.1)',
                    font=dict(color=self.colors['text'])
                )
            ),
            row=2, col=2
        )
        
        fig = self._apply_theme(fig, "Experiment Comparison", 800)
        
        return fig
    
    def create_leaderboard(self, task_type: str = None, 
                         metric_name: str = None,
                         top_k: int = 10) -> go.Figure:
        """
        Create leaderboard of top performing experiments.
        
        Args:
            task_type: Filter by task type
            metric_name: Metric to rank by
            top_k: Number of top experiments to show
            
        Returns:
            Plotly figure with leaderboard
        """
        best_experiments = self.history.get_best_experiments(
            metric_name=metric_name,
            task_type=task_type,
            top_k=top_k
        )
        
        if not best_experiments:
            fig = go.Figure()
            fig.add_annotation(
                text="No experiments found for leaderboard",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return self._apply_theme(fig, "Experiment Leaderboard")
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        experiment_names = [exp['experiment_name'] for exp in best_experiments]
        scores = [exp['score'] for exp in best_experiments]
        
        # Reverse for proper display (highest at top)
        experiment_names.reverse()
        scores.reverse()
        
        fig.add_trace(go.Bar(
            x=scores,
            y=experiment_names,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=f"{best_experiments[0]['metric_name']}")
            ),
            text=[f"{score:.4f}" for score in scores],
            textposition='outside',
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{best_experiments[0]['metric_name']}: %{{x:.4f}}<br>"
                "<extra></extra>"
            )
        ))
        
        fig = self._apply_theme(fig, f"Top {top_k} Experiments - {best_experiments[0]['metric_name']}")
        fig.update_xaxes(title_text=best_experiments[0]['metric_name'])
        fig.update_yaxes(title_text="Experiments")
        
        return fig