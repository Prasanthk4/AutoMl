"""
Advanced Data Analysis Module
Provides automated EDA, data quality assessment, and smart recommendations
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency, normaltest, jarque_bera
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import IsolationForest
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

warnings.filterwarnings('ignore')

class AdvancedDataAnalyzer:
    """
    Comprehensive data analysis class for automated EDA and data quality assessment.
    
    Features:
    - Automated statistical analysis
    - Data quality scoring
    - Correlation analysis
    - Outlier detection
    - Distribution analysis
    - Smart recommendations
    - Interactive visualizations
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str = None):
        """
        Initialize the data analyzer.
        
        Args:
            df: DataFrame to analyze
            target_column: Name of target column (optional)
        """
        self.df = df.copy()
        self.target_column = target_column
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Remove target from feature lists
        if target_column:
            if target_column in self.numeric_columns:
                self.numeric_columns.remove(target_column)
            if target_column in self.categorical_columns:
                self.categorical_columns.remove(target_column)
        
        self.logger = logging.getLogger(__name__)
        
        # Analysis results cache
        self._analysis_cache = {}
        
    def compute_data_quality_score(self) -> Dict[str, Any]:
        """
        Compute comprehensive data quality score and metrics.
        
        Returns:
            Dictionary with quality metrics and overall score
        """
        if 'data_quality' in self._analysis_cache:
            return self._analysis_cache['data_quality']
        
        metrics = {}
        
        # 1. Completeness (missing data)
        total_cells = self.df.size
        missing_cells = self.df.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells
        metrics['completeness'] = completeness
        metrics['missing_percentage'] = (missing_cells / total_cells) * 100
        
        # 2. Uniqueness (duplicates)
        duplicates = self.df.duplicated().sum()
        uniqueness = 1 - (duplicates / len(self.df))
        metrics['uniqueness'] = uniqueness
        metrics['duplicate_rows'] = duplicates
        
        # 3. Consistency (data type consistency)
        consistency_issues = 0
        for col in self.numeric_columns:
            try:
                pd.to_numeric(self.df[col], errors='coerce')
            except:
                consistency_issues += 1
        
        consistency = 1 - (consistency_issues / max(len(self.df.columns), 1))
        metrics['consistency'] = consistency
        
        # 4. Validity (outliers in numeric columns)
        outlier_ratio = 0
        if self.numeric_columns:
            outlier_counts = []
            for col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                outlier_counts.append(outliers)
            
            total_outliers = sum(outlier_counts)
            total_numeric_values = len(self.numeric_columns) * len(self.df)
            outlier_ratio = total_outliers / max(total_numeric_values, 1)
        
        validity = 1 - min(outlier_ratio, 1)
        metrics['validity'] = validity
        metrics['outlier_percentage'] = outlier_ratio * 100
        
        # 5. Cardinality issues
        high_cardinality_cols = []
        for col in self.categorical_columns:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio > 0.9:  # More than 90% unique values
                high_cardinality_cols.append(col)
        
        cardinality_score = 1 - (len(high_cardinality_cols) / max(len(self.categorical_columns), 1))
        metrics['cardinality_score'] = cardinality_score
        metrics['high_cardinality_columns'] = high_cardinality_cols
        
        # Overall quality score (weighted average)
        weights = {
            'completeness': 0.3,
            'uniqueness': 0.2,
            'consistency': 0.2,
            'validity': 0.2,
            'cardinality_score': 0.1
        }
        
        overall_score = sum(metrics[key] * weight for key, weight in weights.items())
        metrics['overall_quality_score'] = overall_score * 100  # Convert to percentage
        
        # Quality level
        if overall_score >= 0.9:
            quality_level = "Excellent"
        elif overall_score >= 0.8:
            quality_level = "Good"
        elif overall_score >= 0.7:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        metrics['quality_level'] = quality_level
        
        self._analysis_cache['data_quality'] = metrics
        return metrics
    
    def generate_smart_recommendations(self) -> List[Dict[str, str]]:
        """
        Generate smart recommendations based on data analysis.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        quality_metrics = self.compute_data_quality_score()
        
        # Missing data recommendations
        if quality_metrics['missing_percentage'] > 5:
            recommendations.append({
                'category': 'Data Quality',
                'type': 'warning',
                'title': 'High Missing Data',
                'description': f"Dataset has {quality_metrics['missing_percentage']:.1f}% missing values",
                'action': 'Consider data imputation or removing columns with excessive missing values'
            })
        
        # Duplicate data recommendations
        if quality_metrics['duplicate_rows'] > 0:
            recommendations.append({
                'category': 'Data Quality',
                'type': 'warning',
                'title': 'Duplicate Rows',
                'description': f"Found {quality_metrics['duplicate_rows']} duplicate rows",
                'action': 'Remove duplicates before training to avoid data leakage'
            })
        
        # High cardinality recommendations
        if quality_metrics['high_cardinality_columns']:
            recommendations.append({
                'category': 'Feature Engineering',
                'type': 'info',
                'title': 'High Cardinality Features',
                'description': f"Columns with high cardinality: {', '.join(quality_metrics['high_cardinality_columns'])}",
                'action': 'Consider feature engineering or dimensionality reduction'
            })
        
        # Outlier recommendations
        if quality_metrics['outlier_percentage'] > 10:
            recommendations.append({
                'category': 'Data Quality',
                'type': 'warning',
                'title': 'High Outlier Rate',
                'description': f"Dataset has {quality_metrics['outlier_percentage']:.1f}% outliers",
                'action': 'Consider outlier treatment or robust algorithms'
            })
        
        # Dataset size recommendations
        n_rows, n_cols = self.df.shape
        if n_rows < 100:
            recommendations.append({
                'category': 'Sample Size',
                'type': 'warning',
                'title': 'Small Dataset',
                'description': f"Only {n_rows} samples available",
                'action': 'Small datasets may lead to overfitting. Consider cross-validation'
            })
        elif n_rows > 100000:
            recommendations.append({
                'category': 'Performance',
                'type': 'info',
                'title': 'Large Dataset',
                'description': f"Dataset has {n_rows:,} rows",
                'action': 'Consider sampling for faster training or use Fast Mode'
            })
        
        # Feature count recommendations
        if n_cols > 100:
            recommendations.append({
                'category': 'Feature Engineering',
                'type': 'info',
                'title': 'High Dimensionality',
                'description': f"Dataset has {n_cols} features",
                'action': 'Consider feature selection or dimensionality reduction'
            })
        
        # Imbalanced target recommendations (if target provided)
        if self.target_column and self.target_column in self.df.columns:
            target_counts = self.df[self.target_column].value_counts()
            if len(target_counts) > 1:
                imbalance_ratio = target_counts.max() / target_counts.min()
                if imbalance_ratio > 5:
                    recommendations.append({
                        'category': 'Target Analysis',
                        'type': 'warning',
                        'title': 'Imbalanced Target',
                        'description': f"Target imbalance ratio: {imbalance_ratio:.1f}:1",
                        'action': 'Consider resampling techniques or stratified sampling'
                    })
        
        # Task type recommendations
        if self.target_column and self.target_column in self.df.columns:
            target_unique = self.df[self.target_column].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(self.df[self.target_column])
            
            if is_numeric and target_unique > 20:
                recommended_task = "regression"
            elif target_unique <= 10:
                recommended_task = "classification"
            else:
                recommended_task = "classification (consider binning)"
            
            recommendations.append({
                'category': 'Task Type',
                'type': 'info',
                'title': 'Recommended Task Type',
                'description': f"Based on target analysis: {recommended_task}",
                'action': f'Use task_type="{recommended_task.split()[0]}" in AutoML'
            })
        
        return recommendations
    
    def analyze_correlations(self) -> Dict[str, Any]:
        """
        Analyze correlations between features and with target.
        
        Returns:
            Dictionary with correlation analysis results
        """
        if 'correlations' in self._analysis_cache:
            return self._analysis_cache['correlations']
        
        results = {}
        
        # Numeric correlations
        if len(self.numeric_columns) > 1:
            numeric_data = self.df[self.numeric_columns]
            if self.target_column and pd.api.types.is_numeric_dtype(self.df[self.target_column]):
                numeric_data = pd.concat([numeric_data, self.df[self.target_column]], axis=1)
            
            corr_matrix = numeric_data.corr()
            results['numeric_correlation_matrix'] = corr_matrix
            results['correlation_matrix'] = corr_matrix  # Alias for compatibility
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.8:  # High correlation threshold
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
            
            results['high_correlation_pairs'] = high_corr_pairs
        
        # Target correlations (if target exists)
        if self.target_column and self.target_column in self.df.columns:
            target_correlations = []
            
            # Numeric features with numeric target
            if pd.api.types.is_numeric_dtype(self.df[self.target_column]):
                for col in self.numeric_columns:
                    corr = self.df[col].corr(self.df[self.target_column])
                    if not pd.isna(corr):
                        target_correlations.append({
                            'feature': col,
                            'correlation': corr,
                            'abs_correlation': abs(corr)
                        })
            
            # Categorical features (using mutual information)
            if self.categorical_columns:
                # Encode categorical features for mutual information
                encoded_data = pd.DataFrame()
                le = LabelEncoder()
                
                for col in self.categorical_columns:
                    encoded_data[col] = le.fit_transform(self.df[col].astype(str))
                
                # Encode target if categorical
                if pd.api.types.is_numeric_dtype(self.df[self.target_column]):
                    target_encoded = self.df[self.target_column]
                    mi_func = mutual_info_regression
                else:
                    target_encoded = le.fit_transform(self.df[self.target_column].astype(str))
                    mi_func = mutual_info_classif
                
                try:
                    mi_scores = mi_func(encoded_data, target_encoded, random_state=42)
                    for i, col in enumerate(self.categorical_columns):
                        target_correlations.append({
                            'feature': col,
                            'mutual_info_score': mi_scores[i],
                            'feature_type': 'categorical'
                        })
                except Exception as e:
                    self.logger.warning(f"Could not compute mutual information: {e}")
            
            # Sort by importance
            target_correlations.sort(key=lambda x: x.get('abs_correlation', x.get('mutual_info_score', 0)), reverse=True)
            results['target_correlations'] = target_correlations
        
        self._analysis_cache['correlations'] = results
        return results
    
    def detect_outliers(self) -> Dict[str, Any]:
        """
        Detect outliers using multiple methods.
        
        Returns:
            Dictionary with outlier detection results
        """
        if 'outliers' in self._analysis_cache:
            return self._analysis_cache['outliers']
        
        results = {}
        
        if not self.numeric_columns:
            return results
        
        numeric_data = self.df[self.numeric_columns]
        
        # IQR method
        iqr_outliers = {}
        for col in self.numeric_columns:
            Q1 = numeric_data[col].quantile(0.25)
            Q3 = numeric_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)
            outlier_indices = outlier_mask[outlier_mask].index.tolist()
            
            iqr_outliers[col] = {
                'count': len(outlier_indices),
                'percentage': (len(outlier_indices) / len(self.df)) * 100,
                'indices': outlier_indices,
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
        
        results['iqr_outliers'] = iqr_outliers
        
        # Isolation Forest (multivariate outliers)
        try:
            if len(self.numeric_columns) >= 2:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(numeric_data.fillna(numeric_data.mean()))
                
                multivariate_outliers = np.where(outlier_labels == -1)[0].tolist()
                results['multivariate_outliers'] = {
                    'count': len(multivariate_outliers),
                    'percentage': (len(multivariate_outliers) / len(self.df)) * 100,
                    'indices': multivariate_outliers
                }
        except Exception as e:
            self.logger.warning(f"Could not perform multivariate outlier detection: {e}")
        
        self._analysis_cache['outliers'] = results
        return results
    
    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze distributions of numeric features.
        
        Returns:
            Dictionary with distribution analysis results
        """
        if 'distributions' in self._analysis_cache:
            return self._analysis_cache['distributions']
        
        results = {}
        
        for col in self.numeric_columns:
            col_data = self.df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            col_results = {}
            
            # Basic statistics
            col_results['mean'] = col_data.mean()
            col_results['median'] = col_data.median()
            col_results['std'] = col_data.std()
            col_results['skewness'] = col_data.skew()
            col_results['kurtosis'] = col_data.kurtosis()
            
            # Normality tests
            if len(col_data) >= 8:  # Minimum samples for normality test
                try:
                    # Jarque-Bera test
                    jb_stat, jb_pvalue = jarque_bera(col_data)
                    col_results['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_pvalue}
                    
                    # Shapiro-Wilk test (for smaller samples)
                    if len(col_data) <= 5000:
                        shapiro_stat, shapiro_pvalue = stats.shapiro(col_data)
                        col_results['shapiro_wilk'] = {'statistic': shapiro_stat, 'p_value': shapiro_pvalue}
                    
                    # Interpretation
                    is_normal = jb_pvalue > 0.05
                    col_results['is_normal'] = is_normal
                    
                except Exception as e:
                    self.logger.warning(f"Could not perform normality test for {col}: {e}")
            
            # Distribution shape
            if abs(col_results['skewness']) > 1:
                if col_results['skewness'] > 1:
                    shape = "highly right-skewed"
                else:
                    shape = "highly left-skewed"
            elif abs(col_results['skewness']) > 0.5:
                if col_results['skewness'] > 0.5:
                    shape = "moderately right-skewed"
                else:
                    shape = "moderately left-skewed"
            else:
                shape = "approximately symmetric"
            
            col_results['distribution_shape'] = shape
            
            results[col] = col_results
        
        self._analysis_cache['distributions'] = results
        return results
    
    def create_correlation_heatmap(self) -> go.Figure:
        """
        Create interactive correlation heatmap.
        
        Returns:
            Plotly figure with correlation heatmap
        """
        correlations = self.analyze_correlations()
        
        if 'numeric_correlation_matrix' not in correlations:
            return None
        
        corr_matrix = correlations['numeric_correlation_matrix']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f9fafb'),
            height=max(400, len(corr_matrix.columns) * 30)
        )
        
        return fig
    
    def create_distribution_plots(self) -> go.Figure:
        """
        Create distribution plots for numeric features.
        
        Returns:
            Plotly figure with distribution plots
        """
        if not self.numeric_columns:
            return None
        
        n_cols = min(3, len(self.numeric_columns))
        n_rows = (len(self.numeric_columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=self.numeric_columns,
            vertical_spacing=0.1
        )
        
        for i, col in enumerate(self.numeric_columns):
            row = (i // n_cols) + 1
            col_pos = (i % n_cols) + 1
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=self.df[col].dropna(),
                    name=col,
                    opacity=0.7,
                    nbinsx=30,
                    showlegend=False
                ),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title="Feature Distributions",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f9fafb'),
            height=max(400, n_rows * 250)
        )
        
        return fig
    
    def create_target_analysis_plot(self) -> go.Figure:
        """
        Create target variable analysis plot.
        
        Returns:
            Plotly figure with target analysis
        """
        if not self.target_column or self.target_column not in self.df.columns:
            return None
        
        target_data = self.df[self.target_column].dropna()
        
        if pd.api.types.is_numeric_dtype(target_data):
            # Numeric target - histogram
            fig = go.Figure(data=[go.Histogram(
                x=target_data,
                nbinsx=30,
                opacity=0.7,
                marker_color='rgb(16, 185, 129)',
                name='Target Distribution'
            )])
            
            fig.update_layout(
                title=f"Target Variable Distribution: {self.target_column}",
                xaxis_title=self.target_column,
                yaxis_title="Count"
            )
        else:
            # Categorical target - bar chart
            value_counts = target_data.value_counts()
            
            fig = go.Figure(data=[go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                marker_color='rgb(16, 185, 129)',
                name='Target Distribution'
            )])
            
            fig.update_layout(
                title=f"Target Variable Distribution: {self.target_column}",
                xaxis_title=self.target_column,
                yaxis_title="Count"
            )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f9fafb'),
            height=400
        )
        
        return fig
    
    def create_outlier_plot(self) -> go.Figure:
        """
        Create box plots for outlier visualization.
        
        Returns:
            Plotly figure with outlier analysis
        """
        if not self.numeric_columns:
            return None
        
        fig = go.Figure()
        
        for col in self.numeric_columns[:6]:  # Limit to 6 features for readability
            fig.add_trace(go.Box(
                y=self.df[col],
                name=col,
                boxpoints='outliers',
                marker_color='rgb(16, 185, 129)'
            ))
        
        fig.update_layout(
            title="Outlier Detection (Box Plots)",
            yaxis_title="Values",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f9fafb'),
            height=500
        )
        
        return fig
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get comprehensive summary statistics.
        
        Returns:
            DataFrame with summary statistics
        """
        summary_stats = []
        
        # Numeric columns
        for col in self.numeric_columns:
            col_data = self.df[col]
            stats_dict = {
                'Feature': col,
                'Type': 'Numeric',
                'Count': len(col_data.dropna()),
                'Missing': col_data.isnull().sum(),
                'Missing %': (col_data.isnull().sum() / len(col_data)) * 100,
                'Unique': col_data.nunique(),
                'Mean': col_data.mean() if len(col_data.dropna()) > 0 else None,
                'Std': col_data.std() if len(col_data.dropna()) > 0 else None,
                'Min': col_data.min() if len(col_data.dropna()) > 0 else None,
                'Max': col_data.max() if len(col_data.dropna()) > 0 else None
            }
            summary_stats.append(stats_dict)
        
        # Categorical columns
        for col in self.categorical_columns:
            col_data = self.df[col]
            most_common = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None
            
            stats_dict = {
                'Feature': col,
                'Type': 'Categorical',
                'Count': len(col_data.dropna()),
                'Missing': col_data.isnull().sum(),
                'Missing %': (col_data.isnull().sum() / len(col_data)) * 100,
                'Unique': col_data.nunique(),
                'Mean': None,
                'Std': None,
                'Min': None,
                'Max': None,
                'Most Common': most_common
            }
            summary_stats.append(stats_dict)
        
        return pd.DataFrame(summary_stats).round(3)