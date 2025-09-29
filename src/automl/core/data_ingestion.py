"""
Data ingestion module for loading and profiling various data formats.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
from pathlib import Path
import logging
try:
    from sqlalchemy import create_engine
except ImportError:
    create_engine = None


class DataIngestion:
    """
    Handles data loading and profiling from various sources.
    
    Supports:
    - CSV files
    - Excel files (.xlsx, .xls)
    - JSON files
    - Parquet files
    - Database connections
    
    Args:
        verbose (bool): Whether to print detailed information
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
    def load_and_profile(
        self, 
        data_source: Union[str, pd.DataFrame], 
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load data from source and generate comprehensive profile.
        
        Args:
            data_source: File path, connection string, or DataFrame
            **kwargs: Additional arguments for data loading
            
        Returns:
            Tuple of (DataFrame, profile_dict)
        """
        # Load data
        if isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
            source_info = "DataFrame"
        else:
            df, source_info = self._load_from_source(data_source, **kwargs)
            
        # Generate profile
        profile = self._generate_profile(df, source_info)
        
        if self.verbose:
            self._print_profile_summary(profile)
            
        return df, profile
        
    def _load_from_source(
        self, 
        source: str, 
        **kwargs
    ) -> Tuple[pd.DataFrame, str]:
        """Load data from various file formats or databases."""
        source_path = Path(source)
        
        if source_path.suffix.lower() == '.csv':
            df = self._load_csv(source, **kwargs)
            source_info = f"CSV file: {source}"
            
        elif source_path.suffix.lower() in ['.xlsx', '.xls']:
            df = self._load_excel(source, **kwargs)
            source_info = f"Excel file: {source}"
            
        elif source_path.suffix.lower() == '.json':
            df = self._load_json(source, **kwargs)
            source_info = f"JSON file: {source}"
            
        elif source_path.suffix.lower() == '.parquet':
            df = self._load_parquet(source, **kwargs)
            source_info = f"Parquet file: {source}"
            
        elif source.startswith(('sqlite://', 'postgresql://', 'mysql://')):
            df = self._load_from_database(source, **kwargs)
            source_info = f"Database: {source.split('://')[0]}"
            
        else:
            # Try to infer format or default to CSV
            try:
                df = pd.read_csv(source, **kwargs)
                source_info = f"CSV file (inferred): {source}"
            except Exception as e:
                raise ValueError(f"Unsupported data source format: {source}. Error: {e}")
                
        return df, source_info
        
    def _load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load CSV file with automatic encoding detection."""
        default_kwargs = {
            'encoding': 'utf-8',
            'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'nan', 'NaN'],
            'keep_default_na': True
        }
        default_kwargs.update(kwargs)
        
        try:
            df = pd.read_csv(filepath, **default_kwargs)
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    default_kwargs['encoding'] = encoding
                    df = pd.read_csv(filepath, **default_kwargs)
                    if self.verbose:
                        self.logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode CSV file with any supported encoding")
                
        return df
        
    def _load_excel(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        default_kwargs = {
            'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'nan', 'NaN']
        }
        default_kwargs.update(kwargs)
        
        df = pd.read_excel(filepath, **default_kwargs)
        return df
        
    def _load_json(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load JSON file."""
        default_kwargs = {
            'orient': 'records'
        }
        default_kwargs.update(kwargs)
        
        df = pd.read_json(filepath, **default_kwargs)
        return df
        
    def _load_parquet(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load Parquet file."""
        df = pd.read_parquet(filepath, **kwargs)
        return df
        
    def _load_from_database(self, connection_string: str, **kwargs) -> pd.DataFrame:
        """Load data from database."""
        if 'query' not in kwargs and 'table_name' not in kwargs:
            raise ValueError("Either 'query' or 'table_name' must be provided for database loading")
            
        engine = create_engine(connection_string)
        
        if 'query' in kwargs:
            df = pd.read_sql_query(kwargs['query'], engine)
        else:
            df = pd.read_sql_table(kwargs['table_name'], engine)
            
        return df
        
    def _generate_profile(self, df: pd.DataFrame, source_info: str) -> Dict[str, Any]:
        """Generate comprehensive data profile."""
        data_quality = self._assess_data_quality(df)
        
        profile = {
            'source_info': source_info,
            'basic_info': self._get_basic_info(df),
            'column_info': self._get_column_info(df),
            'data_quality': data_quality,
            'statistics': self._get_statistics(df),
            'recommendations': self._generate_recommendations(df),
            # Add data quality score at top level for compatibility
            'data_quality_score': data_quality.get('data_quality_score', data_quality.get('overall_quality_score', 0))
        }
        
        return profile
        
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            'shape': df.shape,
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
        
    def _get_column_info(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about each column."""
        column_info = {}
        
        for col in df.columns:
            col_data = df[col]
            
            # Basic info
            info = {
                'dtype': str(col_data.dtype),
                'non_null_count': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'null_percentage': (col_data.isnull().sum() / len(df)) * 100,
                'unique_count': col_data.nunique(),
                'unique_percentage': (col_data.nunique() / len(df)) * 100
            }
            
            # Infer semantic type
            info['semantic_type'] = self._infer_semantic_type(col_data)
            
            # Type-specific analysis
            if pd.api.types.is_numeric_dtype(col_data):
                info.update(self._analyze_numeric_column(col_data))
            elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
                info.update(self._analyze_categorical_column(col_data))
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                info.update(self._analyze_datetime_column(col_data))
                
            column_info[col] = info
            
        return column_info
        
    def _infer_semantic_type(self, series: pd.Series) -> str:
        """Infer the semantic type of a column."""
        # Check for datetime patterns
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
            
        # Check for numeric types
        if pd.api.types.is_numeric_dtype(series):
            if series.dtype in ['int64', 'int32', 'int16', 'int8']:
                # Check if it could be categorical
                unique_ratio = series.nunique() / len(series)
                if unique_ratio < 0.05 and series.nunique() < 50:
                    return 'categorical_int'
                return 'integer'
            else:
                return 'float'
                
        # Check for boolean
        if series.dtype == 'bool':
            return 'boolean'
            
        # For object/string columns
        if series.dtype == 'object':
            # Check if all values look like numbers
            sample_non_null = series.dropna().head(100)
            if len(sample_non_null) > 0:
                try:
                    pd.to_numeric(sample_non_null)
                    return 'numeric_string'
                except ValueError:
                    pass
                    
            # Check for high cardinality (potential ID column)
            unique_ratio = series.nunique() / len(series)
            if unique_ratio > 0.9:
                return 'identifier'
            elif unique_ratio < 0.1:
                return 'categorical'
            else:
                return 'text'
                
        return 'unknown'
        
    def _analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column."""
        stats = series.describe()
        
        analysis = {
            'min': stats['min'],
            'max': stats['max'],
            'mean': stats['mean'],
            'median': stats['50%'],
            'std': stats['std'],
            'q25': stats['25%'],
            'q75': stats['75%'],
            'skewness': series.skew(),
            'kurtosis': series.kurtosis()
        }
        
        # Detect outliers using IQR
        q1, q3 = stats['25%'], stats['75%']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        analysis['outlier_count'] = len(outliers)
        analysis['outlier_percentage'] = (len(outliers) / len(series)) * 100
        
        # Check for potential distributions
        analysis['distribution_hint'] = self._guess_distribution(series)
        
        return analysis
        
    def _analyze_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical column."""
        value_counts = series.value_counts()
        
        analysis = {
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
            'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
            'top_5_values': value_counts.head(5).to_dict()
        }
        
        # Check for imbalance
        if len(value_counts) > 1:
            imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
            analysis['imbalance_ratio'] = imbalance_ratio
            analysis['is_imbalanced'] = imbalance_ratio > 10
            
        return analysis
        
    def _analyze_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze datetime column."""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return {'error': 'No non-null datetime values'}
            
        analysis = {
            'min_date': non_null_series.min(),
            'max_date': non_null_series.max(),
            'date_range_days': (non_null_series.max() - non_null_series.min()).days,
            'most_frequent_date': non_null_series.mode().iloc[0] if len(non_null_series.mode()) > 0 else None
        }
        
        return analysis
        
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        
        quality = {
            'completeness_score': ((total_cells - null_cells) / total_cells) * 100,
            'columns_with_nulls': df.columns[df.isnull().any()].tolist(),
            'high_null_columns': df.columns[df.isnull().sum() / len(df) > 0.5].tolist(),
            'constant_columns': df.columns[df.nunique() <= 1].tolist(),
            'duplicate_columns': self._find_duplicate_columns(df)
        }
        
        # Overall quality score
        quality_factors = [
            quality['completeness_score'] / 100,  # Completeness
            max(0, 1 - len(quality['high_null_columns']) / df.shape[1]),  # Low null rate
            max(0, 1 - len(quality['constant_columns']) / df.shape[1]),  # No constant columns
            max(0, 1 - df.duplicated().sum() / len(df))  # Low duplicate rate
        ]
        
        quality['overall_quality_score'] = np.mean(quality_factors) * 100
        quality['data_quality_score'] = quality['overall_quality_score']  # Alias for compatibility
        
        return quality
        
    def _find_duplicate_columns(self, df: pd.DataFrame) -> List[List[str]]:
        """Find columns with identical values."""
        duplicate_groups = []
        processed = set()
        
        for col1 in df.columns:
            if col1 in processed:
                continue
                
            duplicates = [col1]
            for col2 in df.columns:
                if col2 != col1 and col2 not in processed:
                    if df[col1].equals(df[col2]):
                        duplicates.append(col2)
                        
            if len(duplicates) > 1:
                duplicate_groups.append(duplicates)
                processed.update(duplicates)
            else:
                processed.add(col1)
                
        return duplicate_groups
        
    def _get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get overall dataset statistics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        
        stats = {
            'n_numeric': len(numeric_cols),
            'n_categorical': len(categorical_cols),
            'n_datetime': len(datetime_cols),
            'numeric_columns': numeric_cols.tolist(),
            'categorical_columns': categorical_cols.tolist(),
            'datetime_columns': datetime_cols.tolist()
        }
        
        if len(numeric_cols) > 0:
            stats['correlation_matrix'] = df[numeric_cols].corr().to_dict()
            
        return stats
        
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate data preprocessing recommendations."""
        recommendations = []
        
        # Check for missing values
        null_columns = df.columns[df.isnull().any()]
        if len(null_columns) > 0:
            recommendations.append(f"Handle missing values in {len(null_columns)} columns")
            
        # Check for high cardinality categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        high_cardinality = [col for col in categorical_cols 
                           if df[col].nunique() > 50 and df[col].nunique() / len(df) > 0.1]
        if high_cardinality:
            recommendations.append(f"Consider feature engineering for high-cardinality columns: {high_cardinality}")
            
        # Check for skewed numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skewed_cols = [col for col in numeric_cols 
                      if abs(df[col].skew()) > 2]
        if skewed_cols:
            recommendations.append(f"Consider log transformation for skewed columns: {skewed_cols}")
            
        # Check for outliers
        outlier_cols = []
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
            if len(outliers) / len(df) > 0.05:  # More than 5% outliers
                outlier_cols.append(col)
        if outlier_cols:
            recommendations.append(f"Consider outlier treatment for columns: {outlier_cols}")
            
        # Check for feature scaling needs
        if len(numeric_cols) > 1:
            ranges = [(df[col].max() - df[col].min()) for col in numeric_cols]
            if max(ranges) / min(ranges) > 100:
                recommendations.append("Consider feature scaling due to different value ranges")
                
        return recommendations
        
    def _guess_distribution(self, series: pd.Series) -> str:
        """Guess the distribution type of a numeric series."""
        # Simple heuristic based on skewness and kurtosis
        skew = abs(series.skew())
        kurt = series.kurtosis()
        
        if skew < 0.5 and -1 < kurt < 1:
            return "normal"
        elif skew > 1:
            return "right_skewed"
        elif skew < -1:
            return "left_skewed"
        elif kurt > 3:
            return "heavy_tailed"
        else:
            return "unknown"
            
    def _print_profile_summary(self, profile: Dict[str, Any]) -> None:
        """Print a summary of the data profile."""
        basic = profile['basic_info']
        quality = profile['data_quality']
        
        print("\n" + "="*60)
        print("DATA PROFILE SUMMARY")
        print("="*60)
        print(f"Source: {profile['source_info']}")
        print(f"Shape: {basic['shape']}")
        print(f"Memory Usage: {basic['memory_usage_mb']:.2f} MB")
        print(f"Duplicates: {basic['duplicate_rows']} ({basic['duplicate_percentage']:.1f}%)")
        print(f"Data Quality Score: {quality['overall_quality_score']:.1f}/100")
        
        if quality['columns_with_nulls']:
            print(f"Columns with nulls: {len(quality['columns_with_nulls'])}")
            
        if profile['recommendations']:
            print(f"\nRecommendations:")
            for rec in profile['recommendations']:
                print(f"  • {rec}")
                
        print("="*60)