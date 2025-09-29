"""
Data preprocessing pipeline for AutoML system.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


class Preprocessor:
    """
    Comprehensive data preprocessing pipeline.
    
    Handles:
    - Missing value imputation
    - Categorical encoding
    - Feature scaling
    - Outlier detection and treatment
    - Feature selection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the preprocessor with configuration."""
        self.config = config or {}
        self.is_fitted = False
        
        # Store transformers
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.encoders = {}
        self.feature_selector = None
        
        # Store column information
        self.numeric_features = []
        self.categorical_features = []
        self.feature_names_out = []
        self.target_encoder = None
        
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        feature_engineering: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_engineering: Whether to perform feature engineering
            
        Returns:
            Tuple of (transformed_X, transformed_y)
        """
        X = X.copy()
        y = y.copy()
        
        # Identify column types
        self._identify_column_types(X)
        
        # Handle missing values
        X = self._handle_missing_values(X, fit=True)
        
        # Encode categorical features
        X = self._encode_categorical_features(X, y, fit=True)
        
        # Handle target encoding if needed
        y = self._encode_target(y, fit=True)
        
        # Scale numerical features
        X = self._scale_features(X, fit=True)
        
        # Handle outliers
        X = self._handle_outliers(X, fit=True)
        
        # Feature engineering if requested
        if feature_engineering:
            X = self._generate_features(X, fit=True)
            
        # Feature selection
        X = self._select_features(X, y, fit=True)
        
        self.is_fitted = True
        self.feature_names_out = X.columns.tolist()
        
        return X, y
        
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            X: Feature matrix
            y: Optional target vector
            
        Returns:
            Tuple of (transformed_X, transformed_y)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        X = X.copy()
        if y is not None:
            y = y.copy()
        
        # Apply all transformations in the same order
        X = self._handle_missing_values(X, fit=False)
        X = self._encode_categorical_features(X, y, fit=False)
        
        if y is not None:
            y = self._encode_target(y, fit=False)
            
        X = self._scale_features(X, fit=False)
        X = self._handle_outliers(X, fit=False)
        X = self._select_features(X, y, fit=False)
        
        # Ensure same columns as training
        X = X.reindex(columns=self.feature_names_out, fill_value=0)
        
        return X, y
        
    def _identify_column_types(self, X: pd.DataFrame) -> None:
        """Identify numeric and categorical columns."""
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def _handle_missing_values(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        strategy = self.config.get('missing_value_strategy', 'auto')
        
        if fit:
            # Numeric features
            if self.numeric_features:
                if strategy == 'auto':
                    # Use median for skewed distributions, mean for normal
                    num_strategy = 'median'
                elif strategy in ['mean', 'median', 'most_frequent']:
                    num_strategy = strategy
                else:
                    num_strategy = 'median'
                    
                self.numeric_imputer = SimpleImputer(strategy=num_strategy)
                X[self.numeric_features] = self.numeric_imputer.fit_transform(X[self.numeric_features])
                
            # Categorical features
            if self.categorical_features:
                self.categorical_imputer = SimpleImputer(strategy='most_frequent')
                X[self.categorical_features] = self.categorical_imputer.fit_transform(X[self.categorical_features])
                
        else:
            # Transform only
            if self.numeric_features and self.numeric_imputer:
                X[self.numeric_features] = self.numeric_imputer.transform(X[self.numeric_features])
            if self.categorical_features and self.categorical_imputer:
                X[self.categorical_features] = self.categorical_imputer.transform(X[self.categorical_features])
                
        return X
        
    def _encode_categorical_features(self, X: pd.DataFrame, y: Optional[pd.Series], fit: bool = False) -> pd.DataFrame:
        """Encode categorical features."""
        if not self.categorical_features:
            return X
            
        encoding_method = self.config.get('categorical_encoding', 'auto')
        max_cardinality = self.config.get('max_categorical_cardinality', 50)
        
        if fit:
            encoded_dfs = []
            
            for col in self.categorical_features:
                unique_values = X[col].nunique()
                
                if encoding_method == 'auto':
                    # Use one-hot for low cardinality, label for high cardinality
                    method = 'onehot' if unique_values <= 10 else 'label'
                else:
                    method = encoding_method
                    
                if method == 'onehot' and unique_values <= max_cardinality:
                    # One-hot encoding
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(X[[col]])
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=X.index)
                    encoded_dfs.append(encoded_df)
                    self.encoders[col] = ('onehot', encoder)
                    
                else:
                    # Label encoding
                    encoder = LabelEncoder()
                    X[col] = encoder.fit_transform(X[col].astype(str))
                    self.encoders[col] = ('label', encoder)
                    
            # Combine one-hot encoded features
            if encoded_dfs:
                encoded_df = pd.concat(encoded_dfs, axis=1)
                X = pd.concat([X.drop(columns=[col for col in self.categorical_features 
                                              if self.encoders[col][0] == 'onehot']), encoded_df], axis=1)
                                              
        else:
            # Transform only
            encoded_dfs = []
            
            for col in self.categorical_features:
                if col in self.encoders:
                    method, encoder = self.encoders[col]
                    
                    if method == 'onehot':
                        encoded_data = encoder.transform(X[[col]])
                        feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                        encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=X.index)
                        encoded_dfs.append(encoded_df)
                    else:
                        # Handle unknown categories for label encoding
                        X[col] = X[col].astype(str)
                        unknown_mask = ~X[col].isin(encoder.classes_)
                        X.loc[unknown_mask, col] = encoder.classes_[0]  # Use first class for unknown
                        X[col] = encoder.transform(X[col])
                        
            # Combine one-hot encoded features
            if encoded_dfs:
                encoded_df = pd.concat(encoded_dfs, axis=1)
                X = pd.concat([X.drop(columns=[col for col in self.categorical_features 
                                              if col in self.encoders and self.encoders[col][0] == 'onehot']), 
                              encoded_df], axis=1)
                              
        return X
        
    def _encode_target(self, y: pd.Series, fit: bool = False) -> pd.Series:
        """Encode target variable if it's categorical."""
        if pd.api.types.is_numeric_dtype(y):
            return y
            
        if fit:
            self.target_encoder = LabelEncoder()
            return pd.Series(self.target_encoder.fit_transform(y), index=y.index, name=y.name)
        else:
            if self.target_encoder:
                # Handle unknown categories
                y_str = y.astype(str)
                unknown_mask = ~y_str.isin(self.target_encoder.classes_)
                y_str.loc[unknown_mask] = self.target_encoder.classes_[0]
                return pd.Series(self.target_encoder.transform(y_str), index=y.index, name=y.name)
            return y
            
    def _scale_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale numerical features."""
        scaling_method = self.config.get('feature_scaling', 'auto')
        
        if scaling_method == 'none':
            return X
            
        # Get current numeric features (after encoding)
        current_numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not current_numeric_features:
            return X
            
        if fit:
            if scaling_method == 'auto':
                # Use robust scaler by default (handles outliers well)
                self.scaler = RobustScaler()
            elif scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaling_method == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = RobustScaler()
                
            X[current_numeric_features] = self.scaler.fit_transform(X[current_numeric_features])
            
        else:
            if self.scaler:
                # Only scale features that were scaled during fit
                features_to_scale = [f for f in current_numeric_features if f in X.columns]
                if features_to_scale:
                    X[features_to_scale] = self.scaler.transform(X[features_to_scale])
                    
        return X
        
    def _handle_outliers(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle outliers in numerical features."""
        outlier_method = self.config.get('outlier_handling', 'auto')
        
        if outlier_method == 'none':
            return X
            
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_features:
            return X
            
        if outlier_method in ['auto', 'clip']:
            threshold = self.config.get('outlier_threshold', 1.5)
            
            for col in numeric_features:
                if fit:
                    # Calculate bounds during fit
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    # Store bounds for transform
                    if not hasattr(self, 'outlier_bounds'):
                        self.outlier_bounds = {}
                    self.outlier_bounds[col] = (lower_bound, upper_bound)
                    
                    # Clip outliers
                    X[col] = X[col].clip(lower_bound, upper_bound)
                    
                else:
                    # Apply stored bounds
                    if hasattr(self, 'outlier_bounds') and col in self.outlier_bounds:
                        lower_bound, upper_bound = self.outlier_bounds[col]
                        X[col] = X[col].clip(lower_bound, upper_bound)
                        
        return X
        
    def _generate_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Generate additional features through feature engineering."""
        # Placeholder for feature engineering
        # Could add polynomial features, interactions, etc.
        return X
        
    def _select_features(self, X: pd.DataFrame, y: Optional[pd.Series], fit: bool = False) -> pd.DataFrame:
        """Select the most important features."""
        if not self.config.get('feature_selection', True) or y is None:
            return X
            
        method = self.config.get('feature_selection_method', 'auto')
        
        # Don't select features if we have very few
        if X.shape[1] <= 5:
            return X
            
        if fit:
            # Determine if this is classification or regression
            is_classification = pd.api.types.is_integer_dtype(y) and y.nunique() < 50
            
            if method == 'auto':
                # Use SelectKBest with appropriate scoring function
                k = min(max(10, X.shape[1] // 2), X.shape[1])  # Select at most half the features
                scoring_func = f_classif if is_classification else f_regression
                self.feature_selector = SelectKBest(score_func=scoring_func, k=k)
                
            if self.feature_selector:
                X_selected = self.feature_selector.fit_transform(X, y)
                selected_features = X.columns[self.feature_selector.get_support()].tolist()
                X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                
        else:
            if self.feature_selector:
                X_selected = self.feature_selector.transform(X)
                selected_features = X.columns[self.feature_selector.get_support()].tolist()
                X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                
        return X
        
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return self.feature_names_out.copy()