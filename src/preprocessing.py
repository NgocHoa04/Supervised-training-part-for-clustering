"""
Data Preprocessing Module
Handles data loading, cleaning, and preparation for classification tasks.

This module provides utilities for:
- Loading datasets
- Identifying feature types
- Data splitting (train/test and cross-validation)
- Feature scaling and encoding
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from pathlib import Path


class DataPreprocessor:
    """
    Prepare data for classification models.
    
    Handles feature type identification, data splitting, and preprocessing.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the data preprocessor.
        
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.preprocessor = None
        self.numeric_features = None
        self.categorical_features = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.data_info = {}
        
    def load_data(self, filepath):
        """
        Load data from CSV file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to CSV file
            
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        df = pd.read_csv(filepath)
        self.data_info['filepath'] = str(filepath)
        self.data_info['shape'] = df.shape
        self.data_info['columns'] = df.columns.tolist()
        return df
    
    def get_basic_info(self, df):
        """
        Get basic information about the dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
            
        Returns
        -------
        dict
            Dictionary with shape, dtypes, missing values, etc.
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        return info
    
    def identify_feature_types(self, df, target_column, drop_columns=None):
        """
        Identify numeric and categorical features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        target_column : str
            Name of target/label column
        drop_columns : list, optional
            Columns to exclude from features
            
        Returns
        -------
        tuple
            (numeric_features list, categorical_features list)
        """
        if drop_columns is None:
            drop_columns = []
        
        # Get all feature columns (exclude target and drop_columns)
        exclude_list = [target_column] + drop_columns
        feature_columns = [col for col in df.columns if col not in exclude_list]
        
        # Separate by type
        numeric_features = df[feature_columns].select_dtypes(
            include=['int64', 'float64', 'int32', 'float32']
        ).columns.tolist()
        
        categorical_features = df[feature_columns].select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        return numeric_features, categorical_features
    
    def build_preprocessing_pipeline(self, numeric_features, categorical_features=None):
        """
        Create preprocessing pipeline for features.
        
        Parameters
        ----------
        numeric_features : list
            Names of numeric feature columns
        categorical_features : list, optional
            Names of categorical feature columns
            
        Returns
        -------
        sklearn.compose.ColumnTransformer
            Fitted preprocessor
        """
        if categorical_features is None:
            categorical_features = []
        
        transformers = []
        
        # Add numeric scaler
        if numeric_features:
            transformers.append(
                ('numeric', StandardScaler(), numeric_features)
            )
        
        # Add categorical encoder
        if categorical_features:
            transformers.append(
                ('categorical', 
                 OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
                 categorical_features)
            )
        
        self.preprocessor = ColumnTransformer(transformers=transformers)
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        return self.preprocessor
    
    def split_data(self, X, y, test_size=0.2, stratify=True):
        """
        Split data into training and testing sets.
        
        Parameters
        ----------
        X : pd.DataFrame or array-like
            Features
        y : pd.Series or array-like
            Target labels
        test_size : float
            Fraction of data for testing (default 0.2 = 20%)
        stratify : bool
            Whether to use stratified split to maintain class distribution
            
        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        stratify_arg = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_arg
        )
        return X_train, X_test, y_train, y_test
    
    def get_cross_validation_splitter(self, n_splits=5, stratify=True):
        """
        Create cross-validation splitter.
        
        Parameters
        ----------
        n_splits : int
            Number of cross-validation folds (default 5)
        stratify : bool
            Whether to use stratified k-fold to maintain class distribution
            
        Returns
        -------
        sklearn.model_selection._BaseKFold
            CV splitter object
        """
        if stratify:
            return StratifiedKFold(
                n_splits=n_splits, 
                shuffle=True, 
                random_state=self.random_state
            )
        else:
            return KFold(
                n_splits=n_splits, 
                shuffle=True, 
                random_state=self.random_state
            )
    
    def get_class_distribution(self, y):
        """
        Get class distribution statistics.
        
        Parameters
        ----------
        y : array-like
            Target labels
            
        Returns
        -------
        dict
            Class counts and percentages
        """
        value_counts = pd.Series(y).value_counts().sort_index()
        total = len(y)
        
        distribution = {
            'counts': value_counts.to_dict(),
            'percentages': (value_counts / total * 100).to_dict(),
            'total_samples': total,
            'num_classes': len(value_counts)
        }
        return distribution
