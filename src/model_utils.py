"""
Model Training and Hyperparameter Tuning Utilities
Handles model training, hyperparameter optimization, and results management.

This module provides:
- Model pipeline creation
- Hyperparameter tuning with Grid Search, Random Search, and Bayesian Optimization
- Cross-validation
- Model and results persistence
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import get_scorer

import optuna
from optuna.samplers import TPESampler


class ModelTrainer:
    """
    Train and optimize classification models with multiple algorithms and tuning methods.
    """
    
    SUPPORTED_MODELS = {
        'logistic': {
            'class': LogisticRegression,
            'hyperparameters': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear', 'saga'],
                'max_iter': [1000, 5000],
                'class_weight': ['balanced', None],
            }
        },
        'random_forest': {
            'class': RandomForestClassifier,
            'hyperparameters': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', 'balanced_subsample', None],
            }
        },
        'gradient_boosting': {
            'class': GradientBoostingClassifier,
            'hyperparameters': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.001, 0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0],
            }
        },
        'svm': {
            'class': SVC,
            'hyperparameters': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto'],
                'degree': [2, 3, 4],
                'class_weight': ['balanced', None],
                'probability': [True],
            }
        },
        'knn': {
            'class': KNeighborsClassifier,
            'hyperparameters': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'p': [1, 2, 3],
            }
        },
        'decision_tree': {
            'class': DecisionTreeClassifier,
            'hyperparameters': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', None],
            }
        },
    }
    
    def __init__(self, random_state=42):
        """
        Initialize model trainer.
        
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.trained_models = {}
        self.best_models = {}
        self.cv_results = {}
        self._n_jobs_warned = False
        
    def create_model_pipeline(self, model_name, scale_features=True, **model_kwargs):
        """
        Create a pipeline with optional scaling and model.
        
        Parameters
        ----------
        model_name : str
            Name of the model ('logistic', 'random_forest', etc.)
        scale_features : bool
            Whether to include StandardScaler in pipeline
        **model_kwargs : dict
            Additional keyword arguments for the model
            
        Returns
        -------
        sklearn.pipeline.Pipeline
            Pipeline with scaler and model
        """
        if model_name not in self.SUPPORTED_MODELS:
            supported = list(self.SUPPORTED_MODELS.keys())
            raise ValueError(
                f"Model '{model_name}' is not supported. "
                f"Supported models: {', '.join(supported)}"
            )
        
        pipeline_steps = []
        
        # Add scaler if requested
        if scale_features:
            pipeline_steps.append(('scaler', StandardScaler()))
        
        # Add model
        model_class = self.SUPPORTED_MODELS[model_name]['class']
        model_params = dict(model_kwargs)
        if 'random_state' in model_class().get_params():
            model_params['random_state'] = self.random_state

        pipeline_steps.append((
            model_name,
            model_class(**model_params)
        ))
        
        return Pipeline(pipeline_steps)
    
    def tune_with_grid_search(self, pipeline, X_train, y_train, cv=5, n_jobs=-1, verbose=1):
        """
        Optimize hyperparameters using Grid Search CV.
        
        Tests all combinations of parameters (thorough but slow).
        
        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline
            Model pipeline
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        cv : int
            Number of cross-validation folds
        n_jobs : int
            Number of parallel jobs (-1 = use all cores)
        verbose : int
            Verbosity level
            
        Returns
        -------
        sklearn.model_selection.GridSearchCV
            Fitted grid search object
        """
        # Extract model name from pipeline
        model_name = [name for name, _ in pipeline.steps 
                     if name in self.SUPPORTED_MODELS.keys()][0]
        
        # Get hyperparameters for this model
        hyperparams = self.SUPPORTED_MODELS[model_name]['hyperparameters']
        
        # Create parameter grid with pipeline prefixes
        param_grid = {f"{model_name}__{k}": v for k, v in hyperparams.items()}
        
        # Create and fit grid search
        effective_n_jobs = self._normalize_n_jobs(n_jobs)

        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=cv, n_jobs=effective_n_jobs, verbose=verbose,
            scoring='f1_weighted'
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search
    
    def tune_with_random_search(self, pipeline, X_train, y_train, cv=5, 
                               n_iter=20, n_jobs=-1, verbose=1):
        """
        Optimize hyperparameters using Random Search CV.
        
        Samples random combinations (faster than grid search).
        
        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline
            Model pipeline
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        cv : int
            Number of cross-validation folds
        n_iter : int
            Number of random combinations to try
        n_jobs : int
            Number of parallel jobs
        verbose : int
            Verbosity level
            
        Returns
        -------
        sklearn.model_selection.RandomizedSearchCV
            Fitted random search object
        """
        # Extract model name
        model_name = [name for name, _ in pipeline.steps 
                     if name in self.SUPPORTED_MODELS.keys()][0]
        
        # Get hyperparameters
        hyperparams = self.SUPPORTED_MODELS[model_name]['hyperparameters']
        
        # Create parameter distribution
        param_dist = {f"{model_name}__{k}": v for k, v in hyperparams.items()}
        
        # Create and fit random search
        effective_n_jobs = self._normalize_n_jobs(n_jobs)

        random_search = RandomizedSearchCV(
            pipeline, param_dist,
            n_iter=n_iter, cv=cv, n_jobs=effective_n_jobs, verbose=verbose,
            random_state=self.random_state, scoring='f1_weighted'
        )
        
        random_search.fit(X_train, y_train)
        
        return random_search
    
    def tune_with_bayesian_optimization(self, pipeline, X_train, y_train, 
                                       cv=5, n_trials=50):
        """
        Optimize hyperparameters using Bayesian Optimization (Optuna).
        
        Uses Bayesian approach to intelligently search parameter space (fastest).
        
        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline
            Model pipeline
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        cv : int
            Number of cross-validation folds
        n_trials : int
            Number of trials to run
            
        Returns
        -------
        optuna.study.Study
            Optuna study object with optimization results
        """
        # Extract model name
        model_name = [name for name, _ in pipeline.steps 
                     if name in self.SUPPORTED_MODELS.keys()][0]
        
        def objective(trial):
            """Objective function for Optuna"""
            # Suggest hyperparameters
            suggested_params = self._suggest_parameters_for_optuna(trial, model_name)
            
            # Create pipeline
            pipe = self.create_model_pipeline(model_name)
            
            # Set suggested parameters
            for key, value in suggested_params.items():
                pipe.set_params(**{f"{model_name}__{key}": value})
            
            # Cross-validate
            from sklearn.model_selection import cross_val_score
            effective_n_jobs = self._normalize_n_jobs(-1)
            scores = cross_val_score(
                pipe, X_train, y_train, cv=cv,
                scoring='f1_weighted', n_jobs=effective_n_jobs
            )
            
            return scores.mean()
        
        # Create Optuna study
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Fit best model on full training data
        best_params = study.best_params
        best_pipeline = self.create_model_pipeline(model_name)
        for key, value in best_params.items():
            best_pipeline.set_params(**{f"{model_name}__{key}": value})
        best_pipeline.fit(X_train, y_train)

        return SimpleNamespace(
            best_estimator_=best_pipeline,
            best_params_=best_params,
            best_score_=study.best_value,
        )
    
    def _suggest_parameters_for_optuna(self, trial, model_name):
        """
        Suggest hyperparameters for Optuna trial.
        
        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object
        model_name : str
            Model name
            
        Returns
        -------
        dict
            Dictionary of suggested parameters
        """
        if model_name == 'logistic':
            return {
                'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l2']),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 1000, 5000),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            }
        elif model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            }
        elif model_name == 'gradient_boosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            }
        elif model_name == 'svm':
            return {
                'C': trial.suggest_float('C', 1e-1, 1e2, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'degree': trial.suggest_int('degree', 2, 4),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            }
        elif model_name == 'knn':
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
                'p': trial.suggest_int('p', 1, 3),
            }
        elif model_name == 'decision_tree':
            return {
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            }
    
    def perform_cross_validation(self, pipeline, X_train, y_train, cv=5):
        """
        Perform cross-validation with multiple metrics.
        
        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline
            Model pipeline
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        cv : int
            Number of cross-validation folds
            
        Returns
        -------
        dict
            Dictionary with cross-validation results for each metric
        """
        scoring_metrics = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'f1_macro': 'f1_macro',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
        }
        
        effective_n_jobs = self._normalize_n_jobs(-1)
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring=scoring_metrics, return_train_score=True,
            n_jobs=effective_n_jobs
        )
        
        return cv_results
    
    def save_trained_model(self, model, model_name, output_dir='models'):
        """
        Save trained model to disk.
        
        Parameters
        ----------
        model : sklearn estimator
            Trained model or pipeline
        model_name : str
            Name for the model
        output_dir : str
            Directory to save model
            
        Returns
        -------
        Path
            Path where model was saved
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_path / f"{model_name}_{timestamp}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        return filepath
    
    def load_trained_model(self, filepath):
        """
        Load saved model from disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to saved model file
            
        Returns
        -------
        sklearn estimator
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def save_results_to_json(self, results_dict, output_dir='results'):
        """
        Save results to JSON file.
        
        Parameters
        ----------
        results_dict : dict
            Results dictionary to save
        output_dir : str
            Directory to save results
            
        Returns
        -------
        Path
            Path where results were saved
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_path / f"results_{timestamp}.json"
        
        # Make results JSON serializable
        serializable_results = self._make_serializable(results_dict)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        return filepath
    
    @staticmethod
    def _make_serializable(obj):
        """
        Convert numpy and other non-serializable types to JSON-compatible types.
        
        Parameters
        ----------
        obj : any
            Object to convert
            
        Returns
        -------
        any
            JSON-serializable version
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: ModelTrainer._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ModelTrainer._make_serializable(i) for i in obj]
        return obj

    def _normalize_n_jobs(self, n_jobs):
        """
        Avoid Windows process-spawn memory spikes with joblib.
        """
        if os.name == "nt" and n_jobs not in (None, 1):
            if not self._n_jobs_warned:
                print("Using n_jobs=1 on Windows to reduce memory usage during parallel CV.")
                self._n_jobs_warned = True
            return 1
        return n_jobs


class FeatureImportanceAnalyzer:
    """
    Analyze and rank feature importance from trained models.
    """
    
    @staticmethod
    def get_feature_importance(model, feature_names, top_n=None):
        """
        Extract feature importance from tree-based or linear models.
        
        Parameters
        ----------
        model : sklearn estimator
            Trained model
        feature_names : list
            Names of features
        top_n : int, optional
            Return only top N features
            
        Returns
        -------
        pd.DataFrame
            DataFrame with features and their importance scores
        """
        importance_scores = None
        
        # Try to get coefficients (for linear models)
        if hasattr(model, 'coef_'):
            importance_scores = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        
        # Try to get feature importance (for tree-based models)
        elif hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        
        if importance_scores is None:
            raise ValueError("Model does not have feature importance or coefficients")
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
