"""
Main Analysis Script for Supervised Learning Classification
Complete workflow: data loading, model training, evaluation, and cluster interpretation.

This script answers all project requirements:
1. Multi-class classification metrics with explanations
2. Hyperparameter tuning with multiple models
3. Overfitting detection
4. Feature importance analysis with retraining
5. Cluster interpretation and business insights
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

import config
from preprocessing import DataPreprocessor
from metrics import MetricsCalculator, OverfittingDetector
from model_utils import ModelTrainer, FeatureImportanceAnalyzer
from visualization import ModelVisualizer, ClusterVisualizer


class CompleteAnalysisPipeline:
    """
    Execute complete supervised learning analysis for clustering.
    """
    
    def __init__(self):
        """Initialize pipeline with config settings."""
        self.config = config
        self.preprocessor = DataPreprocessor(random_state=config.RANDOM_STATE)
        self.metrics_calc = MetricsCalculator()
        self.model_trainer = ModelTrainer(random_state=config.RANDOM_STATE)
        self.importance_analyzer = FeatureImportanceAnalyzer()
        self.model_visualizer = ModelVisualizer(
            style=config.PLOT_STYLE,
            figsize=config.PLOT_FIGSIZE,
            dpi=config.PLOT_DPI,
        )
        self.cluster_visualizer = ClusterVisualizer(
            figsize=config.PLOT_FIGSIZE,
            dpi=config.PLOT_DPI,
        )
        
        self.results = {}
        self.all_models = {}
        self.feature_importance_dict = {}
        self.full_df = None
        
    def run_complete_analysis(self):
        """Execute the complete analysis pipeline."""
        print("\n" + "="*80)
        print("SUPERVISED LEARNING CLASSIFICATION ANALYSIS")
        print("="*80)
        
        # Step 1: Load and prepare data
        print("\n[STEP 1] Loading and Preparing Data...")
        df = self._load_data()
        self.full_df = df
        X_train, X_test, y_train, y_test = self._prepare_data(df)
        
        # Step 2: Train models with hyperparameter tuning
        print("\n[STEP 2] Training Models with Hyperparameter Tuning...")
        trained_models = self._train_and_tune_models(X_train, y_train)
        
        # Step 3: Evaluate models
        print("\n[STEP 3] Evaluating Models on Test Set...")
        self._evaluate_models(trained_models, X_test, y_test)
        
        # Step 4: Detect overfitting
        print("\n[STEP 4] Detecting Overfitting...")
        self._detect_overfitting(trained_models, X_train, y_train, X_test, y_test)
        
        # Step 5: Analyze feature importance
        print("\n[STEP 5] Analyzing Feature Importance...")
        top_features = self._analyze_feature_importance(
            trained_models, X_train.columns, y_train
        )
        
        # Step 6: Retrain with important features
        print("\n[STEP 6] Retraining Models with Top Features...")
        self._retrain_with_top_features(X_train, X_test, y_train, y_test, top_features)
        
        # Step 7: Analyze clusters
        print("\n[STEP 7] Analyzing Cluster Profiles...")
        self._analyze_clusters(df, top_features)
        
        # Step 8: Generate visualizations
        print("\n[STEP 8] Generating Visualizations...")
        self._generate_visualizations(trained_models, X_test, y_test, top_features)
        
        # Save results
        print("\n[STEP 9] Saving Results...")
        self._save_results()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {config.RESULTS_OUTPUT_DIR}")
        
    def _load_data(self):
        """Load and display data information."""
        df = self.preprocessor.load_data(config.DATASET_PATH)
        print(f"Data shape: {df.shape}")
        print(f"Data info:\n{self.preprocessor.get_basic_info(df)}")
        return df
    
    def _prepare_data(self, df):
        """Prepare features and target for training."""
        # Separate features and target
        X = df.drop(columns=[config.TARGET_COLUMN] + config.DROP_COLUMNS)
        y = df[config.TARGET_COLUMN]
        
        # Identify feature types
        numeric_features, categorical_features = self.preprocessor.identify_feature_types(
            df, config.TARGET_COLUMN, config.DROP_COLUMNS
        )
        
        print(f"Numeric features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Display class distribution
        print(f"\nClass distribution:")
        dist = self.preprocessor.get_class_distribution(y)
        for class_label, count in dist['counts'].items():
            percentage = dist['percentages'][class_label]
            print(f"  Class {class_label}: {count} samples ({percentage:.1f}%)")
        
        # Build preprocessing pipeline
        self.preprocessor.build_preprocessing_pipeline(numeric_features, categorical_features)
        
        # Fit and transform training data, transform test data
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(
            X, y, test_size=config.TEST_SIZE, stratify=True
        )
        
        # Fit preprocessor on training data
        X_train_processed = self.preprocessor.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.preprocessor.transform(X_test)
        
        # Get feature names after preprocessing
        feature_names = []
        if numeric_features:
            feature_names.extend(numeric_features)
        if categorical_features:
            # Get one-hot encoded column names
            cat_encoder = self.preprocessor.preprocessor.named_transformers_['categorical']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
            feature_names.extend(cat_feature_names)
        
        # Convert to DataFrames for easier handling
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
        
        print(f"\nData split:")
        print(f"  Training set: {X_train_df.shape}")
        print(f"  Test set: {X_test_df.shape}")
        
        return X_train_df, X_test_df, y_train, y_test
    
    def _train_and_tune_models(self, X_train, y_train):
        """Train and tune multiple models."""
        trained_models = {}
        
        for model_name in config.MODELS_TO_TRAIN:
            print(f"\n  Training {model_name.upper()}...")
            
            # Create pipeline
            pipeline = self.model_trainer.create_model_pipeline(model_name)
            
            # Tune hyperparameters
            if config.TUNING_METHOD == 'grid':
                best_model = self.model_trainer.tune_with_grid_search(
                    pipeline, X_train, y_train, **config.GRID_SEARCH_PARAMS
                )
            elif config.TUNING_METHOD == 'random':
                best_model = self.model_trainer.tune_with_random_search(
                    pipeline, X_train, y_train, **config.RANDOM_SEARCH_PARAMS
                )
            elif config.TUNING_METHOD == 'bayesian':
                best_model = self.model_trainer.tune_with_bayesian_optimization(
                    pipeline, X_train, y_train, **config.BAYESIAN_PARAMS
                )
            
            trained_models[model_name] = best_model
            self.all_models[model_name] = best_model
            
            if hasattr(best_model, 'best_score_'):
                print(f"    Best score: {best_model.best_score_:.4f}")
            else:
                print("    Best score: N/A")
        
        return trained_models
    
    def _evaluate_models(self, trained_models, X_test, y_test):
        """Evaluate all models on test set."""
        print("\n  MODEL PERFORMANCE COMPARISON")
        print("  " + "-"*70)
        
        test_scores = {}
        
        for model_name, model_obj in trained_models.items():
            best_model, best_params = self._unwrap_model(model_obj)
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            metrics = self.metrics_calc.calculate_all_metrics(y_test, y_pred)
            test_scores[model_name] = metrics['f1_weighted']
            
            # Store results
            self.results[model_name] = {
                'model': best_model,
                'best_params': best_params,
                'test_metrics': metrics,
                'predictions': y_pred
            }
            
            print(f"\n  {model_name.upper()}")
            summary = self.metrics_calc.get_main_metrics_summary(y_test, y_pred)
            for metric, value in summary.items():
                print(f"    {metric}: {value}")
    
    def _detect_overfitting(self, trained_models, X_train, y_train, X_test, y_test):
        """Detect overfitting by comparing train vs test performance."""
        print("\n  OVERFITTING DETECTION")
        print("  " + "-"*70)
        
        overfitting_results = {}
        
        for model_name, model_obj in trained_models.items():
            best_model, _ = self._unwrap_model(model_obj)
            
            # Train set predictions
            y_train_pred = best_model.predict(X_train)
            train_metrics = self.metrics_calc.calculate_all_metrics(y_train, y_train_pred)
            train_f1 = train_metrics['f1_weighted']
            
            # Test set predictions
            y_test_pred = best_model.predict(X_test)
            test_metrics = self.metrics_calc.calculate_all_metrics(y_test, y_test_pred)
            test_f1 = test_metrics['f1_weighted']
            
            # Detect overfitting
            overfitting_check = OverfittingDetector.detect_overfitting(
                train_f1, test_f1, threshold=config.OVERFITTING_THRESHOLD
            )

            overfitting_check['train_f1'] = train_f1
            overfitting_check['test_f1'] = test_f1
            
            overfitting_results[model_name] = overfitting_check
            
            status = "OVERFITTING DETECTED" if overfitting_check['is_overfitting'] else "No overfitting"
            print(f"\n  {model_name.upper()}: {status}")
            print(f"    Train F1: {train_f1:.4f}")
            print(f"    Test F1:  {test_f1:.4f}")
            print(f"    Gap:      {overfitting_check['gap']:.4f}")
        
        self.results['overfitting_detection'] = overfitting_results
    
    def _analyze_feature_importance(self, trained_models, feature_names, y_train):
        """Analyze feature importance from all models."""
        print("\n  FEATURE IMPORTANCE ANALYSIS")
        print("  " + "-"*70)
        
        # Collect importance from all models
        importance_scores = {}
        
        for model_name, model_obj in trained_models.items():
            best_model, _ = self._unwrap_model(model_obj)
            
            try:
                # Get feature importance
                if hasattr(best_model, 'named_steps'):
                    estimator = best_model.named_steps[list(best_model.named_steps.keys())[-1]]
                else:
                    estimator = best_model

                importance_df = self.importance_analyzer.get_feature_importance(
                    estimator,
                    feature_names
                )
                importance_scores[model_name] = importance_df
                print(f"\n  {model_name.upper()} - Top 5 Features:")
                for idx, row in importance_df.head(5).iterrows():
                    print(f"    {row['feature']}: {row['importance']:.4f}")
            except:
                print(f"\n  {model_name}: Unable to extract feature importance")
        
        # Consensus ranking
        print("\n  CONSENSUS TOP FEATURES (Across all models):")
        if not importance_scores:
            print("\n  No feature importance available from selected models.")
            self.results['top_features'] = []
            return []

        all_importances = pd.concat(
            [df.set_index('feature') for df in importance_scores.values()], axis=1
        )
        consensus_importance = all_importances.mean(axis=1).sort_values(ascending=False)
        top_features = consensus_importance.head(config.TOP_FEATURES_COUNT).index.tolist()
        print(f"  Top {config.TOP_FEATURES_COUNT} features:")
        for i, feature in enumerate(top_features, 1):
            print(f"    {i}. {feature}")
        
        self.feature_importance_dict = importance_scores
        self.results['top_features'] = top_features
        
        return top_features
    
    def _retrain_with_top_features(self, X_train, X_test, y_train, y_test, top_features):
        """Retrain models using only top features."""
        if not config.RETRAIN_WITH_TOP_FEATURES:
            return

        if not top_features:
            print("\n  No top features found. Skipping retraining.")
            return
        
        print("\n  RETRAINING WITH TOP FEATURES ONLY")
        print("  " + "-"*70)
        
        # Select top features
        X_train_top = X_train[top_features]
        X_test_top = X_test[top_features]
        
        retrain_results = {}
        
        for model_name in config.MODELS_TO_TRAIN[:3]:  # Retrain top 3 models
            print(f"\n  {model_name.upper()} (with top {config.TOP_FEATURES_COUNT} features):")
            
            # Create pipeline
            pipeline = self.model_trainer.create_model_pipeline(model_name)
            
            # Train on top features
            if config.TUNING_METHOD == 'grid':
                best_model = self.model_trainer.tune_with_grid_search(
                    pipeline, X_train_top, y_train, cv=config.CV_FOLDS
                )
            elif config.TUNING_METHOD == 'bayesian':
                best_model = self.model_trainer.tune_with_bayesian_optimization(
                    pipeline, X_train_top, y_train, cv=config.CV_FOLDS, n_trials=20
                )
            else:
                best_model = self.model_trainer.tune_with_random_search(
                    pipeline, X_train_top, y_train, cv=config.CV_FOLDS, n_iter=10
                )
            
            # Evaluate
            retrained_model, _ = self._unwrap_model(best_model)
            y_pred = retrained_model.predict(X_test_top)
            metrics = self.metrics_calc.calculate_all_metrics(y_test, y_pred)
            
            retrain_results[model_name] = metrics
            
            # Compare with full features
            original_f1 = self.results[model_name]['test_metrics']['f1_weighted']
            new_f1 = metrics['f1_weighted']
            diff = new_f1 - original_f1
            
            print(f"    Original F1 (all features): {original_f1:.4f}")
            print(f"    New F1 (top features):      {new_f1:.4f}")
            print(f"    Difference:                 {diff:+.4f}")
        
        self.results['retrain_with_top_features'] = retrain_results
    
    def _analyze_clusters(self, df, top_features):
        """Analyze cluster profiles using top features."""
        print("\n  CLUSTER PROFILE ANALYSIS")
        print("  " + "-"*70)
        
        if not top_features:
            print("\n  No top features found. Skipping cluster analysis.")
            return

        # Map encoded feature names back to original columns when possible
        mapped_features = []
        for feature in top_features:
            if feature in df.columns:
                mapped_features.append(feature)
                continue
            base_feature = feature.split('_')[0]
            if base_feature in df.columns:
                mapped_features.append(base_feature)

        # Keep unique features and limit to top N
        seen = set()
        cluster_features = []
        for feature in mapped_features:
            if feature not in seen:
                seen.add(feature)
                cluster_features.append(feature)
        cluster_features = cluster_features[:config.CLUSTER_ANALYSIS_TOP_N_FEATURES]

        if not cluster_features:
            print("\n  No valid original features found for cluster analysis.")
            return

        # Select cluster column and top features
        cluster_col = config.TARGET_COLUMN
        selected_cols = [cluster_col] + cluster_features
        cluster_data = df[selected_cols].copy()
        self.results['cluster_features_used'] = cluster_features
        
        # Analyze each cluster
        print(f"\nCluster profiles (using top {len(cluster_features)} features):\n")
        
        cluster_profiles = {}
        for cluster_id in sorted(cluster_data[cluster_col].unique()):
            cluster_subset = cluster_data[cluster_data[cluster_col] == cluster_id]
            
            profile = {
                'size': len(cluster_subset),
                'percentage': len(cluster_subset) / len(cluster_data) * 100,
                'feature_means': cluster_subset[cluster_features].mean().to_dict()
            }
            
            cluster_profiles[cluster_id] = profile
            
            print(f"Cluster {cluster_id}:")
            print(f"  Size: {profile['size']} samples ({profile['percentage']:.1f}%)")
            print(f"  Key characteristics:")
            for feature, mean_val in profile['feature_means'].items():
                print(f"    {feature}: {mean_val:.4f}")
            print()
        
        self.results['cluster_profiles'] = cluster_profiles
    
    def _generate_visualizations(self, trained_models, X_test, y_test, top_features):
        """Generate all visualizations."""
        print("\n  Generating plots...")
        
        # Model comparison
        test_scores = {
            name: self.results[name]['test_metrics']['f1_weighted']
            for name in self.results if name in config.MODELS_TO_TRAIN
        }

        plot_ext = config.PLOT_FORMAT

        if config.GENERATE_PLOTS.get('model_comparison') and test_scores:
            self.model_visualizer.plot_model_comparison(
                test_scores, 'F1-Score (Weighted)',
                save_path=config.PLOTS_OUTPUT_DIR / f"model_comparison.{plot_ext}"
            )

        # Overfitting detection plot
        overfit = self.results.get('overfitting_detection', {})
        if config.GENERATE_PLOTS.get('overfitting_detection') and overfit:
            train_scores = {k: v['train_f1'] for k, v in overfit.items()}
            test_scores_overfit = {k: v['test_f1'] for k, v in overfit.items()}
            self.model_visualizer.plot_overfitting_detection(
                train_scores,
                test_scores_overfit,
                metric_name='F1-Score (Weighted)',
                save_path=config.PLOTS_OUTPUT_DIR / f"overfitting_detection.{plot_ext}"
            )

        # Feature importance
        if config.GENERATE_PLOTS.get('feature_importance') and self.feature_importance_dict:
            first_key = list(self.feature_importance_dict.keys())[0]
            importance_df = self.feature_importance_dict[first_key]
            self.model_visualizer.plot_feature_importance(
                importance_df,
                top_n=config.TOP_FEATURES_COUNT,
                save_path=config.PLOTS_OUTPUT_DIR / f"feature_importance.{plot_ext}"
            )

        # Confusion matrix for best model
        if config.GENERATE_PLOTS.get('confusion_matrix') and test_scores:
            best_model_name = max(test_scores, key=test_scores.get)
            best_model = self.results[best_model_name]['model']
            y_pred = best_model.predict(X_test)
            class_names = sorted(pd.Series(y_test).unique())
            self.model_visualizer.plot_confusion_matrix(
                y_test,
                y_pred,
                class_names=class_names,
                save_path=config.PLOTS_OUTPUT_DIR / f"confusion_matrix.{plot_ext}"
            )

        # Cluster plots
        if self.full_df is not None and config.ANALYZE_CLUSTER_PROFILES:
            cluster_col = config.TARGET_COLUMN
            if config.GENERATE_PLOTS.get('cluster_distribution'):
                self.cluster_visualizer.plot_cluster_distribution(
                    self.full_df,
                    cluster_column=cluster_col,
                    save_path=config.PLOTS_OUTPUT_DIR / f"cluster_distribution.{plot_ext}"
                )
            cluster_features = self.results.get('cluster_features_used')
            if config.GENERATE_PLOTS.get('cluster_profiles') and cluster_features:
                self.cluster_visualizer.plot_cluster_profiles(
                    self.full_df,
                    cluster_column=cluster_col,
                    top_features=cluster_features,
                    save_path=config.PLOTS_OUTPUT_DIR / f"cluster_profiles.{plot_ext}"
                )
    
    def _save_results(self):
        """Save analysis results to JSON."""
        models_summary = {}
        for model_name in config.MODELS_TO_TRAIN:
            if model_name in self.results:
                models_summary[model_name] = {
                    'best_params': self.results[model_name]['best_params'],
                    'test_metrics': self.results[model_name]['test_metrics'],
                }

        results_to_save = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'tuning_method': config.TUNING_METHOD,
                'cv_folds': config.CV_FOLDS,
                'top_features_count': config.TOP_FEATURES_COUNT,
            },
            'models_trained': config.MODELS_TO_TRAIN,
            'top_features': self.results.get('top_features', []),
            'models_summary': models_summary,
            'overfitting_detection': self.results.get('overfitting_detection', {}),
            'retrain_with_top_features': self.results.get('retrain_with_top_features', {}),
            'cluster_profiles': self.results.get('cluster_profiles', {}),
        }
        
        if config.SAVE_RESULTS:
            results_path = self.model_trainer.save_results_to_json(
                results_to_save,
                output_dir=config.RESULTS_OUTPUT_DIR
            )
            print(f"Results saved to: {results_path}")

    @staticmethod
    def _unwrap_model(model_obj):
        """Return best estimator and best params from a tuning result."""
        if hasattr(model_obj, 'best_estimator_'):
            return model_obj.best_estimator_, getattr(model_obj, 'best_params_', {})
        return model_obj, {}


def main():
    """Main entry point."""
    pipeline = CompleteAnalysisPipeline()
    pipeline.run_complete_analysis()


if __name__ == "__main__":
    main()
