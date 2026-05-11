"""
Configuration for supervised learning classification.
"""

from pathlib import Path

# ============================================================================
# PROJECT SETTINGS
# ============================================================================
PROJECT_NAME = "Supervised Training for Clustering"
PROJECT_DESCRIPTION = "Multi-class classification to explain clusters"
RANDOM_STATE = 42

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_DIR = Path("dataset")
DATASET_PATH = DATA_DIR / "customer_clusters_with_features.csv"
TARGET_COLUMN = "Cluster"
DROP_COLUMNS = []

TEST_SIZE = 0.2
CV_FOLDS = 5

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODELS_TO_TRAIN = [
    "logistic",
    "random_forest",
    "gradient_boosting",
    "svm",
    "knn",
    "decision_tree",
]
PRIMARY_METRIC = "f1_weighted"

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================
TUNING_METHOD = "grid"  # "grid", "random", or "bayesian"

GRID_SEARCH_PARAMS = {
    "cv": CV_FOLDS,
    "n_jobs": -1,
    "verbose": 1,
}

RANDOM_SEARCH_PARAMS = {
    "n_iter": 20,
    "cv": CV_FOLDS,
    "n_jobs": -1,
    "verbose": 1,
}

BAYESIAN_PARAMS = {
    "n_trials": 50,
    "cv": CV_FOLDS,
}

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
TOP_FEATURES_COUNT = 10
RETRAIN_WITH_TOP_FEATURES = True

# ============================================================================
# CLUSTER ANALYSIS
# ============================================================================
ANALYZE_CLUSTER_PROFILES = True
CLUSTER_ANALYSIS_TOP_N_FEATURES = 5

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
RESULTS_OUTPUT_DIR = Path("results")
MODELS_OUTPUT_DIR = Path("models")
PLOTS_OUTPUT_DIR = RESULTS_OUTPUT_DIR / "plots"
LOGS_OUTPUT_DIR = RESULTS_OUTPUT_DIR / "logs"

RESULTS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODELS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LOGS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SAVE_MODELS = True
SAVE_RESULTS = True

# ============================================================================
# VISUALIZATION
# ============================================================================
PLOT_STYLE = "seaborn-v0_8-darkgrid"
PLOT_FIGSIZE = (12, 6)
PLOT_DPI = 300
PLOT_FORMAT = "png"

GENERATE_PLOTS = {
    "model_comparison": True,
    "overfitting_detection": True,
    "feature_importance": True,
    "confusion_matrix": True,
    "cluster_profiles": True,
    "cluster_distribution": True,
}

# ============================================================================
# METRICS
# ============================================================================
EVALUATION_METRICS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1-Score (macro)",
    "precision_weighted": "Precision (weighted)",
    "recall_weighted": "Recall (weighted)",
    "f1_weighted": "F1-Score (weighted)",
    "cohen_kappa": "Cohen's Kappa",
    "matthews_corrcoef": "Matthews Correlation Coefficient",
    "jaccard_macro": "Jaccard (macro)",
    "jaccard_weighted": "Jaccard (weighted)",
}

OVERFITTING_THRESHOLD = 0.05

# ============================================================================
# PREPROCESSING
# ============================================================================
SCALING_METHOD = "standard"
CATEGORICAL_ENCODING = "onehot"
MISSING_VALUE_HANDLER = "mean"

# ============================================================================
# LOGGING
# ============================================================================
ENABLE_LOGGING = True
LOG_FILE = LOGS_OUTPUT_DIR / "analysis_log.txt"

# ============================================================================
# RUN CONTROL
# ============================================================================
RUN_PART_1_METRICS = True
RUN_PART_2_TUNING = True
RUN_PART_3_FEATURES = True
RUN_PART_4_CLUSTERS = True
VERBOSE = 1
