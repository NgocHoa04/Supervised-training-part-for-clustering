"""
Classification Metrics Module
Calculates and explains all metrics for multi-class classification problems.

This module provides comprehensive metrics with mathematical formulas, explanations,
and guidance on when to use each metric for evaluating classification models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef, hamming_loss, jaccard_score,
    balanced_accuracy_score, zero_one_loss
)


class MetricsCalculator:
    """
    Calculate and explain classification metrics for multi-class problems.
    
    Attributes:
        METRICS_DEFINITIONS (dict): Complete definitions of all metrics with formulas,
                                   meaning, usage, and range.
    """
    
    METRICS_DEFINITIONS = {
        'accuracy': {
            'name': 'Accuracy',
            'formula': 'Accuracy = (TP + TN) / (TP + TN + FP + FN)',
            'description': (
                'Percentage of correct predictions out of total predictions. '
                'Simple overall performance metric.'
            ),
            'meaning': (
                'How many predictions were correct in total. '
                'If accuracy is 0.85, then 85% of predictions are correct.'
            ),
            'usage': (
                'Good general metric, but can be misleading with imbalanced data. '
                'Best when all classes are equally important.'
            ),
            'range': '[0, 1] (higher is better)',
            'priority': 'Basic'
        },
        'precision': {
            'name': 'Precision',
            'formula': 'Precision = TP / (TP + FP)',
            'description': (
                'Of all positive predictions made, how many were actually positive. '
                'Focuses on false positives (false alarms).'
            ),
            'meaning': (
                'When the model predicts "positive", how confident are we that it is correct? '
                'High precision means few false alarms.'
            ),
            'usage': (
                'Important when false positives are costly. '
                'Example: Email spam filter - marking legitimate emails as spam is bad.'
            ),
            'range': '[0, 1] (higher is better)',
            'priority': 'Important'
        },
        'recall': {
            'name': 'Recall (Sensitivity)',
            'formula': 'Recall = TP / (TP + FN)',
            'description': (
                'Of all actual positive cases, how many did the model find. '
                'Focuses on false negatives (missed cases).'
            ),
            'meaning': (
                'If there are 100 actual positive cases, how many does the model identify? '
                'High recall means few missed cases.'
            ),
            'usage': (
                'Important when missing positive cases is costly. '
                'Example: Disease detection - missing a sick person is very bad.'
            ),
            'range': '[0, 1] (higher is better)',
            'priority': 'Important'
        },
        'f1_score': {
            'name': 'F1-Score',
            'formula': 'F1 = 2 × (Precision × Recall) / (Precision + Recall)',
            'description': (
                'Harmonic mean of Precision and Recall. '
                'Balances both metrics into single score.'
            ),
            'meaning': (
                'Good balance between precision and recall. '
                'F1-Score considers both false positives and false negatives.'
            ),
            'usage': (
                'Best for imbalanced datasets. Recommended when you care about both '
                'false positives and false negatives equally. Standard metric for most cases.'
            ),
            'range': '[0, 1] (higher is better)',
            'priority': 'Most Important'
        },
        'balanced_accuracy': {
            'name': 'Balanced Accuracy',
            'formula': 'Balanced Accuracy = (1/n) × Σ(TP_i / (TP_i + FN_i))',
            'description': (
                'Average of recall for each class. '
                'Shows performance per class without class weights.'
            ),
            'meaning': (
                'If we have classes A, B, C: average how well we find each class. '
                'Treats all classes equally regardless of frequency.'
            ),
            'usage': (
                'Best for imbalanced datasets where all classes matter equally. '
                'Better than accuracy when classes are unequal in size.'
            ),
            'range': '[0, 1] (higher is better)',
            'priority': 'Important'
        },
        'cohen_kappa': {
            'name': "Cohen's Kappa",
            'formula': "Cohen's Kappa = (p_observed - p_chance) / (1 - p_chance)",
            'description': (
                'Measures agreement between predictions and actual labels. '
                'Removes effect of random agreement (chance).'
            ),
            'meaning': (
                'Beyond random guessing, how much better is our model? '
                'Kappa = 0: same as random guessing. Kappa = 1: perfect agreement.'
            ),
            'usage': (
                'Good for checking real agreement quality. '
                'Not affected by class imbalance. Independent from class distribution.'
            ),
            'range': '[-1, 1] (higher is better, 0 = random)',
            'priority': 'Important'
        },
        'matthews_corrcoef': {
            'name': 'Matthews Correlation Coefficient (MCC)',
            'formula': 'MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))',
            'description': (
                'Correlation coefficient between predictions and actual labels. '
                'Considers all four elements of confusion matrix.'
            ),
            'meaning': (
                'How correlated are predictions with reality? '
                'Good for imbalanced datasets, considers all errors.'
            ),
            'usage': (
                'Best metric for imbalanced data. Handles rare events well. '
                'One of the fairest metrics for imbalanced classification.'
            ),
            'range': '[-1, 1] (higher is better)',
            'priority': 'Most Important for Imbalanced Data'
        },
        'jaccard_score': {
            'name': 'Jaccard Score (Intersection over Union)',
            'formula': 'Jaccard = TP / (TP + FP + FN)',
            'description': (
                'Intersection divided by union of prediction and actual sets. '
                'Sensitive to false positives.'
            ),
            'meaning': (
                'What portion of predicted or actual positive cases are correctly predicted? '
                'Strict metric that penalizes any error.'
            ),
            'usage': (
                'Good for multi-class problems. Stricter than F1-Score. '
                'Useful when errors in any direction are important.'
            ),
            'range': '[0, 1] (higher is better)',
            'priority': 'Useful'
        },
    }
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics_results = {}
        self.macro_average = {}
        self.weighted_average = {}
        
    def calculate_all_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate all metrics for multi-class classification.
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like, optional
            Predicted probabilities (for advanced metrics)
            
        Returns
        -------
        dict
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # Basic overall metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        metrics['zero_one_loss'] = float(zero_one_loss(y_true, y_pred))
        
        # Precision, Recall, F1 - macro (unweighted)
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        
        # Precision, Recall, F1 - weighted (by class frequency)
        metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Special multi-class metrics
        metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
        metrics['matthews_corrcoef'] = float(matthews_corrcoef(y_true, y_pred))
        
        # Per-class detailed report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = class_report
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Jaccard score
        metrics['jaccard_macro'] = float(jaccard_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['jaccard_weighted'] = float(jaccard_score(y_true, y_pred, average='weighted', zero_division=0))
        
        self.metrics_results = metrics
        return metrics
    
    def get_main_metrics_summary(self, y_true, y_pred):
        """
        Get a simple summary of the most important metrics.
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
            
        Returns
        -------
        dict
            Dictionary of key metrics formatted as strings
        """
        summary = {
            'Accuracy': f"{accuracy_score(y_true, y_pred):.4f}",
            'Balanced Accuracy': f"{balanced_accuracy_score(y_true, y_pred):.4f}",
            'Precision (macro)': f"{precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
            'Recall (macro)': f"{recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
            'F1-Score (macro)': f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
            'F1-Score (weighted)': f"{f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}",
            'Cohen Kappa': f"{cohen_kappa_score(y_true, y_pred):.4f}",
            'Matthews Corr. Coef': f"{matthews_corrcoef(y_true, y_pred):.4f}",
        }
        return summary
    
    def get_metric_explanation(self, metric_name):
        """
        Get detailed explanation for a specific metric.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric
            
        Returns
        -------
        dict or None
            Dictionary with formula, meaning, usage, or None if metric not found
        """
        if metric_name in self.METRICS_DEFINITIONS:
            return self.METRICS_DEFINITIONS[metric_name]
        else:
            return None
    
    def generate_detailed_report(self, y_true, y_pred):
        """
        Generate comprehensive metrics report.
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
            
        Returns
        -------
        str
            Formatted text report of all metrics
        """
        report_str = "=" * 90 + "\n"
        report_str += "CLASSIFICATION METRICS DETAILED REPORT\n"
        report_str += "=" * 90 + "\n\n"
        
        metrics = self.calculate_all_metrics(y_true, y_pred)
        
        # Overall metrics section
        report_str += "1. OVERALL PERFORMANCE METRICS\n"
        report_str += "-" * 90 + "\n"
        report_str += f"  Accuracy:                    {metrics['accuracy']:.4f}\n"
        report_str += f"  Balanced Accuracy:           {metrics['balanced_accuracy']:.4f}\n"
        report_str += f"  Cohen's Kappa:               {metrics['cohen_kappa']:.4f}\n"
        report_str += f"  Matthews Correlation Coef:   {metrics['matthews_corrcoef']:.4f}\n"
        report_str += "\n"
        
        # Macro-averaged metrics (unweighted)
        report_str += "2. MACRO-AVERAGED METRICS (all classes weighted equally)\n"
        report_str += "-" * 90 + "\n"
        report_str += f"  Precision (macro):   {metrics['precision_macro']:.4f}\n"
        report_str += f"  Recall (macro):      {metrics['recall_macro']:.4f}\n"
        report_str += f"  F1-Score (macro):    {metrics['f1_macro']:.4f}\n"
        report_str += f"  Jaccard (macro):     {metrics['jaccard_macro']:.4f}\n"
        report_str += "\n"
        
        # Weighted-averaged metrics
        report_str += "3. WEIGHTED-AVERAGED METRICS (by class frequency)\n"
        report_str += "-" * 90 + "\n"
        report_str += f"  Precision (weighted): {metrics['precision_weighted']:.4f}\n"
        report_str += f"  Recall (weighted):    {metrics['recall_weighted']:.4f}\n"
        report_str += f"  F1-Score (weighted):  {metrics['f1_weighted']:.4f}\n"
        report_str += f"  Jaccard (weighted):   {metrics['jaccard_weighted']:.4f}\n"
        report_str += "\n"
        
        # Confusion Matrix
        report_str += "4. CONFUSION MATRIX\n"
        report_str += "-" * 90 + "\n"
        cm = confusion_matrix(y_true, y_pred)
        report_str += str(cm) + "\n\n"
        
        # Detailed classification report
        report_str += "5. DETAILED CLASSIFICATION REPORT (per class)\n"
        report_str += "-" * 90 + "\n"
        report_str += classification_report(y_true, y_pred) + "\n"
        
        return report_str
    
    @staticmethod
    def create_metrics_explanation_table():
        """
        Create a dataframe with all metrics definitions.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Metric, Formula, Meaning, Usage, Range, Priority
        """
        data = {
            'Metric': [],
            'Formula': [],
            'Meaning': [],
            'Usage': [],
            'Range': [],
            'Priority': []
        }
        
        for metric_name, details in MetricsCalculator.METRICS_DEFINITIONS.items():
            data['Metric'].append(details['name'])
            data['Formula'].append(details['formula'])
            data['Meaning'].append(details['meaning'])
            data['Usage'].append(details['usage'])
            data['Range'].append(details['range'])
            data['Priority'].append(details['priority'])
        
        return pd.DataFrame(data)


class OverfittingDetector:
    """
    Detect overfitting by comparing training and validation/test metrics.
    """
    
    @staticmethod
    def detect_overfitting(train_score, val_score, threshold=0.05):
        """
        Simple overfitting detection based on score difference.
        
        Parameters
        ----------
        train_score : float
            Training metric (e.g., F1-score on train set)
        val_score : float
            Validation/test metric (e.g., F1-score on validation set)
        threshold : float
            Threshold for overfitting (default 0.05 = 5% difference)
            
        Returns
        -------
        dict
            Contains 'is_overfitting' bool and 'gap' float
        """
        gap = train_score - val_score
        is_overfitting = gap > threshold
        
        return {
            'is_overfitting': is_overfitting,
            'gap': gap,
            'train_score': train_score,
            'val_score': val_score,
            'threshold': threshold
        }
    
    @staticmethod
    def analyze_cv_scores(cv_scores):
        """
        Analyze cross-validation scores for overfitting signs.
        
        Parameters
        ----------
        cv_scores : array-like
            Cross-validation scores from all folds
            
        Returns
        -------
        dict
            Statistics about variance across folds
        """
        return {
            'mean': float(np.mean(cv_scores)),
            'std': float(np.std(cv_scores)),
            'min': float(np.min(cv_scores)),
            'max': float(np.max(cv_scores)),
            'cv_range': float(np.max(cv_scores) - np.min(cv_scores))
        }
