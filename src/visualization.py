"""
Visualization Utilities Module
Creates plots and visualizations for model analysis and cluster interpretation.

Includes:
- Model comparison plots
- Overfitting detection plots
- Feature importance visualizations
- Cluster profile plots
- Confusion matrices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix


class ModelVisualizer:
    """Create visualizations for model evaluation and comparison."""
    
    def __init__(self, style='seaborn-v0_8-darkgrid', figsize=(12, 6), dpi=300):
        """
        Initialize visualizer with style settings.
        
        Parameters
        ----------
        style : str
            Matplotlib style
        figsize : tuple
            Default figure size (width, height)
        """
        plt.style.use(style)
        self.figsize = figsize
        self.dpi = dpi
        self.output_dir = Path('results')
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def plot_model_comparison(self, models_metrics, metric_name='f1_weighted', save_path=None):
        """
        Compare models on a specific metric.
        
        Parameters
        ----------
        models_metrics : dict
            Dictionary with model names as keys and metric values as values
        metric_name : str
            Name of metric to display on plot
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        models = list(models_metrics.keys())
        scores = list(models_metrics.values())
        
        # Create bar plot
        bars = ax.bar(models, scores, color='steelblue', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison - {metric_name}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_overfitting_detection(self, train_scores, val_scores, metric_name='F1-Score', 
                                   save_path=None):
        """
        Visualize train vs validation scores to detect overfitting.
        
        Parameters
        ----------
        train_scores : dict
            Dictionary with model names and training scores
        val_scores : dict
            Dictionary with model names and validation scores
        metric_name : str
            Name of metric
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        models = list(train_scores.keys())
        x = np.arange(len(models))
        width = 0.35
        
        train_values = [train_scores[m] for m in models]
        val_values = [val_scores[m] for m in models]
        
        # Create grouped bars
        bars1 = ax.bar(x - width/2, train_values, width, label='Training', 
                       color='green', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, val_values, width, label='Validation', 
                       color='orange', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        # Formatting
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_title(f'Overfitting Detection - {metric_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add gap annotations
        for i, model in enumerate(models):
            gap = train_values[i] - val_values[i]
            if gap > 0.05:  # Significant overfitting
                ax.text(i, max(train_values[i], val_values[i]) + 0.05,
                       f'Gap: {gap:.3f}', ha='center', fontsize=9, color='red', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, importance_df, top_n=20, save_path=None):
        """
        Plot feature importance ranking.
        
        Parameters
        ----------
        importance_df : pd.DataFrame
            DataFrame with 'feature' and 'importance' columns
        top_n : int
            Number of top features to display
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        top_features = importance_df.head(top_n).copy()
        top_features = top_features.sort_values('importance')
        
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.3)))
        
        bars = ax.barh(top_features['feature'], top_features['importance'], 
                       color='steelblue', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {width:.4f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """
        Plot confusion matrix as heatmap.
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        class_names : list, optional
            Names of classes
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        # Formatting
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig


class ClusterVisualizer:
    """Create visualizations for cluster analysis and interpretation."""
    
    def __init__(self, figsize=(14, 8), dpi=300):
        """
        Initialize cluster visualizer.
        
        Parameters
        ----------
        figsize : tuple
            Default figure size
        """
        self.figsize = figsize
        self.dpi = dpi
        self.output_dir = Path('results')
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def plot_cluster_profiles(self, cluster_data, cluster_column='Cluster', 
                             top_features=None, save_path=None):
        """
        Create heatmap showing cluster profiles.
        
        Parameters
        ----------
        cluster_data : pd.DataFrame
            Data with cluster labels and features
        cluster_column : str
            Name of cluster column
        top_features : list, optional
            Specific features to plot (if None, use first 10 numeric)
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        # Select features to plot
        if top_features is None:
            numeric_cols = cluster_data.select_dtypes(
                include=[np.number]).columns.tolist()
            # Remove cluster column if it's numeric
            if cluster_column in numeric_cols:
                numeric_cols.remove(cluster_column)
            top_features = numeric_cols[:10]
        
        # Calculate mean values per cluster
        cluster_means = cluster_data.groupby(cluster_column)[top_features].mean()
        
        # Normalize for better visualization
        cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(cluster_means_norm.T, annot=cluster_means.T.round(3),
                   fmt='.3f', cmap='RdYlGn', cbar=True, ax=ax,
                   cbar_kws={'label': 'Normalized Value'})
        
        # Formatting
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_title('Cluster Profiles (Mean Feature Values)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_cluster_distribution(self, cluster_data, cluster_column='Cluster', 
                                 save_path=None):
        """
        Plot cluster size distribution.
        
        Parameters
        ----------
        cluster_data : pd.DataFrame
            Data with cluster labels
        cluster_column : str
            Name of cluster column
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        cluster_counts = cluster_data[cluster_column].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(cluster_counts.index, cluster_counts.values, 
                     color='steelblue', alpha=0.8, edgecolor='black')
        
        # Add value labels and percentages
        total = cluster_counts.sum()
        for bar in bars:
            height = bar.get_height()
            percentage = (height / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(cluster_counts.index)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
