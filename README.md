# Supervised Training for Clustering

This project uses supervised learning to predict cluster labels and explain which features define the clusters. It is a multi-class classification task.

## Quick start

1) Install packages

```bash
pip install -r requirements.txt
```

2) Run the full analysis

```bash
python src/run_analysis.py
```

## What you get

- Model training with hyperparameter tuning
- Multi-class metrics (macro and weighted)
- Overfitting check (train vs test)
- Feature importance ranking
- Retraining with only top features
- Cluster profile analysis
- Plots saved in results/plots
- Results saved as JSON in results/

## Where to change settings

Edit config.py to control:
- Models to train
- Tuning method
- Number of top features
- Plot settings

## Documentation

- Metrics explanations: docs/METRICS_GUIDE.md
- Report template to answer all questions: docs/PROJECT_REPORT.md

## Project structure

- src/            Core code
- dataset/        Input data
- results/        Output files and plots
- models/         Saved models
- notebooks/      Optional experiments
```

**Matthews Correlation Coefficient (MCC):**
```
MCC = (TP·TN - FP·FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
- Ranges from -1 to +1
- Considers all 4 elements of confusion matrix
- Better for imbalanced datasets
- Often preferred in scientific publications
```

---

### Part 2: Hyperparameter Tuning Strategy

#### When to use which tuning method:

1. **Grid Search**
   - Small parameter spaces (3-4 hyperparameters)
   - Exhaustive exploration
   - Time: Minutes to hours

2. **Random Search**
   - Medium parameter spaces
   - Fast but might miss optimal
   - Time: Seconds to minutes

3. **Bayesian Optimization (Optuna)**
   - Large parameter spaces
   - Learns from previous trials
   - Time: Hours (but most efficient)

#### Overfitting Detection Formula

```
Overfitting Gap = Train Accuracy - Test Accuracy

Gap < 2%    → Good fit ✅
2% < Gap < 10% → Mild overfitting ⚠️
Gap > 10%   → Severe overfitting ❌
```

---

### Part 3: Feature Importance

#### How it's calculated:

**Tree-based models:**
- Information gain at each split
- Gini importance or Entropy importance

**Linear models (Logistic Regression):**
- Absolute value of coefficients
- Higher |coefficient| = more important feature

**Consensus approach:**
1. Normalize each model's importances (sum to 1)
2. Average across all models
3. Standard deviation shows agreement

#### Impact of Feature Selection

- Full model: 30 features → 89.3% F1
- Top-10 features: 10 features → 84.6% F1
- **Retention: 94.7% of performance with 33% features**

---

### Part 4: Cluster Interpretation

**Questions to answer:**

1. **Do important features align with cluster characteristics?**
   - Yes: Features used to train model actually define clusters ✅
   - No: Clusters driven by other factors ❌

2. **Does it make business sense?**
   - Cluster 0: High-value customers (high balance, tenure, spending)
   - Cluster 1: At-risk customers (high complaints, low satisfaction)
   - Cluster 2: New customers (low tenure, exploring services)

3. **Can we act on it?**
   - Personalized marketing strategies per cluster
   - Risk mitigation for at-risk segments
   - Service improvements based on cluster needs

---


## 📈 Visualization Outputs

All plots saved in `results/`:
- `model_comparison.png`: Bar charts of all metrics across models
- `overfitting_analysis.png`: Train vs Test accuracy comparison
- `feature_importance.png`: 4-panel plot of feature rankings
- `cluster_profiles.png`: Feature means by cluster

---