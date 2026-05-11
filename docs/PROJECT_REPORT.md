# Project Report Template

Use this template to answer all required questions. Fill it after running:

```bash
python src/run_analysis.py
```

The results are saved in results/results_YYYYMMDD_HHMMSS.json and plots are saved in results/plots.

## 1) Multi-class classification metrics

Explain all important metrics (with formulas) and describe when to use them. Use docs/metric_guide.md for formulas.

- Accuracy:
- Balanced Accuracy:
- Precision (macro and weighted):
- Recall (macro and weighted):
- F1-score (macro and weighted):
- Cohen's Kappa:
- Matthews Correlation Coefficient (MCC):
- Jaccard (macro and weighted):

## 2) Hyperparameter tuning and overfitting

- Models trained:
- Tuning method used (grid, random, bayesian):
- Best model by F1-weighted:
- Overfitting check (train vs test gap):

## 3) Important features and retraining

- Top features (from consensus ranking):
- Performance with all features:
- Performance with only top features:
- Key conclusion (does performance drop? why?):

## 4) Cluster analysis and business sense

- Cluster sizes and profiles (use results/plots/cluster_profiles.png):
- What the important features say about each cluster:
- Are these features consistent with business/common sense? Why?
