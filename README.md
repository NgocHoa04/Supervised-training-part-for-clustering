# Supervised Training for Clustering

A multi-class classification project that applies supervised learning to predict cluster labels and identify the most important features that characterize each cluster. This analysis reveals the predictive power of features and their interpretability for business insights.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete analysis
python src/run_analysis.py
```

---

## 📊 1. Multi-Class Metrics & Evaluation

### 1.1 Core Metrics Explained

#### **ACCURACY**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- Overall percentage of correct predictions
- **Use when:** Balanced data with equal importance on all classes
- **Limitation:** Unreliable with imbalanced data

#### **PRECISION**
$$\text{Precision}_i = \frac{TP_i}{TP_i + FP_i}$$
- Of all predicted as class i, how many are actually class i?
- **Use when:** False positives are costly (e.g., spam detection)

#### **RECALL** (Sensitivity)
$$\text{Recall}_i = \frac{TP_i}{TP_i + FN_i}$$
- Of all actual class i instances, how many did we find?
- **Use when:** False negatives are costly (e.g., fraud detection)

#### **F1-SCORE** (Harmonic Mean)
$$F1_i = 2 \times \frac{\text{Precision}_i \times \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$$
- Balances Precision and Recall
- **Macro F1:** Simple average across classes (treats all classes equally)
- **Weighted F1:** Weighted average by class size (accounts for imbalance)
- **Recommended for:** Imbalanced multi-class problems ⭐

#### **BALANCED ACCURACY**
$$\text{Balanced Accuracy} = \frac{1}{n} \sum_{i=1}^{n} \text{Recall}_i$$
- Average recall across all classes
- **Use when:** Need equal performance on all classes

#### **COHEN'S KAPPA**
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
- Agreement beyond random chance (adjusts for class imbalance)
- **Interpretation:** $\kappa > 0.9$ (excellent) | $0.8-0.9$ (very good) | $0.6-0.8$ (good)

#### **MATTHEWS CORRELATION COEFFICIENT (MCC)**
$$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$
- Considers all confusion matrix elements, unbiased by class imbalance
- **Best for:** Highly imbalanced datasets

#### **CONFUSION MATRIX**
Detailed breakdown of predictions:
- **TP:** Correctly predicted positive
- **TN:** Correctly predicted negative  
- **FP:** False positive (false alarm)
- **FN:** False negative (miss)

#### **JACCARD SCORE**
$$\text{Jaccard} = \frac{TP}{TP + FP + FN}$$
- Intersection over union ratio
- Sensitive to both false positives and false negatives

---

## 🎯 2. Hyperparameter Tuning & Overfitting Analysis

### 2.1 Tuning Methods

| Method | Best For | Time | Trade-off |
|--------|----------|------|-----------|
| **Grid Search** | Small parameter spaces | Minutes-hours | Exhaustive but slow |
| **Random Search** | Medium parameter spaces | Seconds-minutes | Fast but may miss optimal |
| **Bayesian Optimization** | Large parameter spaces | Hours | Most efficient |

### 2.2 Overfitting Detection

$$\text{Overfitting Gap} = \text{Train Accuracy} - \text{Test Accuracy}$$

- **Gap < 2%** ✅ Good fit
- **Gap 2-10%** ⚠️ Mild overfitting  
- **Gap > 10%** ❌ Severe overfitting (retrain with regularization)

---

## 📈 3. Feature Importance Analysis

### 3.1 Calculation Methods

**Tree-based models:**
- Information gain at each split (Gini or Entropy importance)

**Linear models (Logistic Regression):**
- Absolute value of coefficients (higher |coeff| = more important)

**Consensus approach:**
1. Normalize each model's importances (sum to 1)
2. Average across all models
3. Standard deviation shows agreement level

### 3.2 Feature Selection Impact

- **Full model (30 features):** 89.3% F1-Score
- **Top-10 features:** 84.6% F1-Score
- **Retention:** 94.7% performance with only 33% of features ✨

### 3.3 Retraining with Top Features

After identifying important features:
1. Filter dataset to top features only
2. Retrain all models with reduced feature set
3. Compare metrics with full model
4. Assess business interpretability vs performance trade-off

---

## 🔍 4. Cluster Interpretation & Business Insights

### Cluster Profiles (Based on Important Features)

**Cluster 0: Dormant Customers**
- Strong positive driver: High inactivity periods (+3.5)
- Secondary factors: Low credit limits, younger age
- **Profile:** Disengaged customers with long periods without activity
- **Business Action:** Re-engagement campaigns, incentive programs

**Cluster 1: Long-term, Low-Activity Customers**
- Drivers: Long tenure (+0.2), multiple products (+0.2), higher-tier cards (+0.2)
- Strong negative: Transaction activity (-20.0), spending (-17.5)
- **Profile:** Relationship banking customers - hold accounts but rarely use them
- **Business Action:** Activation programs, product cross-sell

**Cluster 2: High-Value Active Customers**
- Strong drivers: High transaction volume (+7.5), high spending (+6.5)
- Secondary: Higher credit limits (+0.1)
- Negative: Lower income, fewer products
- **Profile:** Highly engaged transactors, most valuable from usage perspective
- **Business Action:** Premium services, loyalty programs

**Cluster 3: Growth-Oriented Customers**
- Key driver: Almost never inactive (-6.0 inactive months)
- Positive trends: Increasing transaction frequency (+0.12), increasing spending (+0.06)
- Minor factors: Divorced status, higher education
- **Profile:** Customers with growing engagement, life-event driven
- **Business Action:** Personalized growth incentives, relationship building

### Key Insights

**Central Story:** Transaction behavior is the primary differentiator across clusters

**The Main Divide:** **Inactivity** acts as a wall separating:
- Engaged customers (Clusters 2 & 3) → Active, frequent transactions
- Disengaged customers (Clusters 0 & 1) → Low activity, dormant

**Products vs Usage:** A clear tension exists between:
- Cluster 1: Relationship focus (many products, minimal usage)
- Cluster 2: Usage focus (heavy transactions, fewer products)

**Demographics Matter Less:** Personal attributes (income, education, age) are minor factors compared to behavioral patterns - **What customers do trumps who they are**

---

## ⚙️ 5. Configuration & Customization

Edit `config.py` to customize:
- **Models to train** (Logistic Regression, Decision Tree, Random Forest, SVM, etc.)
- **Tuning method** (Grid Search, Random Search, Bayesian Optimization)
- **Number of top features** for retraining
- **Plot settings** (figure size, color scheme, DPI)
- **Random state** for reproducibility

---

## 📂 Project Structure

```
📦 Supervised-training-part-for-clustering
│
├── 📂 src/                          ✨ Core code
│   ├── __init__.py
│   ├── preprocessing.py             Data cleaning & feature engineering
│   ├── model_utils.py              Model training utilities
│   ├── metrics.py                  Performance metrics (all metrics explained above)
│   ├── visualization.py            Plotting functions
│   └── run_analysis.py             Main entry point ⭐
│
├── 📂 dataset/                      📊 Input data
│   ├── BankChurners_Final.csv      Original customer data with cluster labels
│   └── customer_clusters_with_features.csv  Pre-computed clusters
│
├── 📂 results/                      📈 Output files & visualizations
│   ├── logs/                       Training logs
│   ├── model_comparison.png        Multi-metric comparison across models
│   ├── overfitting_analysis.png    Train vs Test accuracy gap detection
│   ├── feature_importance.png      Feature rankings (consensus across models)
│   └── cluster_profiles.png        Cluster-specific feature patterns
│
├── 📂 models/                       💾 Best performing models & configs
│   ├── best_params_*.json          Optimal hyperparameters
│   └── performance_*.json          Metrics on test set
│
├── 📂 notebooks/                    🔬 Experimental notebooks
│   ├── 01_cluster_analysis.ipynb   Initial cluster exploration
│   ├── 02_complete_analysis.ipynb  Full pipeline development
│   ├── 03_logistics_regression.ipynb  Logistic regression deep-dive
│   └── 04_optimize_decision_tree_main.ipynb  Decision tree optimization
│
├── 📂 docs/                         📚 Reference documentation
│   ├── METRICS_GUIDE.md            Extended metric explanations
│   └── PROJECT_REPORT.md           Analysis report template
│
├── ⚙️  config.py                    Configuration settings
├── 📄 README.md                     This file
└── 📋 requirements.txt              Python dependencies
```

---

## 🎯 Key Outputs & Deliverables

✅ **Predictive Models:** Trained and tuned on multi-class classification task  
✅ **Performance Metrics:** All major metrics with mathematical definitions  
✅ **Overfitting Analysis:** Train vs test gap detection with interpretation  
✅ **Feature Importance:** Ranked features with consensus scoring  
✅ **Feature Selection Impact:** Retraining results with top features only  
✅ **Cluster Profiles:** Business-meaningful interpretation of each cluster  
✅ **Visualizations:** 4+ publication-quality plots in `results/`  
✅ **Results:** JSON export of metrics and hyperparameters  

---

## 📖 Further Reading

- **Metrics Guide:** [docs/METRICS_GUIDE.md](docs/METRICS_GUIDE.md) - Extended explanations
- **Project Report:** [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) - Analysis template
- **Notebooks:** Check `notebooks/` for step-by-step implementations

---