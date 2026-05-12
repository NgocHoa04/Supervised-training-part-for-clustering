# Project Report — Subject 3: Supervised Learning for Cluster Characterisation

---

## Overview

In the previous step (Subject 2), unsupervised clustering divided the bank's customers into **4 distinct groups** (Cluster 0, 1, 2, 3) based on behavioural and demographic features. Subject 3 asks: can a supervised model learn to predict these cluster labels, and if so — which features drive that prediction?

This report answers all four questions in the Subject 3 brief:

1. Multi-class classification metrics (mathematical formulae, meaning, usage)
2. Hyperparameter tuning with overfitting check
3. Feature importance analysis and retraining with top features
4. Cluster interpretation — business and common-sense alignment

---

## Part 1 — Evaluation Metrics for Multi-Class Classification

### Why is this a multi-class problem?

There are **4 possible output labels** (Cluster 0, 1, 2, 3), not just yes/no. Standard binary metrics do not fully generalise here — we need metrics designed for multiple classes. The goal is to train a classifier that, given a new customer's data, outputs one of these four labels correctly.

Before listing the metrics, it helps to define the four fundamental quantities that all metrics are built from. In a multi-class setting, these are computed **per class** using the One-vs-Rest strategy (e.g., "Is this customer Cluster 2, or not?"):

| Term | Definition |
|---|---|
| **True Positive (TP)** for class $i$ | Model predicted class $i$, and the customer actually belongs to class $i$ ✅ |
| **True Negative (TN)** for class $i$ | Model predicted "not class $i$", and the customer is indeed not class $i$ ✅ |
| **False Positive (FP)** for class $i$ | Model predicted class $i$, but the customer actually belongs to a different class ❌ |
| **False Negative (FN)** for class $i$ | Model predicted "not class $i$", but the customer actually belongs to class $i$ ❌ |

---

### Metric 1 — Accuracy

$$\text{Accuracy} = \frac{\sum_{i} TP_i}{\text{Total predictions}} = \frac{\text{Number of correct predictions}}{N}$$

**Mathematical interpretation:** Of every prediction the model made, what fraction was correct?

**When it works well:** When all 4 clusters are approximately equal in size, and every error is equally costly.

**When it fails:** If Cluster 0 contains 80% of all customers, a model that always predicts "Cluster 0" achieves 80% accuracy while completely ignoring the other three clusters. This is called the **majority class bias** problem. Accuracy alone is therefore an insufficient metric for imbalanced multi-class problems.

**Usage in this project:** Reported as a quick sanity check, but *not* used as the primary metric.

---

### Metric 2 — Precision

$$\text{Precision}_i = \frac{TP_i}{TP_i + FP_i}$$

**Mathematical interpretation:** Of all customers the model labelled as Cluster $i$, what proportion actually belong there? Precision measures how *trustworthy* a positive prediction is.

**Intuitive analogy:** Imagine the classifier is a fishing net. Precision answers: "Of everything the net caught, how much is actually the fish we wanted?"

**Practical example:** If the model labels 100 customers as Cluster 2 but only 85 are genuinely Cluster 2 — Precision = 85%.

**When it matters most:** When a false positive is costly. Sending a premium retention offer to the wrong customer wastes marketing budget and may create a negative impression.

**Multi-class extension:**

$$\text{Macro Precision} = \frac{1}{K} \sum_{i=1}^{K} \text{Precision}_i \quad \text{(all classes weighted equally)}$$

$$\text{Weighted Precision} = \sum_{i=1}^{K} \frac{n_i}{N} \cdot \text{Precision}_i \quad \text{(weighted by class size)}$$

---

### Metric 3 — Recall (Sensitivity)

$$\text{Recall}_i = \frac{TP_i}{TP_i + FN_i}$$

**Mathematical interpretation:** Of all customers who *truly belong* to Cluster $i$, what proportion did the model successfully identify? Recall measures how *complete* the model's detection is.

**Intuitive analogy:** Using the same fishing net: Recall answers "Of all the fish that actually exist in the lake, how many did we catch?"

**Practical example:** If there are 200 real Cluster 2 customers but the model only finds 160 — Recall = 80%. We missed 40 high-value customers.

**When it matters most:** When missing a true member is costly. Failing to identify a high-value customer for retention means losing real revenue. Missing a dormant customer for reactivation means losing a reactivation opportunity.

**Precision–Recall trade-off:** These two metrics are in tension. Making a model more conservative (requiring higher confidence before assigning class $i$) raises Precision but lowers Recall. F1-Score, defined below, is designed to balance this trade-off.

---

### Metric 4 — F1-Score

$$F1_i = 2 \cdot \frac{\text{Precision}_i \times \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$$

**Mathematical interpretation:** The harmonic mean of Precision and Recall. The harmonic mean is used instead of the arithmetic mean because it is biased toward the smaller value — if either Precision or Recall is very low, F1 will also be low, even if the other is high.

**Why harmonic mean, not arithmetic mean?**

Consider a model with Precision = 1.0 and Recall = 0.0:
- Arithmetic mean = (1.0 + 0.0) / 2 = **0.5** — incorrectly suggests "acceptable"
- Harmonic mean = 2·(1.0 × 0.0)/(1.0 + 0.0) = **0.0** — correctly shows the model is useless

Both Precision and Recall must be high for F1 to be high.

**Multi-class extensions:**

$$\text{Macro F1} = \frac{1}{K} \sum_{i=1}^{K} F1_i$$

This treats all clusters equally regardless of size. A cluster with only 100 customers is as important as one with 1,000.

$$\text{Weighted F1} = \sum_{i=1}^{K} \frac{n_i}{N} \cdot F1_i$$

This weights each cluster by its proportion of the data. Clusters with more customers have a larger influence on the aggregate score.

**Which to use for this project:** **Macro F1 is the primary metric.** The brief asks us to characterise all clusters equally, so giving extra weight to larger clusters would misrepresent performance on smaller ones.

---

### Metric 5 — Balanced Accuracy

$$\text{Balanced Accuracy} = \frac{1}{K} \sum_{i=1}^{K} \text{Recall}_i$$

**Mathematical interpretation:** The arithmetic mean of per-class Recall scores. Every class contributes equally to the final score, regardless of size. This directly corrects for the majority-class bias problem that regular Accuracy suffers from.

**Numerical example:** If per-cluster Recall scores are {Cluster 0: 0.95, Cluster 1: 0.88, Cluster 2: 0.91, Cluster 3: 0.93}:

$$\text{Balanced Accuracy} = \frac{0.95 + 0.88 + 0.91 + 0.93}{4} = 0.9175$$

**Relationship to regular Accuracy:** If all clusters are equally sized, Balanced Accuracy equals regular Accuracy. As cluster sizes diverge, they increasingly differ. Balanced Accuracy is the honest version for imbalanced datasets.

**Usage in this project:** Used as a secondary verification metric alongside Macro F1.

---

### Metric 6 — Cohen's Kappa

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Where:
- $p_o$ = observed agreement = the model's actual accuracy on the test set
- $p_e$ = expected agreement by chance = the accuracy a random classifier would achieve if it guessed proportionally to class distribution

**What is $p_e$?**

$$p_e = \sum_{i=1}^{K} \frac{n_i^{\text{actual}} \cdot n_i^{\text{predicted}}}{N^2}$$

This computes how often a random guesser, who knows the overall class frequencies, would get lucky.

**Why it matters:** A model that simply memorises class frequencies and always guesses the most common class will score well on Accuracy, but Kappa corrects for this. Kappa = 0 means the model is no better than random; Kappa = 1 is perfect.

**Example:** If $p_o = 0.92$ and $p_e = 0.50$ (random baseline):

$$\kappa = \frac{0.92 - 0.50}{1 - 0.50} = \frac{0.42}{0.50} = 0.84 \quad \text{(Very good)}$$

**Interpretation scale:**

| Kappa value | Meaning |
|---|---|
| > 0.90 | Almost perfect |
| 0.80 – 0.90 | Very good |
| 0.60 – 0.80 | Good |
| 0.40 – 0.60 | Moderate |
| < 0.40 | Poor / little better than chance |

**Usage:** Best for comparing models fairly when class distributions differ between experiments.

---

### Metric 7 — Matthews Correlation Coefficient (MCC)

For multi-class problems, MCC generalises to:

$$MCC = \frac{c \cdot s - \sum_k p_k \cdot t_k}{\sqrt{(s^2 - \sum_k p_k^2)(s^2 - \sum_k t_k^2)}}$$

Where:
- $c = \sum_k M_{kk}$ = sum of correct predictions (diagonal of confusion matrix)
- $s = \sum_{ij} M_{ij}$ = total number of samples
- $p_k = \sum_j M_{kj}$ = number of samples predicted as class $k$
- $t_k = \sum_j M_{jk}$ = number of samples that truly belong to class $k$

In binary (one-vs-rest) form, each class's contribution is:

$$MCC_i = \frac{TP_i \cdot TN_i - FP_i \cdot FN_i}{\sqrt{(TP_i+FP_i)(TP_i+FN_i)(TN_i+FP_i)(TN_i+FN_i)}}$$

**Scale:** Ranges from −1 to +1:
- **+1** = perfect predictions
- **0** = no better than random guessing
- **−1** = perfect inverse (every prediction is wrong)

**Why MCC is uniquely reliable:** MCC takes *all four* confusion matrix elements into account simultaneously. It cannot be inflated by predicting only the majority class. For a model to score high MCC, it must perform well on *all* classes, including small ones.

**Usage:** Used alongside Macro F1 as a secondary honesty check — a high MCC score confirms the model is genuinely performing, not just guessing the dominant class.

---

### Metric 8 — Confusion Matrix

A confusion matrix is a $K \times K$ table showing every combination of actual vs. predicted labels. For our 4-cluster problem:

|  | Predicted C0 | Predicted C1 | Predicted C2 | Predicted C3 |
|---|---|---|---|---|
| **Actual C0** | TP₀ ✅ | Error | Error | Error |
| **Actual C1** | Error | TP₁ ✅ | Error | Error |
| **Actual C2** | Error | Error | TP₂ ✅ | Error |
| **Actual C3** | Error | Error | Error | TP₃ ✅ |

The diagonal holds correct predictions. Off-diagonal cells reveal which clusters are being confused with each other, and in which direction.

**What the matrix revealed in this project:** The most common misclassification was confusing Cluster 0 customers with Cluster 3 (136 cases in the baseline model). This is interpretable: both clusters contain relatively inactive customers, so their feature profiles overlap. No single-number metric can reveal this — only the full confusion matrix can.

---

### Metric 9 — Jaccard Score (Intersection over Union)

$$\text{Jaccard}_i = \frac{TP_i}{TP_i + FP_i + FN_i}$$

**Mathematical interpretation:** The size of the intersection between predicted Cluster $i$ membership and actual Cluster $i$ membership, divided by their union.

**Geometric analogy:** Draw two circles — one for all customers the model called Cluster $i$, one for all customers who actually are Cluster $i$. Jaccard = overlap area / total area covered by both circles.

**Why it is strict:** Unlike Precision (which only penalises FP) or Recall (which only penalises FN), Jaccard penalises *both* simultaneously. A model that over-predicts (many FP) and under-predicts (many FN) will score low on Jaccard even if each mistake is small in isolation.

**Usage:** Particularly useful when the cost of false positives and false negatives is symmetric and both must be minimised.

---

### Summary — Metric Selection Guide

| Goal | Recommended metric |
|---|---|
| Quick overall check | Accuracy |
| Evaluate all 4 clusters fairly | **Macro F1 ⭐ (primary metric)** |
| Correct for class-size imbalance | Balanced Accuracy |
| Remove "lucky guessing" from the score | Cohen's Kappa |
| Single most honest and robust metric | **MCC** |
| Understand *which* errors were made | Confusion Matrix |
| Strict per-class overlap check | Jaccard Score |

**Primary metric: Macro F1. Supporting metrics: MCC, Balanced Accuracy, Confusion Matrix.**

---

## Part 2 — Hyperparameter Tuning & Overfitting Check

### What are hyperparameters?

A machine learning model has two kinds of parameters:
- **Learned parameters** (e.g., logistic regression coefficients) — the model adjusts these automatically during training
- **Hyperparameters** (e.g., regularisation strength `C`, tree depth, number of neighbours) — the user sets these *before* training; they control *how* the model learns

Tuning means searching for the hyperparameter combination that maximises performance on unseen data.

---

### Models trained and tuning methods

**Notebook `02_complete_analysis.ipynb`** trained 6 model types:

| Model | Type | Key hyperparameters tuned |
|---|---|---|
| Logistic Regression | Linear | `C`, `penalty`, `solver`, `class_weight` |
| Random Forest | Ensemble (bagging) | `n_estimators`, `max_depth`, `min_samples_split`, `max_features` |
| Gradient Boosting | Ensemble (boosting) | `n_estimators`, `max_depth`, `learning_rate`, `subsample` |
| Support Vector Machine | Kernel-based | `C`, `kernel`, `gamma` |
| K-Nearest Neighbours | Instance-based | `n_neighbors`, `weights`, `metric` |
| Decision Tree | Tree | `max_depth`, `min_samples_split`, `criterion` |

**Tuning method:** Grid Search CV — exhaustively tests every combination of specified values, evaluating each with 5-fold Stratified Cross-Validation (folds maintain the same class proportions as the full dataset). Optimisation target: Macro F1.

**Notebook `03_logistics_regression.ipynb`** focused on Logistic Regression with a more sophisticated approach.

**Tuning method:** Optuna (Bayesian Optimisation, TPE Sampler) — rather than blindly testing all combinations, Optuna uses a probabilistic surrogate model to learn which regions of the parameter space are most promising, and directs future trials there.

**Tuning method comparison:**

| Method | Strategy | Speed | Best for |
|---|---|---|---|
| **Grid Search** | Tests every combination | Slow (exponential with parameters) | Small, well-understood parameter spaces |
| **Random Search** | Tests random samples | Medium | Moderate parameter spaces |
| **Bayesian Optimisation (Optuna)** | Learns from past trials, targets promising regions | Most sample-efficient | Large, high-dimensional parameter spaces |

---

### Hyperparameter search space — Logistic Regression (Optuna, 100 trials)

| Parameter | What it controls | Range searched |
|---|---|---|
| `C` | Regularisation strength. Lower C = stricter (shrinks coefficients harder). | $10^{-4}$ to $10^{2}$, log-uniform scale |
| `penalty` | Type of regularisation: L1 (sparsity), L2 (smoothness), ElasticNet (both), None | {l1, l2, elasticnet, none} |
| `solver` | Internal optimisation algorithm | {lbfgs, saga, newton-cg, sag} |
| `class_weight` | Whether to penalise misclassifying smaller clusters more heavily | {None, balanced} |
| `l1_ratio` | Only active when `penalty=elasticnet`: balance between L1 and L2 | Continuous, [0.0, 1.0] |

All 100 trials were evaluated using **5-fold Stratified Cross-Validation**, optimising for **Macro F1**.

---

### Tuning results — Logistic Regression

**Baseline model (default sklearn settings, no tuning):**
- Clusters 0, 1, and 3: high accuracy
- Cluster 2: 185 / 190 correctly classified (97.4%)
- Largest mistake: 136 Cluster 0 customers misclassified as Cluster 3

This Cluster 0 / Cluster 3 confusion is expected — both contain relatively inactive customers with overlapping feature profiles.

**After Optuna tuning:**

| Metric | Baseline | After Tuning | Change |
|---|---|---|---|
| F1-Macro (test set) | ~0.90 | **0.9175** | ↑ +1.75 pp |
| Cluster 2 accuracy | 185 / 190 | **190 / 190 (100%)** | ↑ Perfect recall |
| Cluster 0 vs 3 confusion | 136 wrong | 162 wrong | ↑ Slightly worse |

**Why did Cluster 0 vs 3 confusion increase after tuning?**

Optuna optimised Macro F1, which averages performance equally across all 4 clusters. Cluster 2 was the weakest link — improving it required shifting the model's decision boundaries in ways that blurred the already-difficult Cluster 0 / Cluster 3 boundary. This is a fundamental trade-off in multi-class optimisation: **improving performance on one weak class can slightly reduce performance on adjacent classes**.

**Observations from the Optuna Optimisation History:**

- The majority of 100 trials converged around F1 = 0.92, showing a well-behaved and stable hyperparameter landscape
- 3 trials failed catastrophically (F1 = 0.15 – 0.57) due to incompatible solver/penalty combinations (e.g., L1 penalty is not supported by the `lbfgs` solver)
- The best score was found early (~trial 20–30) and barely changed thereafter — the search could have been stopped at ~50 trials without losing quality

**Most important hyperparameter (Optuna Importance Analysis):**

`C` (regularisation strength) had by far the largest effect on Macro F1. The choice of penalty type or solver had minimal impact in comparison. **Practical implication:** When tuning Logistic Regression on this dataset, prioritise finding the right value of `C` above all other parameters.

**Best final result: F1-Macro = 91.75%**, achieved by Optuna-tuned Logistic Regression with all features.

---

### Overfitting check

**What is overfitting?**

A model overfits when it memorises the training data — including its noise — too closely. It performs very well on training data but fails on new, unseen data. The overfitting gap is:

$$\text{Overfitting Gap} = \text{Train Score} - \text{Test Score}$$

| Gap | Diagnosis |
|---|---|
| < 2% | ✅ Healthy — model generalises well |
| 2% – 10% | ⚠️ Mild overfitting — monitor, consider stronger regularisation |
| > 10% | ❌ Serious overfitting — requires regularisation, simpler model, or more data |

**Learning curve analysis (notebook 03):**

A learning curve plots model performance (Train score and Validation score) as a function of training set size. A healthy model shows:
1. Train and validation scores converging toward the same value as more data is added
2. A consistently small gap between them

Our Optuna-tuned Logistic Regression displayed this pattern exactly — a small, stable gap across all training sizes, with both curves converging toward ~0.92. The gap never exceeded 2%.

**Conclusion: no significant overfitting. The model generalises reliably to new customers.**

**Why Logistic Regression is naturally resistant to overfitting:**

L2 regularisation (the L2 penalty in logistic regression) penalises large coefficients by adding $\lambda ||w||_2^2$ to the loss function. This prevents any single feature from dominating the model, which is the primary mechanism of overfitting in linear models. The tuned value of `C` directly controls this: a small `C` means strong regularisation.

---

### Performance comparison — all 6 models (from notebook 02)

| Model | Train F1 | Test F1 | Gap | Verdict |
|---|---|---|---|---|
| Logistic Regression | ~0.92 | **0.9175** | ~0.002 | ✅ Best, no overfitting |
| Gradient Boosting | ~0.98 | ~0.91 | ~0.07 | ⚠️ Mild overfit |
| Random Forest | ~0.99 | ~0.90 | ~0.09 | ⚠️ Mild overfit |
| SVM | ~0.93 | ~0.90 | ~0.03 | ✅ Good |
| Decision Tree | ~1.00 | ~0.85 | ~0.15 | ❌ Overfit |
| KNN | ~0.94 | ~0.87 | ~0.07 | ⚠️ Mild overfit |

Decision Tree shows the worst overfitting — without depth constraints, it memorises training examples perfectly (Train F1 = 1.00) but generalises poorly. Logistic Regression, despite being the simplest model, achieves the best test performance due to its natural regularisation and the linear separability of the clusters in feature space.

---

## Part 3 — Feature Importance Analysis

### What is feature importance?

Feature importance quantifies **which customer attributes the model relied on most** when making predictions. This answers two questions:

1. **Interpretability:** Are the model's decisions explainable in business terms? If the model relies on features that make intuitive sense, we can trust and act on its predictions.
2. **Dimensionality reduction:** Can we achieve similar performance with fewer features? This reduces model complexity, speeds up inference, and lowers overfitting risk.

---

### How feature importance is computed — by model type

**Tree-based models (Random Forest, Gradient Boosting, Decision Tree):**

Each split in a decision tree selects the feature that most reduces impurity (Gini impurity or information gain). The **Gini importance** of feature $j$ is:

$$\text{Importance}_j = \sum_{\text{nodes using } j} \frac{n_t}{N} \cdot \left[ \text{Gini}(t) - \frac{n_{tL}}{n_t}\text{Gini}(t_L) - \frac{n_{tR}}{n_t}\text{Gini}(t_R) \right]$$

Where $n_t$ is the number of samples at node $t$, and $t_L$, $t_R$ are the left and right child nodes. This is summed across all nodes that use feature $j$, across all trees, then normalised so all importances sum to 1.

**Logistic Regression:**

After standardising all features (using StandardScaler, so they are all on the same scale), the model assigns a coefficient $w_j$ to each feature. Feature importance is simply the absolute value:

$$\text{Importance}_j = |w_j|$$

This is valid *only* after standardisation — without it, features with larger raw scales would appear artificially more important.

**ElasticNet for automatic feature selection (notebook 03):**

ElasticNet adds two regularisation penalties simultaneously to the logistic regression loss:

$$\mathcal{L}_{\text{ElasticNet}} = \mathcal{L}_{\text{original}} + \alpha \left[ \rho \cdot ||w||_1 + (1 - \rho) \cdot ||w||_2^2 \right]$$

- The **L1 component** ($\rho \cdot ||w||_1$) shrinks some coefficients all the way to zero, performing automatic feature elimination
- The **L2 component** ($(1-\rho) \cdot ||w||_2^2$) prevents any single non-eliminated feature from dominating

The hyperparameter $\rho$ (`l1_ratio`) controls the L1/L2 balance. Five values {0.1, 0.3, 0.5, 0.7, 0.9} were evaluated using cross-validation; the best Macro F1 was used to select $\rho$.

**Consensus approach (notebook 02):**

Rather than trusting a single model's perspective, importance rankings were combined across all 6 models:

1. Normalise each model's importance scores to sum to 1 (making them comparable)
2. Average the normalised scores across all models for each feature
3. Compute the standard deviation across models — low SD means broad consensus; high SD means the models disagree

This produces a more robust, model-agnostic feature ranking.

---

### Top features — consensus ranking across all models

| Rank | Feature | What it represents | Consensus SD |
|---|---|---|---|
| 1 | `Months_Inactive_12_mon` | Number of months in the past year with zero transactions | Low (all models agree) |
| 2 | `Total_Trans_Amt` | Total dollar amount spent across all transactions | Low |
| 3 | `Total_Trans_Ct` | Total number of individual transactions | Low |
| 4 | `Total_Revolving_Bal` | Balance that carries over to the next billing cycle | Medium |
| 5 | `Total_Ct_Chng_Q4_Q1` | Change in transaction frequency from Q1 to Q4 of the year | Medium |

**Key observation:** All five top features are **behavioural** — they describe what the customer *does* with their account, not who they are demographically. Age, income, education, and marital status ranked consistently lower across all models.

---

### What happens when we retrain with only the top features?

We retrained using only the features selected by ElasticNet (features whose coefficients were not zeroed out by the L1 penalty), then tuned those reduced models with Optuna.

**Results:**

| Configuration | Test F1-Macro | Test Accuracy |
|---|---|---|
| Baseline (all features, no tuning) | ~0.90 | ~0.90 |
| **Optuna-tuned (all features)** | **0.9175** | **~0.92** |
| Optuna-tuned + Top features only | Lower | Lower |

**The full-feature model outperformed the reduced model.**

**Why did dropping features hurt performance?**

This result is counterintuitive but has a clear explanation. Feature elimination — removing a feature entirely — discards its contribution permanently. Features that appear weak on their own may contribute useful signal when **combined** with others (interaction effects). Regularisation does not discard weak features; it suppresses them. A coefficient of 0.001 is not zero — it still contributes a tiny amount, and across hundreds of customers and many features, those tiny contributions accumulate.

**The key insight:** For this dataset, **regularisation is a strictly better strategy than feature selection.** The optimal model keeps all features but uses a carefully tuned `C` value to control how strongly each feature can influence predictions.

**Practical takeaway:** Feature selection is most valuable when inference speed, storage constraints, or interpretability for non-technical stakeholders is the priority — not raw predictive accuracy.

---

## Part 4 — Cluster Analysis & Business Insights

### Do the important features explain the clusters?

Yes — clearly, consistently, and in a way that aligns with business intuition. The features the model relies on most are exactly the ones that divide customers into meaningful, actionable groups. This alignment between data-driven results and domain knowledge is strong evidence that the clusters from Subject 2 captured **real, stable patterns** — not statistical noise.

The two dimensions that matter most are:

1. **Activity level** — how often does the customer transact, and how much do they spend?
2. **Engagement trajectory** — is their activity increasing, stable, or declining?

---

### Cluster 0 — Dormant Customers

**Profile summary:**

| Feature | Direction | Signal strength |
|---|---|---|
| `Months_Inactive_12_mon` | High (many inactive months) | Very strong (+3.5) |
| `Total_Trans_Amt` | Low | Moderate |
| `Total_Trans_Ct` | Low | Moderate |
| Credit limit | Lower than average | Weak |
| Age | Slightly younger | Weak |

**What the model learned:** Prolonged inactivity is the overwhelming defining signal. Any customer with many inactive months in the past year lands almost certainly in Cluster 0.

**Business interpretation:** These customers have not churned — their accounts are still open — but they have gone quiet. They are in a passive drift state. Without intervention, they are the segment most at risk of letting their account atrophy or eventually closing it.

**Is this business-sensible?** Yes. Banks and card issuers universally treat inactivity as a leading indicator of churn risk. A customer who has not transacted in 4–6 of the last 12 months is exhibiting classic pre-churn behaviour.

**Recommended action:** **Re-engagement campaigns.** Cashback bonuses on the first transaction, limited-time zero-interest balance transfers, or personalised reminders of card benefits. Since this cluster skews younger, digital-first outreach (in-app push notifications, personalised email, SMS) is likely more effective than direct mail.

---

### Cluster 1 — Long-Term, Low-Activity Customers

**Profile summary:**

| Feature | Direction | Signal strength |
|---|---|---|
| `Total_Trans_Amt` | Very low | Very strong (−20.0 — high spend *rules out* Cluster 1) |
| `Total_Trans_Ct` | Very low | Very strong (−17.5) |
| Tenure (months on book) | Long | Moderate (+0.2) |
| Number of products | Many | Moderate (+0.2) |
| Card category | Higher tier | Moderate (+0.2) |

**What the model learned:** The defining pattern is not just low activity — it is the combination of **long tenure + multiple products + near-zero transactions**. The model uses high spending as a *negative* signal — a customer who spends a lot is almost certainly *not* in this cluster.

**Business interpretation:** These are loyal customers who value the *relationship* with the bank more than actively using its products. They may hold accounts for emergency access, credit score maintenance, or social status (premium card tier). They are "breadth without depth" — they have signed up for many products but barely engage with any of them.

**Is this business-sensible?** Yes. This is a well-recognised customer archetype in retail banking: the "passive loyalist." They are unlikely to churn (they have been here for years) but they generate little fee revenue or interchange commission from card transactions.

**Recommended action:** **Activation, not acquisition.** The goal is to convert passive account holders into active card users. Spending-linked rewards (points per dollar, cashback on the first $500 of purchases per month) lower the behavioural activation barrier. Cross-selling more products is less urgent than getting them to use what they already have.

---

### Cluster 2 — High-Value Active Customers

**Profile summary:**

| Feature | Direction | Signal strength |
|---|---|---|
| `Total_Trans_Ct` | Very high | Very strong (+7.5) |
| `Total_Trans_Amt` | Very high | Very strong (+6.5) |
| Credit limit | Higher than average | Weak (+0.1) |
| Number of products | Fewer than average | Moderate (−0.2) |
| Income | Slightly lower reported | Weak (−0.3) |

**What the model learned:** Pure transactional volume and value dominate everything else. If a customer spends a lot and transacts frequently, they almost certainly belong here — no other signals come close in predictive power.

**Business interpretation:** These are the bank's most engaged customers from a revenue standpoint. Despite holding fewer products than Cluster 1, they generate more revenue through card fees, merchant interchange commissions, and (for those who carry a balance) revolving interest. They are the most profitable segment to retain.

**Is this business-sensible?** Yes. Transaction volume is the single most direct driver of card-business revenue. Banks specifically design loyalty programmes to reward high-frequency, high-value spenders because they generate the most interchange revenue per customer.

**Interesting tension:** High activity but fewer products. This is not a contradiction — it may reflect customers who treat this card as their primary spending vehicle while keeping banking elsewhere. It is a clear **upsell opportunity**.

**Recommended action:** **Reward and retain aggressively.** Premium perks — airport lounge access, elevated cashback tiers, concierge services, priority customer support — resonate most with high-activity customers who already see the card as central to their financial life. Upselling savings accounts, investment products, or insurance packages represents the largest revenue growth opportunity in the dataset.

---

### Cluster 3 — Growth-Oriented Customers

**Profile summary:**

| Feature | Direction | Signal strength |
|---|---|---|
| `Months_Inactive_12_mon` | Very low (almost never inactive) | Very strong (−6.0 — inactivity *rules out* Cluster 3) |
| `Total_Ct_Chng_Q4_Q1` | Positive (increasing frequency) | Moderate (+0.12) |
| `Total_Amt_Chng_Q4_Q1` | Positive (increasing spend) | Weak (+0.06) |
| Marital status: divorced | Slight positive | Very weak (+0.1) |
| Education level | Higher than average | Very weak (+0.04) |

**What the model learned:** Two things define this cluster: (1) the customer is almost never inactive, and (2) their transaction frequency and spending have been *increasing* over the year. It is the upward trajectory — not just current activity level — that distinguishes this group from Cluster 2.

**Business interpretation:** These customers are in transition. Possible triggers: a new job and higher income, financial independence after a separation, or simply developing stronger spending habits and card awareness. Whatever the cause, they are on an upward path. They represent the strongest long-term growth opportunity.

**Is this business-sensible?** Yes. Customers whose engagement is actively growing are more likely to become long-term high-value customers, and they are more receptive to product expansion offers than customers at any other stage. This is the lifecycle stage retail banks actively try to identify.

**Interesting signal:** The small contribution of "divorced" status and higher education. These demographic signals are far weaker than behavioural ones, but they suggest this cluster may skew toward educated professionals undergoing life changes that create new financial needs (independent accounts, savings goals, insurance). The model is picking up a faint demographic echo of a major life event.

**Recommended action:** **Invest in this relationship now, before the trajectory levels off.** Products that match growing financial needs — higher credit limits, savings goals, early investment products — are well-timed here. Loyalty programmes that reward *increasing* usage (rather than just high usage) are especially well-suited, as they reinforce the existing upward trend.

---

### The bigger picture — what do the four clusters tell us?

**1. Transaction behaviour is the dominant dimension.**

Across all models and all feature importance methods, the same features consistently ranked highest: how often a customer transacts, how much they spend, and how long they go without activity. Demographics — income, education, age — appeared in results but only as minor supporting signals.

> **The bottom line: what a customer does with their account matters far more than who they are.**

**2. Inactivity is the primary dividing line.**

There is a clear binary split in this dataset:

| Engaged customers | Disengaged customers |
|---|---|
| Clusters 2 & 3 | Clusters 0 & 1 |
| Transact frequently, almost never inactive | Rarely transact, often inactive for months |
| Drive card revenue | High churn risk / activation potential |

`Months_Inactive_12_mon` alone is powerful enough to separate these two halves. Every model in the analysis ranked it as a top feature.

**3. Products held vs. actual card usage — a meaningful tension.**

Cluster 1 holds many banking products but barely uses its card. Cluster 2 uses its card intensely but holds fewer products. This is not a modelling artefact — it reflects a genuine behavioural difference in how customers relate to the bank, and it points directly to different CRM strategies for each group.

**4. The supervised model validates the unsupervised clustering.**

The fact that a supervised model achieves **91.75% Macro F1** on labels generated entirely by unsupervised clustering is strong evidence that those clusters are **stable, well-separated, and real**. If the clusters were just mathematical noise — arbitrary groupings with no coherent structure — a classifier could not learn to predict them. Clean separability (confirmed by the confusion matrix, with mistakes only between the most similar adjacent clusters) means the clustering captured genuine patterns in customer behaviour.

**5. Full alignment with banking domain knowledge.**

The features the model relies on — inactivity, transaction volume, spending trends — are exactly what a banking analyst or CRM strategist would examine when deciding which customers to retain, reactivate, or upsell. This alignment between data-driven results and domain knowledge gives high confidence that the entire pipeline, from clustering to supervised classification, is grounded in real customer dynamics, not statistical noise.

---
