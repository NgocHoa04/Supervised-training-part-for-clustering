# Metrics Guide (Multi-class Classification)

This guide explains the main metrics for multi-class classification in simple English. Use it to explain results in your report.

## Confusion matrix

A confusion matrix counts how often each class was predicted as each class.

- Diagonal cells: correct predictions
- Off-diagonal cells: mistakes

For a class $i$:
- True positive $TP_i$: predicted $i$ and actually $i$
- False positive $FP_i$: predicted $i$ but actually not $i$
- False negative $FN_i$: actually $i$ but predicted something else
- True negative $TN_i$: all other cases

## Accuracy

$$
\text{Accuracy} = \frac{\sum_i TP_i}{N}
$$

Meaning: the percent of correct predictions.

Use when classes are balanced and all errors are equally important.

## Precision (per class)

$$
\text{Precision}_i = \frac{TP_i}{TP_i + FP_i}
$$

Meaning: when the model predicts class $i$, how often it is correct.

Use when false positives are costly.

## Recall (per class)

$$
\text{Recall}_i = \frac{TP_i}{TP_i + FN_i}
$$

Meaning: of all true class $i$ items, how many are found.

Use when missing a class is costly.

## F1-score (per class)

$$
\text{F1}_i = 2 \times \frac{\text{Precision}_i \times \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}
$$

Meaning: balance between precision and recall.

Use when you want a single score that balances both kinds of errors.

## Macro average

Macro average treats all classes equally.

$$
\text{Macro Metric} = \frac{1}{K} \sum_{i=1}^K \text{Metric}_i
$$

Meaning: small classes matter as much as large classes.

Use when all classes are equally important.

## Weighted average

Weighted average uses class size as weights.

$$
\text{Weighted Metric} = \sum_{i=1}^K w_i \times \text{Metric}_i
$$

where $w_i = \frac{n_i}{N}$.

Meaning: large classes have more influence.

Use when you want a score close to overall performance.

## Balanced accuracy

$$
\text{Balanced Accuracy} = \frac{1}{K} \sum_{i=1}^K \text{Recall}_i
$$

Meaning: average recall across classes.

Use when classes are imbalanced.

## Cohen's Kappa

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

- $p_o$: observed agreement
- $p_e$: expected agreement by chance

Meaning: how much better the model is than random guessing.

Use to check if the model is truly useful beyond chance.

## Matthews Correlation Coefficient (MCC)

For multi-class, MCC uses the confusion matrix to measure correlation between predictions and true labels.

A simple view is:

$$
\text{MCC} = \frac{\text{cov}(y, \hat{y})}{\sqrt{\text{cov}(y, y) \times \text{cov}(\hat{y}, \hat{y})}}
$$

Meaning: a balanced correlation score, robust to class imbalance.

Use when data is imbalanced and you want a fair, overall metric.

## Jaccard score

For a class $i$:

$$
\text{Jaccard}_i = \frac{TP_i}{TP_i + FP_i + FN_i}
$$

Meaning: how much the predicted set overlaps the true set.

Use for strict evaluation where any error is costly.

## Overfitting check

Compare training and test scores:

$$
\text{Gap} = \text{Train Score} - \text{Test Score}
$$

If the gap is large (for example $> 0.05$), the model may be overfitting.
