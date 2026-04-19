# Results Index

## Variant Summary

| Variant | Description | Key Files |
|---------|-------------|----------|
| baseline | Basic Logistic Regression without preprocessing | trained_model.pkl, Unprocessed graph.png |
| missing_values_imputed | KNN Imputation + SMOTE | trained_model.pkl, KNN fix.png, results.png |
| missing_values_dropped | Drop rows with missing values + SMOTE | trained_model.pkl, drop row fix.png, Results.png |
| smote_scaling | StandardScaler + SMOTE | trained_model.pkl, SMOTE applied graph.png, results.png |
| max_iter | Increased max_iter to 10000 | trained_model.pkl, Max iter increase graph.png |
| recall_fix | Class weight balanced | trained_model.pkl, Recall fix graph.png, class imbalance graph.png |

## Output Locations

- Models: `../outputs/<variant>/trained_model.pkl`
- Visualizations: `../outputs/<variant>/*.png`

## Model Performance

Each variant outputs a classification_report with:
- Precision, Recall, F1-score per class
- Macro and weighted averages

See individual console outputs for detailed metrics.